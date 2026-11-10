"""
dynamic_multi_scenario_trainer.py
=================================
多场景分层强化学习 HPPO 训练驱动器。

功能说明：
  1. 解析动态场景数据集（包含 train / val / test 划分与 manifest 元数据）；
  2. 在训练迭代中随机抽取训练场景，驱动 DynamicEventManager 推进动态插单事件；
  3. 对奖励差值执行尺度缩放，稳定 Critic 训练梯度；
  4. 联合更新 HGNN、ManagerPolicy、WorkerPolicy 与共享 Critic 网络；
  5. 周期性在验证集上执行确定性策略评估，追踪 Actor Cmax、完整事件可行率与 PPO 损失；
  6. 自动保存最优 Checkpoint 与全量训练日志。
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_ENVIRONMENT_DIR = os.path.join(_ROOT_DIR, "environment")
_ACTIVE_DATASET_POINTER = os.path.join(_ROOT_DIR, "data", "active_experiment_package.json")
_TRAIN_PROCESS_LOCK = os.path.join(_ROOT_DIR, ".dynamic_multi_scenario_trainer.lock")
_TRAIN_PROCESS_LOCK_HANDLE = None
for _p in (_THIS_DIR, _ENVIRONMENT_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.experiment_protocol import DEFAULT_FORMAL_SEED, FORMAL_SEEDS

try:
    import msvcrt
except ImportError:
    msvcrt = None

STAGE_VALIDATION_FILENAMES: Dict[int, Tuple[str, ...]] = {
    1: (
        "scenario_val_000.xlsx",
        "scenario_val_001.xlsx",
    ),
    2: (
        "scenario_val_002.xlsx",
        "scenario_val_003.xlsx",
    ),
    3: (
        "scenario_val_004.xlsx",
        "scenario_val_005.xlsx",
    ),
}
_CHECKPOINT_SELECTION_EPS = 1e-2


def resolve_trainer_identity(trainer: Any) -> Tuple[str, str]:
    """Return the concrete trainer identity used by logs and checkpoints.

    Class-level identifiers take precedence; ``checkpoint_contract`` supplies
    the same identifiers when a trainer does not define those attributes.
    """
    trainer_contract = dict(getattr(trainer, "checkpoint_contract", {}) or {})
    method_id = str(
        getattr(
            trainer,
            "method_id",
            trainer_contract.get("method_id", "AD-HG-DAPPO"),
        )
    )
    code_id = str(
        getattr(
            trainer,
            "code_id",
            trainer_contract.get("code_id", "dual_actor_hppo"),
        )
    )
    return method_id, code_id


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_validation_protocol(stage_index: int) -> str:
    """Return the versioned checkpoint-selection protocol for one Stage."""
    stage = int(stage_index)
    filenames = STAGE_VALIDATION_FILENAMES.get(stage)
    if filenames is None:
        raise ValueError(f"阶段编号必须为 1..3，收到 {stage_index!r}。")
    suffix = "".join(os.path.splitext(filename)[0].rsplit("_", 1)[-1][-1] for filename in filenames)
    return f"stage{stage}_val{suffix}_all_feasible_mean_then_worst_v1"


def summarize_stage_validation_selection(
    stage_index: int,
    details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one auditable two-scenario Stage checkpoint score.

    A candidate is eligible only when both fixed validation scenarios are
    autonomously feasible.  Eligible candidates are ranked by lower equal-
    weight mean Cmax and, for near-equal means, lower worst-case Cmax.
    """
    stage = int(stage_index)
    expected = list(STAGE_VALIDATION_FILENAMES.get(stage, ()))
    if not expected:
        raise ValueError(f"阶段编号必须为 1..3，收到 {stage_index!r}。")
    detail_by_file: Dict[str, Dict[str, Any]] = {}
    for detail in details:
        filename = os.path.basename(str(detail.get("file", "")))
        if filename in detail_by_file:
            raise ValueError(f"Stage {stage} 验证结果存在重复场景: {filename}")
        detail_by_file[filename] = detail

    missing_files = [filename for filename in expected if filename not in detail_by_file]
    unexpected_files = sorted(filename for filename in detail_by_file if filename not in expected)
    per_file: List[Dict[str, Any]] = []
    cmax_values: List[float] = []
    successful_scenarios = 0
    for filename in expected:
        detail = detail_by_file.get(filename)
        success = bool(detail and detail.get("autonomous_success"))
        raw_cmax = float(detail.get("raw_actor_cmax", float("inf"))) if detail else float("inf")
        finite_cmax = raw_cmax != float("inf") and raw_cmax == raw_cmax
        if success and finite_cmax:
            successful_scenarios += 1
            cmax_values.append(raw_cmax)
        per_file.append(
            {
                "file": filename,
                "autonomous_success": success,
                "raw_actor_cmax": raw_cmax,
                "failed_event_count": int(detail.get("failed_event_count", 0)) if detail else None,
            }
        )

    eligible = (
        not missing_files
        and not unexpected_files
        and successful_scenarios == len(expected)
        and len(cmax_values) == len(expected)
    )
    return {
        "protocol": stage_validation_protocol(stage),
        "stage": stage,
        "validation_files": expected,
        "eligible": eligible,
        "successful_scenarios": successful_scenarios,
        "mean_cmax": (sum(cmax_values) / len(cmax_values)) if eligible else None,
        "worst_cmax": max(cmax_values) if eligible else None,
        "missing_files": missing_files,
        "unexpected_files": unexpected_files,
        "per_file": per_file,
    }


def is_stage_validation_candidate_better(
    candidate: Dict[str, Any],
    best_mean_cmax: float,
    best_worst_cmax: float,
) -> bool:
    """Apply the common all-feasible mean-then-worst Stage ordering."""
    if not candidate.get("eligible"):
        return False
    candidate_mean = float(candidate["mean_cmax"])
    candidate_worst = float(candidate["worst_cmax"])
    if candidate_mean < best_mean_cmax - _CHECKPOINT_SELECTION_EPS:
        return True
    return (
        abs(candidate_mean - best_mean_cmax) <= _CHECKPOINT_SELECTION_EPS
        and candidate_worst < best_worst_cmax - _CHECKPOINT_SELECTION_EPS
    )



def _acquire_single_training_process_lock(lock_dir: Optional[str] = None) -> None:
    """Prevent two PyCharm/shell launches from writing the same live artifacts."""
    global _TRAIN_PROCESS_LOCK_HANDLE
    target_lock = os.path.join(lock_dir, ".process.lock") if lock_dir else _TRAIN_PROCESS_LOCK
    os.makedirs(os.path.dirname(os.path.abspath(target_lock)), exist_ok=True)
    lock_handle = open(target_lock, "a+b")
    lock_handle.seek(0, os.SEEK_END)
    if lock_handle.tell() == 0:
        lock_handle.write(b"0")
        lock_handle.flush()
    lock_handle.seek(0)
    if msvcrt is not None:
        try:
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            lock_handle.close()
            raise RuntimeError(
                "检测到另一个 dynamic_multi_scenario_trainer.py 训练进程正在运行。"
                "本次启动已停止，以防两个进程同时写入断点与日志。"
            ) from exc
    else:
        try:
            import fcntl
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, OSError):
            pass

    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(str(os.getpid()).encode("ascii"))
    lock_handle.flush()
    lock_handle.seek(0)
    _TRAIN_PROCESS_LOCK_HANDLE = lock_handle

    def _release_lock() -> None:
        global _TRAIN_PROCESS_LOCK_HANDLE
        if _TRAIN_PROCESS_LOCK_HANDLE is None:
            return
        try:
            _TRAIN_PROCESS_LOCK_HANDLE.seek(0)
            if msvcrt is not None:
                msvcrt.locking(_TRAIN_PROCESS_LOCK_HANDLE.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                try:
                    import fcntl
                    fcntl.flock(_TRAIN_PROCESS_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
        finally:
            _TRAIN_PROCESS_LOCK_HANDLE.close()
            _TRAIN_PROCESS_LOCK_HANDLE = None

    atexit.register(_release_lock)

import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from data_models import POOL_B, POOL_D, Order  # type: ignore
from dynamic_validator import load_dynamic_orders_and_events, validate_dynamic_excel  # type: ignore
from dynamic_event_manager import (  # type: ignore
    DynamicEventManager,
    INITIAL_SOLUTION_BUILDER_VERSION,
    INITIAL_SOLUTION_DISPATCH_RULE,
)
from eligibility_builder import eligibility_policy_descriptor  # type: ignore
from hppo_trainer import (  # type: ignore
    FORMAL_DYNAMIC_C_ENT,
    FORMAL_DYNAMIC_LR,
    FORMAL_DYNAMIC_PLACEMENT_LR,
    FORMAL_DYNAMIC_PROTOCOL_VERSION,
    FORMAL_DYNAMIC_REWARD_SCALE,
    HPPOTrainer,
)


_SCENARIO_RECORD_REQUIRED_FIELDS = {
    "scenario_id",
    "split",
    "seed",
    "rng_stream_id",
    "base_instance_id",
    "source_sheets",
    "coverage_classes",
    "initial_orders",
    "dynamic_events",
    "cmax_reference",
    "relative_path",
    "filepath",
    "sha256",
    "validation_status",
}
_CURRICULUM_POLICY_REQUIRED_FIELDS = {
    "policy_version",
    "selection_rule",
    "scope",
    "ranking_fields",
    "ranking_direction",
    "formal_train_scenarios",
    "formal_total_episodes",
    "base_quota",
    "quota_step",
    "quota_assignment",
}
_SCALE_RANKING_FIELDS = ["initial_orders", "dynamic_events", "scenario_id"]
_DATASET_LEDGER_VERSION = "V20261027_THREE_GROUP_WORKBOOK_DECODED_V1"
_DATASET_LEDGER_REQUIRED_FIELDS = {
    "ledger_version",
    "derivation",
    "total_scenarios",
    "total_initial_orders",
    "total_dynamic_events",
    "event_class_counts",
    "inserted_order_process_type_counts",
    "forecast_status_counts",
    "split_summary",
}
_DATASET_LEDGER_SPLIT_FIELDS = {
    "scenarios",
    "total_initial_orders",
    "total_dynamic_events",
    "event_class_counts",
    "inserted_order_process_type_counts",
    "forecast_status_counts",
}


def _scenario_scale_rank_key(record: Dict[str, Any]) -> Tuple[int, int, str]:
    """Return the deterministic small-to-large key used by curriculum B."""
    scenario_id = record.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("课程学习场景缺少合法 scenario_id。")

    counts: List[int] = []
    for field in _SCALE_RANKING_FIELDS[:2]:
        value = record.get(field)
        if isinstance(value, bool):
            raise ValueError(f"课程学习场景 {scenario_id} 的 {field} 不能为布尔值。")
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"课程学习场景 {scenario_id} 的 {field} 必须为正整数。") from exc
        if count <= 0:
            raise ValueError(f"课程学习场景 {scenario_id} 的 {field} 必须为正整数。")
        counts.append(count)
    return counts[0], counts[1], scenario_id


def validate_scale_ranked_curriculum_policy(
    manifest: Dict[str, Any],
    expected_scenarios: Dict[str, int],
) -> Dict[str, Any]:
    """Validate the formal B policy: scale rank, then progressive quota."""
    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, dict):
        raise ValueError("活动数据 Manifest 缺少 split_policy。")
    policy = split_policy.get("training_curriculum_policy")
    if not isinstance(policy, dict):
        raise ValueError("活动数据 Manifest 缺少按算例规模分配轮数的 training_curriculum_policy。")
    missing = sorted(_CURRICULUM_POLICY_REQUIRED_FIELDS - set(policy))
    if missing:
        raise ValueError(f"活动数据 Manifest 的课程学习政策缺少字段：{missing}。")
    if policy["selection_rule"] != "B_GROUPED_SCALE_RANKED_PROGRESSIVE_QUOTAS":
        raise ValueError("活动数据 Manifest 的课程学习政策必须采用 B_GROUPED_SCALE_RANKED_PROGRESSIVE_QUOTAS。")
    if policy["scope"] != "train":
        raise ValueError("活动数据 Manifest 的课程学习政策只能作用于 train 划分。")
    if policy["ranking_fields"] != _SCALE_RANKING_FIELDS or policy["ranking_direction"] != "ascending":
        raise ValueError(
            "活动数据 Manifest 的课程学习排序必须为 "
            "initial_orders → dynamic_events → scenario_id 的升序。"
        )
    if policy["formal_train_scenarios"] != expected_scenarios.get("train"):
        raise ValueError("活动数据 Manifest 的课程学习训练场景数与活动合同不一致。")
    if policy["formal_total_episodes"] != 5400 or policy["base_quota"] != 300 or policy["quota_step"] != 0:
        raise ValueError("活动数据 Manifest 的正式课程学习预算必须为 18 场景、5400 轮（每组 1800 轮循环）、300 起点、0 步长。")
    return dict(policy)


def _sha256_file(filepath: str) -> str:
    """Return the SHA-256 digest of one scenario workbook."""
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset_record_path(dataset_dir: str, record: Dict[str, Any]) -> str:
    """Resolve one manifest record and reject paths outside the active root."""
    relative_path = record.get("filepath")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("活动数据 Manifest 含无 filepath 的场景记录。")
    if record.get("relative_path") != relative_path:
        raise ValueError(
            f"场景 {record.get('scenario_id')!r} 的 relative_path 与 filepath 不一致。"
        )
    scenario_path = os.path.abspath(os.path.join(dataset_dir, relative_path))
    if os.path.commonpath([os.path.normcase(dataset_dir), os.path.normcase(scenario_path)]) != os.path.normcase(dataset_dir):
        raise ValueError(f"活动数据 Manifest 的场景路径越出数据集根目录：{relative_path}")
    if not os.path.isfile(scenario_path):
        raise FileNotFoundError(f"活动数据 Manifest 引用的场景不存在：{scenario_path}")
    return scenario_path


def validate_dataset_manifest_integrity(
    manifest: Dict[str, Any],
    dataset_dir: str,
    expected_scenarios: Dict[str, int],
) -> Dict[str, int]:
    """Verify manifest views, workbook hashes, and decoded scenario cardinalities.

    ``splits`` and ``scenarios`` are two views of the same immutable workbook
    records.  The trainer must not select a file unless both views are exactly
    identical and its declared SHA-256, initial-order count, and dynamic-event
    count agree with the current V20261027 workbook.
    """
    split_index = manifest.get("splits")
    scenario_records = manifest.get("scenarios")
    if not isinstance(split_index, dict) or not isinstance(scenario_records, list):
        raise ValueError("活动数据 Manifest 必须同时包含 splits 索引和 scenarios 列表。")
    expected_split_names = set(expected_scenarios)
    if set(split_index) != expected_split_names:
        raise ValueError(
            "活动数据 Manifest 的 splits 必须且只能包含活动划分："
            f"{sorted(expected_split_names)}。"
        )

    scenarios_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for record in scenario_records:
        if not isinstance(record, dict):
            raise ValueError("活动数据 Manifest 的 scenarios 含非对象记录。")
        split = record.get("split")
        scenario_id = record.get("scenario_id")
        if not isinstance(split, str) or not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("活动数据 Manifest 的 scenarios 含缺少 split 或 scenario_id 的记录。")
        per_split = scenarios_by_split.setdefault(split, {})
        if scenario_id in per_split:
            raise ValueError(f"活动数据 Manifest 的 scenarios 含重复场景：{split}/{scenario_id}。")
        per_split[scenario_id] = record
    if set(scenarios_by_split) != expected_split_names:
        raise ValueError(
            "活动数据 Manifest 的 scenarios 必须且只能包含活动划分："
            f"{sorted(expected_split_names)}。"
        )

    checked_by_split: Dict[str, int] = {}
    for split, expected_count in expected_scenarios.items():
        records = split_index.get(split)
        if not isinstance(records, list) or len(records) != expected_count:
            actual_count = len(records) if isinstance(records, list) else 0
            raise ValueError(
                f"活动数据 Manifest 的 {split} 场景数错误："
                f"期望 {expected_count}，实际 {actual_count}。"
            )

        split_records: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"活动数据 Manifest 的 {split} 含非对象场景记录。")
            missing = sorted(_SCENARIO_RECORD_REQUIRED_FIELDS - set(record))
            if missing:
                raise ValueError(f"活动数据 Manifest 的 {split} 场景缺少字段：{missing}。")
            scenario_id = record.get("scenario_id")
            if record.get("split") != split or not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError(f"活动数据 Manifest 的 {split} 含 split/scenario_id 不一致的记录。")
            if scenario_id in split_records:
                raise ValueError(f"活动数据 Manifest 的 {split} 含重复场景：{scenario_id}。")
            split_records[scenario_id] = record

        scenario_view = scenarios_by_split.get(split, {})
        if set(split_records) != set(scenario_view):
            raise ValueError(f"活动数据 Manifest 的 {split} 在 splits 与 scenarios 两个视图中不一致。")

        for scenario_id, record in split_records.items():
            if record != scenario_view[scenario_id]:
                raise ValueError(
                    f"活动数据 Manifest 的 {split}/{scenario_id} 在 splits 与 scenarios 两个视图中不一致。"
                )
            if record["validation_status"] != "PASS":
                raise ValueError(f"活动数据 Manifest 的 {split}/{scenario_id} 未通过动态数据结构校验。")

            scenario_path = _resolve_dataset_record_path(dataset_dir, record)
            actual_hash = _sha256_file(scenario_path)
            if actual_hash != record["sha256"]:
                raise ValueError(
                    f"活动数据 Manifest 的 {split}/{scenario_id} SHA-256 不匹配："
                    f"声明 {record['sha256']}，实际 {actual_hash}。"
                )

            initial_orders, dynamic_events = load_dynamic_orders_and_events(scenario_path, "Sheet1")
            if len(initial_orders) != record["initial_orders"]:
                raise ValueError(
                    f"活动数据 Manifest 的 {split}/{scenario_id} 初始订单数不匹配："
                    f"声明 {record['initial_orders']}，实际 {len(initial_orders)}。"
                )
            if len(dynamic_events) != record["dynamic_events"]:
                raise ValueError(
                    f"活动数据 Manifest 的 {split}/{scenario_id} 动态事件数不匹配："
                    f"声明 {record['dynamic_events']}，实际 {len(dynamic_events)}。"
                )
        checked_by_split[split] = len(split_records)

    return checked_by_split


def _increment_count(counts: Dict[str, int], key: str) -> None:
    """Increment one ledger category without silently coercing its identifier."""
    if not isinstance(key, str) or not key:
        raise ValueError("V20261027 数据总账含空的事件分类标识。")
    counts[key] = counts.get(key, 0) + 1


def build_decoded_dataset_ledger(
    manifest: Dict[str, Any],
    dataset_dir: str,
    expected_scenarios: Dict[str, int],
) -> Dict[str, Any]:
    """Decode aggregate V20261027 facts from the hash-bound scenario workbooks.

    The formal Manifest deliberately stores compact scenario metadata rather
    than duplicating every event row.  This builder therefore establishes the
    aggregate ledger from the same authoritative loader that the trainer uses,
    after ``validate_dataset_manifest_integrity`` has verified each workbook
    hash and per-scenario cardinality.
    """
    split_summary: Dict[str, Dict[str, Any]] = {}
    total_scenarios = 0
    total_initial_orders = 0
    total_dynamic_events = 0
    event_class_counts: Dict[str, int] = {}
    inserted_order_process_type_counts: Dict[str, int] = {}
    forecast_status_counts: Dict[str, int] = {}

    for split, expected_count in expected_scenarios.items():
        records = manifest.get("splits", {}).get(split)
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(f"无法从活动 Manifest 建立 {split} 的数据总账。")

        split_initial_orders = 0
        split_dynamic_events = 0
        split_event_class_counts: Dict[str, int] = {}
        split_process_counts: Dict[str, int] = {}
        split_forecast_counts: Dict[str, int] = {}
        for record in records:
            scenario_path = _resolve_dataset_record_path(dataset_dir, record)
            initial_orders, dynamic_events = load_dynamic_orders_and_events(scenario_path, "Sheet1")
            split_initial_orders += len(initial_orders)
            split_dynamic_events += len(dynamic_events)
            for event in dynamic_events:
                _increment_count(split_event_class_counts, type(event).__name__)
                _increment_count(split_process_counts, event.order.process_type)
                _increment_count(
                    split_forecast_counts,
                    "forecasted" if event.is_forecasted else "unforecasted",
                )

        split_summary[split] = {
            "scenarios": len(records),
            "total_initial_orders": split_initial_orders,
            "total_dynamic_events": split_dynamic_events,
            "event_class_counts": dict(sorted(split_event_class_counts.items())),
            "inserted_order_process_type_counts": dict(sorted(split_process_counts.items())),
            "forecast_status_counts": dict(sorted(split_forecast_counts.items())),
        }
        total_scenarios += len(records)
        total_initial_orders += split_initial_orders
        total_dynamic_events += split_dynamic_events
        for key, value in split_event_class_counts.items():
            event_class_counts[key] = event_class_counts.get(key, 0) + value
        for key, value in split_process_counts.items():
            inserted_order_process_type_counts[key] = inserted_order_process_type_counts.get(key, 0) + value
        for key, value in split_forecast_counts.items():
            forecast_status_counts[key] = forecast_status_counts.get(key, 0) + value

    return {
        "total_scenarios": total_scenarios,
        "total_initial_orders": total_initial_orders,
        "total_dynamic_events": total_dynamic_events,
        "event_class_counts": dict(sorted(event_class_counts.items())),
        "inserted_order_process_type_counts": dict(sorted(inserted_order_process_type_counts.items())),
        "forecast_status_counts": dict(sorted(forecast_status_counts.items())),
        "split_summary": split_summary,
    }


def validate_dataset_ledger_integrity(
    manifest: Dict[str, Any],
    dataset_dir: str,
    expected_scenarios: Dict[str, int],
) -> Dict[str, Any]:
    """Reject a declared V20261027 aggregate ledger that differs from workbooks."""
    ledger = manifest.get("dataset_ledger")
    if not isinstance(ledger, dict):
        raise ValueError("活动数据 Manifest 缺少 V20261027 dataset_ledger。")
    missing = sorted(_DATASET_LEDGER_REQUIRED_FIELDS - set(ledger))
    if missing:
        raise ValueError(f"活动数据 Manifest 的 dataset_ledger 缺少字段：{missing}。")
    if ledger.get("ledger_version") != _DATASET_LEDGER_VERSION:
        raise ValueError("活动数据 Manifest 的 dataset_ledger 版本不受当前 V20261027 合同支持。")
    derivation = ledger.get("derivation")
    if not isinstance(derivation, dict) or not all(
        isinstance(derivation.get(field), str) and derivation[field]
        for field in ("source", "decoder", "scope")
    ):
        raise ValueError("活动数据 Manifest 的 dataset_ledger 缺少可审计的派生说明。")

    decoded = build_decoded_dataset_ledger(manifest, dataset_dir, expected_scenarios)
    for field, observed in decoded.items():
        declared = ledger.get(field)
        if declared != observed:
            raise ValueError(
                "活动数据 Manifest 的 dataset_ledger 与哈希校验后的工作簿不一致："
                f"{field} 声明 {declared!r}，实际 {observed!r}。"
            )
    return decoded


def load_active_dataset_contract() -> Dict[str, Any]:
    """Load and validate the sole active dynamic-experiment dataset contract.

    The project has one active dynamic dataset only.  Historical packages may
    remain on disk for audit and rollback, but they are intentionally not
    selectable by this trainer.
    """
    if not os.path.isfile(_ACTIVE_DATASET_POINTER):
        raise FileNotFoundError(
            "缺少活动动态实验数据合同："
            f"{_ACTIVE_DATASET_POINTER}"
        )

    with open(_ACTIVE_DATASET_POINTER, "r", encoding="utf-8") as f:
        contract = json.load(f)

    required = {
        "contract_version",
        "selection_policy",
        "dataset_relative_path",
        "manifest_filename",
        "manifest_version",
        "manifest_sha256",
        "source_workbook",
        "expected_scenarios",
        "data_evidence_status",
        "eligibility_policy",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"活动动态实验数据合同缺少字段：{missing}")
    if contract["selection_policy"] != "SOLE_ACTIVE_DYNAMIC_EXPERIMENT_DATASET":
        raise ValueError(
            "活动动态实验数据合同必须声明唯一选择政策 "
            "SOLE_ACTIVE_DYNAMIC_EXPERIMENT_DATASET。"
        )
    if contract["contract_version"] != contract["manifest_version"]:
        raise ValueError(
            "活动动态实验数据合同版本与其 Manifest 版本不一致："
            f"{contract['contract_version']} != {contract['manifest_version']}"
        )
    if contract["data_evidence_status"] != "PRE_FORMAL_ONLY_PROXY_DATA":
        raise ValueError(
            "活动动态实验数据合同必须声明 PRE_FORMAL_ONLY_PROXY_DATA，"
            "不得把代理动态场景表述为现场 MES 实测事件。"
        )
    policy_ref = contract["eligibility_policy"]
    if not isinstance(policy_ref, dict):
        raise ValueError("活动动态实验数据合同的 eligibility_policy 必须为对象。")
    policy_required = {"policy_id", "relative_path", "sha256"}
    policy_missing = sorted(policy_required - set(policy_ref))
    if policy_missing:
        raise ValueError(f"活动动态实验数据合同的 eligibility_policy 缺少字段：{policy_missing}")
    policy_path = os.path.abspath(os.path.join(_ROOT_DIR, policy_ref["relative_path"]))
    if os.path.commonpath([_ROOT_DIR, policy_path]) != _ROOT_DIR:
        raise ValueError(f"资格策略文件越出项目根目录：{policy_path}")
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"未找到活动资格策略文件：{policy_path}")
    if _sha256_file(policy_path) != policy_ref["sha256"]:
        raise ValueError("活动资格策略文件 SHA-256 与活动合同不一致。")
    with open(policy_path, "r", encoding="utf-8") as f:
        frozen_policy = json.load(f)
    if frozen_policy != eligibility_policy_descriptor():
        raise ValueError(
            "活动资格策略文件与 eligibility_builder 的运行时资格语义不一致；"
            "请同步冻结映射并更新合同哈希。"
        )
    if frozen_policy.get("policy_id") != policy_ref["policy_id"]:
        raise ValueError("活动资格策略 ID 与活动合同不一致。")

    dataset_dir = os.path.abspath(os.path.join(_ROOT_DIR, contract["dataset_relative_path"]))
    if os.path.commonpath([_ROOT_DIR, dataset_dir]) != _ROOT_DIR:
        raise ValueError(f"活动数据目录越出项目根目录：{dataset_dir}")
    manifest_path = os.path.join(dataset_dir, contract["manifest_filename"])
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"未找到活动数据 Manifest：{manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if contract["manifest_sha256"] != actual_manifest_sha256:
        raise ValueError(
            "活动动态实验数据合同的 Manifest SHA-256 不一致："
            f"声明 {contract['manifest_sha256']}，实际 {actual_manifest_sha256}。"
        )
    if manifest.get("manifest_version") != contract["manifest_version"]:
        raise ValueError(
            "活动数据 Manifest 版本与活动合同不一致："
            f"{manifest.get('manifest_version')!r} != {contract['manifest_version']!r}"
        )
    if manifest.get("source_workbook", {}).get("sha256") != contract["source_workbook"].get("sha256"):
        raise ValueError("活动数据 Manifest 的源工作簿 SHA-256 与活动合同不一致。")
    if manifest.get("expected_scenarios") != contract["expected_scenarios"]:
        raise ValueError("活动数据 Manifest 的 train/val/test 划分与活动合同不一致。")
    if manifest.get("data_evidence_status") != contract["data_evidence_status"]:
        raise ValueError("活动数据 Manifest 的数据证据状态与活动合同不一致。")
    if manifest.get("eligibility_policy") != policy_ref:
        raise ValueError("活动数据 Manifest 的资格策略绑定与活动合同不一致。")

    curriculum_policy = validate_scale_ranked_curriculum_policy(
        manifest,
        contract["expected_scenarios"],
    )
    checked_by_split = validate_dataset_manifest_integrity(
        manifest,
        dataset_dir,
        contract["expected_scenarios"],
    )
    decoded_ledger = validate_dataset_ledger_integrity(
        manifest,
        dataset_dir,
        contract["expected_scenarios"],
    )

    resolved = dict(contract)
    resolved["resolved_dataset_dir"] = dataset_dir
    resolved["resolved_manifest_path"] = manifest_path
    resolved["resolved_eligibility_policy_path"] = policy_path
    resolved["eligibility_policy_descriptor"] = frozen_policy
    resolved["manifest"] = manifest
    resolved["training_curriculum_policy"] = curriculum_policy
    resolved["verified_scenarios"] = checked_by_split
    resolved["verified_dataset_ledger"] = decoded_ledger
    return resolved


def compute_progressive_quotas(
    n_scenarios: int,
    total_episodes: int = 1800,
    base_quota: int = 300,
    step_quota: int = 0,
) -> List[int]:
    """计算各训练算例依次严格循环的训练轮数配额。

    在 N=18 个算例时，严格匹配三组数据集的训练设计：
      每个阶段 6 个算例，每算例 300 轮（共 1800 轮），三阶段总和 5400 轮。
    在 N=6 个算例时，对应单个独立阶段，配额为 [300, 300, 300, 300, 300, 300]，总和 1800 轮。
    """
    if n_scenarios <= 0:
        return []
    if n_scenarios == 1:
        return [int(total_episodes)]

    if n_scenarios == 6 and (total_episodes == 1800 or total_episodes is None):
        return [300, 300, 300, 300, 300, 300]

    if n_scenarios == 18 and (total_episodes == 5400 or total_episodes is None):
        return [300] * 18

    ideal = [base_quota + step_quota * i for i in range(n_scenarios)]
    if sum(ideal) == total_episodes:
        return ideal

    target_sum = int(total_episodes)
    per_sc = target_sum // n_scenarios
    quotas = [per_sc] * n_scenarios
    diff = target_sum - sum(quotas)
    if diff != 0:
        quotas[-1] += diff
    return quotas


def compute_stage_only_bounds(
    stage_index: int,
    n_scenarios: int = 18,
    stage_count: int = 3,
) -> Tuple[int, int]:
    """Return the zero-based half-open scenario interval for one formal stage.

    The 18 formal training scenarios are sorted by the manifest scale rank and
    split into three contiguous groups of six. Keeping the boundary in one
    helper prevents a stage-only run from silently selecting the first group
    when the requested target is a later group.
    """
    if stage_count <= 0 or n_scenarios <= 0 or n_scenarios % stage_count != 0:
        raise ValueError(
            "独立阶段训练要求训练场景数能被阶段数整除："
            f"n_scenarios={n_scenarios}, stage_count={stage_count}。"
        )
    if not 1 <= int(stage_index) <= stage_count:
        raise ValueError(
            f"阶段编号必须在 1..{stage_count} 内，收到 {stage_index!r}。"
        )
    width = n_scenarios // stage_count
    start = (int(stage_index) - 1) * width
    return start, start + width


def compute_stage_only_quotas(
    stage_index: int,
    n_scenarios: int = 18,
    total_episodes: int = 5400,
    base_quota: int = 300,
    step_quota: int = 0,
    stage_count: int = 3,
) -> List[int]:
    """Extract one stage's original quotas from the full formal curriculum.

    A standalone stage is an isolated training run, not a shortened new
    curriculum. Stage 3 therefore retains [470, 490, 510, 530, 550, 570]
    rather than recalculating six arbitrary quotas from a 3,120-episode total.
    """
    start, end = compute_stage_only_bounds(
        stage_index=stage_index,
        n_scenarios=n_scenarios,
        stage_count=stage_count,
    )
    full_quotas = compute_progressive_quotas(
        n_scenarios=n_scenarios,
        total_episodes=total_episodes,
        base_quota=base_quota,
        step_quota=step_quota,
    )
    return full_quotas[start:end]


def compute_tier_thresholds(quotas: List[int]) -> List[int]:
    """计算三个订单属性组的累积训练轮数阈值。"""
    n = len(quotas)
    if n <= 3 or n % 3 != 0:
        return [sum(quotas)]
    group_width = n // 3
    t1_sum = sum(quotas[:group_width])
    t2_sum = sum(quotas[group_width: 2 * group_width])
    t3_sum = sum(quotas[2 * group_width:])
    return [t1_sum, t1_sum + t2_sum, t1_sum + t2_sum + t3_sum]


def compute_transition_lr_multiplier(
    ep: int,
    thresholds: List[int],
    warmup_episodes: int = 100,
    min_multiplier: float = 0.30,
) -> float:
    """计算当前轮数在阶梯过渡期的学习率阻尼与平滑 Warmup 乘子。

    机制：
      在稳态运行区间，学习率倍率为 1.0；
      当刚跨越阶梯累积轮数阈值 T_k (如 1250, 3000, 5250) 时，
      由于算例规模与异构图拓扑跳跃，将学习率瞬时退火至 min_multiplier (0.30)，
      并在随后的 warmup_episodes (100) 轮内线性爬坡平滑恢复至 1.0，
      防止陌生的大规模图特征直接冲毁策略网络特征空间与信任域。
    """
    if ep <= 0 or not thresholds:
        return 1.0

    # 遍历前 3 个阶梯交界阈值（排除最后一个终点阈值）
    for t in thresholds[:-1]:
        if t < ep <= t + warmup_episodes:
            progress = (ep - t) / max(1, warmup_episodes)
            return float(min_multiplier + (1.0 - min_multiplier) * progress)

    return 1.0


class ProgressiveCurriculumSampler:
    """正式课程学习采样器：每阶段 6 个算例，每阶段 1800 轮（每算例 300 轮），三阶段总计 5400 轮，按规模升序严格循环遍历，每走完一轮循环（6个算例累积）集中更新一次 PPO"""

    def __init__(
        self,
        train_files: List[str],
        total_episodes: int = 1800,
        base_quota: int = 300,
        step_quota: int = 0,
        seed: int = DEFAULT_FORMAL_SEED,
        scenario_scale_keys: Optional[Dict[str, Tuple[int, int, str]]] = None,
        quota_override: Optional[List[int]] = None,
        stage_label: Optional[str] = None,
    ):
        supplied_files = list(train_files)
        if scenario_scale_keys is not None:
            missing_scale_keys = [path for path in supplied_files if path not in scenario_scale_keys]
            if missing_scale_keys:
                raise ValueError(
                    "按算例规模分配训练轮数时，每个训练文件必须具有规模排名键："
                    f"{missing_scale_keys}"
                )
            self.train_files = sorted(supplied_files, key=lambda path: scenario_scale_keys[path])
            self.scale_rank_keys = [scenario_scale_keys[path] for path in self.train_files]
        else:
            self.train_files = supplied_files
            self.scale_rank_keys = []
        self.n_total = len(train_files)
        self.total_episodes = int(total_episodes)
        if quota_override is None:
            self.quotas = compute_progressive_quotas(
                n_scenarios=self.n_total,
                total_episodes=self.total_episodes,
                base_quota=base_quota,
                step_quota=step_quota,
            )
        else:
            self.quotas = [int(value) for value in quota_override]
            if len(self.quotas) != self.n_total:
                raise ValueError(
                    "独立阶段配额数必须与训练场景数一致："
                    f"quotas={len(self.quotas)}, scenes={self.n_total}。"
                )
            if any(value <= 0 for value in self.quotas):
                raise ValueError("独立阶段配额必须全部为正整数。")
            if sum(self.quotas) != self.total_episodes:
                raise ValueError(
                    "独立阶段配额和必须等于总训练轮数："
                    f"sum={sum(self.quotas)}, total={self.total_episodes}。"
                )
        self.stage_label = stage_label
        self.thresholds = compute_tier_thresholds(self.quotas)
        self.scale_rank_plan = [
            {
                "rank": rank,
                "train_file": os.path.basename(path),
                "initial_orders": key[0],
                "dynamic_events": key[1],
                "scenario_id": key[2],
                "quota": self.quotas[rank],
            }
            for rank, (path, key) in enumerate(zip(self.train_files, self.scale_rank_keys))
        ]
        self.counts = [0] * self.n_total
        self.rng = random.Random(seed)

    def set_counts(self, counts: List[int]):
        """从历史断点恢复已采样的轮数统计"""
        self.counts = list(counts[:self.n_total])
        if len(self.counts) < self.n_total:
            self.counts.extend([0] * (self.n_total - len(self.counts)))

    def sample(self, ep: int, mode: str = "curriculum") -> Tuple[str, str]:
        if mode in ("stage_only", "round_robin", "curriculum", "sequential", "cyclic"):
            idx = (ep - 1) % self.n_total
            self.counts[idx] += 1
            cur_c = self.counts[idx]
            tar_q = self.quotas[idx] if idx < len(self.quotas) else (self.total_episodes // self.n_total)
            label = self.stage_label or "阶段"
            return self.train_files[idx], f"{label}[s{idx + 1}({cur_c}/{tar_q})]"

        if mode == "sequential":
            cum = 0
            chosen_idx = self.n_total - 1
            local_ep = 1
            for i, q in enumerate(self.quotas):
                if ep <= cum + q:
                    chosen_idx = i
                    local_ep = ep - cum
                    break
                cum += q
            self.counts[chosen_idx] += 1
            stage_tag = f"顺序递增[算例{chosen_idx+1}/{self.n_total},第{local_ep}/{self.quotas[chosen_idx]}轮]"
            return self.train_files[chosen_idx], stage_tag

        # 未提供规模键时采用均匀抽样；提供规模键后，
        # B 政策在任意场景数下都必须按规模排名与配额执行。
        if mode == "uniform" or (self.n_total <= 4 and not self.scale_rank_keys):
            idx = self.rng.randint(0, self.n_total - 1)
            self.counts[idx] += 1
            return self.train_files[idx], "均匀随机"

        # 三个订单属性组的连续平滑混合课程流。
        n = self.n_total
        if len(self.thresholds) < 3 or n < 3 or n % 3 != 0:
            avail = [i for i in range(n) if self.counts[i] < self.quotas[i]] or list(range(n))
            weights = [self.quotas[i] - self.counts[i] for i in avail]
            chosen_idx = self.rng.choices(avail, weights=weights, k=1)[0] if sum(weights) > 0 else self.rng.choice(avail)
            self.counts[chosen_idx] += 1
            cur_c = self.counts[chosen_idx]
            tar_q = self.quotas[chosen_idx]
            return self.train_files[chosen_idx], f"基础阶梯[算例{chosen_idx+1}({cur_c}/{tar_q})]"

        group_width = n // 3
        t1_end = group_width
        t2_end = 2 * group_width

        # 每个组的切换使用固定半窗的连续概率混合。
        w_trans = 100
        t1 = self.thresholds[0]
        t2 = self.thresholds[1]

        def _smooth_lambda(current_ep: int, threshold: int, target_prob: float = 0.70) -> float:
            """在 [threshold - w_trans, threshold + w_trans] 窗口内平滑余弦插值爬坡"""
            if current_ep < threshold - w_trans:
                return 0.0
            if current_ep > threshold + w_trans:
                return target_prob
            u = (current_ep - (threshold - w_trans)) / (2.0 * w_trans)
            import math
            return float(target_prob * (math.sin(0.5 * math.pi * u) ** 2))

        if ep < t1 - w_trans:
            tier_name = f"组1:小规模低负荷[0..{t1_end-1}]"
            avail = [i for i in range(0, t1_end) if self.counts[i] < self.quotas[i]]
        elif ep < t2 - w_trans:
            tier_name = f"组2:中规模混合负荷[{t1_end}..{t2_end-1}]"
            new_avail = [i for i in range(t1_end, t2_end) if self.counts[i] < self.quotas[i]]
            old_avail = [i for i in range(0, t1_end) if self.counts[i] < self.quotas[i]]
            p_new = _smooth_lambda(ep, t1, target_prob=0.70)
            if new_avail and (not old_avail or self.rng.random() < p_new):
                avail = new_avail
            elif old_avail:
                avail = old_avail
            else:
                avail = new_avail or [i for i in range(t2_end) if self.counts[i] < self.quotas[i]]
        else:
            tier_name = f"组3:大规模高负荷高扰动[{t2_end}..{n-1}]"
            new_avail = [i for i in range(t2_end, n) if self.counts[i] < self.quotas[i]]
            old_avail = [i for i in range(0, t2_end) if self.counts[i] < self.quotas[i]]
            p_new = _smooth_lambda(ep, t2, target_prob=0.70)
            if new_avail and (not old_avail or self.rng.random() < p_new):
                avail = new_avail
            elif old_avail:
                avail = old_avail
            else:
                avail = new_avail or [i for i in range(n) if self.counts[i] < self.quotas[i]]

        if not avail:
            avail = [i for i in range(n) if self.counts[i] < self.quotas[i]]
        if not avail:
            avail = list(range(n))

        weights = [self.quotas[i] - self.counts[i] for i in avail]
        if sum(weights) > 0:
            chosen_idx = self.rng.choices(avail, weights=weights, k=1)[0]
        else:
            chosen_idx = self.rng.choice(avail)

        self.counts[chosen_idx] += 1
        cur_c = self.counts[chosen_idx]
        tar_q = self.quotas[chosen_idx]
        stage_tag = f"{tier_name}[算例{chosen_idx+1}({cur_c}/{tar_q})]"
        return self.train_files[chosen_idx], stage_tag


class DynamicMultiScenarioTrainer:
    """多场景 HPPO 联合训练驱动器"""

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        sheet_name: str = "Sheet1",
        pool_type: str = POOL_D,
        d_h: int = 32,
        lr: float = FORMAL_DYNAMIC_LR,
        placement_lr: float = FORMAL_DYNAMIC_PLACEMENT_LR,
        ppo_epochs: int = 4,
        reward_scale: float = FORMAL_DYNAMIC_REWARD_SCALE,
        checkpoint_in: Optional[str] = None,
        device: Optional[torch.device] = None,
        max_train_scenarios: Optional[int] = None,
        max_val_scenarios: Optional[int] = None,
        single_scenario_file: Optional[str] = None,
        stage_only: Optional[int] = None,
        rollout_scenarios: int = 6,
        reward_mode: str = "scaled_relative",
        target_kl: float = 0.05,
        max_kl: float = 0.15,
        base_quota: int = 300,
        step_quota: int = 0,
        c_ent: float = FORMAL_DYNAMIC_C_ENT,
    ):
        self.dataset_contract = load_active_dataset_contract()
        active_dataset_dir = self.dataset_contract["resolved_dataset_dir"]
        if dataset_dir is None:
            dataset_dir = active_dataset_dir
        elif not os.path.isabs(dataset_dir):
            dataset_dir = os.path.join(_ROOT_DIR, dataset_dir)
        self.dataset_dir = os.path.abspath(dataset_dir)
        if os.path.normcase(self.dataset_dir) != os.path.normcase(active_dataset_dir):
            raise ValueError(
                "当前训练器只接受活动动态实验数据集 "
                f"{self.dataset_contract['contract_version']}：{active_dataset_dir}；"
                f"不接受 {self.dataset_dir}。"
            )
        self.dataset_manifest = self.dataset_contract["manifest"]
        self.training_curriculum_policy = self.dataset_contract["training_curriculum_policy"]
        self.sheet_name = sheet_name
        self.pool_type = pool_type
        self.reward_scale = max(1.0, float(reward_scale))
        self.ppo_epochs = max(1, int(ppo_epochs))
        self.rollout_scenarios = max(1, int(rollout_scenarios))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_quota = int(base_quota)
        self.step_quota = int(step_quota)
        self.c_ent = float(c_ent)
        self.curriculum_sampler: Optional[ProgressiveCurriculumSampler] = None
        self.train_scale_keys: Dict[str, Tuple[int, int, str]] = {}
        self.stage_only = int(stage_only) if stage_only is not None else None
        self.stage_quota_override: Optional[List[int]] = None
        self.stage_label: Optional[str] = None
        self.model_selection_protocol = "feasibility_then_mean_cmax_v1"
        self.model_selection_val_files: List[str] = []

        if self.stage_only is not None:
            if single_scenario_file:
                raise ValueError("--stage-only 不能与 --scenario-file 同时使用。")
            if max_train_scenarios is not None or max_val_scenarios is not None:
                raise ValueError("--stage-only 会按正式阶段边界选择场景，不能再叠加 --max-*-scenarios。")

        # 加载数据集文件清单
        if single_scenario_file:
            if not os.path.isabs(single_scenario_file):
                single_scenario_file = os.path.join(_ROOT_DIR, single_scenario_file)
            if os.path.exists(single_scenario_file):
                single_abs = os.path.abspath(single_scenario_file)
                if os.path.commonpath([self.dataset_dir, single_abs]) != self.dataset_dir:
                    raise ValueError(
                        "单算例调试文件必须属于当前活动动态实验数据集："
                        f"{self.dataset_dir}"
                    )
                self.train_files = [single_abs]
                self.val_files = [single_abs]
                train_records = self.dataset_manifest.get("splits", {}).get("train", [])
                for record in train_records:
                    record_path = os.path.abspath(os.path.join(self.dataset_dir, record["filepath"]))
                    if os.path.normcase(record_path) == os.path.normcase(single_abs):
                        self.train_scale_keys[single_abs] = _scenario_scale_rank_key(record)
                        break
                print(f"模式：单算例专属训练，目标算例: {single_abs}")
            else:
                raise FileNotFoundError(f"未找到指定的单算例文件: {single_scenario_file}")
        else:
            all_train_files = self._discover_split_files("train")
            if self.stage_only is None:
                self.train_files = all_train_files
                self.val_files = self._discover_split_files("val")
                if max_train_scenarios is not None and max_train_scenarios > 0:
                    self.train_files = self.train_files[:max_train_scenarios]
                if max_val_scenarios is not None and max_val_scenarios > 0:
                    self.val_files = self.val_files[:max_val_scenarios]
            else:
                n_formal = int(self.training_curriculum_policy["formal_train_scenarios"])
                if len(all_train_files) != n_formal:
                    raise ValueError(
                        "活动 Manifest 的训练场景数与正式课程策略不一致："
                        f"files={len(all_train_files)}, formal_train_scenarios={n_formal}。"
                    )
                start, end = compute_stage_only_bounds(
                    stage_index=self.stage_only,
                    n_scenarios=n_formal,
                )
                self.train_files = all_train_files[start:end]
                self.val_files = self._discover_stage_validation_files(self.stage_only)
                self.stage_quota_override = compute_stage_only_quotas(
                    stage_index=self.stage_only,
                    n_scenarios=n_formal,
                    total_episodes=int(self.training_curriculum_policy["formal_total_episodes"]),
                    base_quota=int(self.training_curriculum_policy["base_quota"]),
                    step_quota=int(self.training_curriculum_policy["quota_step"]),
                )
                self.stage_label = (
                    "独立阶段3:大规模高负荷高扰动"
                    if self.stage_only == 3
                    else f"独立阶段{self.stage_only}"
                )
                print(
                    f"模式：独立 Stage {self.stage_only} 训练（非 18 场景正式主实验续跑）；"
                    f"训练场景排名 {start + 1}..{end}，固定配额 {self.stage_quota_override}，"
                    f"总轮数 {sum(self.stage_quota_override)}。"
                )

        if not self.train_files:
            raise FileNotFoundError(f"在 {self.dataset_dir} 中未找到训练场景文件。")
        if self.stage_only is not None:
            self.model_selection_protocol = stage_validation_protocol(self.stage_only)
            self.model_selection_val_files = [os.path.basename(path) for path in self.val_files]

        # 使用首个训练文件的初始订单初始化 HGNN 与 Policy 骨干网络
        sample_init_orders, _ = load_dynamic_orders_and_events(self.train_files[0], self.sheet_name)
        self.reward_mode = str(reward_mode)
        self.target_kl = float(target_kl)
        self.max_kl = float(max_kl)
        self.trainer = HPPOTrainer(
            orders=sample_init_orders,
            pool_type=self.pool_type,
            d_h=d_h,
            lr=lr,
            placement_lr=placement_lr,
            ppo_epochs=self.ppo_epochs,
            c_ent=self.c_ent,
            device=self.device,
            reward_mode=self.reward_mode,
            target_kl=self.target_kl,
            max_kl=self.max_kl,
        )

        self.checkpoint_payload: Optional[Dict[str, Any]] = None
        if checkpoint_in and os.path.exists(checkpoint_in):
            print(f"正在加载已有权重: {checkpoint_in}")
            self.checkpoint_payload = self.trainer.load_checkpoint(checkpoint_in, load_optimizer=True)

    def _discover_split_files(self, split: str) -> List[str]:
        """仅从已锁定的活动 V20261027 Manifest 解析场景文件。"""
        expected_count = self.dataset_contract["expected_scenarios"].get(split)
        if expected_count is None:
            raise ValueError(f"活动动态实验数据合同未定义 split={split!r}。")
        records = self.dataset_manifest.get("splits", {}).get(split, [])
        if len(records) != expected_count:
            raise ValueError(
                f"活动数据 Manifest 的 {split} 场景数错误："
                f"期望 {expected_count}，实际 {len(records)}。"
            )
        if split == "train":
            records = sorted(records, key=_scenario_scale_rank_key)
        files = [
            os.path.abspath(os.path.join(self.dataset_dir, record["filepath"]))
            for record in records
        ]
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"活动数据 Manifest 的 {split} 场景文件缺失：{missing}"
            )
        if split == "train":
            self.train_scale_keys = {
                path: _scenario_scale_rank_key(record)
                for path, record in zip(files, records)
            }
        return files

    def _discover_stage_validation_files(self, stage_index: int) -> List[str]:
        """Resolve the fixed 01/23/45 validation window for one Stage."""
        stage = int(stage_index)
        expected_filenames = STAGE_VALIDATION_FILENAMES.get(stage)
        if expected_filenames is None:
            raise ValueError(f"未知独立阶段: {stage_index!r}。")
        records = self.dataset_manifest.get("splits", {}).get("val", [])
        paths_by_name: Dict[str, List[str]] = {}
        for record in records:
            name = os.path.basename(str(record.get("filepath", "")))
            path = os.path.abspath(os.path.join(self.dataset_dir, record["filepath"]))
            paths_by_name.setdefault(name, []).append(path)

        selected: List[str] = []
        for filename in expected_filenames:
            matches = paths_by_name.get(filename, [])
            if len(matches) != 1:
                raise ValueError(
                    f"独立 Stage {stage} 的固定验证窗口要求且仅要求一个文件: "
                    f"{filename}; 实际找到 {len(matches)} 个。"
                )
            if not os.path.isfile(matches[0]):
                raise FileNotFoundError(f"独立 Stage {stage} 验证场景缺失: {matches[0]}")
            selected.append(matches[0])
        return selected

    def _sample_curriculum_scenario(
        self,
        ep: int,
        total_episodes: int,
        rng: random.Random,
        mode: str = "curriculum",
    ) -> Tuple[str, str]:
        """按三组规模顺序执行 18 场景、5400 episode 的正式课程采样。"""
        if self.curriculum_sampler is None:
            self.curriculum_sampler = ProgressiveCurriculumSampler(
                train_files=self.train_files,
                total_episodes=total_episodes,
                base_quota=self.base_quota,
                step_quota=self.step_quota,
                seed=DEFAULT_FORMAL_SEED,
                scenario_scale_keys=self.train_scale_keys or None,
                quota_override=self.stage_quota_override,
                stage_label=self.stage_label,
            )
        effective_mode = "stage_only" if self.stage_only is not None else mode
        return self.curriculum_sampler.sample(ep, mode=effective_mode)

    def _new_evaluation_manager(self, scenario_file: str) -> DynamicEventManager:
        """Build one independent state trajectory for a fixed validation scenario."""
        init_orders, events = load_dynamic_orders_and_events(scenario_file, self.sheet_name)
        return DynamicEventManager(
            pool_type=self.pool_type,
            initial_orders=init_orders,
            events=events,
            instance_name=self.sheet_name,
            formal_mode=True,
        )

    def evaluate_on_val(self) -> Dict[str, Any]:
        """Evaluate fixed validation scenarios without altering PPO rollout state.

        每个场景只执行一条确定性 Actor 轨迹。无参考轨、无质量门、无替代交付轨。
        任一事件失败即将该场景记为自主失败，失败场景不伪造最终可行 Cmax。

        All Actor calls use ``record_rollout=False`` and ``clear_buffer=False``.
        Consequently evaluation is read-only with respect to existing training
        buffers, optimizer state, and the next PPO batch.
        """
        if not self.val_files:
            return {
                "count": 0,
                "mean_cmax": float("inf"),
                "mean_actor_cmax": float("inf"),
                "autonomous_feasibility_rate": 0.0,
                "details": [],
            }

        results: List[Dict[str, Any]] = []
        all_raw_cmax: List[float] = []
        successful_scenarios = 0

        # Evaluation must not update batch-normalization/dropout state either.
        eval_modules = (
            self.trainer.hgnn,
            self.trainer.manager_policy,
            self.trainer.worker_policy,
            self.trainer.shared_critic,
        )
        was_training = [module.training for module in eval_modules]
        for module in eval_modules:
            module.eval()

        try:
            with torch.inference_mode():
                for v_file in self.val_files:
                    raw_manager = self._new_evaluation_manager(v_file)

                    raw_decisions = self.trainer.collect_dynamic_episode(
                        raw_manager,
                        deterministic=True,
                        clear_buffer=False,
                        record_rollout=False,
                    )
                    expected_events = len(raw_manager.events)
                    autonomous_success = (
                        len(raw_decisions) == expected_events
                        and all(decision.committed for decision in raw_decisions)
                        and not raw_manager.pending_event_ids
                        and raw_manager.active_context is None
                    )
                    raw_cmax = float(raw_manager.current_solution.makespan) if autonomous_success else float("inf")
                    if autonomous_success:
                        successful_scenarios += 1
                        all_raw_cmax.append(raw_cmax)

                    results.append({
                        "file": os.path.basename(v_file),
                        "final_cmax": raw_cmax,
                        "actor_cmax": raw_cmax,
                        "raw_actor_cmax": raw_cmax,
                        "autonomous_success": autonomous_success,
                        "event_count": len(raw_decisions),
                        "raw_event_count": len(raw_decisions),
                        "failed_event_count": sum(int(not d.committed) for d in raw_decisions),
                    })
        finally:
            for module, training in zip(eval_modules, was_training):
                module.train(training)

        mean_raw_cmax = sum(all_raw_cmax) / len(all_raw_cmax) if all_raw_cmax else float("inf")
        selection_summary = None
        if self.stage_only is not None:
            selection_summary = summarize_stage_validation_selection(self.stage_only, results)

        return {
            "count": len(results),
            "mean_cmax": mean_raw_cmax,
            "mean_actor_cmax": mean_raw_cmax,
            "autonomous_feasibility_rate": successful_scenarios / max(1, len(results)),
            "details": results,
            "selection_protocol": self.model_selection_protocol,
            "selection_summary": selection_summary,
        }

    def train(
        self,
        total_episodes: Optional[int] = None,
        episodes_per_scenario: int = 400,
        eval_interval: int = 20,
        checkpoint_out: Optional[str] = None,
        latest_checkpoint_out: Optional[str] = None,
        log_out: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        curriculum_mode: str = "curriculum",
        resume: bool = False,
        seed: int = DEFAULT_FORMAL_SEED,
    ) -> Dict[str, Any]:
        """运行多场景联合训练主循环。

        核心工程能力：
          1. 课程学习与轮询调度：支持三组订单属性渐进采样与 round_robin 轮询调度；
          2. 正式预生成 Manifest 固定 18 训练场景合计 5400 轮（每阶段 1800 轮，6 个算例循环 300 次，每走完一轮循环 6 个算例集中更新一次 PPO，共 300 次 PPO更新）；
          3. 实时中途曲线落盘：每个 Episode 结束后立即将训练轨迹与评估曲线刷新写入 log_out；
          4. 支持优雅随时暂停：随时按 Ctrl+C 或创建 'pause.flag' 标记文件安全挂起，不丢数据并自动保存中断断点；
          5. 支持断点无缝续训：指定 resume=True 自动载入历史轮次与日志向后继续推进，曲线无缝衔接；
          6. Checkpoint 先按完整自主可行率、再按成功场景平均 Cmax 排序。
        """
        random.seed(seed)
        torch.manual_seed(seed)
        rng = random.Random(seed)

        n_scenes = len(self.train_files)
        if total_episodes is None or total_episodes <= 0:
            total_episodes = n_scenes * episodes_per_scenario
        else:
            episodes_per_scenario = max(1, total_episodes // max(1, n_scenes))

        best_autonomous_feasibility_rate = -1.0
        best_val_cmax = float("inf")
        best_worst_val_cmax = float("inf")
        history: List[Dict[str, Any]] = []
        val_history: List[Dict[str, Any]] = []
        selection_switch: Optional[Dict[str, Any]] = None
        start_episode = 1
        manifest_path = os.path.abspath(self.dataset_contract["resolved_manifest_path"])
        method_id, code_id = resolve_trainer_identity(self.trainer)
        checkpoint_contract_meta = {
            "method_id": method_id,
            "code_id": code_id,
            "protocol_version": FORMAL_DYNAMIC_PROTOCOL_VERSION,
            "initial_solution_builder_version": INITIAL_SOLUTION_BUILDER_VERSION,
            "initial_solution_dispatch_rule": INITIAL_SOLUTION_DISPATCH_RULE,
            "stage": self.stage_only,
            "seed": int(seed),
            "dataset_contract_version": self.dataset_contract["contract_version"],
            "dataset_manifest_version": self.dataset_contract["manifest_version"],
            "dataset_manifest_path": manifest_path,
            "dataset_manifest_sha256": _sha256_file(manifest_path),
            "active_package_path": os.path.abspath(_ACTIVE_DATASET_POINTER),
            "active_package_sha256": _sha256_file(_ACTIVE_DATASET_POINTER),
            "split_learning_rate": self.trainer.lr,
            "placement_learning_rate": self.trainer.worker_lr,
        }
        dual_actor_semantics_version = getattr(
            self.trainer, "dual_actor_semantics_version", None
        )
        if dual_actor_semantics_version is not None:
            checkpoint_contract_meta["dual_actor_semantics_version"] = str(
                dual_actor_semantics_version
            )

        # 尝试断点续训恢复。日志和 latest checkpoint 分别原子落盘，
        # 进程若恰好在两次写入之间中断，checkpoint 最多可能比日志领先 1 轮。
        if resume and log_out and os.path.exists(log_out):
            try:
                with open(log_out, "r", encoding="utf-8") as f:
                    past_report = json.load(f)
                history = past_report.get("training_history", [])
                val_history = past_report.get("validation_history", [])
                saved_protocol = past_report.get("model_selection_protocol")
                if self.stage_only is not None:
                    if saved_protocol != self.model_selection_protocol:
                        raise RuntimeError(
                            f"Stage {self.stage_only} 验证选模协议与历史日志不一致；"
                            "禁止把旧的单场景指标混入新的两场景综合选模。"
                        )
                    saved_val_files = list(past_report.get("model_selection_val_files", []))
                    if saved_val_files != self.model_selection_val_files:
                        raise RuntimeError(
                            f"Stage {self.stage_only} 验证文件与历史日志不一致："
                            f"saved={saved_val_files}, active={self.model_selection_val_files}"
                        )
                best_autonomous_feasibility_rate = past_report.get(
                    "best_autonomous_feasibility_rate", -1.0
                )
                best_val_cmax = past_report.get("best_val_cmax", float("inf"))
                best_worst_val_cmax = past_report.get("best_worst_val_cmax", float("inf"))
                selection_switch = past_report.get("model_selection_switch")
                if history:
                    log_episode = int(history[-1]["episode"])
                    checkpoint_episode = int(
                        (self.checkpoint_payload or {}).get("episode", log_episode)
                    )
                    if checkpoint_episode < log_episode or checkpoint_episode > log_episode + 1:
                        raise RuntimeError(
                            "断点与日志不一致："
                            f"checkpoint Episode={checkpoint_episode}, log Episode={log_episode}"
                        )
                    start_episode = max(log_episode, checkpoint_episode) + 1
                    print(
                        f"  >>> [Resume] 成功从已有日志恢复！续训起始轮数: Episode {start_episode}/{total_episodes} "
                        f"(已继承历史样本 {len(history)} 条, 最优自主可行率={best_autonomous_feasibility_rate * 100:.1f}%)"
                    )
                    if checkpoint_episode == log_episode + 1:
                        print(
                            f"  >>> [Resume] checkpoint 已包含 Episode {checkpoint_episode} 的权重状态；"
                            "该轮日志在中断前尚未落盘，不伪造补写指标。"
                        )
            except Exception as e:
                raise RuntimeError(
                    f"续训恢复失败：{e}。已停止启动以防止覆盖历史；"
                    "如确需从头训练，请显式使用 --fresh-start。"
                ) from e

        if self.curriculum_sampler is None:
            self.curriculum_sampler = ProgressiveCurriculumSampler(
                train_files=self.train_files,
                total_episodes=total_episodes,
                base_quota=self.base_quota,
                step_quota=self.step_quota,
                seed=seed,
                scenario_scale_keys=self.train_scale_keys or None,
                quota_override=self.stage_quota_override,
                stage_label=self.stage_label,
            )
        if resume and history:
            counts = [0] * len(self.train_files)
            train_basenames = [os.path.basename(f) for f in self.train_files]
            for h in history:
                tf = h.get("train_file")
                if tf in train_basenames:
                    counts[train_basenames.index(tf)] += 1
            self.curriculum_sampler.set_counts(counts)
            print(
                f"  >>> [Resume] 成功恢复 {len(self.train_files)} 个训练算例的采样计数，"
                f"当前已完成轮数: {sum(counts)}/{total_episodes}"
            )

        writer = None
        if tensorboard_dir:
            # 统一锁定单一运行目录 active_run：
            run_dir = os.path.join(tensorboard_dir, "active_run")
            if not resume:
                # 全新开始新进程时，清理历史事件文件，确保 TensorBoard 曲线从 Episode 1 全新绘制
                if os.path.exists(run_dir):
                    for f in os.listdir(run_dir):
                        if f.startswith("events.out.tfevents"):
                            try:
                                os.remove(os.path.join(run_dir, f))
                            except Exception:
                                pass
            os.makedirs(run_dir, exist_ok=True)
            if SummaryWriter is not None:
                writer = SummaryWriter(log_dir=run_dir)
                print(f"TensorBoard 实时监控已就绪，统一日志目录: {run_dir}")
                # 仅在全新子目录（无已有事件文件）且存在历史日志时，进行全量历史指标无缝回填
                has_existing_events = any(f.startswith("events.out.tfevents") for f in os.listdir(run_dir))
                if resume and history and not has_existing_events:
                    print(f"  >>> [TensorBoard] 正在快速同步回填历史 {len(history)} 轮训练曲线与 {len(val_history)} 轮评估曲线...")
                    for h in history:
                        h_ep = h.get("episode", 0)
                        if "final_cmax" in h:
                            writer.add_scalar("Train/Makespan_Cmax", h["final_cmax"], h_ep)
                        if "mean_reward" in h:
                            writer.add_scalar("Train/Mean_Reward", h["mean_reward"], h_ep)
                        if "trainable_events" in h:
                            writer.add_scalar("Train/Trainable_Events", h["trainable_events"], h_ep)
                        if "committed_events" in h:
                            writer.add_scalar("Train/Committed_Events", h["committed_events"], h_ep)
                        loss_h = h.get("loss", {})
                        if loss_h and bool(h.get("ppo_updated", True)):
                            if "total_loss" in loss_h:
                                writer.add_scalar("Train/Total_Loss", loss_h.get("total_loss", 0.0), h_ep)
                            if "m_policy_loss" in loss_h:
                                writer.add_scalar("Train/Manager_Policy_Loss", loss_h.get("m_policy_loss", 0.0), h_ep)
                            if "m_val_loss" in loss_h:
                                writer.add_scalar("Train/Manager_Value_Loss", loss_h.get("m_val_loss", 0.0), h_ep)
                            if "w_policy_loss" in loss_h:
                                writer.add_scalar("Train/Worker_Policy_Loss", loss_h.get("w_policy_loss", 0.0), h_ep)
                            if "w_val_loss" in loss_h:
                                writer.add_scalar("Train/Worker_Value_Loss", loss_h.get("w_val_loss", 0.0), h_ep)
                            if "m_clip_fraction" in loss_h:
                                writer.add_scalar("Train/Manager_Clip_Fraction", loss_h.get("m_clip_fraction", 0.0), h_ep)
                            if "w_clip_fraction" in loss_h:
                                writer.add_scalar("Train/Worker_Clip_Fraction", loss_h.get("w_clip_fraction", 0.0), h_ep)
                            if "m_approx_kl" in loss_h:
                                writer.add_scalar("Train/Manager_Approx_KL", loss_h.get("m_approx_kl", 0.0), h_ep)
                            if "w_approx_kl" in loss_h:
                                writer.add_scalar("Train/Worker_Approx_KL", loss_h.get("w_approx_kl", 0.0), h_ep)
                            if "m_ratio_mean" in loss_h:
                                writer.add_scalar("Train/Manager_Ratio_Mean", loss_h.get("m_ratio_mean", 1.0), h_ep)
                            if "w_ratio_mean" in loss_h:
                                writer.add_scalar("Train/Worker_Ratio_Mean", loss_h.get("w_ratio_mean", 1.0), h_ep)
                            if "grad_norm" in loss_h:
                                writer.add_scalar("Train/Grad_Norm", loss_h.get("grad_norm", 0.0), h_ep)
                            if "c_ent" in loss_h:
                                writer.add_scalar("Train/Entropy_Coeff", float(loss_h.get("c_ent", 0.0)), h_ep)
                            if "kl_early_stopped" in loss_h:
                                writer.add_scalar("Train/KL_Early_Stopped", float(loss_h.get("kl_early_stopped", 0.0)), h_ep)
                            if "ppo_epochs_completed" in loss_h:
                                writer.add_scalar("Train/PPO_Epochs_Completed", float(loss_h.get("ppo_epochs_completed", 4.0)), h_ep)
                    for vh in val_history:
                        vh_ep = vh.get("episode", 0)
                        if "mean_actor_cmax" in vh:
                            writer.add_scalar("Val/Actor_Makespan_Cmax", vh["mean_actor_cmax"], vh_ep)
                        if "autonomous_feasibility_rate" in vh:
                            writer.add_scalar(
                                "Val/Autonomous_Feasibility_Rate_Percent",
                                vh["autonomous_feasibility_rate"] * 100.0,
                                vh_ep,
                            )
                    writer.flush()
                    print("  >>> [TensorBoard] 历史全量曲线同步完成！")
            else:
                print("未检测到 torch.utils.tensorboard，跳过 TensorBoard 写入。")

        print(
            f"=== 开始 HPPO 多场景训练: 训练算例数 {n_scenes}, 每订单/算例目标 {episodes_per_scenario} 轮, "
            f"计划总轮数 {total_episodes} (起始 Ep {start_episode}), 模式: {curriculum_mode.upper()}, "
            f"验证算例: {len(self.val_files)} ==="
        )
        print("提示：支持随时按 Ctrl+C 或创建 'pause.flag' 优雅暂停训练；训练曲线每轮实时原子落盘，绝不丢数据。\n")

        def _persist_live_log(current_ep: int, is_paused: bool = False):
            """实时将当前曲线与总结写入 JSON 文件，采用原子写出防止中断时损坏。"""
            if not log_out:
                return
            log_out_abs = os.path.abspath(log_out)
            os.makedirs(os.path.dirname(log_out_abs), exist_ok=True)
            report_data = {
                "total_episodes": total_episodes,
                "episodes_per_scenario": episodes_per_scenario,
                "current_episode": current_ep,
                "status": "PAUSED" if is_paused else ("COMPLETED" if current_ep >= total_episodes else "RUNNING"),
                "curriculum_mode": curriculum_mode,
                "run_scope": "INDEPENDENT_STAGE_ONLY" if self.stage_only is not None else "FULL_FORMAL_CURRICULUM",
                "stage_only": self.stage_only,
                "method_id": method_id,
                "code_id": code_id,
                "protocol_version": FORMAL_DYNAMIC_PROTOCOL_VERSION,
                "initial_solution_builder_version": INITIAL_SOLUTION_BUILDER_VERSION,
                "initial_solution_dispatch_rule": INITIAL_SOLUTION_DISPATCH_RULE,
                "dual_actor_semantics_version": dual_actor_semantics_version,
                "split_learning_rate": self.trainer.lr,
                "placement_learning_rate": self.trainer.worker_lr,
                "seed": int(seed),
                # 实验复现绑定：B 规则、Manifest 版本及源输入哈希必须和
                # 每一份逐轮日志一起落盘，不能仅由目录名称或人工说明推断。
                "dataset_contract_version": self.dataset_contract["contract_version"],
                "dataset_manifest_version": self.dataset_contract["manifest_version"],
                "dataset_manifest_path": self.dataset_contract["resolved_manifest_path"],
                "dataset_manifest_sha256": checkpoint_contract_meta["dataset_manifest_sha256"],
                "active_package_path": checkpoint_contract_meta["active_package_path"],
                "active_package_sha256": checkpoint_contract_meta["active_package_sha256"],
                "source_workbook_sha256": self.dataset_contract["source_workbook"]["sha256"],
                "training_curriculum_policy": self.training_curriculum_policy,
                "train_scenes_count": len(self.train_files),
                "val_scenes_count": len(self.val_files),
                "curriculum_quotas": self.curriculum_sampler.quotas if self.curriculum_sampler else [],
                "curriculum_counts": self.curriculum_sampler.counts if self.curriculum_sampler else [],
                "curriculum_scale_rank_plan": self.curriculum_sampler.scale_rank_plan if self.curriculum_sampler else [],
                "model_selection_protocol": self.model_selection_protocol,
                "model_selection_val_files": self.model_selection_val_files,
                "best_autonomous_feasibility_rate": best_autonomous_feasibility_rate,
                "best_val_cmax": best_val_cmax,
                "best_worst_val_cmax": best_worst_val_cmax if best_worst_val_cmax != float("inf") else None,
                "model_selection_switch": selection_switch,
                "reward_scale": self.reward_scale,
                "checkpoint_out": checkpoint_out,
                "latest_checkpoint_out": latest_checkpoint_out,
                "training_history": history,
                "validation_history": val_history,
            }
            tmp_path = log_out_abs + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2)
                if os.path.exists(log_out_abs):
                    try:
                        os.replace(tmp_path, log_out_abs)
                    except OSError:
                        with open(log_out_abs, "w", encoding="utf-8") as f:
                            json.dump(report_data, f, ensure_ascii=False, indent=2)
                else:
                    os.rename(tmp_path, log_out_abs)
            except Exception as exc:
                print(f"[警告] 实时日志落盘波动: {exc}")

        pause_flag_file = os.path.join(_ROOT_DIR, "pause.flag")
        paused_requested = False
        last_executed_ep = start_episode - 1
        scenarios_in_batch = 0
        last_loss_dict: Dict[str, float] = {}

        try:
            for ep in range(start_episode, total_episodes + 1):
                # 检测外部文件暂停指令
                if os.path.exists(pause_flag_file):
                    print(f"\n[检测到 pause.flag 暂停请求文件] 正在优雅暂停训练...")
                    try:
                        os.remove(pause_flag_file)
                    except Exception:
                        pass
                    paused_requested = True
                    break
                last_executed_ep = ep

                # 依据 Bengio 2009 课程学习难度阶梯挑选训练场景
                train_file, stage_tag = self._sample_curriculum_scenario(
                    ep=ep,
                    total_episodes=total_episodes,
                    rng=rng,
                    mode=curriculum_mode,
                )
                init_orders, events = load_dynamic_orders_and_events(train_file, self.sheet_name)
                manager = DynamicEventManager(
                    pool_type=self.pool_type,
                    initial_orders=init_orders,
                    events=events,
                    instance_name=self.sheet_name,
                    formal_mode=True,
                )

                # 多场景经验累积机制 (Rollout Scenarios Accumulation)
                # 仅在每个累积周期的首个场景清理经验池，跨多个场景收集轨迹，以增强 PPO batch 统计鲁棒性与重要性采样方差
                scenarios_in_batch += 1
                is_batch_start = (scenarios_in_batch == 1)

                # 收集纯自主同策略探索轨迹；无质量门、无参考解回退。
                decisions = self.trainer.collect_dynamic_episode(
                    manager,
                    deterministic=False,
                    reward_scale=self.reward_scale,
                    clear_buffer=is_batch_start,
                )

                # 检查是否满足多场景累积更新条件
                should_update = (scenarios_in_batch >= self.rollout_scenarios) or (ep == total_episodes)
                batch_scenarios_before_update = scenarios_in_batch
                manager_buffer_before_update = len(self.trainer.manager_buffer)
                worker_buffer_before_update = len(self.trainer.worker_buffer)
                if should_update:
                    # 动态阶段过渡学习率调节 (Transition LR Annealing & Warmup)
                    if self.stage_only is not None:
                        cur_lr_mult = 1.0
                    elif self.curriculum_sampler and hasattr(self.curriculum_sampler, "thresholds"):
                        cur_lr_mult = compute_transition_lr_multiplier(ep, self.curriculum_sampler.thresholds)
                    else:
                        cur_lr_mult = 1.0
                    if hasattr(self.trainer, "set_lr_multiplier"):
                        self.trainer.set_lr_multiplier(cur_lr_mult)

                    # 只有真正执行 PPO 更新的 episode 才保存 loss。以前把
                    # last_loss_dict 复制到每一个累积 episode，造成价值损失和
                    # KL 曲线被重复点伪造为连续训练信号。
                    loss_dict = self.trainer.update_policies()
                    ppo_updated = bool(loss_dict)
                    if ppo_updated:
                        last_loss_dict = loss_dict.copy()
                    scenarios_in_batch = 0
                    self.trainer.step_entropy_anneal()
                else:
                    loss_dict = {}
                    ppo_updated = False
                    cur_lr_mult = (
                        1.0
                        if self.stage_only is not None
                        else (compute_transition_lr_multiplier(ep, self.curriculum_sampler.thresholds) if self.curriculum_sampler else 1.0)
                    )

                # 控制台可展示最近一次更新的诊断，但它不进入 JSON/TensorBoard
                # 曲线，避免把同一更新重复解释为多次梯度步骤。
                display_loss_dict = loss_dict if ppo_updated else last_loss_dict

                committed_count = sum(int(d.committed) for d in decisions)
                trainable_count = sum(int(d.trainable) for d in decisions)
                mean_ep_reward = sum(d.reward for d in decisions) / max(1, len(decisions))

                ep_record = {
                    "episode": ep,
                    "curriculum_stage": stage_tag,
                    "train_file": os.path.basename(train_file),
                    "committed_events": committed_count,
                    "trainable_events": trainable_count,
                    "manager_buffer_size": len(self.trainer.manager_buffer),
                    "worker_buffer_size": len(self.trainer.worker_buffer),
                    "ppo_updated": ppo_updated,
                    "rollout_batch_scenarios": batch_scenarios_before_update if should_update else scenarios_in_batch,
                    "manager_buffer_size_before_update": manager_buffer_before_update if should_update else None,
                    "worker_buffer_size_before_update": worker_buffer_before_update if should_update else None,
                    "mean_reward": mean_ep_reward,
                    "final_cmax": float(manager.current_solution.makespan),
                    "lr_multiplier": cur_lr_mult,
                    "loss": loss_dict,
                }
                history.append(ep_record)

                # 记录 TensorBoard 训练指标并即刻 flush
                if writer is not None:
                    writer.add_scalar("Train/Makespan_Cmax", manager.current_solution.makespan, ep)
                    writer.add_scalar("Train/Mean_Reward", mean_ep_reward, ep)
                    writer.add_scalar("Train/LR_Multiplier", cur_lr_mult, ep)
                    writer.add_scalar("Train/Manager_LR", self.trainer.lr * cur_lr_mult, ep)
                    writer.add_scalar("Train/Worker_LR", self.trainer.worker_lr * cur_lr_mult, ep)
                    if ppo_updated:
                        writer.add_scalar("Train/Total_Loss", loss_dict.get("total_loss", 0.0), ep)
                        writer.add_scalar("Train/Manager_Policy_Loss", loss_dict.get("m_policy_loss", 0.0), ep)
                        writer.add_scalar("Train/Manager_Value_Loss", loss_dict.get("m_val_loss", 0.0), ep)
                        writer.add_scalar("Train/Worker_Policy_Loss", loss_dict.get("w_policy_loss", 0.0), ep)
                        writer.add_scalar("Train/Worker_Value_Loss", loss_dict.get("w_val_loss", 0.0), ep)
                        writer.add_scalar("Train/Manager_Clip_Fraction", loss_dict.get("m_clip_fraction", 0.0), ep)
                        writer.add_scalar("Train/Worker_Clip_Fraction", loss_dict.get("w_clip_fraction", 0.0), ep)
                        writer.add_scalar("Train/Manager_Approx_KL", loss_dict.get("m_approx_kl", 0.0), ep)
                        writer.add_scalar("Train/Worker_Approx_KL", loss_dict.get("w_approx_kl", 0.0), ep)
                        writer.add_scalar("Train/Manager_Ratio_Mean", loss_dict.get("m_ratio_mean", 1.0), ep)
                        writer.add_scalar("Train/Worker_Ratio_Mean", loss_dict.get("w_ratio_mean", 1.0), ep)
                        writer.add_scalar("Train/Grad_Norm", loss_dict.get("grad_norm", 0.0), ep)
                        writer.add_scalar("Train/Entropy_Coeff", float(self.trainer.c_ent), ep)
                        writer.add_scalar("Train/KL_Early_Stopped", float(loss_dict.get("kl_early_stopped", 0.0)), ep)
                        writer.add_scalar("Train/PPO_Epochs_Completed", float(loss_dict.get("ppo_epochs_completed", 4.0)), ep)
                    writer.add_scalar("Train/Trainable_Events", trainable_count, ep)
                    writer.add_scalar("Train/Committed_Events", committed_count, ep)
                    writer.flush()

                # 每轮保存最新断点，确保随时可恢复
                if latest_checkpoint_out:
                    try:
                        self.trainer.save_checkpoint(
                            latest_checkpoint_out,
                            extra_meta={
                                **checkpoint_contract_meta,
                                "episode": ep,
                                "curriculum_stage": stage_tag,
                                "model_selection_protocol": self.model_selection_protocol,
                                "model_selection_val_files": self.model_selection_val_files,
                                "best_autonomous_feasibility_rate": best_autonomous_feasibility_rate,
                                "best_val_cmax": best_val_cmax,
                                "best_worst_val_cmax": best_worst_val_cmax,
                            },
                        )
                    except Exception as ckpt_err:
                        print(f"[提示] 最新断点落盘临时受限 ({ckpt_err})，不影响训练正常推进。")

                if ep % max(1, eval_interval // 2) == 0 or ep == total_episodes or should_update:
                    m_pl = display_loss_dict.get('m_policy_loss', 0.0)
                    w_pl = display_loss_dict.get('w_policy_loss', 0.0)
                    m_cf = display_loss_dict.get('m_clip_fraction', 0.0) * 100.0
                    w_cf = display_loss_dict.get('w_clip_fraction', 0.0) * 100.0
                    m_kl = display_loss_dict.get('m_approx_kl', 0.0)
                    w_kl = display_loss_dict.get('w_approx_kl', 0.0)
                    upd_tag = "[PPO更新]" if should_update else f"[累积 {scenarios_in_batch}/{self.rollout_scenarios}]"
                    print(
                        f"Episode {ep:03d}/{total_episodes}: [{stage_tag}] {upd_tag} 场景={os.path.basename(train_file)} | "
                        f"样本={trainable_count}/{len(decisions)} | "
                        f"Cmax={manager.current_solution.makespan:.1f} | "
                        f"P_Loss(M/W)={m_pl:.4f}/{w_pl:.4f} | "
                        f"KL(M/W)={m_kl:.2e}/{w_kl:.2e} | "
                        f"Clip(M/W)={m_cf:.1f}%/{w_cf:.1f}%"
                    )

                # 周期性验证评估（纯双 Actor 策略评估）
                if ep % eval_interval == 0 or ep == total_episodes:
                    val_res = self.evaluate_on_val()
                    val_record = {
                        "episode": ep,
                        "mean_cmax": val_res["mean_cmax"],
                        "mean_actor_cmax": val_res["mean_actor_cmax"],
                        "autonomous_feasibility_rate": val_res["autonomous_feasibility_rate"],
                        "details": val_res["details"],
                        "selection_protocol": val_res.get("selection_protocol"),
                        "selection_summary": val_res.get("selection_summary"),
                    }
                    val_history.append(val_record)
                    print(
                        f"  >>> [Validation @ Ep {ep:03d}] Actor Cmax={val_res['mean_actor_cmax']:.1f} min | "
                        f"完整自主可行率={val_res['autonomous_feasibility_rate'] * 100:.1f}%"
                    )
                    if val_res.get("details"):
                        sc_infos = [
                            f"{d['file'].replace('.xlsx','')}: Cmax={d['raw_actor_cmax']:.1f}, success={int(d['autonomous_success'])}"
                            for d in val_res["details"]
                        ]
                        print("      阶梯算例明细: " + " | ".join(sc_infos))

                    if writer is not None:
                        writer.add_scalar("Val/Actor_Makespan_Cmax", val_res["mean_actor_cmax"], ep)
                        writer.add_scalar("Val/Autonomous_Feasibility_Percent", val_res["autonomous_feasibility_rate"] * 100.0, ep)
                        writer.flush()

                    # 三个独立 Stage 均只接受其固定两场景验证窗口全部可行的
                    # 候选，再按等权平均 Cmax 与最差 Cmax 排序。
                    is_improved = False
                    feasibility_rate = val_res["autonomous_feasibility_rate"]
                    actor_cmax = val_res["mean_actor_cmax"]
                    selection_summary = val_res.get("selection_summary")
                    if self.stage_only is not None:
                        if selection_summary is None:
                            raise RuntimeError(
                                f"Stage {self.stage_only} 两场景选模未生成综合验证摘要。"
                            )
                        is_improved = is_stage_validation_candidate_better(
                            candidate=selection_summary,
                            best_mean_cmax=best_val_cmax,
                            best_worst_cmax=best_worst_val_cmax,
                        )
                    else:
                        if feasibility_rate > best_autonomous_feasibility_rate + 1e-4:
                            is_improved = True
                        elif abs(feasibility_rate - best_autonomous_feasibility_rate) <= 1e-4:
                            if actor_cmax < best_val_cmax - _CHECKPOINT_SELECTION_EPS:
                                is_improved = True

                    if is_improved:
                        if self.stage_only is not None:
                            best_autonomous_feasibility_rate = 1.0
                            best_val_cmax = float(selection_summary["mean_cmax"])
                            best_worst_val_cmax = float(selection_summary["worst_cmax"])
                        else:
                            best_autonomous_feasibility_rate = feasibility_rate
                            best_val_cmax = actor_cmax
                        if checkpoint_out:
                            try:
                                saved_path = self.trainer.save_checkpoint(
                                    checkpoint_out,
                                    extra_meta={
                                        **checkpoint_contract_meta,
                                        "episode": ep,
                                        "model_selection_protocol": self.model_selection_protocol,
                                        "model_selection_val_files": self.model_selection_val_files,
                                        "best_autonomous_feasibility_rate": best_autonomous_feasibility_rate,
                                        "best_val_cmax": best_val_cmax,
                                        "best_worst_val_cmax": best_worst_val_cmax,
                                    },
                                )
                                if self.stage_only is not None:
                                    print(
                                        f"  >>> [Checkpoint] 刷新 Stage {self.stage_only} 两场景综合最优策略！"
                                        f"2/2 可行，平均 Cmax={best_val_cmax:.1f} min，"
                                        f"最差 Cmax={best_worst_val_cmax:.1f} min -> 已导出: {saved_path}"
                                    )
                                else:
                                    print(
                                        f"  >>> [Checkpoint] 刷新最优自主策略！完整可行率={best_autonomous_feasibility_rate * 100:.1f}%, "
                                        f"Actor Cmax={best_val_cmax:.1f} min -> 已导出: {saved_path}"
                                    )
                            except Exception as best_err:
                                print(f"[提示] 最优策略权重写入受限 ({best_err})，暂缓写入以防中断。")

                # 每一轮结束都执行实时中途曲线落盘
                _persist_live_log(current_ep=ep, is_paused=False)

        except KeyboardInterrupt:
            print(f"\n\n[检测到用户 Ctrl+C 按键中断] 正在优雅暂停并保护现场...")
            paused_requested = True

        # 退出循环后最终保护性落盘
        _persist_live_log(current_ep=last_executed_ep, is_paused=paused_requested)

        if writer is not None:
            writer.flush()
            writer.close()

        if paused_requested:
            print(f"\n========================================================")
            print(f"  训练已安全暂停在 Episode {last_executed_ep}！")
            if latest_checkpoint_out:
                print(f"  最新权重断点已保存至: {latest_checkpoint_out}")
            if log_out:
                print(f"  当前最新曲线日志已落盘至: {log_out}")
            print(f"  续训方法：下次启动时追加 '--resume' 参数即可无缝继续！")
            print(f"========================================================\n")
        else:
            print(f"\n=== HPPO 训练全部完成！总轮数: {total_episodes} ===")

        return {
            "total_episodes": total_episodes,
            "episodes_per_scenario": episodes_per_scenario,
            "completed_episodes": last_executed_ep,
            "status": "PAUSED" if paused_requested else "COMPLETED",
            "train_scenes_count": len(self.train_files),
            "val_scenes_count": len(self.val_files),
            "model_selection_protocol": self.model_selection_protocol,
            "model_selection_val_files": self.model_selection_val_files,
            "best_autonomous_feasibility_rate": best_autonomous_feasibility_rate,
            "best_val_cmax": best_val_cmax,
            "best_worst_val_cmax": best_worst_val_cmax if best_worst_val_cmax != float("inf") else None,
            "reward_scale": self.reward_scale,
            "checkpoint_out": checkpoint_out,
            "latest_checkpoint_out": latest_checkpoint_out,
            "training_history": history,
            "validation_history": val_history,
        }


def main():
    parser = argparse.ArgumentParser(description="PCB 多场景动态调度 HPPO 训练驱动器")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="活动 V20261027 三组动态场景集根目录；省略时按活动数据合同解析，其他目录会被拒绝",
    )
    parser.add_argument("--sheet", default="Sheet1", help="目标实例工作表名，如 Sheet1")
    parser.add_argument("--episodes", type=int, default=None, help="仅用于单算例调试；独立 Stage 训练始终采用每组 1800 轮循环预设配额（s1..s6 依次循环，每轮循环更新一次 PPO）")
    parser.add_argument("--episodes-per-scenario", "--episodes-per-order", dest="episodes_per_scenario", type=int, default=300, help="每个订单/算例的平均训练轮数（默认 300 轮）")
    parser.add_argument("--base-quota", type=int, default=300, help="每个算例的目标配额（默认 300 轮，6 算例共 1800 轮）")
    parser.add_argument("--step-quota", type=int, default=0, help="算例难度递增步长（循环模式默认 0）")
    parser.add_argument("--c-ent", type=float, default=FORMAL_DYNAMIC_C_ENT, help=f"{FORMAL_DYNAMIC_PROTOCOL_VERSION} 动作熵探索奖励初始系数（默认 {FORMAL_DYNAMIC_C_ENT}）")
    parser.add_argument("--eval-interval", type=int, default=10, help="验证集评估周期")
    parser.add_argument("--d-h", type=int, default=32, help="HGNN 特征维度")
    parser.add_argument("--lr", type=float, default=FORMAL_DYNAMIC_LR, help=f"{FORMAL_DYNAMIC_PROTOCOL_VERSION} Split/HGNN/Critic 学习率（默认 {FORMAL_DYNAMIC_LR}）")
    parser.add_argument("--placement-lr", type=float, default=FORMAL_DYNAMIC_PLACEMENT_LR, help=f"{FORMAL_DYNAMIC_PROTOCOL_VERSION} PlacementPolicy 学习率（默认 {FORMAL_DYNAMIC_PLACEMENT_LR}）")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO 每轮经验重用迭代优化 Epoch 数（默认 4）")
    parser.add_argument("--reward-scale", type=float, default=FORMAL_DYNAMIC_REWARD_SCALE, help=f"{FORMAL_DYNAMIC_PROTOCOL_VERSION} 回报保序缩放因子（默认 {FORMAL_DYNAMIC_REWARD_SCALE}）")
    parser.add_argument("--reward-mode", default="scaled_relative", choices=["scaled_relative", "normalized_relative", "relative", "absolute"], help="Reward calculation mode")
    parser.add_argument("--max-kl", type=float, default=0.15, help="PPO KL Early-Stopping 阈值（默认 0.15 杜绝策略发散）")
    parser.add_argument("--target-kl", type=float, default=0.05, help="PPO Target KL divergence")
    _DEFAULT_EXP_DIR = os.path.join(_ROOT_DIR, "baselines")
    parser.add_argument("--checkpoint-in", help="仅保留给单算例调试；独立 Stage 禁止外部初始权重")
    parser.add_argument("--checkpoint-out", default=None, help="最优 Checkpoint 导出路径；Stage 模式默认写入 result/主算法/stageN/seed_S/")
    parser.add_argument("--latest-checkpoint-out", default=None, help="同 Stage 中断续训的最新断点路径")
    parser.add_argument("--log-out", default=None, help="实时训练报告 JSON 导出路径")
    parser.add_argument("--tensorboard-dir", default=None, help="TensorBoard 日志目录")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备 (cuda/cpu)")
    parser.add_argument(
        "--seed",
        type=int,
        choices=FORMAL_SEEDS,
        default=DEFAULT_FORMAL_SEED,
        help=f"正式随机种子（{list(FORMAL_SEEDS)}，默认 {DEFAULT_FORMAL_SEED}）",
    )
    parser.add_argument("--max-train-scenarios", type=int, default=None, help="使用的最大训练算例数量（默认全量 18 个）")
    parser.add_argument("--max-val-scenarios", type=int, default=None, help="使用的最大验证算例数量（默认全量 6 个）")
    run_scope = parser.add_mutually_exclusive_group(required=True)
    run_scope.add_argument("--scenario-file", default=None, help="指定单个活动数据集算例，仅用于调试")
    run_scope.add_argument("--stage-only", type=int, choices=[1, 2, 3], default=None, help="主实验必选：从随机初始化独立训练指定组（6算例跑1800轮，s1..s6循环，过完一整轮循环6个算例更新一次PPO，共300次PPO更新）")
    parser.add_argument("--curriculum-mode", default="curriculum", choices=["curriculum", "round_robin", "sequential", "uniform"], help="采样调度模式：'curriculum' 启用三组渐进课程学习；'round_robin' 轮流循环遍历；'sequential' 按算例顺序串行训练；'uniform' 纯随机均匀采样")
    parser.add_argument("--rollout-scenarios", type=int, default=6, help="PPO 经验累积场景数（默认 6：s1到s6完成一次循环更新一次 PPO，共 300 次更新）")
    parser.add_argument("--resume", action="store_true", help="是否从上次中断的断点和日志继续训练")
    parser.add_argument("--fresh-start", action="store_true", help="强制从第 1 轮全新开始（忽略已有断点与日志）")

    args = parser.parse_args()

    if args.stage_only is not None and args.checkpoint_in:
        parser.error("独立 Stage 不允许 --checkpoint-in；仅允许通过本 Stage 自身的 --resume/自动断点恢复继续同一次训练。")

    if args.stage_only is not None:
        default_run_dir = os.path.join(
            _ROOT_DIR,
            "result",
            "主算法",
            f"stage{args.stage_only}",
            f"seed_{args.seed}",
        )
        args.checkpoint_out = args.checkpoint_out or os.path.join(default_run_dir, f"stage{args.stage_only}_best.pt")
        args.latest_checkpoint_out = args.latest_checkpoint_out or os.path.join(default_run_dir, f"stage{args.stage_only}_latest.pt")
        args.log_out = args.log_out or os.path.join(default_run_dir, f"stage{args.stage_only}_train_log.json")
        args.tensorboard_dir = args.tensorboard_dir or os.path.join(default_run_dir, "tensorboard")
    else:
        debug_run_dir = os.path.join(_DEFAULT_EXP_DIR, "debug_single_scenario")
        args.checkpoint_out = args.checkpoint_out or os.path.join(debug_run_dir, "best.pt")
        args.latest_checkpoint_out = args.latest_checkpoint_out or os.path.join(debug_run_dir, "latest.pt")
        args.log_out = args.log_out or os.path.join(debug_run_dir, "train_log.json")
        args.tensorboard_dir = args.tensorboard_dir or os.path.join(debug_run_dir, "tensorboard")

    _acquire_single_training_process_lock(default_run_dir if args.stage_only is not None else debug_run_dir)

    device = torch.device(args.device)
    gpu_info = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" and torch.cuda.is_available() else ""
    print(f"HPPO 多场景训练计算设备: {device}{gpu_info}")

    trainer_driver = DynamicMultiScenarioTrainer(
        dataset_dir=args.dataset_dir,
        sheet_name=args.sheet,
        d_h=args.d_h,
        lr=args.lr,
        placement_lr=args.placement_lr,
        ppo_epochs=args.ppo_epochs,
        reward_scale=args.reward_scale,
        checkpoint_in=args.checkpoint_in,
        device=device,
        max_train_scenarios=args.max_train_scenarios,
        max_val_scenarios=args.max_val_scenarios,
        single_scenario_file=args.scenario_file,
        stage_only=args.stage_only,
        rollout_scenarios=args.rollout_scenarios,
        reward_mode=args.reward_mode,
        max_kl=args.max_kl,
        target_kl=args.target_kl,
        base_quota=args.base_quota,
        step_quota=args.step_quota,
        c_ent=args.c_ent,
    )

    n_scenes = len(trainer_driver.train_files)
    if args.stage_only is not None:
        # 独立阶段同样允许断点续训（例如安全暂停后继续）。续训前会在下方
        # resume 分支校验日志的选模协议与验证文件是否和当前代码一致，
        # 防止把旧的单算例选模指标混入新的综合选模协议。
        expected_stage_episodes = sum(trainer_driver.stage_quota_override or [])
        if not expected_stage_episodes:
            raise RuntimeError("独立阶段训练未生成有效的阶段配额。")
        if args.episodes is not None and args.episodes != expected_stage_episodes:
            parser.error(
                f"Stage {args.stage_only} 的固定原始配额总和为 {expected_stage_episodes}；"
                f"不能指定 --episodes {args.episodes}。"
            )
        effective_total_episodes = expected_stage_episodes
        effective_curriculum_mode = "stage_only"
    else:
        formal_default_episodes = int(trainer_driver.training_curriculum_policy["formal_total_episodes"])
        effective_total_episodes = args.episodes if args.episodes is not None else formal_default_episodes
        effective_curriculum_mode = args.curriculum_mode

    # 续训与断点检测：PyCharm 无参数运行时，若存在成对且
    # 一致的 latest checkpoint + 日志，默认续训。只有 --fresh-start
    # 可以明确允许从头开始，防止误点运行覆盖数千轮结果。
    def _to_abs(p: Optional[str]) -> Optional[str]:
        if p and not os.path.isabs(p):
            return os.path.normpath(os.path.join(_DEFAULT_EXP_DIR, p))
        return p

    ckpt_in = args.checkpoint_in
    log_file_abs = _to_abs(args.log_out)
    default_latest = _to_abs(args.latest_checkpoint_out)
    has_log = os.path.exists(log_file_abs)
    has_latest = os.path.exists(default_latest)
    auto_resume = not args.fresh_start and not ckpt_in and has_log and has_latest
    resume_requested = (args.resume or auto_resume) and not args.fresh_start

    if resume_requested:
        try:
            if not (has_log and has_latest):
                raise FileNotFoundError(
                    f"续训需要成对文件：log={has_log}, latest_checkpoint={has_latest}"
                )
            with open(log_file_abs, "r", encoding="utf-8") as f:
                saved_log = json.load(f)
            # 三个独立 Stage 都采用固定两场景综合选模；旧单场景
            # 日志不得与新协议混用。断点还必须属于当前 Stage。
            if trainer_driver.stage_only is not None:
                saved_protocol = saved_log.get("model_selection_protocol")
                if saved_protocol != trainer_driver.model_selection_protocol:
                    raise RuntimeError(
                        f"Stage {trainer_driver.stage_only} 日志的验证选模协议不一致"
                        f"（saved={saved_protocol!r}，当前要求={trainer_driver.model_selection_protocol!r}）。"
                    )
                if saved_log.get("stage_only") != trainer_driver.stage_only:
                    raise RuntimeError(
                        f"断点日志属于 Stage {saved_log.get('stage_only')!r}，"
                        f"不能用于独立 Stage {trainer_driver.stage_only}。"
                    )
                saved_val_files = list(saved_log.get("model_selection_val_files", []))
                if saved_val_files != trainer_driver.model_selection_val_files:
                    raise RuntimeError(
                        f"Stage {trainer_driver.stage_only} 验证文件与日志不一致："
                        f"log={saved_val_files}, active={trainer_driver.model_selection_val_files}"
                    )
            history_len = len(saved_log.get("training_history", []))
            completed_ep = int(saved_log.get("current_episode", history_len))
            if not (0 < completed_ep < effective_total_episodes):
                raise RuntimeError(
                    f"日志轮数 {completed_ep} 不在可续训区间 "
                    f"1..{effective_total_episodes - 1}"
                )
            ckpt_in = default_latest
            was_auto_resume = auto_resume and not args.resume
            payload = trainer_driver.trainer.load_checkpoint(ckpt_in, load_optimizer=True)
            checkpoint_ep = int(payload.get("episode", completed_ep))
            if checkpoint_ep < completed_ep or checkpoint_ep > completed_ep + 1:
                raise RuntimeError(
                    "断点与日志轮数不一致："
                    f"checkpoint Episode={checkpoint_ep}, log Episode={completed_ep}"
                )
            trainer_driver.checkpoint_payload = payload
            args.resume = True
            mode_label = "自动续训" if was_auto_resume else "续训"
            print(
                f"\n[{mode_label}模式] 成功加载昨晚实验断点："
                f"checkpoint Episode {checkpoint_ep}, log Episode {completed_ep}, "
                f"将从 Episode {max(checkpoint_ep, completed_ep) + 1} 继续。\n"
            )
        except Exception as exc:
            raise RuntimeError(
                f"续训断点校验失败：{exc}。程序已停止，不会覆盖现有结果；"
                "只有确认丢弃历史时才使用 --fresh-start。"
            ) from exc
    elif args.fresh_start:
        args.resume = False
        print(f"\n[启动模式] 已显式指定 --fresh-start，从 Episode 1/{effective_total_episodes} 开始。\n")
    elif has_log != has_latest:
        raise RuntimeError(
            "检测到孤立的日志或断点，为防止覆盖已有实验已停止启动。"
            "请恢复成对文件，或显式使用 --fresh-start。"
        )
    else:
        args.resume = False
        print(f"\n[启动模式] 未发现历史断点，从 Episode 1/{effective_total_episodes} 开始。\n")

    trainer_driver.train(
        total_episodes=effective_total_episodes,
        episodes_per_scenario=args.episodes_per_scenario,
        eval_interval=args.eval_interval,
        checkpoint_out=_to_abs(args.checkpoint_out),
        latest_checkpoint_out=_to_abs(args.latest_checkpoint_out),
        log_out=_to_abs(args.log_out),
        tensorboard_dir=_to_abs(args.tensorboard_dir),
        curriculum_mode=effective_curriculum_mode,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
