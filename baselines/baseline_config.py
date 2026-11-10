"""Configuration and manifest resolution for the six approved rule baselines."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_DEFAULT_ARTIFACTS_DIR = _PROJECT_ROOT / "result" / "对比算法"


def find_project_root() -> str:
    """Return the dynamic-project root without relying on a locale-sensitive cwd."""
    return str(_PROJECT_ROOT)


def resolve_layer_paths(root: str) -> List[str]:
    """Expose only the project root and its environment/agent dependencies."""
    root_path = Path(root).resolve()
    paths = [str(root_path)]
    for directory in ("environment", "agent"):
        candidate = root_path / directory
        if candidate.is_dir():
            paths.append(str(candidate))
    return paths


for _p in [
    str(_THIS_DIR),
    str(_THIS_DIR / "configs"),
    *resolve_layer_paths(str(_PROJECT_ROOT)),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.experiment_protocol import FORMAL_SEED_COUNT


# =====================================================================
# 1. Rule Decision Contract
# =====================================================================

_SHARED_PHYSICS = {
    "active_manifest": True,
    "event_manager": "DynamicEventManager",
    "scope": {"k_seed": 5, "k_scope": 8, "delta_k": 4},
    "time_model": "authoritative_time_model",
    "decoder": "shared_full_decode",
    "transaction_validator": "shared_atomic_validation",
    "failure_policy": "controlled_failure_without_fabricated_cmax",
}

_RULE_CONTRACT = {
    "tier": "RULE_BASELINE",
    "learned": False,
    "decision_interface": "deterministic_dispatching_rule",
    "can_resplit_eligible_old_orders": False,
    "direct_template_choice": False,
    "direct_machine_slot_choice": False,
    "academic_role_zh": "确定性调度规则或离线贪心基线",
}


def method_contract(method: str) -> Dict[str, Any]:
    """Return a copy of one method's declared permissions and learning role."""
    key = str(method).strip().upper().replace("-", "_").replace(" ", "_")
    if key in {"S_BASE", "SBASE", "FIFO", "SPT", "LPT", "EDD", "ECT"}:
        payload = {**_RULE_CONTRACT, "method": "S_BASE" if key in {"S_BASE", "SBASE"} else key}
    else:
        raise ValueError(f"未知规则基线: {method!r}")

    payload.setdefault("display_name", payload["method"])
    payload["shared_physics"] = deepcopy(_SHARED_PHYSICS)
    return payload


def comparison_contract() -> Dict[str, Any]:
    """Declare the common physical contract for the six rule baselines."""
    return {
        "schema": "pcb_dynamic_six_rule_contract_v1",
        "approved_rules": ["S_BASE", "FIFO", "SPT", "LPT", "EDD", "ECT"],
        "comparison_level": "deterministic_rule",
        "same_problem_physics": True,
        "primary_claim_zh": (
            "六种规则在同一 PCB 动态调度物理模型、事件流、解码器和事务校验下独立回放。"
        ),
        "same_final_evaluation_contract": True,
    }


# =====================================================================
# 2. Rule Configuration & Manifest Plans
# =====================================================================

@dataclass(frozen=True)
class CurriculumPlan:
    """A manifest-verified full curriculum or one independently runnable stage."""

    stage: Optional[int]
    label: str
    train_files: List[str]
    val_files: List[str]
    quotas: List[int]
    episode_offset: int
    scenario_ids: List[str]
    contract_version: str

    @property
    def total_episodes(self) -> int:
        return sum(self.quotas)


@dataclass
class BaselineConfig:
    """Versioned defaults for six-rule offline replay."""

    project_root: str = str(_PROJECT_ROOT)
    comparison_root: str = str(_THIS_DIR)
    artifacts_dir: str = str(_DEFAULT_ARTIFACTS_DIR)
    k_seed: int = 5
    k_scope: int = 8
    delta_k: int = 4
    max_partition_templates: int = 16

    formal_min_completion_rate: float = 1.0
    formal_min_completed_scenarios: Optional[int] = None
    formal_seed_start: int = 1
    formal_seed_count: int = FORMAL_SEED_COUNT

    sheet_name: str = "Sheet1"
    pool_type: str = "Pool_D"
    formal_mode: bool = True
    create_output_dirs: bool = True

    def __post_init__(self) -> None:
        if self.k_seed <= 0 or self.k_scope < self.k_seed:
            raise ValueError("正式主设置要求 1 <= k_seed <= k_scope。")
        if self.delta_k <= 0:
            raise ValueError("delta_k 必须为正整数。")
        if not 1 <= self.max_partition_templates <= 16:
            raise ValueError("分批模板候选上限必须位于 1..16。")
        if not 0.0 < float(self.formal_min_completion_rate) <= 1.0:
            raise ValueError("formal_min_completion_rate 必须位于 (0, 1]。")
        if self.formal_min_completed_scenarios is not None and int(self.formal_min_completed_scenarios) <= 0:
            raise ValueError("formal_min_completed_scenarios 必须为正整数或 None。")
        if self.formal_seed_start <= 0 or self.formal_seed_count <= 0:
            raise ValueError("正式种子起点与数量必须为正整数。")
        if self.create_output_dirs:
            os.makedirs(self.artifacts_dir, exist_ok=True)

    def audit_dict(self) -> Dict[str, Any]:
        """Return the complete JSON-serializable effective configuration."""
        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_")
            and isinstance(value, (int, float, str, bool, list, dict, tuple, type(None)))
        }

    def config_sha256(self) -> str:
        """Hash the effective configuration for checkpoints and result provenance."""
        payload = json.dumps(
            self.audit_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def formal_seeds(self) -> List[int]:
        """Return the preregistered contiguous formal seed panel."""
        return list(range(self.formal_seed_start, self.formal_seed_start + self.formal_seed_count))

    def artifact_directory(self, algorithm: str, stage: Optional[int], seed: int) -> str:
        stage_name = "full_curriculum" if stage is None else f"stage{int(stage)}"
        return os.path.join(
            self.artifacts_dir,
            algorithm.lower(),
            stage_name,
            f"seed_{int(seed):02d}",
        )


def _load_active_contract() -> Dict[str, object]:
    from dynamic_multi_scenario_trainer import load_active_dataset_contract

    return load_active_dataset_contract()


def _scale_key(record: Dict[str, object]) -> Tuple[int, int, str]:
    try:
        initial_orders = int(record["initial_orders"])
        dynamic_events = int(record["dynamic_events"])
        scenario_id = str(record["scenario_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"活动 Manifest 包含无法排序的训练场景记录: {record!r}") from exc
    return initial_orders, dynamic_events, scenario_id


def _manifest_files(
    contract: Dict[str, object],
    split: str,
    *,
    sort_train_by_scale: bool,
) -> Tuple[List[str], List[Dict[str, object]]]:
    manifest = contract["manifest"]
    dataset_dir = Path(str(contract["resolved_dataset_dir"]))
    records = list(manifest["splits"][split])
    if sort_train_by_scale:
        records.sort(key=_scale_key)
    files = [str((dataset_dir / str(record["filepath"])).resolve()) for record in records]
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"活动 Manifest 所列 {split} 场景缺失: {missing}")
    return files, records


def _stage_validation_files(contract: Dict[str, object], stage: int) -> List[str]:
    from dynamic_multi_scenario_trainer import STAGE_VALIDATION_FILENAMES

    expected_names = STAGE_VALIDATION_FILENAMES.get(int(stage))
    if expected_names is None:
        raise ValueError(f"阶段编号必须为 1..3，收到 {stage!r}。")
    files, _ = _manifest_files(contract, "val", sort_train_by_scale=False)
    paths_by_name: Dict[str, List[str]] = {}
    for path in files:
        paths_by_name.setdefault(Path(path).name, []).append(path)
    selected: List[str] = []
    for filename in expected_names:
        matches = paths_by_name.get(filename, [])
        if not matches:
            raise FileNotFoundError(f"活动 Manifest 中未找到 Stage {stage} 验证场景: {filename}")
        selected.append(matches[0])
    return selected


STAGE_QUOTAS = {
    1: ([300, 300, 300, 300, 300, 300], 0),
    2: ([300, 300, 300, 300, 300, 300], 1800),
    3: ([300, 300, 300, 300, 300, 300], 3600),
}


def resolve_curriculum_plan(
    config: BaselineConfig,
    stage: Optional[int] = None,
) -> CurriculumPlan:
    """Resolve an unscaled, manifest-bound plan for one stage or the full curriculum."""
    contract = _load_active_contract()
    train_files, train_records = _manifest_files(contract, "train", sort_train_by_scale=True)
    manifest = contract["manifest"]
    contract_version = str(manifest.get("manifest_version", "unknown"))

    if stage is not None:
        int_stage = int(stage)
        if int_stage not in STAGE_QUOTAS:
            raise ValueError(f"Stage 编号必须位于 1..3，收到 {stage!r}。")
        quotas, offset = STAGE_QUOTAS[int_stage]
        slice_start = (int_stage - 1) * 6
        slice_end = slice_start + 6
        stage_train_files = train_files[slice_start:slice_end]
        stage_train_records = train_records[slice_start:slice_end]
        scenario_ids = [str(r["scenario_id"]) for r in stage_train_records]
        val_files = _stage_validation_files(contract, int_stage)
        return CurriculumPlan(
            stage=int_stage,
            label=f"Stage {int_stage} (Independent {quotas})",
            train_files=stage_train_files,
            val_files=val_files,
            quotas=quotas,
            episode_offset=offset,
            scenario_ids=scenario_ids,
            contract_version=contract_version,
        )

    # Full curriculum plan (5400 episodes total, 1800 per stage)
    quotas = [300] * len(train_files)
    val_files, _ = _manifest_files(contract, "val", sort_train_by_scale=False)
    scenario_ids = [str(r["scenario_id"]) for r in train_records]
    return CurriculumPlan(
        stage=None,
        label="Full Curriculum (5400 episodes)",
        train_files=train_files,
        val_files=val_files,
        quotas=quotas,
        episode_offset=0,
        scenario_ids=scenario_ids,
        contract_version=contract_version,
    )


def resolve_evaluation_files(
    config: BaselineConfig,
    split: str,
    stage: Optional[int] = None,
) -> List[str]:
    """Resolve file paths for validation, testing, or stage-specific evaluation."""
    contract = _load_active_contract()
    if split == "train" and stage is not None:
        plan = resolve_curriculum_plan(config, stage=stage)
        return plan.train_files
    files, _ = _manifest_files(contract, split, sort_train_by_scale=(split == "train"))
    return files
