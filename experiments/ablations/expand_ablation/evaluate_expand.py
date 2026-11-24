"""Evaluate L11 (DAPPO-Expand) vs AD-HG-DAPPO on quality, graph bloat and latency.

This script compares the main method's on-demand progressive graph construction
(``PCBGraphBuilder``) against pre-expanding all candidate templates' subbatch
nodes in the Split stage (``PreExpandedPCBGraphBuilder`` / ``DAPPO-Expand``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
for directory in (
    THIS_DIR,
    ROOT,
    ROOT / "agent",
    ROOT / "environment",
    ROOT / "experiments" / "ablations" / "scope_ablation",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch

from data_models import POOL_D
from dynamic_event_manager import DynamicEventManager
from dynamic_validator import load_dynamic_orders_and_events
from dappo_expand_trainer import (
    DAPPO_EXPAND_METHOD_ID,
    DAPPOExpandTrainer,
    ExpandedTwoAgentRunner,
    InstrumentedTwoAgentRunner,
)
from expanded_graph_builder import TimedOnDemandPCBGraphBuilder
from hppo_trainer import HPPOTrainer
from scope_metrics import event_perturbation

MAIN_METHOD_ID = "AD-HG-DAPPO"

# Verified pre-migration hashes recorded in result/data_layout_migration_20261117/migration_receipt.json
# so existing seed_1 main checkpoints trained before directory migration remain auditable.
_HISTORICAL_MANIFEST_HASHES = frozenset({
    "135937d2777fc3c235cd1c9f4f2f5558232ee2fbf87e301ad5b60b81870e6b3d",
    "480851e450ce729d282528de44e576b709a5248ba770355bd8094ac5bb28fd21",
})
_HISTORICAL_PACKAGE_HASHES = frozenset({
    "b21c4422fc72ea4c1372d86e47468d43c781e5dbd362055c6a72b85568c581b9",
    "00ccc6bf9011ea373c67a70e3154b7cf47aa49fd5506d0209e13306a64800652",
})


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest() -> str:
    result = hashlib.sha256()
    for folder in (
        ROOT / "agent",
        ROOT / "environment",
        ROOT / "experiments" / "ablations" / "expand_ablation",
    ):
        for path in sorted(folder.rglob("*.py")):
            result.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            result.update(path.read_bytes())
    return result.hexdigest()


def load_config(path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    file = Path(path) if path else ROOT / "configs" / "expand_ablation.json"
    config = json.loads(file.read_text(encoding="utf-8-sig"))
    if (
        config.get("schema") != "expand_ablation_v1"
        or config.get("methods") != [MAIN_METHOD_ID, DAPPO_EXPAND_METHOD_ID]
        or int(config.get("max_partition_templates", 0)) != 16
        or int(config.get("default_k_scope", 0)) != 8
        or int(config.get("default_k_seed", 0)) != 5
        or int(config.get("delta_k", 0)) != 4
    ):
        raise ValueError("Expand ablation configuration violates the locked expand_ablation_v1 contract.")
    return config, _digest(file)


def select_scenarios(
    split: str = "test",
    stages: Sequence[int] = (1, 2, 3),
    scenario_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    base = ROOT / "data" / "dynamic_scenarios"
    index_path = base / "scenario_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    stage_set = {int(s) for s in stages}
    requested = set(scenario_ids or ())
    found: List[Dict[str, Any]] = []
    for row in index["scenarios"]:
        if row["role"] != split or int(row["stage"]) not in stage_set:
            continue
        if requested and row["scenario_id"] not in requested:
            continue
        path = (base / row["relative_path"]).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file() or _digest(path) != row["sha256"]:
            raise ValueError(f"L11 scenario path/hash mismatch: {row['scenario_id']}")
        found.append({**row, "path": path})
    if not found or (requested and requested != {r["scenario_id"] for r in found}):
        raise ValueError("Requested scenario is absent from the selected split/stage.")
    if len({r["scenario_id"] for r in found}) != len(found):
        raise ValueError("Duplicate scenario IDs in scenario_index.json.")
    return found, _digest(index_path)


def _validate_checkpoint_meta(
    payload: Dict[str, Any],
    expected_method: str,
    stage: int,
    seed: int,
    active_manifest_sha: str,
    active_package_sha: str,
) -> None:
    if payload.get("method_id") != expected_method:
        raise ValueError(f"Checkpoint method_id mismatch: got {payload.get('method_id')!r}, expected {expected_method!r}.")
    if int(payload.get("stage", -1)) != int(stage) or int(payload.get("seed", -1)) != int(seed):
        raise ValueError(
            f"{expected_method} checkpoint stage/seed mismatch: got stage={payload.get('stage')}, "
            f"seed={payload.get('seed')}, expected stage={stage}, seed={seed}."
        )
    manifest_sha = str(payload.get("dataset_manifest_sha256", ""))
    if manifest_sha and manifest_sha != active_manifest_sha and manifest_sha not in _HISTORICAL_MANIFEST_HASHES:
        raise ValueError(f"{expected_method} checkpoint manifest SHA mismatch: {manifest_sha}")
    package_sha = str(payload.get("active_package_sha256", ""))
    if package_sha and package_sha != active_package_sha and package_sha not in _HISTORICAL_PACKAGE_HASHES:
        raise ValueError(f"{expected_method} checkpoint active_package SHA mismatch: {package_sha}")


def _load_or_build_trainers(
    config: Dict[str, Any],
    by_stage: Dict[int, List[Dict[str, Any]]],
    seeds: Sequence[int],
    device: torch.device,
    allow_untrained_smoke: bool,
) -> Tuple[Dict[Tuple[str, int, int], Any], Dict[str, Any]]:
    manifest_sha = _digest(ROOT / "data" / "dynamic_scenarios" / "dataset_manifest.json")
    package_sha = _digest(ROOT / "data" / "active_experiment_package.json")
    trainers: Dict[Tuple[str, int, int], Any] = {}
    checkpoints_meta: Dict[str, Any] = {}

    for stage, items in sorted(by_stage.items()):
        initial_orders, events = load_dynamic_orders_and_events(str(items[0]["path"]), config["sheet_name"])
        sample_orders = [*initial_orders, *(ev.order for ev in events)]
        for seed in seeds:
            if allow_untrained_smoke:
                torch.manual_seed(int(seed))
                main_trainer = HPPOTrainer(orders=sample_orders, pool_type=POOL_D, d_h=config["d_h"], device=device)
                torch.manual_seed(int(seed))
                expand_trainer = DAPPOExpandTrainer(orders=sample_orders, pool_type=POOL_D, d_h=config["d_h"], device=device)
                checkpoints_meta[f"{MAIN_METHOD_ID}:s{stage}:seed{seed}"] = {"mode": "SMOKE_UNTRAINED"}
                checkpoints_meta[f"{DAPPO_EXPAND_METHOD_ID}:s{stage}:seed{seed}"] = {"mode": "SMOKE_UNTRAINED"}
            else:
                main_rel = config["proposed_checkpoints"][str(stage)].format(seed=seed)
                expand_rel = config["expand_checkpoints"][str(stage)].format(seed=seed)
                main_ckpt = (ROOT / main_rel).resolve()
                expand_ckpt = (ROOT / expand_rel).resolve()
                if not main_ckpt.is_file():
                    raise FileNotFoundError(f"缺少主方法检查点: {main_ckpt}")
                if not expand_ckpt.is_file():
                    raise FileNotFoundError(
                        f"缺少 DAPPO-Expand 检查点: {expand_ckpt}。请先运行 L11 训练脚本或传入 --allow-untrained-smoke 进行干跑验证。"
                    )
                main_trainer = HPPOTrainer(orders=sample_orders, pool_type=POOL_D, d_h=config["d_h"], device=device)
                main_payload = main_trainer.load_checkpoint(str(main_ckpt), load_optimizer=False)
                _validate_checkpoint_meta(main_payload, MAIN_METHOD_ID, stage, seed, manifest_sha, package_sha)

                expand_trainer = DAPPOExpandTrainer(orders=sample_orders, pool_type=POOL_D, d_h=config["d_h"], device=device)
                expand_payload = expand_trainer.load_checkpoint(str(expand_ckpt), load_optimizer=False)
                _validate_checkpoint_meta(expand_payload, DAPPO_EXPAND_METHOD_ID, stage, seed, manifest_sha, package_sha)

                checkpoints_meta[f"{MAIN_METHOD_ID}:s{stage}:seed{seed}"] = {
                    "path": str(main_ckpt), "sha256": _digest(main_ckpt),
                }
                checkpoints_meta[f"{DAPPO_EXPAND_METHOD_ID}:s{stage}:seed{seed}"] = {
                    "path": str(expand_ckpt), "sha256": _digest(expand_ckpt),
                }

            for trainer in (main_trainer, expand_trainer):
                for mod in (trainer.hgnn, trainer.manager_policy, trainer.worker_policy, trainer.shared_critic):
                    mod.eval()
            trainers[(MAIN_METHOD_ID, stage, seed)] = main_trainer
            trainers[(DAPPO_EXPAND_METHOD_ID, stage, seed)] = expand_trainer

    return trainers, checkpoints_meta


def replay_scenario(
    item: Dict[str, Any],
    method_id: str,
    seed: int,
    trainer: Any,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Replay one scenario independently and collect quality, graph bloat, and latency."""
    setup_t0 = time.perf_counter()
    initial, events = load_dynamic_orders_and_events(str(item["path"]), config["sheet_name"])
    manager = DynamicEventManager(
        pool_type=POOL_D,
        initial_orders=initial,
        events=events,
        instance_name=config["sheet_name"],
        k_scope=int(config["default_k_scope"]),
        k_seed=int(config["default_k_seed"]),
        delta_k=int(config["delta_k"]),
        formal_mode=True,
    )
    runner_cls = ExpandedTwoAgentRunner if method_id == DAPPO_EXPAND_METHOD_ID else InstrumentedTwoAgentRunner
    runner_kwargs: Dict[str, Any] = {}
    if method_id == MAIN_METHOD_ID:
        runner_kwargs["graph_builder_cls"] = TimedOnDemandPCBGraphBuilder
    runner = runner_cls(
        manager=manager,
        hgnn=trainer.hgnn,
        manager_policy=trainer.manager_policy,
        worker_policy=trainer.worker_policy,
        value_critic=trainer.shared_critic,
        device=device,
        max_templates=int(config["max_partition_templates"]),
        **runner_kwargs,
    )
    setup_s = time.perf_counter() - setup_t0

    event_rows: List[Dict[str, Any]] = []
    failure: Optional[str] = None
    while manager.pending_event_ids:
        event_id = manager.pending_event_ids[0]
        t_start = time.perf_counter()
        context = None
        decision = None
        try:
            context = manager.trigger_next_event()
            before = context.base_solution
            with torch.inference_mode():
                decision = runner.run_next_event(deterministic=True, context=context)
            committed = bool(decision.committed)
            after = manager.current_solution if committed else None
            if not committed:
                failure = "; ".join(map(str, decision.errors)) or "Candidate did not pass atomic commit"
        except Exception as exc:
            committed = False
            before = context.base_solution if context is not None else manager.current_solution
            after = None
            failure = f"{type(exc).__name__}: {exc}"
        decision_ms = (time.perf_counter() - t_start) * 1000.0

        metrics = event_perturbation(before, after, context.frozen_uids) if committed and context is not None else {}
        tel = getattr(decision, "graph_telemetry", {}) if decision is not None else {}
        event_rows.append({
            "scenario_id": item["scenario_id"],
            "stage": int(item["stage"]),
            "split": item["role"],
            "origin": item.get("origin", ""),
            "scenario_sha256": item["sha256"],
            "method_id": method_id,
            "seed": int(seed),
            "event_id": event_id,
            "event_time_min": context.event_time if context is not None else None,
            "committed": bool(committed),
            "cmax_before": float(before.makespan) if context is not None else None,
            "cmax_after": float(after.makespan) if after is not None else None,
            "decision_ms": decision_ms,
            "split_graph_build_ms": tel.get("split_graph_build_ms", 0.0),
            "placement_graph_build_ms": tel.get("placement_graph_build_ms", 0.0),
            "event_graph_build_ms": tel.get("event_graph_build_ms", 0.0),
            "split_hgnn_forward_ms": tel.get("split_hgnn_forward_ms", 0.0),
            "placement_hgnn_forward_ms": tel.get("placement_hgnn_forward_ms", 0.0),
            "event_hgnn_forward_ms": tel.get("event_hgnn_forward_ms", 0.0),
            "split_graph_calls": tel.get("split_graph_calls", 0),
            "placement_graph_calls": tel.get("placement_graph_calls", 0),
            "split_subbatch_nodes_mean": tel.get("split_subbatch_nodes_mean", 0.0),
            "split_subbatch_nodes_max": tel.get("split_subbatch_nodes_max", 0),
            "split_total_edges_mean": tel.get("split_total_edges_mean", 0.0),
            "split_total_edges_max": tel.get("split_total_edges_max", 0),
            "placement_subbatch_nodes_mean": tel.get("placement_subbatch_nodes_mean", 0.0),
            "placement_total_edges_mean": tel.get("placement_total_edges_mean", 0.0),
            "moved_old_batches": metrics.get("moved_old_batches"),
            "cross_machine_migrations": metrics.get("cross_machine_migrations"),
            "replaced_old_batches": metrics.get("replaced_old_batches"),
            "start_shift_abs_sum_min": metrics.get("start_shift_abs_sum_min"),
            "start_shift_observed_batches": metrics.get("start_shift_observed_batches"),
            "mean_start_shift_min": metrics.get("mean_start_shift_min"),
            "failure_reason": failure if not committed else None,
        })
        if not committed:
            manager.rollback_active()
            break

    total_events = len(manager.events)
    valid = [r for r in event_rows if r["committed"]]
    completed = (
        failure is None
        and len(event_rows) == total_events
        and len(valid) == total_events
        and not manager.pending_event_ids
        and manager.active_context is None
    )
    shift_obs = sum(int(r["start_shift_observed_batches"] or 0) for r in valid)
    shift_sum = sum(float(r["start_shift_abs_sum_min"] or 0.0) for r in valid)

    eval_row = {
        "scenario_id": item["scenario_id"],
        "stage": int(item["stage"]),
        "split": item["role"],
        "origin": item.get("origin", ""),
        "scenario_sha256": item["sha256"],
        "method_id": method_id,
        "seed": int(seed),
        "completed": completed,
        "feasible": completed,
        "final_cmax": float(manager.current_solution.makespan) if completed else None,
        "event_count": total_events,
        "committed_events": len(valid),
        "accept_rate": len(valid) / max(1, total_events),
        "n_fail_events": sum(1 for r in event_rows if not r["committed"]),
        "mean_split_subbatch_nodes": statistics.mean(r["split_subbatch_nodes_mean"] for r in valid) if valid else None,
        "mean_split_total_edges": statistics.mean(r["split_total_edges_mean"] for r in valid) if valid else None,
        "mean_split_graph_build_ms": statistics.mean(r["split_graph_build_ms"] for r in valid) if valid else None,
        "mean_split_hgnn_forward_ms": statistics.mean(r["split_hgnn_forward_ms"] for r in valid) if valid else None,
        "mean_event_graph_build_ms": statistics.mean(r["event_graph_build_ms"] for r in valid) if valid else None,
        "mean_event_decision_ms": statistics.mean(r["decision_ms"] for r in valid) if valid else None,
        "moved_old_per_event": statistics.mean(r["moved_old_batches"] for r in valid) if completed else None,
        "migrations_per_event": statistics.mean(r["cross_machine_migrations"] for r in valid) if completed else None,
        "start_shift_mean_min": (shift_sum / shift_obs) if (completed and shift_obs > 0) else (0.0 if completed else None),
        "setup_s": setup_s,
        "failure_reason": failure,
    }
    return eval_row, event_rows


def summarize_eval_rows(eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate scenario-level L11 results by (stage, method_id) without inflating N by seeds."""
    summary_rows: List[Dict[str, Any]] = []
    groups = sorted({(int(r["stage"]), str(r["method_id"])) for r in eval_rows})
    metric_keys = (
        "final_cmax",
        "mean_split_subbatch_nodes",
        "mean_split_total_edges",
        "mean_split_graph_build_ms",
        "mean_split_hgnn_forward_ms",
        "mean_event_graph_build_ms",
        "mean_event_decision_ms",
        "moved_old_per_event",
        "migrations_per_event",
        "start_shift_mean_min",
    )
    for stage, method_id in groups:
        group = [r for r in eval_rows if int(r["stage"]) == stage and r["method_id"] == method_id]
        by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_scenario[row["scenario_id"]].append(row)

        scenario_means: Dict[str, List[float]] = {k: [] for k in metric_keys}
        completed_scenarios = 0
        for sid, s_rows in sorted(by_scenario.items()):
            del sid
            if all(bool(r["completed"]) for r in s_rows):
                completed_scenarios += 1
                for k in metric_keys:
                    vals = [float(r[k]) for r in s_rows if r[k] is not None]
                    if len(vals) == len(s_rows):
                        scenario_means[k].append(statistics.mean(vals))

        completed_runs = sum(1 for r in group if r["completed"])
        item: Dict[str, Any] = {
            "stage": stage,
            "method_id": method_id,
            "scenario_count": len(by_scenario),
            "seed_count": len({int(r["seed"]) for r in group}),
            "total_runs": len(group),
            "completed_runs": completed_runs,
            "feasible_rate": completed_runs / max(1, len(group)),
            "valid_scenario_n": completed_scenarios,
        }
        for k in metric_keys:
            vals = scenario_means[k]
            item[f"mean_{k}"] = statistics.mean(vals) if vals else None
            item[f"std_{k}"] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if len(vals) == 1 else None)
        summary_rows.append(item)
    return summary_rows


def compute_convergence_indicators(val_history: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute steady convergence metrics from section 2.5 of the refactoring plan.

    - ``episodes_to_within_1pct_final``: first evaluation episode ``k`` such that
      for all evaluation points ``j >= k``, ``val_cmax[j] <= final_val_cmax * 1.01``.
    - ``normalized_val_auc``: trapezoidal AUC over normalized episode progress
      ``[0, 1]`` and min-max normalized validation Cmax ``[0, 1]``.
    """
    points: List[Tuple[int, float]] = []
    for row in val_history:
        ep = int(row.get("episode", 0))
        val = row.get("mean_actor_cmax", row.get("mean_cmax"))
        feas = float(row.get("autonomous_feasibility_rate", 0.0))
        if val is None:
            continue
        fval = float(val)
        if math.isfinite(fval) and feas >= 1.0 - 1e-9:
            points.append((ep, fval))
    if not points:
        return {
            "best_val_cmax": None,
            "best_val_episode": None,
            "final_val_cmax": None,
            "final_val_episode": None,
            "episodes_to_within_1pct_final": None,
            "normalized_val_auc": None,
        }

    points.sort(key=lambda pair: pair[0])
    best_ep, best_val = min(points, key=lambda pair: (pair[1], pair[0]))
    final_ep, final_val = points[-1]
    threshold = final_val * 1.01

    ep_within_1pct: Optional[int] = None
    for idx in range(len(points)):
        if all(val <= threshold for _, val in points[idx:]):
            ep_within_1pct = points[idx][0]
            break

    max_ep = max(ep for ep, _ in points)
    min_ep = min(ep for ep, _ in points)
    ep_span = max(1.0, float(max_ep - min_ep))
    min_c = min(v for _, v in points)
    max_c = max(v for _, v in points)
    c_span = max_c - min_c

    if len(points) == 1 or c_span <= 1e-12:
        norm_auc = 0.0
    else:
        norm_auc = 0.0
        for (e1, v1), (e2, v2) in zip(points[:-1], points[1:]):
            dx = (e2 - e1) / ep_span
            y1 = (v1 - min_c) / c_span
            y2 = (v2 - min_c) / c_span
            norm_auc += 0.5 * (y1 + y2) * dx

    return {
        "best_val_cmax": float(best_val),
        "best_val_episode": float(best_ep),
        "final_val_cmax": float(final_val),
        "final_val_episode": float(final_ep),
        "episodes_to_within_1pct_final": float(ep_within_1pct) if ep_within_1pct is not None else None,
        "normalized_val_auc": float(norm_auc),
    }


def summarize_convergence_logs(
    stages: Sequence[int],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    """Read training logs for AD-HG-DAPPO and DAPPO-Expand when available."""
    rows: List[Dict[str, Any]] = []
    method_dirs = (
        (MAIN_METHOD_ID, ROOT / "result" / "主算法"),
        (DAPPO_EXPAND_METHOD_ID, ROOT / "result" / "消融实验" / "全模板预展开"),
    )
    for stage in stages:
        for method_id, base_dir in method_dirs:
            for seed in seeds:
                log_file = base_dir / f"stage{stage}" / f"seed_{seed}" / f"stage{stage}_train_log.json"
                if not log_file.is_file():
                    continue
                payload = json.loads(log_file.read_text(encoding="utf-8"))
                indicators = compute_convergence_indicators(payload.get("validation_history", []))
                rows.append({
                    "stage": int(stage),
                    "method_id": method_id,
                    "seed": int(seed),
                    "status": payload.get("status", "UNKNOWN"),
                    "current_episode": payload.get("current_episode"),
                    "log_path": str(log_file.relative_to(ROOT)),
                    **indicators,
                })
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _format_report(
    summary_rows: Sequence[Dict[str, Any]],
    convergence_rows: Sequence[Dict[str, Any]],
    split: str,
    seeds: Sequence[int],
    smoke_mode: bool,
) -> str:
    lines = [
        "# L11 C2 独立消融报告：按需渐进构图 vs. 全模板预展开（DAPPO-Expand）",
        "",
        f"- 主方法：`{MAIN_METHOD_ID}`（按需渐进构图：`PCBGraphBuilder`）",
        f"- C2 消融变体：`{DAPPO_EXPAND_METHOD_ID}`（拆批阶段预先展开全部候选模板子批：`PreExpandedPCBGraphBuilder`）",
        f"- 评估数据划分：`{split}` | 种子：`{list(seeds)}`",
    ]
    if smoke_mode:
        lines.append("- **状态警示**：`SMOKE_UNTRAINED`（仅用于代码结构与构图耗时接口验证，**不得用于论文性能结论**）")
    lines.extend([
        "",
        "| Stage | 方法 | 有效场景数 | 完工时间 $C_{\\max}$ (min) | 拆批子批节点数 | 拆批有向边数 | 拆批构图耗时 (ms) | 拆批 HGNN 耗时 (ms) | 单事件总决策耗时 (ms) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for row in summary_rows:
        cmax_str = (
            f"{row['mean_final_cmax']:.2f} ± {row['std_final_cmax']:.2f}"
            if row["mean_final_cmax"] is not None
            else "N/A"
        )
        nodes_str = f"{row['mean_mean_split_subbatch_nodes']:.1f}" if row["mean_mean_split_subbatch_nodes"] is not None else "N/A"
        edges_str = f"{row['mean_mean_split_total_edges']:.1f}" if row["mean_mean_split_total_edges"] is not None else "N/A"
        gb_str = f"{row['mean_mean_split_graph_build_ms']:.3f}" if row["mean_mean_split_graph_build_ms"] is not None else "N/A"
        hf_str = f"{row['mean_mean_split_hgnn_forward_ms']:.3f}" if row["mean_mean_split_hgnn_forward_ms"] is not None else "N/A"
        dec_str = f"{row['mean_mean_event_decision_ms']:.3f}" if row["mean_mean_event_decision_ms"] is not None else "N/A"
        lines.append(
            f"| Stage {row['stage']} | {row['method_id']} | {row['valid_scenario_n']}/{row['scenario_count']} | "
            f"{cmax_str} | {nodes_str} | {edges_str} | {gb_str} | {hf_str} | {dec_str} |"
        )
    if convergence_rows:
        lines.extend([
            "",
            "## 训练收敛指标对比（2.5 节稳健定义）",
            "",
            "| Stage | 方法 | Seed | 状态 | 达到末值 1% 内轮次 | 归一化曲线下面积 (AUC) | 末轮验证 $C_{\\max}$ | 最佳验证 $C_{\\max}$ |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for crow in convergence_rows:
            ep1 = f"{int(crow['episodes_to_within_1pct_final'])}" if crow["episodes_to_within_1pct_final"] is not None else "N/A"
            auc = f"{crow['normalized_val_auc']:.4f}" if crow["normalized_val_auc"] is not None else "N/A"
            fval = f"{crow['final_val_cmax']:.2f}" if crow["final_val_cmax"] is not None else "N/A"
            bval = f"{crow['best_val_cmax']:.2f}" if crow["best_val_cmax"] is not None else "N/A"
            lines.append(
                f"| Stage {crow['stage']} | {crow['method_id']} | {crow['seed']} | {crow['status']} | "
                f"{ep1} | {auc} | {fval} | {bval} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default="all", choices=("1", "2", "3", "all"))
    parser.add_argument("--split", default="test", choices=("val", "test", "factor_test"))
    parser.add_argument("--seeds", default="1", help="Comma-separated seed list, e.g. 1 or 1,2,3")
    parser.add_argument("--scenario", action="append", help="Optional scenario ID; repeatable")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-untrained-smoke",
        action="store_true",
        help="Dry-run with untrained weights only (marked SMOKE_UNTRAINED in manifest).",
    )
    args = parser.parse_args(argv)

    config, config_sha = load_config(args.config)
    stages = (1, 2, 3) if args.stage == "all" else (int(args.stage),)
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    if not seeds:
        raise ValueError("At least one seed must be specified via --seeds.")

    scenarios, index_sha = select_scenarios(split=args.split, stages=stages, scenario_ids=args.scenario)
    torch.set_num_threads(int(config["cpu_threads"]))
    selected_device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    by_stage: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in scenarios:
        by_stage[int(item["stage"])].append(item)

    trainers, checkpoints_meta = _load_or_build_trainers(
        config=config,
        by_stage=by_stage,
        seeds=seeds,
        device=selected_device,
        allow_untrained_smoke=bool(args.allow_untrained_smoke),
    )

    out_dir = (
        Path(args.output)
        if args.output
        else ROOT / "result" / "消融实验" / "全模板预展开" / "评测" / ("expand_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    eval_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    for item in scenarios:
        stage = int(item["stage"])
        for method_id in (MAIN_METHOD_ID, DAPPO_EXPAND_METHOD_ID):
            for seed in seeds:
                row, ev_rows = replay_scenario(
                    item=item,
                    method_id=method_id,
                    seed=seed,
                    trainer=trainers[(method_id, stage, seed)],
                    config=config,
                    device=selected_device,
                )
                eval_rows.append(row)
                event_rows.extend(ev_rows)

    summary_rows = summarize_eval_rows(eval_rows)
    convergence_rows = summarize_convergence_logs(stages=stages, seeds=seeds)

    _write_csv(out_dir / "expand_events.csv", event_rows)
    _write_csv(out_dir / "expand_eval.csv", eval_rows)
    _write_csv(out_dir / "expand_summary.csv", summary_rows)
    if convergence_rows:
        _write_csv(out_dir / "expand_convergence.csv", convergence_rows)

    report_md = _format_report(
        summary_rows=summary_rows,
        convergence_rows=convergence_rows,
        split=args.split,
        seeds=seeds,
        smoke_mode=bool(args.allow_untrained_smoke),
    )
    (out_dir / "l11_c2_report.md").write_text(report_md, encoding="utf-8")

    manifest = {
        "schema": config["schema"],
        "claim": config["claim"],
        "config_sha256": config_sha,
        "source_sha256": _source_digest(),
        "scenario_index_sha256": index_sha,
        "evidence_status": "SMOKE_UNTRAINED" if args.allow_untrained_smoke else "TRAINED_CHECKPOINT_EVALUATION",
        "data_status": "PRE_FORMAL_ONLY_PROXY_DATA",
        "allow_untrained_smoke": bool(args.allow_untrained_smoke),
        "split": args.split,
        "stages": list(stages),
        "seeds": seeds,
        "device": str(selected_device),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "checkpoints": checkpoints_meta,
        "selected_scenarios": [{k: v for k, v in r.items() if k != "path"} for r in scenarios],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)
    return 0 if all(bool(r["completed"]) for r in eval_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
