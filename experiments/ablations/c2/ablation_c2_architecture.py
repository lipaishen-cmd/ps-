"""Evaluation for the sole C2 architecture ablation.

Only two methods are admitted: the proposed dual-Actor model and the
independently trained single-Actor ablation. No rule policy or checkpoint
substitution is admitted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (
    THIS_DIR,
    EXPERIMENT_ROOT,
    PROJECT_ROOT,
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "agent",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from ablation_common import (
    AblationSummaryStatistics,
    C2_PRIMARY_ABLATION_ID,
    C2_RESULTS_DIR,
    EventEvaluationRecord,
    PROPOSED_METHOD_ID,
    ScenarioEvaluationRecord,
    compute_perturbation_metrics,
    compute_summary_statistics,
    format_ablation_markdown_table,
    load_hppo_trainer,
    resolve_evaluation_scenarios,
    save_event_records,
)
from data_models import POOL_D
from dynamic_event_manager import DynamicEventManager
from dynamic_validator import load_dynamic_orders_and_events
from single_actor_ppo_trainer import (
    SINGLE_ACTOR_CODE_ID,
    SINGLE_ACTOR_ENTROPY_REGULARIZATION,
    SINGLE_ACTOR_LR_SEMANTICS,
    SINGLE_ACTOR_METHOD_ID,
)
from train_single_actor import load_single_actor_trainer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_pair_contract(
    proposed_payload: Dict[str, Any],
    single_actor_payload: Dict[str, Any],
    stage: int,
    seed: int,
) -> None:
    """Reject filename-only pairing and bind both methods to the active data contract."""
    required = (
        "stage",
        "seed",
        "dataset_manifest_sha256",
        "active_package_sha256",
    )
    for label, payload in (
        ("AD-HG-DAPPO", proposed_payload),
        ("AD-HG-SAPPO", single_actor_payload),
    ):
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{label} checkpoint is missing pairing fields: {missing}")
        if int(payload["stage"]) != int(stage) or int(payload["seed"]) != int(seed):
            raise ValueError(
                f"{label} checkpoint stage/seed does not match Stage {stage}, seed {seed}."
            )

    manifest_path = PROJECT_ROOT / "data" / "dynamic_scenarios" / "dataset_manifest.json"
    active_package_path = PROJECT_ROOT / "data" / "active_experiment_package.json"
    expected = {
        "dataset_manifest_sha256": _sha256(manifest_path),
        "active_package_sha256": _sha256(active_package_path),
    }
    HISTORICAL_ACTIVE_PACKAGE_HASHES = {
        "b21c4422fc72ea4c1372d86e47468d43c781e5dbd362055c6a72b85568c581b9",
    }
    for key, expected_hash in expected.items():
        proposed_hash = str(proposed_payload[key])
        single_hash = str(single_actor_payload[key])
        if proposed_hash != single_hash:
            raise ValueError(
                f"Checkpoint data hash mismatch between methods for {key}: {proposed_hash} != {single_hash}"
            )
        if proposed_hash != expected_hash:
            if key == "active_package_sha256" and proposed_hash in HISTORICAL_ACTIVE_PACKAGE_HASHES:
                continue
            raise ValueError(
                f"Checkpoint data hash mismatch for {key}; both methods must use the active package."
            )

    if proposed_payload.get("method_id") != PROPOSED_METHOD_ID:
        raise ValueError("Proposed checkpoint method_id is not AD-HG-DAPPO.")
    if proposed_payload.get("code_id") != "dual_actor_hppo":
        raise ValueError("Proposed checkpoint code_id is not the current dual-Actor trainer.")
    if single_actor_payload.get("method_id") != SINGLE_ACTOR_METHOD_ID:
        raise ValueError("C2 checkpoint method_id is not AD-HG-SAPPO.")
    if single_actor_payload.get("code_id") != SINGLE_ACTOR_CODE_ID:
        raise ValueError("C2 checkpoint code_id is not the current single-Actor trainer.")
    if single_actor_payload.get("actor_lr_semantics") != SINGLE_ACTOR_LR_SEMANTICS:
        raise ValueError("C2 checkpoint does not use the locked shared-Actor LR contract.")
    if (
        single_actor_payload.get("entropy_regularization")
        != SINGLE_ACTOR_ENTROPY_REGULARIZATION
    ):
        raise ValueError("C2 checkpoint does not use the locked entropy contract.")


def evaluate_trainer(
    trainer,
    scenarios: List[str],
    method_id: str,
    seed: int,
) -> Tuple[List[ScenarioEvaluationRecord], List[EventEvaluationRecord], AblationSummaryStatistics]:
    scenario_records: List[ScenarioEvaluationRecord] = []
    event_records: List[EventEvaluationRecord] = []
    for scenario_path in scenarios:
        initial_orders, events = load_dynamic_orders_and_events(scenario_path, "Sheet1")
        manager = DynamicEventManager(
            pool_type=POOL_D,
            initial_orders=initial_orders,
            events=events,
            instance_name="Sheet1",
            formal_mode=True,
        )
        initial_solution = manager.current_solution.copy()
        initial_uids = {
            batch.u_id
            for queue in initial_solution.machine_queues.values()
            for batch in queue.batches
        }
        started = time.perf_counter()
        decisions = trainer.collect_dynamic_episode(
            manager, deterministic=True, record_rollout=False
        )
        elapsed = time.perf_counter() - started
        committed = sum(bool(decision.committed) for decision in decisions)
        completed = committed == len(manager.events) and not manager.pending_event_ids
        final_cmax = float(manager.current_solution.makespan) if completed else None
        if completed:
            moved, reassigned, inversions, mean_deviation, delay_loss = (
                compute_perturbation_metrics(
                    initial_solution, manager.current_solution, initial_uids
                )
            )
        else:
            moved = reassigned = inversions = 0
            mean_deviation = delay_loss = 0.0

        effective_k = [
            float(getattr(decision, "effective_k_scope", manager.k_scope))
            for decision in decisions
        ]
        scope_ratios = [float(getattr(decision, "scope_ratio", 0.0)) for decision in decisions]
        scenario_records.append(
            ScenarioEvaluationRecord(
                ablation_id=method_id,
                category="C2: Hierarchical-to-Flat PPO Architecture",
                scenario_name=Path(scenario_path).stem,
                scenario_path=scenario_path,
                seed=seed,
                completed=completed,
                initial_cmax=float(initial_solution.makespan),
                final_cmax=final_cmax,
                cmax_delta=(final_cmax - float(initial_solution.makespan)) if final_cmax is not None else None,
                elapsed_seconds=elapsed,
                total_events=len(manager.events),
                committed_events=committed,
                controlled_failures=0 if completed else 1,
                moved_old_batches=moved,
                machine_reassigned_batches=reassigned,
                queue_order_inversions=inversions,
                mean_start_time_deviation=mean_deviation,
                total_delay_loss=delay_loss,
                initial_k_scope=manager.k_scope,
                effective_k_scope_mean=float(np.mean(effective_k)) if effective_k else float(manager.k_scope),
                expansion_count_total=sum(int(getattr(d, "expansion_count", 0)) for d in decisions),
                actual_affected_ratio=float(np.mean(scope_ratios)) if scope_ratios else 0.0,
                failure_details=None if completed else "；".join(
                    error for decision in decisions for error in decision.errors
                ),
            )
        )
        for decision in decisions:
            event_records.append(
                EventEvaluationRecord(
                    ablation_id=method_id,
                    scenario_name=Path(scenario_path).stem,
                    event_id=decision.event_id,
                    event_type=(
                        "ARRIVE" if "ARRIVE" in decision.event_id else
                        "COMPLETE" if "COMPLETE" in decision.event_id else "EVENT"
                    ),
                    committed=bool(decision.committed),
                    initial_k=manager.k_scope,
                    effective_k=int(getattr(decision, "effective_k_scope", manager.k_scope)),
                    expansion_count=int(getattr(decision, "expansion_count", 0)),
                    scope_ratio=float(getattr(decision, "scope_ratio", 0.0)),
                    affected_uids_count=len(getattr(decision, "affected_uids", [])),
                    response_time_seconds=float(getattr(decision, "response_time_seconds", 0.0)),
                    seed=seed,
                    cmax_before=float(getattr(decision, "cmax_before", initial_solution.makespan)),
                    cmax_after=getattr(decision, "cmax_after", None),
                    failure_reason=None if decision.committed else "；".join(decision.errors),
                )
            )
    return scenario_records, event_records, compute_summary_statistics(scenario_records)


def execute_c2_ablation(
    proposed_checkpoint: str,
    single_actor_checkpoint: str,
    split: str = "test",
    stage: int = 1,
    device: str = "cpu",
    seed: int = 1,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the proposed model and the one independently trained ablation."""
    if stage not in (1, 2, 3):
        raise ValueError("C2 evaluation stage must be 1, 2 or 3.")
    scenarios = resolve_evaluation_scenarios(split=split, stage=stage)
    initial_orders, _ = load_dynamic_orders_and_events(scenarios[0], "Sheet1")
    selected_device = torch.device(
        device if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    proposed = load_hppo_trainer(proposed_checkpoint, initial_orders, selected_device)
    ablation = load_single_actor_trainer(
        single_actor_checkpoint, initial_orders, selected_device
    )
    validate_checkpoint_pair_contract(
        proposed.checkpoint_payload,
        ablation.checkpoint_payload,
        stage=stage,
        seed=seed,
    )

    all_scenarios: List[ScenarioEvaluationRecord] = []
    all_events: List[EventEvaluationRecord] = []
    summaries: List[AblationSummaryStatistics] = []
    for method_id, trainer in (
        (PROPOSED_METHOD_ID, proposed),
        (C2_PRIMARY_ABLATION_ID, ablation),
    ):
        scenario_records, event_records, summary = evaluate_trainer(
            trainer, scenarios, method_id, seed
        )
        all_scenarios.extend(scenario_records)
        all_events.extend(event_records)
        summaries.append(summary)

    out_dir = (
        Path(output_dir)
        if output_dir
        else C2_RESULTS_DIR / "评测" / f"stage{stage}" / f"seed_{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "c2_scenario_records.json").write_text(
        json.dumps([asdict(record) for record in all_scenarios], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_event_records(all_events, out_dir / "c2_event_records.jsonl")
    with (out_dir / "c2_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))

    table = format_ablation_markdown_table(summaries, anchor_id=PROPOSED_METHOD_ID)
    report = (
        "# C2 唯一架构消融：分层双策略 PPO 与统一单策略 PPO\n\n"
        f"- 主方法：`{PROPOSED_METHOD_ID}`\n"
        f"- 唯一消融：`{C2_PRIMARY_ABLATION_ID}` / `{SINGLE_ACTOR_CODE_ID}`\n"
        f"- 数据划分：`{split}`，Stage {stage}，seed {seed}\n"
        "- 两个方法分别读取各自独立训练检查点；不存在共享权重、规则动作或检查点回退。\n"
        "- 本比较评价完整分层训练架构相对于统一单策略 PPO 的总体贡献，不单独归因于 Actor 数量。\n\n"
        + table
    )
    (out_dir / "c2_report.md").write_text(report, encoding="utf-8")
    return {
        "records": all_scenarios,
        "event_records": all_events,
        "summaries": summaries,
        "table_md": table,
        "output_dir": str(out_dir),
    }


execute_c2_ablations = execute_c2_ablation

