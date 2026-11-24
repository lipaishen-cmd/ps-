"""Measure actual CPU main-policy event decisions for the 事件级 IG 10x budget."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.iterated_greedy.event_ig import IGConfig, _build_manager
from baselines.iterated_greedy.run_ig import (
    machine_info, new_output, select_scenarios, sha256, source_hash, write_csv, write_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Existing trained main-method checkpoint.")
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["test", "factor_test", "val"], default="test")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--d-h", type=int, default=32, help="Must match the saved model architecture.")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = IGConfig.load(args.config)
    scenarios = select_scenarios(args.split, args.stage, args.scenario)
    import torch
    from data_models import POOL_D
    from dynamic_validator import load_dynamic_orders_and_events
    from dynamic_two_agent_runner import DynamicTwoAgentRunner
    from hppo_trainer import HPPOTrainer
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(0)
    initial, dynamic = load_dynamic_orders_and_events(str(scenarios[0]["path"]), config.sheet_name)
    trainer = HPPOTrainer(orders=[*initial, *(event.order for event in dynamic)],
                          pool_type=POOL_D, d_h=args.d_h, device=torch.device("cpu"))
    trainer.load_checkpoint(str(checkpoint), load_optimizer=False)
    for module in (trainer.hgnn, trainer.manager_policy, trainer.worker_policy, trainer.shared_critic):
        module.eval()

    def make_runner(path):
        manager = _build_manager(str(path), config.baseline())
        return manager, DynamicTwoAgentRunner(manager, trainer.hgnn, trainer.manager_policy,
            trainer.worker_policy, value_critic=trainer.shared_critic, device=torch.device("cpu"),
            max_templates=config.max_partition_templates)

    # One discarded event on a fresh manager warms up kernels; no weights or replay buffer change.
    warm_manager, warm_runner = make_runner(scenarios[0]["path"])
    with torch.inference_mode():
        warm_runner.run_next_event(deterministic=True)
    del warm_manager, warm_runner
    output = new_output(args.output, prefix=f"main_timing_stage{args.stage}")
    rows, runs = [], []
    with torch.inference_mode():
        for item in scenarios:
            manager, runner = make_runner(item["path"])
            scenario_rows = []
            failure = None
            while manager.pending_event_ids:
                event_id = manager.pending_event_ids[0]
                start = time.perf_counter()
                try:
                    decision = runner.run_next_event(deterministic=True)
                    committed = bool(decision.committed)
                    errors = decision.errors
                except Exception as exc:
                    committed, errors = False, [f"{type(exc).__name__}: {exc}"]
                duration = time.perf_counter() - start
                scenario_rows.append({"scenario_id": item["scenario_id"], "stage": item["stage"],
                    "split": item["role"], "event_id": event_id, "decision_s": duration,
                    "committed": committed, "scenario_sha256": item["sha256"],
                    "checkpoint_sha256": sha256(checkpoint)})
                if not committed:
                    failure = str(errors)
                    manager.rollback_active()
                    break
            completed = (failure is None and len(scenario_rows) == len(manager.events)
                         and not manager.pending_event_ids and manager.active_context is None)
            for row in scenario_rows:
                row["completed"] = completed
            rows.extend(scenario_rows)
            runs.append({"scenario_id": item["scenario_id"], "stage": item["stage"],
                "split": item["role"], "completed": completed,
                "cmax": float(manager.current_solution.makespan) if completed else None,
                "total_decision_s": sum(r["decision_s"] for r in scenario_rows),
                "event_count": len(manager.events),
                "failure_reason": failure, "scenario_sha256": item["sha256"],
                "checkpoint_sha256": sha256(checkpoint)})
            write_csv(output / "main_event_times.csv", rows)
            write_csv(output / "main_runs.csv", runs)
            print(f"{item['scenario_id']} main completed={completed}", flush=True)
    csv_path = output / "main_event_times.csv"
    write_json(csv_path.with_suffix(".meta.json"), {"source_sha256": source_hash(),
        "config_sha256": config.digest(), "machine": machine_info(config), "csv_sha256": sha256(csv_path),
        "timing_scope": "trigger_to_commit_cpu_wall", "method_id": "AD-HG-DAPPO",
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "torch": torch.__version__, "warmup_events_excluded": 1, "arguments": vars(args)})
    print(f"Artifacts: {output}")
    return 0 if all(r["completed"] for r in runs) else 2


if __name__ == "__main__":
    raise SystemExit(main())

