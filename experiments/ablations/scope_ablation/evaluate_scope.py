"""L7: three affected-scope variants for raw dual-Actor and S_BASE on test cases."""
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

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "agent", ROOT / "environment"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.ablations.scope_ablation.scope_metrics import event_perturbation
from data_models import POOL_D
from dynamic_event_manager import DynamicEventManager
from dynamic_validator import load_dynamic_orders_and_events


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    result = hashlib.sha256()
    for folder in (ROOT / "agent", ROOT / "environment", ROOT / "experiments/ablations/scope_ablation"):
        for path in sorted(folder.rglob("*.py")):
            result.update(path.relative_to(ROOT).as_posix().encode())
            result.update(path.read_bytes())
    return result.hexdigest()


def load_config(path=None):
    file = Path(path) if path else ROOT / "configs/scope_ablation.json"
    config = json.loads(file.read_text(encoding="utf-8-sig"))
    if (config.get("schema") != "scope_ablation_v1" or
        config.get("scope_labels") != ["zero", "default", "full_tail"] or
        config.get("methods") != ["AD-HG-DAPPO", "S_BASE"] or
        config.get("default_k_scope") != 8 or config.get("default_k_seed") != 5 or
        config.get("delta_k") != 4 or config.get("full_tail_sentinel", 0) < 900000 or
        config.get("max_partition_templates") != 16 or config.get("cpu_threads", 0) < 1):
        raise ValueError("L7 configuration violates the fixed three-scope contract")
    return config, digest(file)


def select_test(stage=None, scenario_ids=None):
    base = ROOT / "data/dynamic_scenarios"
    index_path = base / "scenario_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if index["counts"]["test"] != 36:
        raise ValueError("L7 requires the registered 36 test scenarios")
    requested = set(scenario_ids or [])
    found = []
    for row in index["scenarios"]:
        if row["role"] != "test" or (stage is not None and row["stage"] != stage):
            continue
        if requested and row["scenario_id"] not in requested:
            continue
        path = (base / row["relative_path"]).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"L7 scenario path/hash mismatch: {row['scenario_id']}")
        found.append({**row, "path": path})
    if not found or (requested and requested != {r["scenario_id"] for r in found}):
        raise ValueError("Requested scenario is absent from the selected test stage")
    if len({r["scenario_id"] for r in found}) != len(found):
        raise ValueError("Duplicate scenario IDs")
    if not requested and len(found) != (36 if stage is None else 12):
        raise ValueError("Incomplete L7 test coverage")
    return found, digest(index_path)


def scope_parameters(label, config):
    if label == "zero":
        return 0, 0
    if label == "default":
        return config["default_k_scope"], config["default_k_seed"]
    if label == "full_tail":
        return config["full_tail_sentinel"], config["default_k_seed"]
    raise ValueError(f"Unknown scope {label}")


def build_manager(item, label, config):
    initial, events = load_dynamic_orders_and_events(str(item["path"]), config["sheet_name"])
    k_scope, k_seed = scope_parameters(label, config)
    return DynamicEventManager(pool_type=POOL_D, initial_orders=initial, events=events,
        instance_name=config["sheet_name"], k_scope=k_scope, k_seed=k_seed,
        delta_k=config["delta_k"], formal_mode=True)


def load_trainer(item, checkpoint, config):
    import torch
    from hppo_trainer import HPPOTrainer
    initial, events = load_dynamic_orders_and_events(str(item["path"]), config["sheet_name"])
    trainer = HPPOTrainer(orders=[*initial, *(e.order for e in events)],
                          pool_type=POOL_D, device=torch.device("cpu"))
    trainer.load_checkpoint(str(checkpoint), load_optimizer=False)
    for module in (trainer.hgnn, trainer.manager_policy, trainer.worker_policy, trainer.shared_critic):
        module.eval()
    return trainer


def replay(item, label, method, config, trainer=None):
    """One independent replay; failed raw Actor decisions never get S_BASE repair."""
    from dynamic_two_agent_runner import DynamicTwoAgentRunner
    import torch
    setup_start = time.perf_counter()
    manager = build_manager(item, label, config)
    setup_seconds = time.perf_counter() - setup_start
    runner = None
    if method == "AD-HG-DAPPO":
        if trainer is None:
            raise ValueError("Main-method checkpoint has not been loaded")
        runner = DynamicTwoAgentRunner(manager, trainer.hgnn, trainer.manager_policy,
            trainer.worker_policy, value_critic=trainer.shared_critic,
            device=torch.device("cpu"), max_templates=config["max_partition_templates"])
    elif method != "S_BASE":
        raise ValueError(method)

    event_rows = []
    failure = None
    while manager.pending_event_ids:
        event_id = manager.pending_event_ids[0]
        started = time.perf_counter()  # includes trigger, scope, graph/selection, decode, validation and commit
        context = None
        decision = None
        effective_scope = None
        try:
            context = manager.trigger_next_event()
            before = context.base_solution
            if method == "AD-HG-DAPPO":
                with torch.inference_mode():
                    decision = runner.run_next_event(deterministic=True, context=context)
                committed = bool(decision.committed)
                errors = list(decision.errors)
                effective_scope = len(decision.affected_uids)
            else:
                if label == "zero":
                    # The existing S_BASE expansion loop would otherwise change k=0.
                    candidate = manager.build_independent_baseline_candidate(
                        context, max_templates=config["max_partition_templates"])
                else:
                    candidate = manager.run_independent_baseline(
                        context, max_templates=config["max_partition_templates"])
                current_context = manager.active_context
                effective_scope = len(current_context.affected_uids)
                committed, _, errors = manager.begin_transaction(current_context).commit(candidate)
            after = manager.current_solution if committed else None
            if not committed:
                failure = "; ".join(map(str, errors)) or "Candidate did not pass atomic commit"
        except Exception as exc:
            committed = False
            after = None
            failure = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        metrics = event_perturbation(before, after, context.frozen_uids) if committed else {}
        old_unfrozen = (len(set(before.get_all_active_uids()) - set(context.frozen_uids))
                        if context is not None else None)
        if committed and label == "zero" and (metrics["moved_old_batches"] != 0 or effective_scope != 0):
            raise AssertionError("k_scope=0 moved old batches or selected an old affected UID")
        if committed and label == "full_tail" and effective_scope != old_unfrozen:
            raise AssertionError("full-tail scope did not include every unfrozen old batch")
        event_rows.append({
            "scenario_id": item["scenario_id"], "stage": item["stage"], "scenario_sha256": item["sha256"],
            "method_id": method, "scope_variant": label, "event_id": event_id,
            "event_time_min": context.event_time if context else None,
            "committed": bool(committed), "cmax_before": float(before.makespan) if context else None,
            "cmax_after": float(after.makespan) if after else None,
            "initial_affected_old_count": len(context.affected_uids) if context else None,
            "effective_affected_old_count": effective_scope,
            "expansion_count": int(decision.expansion_count) if decision is not None else
                (max(0, (manager.active_context.k_scope - context.k_scope) // config["delta_k"])
                 if context and manager.active_context and label == "default" else 0),
            "decision_s": elapsed, "failure_reason": failure,
            **metrics,
        })
        if failure:
            manager.rollback_active()
            break

    completed = (failure is None and len(event_rows) == len(manager.events)
                 and all(row["committed"] for row in event_rows)
                 and not manager.pending_event_ids and manager.active_context is None)
    valid = [r for r in event_rows if r["committed"]]
    result = {
        "scenario_id": item["scenario_id"], "stage": item["stage"], "split": "test",
        "origin": item["origin"], "scenario_sha256": item["sha256"],
        "method_id": method, "scope_variant": label, "k_scope_requested": scope_parameters(label, config)[0],
        "completed": completed, "final_cmax": float(manager.current_solution.makespan) if completed else None,
        "event_count": len(manager.events), "committed_events": len(valid),
        "mean_moved_old_per_event": statistics.mean(r["moved_old_batches"] for r in valid) if completed else None,
        "mean_cross_machine_per_event": statistics.mean(r["cross_machine_migrations"] for r in valid) if completed else None,
        "mean_replaced_old_per_event": statistics.mean(r["replaced_old_batches"] for r in valid) if completed else None,
        "mean_start_shift_min": (sum(r["start_shift_abs_sum_min"] for r in valid) /
            sum(r["start_shift_observed_batches"] for r in valid)
            if completed and sum(r["start_shift_observed_batches"] for r in valid) else
            (0.0 if completed else None)),
        "mean_event_decision_s": statistics.mean(r["decision_s"] for r in valid) if completed else None,
        "mean_effective_affected_old": statistics.mean(r["effective_affected_old_count"] for r in valid) if completed else None,
        "setup_s": setup_seconds, "failure_reason": failure,
    }
    return result, event_rows


def summarize(rows):
    output = []
    for stage, method, label in sorted({(r["stage"], r["method_id"], r["scope_variant"]) for r in rows}):
        group = [r for r in rows if (r["stage"], r["method_id"], r["scope_variant"]) == (stage, method, label)]
        good = [r for r in group if r["completed"]]
        item = {"stage": stage, "method_id": method, "scope_variant": label,
                "scenario_count": len(group), "completed_count": len(good),
                "completion_rate": len(good) / len(group), "valid_cmax_n": len(good),
                "all_completed": len(group) == len(good)}
        for field in ("final_cmax", "mean_moved_old_per_event", "mean_cross_machine_per_event",
                      "mean_replaced_old_per_event", "mean_start_shift_min", "mean_event_decision_s",
                      "mean_effective_affected_old"):
            item["group_" + field] = statistics.mean(r[field] for r in good) if good else None
        output.append(item)
    return output


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3])
    parser.add_argument("--scenario", action="append", help="Registered test scenario ID; repeatable")
    parser.add_argument("--scope", choices=["all", "zero", "default", "full_tail"], default="all")
    parser.add_argument("--method", choices=["both", "AD-HG-DAPPO", "S_BASE"], default="both")
    parser.add_argument("--output", help="New result directory; existing paths are never overwritten")
    parser.add_argument("--resume", help="Resume a previously created L7 result directory")
    args = parser.parse_args(argv)
    if args.output and args.resume:
        parser.error("--output and --resume cannot be combined")
    config, config_hash = load_config(args.config)
    scenarios, index_hash = select_test(args.stage, args.scenario)
    import torch
    torch.set_num_threads(config["cpu_threads"])
    scope_labels = config["scope_labels"] if args.scope == "all" else [args.scope]
    methods = config["methods"] if args.method == "both" else [args.method]
    by_stage = defaultdict(list)
    for item in scenarios:
        by_stage[item["stage"]].append(item)
    checkpoints = {}
    trainers = {}
    if "AD-HG-DAPPO" in methods:
        for stage, items in by_stage.items():
            checkpoint = (ROOT / config["checkpoints"][str(stage)]).resolve()
            if not checkpoint.is_relative_to(ROOT) or not checkpoint.is_file():
                raise FileNotFoundError(f"Missing in-project trained checkpoint: {checkpoint}")
            trainers[stage] = load_trainer(items[0], checkpoint, config)
            checkpoints[str(stage)] = {"path": str(checkpoint), "sha256": digest(checkpoint)}
    output = Path(args.resume) if args.resume else (Path(args.output) if args.output else
        ROOT / "result/消融实验/受影响范围" /
        ("scope_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    manifest = {
        "schema": config["schema"], "config": config, "config_sha256": config_hash,
        "source_sha256": source_digest(),
        "scenario_index_sha256": index_hash, "checkpoints": checkpoints,
        "device": "cpu", "torch": torch.__version__, "python": sys.version,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "cpu_threads": config["cpu_threads"],
        "zero_shot_scope_change": True, "training_scope": 8,
        "data_status": "PRE_FORMAL_ONLY_PROXY_DATA", "arguments": vars(args),
        "selected_scenarios": [{k: v for k, v in r.items() if k != "path"} for r in scenarios],
    }
    if args.resume:
        previous = json.loads((output / "run_manifest.json").read_text(encoding="utf-8-sig"))
        for key in ("config_sha256", "source_sha256", "scenario_index_sha256", "checkpoints",
                    "selected_scenarios", "device", "cpu_threads"):
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Cannot resume: L7 provenance changed ({key})")
        old_args = previous["arguments"]
        if any(old_args.get(key) != vars(args)[key] for key in ("stage", "scenario", "scope", "method")):
            raise ValueError("Cannot resume a different L7 scenario/method/scope selection")
        scenario_rows = read_csv(output / "scope_eval.csv") if (output / "scope_eval.csv").is_file() else []
        event_rows = read_csv(output / "scope_events.csv") if (output / "scope_events.csv").is_file() else []
        for row in scenario_rows:
            row["stage"] = int(row["stage"])
            row["completed"] = row["completed"].lower() == "true"
            for key in ("final_cmax", "mean_moved_old_per_event", "mean_cross_machine_per_event",
                        "mean_replaced_old_per_event", "mean_start_shift_min", "mean_event_decision_s",
                        "mean_effective_affected_old"):
                row[key] = None if row[key] in ("", "None") else float(row[key])
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
        scenario_rows, event_rows = [], []
    completed_keys = {(r["scenario_id"], r["method_id"], r["scope_variant"]) for r in scenario_rows}
    for item in scenarios:
        for method in methods:
            for label in scope_labels:
                if (item["scenario_id"], method, label) in completed_keys:
                    continue
                record, event_records = replay(item, label, method, config, trainers.get(item["stage"]))
                scenario_rows.append(record)
                event_rows.extend(event_records)
                write_csv(output / "scope_eval.csv", scenario_rows)
                write_csv(output / "scope_events.csv", event_rows)
                write_csv(output / "scope_summary.csv", summarize(scenario_rows))
                print(f"{item['scenario_id']} {method} {label}: complete={record['completed']}, "
                      f"Cmax={record['final_cmax']}", flush=True)
    print(output)
    return 0 if all(r["completed"] for r in scenario_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
