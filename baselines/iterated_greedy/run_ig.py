"""Run 事件级 IG against the unified stage/role index (original and L4 scenarios)."""
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
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.iterated_greedy.event_ig import IGConfig, ROOT, run_ig_scenario


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for folder in ("environment", "agent", "baselines", "configs"):
        for p in sorted((ROOT / folder).rglob("*")):
            path: Path = p
            if path.suffix in {".py", ".json"}:
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def machine_info(config):
    return {"node": platform.node(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "python": sys.version, "device": "cpu", "cpu_threads": config.cpu_threads}


def select_scenarios(split="test", stage=None, scenario_ids=None):
    root = ROOT / "data/dynamic_scenarios"
    index = json.loads((root / "scenario_index.json").read_text(encoding="utf-8-sig"))
    requested = set(scenario_ids or [])
    selected = []
    for item in index["scenarios"]:
        if item["role"] != split or (stage is not None and item["stage"] != stage):
            continue
        if requested and item["scenario_id"] not in requested:
            continue
        path = (root / item["relative_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != item["sha256"]:
            raise ValueError(f"Scenario path/hash mismatch: {path}")
        selected.append({**item, "path": path})
    if not selected or (requested and requested != {r["scenario_id"] for r in selected}):
        raise ValueError("No matching scenarios, or requested IDs do not belong to this stage/role.")
    if len({r["scenario_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate scenario IDs.")
    return selected


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_output(path=None, prefix="ig"):
    path = Path(path) if path else ROOT / "result/对比算法/ig" / (
        prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_main_times(path, scenarios, config):
    """A calibrated bundle is hash-bound to this machine, source, config and data."""
    if not path:
        raise ValueError("10x requires --main-timing from calibrate_main.py; no estimated timing is substituted.")
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8-sig"))
    import torch
    expected = {"source_sha256": source_hash(), "config_sha256": config.digest(),
                "machine": machine_info(config), "csv_sha256": sha256(path),
                "timing_scope": "trigger_to_commit_cpu_wall", "method_id": "AD-HG-DAPPO",
                "torch": torch.__version__}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Main timing provenance mismatch: {key}. Recalibrate.")
    selected = {r["scenario_id"]: r for r in scenarios}
    result = {sid: {} for sid in selected}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sid = row["scenario_id"]
            if sid not in selected:
                continue
            if (row["scenario_sha256"] != selected[sid]["sha256"] or
                    int(row["stage"]) != selected[sid]["stage"] or row["split"] != selected[sid]["role"] or
                    row["checkpoint_sha256"] != meta.get("checkpoint_sha256") or
                    row["completed"].lower() != "true" or row["committed"].lower() != "true"):
                raise ValueError(f"Invalid/incomplete main timing: {sid}")
            duration = float(row["decision_s"])
            event = row["event_id"]
            if not math.isfinite(duration) or duration <= 0 or event in result[sid]:
                raise ValueError(f"Duplicate/invalid main timing: {sid}/{event}")
            result[sid][event] = duration
    # Check actual event identities before any IG run, not just expected counts.
    from dynamic_validator import load_dynamic_orders_and_events
    for sid, item in selected.items():
        _, events = load_dynamic_orders_and_events(str(item["path"]), config.sheet_name)
        if set(result[sid]) != {e.event_id for e in events}:
            raise ValueError(f"Missing/extra main timings: {sid}")
    return result


def summarize(rows):
    summaries = []
    for sid, budget in sorted({(r["scenario_id"], r["budget_tag"]) for r in rows}):
        group = [r for r in rows if (r["scenario_id"], r["budget_tag"]) == (sid, budget)]
        done = [r for r in group if r["completed"]]
        compliant = [r for r in done if r.get("budget_met")]
        all_three = len(group) == len(done) == 3
        all_three_compliant = len(group) == len(compliant) == 3
        values = [r["cmax"] for r in done]
        strict_values = [r["cmax"] for r in compliant]
        total_events = sum(int(r.get("event_count") or 0) for r in group)
        total_misses = sum(int(r.get("budget_misses") or 0) for r in group)
        event_met_rate = ((total_events - total_misses) / total_events) if total_events > 0 else (
            1.0 if total_misses == 0 else None)
        summaries.append({"scenario_id": sid, "stage": group[0]["stage"], "split": group[0]["split"],
            "budget_tag": budget, "restarts": len(group), "completed_restarts": len(done),
            "compliant_restarts": len(compliant),
            "completion_rate": len(done) / len(group),
            "scenario_budget_met_rate": len(compliant) / len(group),
            "event_budget_met_rate": event_met_rate,
            "strict_cmax_mean": statistics.mean(strict_values) if all_three_compliant else None,
            "strict_cmax_std": statistics.stdev(strict_values) if all_three_compliant else None,
            "cmax_mean": statistics.mean(values) if all_three else None,
            "cmax_std": statistics.stdev(values) if all_three else None,
            "mean_decision_s": statistics.mean(r["total_decision_s"] for r in group),
            "mean_search_s": statistics.mean(r["total_search_s"] for r in group),
            "budget_misses": total_misses,
            "all_budgets_met": all_three and all(r["budget_met"] for r in group),
            "run_kind": group[0]["run_kind"]})
    return summaries


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--split", choices=["test", "factor_test", "val"], default="test")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Stage 1, 2, or 3 (default: 1)")
    p.add_argument("--scenario", action="append", help="Registered scenario ID; repeat to select multiple.")
    p.add_argument("--budget", choices=["10x", "5s", "both"], default="5s", help="Budget tag (default: 5s)")
    p.add_argument("--main-timing", help="main_event_times.csv produced by calibrate_main.py")
    p.add_argument("--output", help="New directory; an existing directory is never overwritten.")
    p.add_argument("--smoke-iterations", type=int, help="Iteration cap; all results explicitly marked smoke.")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = IGConfig.load(args.config)
    if args.smoke_iterations is not None and args.smoke_iterations < 0:
        raise ValueError("smoke-iterations must be nonnegative")
    import torch
    torch.set_num_threads(config.cpu_threads)
    scenarios = select_scenarios(args.split, args.stage, args.scenario)
    budgets = config.budgets if args.budget == "both" else [args.budget]
    times = load_main_times(args.main_timing, scenarios, config) if "10x" in budgets else {}
    output = new_output(args.output)
    write_json(output / "run_manifest.json", {"config": asdict(config), "config_sha256": config.digest(),
        "source_sha256": source_hash(), "machine": machine_info(config), "torch": torch.__version__,
        "arguments": vars(args), "scenario_index_sha256": sha256(ROOT / "data/dynamic_scenarios/scenario_index.json"),
        "main_timing_sha256": sha256(args.main_timing) if args.main_timing else None,
        "data_status": "PRE_FORMAL_ONLY_PROXY_DATA", "repeat_unit": "independent_scenario_restart",
        "selected_scenarios": [{k: v for k, v in r.items() if k != "path"} for r in scenarios]})
    rows, events = [], []
    for item in scenarios:
        for budget in budgets:
            for restart, seed in enumerate(config.restart_seeds, 1):
                result = run_ig_scenario(item["path"], config, budget, seed,
                    main_event_seconds=times.get(item["scenario_id"]), iteration_limit=args.smoke_iterations)
                result.update(restart=restart, stage=item["stage"], split=item["role"], origin=item["origin"],
                              scenario_sha256=item["sha256"], scenario_path=item["relative_path"])
                write_json(output / f"{item['scenario_id']}__{budget}__r{restart}.json", result)
                rows.append({k: v for k, v in result.items() if k != "events"})
                for event in result["events"]:
                    events.append({"scenario_id": item["scenario_id"], "stage": item["stage"],
                        "split": item["role"], "budget_tag": budget, "restart": restart,
                        **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                           for k, v in event.items()}})
                write_csv(output / "ig_runs.csv", rows)
                write_csv(output / "ig_events.csv", events)
                write_csv(output / "ig_summary.csv", summarize(rows))
                print(f"{item['scenario_id']} {budget} restart={restart} completed={result['completed']} "
                      f"Cmax={result['cmax']} budget_misses={result['budget_misses']}", flush=True)
    print(f"Artifacts: {output}")
    return 0 if all(row["completed"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())

