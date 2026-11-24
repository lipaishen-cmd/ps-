"""Measure rule decisions/training, and report verified event latency.

Run `python experiments/runtime/decision_training_runtime.py --help` from the project root.
No solver run is started by the `report` subcommand.
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
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "result" / "耗时实验" / "决策与训练耗时"
METHODS = ("AD-HG-DAPPO", "S_BASE", "FIFO", "SPT", "LPT", "EDD", "ECT",
           "IG-10x", "IG-5s", "AD-HG-SAPPO")
RULES = ("S_BASE", "FIFO", "SPT", "LPT", "EDD", "ECT")
BOUNDARY = "event_trigger_to_validated_atomic_commit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def new_dir(path: str | None, tag: str) -> Path:
    target = Path(path).resolve() if path else OUT_ROOT / (tag + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    target.mkdir(parents=True, exist_ok=False)
    return target


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1", "yes"):
        return True
    if str(value).strip().lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def scenario_index():
    base = ROOT / "data" / "dynamic_scenarios"
    path = base / "scenario_index.json"
    data = read_json(path)
    lookup = {(item["role"], int(item["stage"]), item["scenario_id"]): item
              for item in data["scenarios"]}
    if len(lookup) != len(data["scenarios"]):
        raise ValueError("Duplicate scenario in scenario_index.json")
    return lookup, sha256(path)


def hardware(device: str = "cpu"):
    info = {"node": platform.node(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "python": sys.version, "device": device}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


def benchmark_rules(args):
    """Reuse the six original rules/decoder/transaction; never create a new heuristic."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "baselines"))
    from baseline_config import BaselineConfig
    from baseline_runtime import RULE_NAMES, run_rule_scenario
    if tuple(RULE_NAMES) != RULES:
        raise ValueError("Rule registry changed; check runtime-report method list")
    index, index_hash = scenario_index()
    selected = [item for (split, stage, sid), item in index.items()
                if split == args.split and (args.stage is None or stage == args.stage)
                and (not args.scenario or sid in args.scenario)]
    if not selected or (args.scenario and set(args.scenario) != {x["scenario_id"] for x in selected}):
        raise ValueError("Requested scenarios are not registered in this split")
    folder = new_dir(args.output, "rules")
    config = BaselineConfig()
    records, scenario_results = [], []
    for item in selected:
        source = (ROOT / "data" / "dynamic_scenarios" / item["relative_path"]).resolve()
        if not source.is_file() or sha256(source) != item["sha256"]:
            raise ValueError(f"Scenario hash mismatch: {item['scenario_id']}")
        for method in args.rule or RULES:
            result = run_rule_scenario(str(source), method, config)
            scenario_results.append({"method_id": method, "split": args.split,
                "stage": item["stage"], "scenario_id": item["scenario_id"],
                "scenario_sha256": item["sha256"], "seed": "", "repeat": 1,
                "completed": result.completed, "final_cmax": result.final_cmax,
                "checkpoint_type": "not_applicable", "failure_reason": result.failure_reason or ""})
            for event in result.event_records:
                records.append({"stage": item["stage"], "split": args.split,
                    "scenario_id": item["scenario_id"], "scenario_sha256": item["sha256"],
                    "method_id": method, "event_id": event.event_id,
                    "committed": event.committed, "completed": result.completed,
                    "decision_s": event.decision_s,
                    "failure_reason": result.failure_reason or ""})
            write_csv(folder / "rule_event_times.csv", records)
            write_csv(folder / "rule_scenario_results.csv", scenario_results)
            print(f"{method} {item['scenario_id']}: completed={result.completed}", flush=True)
    write_json(folder / "run_manifest.json", {"schema": "rule_decision_timing_v1",
        "timing_boundary": BOUNDARY, "scenario_index_sha256": index_hash,
        "machine": hardware(), "selected_scenarios": selected,
        "methods": args.rule or list(RULES), "source_sha256": sha256(Path(__file__)),
        "event_csv_sha256": sha256(folder / "rule_event_times.csv"),
        "scenario_csv_sha256": sha256(folder / "rule_scenario_results.csv")})
    print(folder)
    return 0


def event_rows(path: Path, index: dict, index_hash: str):
    """Accept only event-level, trigger-to-commit evidence with a known source schema."""
    name = path.name
    if name not in ("scope_events.csv", "ig_events.csv", "main_event_times.csv",
                    "rule_event_times.csv", "event_decisions.csv"):
        raise ValueError(f"Unsupported event source: {path}")
    manifest_path = path.with_suffix(".meta.json") if name == "main_event_times.csv" else path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Timing metadata is required: {manifest_path}")
    meta = read_json(manifest_path)
    expected_boundary = "trigger_to_commit_cpu_wall" if name == "main_event_times.csv" else BOUNDARY
    if name == "main_event_times.csv":
        if meta.get("timing_scope") != expected_boundary or meta.get("csv_sha256") != sha256(path):
            raise ValueError("Main timing metadata boundary/hash mismatch")
    elif name in ("rule_event_times.csv", "event_decisions.csv"):
        if meta.get("timing_boundary") != BOUNDARY or meta.get("event_csv_sha256") != sha256(path):
            raise ValueError("Event timing metadata boundary/hash mismatch")
        if name == "event_decisions.csv" and meta.get("schema") != "event_decisions_v1":
            raise ValueError("Generic event timing schema mismatch")
    elif name == "ig_events.csv":
        if meta.get("arguments", {}).get("smoke_iterations") is not None:
            raise ValueError("IG smoke run cannot enter the decision-time report")
    if name != "main_event_times.csv" and meta.get("scenario_index_sha256") != index_hash:
        raise ValueError(f"Scenario index hash mismatch: {path}")
    run_map = {}
    scope_map = {}
    if name == "ig_events.csv":
        for run in read_csv(path.parent / "ig_runs.csv"):
            run_map[(run["scenario_id"], run["budget_tag"], run["restart_seed"])] = run
    if name == "scope_events.csv":
        for run in read_csv(path.parent / "scope_eval.csv"):
            scope_map[(run["scenario_id"], run["method_id"], run["scope_variant"])] = run
    normalized = []
    for row in read_csv(path):
        if name == "scope_events.csv" and row["scope_variant"] != "default":
            continue  # Ordinary k_scope=8 method only; exclude range ablations.
        if name == "ig_events.csv":
            method = "IG-" + row["budget_tag"]
            restart = str(row["restart"])
            run = run_map.get((row["scenario_id"], row["budget_tag"], restart))
            if not run or run["run_kind"] != "budgeted":
                raise ValueError("IG event has no matching formal-budget restart")
            completed = as_bool(run["completed"])
        else:
            method = row.get("method_id") or meta.get("method_id")
            restart = str(row.get("restart") or "1")
            if name == "scope_events.csv":
                run = scope_map.get((row["scenario_id"], method, row["scope_variant"]))
                if run is None:
                    raise ValueError("L7 event has no matching scenario replay")
                completed = as_bool(run["completed"])
            else:
                completed = as_bool(row["completed"])
        if method not in METHODS:
            raise ValueError(f"Unknown runtime-report method: {method}")
        stage = int(row["stage"])
        split = row["split"] if "split" in row else "test"
        if split != "test":
            continue
        key = (split, stage, row["scenario_id"])
        item = index.get(key)
        observed_hash = run["scenario_sha256"] if name == "ig_events.csv" else row.get("scenario_sha256")
        if item is None or observed_hash != item["sha256"]:
            raise ValueError(f"Unregistered or stale test scenario: {key}")
        duration = float(row["decision_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid event duration: {path} / {row['event_id']}")
        normalized.append({"stage": stage, "method_id": method,
            "scenario_id": row["scenario_id"], "event_id": row["event_id"],
            "restart": restart, "committed": as_bool(row["committed"]),
            "completed": completed, "decision_ms": duration * 1000.0,
            "budget_met": (as_bool(row["budget_met"]) if name == "ig_events.csv" and "budget_met" in row
                           else (True if method == "AD-HG-DAPPO" else None)),
            "source": str(path.resolve())})
    if not normalized:
        raise ValueError(f"No eligible test/default-scope event rows: {path}")
    return normalized, {"path": str(path.resolve()), "sha256": sha256(path),
        "metadata_path": str(manifest_path.resolve()), "metadata_sha256": sha256(manifest_path),
        "machine": meta.get("machine") or {"device": meta.get("device", "unknown")}}


def percentile_nearest_rank(values: list[float], fraction: float):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def report(args):
    index, index_hash = scenario_index()
    raw, sources = [], []
    for source in args.event_file:
        rows, provenance = event_rows(Path(source).resolve(), index, index_hash)
        raw.extend(rows)
        sources.append(provenance)
    seen = set()
    for row in raw:
        key = (row["stage"], row["method_id"], row["scenario_id"], row["event_id"], row["restart"])
        if key in seen:
            raise ValueError(f"Duplicate event observation across files: {key}")
        seen.add(key)
    groups = defaultdict(list)
    for row in raw:
        groups[(row["stage"], row["method_id"])].append(row)
    table = []
    for stage in (1, 2, 3):
        for method in METHODS:
            observations = groups[(stage, method)]
            successful = [r for r in observations if r["committed"] and r["completed"]]
            values = [r["decision_ms"] for r in successful]
            scenarios = {r["scenario_id"] for r in observations}
            required_restarts = 3 if method.startswith("IG-") else 1
            expected_complete = all(
                len([r for r in observations if r["scenario_id"] == sid]) ==
                int(index[("test", stage, sid)]["dynamic_events"]) * required_restarts
                for sid in scenarios)
            all_committed = all(r["committed"] and r["completed"] for r in observations)
            coverage = ("missing" if not observations else
                        "complete" if len(scenarios) == 12 and expected_complete and all_committed else
                        "partial_or_failed")
            budget_obs = [r["budget_met"] for r in observations if r.get("budget_met") is not None]
            budget_met_n = sum(bool(v) for v in budget_obs) if budget_obs else None
            budget_met_rate = (budget_met_n / len(budget_obs)) if budget_obs else None
            table.append({"stage": stage, "method_id": method,
                "scenario_n": len(scenarios), "expected_scenarios": 12,
                "event_attempt_n": len(observations), "successful_event_n": len(values),
                "failed_or_incomplete_n": len(observations) - len(values),
                "budget_met_n": budget_met_n,
                "budget_met_rate": budget_met_rate,
                "mean_ms": statistics.mean(values) if values else None,
                "p95_ms": percentile_nearest_rank(values, .95) if values else None,
                "max_ms": max(values) if values else None,
                "coverage": coverage, "timing_boundary": BOUNDARY})
    training = []
    train_files = [Path(x).resolve() for x in args.training_file]
    train_records = []
    train_keys = set()
    for file in train_files:
        info = read_json(file)
        if (info.get("schema") != "training_wall_clock_v1" or
                info.get("status") != "completed" or
                info.get("training_log_status") != "matched_completed_training" or
                int(info.get("stage", 0)) not in (1, 2, 3) or
                int(info.get("seed", 0)) < 1 or
                info.get("method_id") not in METHODS or
                float(info.get("wall_clock_s", 0)) <= 0):
            raise ValueError(f"Invalid completed training timer: {file}")
        log = Path(info["training_log"])
        if not log.is_file() or sha256(log) != info.get("training_log_sha256"):
            raise ValueError(f"Training log changed since timing: {file}")
        key = (info["stage"], info["method_id"], info["seed"])
        if key in train_keys:
            raise ValueError(f"Duplicate training timer: {key}")
        train_keys.add(key)
        train_records.append((file, info))
    for stage in (1, 2, 3):
        matched = [(file, info) for file, info in train_records if info["stage"] == stage]
        if not matched:
            training.append({"stage": stage, "method_id": "AD-HG-DAPPO",
                "seed": None, "wall_clock_s": None, "hardware": None,
                "status": "not_recorded", "source": None})
        for file, info in matched:
            training.append({"stage": stage, "method_id": info["method_id"],
                "seed": info["seed"], "wall_clock_s": info["wall_clock_s"],
                "hardware": json.dumps(info["hardware"], ensure_ascii=False),
                "status": "measured", "source": str(file)})
    folder = new_dir(args.output, "report")
    write_csv(folder / "event_latency_by_stage.csv", table)
    write_csv(folder / "training_wall_clock.csv", training)
    write_json(folder / "report_manifest.json", {"schema": "decision_training_runtime_report_v1",
        "timing_boundary": BOUNDARY, "percentile": "nearest_rank_p95",
        "aggregation": "pooled_successful_events_from_complete_scenario_replays",
        "scenario_index_sha256": index_hash, "event_sources": sources,
        "training_sources": [{"path": str(p), "sha256": sha256(p)} for p in train_files],
        "data_status": "PRE_FORMAL_ONLY_PROXY_DATA"})
    print(folder)
    return 0


def measure_training(args):
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("Supply training command after --")
    train_log = Path(args.training_log)
    if not train_log.is_absolute():
        train_log = ROOT / train_log
    train_log = train_log.resolve()
    if train_log.exists():
        raise FileExistsError(f"Only a fresh, single training run can be timed: {train_log}")
    if "--fresh-start" not in command or not any(
            part == "--seed" and i + 1 < len(command) and command[i + 1] == str(args.seed)
            or part == f"--seed={args.seed}" for i, part in enumerate(command)):
        raise ValueError("Training command must explicitly use --fresh-start and the declared --seed")
    folder = new_dir(args.output, "training")
    start_utc = datetime.now(timezone.utc).isoformat()
    start_epoch = time.time()
    start = time.perf_counter()
    selected_device = args.compute_device
    if selected_device == "auto":
        if "--device" in command and command.index("--device") + 1 < len(command):
            selected_device = command[command.index("--device") + 1]
        else:
            try:
                import torch
                selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                selected_device = "unknown"
    machine = hardware(selected_device)
    try:
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            code = subprocess.call(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    except OSError as exc:
        code = None
        (folder / "launch_error.txt").write_text(str(exc), encoding="utf-8")
    elapsed = time.perf_counter() - start
    log_status = "missing_or_stale"
    log_hash = None
    if train_log.is_file() and train_log.stat().st_mtime >= start_epoch - 1:
        log_hash = sha256(train_log)
        try:
            log = read_json(train_log)
            if (log.get("status") == "COMPLETED" and
                    int(log.get("stage_only", -1)) == args.stage and
                    int(log.get("seed", -1)) == args.seed and
                    log.get("method_id") == args.method):
                log_status = "matched_completed_training"
            else:
                log_status = "mismatched_or_incomplete_training"
        except (ValueError, TypeError, json.JSONDecodeError):
            log_status = "invalid_training_log"
    payload = {"schema": "training_wall_clock_v1", "stage": args.stage,
        "seed": args.seed, "method_id": args.method, "command": command,
        "cwd": str(ROOT), "start_utc": start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": elapsed, "hardware": machine,
        "training_log": str(train_log), "training_log_sha256": log_hash,
        "training_log_status": log_status,
        "exit_code": code, "status": "completed" if code == 0 and
        log_status == "matched_completed_training" else "failed"}
    write_json(folder / "training_time.json", payload)
    print(folder)
    return 0 if payload["status"] == "completed" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    rules = sub.add_parser("rules", help="Replay six rules on a registered train/val/test split")
    rules.add_argument("--split", choices=("train", "val", "test"), default="test")
    rules.add_argument("--stage", type=int, choices=(1, 2, 3))
    rules.add_argument("--scenario", action="append")
    rules.add_argument("--rule", action="append", choices=RULES)
    rules.add_argument("--output")
    rep = sub.add_parser("report", help="Aggregate measured event/training records; never run solvers")
    rep.add_argument("--event-file", action="append", required=True)
    rep.add_argument("--training-file", action="append", default=[])
    rep.add_argument("--output")
    train = sub.add_parser("train", help="Run one explicitly supplied training command and time it")
    train.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--method", default="AD-HG-DAPPO")
    train.add_argument("--compute-device", default="auto", choices=("auto", "cpu", "cuda"))
    train.add_argument("--training-log", required=True,
                       help="Log written by this training command; stage/seed/status must match")
    train.add_argument("--output")
    train.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return {"rules": benchmark_rules, "report": report, "train": measure_training}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())

