"""Load registered scheduling outcomes and assemble the paired test panel."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = ("S_BASE", "FIFO", "SPT", "LPT", "EDD", "ECT")
MAIN = "AD-HG-DAPPO"
SINGLE = "AD-HG-SAPPO"
DEPLOYABLE = "DEPLOYABLE_BEST_RULE"
IG = ("IG-10x", "IG-5s")
METHODS = (MAIN, *RULES, DEPLOYABLE, *IG, SINGLE)
MEASURED_METHODS = (MAIN, *RULES, *IG, SINGLE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1", "yes"):
        return True
    if str(value).strip().lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid completed value {value!r}")


def registered_scenarios():
    base = ROOT / "data" / "dynamic_scenarios"
    path = base / "scenario_index.json"
    index = json.loads(path.read_text(encoding="utf-8-sig"))
    if any(index["counts"].get(role) != count for role, count in
           (("train", 18), ("val", 6), ("test", 36))):
        raise ValueError("Unexpected train/val/test scenario counts")
    lookup = {}
    for row in index["scenarios"]:
        key = (row["role"], int(row["stage"]), row["scenario_id"])
        if key in lookup:
            raise ValueError(f"Duplicate registered scenario {key}")
        if row["role"] in ("train", "val", "test"):
            scenario_path = base / row["relative_path"]
            if not scenario_path.is_file() or sha256(scenario_path) != row["sha256"]:
                raise ValueError(f"Registered scenario file missing or changed: {scenario_path}")
        lookup[key] = row
    for role, per_stage in (("train", 6), ("val", 2), ("test", 12)):
        for stage in (1, 2, 3):
            if sum(k[0] == role and k[1] == stage for k in lookup) != per_stage:
                raise ValueError(f"Incomplete {role} stage {stage} index")
    test_origins = [row["origin"] for row in lookup.values() if row["role"] == "test"]
    if test_origins.count("original") != 6 or test_origins.count("L4") != 30:
        raise ValueError("Expected six original and thirty added test scenarios")
    return lookup, sha256(path)


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _best_seed_from_scope_manifest(path: Path, method: str, stage: int):
    if method in RULES:
        return None
    manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8-sig"))
    checkpoint = manifest.get("checkpoints", {}).get(str(stage), {}).get("path", "")
    match = re.search(r"seed_(\d+)[\\/]stage\d+_best\.pt$", checkpoint)
    if manifest.get("training_scope") != 8 or not match:
        raise ValueError(f"No verifiable stage best checkpoint in {path}")
    return int(match.group(1))


def load_records(paths: list[Path], index: dict, index_hash: str):
    """Accept canonical evaluation CSV, IG runs, scope replay and rule replay JSON."""
    records, sources = [], []
    for source in paths:
        path = source.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        name = path.name
        rows = []
        if name == "ig_runs.csv":
            manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8-sig"))
            if (manifest.get("scenario_index_sha256") != index_hash or
                    manifest.get("arguments", {}).get("smoke_iterations") is not None):
                raise ValueError(f"Stale or smoke IG runs: {path}")
            for row in _csv(path):
                if row["run_kind"] != "budgeted":
                    raise ValueError(f"Non-budgeted IG run: {path}")
                rows.append({"method_id": "IG-" + row["budget_tag"],
                    "split": row["split"], "stage": row["stage"],
                    "scenario_id": row["scenario_id"], "scenario_sha256": row["scenario_sha256"],
                    "seed": "", "repeat": row["restart_seed"], "completed": row["completed"],
                    "final_cmax": row["cmax"], "checkpoint_type": "not_applicable",
                    "run_kind": "budgeted"})
        elif name == "scope_eval.csv":
            manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8-sig"))
            if manifest.get("scenario_index_sha256") != index_hash:
                raise ValueError(f"Stale scope replay: {path}")
            for row in _csv(path):
                if row["scope_variant"] != "default" or row["method_id"] != MAIN:
                    continue
                method = row["method_id"]
                stage = int(row["stage"])
                rows.append({"method_id": method, "split": row["split"],
                    "stage": stage, "scenario_id": row["scenario_id"],
                    "scenario_sha256": row["scenario_sha256"],
                    "seed": _best_seed_from_scope_manifest(path, method, stage),
                    "repeat": 1, "completed": row["completed"],
                    "final_cmax": row["final_cmax"],
                    "checkpoint_type": "best" if method == MAIN else "not_applicable"})
        elif name == "rule_replay.json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if payload.get("schema") != "six_rule_replay_v1":
                raise ValueError(f"Unknown rule replay schema: {path}")
            for method, attempts in payload["details"].items():
                for item in attempts:
                    scenario_path = Path(item["scenario_path"])
                    if not scenario_path.is_file():
                        raise ValueError(f"Rule replay source no longer exists: {scenario_path}")
                    sid = scenario_path.stem
                    candidates = [(key, row) for key, row in index.items() if key[2] == sid]
                    if len(candidates) != 1:
                        raise ValueError(f"Ambiguous rule scenario {sid}")
                    key, registered = candidates[0]
                    rows.append({"method_id": method, "split": key[0], "stage": key[1],
                        "scenario_id": sid, "scenario_sha256": sha256(scenario_path),
                        "seed": "", "repeat": 1, "completed": item["completed"],
                        "final_cmax": item["final_cmax"], "checkpoint_type": "not_applicable"})
        elif path.suffix.lower() == ".csv":
            rows = _csv(path)
            if name == "rule_scenario_results.csv":
                manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8-sig"))
                if (manifest.get("scenario_index_sha256") != index_hash or
                        manifest.get("scenario_csv_sha256") != sha256(path)):
                    raise ValueError(f"Stale rule scenario results: {path}")
            required = {"method_id", "split", "stage", "scenario_id", "scenario_sha256",
                        "seed", "completed", "final_cmax", "checkpoint_type"}
            if not rows or not required.issubset(rows[0]):
                raise ValueError(f"Canonical evaluation CSV lacks fields {required}: {path}")
        else:
            raise ValueError(f"Unsupported evaluation artifact: {path}")
        for raw in rows:
            method = str(raw["method_id"]).strip()
            if method not in MEASURED_METHODS:
                raise ValueError(f"Unexpected method {method!r}: {path}")
            role, stage, sid = str(raw["split"]).strip(), int(raw["stage"]), str(raw["scenario_id"]).strip()
            if role not in ("train", "val", "test"):
                raise ValueError(f"Statistics accept train/val/test only: {role}")
            registered = index.get((role, stage, sid))
            if registered is None or raw["scenario_sha256"] != registered["sha256"]:
                raise ValueError(f"Scenario hash/role/stage mismatch: {sid} from {path}")
            finished = as_bool(raw["completed"])
            cmax = raw.get("final_cmax")
            if finished:
                cmax = float(cmax)
                if not math.isfinite(cmax) or cmax < 0:
                    raise ValueError(f"Invalid complete Cmax: {path} / {sid}")
            elif cmax not in (None, "", "None", "null"):
                raise ValueError(f"Failed run has a Cmax: {path} / {sid}")
            else:
                cmax = None
            seed = raw.get("seed")
            seed = int(seed) if seed not in (None, "", "None") else None
            repeat = int(raw.get("repeat") or 1)
            if method in (MAIN, SINGLE):
                if seed is None or raw.get("checkpoint_type") != "best" or repeat != 1:
                    raise ValueError(f"Learned method requires seed and best checkpoint: {path} / {sid}")
            elif seed is not None:
                raise ValueError(f"Deterministic/search method must not use a training seed: {path}")
            if method in IG and raw.get("run_kind") != "budgeted":
                raise ValueError(f"IG statistics require a real budgeted run: {path} / {sid}")
            if repeat < 1 or (method not in IG and repeat != 1):
                raise ValueError(f"Invalid repeat: {path} / {sid}")
            records.append({"method_id": method, "split": role, "stage": stage,
                "scenario_id": sid, "scenario_sha256": registered["sha256"],
                "seed": seed, "repeat": repeat, "completed": finished,
                "final_cmax": cmax, "source": str(path)})
        sources.append({"path": str(path), "sha256": sha256(path), "row_count": len(rows)})
    seen = set()
    for row in records:
        key = (row["method_id"], row["split"], row["stage"], row["scenario_id"],
               row["seed"], row["repeat"])
        if key in seen:
            raise ValueError(f"Duplicate evaluation observation: {key}")
        seen.add(key)
    return records, sources


def select_deployable_rules(records: list[dict], index: dict):
    by_key = {(r["method_id"], r["split"], r["stage"], r["scenario_id"]): r
              for r in records if r["method_id"] in RULES}
    selected, diagnostics = {}, []
    for stage in (1, 2, 3):
        scenario_keys = [key for key in index if key[1] == stage and key[0] in ("train", "val")]
        expected_n = len(scenario_keys)
        means = {}
        for method in RULES:
            attempts = [by_key.get((method, role, st, sid)) for role, st, sid in scenario_keys]
            valid = [r for r in attempts if r and r["completed"]]
            mean = sum(r["final_cmax"] for r in valid) / len(valid) if expected_n > 0 and len(valid) == expected_n else None
            diagnostics.append({"stage": stage, "method_id": method,
                "train_val_n": len(valid), "expected_n": expected_n, "mean_cmax": mean})
            if mean is not None:
                means[method] = mean
        if len(means) == len(RULES):
            selected[stage] = min(RULES, key=lambda method: (means[method], RULES.index(method)))
    return selected, diagnostics


def assemble_test_panel(records: list[dict], index: dict, selected: dict, seeds: tuple[int, ...]):
    lookup = defaultdict(list)
    for row in records:
        if row["split"] == "test":
            lookup[(row["stage"], row["scenario_id"], row["method_id"])].append(row)
    panel, coverage = [], []
    for stage in (1, 2, 3):
        test_ids = sorted(sid for role, st, sid in index if role == "test" and st == stage)
        for method in MEASURED_METHODS:
            good = 0
            for sid in test_ids:
                attempts = lookup[(stage, sid, method)]
                expected = (set(seeds) if method in (MAIN, SINGLE) else
                            {1, 2, 3} if method in IG else {1})
                observed = ({r["seed"] for r in attempts} if method in (MAIN, SINGLE) else
                            {r["repeat"] for r in attempts})
                if len(attempts) == len(expected) and observed == expected and all(r["completed"] for r in attempts):
                    good += 1
            coverage.append({"stage": stage, "method_id": method, "complete_scenarios": good,
                "expected_scenarios": 12, "status": "complete" if good == 12 else "missing_or_failed"})
        for sid in test_ids:
            registered = index[("test", stage, sid)]
            row = {"stage": stage, "scenario_id": sid,
                "cohort": "T0" if registered["origin"] == "original" else "T-ID",
                "scenario_sha256": registered["sha256"]}
            for method in MEASURED_METHODS:
                attempts = lookup[(stage, sid, method)]
                expected = (set(seeds) if method in (MAIN, SINGLE) else
                            {1, 2, 3} if method in IG else {1})
                observed = ({r["seed"] for r in attempts} if method in (MAIN, SINGLE) else
                            {r["repeat"] for r in attempts})
                row[method] = (sum(r["final_cmax"] for r in attempts) / len(attempts)
                               if len(attempts) == len(expected) and observed == expected and
                               all(r["completed"] for r in attempts) else None)
            row[DEPLOYABLE] = row.get(selected.get(stage, "")) if stage in selected else None
            panel.append(row)
        coverage.append({"stage": stage, "method_id": DEPLOYABLE,
            "complete_scenarios": sum(row[DEPLOYABLE] is not None for row in panel if row["stage"] == stage),
            "expected_scenarios": 12,
            "status": "complete" if stage in selected and all(
                row[DEPLOYABLE] is not None for row in panel if row["stage"] == stage) else "missing_or_failed"})
    return panel, coverage
