"""Build the authoritative wide comparison CSV and 6B long-form CSV artifacts.

Re-evaluates AD-HG-DAPPO, AD-HG-SAPPO and the six dispatching rules on the
registered scenarios under identical constraints, storing:
1. result/综合对比/三阶段_调度规则_消融模型全量对比.csv (and .json)
2. 6B long-form CSVs in output_root:
   - scenarios.csv
   - test_eval.csv
   - val_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
for directory in (
    THIS_DIR,
    ROOT,
    ROOT / "agent",
    ROOT / "environment",
    ROOT / "baselines",
    ROOT / "experiments" / "ablations",
    ROOT / "experiments" / "ablations" / "single_actor",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch

from audit_anomalies import _load_trainer, load_l3_config
from baselines.baseline_config import BaselineConfig
from baselines.baseline_runtime import run_rule_scenario
from data_models import POOL_D
from dynamic_event_manager import DynamicEventManager
from dynamic_validator import load_dynamic_orders_and_events

_GROUP_UNFORECAST_MAP = {
    "G1_SMALL_LOW_LOAD": 0.10,
    "G2_MEDIUM_MIXED_LOAD": 0.20,
    "G3_LARGE_HIGH_LOAD": 0.35,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_scenario_metadata_records(include_l4: bool) -> List[Dict[str, Any]]:
    base = ROOT / "data" / "dynamic_scenarios"
    index = json.loads((base / "scenario_index.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((base / "dataset_manifest.json").read_text(encoding="utf-8-sig"))

    orig_meta: Dict[str, Dict[str, Any]] = {}
    for split_rows in manifest.get("splits", {}).values():
        for rec in split_rows:
            orig_meta[rec["scenario_id"]] = rec

    l4_meta: Dict[str, Dict[str, Any]] = {}
    l4_csv = ROOT / "data" / "l4_test_extension_v20261117" / "scenarios.csv"
    if l4_csv.is_file():
        with l4_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                l4_meta[row["scenario_id"]] = row

    items: List[Dict[str, Any]] = []
    for row in index["scenarios"]:
        is_original = row.get("origin") == "original"
        if not include_l4 and not is_original:
            continue
        sid = str(row["scenario_id"])
        role = str(row["role"])
        stage = int(row["stage"])
        path = (base / row["relative_path"]).resolve()
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"Scenario path or SHA-256 mismatch: {sid} ({path})")

        if role == "train":
            split_6b = "train"
        elif role == "val":
            split_6b = "val"
        elif role == "test":
            split_6b = "T0" if is_original else "T-ID"
        elif role == "factor_test":
            split_6b = "T-F"
        else:
            split_6b = role

        if sid in orig_meta:
            m = orig_meta[sid]
            cov = m.get("coverage_classes", [])
            arrival = str(cov[1]) if len(cov) >= 2 else ""
            load_layer = str(cov[2]) if len(cov) >= 3 else ""
            unforecast_ratio = _GROUP_UNFORECAST_MAP.get(str(m.get("group_id", "")), 0.10 * stage)
            id_or_ood = "ID"
            gen_seed = m.get("seed", "")
            cmax_ref = m.get("cmax_reference")
        elif sid in l4_meta:
            m = l4_meta[sid]
            arrival = str(m["arrival"])
            load_layer = str(m["load_layer"])
            unforecast_ratio = float(m["unforecast_probability"])
            id_or_ood = str(m["id_or_ood"])
            gen_seed = m.get("seed", "")
            cmax_ref = None
        else:
            arrival = ""
            load_layer = ""
            unforecast_ratio = None
            id_or_ood = "ID"
            gen_seed = ""
            cmax_ref = None

        items.append({
            "scenario_id": sid,
            "stage": stage,
            "role": role,
            "split": split_6b,
            "origin": row.get("origin", ""),
            "n_orders": int(row["initial_orders"]),
            "n_events": int(row["dynamic_events"]),
            "arrival": arrival,
            "unforecast_ratio": unforecast_ratio,
            "load_layer": load_layer,
            "id_or_ood": id_or_ood,
            "seed": gen_seed,
            "sha256": row["sha256"],
            "cmax_reference": cmax_ref,
            "path": path,
        })

    split_order = {"train": 0, "val": 1, "T0": 2, "T-ID": 3, "T-F": 4}
    items.sort(key=lambda r: (r["stage"], split_order.get(r["split"], 9), r["scenario_id"]))
    return items


def _eval_learning_model(
    trainer: Any,
    scenario_path: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    init_orders, events = load_dynamic_orders_and_events(str(scenario_path), config["sheet_name"])
    manager = DynamicEventManager(
        pool_type=POOL_D,
        initial_orders=init_orders,
        events=events,
        instance_name=config["sheet_name"],
        k_scope=int(config["k_scope"]),
        k_seed=int(config["k_seed"]),
        delta_k=int(config["delta_k"]),
        formal_mode=True,
    )
    with torch.inference_mode():
        decisions = trainer.collect_dynamic_episode(
            manager, deterministic=True, clear_buffer=False, record_rollout=False
        )
    total_events = len(manager.events)
    committed = sum(1 for d in decisions if d.committed)
    n_fail = sum(1 for d in decisions if not d.committed)
    completed = committed == total_events and not manager.pending_event_ids
    return {
        "completed": completed,
        "cmax": float(manager.current_solution.makespan) if completed else None,
        "accept_rate": committed / max(1, total_events),
        "n_fail_events": n_fail,
    }


def extract_val_curve_csv(
    config: Dict[str, Any],
    seeds: Sequence[int],
    out_path: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method_id in (config["main_method"], config["ablation_method"]):
        for stage in (1, 2, 3):
            for seed in seeds:
                log_path = ROOT / config["train_logs"][method_id].format(stage=stage, seed=seed)
                if not log_path.is_file():
                    continue
                payload = json.loads(log_path.read_text(encoding="utf-8"))
                for vh in payload.get("validation_history", []):
                    val_cmax = vh.get("mean_actor_cmax", vh.get("mean_cmax"))
                    feas = float(vh.get("autonomous_feasibility_rate", 0.0))
                    rows.append({
                        "stage": stage,
                        "method": method_id,
                        "seed": seed,
                        "episode": int(vh["episode"]),
                        "val_mean_cmax": float(val_cmax) if val_cmax is not None else None,
                        "val_feasible": feas,
                    })
    _write_csv(out_path, rows)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def build_authoritative_artifacts(
    config_path: Optional[str] = None,
    seeds: Sequence[int] = (1,),
    include_l4: bool = False,
    output_dir: Optional[str] = None,
    skip_missing_seeds: bool = False,
) -> Dict[str, Any]:
    """Build independent authoritative CSVs & 6B files per seed (seed_1, seed_2, seed_3)."""
    config, config_sha = load_l3_config(config_path)
    torch.set_num_threads(int(config["cpu_threads"]))
    scenarios = _load_scenario_metadata_records(include_l4=include_l4)

    by_stage: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in scenarios:
        by_stage[item["stage"]].append(item)

    rule_cfg = BaselineConfig(
        k_seed=int(config["k_seed"]),
        k_scope=int(config["k_scope"]),
        delta_k=int(config["delta_k"]),
        max_partition_templates=int(config["max_partition_templates"]),
        sheet_name=str(config["sheet_name"]),
        create_output_dirs=False,
    )

    # Deterministic 6-rule replay (seed-independent, run once and reused per seed)
    rule_cache: Dict[str, Dict[str, Optional[float]]] = {}
    rule_long_rows: List[Dict[str, Any]] = []
    for item in scenarios:
        sid = item["scenario_id"]
        stage = item["stage"]
        path = item["path"]
        rule_results: Dict[str, Optional[float]] = {}
        for rule in config["rules"]:
            res = run_rule_scenario(str(path), rule, rule_cfg)
            rule_results[rule] = float(res.final_cmax) if res.completed else None
            rule_long_rows.append({
                "scenario_id": sid,
                "stage": stage,
                "split": item["split"],
                "method": rule,
                "seed": "",
                "cmax": rule_results[rule],
                "feasible": bool(res.completed),
                "accept_rate": res.committed_events / max(1, res.event_count),
                "n_fail_events": res.event_count - res.committed_events,
            })
        rule_cache[sid] = rule_results

    # Select deployable best single rule per stage strictly from train + val (L10)
    deployable_rule_by_stage: Dict[int, str] = {}
    for stage in sorted(by_stage.keys()):
        train_val_sids = [
            it["scenario_id"] for it in by_stage[stage] if it["split"] in ("train", "val")
        ]
        rule_means: List[Tuple[float, int, str]] = []
        for idx, rule in enumerate(config["rules"]):
            vals = [rule_cache[sid][rule] for sid in train_val_sids]
            if vals and all(v is not None for v in vals):
                rule_means.append((statistics.mean(float(v) for v in vals), idx, rule))
        if not rule_means:
            raise RuntimeError(f"Stage {stage} has no rule completing all train+val scenarios.")
        rule_means.sort(key=lambda t: (t[0], t[1]))
        deployable_rule_by_stage[stage] = rule_means[0][2]

    scenarios_6b_rows = [
        {
            "scenario_id": it["scenario_id"],
            "stage": f"S{it['stage']}",
            "split": it["split"],
            "n_orders": it["n_orders"],
            "n_events": it["n_events"],
            "arrival": it["arrival"],
            "unforecast_ratio": it["unforecast_ratio"],
            "load_layer": it["load_layer"],
            "id_or_ood": it["id_or_ood"],
            "seed": it["seed"],
            "sha256": it["sha256"],
        }
        for it in scenarios
    ]

    per_seed_outputs: Dict[str, Any] = {}
    for seed in seeds:
        missing = [
            ROOT / config["checkpoints"][m].format(stage=s, seed=seed)
            for m in (config["main_method"], config["ablation_method"])
            for s in sorted(by_stage.keys())
            if not (ROOT / config["checkpoints"][m].format(stage=s, seed=seed)).is_file()
        ]
        if missing:
            if skip_missing_seeds:
                print(f"[Skip seed_{seed}] Missing checkpoints: {[str(p.relative_to(ROOT)) for p in missing]}")
                continue
            raise FileNotFoundError(f"Missing checkpoints for seed_{seed}: {[str(p) for p in missing]}")

        trainers: Dict[Tuple[str, int], Any] = {}
        ckpt_hashes: Dict[str, str] = {}
        for stage, stage_items in sorted(by_stage.items()):
            sample_orders, events = load_dynamic_orders_and_events(str(stage_items[0]["path"]), config["sheet_name"])
            all_orders = [*sample_orders, *(ev.order for ev in events)]
            for method_id in (config["main_method"], config["ablation_method"]):
                ckpt_path = (ROOT / config["checkpoints"][method_id].format(stage=stage, seed=seed)).resolve()
                trainers[(method_id, stage)] = _load_trainer(method_id, ckpt_path, all_orders)
                ckpt_hashes[f"{method_id}:S{stage}:seed{seed}"] = _sha256(ckpt_path)

        long_eval_rows: List[Dict[str, Any]] = list(rule_long_rows)
        wide_rows: List[Dict[str, Any]] = []
        for item in scenarios:
            sid = item["scenario_id"]
            stage = item["stage"]
            path = item["path"]
            rule_results = rule_cache[sid]

            method_vals: Dict[str, Optional[float]] = {}
            for method_id in (config["main_method"], config["ablation_method"]):
                res_m = _eval_learning_model(trainers[(method_id, stage)], path, config)
                long_eval_rows.append({
                    "scenario_id": sid,
                    "stage": stage,
                    "split": item["split"],
                    "method": method_id,
                    "seed": seed,
                    "cmax": res_m["cmax"],
                    "feasible": bool(res_m["completed"]),
                    "accept_rate": res_m["accept_rate"],
                    "n_fail_events": res_m["n_fail_events"],
                })
                method_vals[method_id] = float(res_m["cmax"]) if (res_m["completed"] and res_m["cmax"] is not None) else None

            prop_cmax = method_vals[config["main_method"]]
            abla_cmax = method_vals[config["ablation_method"]]
            valid_rules = [(float(rule_results[r]), idx, r) for idx, r in enumerate(config["rules"]) if rule_results[r] is not None]
            min_rule_cmax, _, min_rule_name = min(valid_rules, key=lambda t: (t[0], t[1]))
            dep_name = deployable_rule_by_stage[stage]
            dep_cmax = rule_results[dep_name]

            def _imp(ref: Optional[float]) -> Optional[float]:
                if prop_cmax is None or ref is None or ref == 0.0:
                    return None
                return (ref - prop_cmax) / ref * 100.0

            wide_rows.append({
                "scenario_id": sid,
                "stage": f"Stage {stage}",
                "stage_num": stage,
                "split": item["split"],
                "role": item["role"],
                "origin": item["origin"],
                "initial_orders": item["n_orders"],
                "dynamic_events": item["n_events"],
                "arrival": item["arrival"],
                "unforecast_ratio": item["unforecast_ratio"],
                "id_or_ood": item["id_or_ood"],
                "seed": seed,
                "proposed_cmax": prop_cmax,
                "ablation_cmax": abla_cmax,
                "deployable_rule_name": dep_name,
                "deployable_rule_cmax": dep_cmax,
                "min_rule_name": min_rule_name,
                "min_rule_cmax": min_rule_cmax,
                "imp_vs_deployable_pct": _imp(dep_cmax),
                "imp_vs_min_rule_pct": _imp(min_rule_cmax),
                "imp_vs_s_base_pct": _imp(rule_results.get("S_BASE")),
                "imp_vs_ablation_pct": _imp(abla_cmax),
                **{r: rule_results[r] for r in config["rules"]},
                "scenario_sha256": item["sha256"],
            })

        auth_csv = ROOT / config["authoritative_csv_path"].format(seed=seed)
        auth_json = ROOT / config["authoritative_json_path"].format(seed=seed)
        _write_csv(auth_csv, wide_rows)
        auth_json.parent.mkdir(parents=True, exist_ok=True)
        auth_json.write_text(
            json.dumps(
                {
                    "schema": "authoritative_comparison_v1",
                    "generated_at": datetime.now().isoformat(),
                    "config_sha256": config_sha,
                    "include_l4": include_l4,
                    "seed": seed,
                    "deployable_rule_by_stage": {f"Stage {k}": v for k, v in deployable_rule_by_stage.items()},
                    "checkpoint_sha256": ckpt_hashes,
                    "records": wide_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        seed_out_dir = (Path(output_dir) if output_dir else (ROOT / config["output_root"])) / f"seed_{seed}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(seed_out_dir / "scenarios.csv", scenarios_6b_rows)
        _write_csv(seed_out_dir / "test_eval.csv", long_eval_rows)
        val_curve_rows = extract_val_curve_csv(config, seeds=(seed,), out_path=seed_out_dir / "val_curve.csv")

        per_seed_outputs[f"seed_{seed}"] = {
            "authoritative_csv": str(auth_csv),
            "authoritative_json": str(auth_json),
            "scenarios_csv": str(seed_out_dir / "scenarios.csv"),
            "test_eval_csv": str(seed_out_dir / "test_eval.csv"),
            "val_curve_csv": str(seed_out_dir / "val_curve.csv"),
            "scenario_count": len(wide_rows),
            "val_curve_count": len(val_curve_rows),
        }

    return {
        "deployable_rule_by_stage": deployable_rule_by_stage,
        "seeds_processed": list(per_seed_outputs.keys()),
        "per_seed": per_seed_outputs,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds, e.g. 1 or 1,2,3")
    parser.add_argument("--include-l4", action="store_true", help="Include 57 L4 T-ID and T-F scenarios (all 87)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-missing-seeds", action="store_true", help="Skip seeds whose checkpoints do not exist yet")
    args = parser.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    summary = build_authoritative_artifacts(
        config_path=args.config,
        seeds=seeds,
        include_l4=bool(args.include_l4),
        output_dir=args.output,
        skip_missing_seeds=bool(args.skip_missing_seeds),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
