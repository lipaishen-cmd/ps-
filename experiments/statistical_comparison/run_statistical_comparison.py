"""Build a leakage-safe 36-scenario panel, then run paired nonparametric tests."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.statistical_comparison.prepare_panel import (
    METHODS, assemble_test_panel, load_records, registered_scenarios,
    select_deployable_rules, sha256,
)
from experiments.statistical_comparison.paired_inference import (
    compare_main, friedman_nemenyi, stage_descriptives,
)
from experiments.statistical_comparison.critical_difference import draw_cd

OUT_ROOT = ROOT / "result" / "统计检验" / "测试集配对比较"


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def seeds_from_text(value: str):
    seeds = tuple(int(token.strip()) for token in value.split(","))
    if not seeds or any(seed < 1 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be unique positive integers")
    if len(seeds) not in (1, 3):
        raise ValueError("Use the full three-seed protocol or the documented single-run fallback")
    return seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="Canonical evaluation CSV, scope_eval.csv, ig_runs.csv, or rule_replay.json; repeat")
    parser.add_argument("--seeds", default="1,2,3", help="Best-checkpoint seeds: default 1,2,3; 1 is single-run fallback")
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--output", help="New output directory; existing directory is rejected")
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie in (0,1)")
    seeds = seeds_from_text(args.seeds)
    index, index_hash = registered_scenarios()
    records, sources = load_records([Path(x) for x in args.input], index, index_hash)
    selected, selection_candidates = select_deployable_rules(records, index)
    panel, coverage = assemble_test_panel(records, index, selected, seeds)
    folder = (Path(args.output).resolve() if args.output else OUT_ROOT /
              ("comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    folder.mkdir(parents=True, exist_ok=False)

    write_csv(folder / "coverage.csv", coverage)
    write_csv(folder / "rule_selection_candidates.csv", selection_candidates)
    write_csv(folder / "selected_deployable_rules.csv", [
        {"stage": stage, "selected_rule": selected.get(stage), "selection_split": "train_val",
         "complete_train_val_scenarios": 8 if stage in selected else 0}
        for stage in (1, 2, 3)])
    write_csv(folder / "panel_diagnostic.csv", panel)
    scripts = [Path(__file__), Path(__file__).with_name("prepare_panel.py"),
               Path(__file__).with_name("paired_inference.py"),
               Path(__file__).with_name("critical_difference.py")]
    import scipy
    manifest = {"schema": "paired_test_comparison_v1", "scenario_index_sha256": index_hash,
                "sources": sources, "script_sha256": {p.name: sha256(p) for p in scripts},
                "scipy": scipy.__version__, "training_seeds": seeds,
                "seed_protocol": "three_seed" if len(seeds) == 3 else "single_run_fallback",
                "alpha": args.alpha, "methods": METHODS,
                "data_status": "PRE_FORMAL_ONLY_PROXY_DATA",
                "sample_unit": "one registered test scenario; seed/restart Cmax averaged within scenario",
                "test_cohorts": {"T0": 6, "T-ID": 30},
                "selected_rules": selected}
    ready = (len(selected) == 3 and len(panel) == 36 and
             Counter(r["cohort"] for r in panel) == {"T0": 6, "T-ID": 30} and
             all(row["status"] == "complete" for row in coverage))
    if not ready:
        manifest["status"] = "INCOMPLETE_INPUT_NO_INFERENCE"
        write_json(folder / "run_manifest.json", manifest)
        print(f"Incomplete input: coverage only; no p-values or CD figure. {folder}")
        return 2

    comparisons = compare_main(panel, alpha=args.alpha)
    friedman, ranks = friedman_nemenyi(panel, alpha=args.alpha)
    write_csv(folder / "paired_comparisons.csv", comparisons)
    write_csv(folder / "stage_descriptive.csv", stage_descriptives(panel))
    write_csv(folder / "average_ranks.csv", ranks)
    write_json(folder / "friedman_nemenyi.json", friedman)
    draw_cd(ranks, friedman, folder / "critical_difference.svg")
    draw_cd(ranks, friedman, folder / "critical_difference.png")
    manifest["status"] = "COMPLETE_TEST_PANEL"
    manifest["wilcoxon"] = {"alternative": "two-sided", "zero_method": "wilcox",
                            "difference_round_decimals": 9,
                            "family": "main_vs_10_opponents", "adjustment": "Holm"}
    manifest["friedman_posthoc"] = "Nemenyi CD (studentized range / sqrt(2))"
    write_json(folder / "run_manifest.json", manifest)
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
