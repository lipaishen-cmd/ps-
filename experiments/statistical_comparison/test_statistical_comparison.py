"""Focused synthetic checks for pairing, leakage boundary and corrections."""
from __future__ import annotations

import unittest
import csv
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from .paired_inference import compare_main, friedman_nemenyi, holm_adjust
from .prepare_panel import (
    METHODS, RULES, assemble_test_panel, registered_scenarios, select_deployable_rules,
)


class StatisticalComparisonTests(unittest.TestCase):
    def test_holm_step_down_original_order(self):
        adjusted = holm_adjust([.01, .02, .005])
        self.assertEqual(adjusted, [.02, .02, .015])

    def test_all_ties_have_p_one(self):
        panel = [{"scenario_id": f"s{i}", **{method: 100.0 for method in METHODS}}
                 for i in range(36)]
        comparisons = compare_main(panel)
        self.assertEqual(len(comparisons), 10)
        self.assertTrue(all(r["win_tie_loss"] == "0/36/0" and r["p_holm"] == 1
                            for r in comparisons))
        omnibus, ranks = friedman_nemenyi(panel)
        self.assertEqual(omnibus["p_value"], 1)
        self.assertEqual(len(ranks), 11)

    def test_complete_panel_uses_36_scenarios(self):
        panel = [{"scenario_id": f"s{i}", **{
            method: 100.0 if method == "AD-HG-DAPPO" else 110.0 + METHODS.index(method)
            for method in METHODS}} for i in range(36)]
        comparisons = compare_main(panel)
        self.assertTrue(all(r["wins"] == 36 and r["p_holm"] < .05 for r in comparisons))
        omnibus, ranks = friedman_nemenyi(panel)
        self.assertGreater(omnibus["critical_difference"], 0)
        self.assertLess(omnibus["p_value"], .05)
        self.assertEqual(ranks[0]["average_rank"], 1)

    def test_best_rule_selection_never_reads_test(self):
        index, records = {}, []
        for role, count in (("train", 6), ("val", 2), ("test", 12)):
            for i in range(count):
                sid = f"{role}{i}"
                index[(role, 1, sid)] = {"role": role, "stage": 1, "scenario_id": sid}
                for method in RULES:
                    cmax = (1.0 if method == "FIFO" else 10.0) if role != "test" else (
                        1.0 if method == "ECT" else 10.0)
                    records.append({"method_id": method, "split": role, "stage": 1,
                                    "scenario_id": sid, "completed": True, "final_cmax": cmax})
        selected, diagnostics = select_deployable_rules(records, index)
        self.assertEqual(selected[1], "FIFO")
        self.assertEqual(len(diagnostics), 18)

    def test_incomplete_panel_has_no_p_value(self):
        with self.assertRaises(ValueError):
            compare_main([])

    def test_complete_synthetic_scenarios_do_not_multiply_seeds_or_restarts(self):
        index, records = {}, []
        for stage in (1, 2, 3):
            for role, count in (("train", 6), ("val", 2), ("test", 12)):
                for number in range(count):
                    sid = f"{role}_{stage}_{number}"
                    index[(role, stage, sid)] = {"origin": "original" if number < 2 else "L4",
                                                 "sha256": f"synthetic_{sid}"}
                    for method in RULES:
                        records.append({"method_id": method, "split": role, "stage": stage,
                            "scenario_id": sid, "seed": None, "repeat": 1, "completed": True,
                            "final_cmax": 120.0 if method == "SPT" else 130.0})
                    if role == "test":
                        for method, value in (("AD-HG-DAPPO", 100.0), ("AD-HG-SAPPO", 110.0)):
                            for seed in (1, 2, 3):
                                records.append({"method_id": method, "split": role,
                                    "stage": stage, "scenario_id": sid, "seed": seed,
                                    "repeat": 1, "completed": True, "final_cmax": value + seed})
                        for method in ("IG-10x", "IG-5s"):
                            for restart in (1, 2, 3):
                                records.append({"method_id": method, "split": role,
                                    "stage": stage, "scenario_id": sid, "seed": None,
                                    "repeat": restart, "completed": True,
                                    "final_cmax": 115.0 + restart})
        selected, _ = select_deployable_rules(records, index)
        self.assertEqual(selected, {1: "SPT", 2: "SPT", 3: "SPT"})
        panel, coverage = assemble_test_panel(records, index, selected, (1, 2, 3))
        self.assertEqual(len(panel), 36)
        self.assertTrue(all(row["status"] == "complete" for row in coverage))
        self.assertEqual(panel[0]["AD-HG-DAPPO"], 102.0)
        self.assertEqual(panel[0]["IG-5s"], 117.0)
        self.assertEqual(panel[0]["DEPLOYABLE_BEST_RULE"], 120.0)
        self.assertEqual(len(compare_main(panel)), 10)

    def test_synthetic_full_cli_writes_cd_only_after_complete_panel(self):
        from .run_statistical_comparison import main
        index, _ = registered_scenarios()
        rows = []
        for (role, stage, sid), item in index.items():
            if role not in ("train", "val", "test"):
                continue
            for method in RULES:
                rows.append({"method_id": method, "split": role, "stage": stage,
                    "scenario_id": sid, "scenario_sha256": item["sha256"],
                    "seed": "", "repeat": 1, "completed": True,
                    "final_cmax": 120 + RULES.index(method),
                    "checkpoint_type": "not_applicable", "run_kind": ""})
            if role == "test":
                for method, base in (("AD-HG-DAPPO", 100), ("AD-HG-SAPPO", 110)):
                    for seed in (1, 2, 3):
                        rows.append({"method_id": method, "split": role, "stage": stage,
                            "scenario_id": sid, "scenario_sha256": item["sha256"],
                            "seed": seed, "repeat": 1, "completed": True,
                            "final_cmax": base + seed,
                            "checkpoint_type": "best", "run_kind": ""})
                for method in ("IG-10x", "IG-5s"):
                    for repeat in (1, 2, 3):
                        rows.append({"method_id": method, "split": role, "stage": stage,
                            "scenario_id": sid, "scenario_sha256": item["sha256"],
                            "seed": "", "repeat": repeat, "completed": True,
                            "final_cmax": 115 + repeat,
                            "checkpoint_type": "not_applicable", "run_kind": "budgeted"})
        with tempfile.TemporaryDirectory(prefix="synthetic_stat_test_") as temp:
            source = Path(temp) / "synthetic_only.csv"
            output = Path(temp) / "output"
            with source.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with redirect_stdout(io.StringIO()):
                exit_code = main(["--input", str(source), "--output", str(output)])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["status"],
                             "COMPLETE_TEST_PANEL")
            self.assertTrue((output / "critical_difference.svg").is_file())
            self.assertTrue((output / "critical_difference.png").is_file())
            with (output / "paired_comparisons.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)


if __name__ == "__main__":
    unittest.main()
