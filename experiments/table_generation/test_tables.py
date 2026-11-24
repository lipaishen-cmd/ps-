"""Unit and regression tests for L3 anomaly audit and single-CSV table generation."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
for p in (str(THIS_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from audit_anomalies import audit_checkpoints_and_logs, load_l3_config
from build_authoritative_csv import extract_val_curve_csv
from generate_paper_tables import (
    _compute_val_convergence,
    generate_all_tables_from_csv,
    win_tie_loss,
)

# Reference numbers from Table 23a / PDF Section 1.2 & 1.3 for deterministic regression check
_BASELINE_30_FIXTURE = [
    # stage_num, sid, split, orders, events, arrival, proposed, ablation, S_BASE, FIFO, SPT, LPT, EDD, ECT
    (1, "scenario_train_000", "train", 14, 5, "uniform", 176.62, 185.87, 225.3, 208.48, 221.7, 210.1, 221.7, 318.4),
    (1, "scenario_train_001", "train", 15, 5, "poisson", 299.05, 299.05, 327.74, 327.74, 327.74, 327.74, 327.74, 335.7),
    (1, "scenario_train_002", "train", 16, 6, "uniform", 282.21, 321.99, 282.21, 282.21, 282.21, 282.21, 282.21, 282.21),
    (1, "scenario_train_003", "train", 17, 6, "poisson", 250.47, 262.87, 263.13, 263.13, 263.13, 263.13, 263.13, 263.13),
    (1, "scenario_train_004", "train", 18, 7, "uniform", 295.31, 300.29, 295.31, 295.31, 295.31, 295.31, 333.6, 295.31),
    (1, "scenario_train_005", "train", 20, 7, "poisson", 321.01, 343.49, 344.1, 321.38, 344.1, 321.38, 344.1, 374.6),
    (1, "scenario_val_000", "val", 16, 6, "uniform", 281.56, 269.16, 297.04, 287.2, 324.5, 284.32, 297.04, 298.9),
    (1, "scenario_val_001", "val", 18, 7, "poisson", 253.00, 261.09, 295.68, 293.9, 272.5, 264.0, 257.28, 440.2),
    (1, "scenario_test_000", "T0", 17, 6, "uniform", 283.75, 283.75, 308.95, 308.95, 308.95, 308.95, 308.95, 320.4),
    (1, "scenario_test_001", "T0", 18, 7, "poisson", 341.89, 341.89, 362.85, 353.6, 341.81, 350.7, 356.0, 376.7),
    (2, "scenario_train_006", "train", 28, 10, "uniform", 1135.67, 1180.1, 1203.68, 1251.8, 1224.5, 1269.6, 1227.0, 1907.3),
    (2, "scenario_train_007", "train", 29, 10, "poisson", 1100.65, 1357.9, 1176.51, 1204.2, 1328.8, 1221.6, 1295.8, 3266.4),
    (2, "scenario_train_008", "train", 30, 11, "burst", 1164.00, 1248.6, 1200.69, 1395.9, 1383.5, 1268.3, 1315.4, 2251.7),
    (2, "scenario_train_009", "train", 32, 11, "uniform", 1313.85, 1357.2, 1374.2, 1397.1, 1376.8, 1341.86, 1383.5, 1931.8),
    (2, "scenario_train_010", "train", 34, 12, "poisson", 1411.26, 1372.2, 1566.1, 1513.11, 1644.3, 1552.4, 1605.0, 1954.0),
    (2, "scenario_train_011", "train", 36, 12, "burst", 1488.71, 1577.6, 1639.5, 1599.7, 1732.5, 1652.3, 1582.64, 2689.2),
    (2, "scenario_val_002", "val", 30, 11, "uniform", 1268.74, 1289.36, 1378.63, 1528.5, 1516.8, 1467.2, 1459.6, 2317.5),
    (2, "scenario_val_003", "val", 36, 13, "poisson", 1416.49, 1435.79, 1555.06, 1564.7, 1587.8, 1527.92, 1618.9, 3312.6),
    (2, "scenario_test_002", "T0", 32, 11, "uniform", 1425.37, 1237.74, 1247.65, 1339.3, 1350.4, 1333.5, 1359.3, 1712.8),
    (2, "scenario_test_003", "T0", 36, 13, "poisson", 1274.73, 1318.51, 1379.09, 1384.4, 1413.0, 1315.08, 1382.3, 1717.8),
    (3, "scenario_train_012", "train", 44, 16, "burst", 4377.19, 4359.43, 4557.7, 4577.3, 4569.1, 4488.61, 4663.4, 7057.2),
    (3, "scenario_train_013", "train", 45, 17, "poisson", 4510.24, 4624.55, 4866.0, 4828.0, 4709.57, 4764.5, 4858.9, 7560.2),
    (3, "scenario_train_014", "train", 46, 18, "burst", 4695.57, 4791.65, 4677.78, 4770.2, 4874.2, 4799.4, 4722.6, 10566.0),
    (3, "scenario_train_015", "train", 48, 18, "poisson", 4645.65, 4685.00, 4798.6, 4976.9, 4835.6, 4855.0, 4757.32, 6108.1),
    (3, "scenario_train_016", "train", 50, 19, "burst", 4854.72, 4837.39, 4886.3, 5043.5, 4908.5, 4818.22, 4837.3, 9134.3),
    (3, "scenario_train_017", "train", 52, 20, "poisson", 5051.11, 5187.31, 5200.44, 5263.1, 5398.4, 5319.0, 5387.0, 6416.4),
    (3, "scenario_val_004", "val", 46, 18, "burst", 4824.81, 4860.83, 5018.81, 5089.0, 5015.6, 4910.25, 5013.9, 9316.6),
    (3, "scenario_val_005", "val", 50, 20, "poisson", 4550.87, 4620.15, 4649.11, 4741.0, 4680.3, 4570.39, 4698.0, 6391.2),
    (3, "scenario_test_004", "T0", 48, 19, "burst", 4776.79, 4922.47, 5005.67, 4952.4, 5063.7, 5008.8, 4928.22, 9207.8),
    (3, "scenario_test_005", "T0", 52, 20, "poisson", 5025.09, 5093.70, 5135.32, 5249.7, 5321.7, 5088.34, 5138.4, 7842.7),
]


class L3TablesTests(unittest.TestCase):
    def test_tie_tolerance_and_section_12_13_numbers(self) -> None:
        self.assertEqual(win_tie_loss([341.89], [341.81], tie_tol_min=0.1), (0, 1, 0))
        self.assertEqual(win_tie_loss([283.75], [308.95], tie_tol_min=0.1), (1, 0, 0))
        self.assertEqual(win_tie_loss([1425.37], [1247.65], tie_tol_min=0.1), (0, 0, 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "auth.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                fieldnames = [
                    "scenario_id", "stage", "stage_num", "split", "initial_orders", "dynamic_events",
                    "arrival", "proposed_cmax", "ablation_cmax", "S_BASE", "FIFO", "SPT", "LPT", "EDD", "ECT",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for st, sid, sp, ord_n, ev_n, arr, prop, abla, sb, fi, spt, lpt, edd, ect in _BASELINE_30_FIXTURE:
                    writer.writerow({
                        "scenario_id": sid, "stage": f"Stage {st}", "stage_num": st, "split": sp,
                        "initial_orders": ord_n, "dynamic_events": ev_n, "arrival": arr,
                        "proposed_cmax": prop, "ablation_cmax": abla,
                        "S_BASE": sb, "FIFO": fi, "SPT": spt, "LPT": lpt, "EDD": edd, "ECT": ect,
                    })

            out_dir = tmp / "tables"
            inline = generate_all_tables_from_csv(csv_path=str(csv_path), output_dir=str(out_dir))

            # Exact PDF Section 1.2 & 1.3 assertions
            self.assertEqual(inline["deployable_rule_by_stage"], {"S1": "LPT", "S2": "S_BASE", "S3": "LPT"})
            self.assertEqual(inline["win_tie_loss"]["all_main_vs_s_base"], [26, 2, 2])
            self.assertEqual(inline["win_tie_loss"]["all_main_vs_min_rule"], [24, 3, 3])
            self.assertEqual(inline["win_tie_loss"]["s3_main_vs_sappo"], [8, 0, 2])
            self.assertEqual(inline["win_tie_loss"]["pooled_train_vs_min_rule"], [14, 2, 2])
            self.assertEqual(inline["win_tie_loss"]["pooled_val_vs_min_rule"], [6, 0, 0])
            self.assertEqual(inline["win_tie_loss"]["pooled_test_vs_min_rule"], [4, 1, 1])
            self.assertEqual(inline["win_tie_loss"]["pooled_test_vs_sappo"], [3, 2, 1])
            self.assertAlmostEqual(inline["pooled_test_per_case_mean_imp_vs_min_rule_pct"], 0.21, places=2)
            self.assertEqual(inline["per_stage_test_ratio_imp_vs_deployable_pct"], {"S1": 5.16, "S2": -2.79, "S3": 2.92})
            self.assertEqual(inline["worst_test_scenario"]["scenario_id"], "scenario_test_002")

            # Cross-table consistency: test_004/005 identical across Table 23a, 23b, 37, 39; no universal column
            for name in ("table_23b.md", "table_35.md", "table_36.md", "table_37.md"):
                content = (out_dir / name).read_text(encoding="utf-8")
                self.assertNotIn("通用单策略消融", content)
                self.assertNotIn("跨阶", content)
            for name in ("table_23b.csv", "table_35.csv", "table_36.csv", "table_37.csv"):
                with (out_dir / name).open("r", encoding="utf-8-sig") as f:
                    header_line = f.readline()
                self.assertNotIn("通用", header_line)
                self.assertNotIn("universal", header_line.lower())

            with (out_dir / "table_23a.csv").open("r", encoding="utf-8-sig") as f:
                r23a = {row["scenario_id"]: row for row in csv.DictReader(f)}
            with (out_dir / "table_23b.csv").open("r", encoding="utf-8-sig") as f:
                r23b = {row["scenario_id"]: row for row in csv.DictReader(f)}
            with (out_dir / "table_37.csv").open("r", encoding="utf-8-sig") as f:
                r37 = {row["scenario_id"]: row for row in csv.DictReader(f)}
            with (out_dir / "table_39.csv").open("r", encoding="utf-8-sig") as f:
                r39 = {row["scenario_id"]: row for row in csv.DictReader(f)}

            for sid, expected_dappo, expected_sappo in (
                ("scenario_test_004", "4776.79", "4922.47"),
                ("scenario_test_005", "5025.09", "5093.7"),
            ):
                self.assertEqual(float(r23a[sid]["AD-HG-DAPPO"]), float(expected_dappo))
                self.assertEqual(float(r23b[sid]["AD-HG-DAPPO"]), float(expected_dappo))
                self.assertEqual(float(r37[sid]["AD-HG-DAPPO"]), float(expected_dappo))
                self.assertEqual(float(r39[sid]["AD-HG-DAPPO"]), float(expected_dappo))
                self.assertEqual(float(r23a[sid]["AD-HG-SAPPO"]), float(expected_sappo))
                self.assertEqual(float(r23b[sid]["AD-HG-SAPPO"]), float(expected_sappo))
                self.assertEqual(float(r37[sid]["AD-HG-SAPPO"]), float(expected_sappo))
                self.assertEqual(float(r39[sid]["AD-HG-SAPPO"]), float(expected_sappo))

    def test_checkpoint_log_audit_and_val_curve_extraction(self) -> None:
        config, _ = load_l3_config()
        audit_rows = audit_checkpoints_and_logs(config, seed=1)
        self.assertEqual(len(audit_rows), 6)
        self.assertTrue(all(r["identity_matches"] and r["val_cmax_matches"] for r in audit_rows))

        with tempfile.TemporaryDirectory() as tmpdir:
            vc_path = Path(tmpdir) / "val_curve.csv"
            rows = extract_val_curve_csv(config, seeds=(1,), out_path=vc_path)
            self.assertGreater(len(rows), 0)
            conv = _compute_val_convergence(vc_path)
            self.assertIn((1, "AD-HG-DAPPO"), conv)
            self.assertIn((3, "AD-HG-SAPPO"), conv)
            self.assertIsNotNone(conv[(1, "AD-HG-DAPPO")]["episodes_to_within_1pct_final_mean"])
            self.assertIsNotNone(conv[(1, "AD-HG-DAPPO")]["normalized_val_auc_mean"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
