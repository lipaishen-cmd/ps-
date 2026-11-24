"""The 事件级 IG figure must use per-event time and test-set relative Cmax."""
import unittest

from baselines.iterated_greedy.plot_quality_time import quality_time_points


def fixture():
    refs, main, ig = [], [], []
    for i in range(2):
        sid, digest = f"scenario_{i}", f"hash_{i}"
        refs.append({"scenario_id": sid, "stage": "1", "rule_id": "SPT",
                     "scenario_sha256": digest, "reference_cmax": "100",
                     "selection_split": "train_val"})
        main.append({"scenario_id": sid, "stage": "1", "split": "test",
                     "completed": "True", "scenario_sha256": digest, "cmax": "90",
                     "event_count": "2", "total_decision_s": "2"})
        for tag, cmax, duration in (("10x", "85", "4"), ("5s", "80", "6")):
            for restart in range(1, 4):
                ig.append({"scenario_id": sid, "stage": "1", "split": "test",
                           "run_kind": "budgeted", "completed": "True", "budget_tag": tag,
                           "restart": str(restart), "scenario_sha256": digest,
                           "cmax": cmax, "total_decision_s": duration,
                           "event_count": "2", "budget_misses": "0"})
    return ig, main, refs


class QualityTimeTests(unittest.TestCase):
    def test_relative_quality_and_per_event_time(self):
        points = quality_time_points(*fixture(), expected_scenarios=2)
        self.assertEqual([p["mean_relative_improvement_pct"] for p in points], [10, 15, 20])
        self.assertEqual([p["mean_event_decision_s"] for p in points], [1, 2, 3])
        self.assertEqual([p["event_budget_met_rate"] for p in points], [1.0, 1.0, 1.0])

    def test_timed_out_ig_runs_are_excluded_from_strict_valid_solutions(self):
        ig, main, refs = fixture()
        # Scenario 0 fails 10x budget on 1 event; only Scenario 1 (with budget_misses=0) counts as strictly valid.
        for row in ig:
            if row["scenario_id"] == "scenario_0" and row["budget_tag"] == "10x":
                row["budget_misses"] = "1"
                row["budget_met"] = "False"
                row["cmax"] = "50"  # Even if raw Cmax were lower, it must NOT count as valid within 10x!
        points = quality_time_points(ig, main, refs, expected_scenarios=2)
        ig10 = next(p for p in points if p["method"] == "IG-10x")
        self.assertEqual(ig10["strict_compliant_scenarios"], 1)
        self.assertAlmostEqual(ig10["scenario_budget_met_rate"], 0.5)
        self.assertAlmostEqual(ig10["event_budget_met_rate"], 9 / 12)
        self.assertAlmostEqual(ig10["mean_relative_improvement_pct"], 15.0)
        self.assertGreater(ig10["raw_mean_relative_improvement_pct"], 15.0)

    def test_test_selected_rule_and_missing_restart_are_rejected(self):
        ig, main, refs = fixture()
        refs[1]["rule_id"] = "FIFO"
        with self.assertRaises(ValueError):
            quality_time_points(ig, main, refs, expected_scenarios=2)
        ig, main, refs = fixture()
        ig.pop()
        with self.assertRaises(ValueError):
            quality_time_points(ig, main, refs, expected_scenarios=2)

    def test_hash_and_event_coverage_are_required(self):
        ig, main, refs = fixture()
        main[0]["scenario_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            quality_time_points(ig, main, refs, expected_scenarios=2)
        ig, main, refs = fixture()
        main[0]["event_count"] = "3"
        with self.assertRaises(ValueError):
            quality_time_points(ig, main, refs, expected_scenarios=2)

    def test_smoke_or_incomplete_runs_are_rejected(self):
        ig, main, refs = fixture()
        ig[0]["run_kind"] = "smoke"
        with self.assertRaises(ValueError):
            quality_time_points(ig, main, refs, expected_scenarios=2)
        ig, main, refs = fixture()
        ig[0]["completed"] = "False"
        with self.assertRaises(ValueError):
            quality_time_points(ig, main, refs, expected_scenarios=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

