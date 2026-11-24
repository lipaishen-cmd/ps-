"""Focused 事件级 IG regression tests; synthetic timing fixtures are not experiment data."""
import math
import random
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from baselines.iterated_greedy.event_ig import EventIG, IGConfig, _build_manager, run_ig_scenario
from baselines.iterated_greedy.run_ig import select_scenarios, summarize, load_main_times


def signature(solution):
    return [(mid, [(b.u_id, b.order_id, b.stack_count, b.machine_id, b.start_time, b.finish_time)
                   for b in q.batches]) for mid, q in solution.machine_queues.items()]


class EventIGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = IGConfig.load()
        cls.path = select_scenarios("test", 1, ["scenario_test_000"])[0]["path"]

    def fixture(self):
        manager = _build_manager(str(self.path), self.config.baseline())
        context = manager.trigger_next_event()
        initial = manager.run_independent_baseline(context)
        context = manager.active_context
        search = EventIG(manager, context, self.config, random.Random(17), time.perf_counter() + 30)
        return manager, context, initial, search

    def test_all_stages_include_original_and_new_by_role(self):
        for stage in (1, 2, 3):
            rows = select_scenarios("test", stage)
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(r["origin"] == "original" for r in rows), 2)
            self.assertEqual(len(select_scenarios("factor_test", stage)), 9)
        with self.assertRaises(ValueError):
            select_scenarios("test", 2, ["scenario_test_000"])

    def test_neighborhoods_preserve_demands_and_do_not_mutate_live_state(self):
        manager, context, initial, search = self.fixture()
        before = signature(manager.current_solution)
        initial_before = signature(initial)
        for kind in ("destroy_repair", "split", "relocate", "swap"):
            candidate = None
            for _ in range(12):
                candidate = search.neighbor(initial, kind)
                if candidate is not None:
                    break
            self.assertIsNotNone(candidate, kind)
            self.assertEqual(search.tx.validate(candidate), [], kind)
            for oid, order in context.orders.items():
                total = sum(b.stack_count for q in candidate.machine_queues.values()
                            for b in q.batches if b.order_id == oid)
                self.assertEqual(total, order.W, (kind, oid))
        self.assertEqual(signature(manager.current_solution), before)
        self.assertEqual(signature(initial), initial_before)

    def test_frozen_and_stable_batches_are_rejected_if_corrupted(self):
        _, context, initial, search = self.fixture()
        for protected in (context.frozen_uids, context.stable_uids):
            self.assertTrue(protected, "Fixture must exercise both F and K")
            corrupt = initial.copy()
            batch = next(b for q in corrupt.machine_queues.values() for b in q.batches if b.u_id in protected)
            batch.stack_count += 1
            self.assertTrue(search.tx.validate(corrupt))
        blocked = {b.order_id for q in initial.machine_queues.values() for b in q.batches
                   if b.u_id in search.protected}
        self.assertFalse(blocked & set(search.splittable_orders(initial)))

    def test_incumbent_is_complete_and_no_worse_than_sbase(self):
        _, _, initial, search = self.fixture()
        best, stats = search.search(initial, iteration_limit=4)
        self.assertLessEqual(best.makespan, initial.makespan)
        self.assertEqual(search.tx.validate(best), [])
        self.assertEqual(stats["iterations"], 4)
        self.assertEqual(stats["attempts"]["destroy_repair"] + stats["attempts"]["split"], 4)
        self.assertEqual(stats["attempts"]["relocate"], 4)
        self.assertEqual(stats["attempts"]["swap"], 4)

    def test_single_path_p_split_and_lightweight_local_search(self):
        _, _, initial, search = self.fixture()
        search.config = replace(self.config, p_split=0.0)
        _, stats_dr = search.search(initial, iteration_limit=3)
        self.assertEqual(stats_dr["attempts"]["destroy_repair"], 3)
        self.assertEqual(stats_dr["attempts"]["split"], 0)
        _, _, initial2, search2 = self.fixture()
        search2.config = replace(self.config, p_split=1.0)
        _, stats_sp = search2.search(initial2, iteration_limit=3)
        self.assertGreaterEqual(stats_sp["attempts"]["split"], 1)
        self.assertEqual(stats_sp["attempts"]["relocate"], 3)
        self.assertEqual(stats_sp["attempts"]["swap"], 3)

    def test_expired_budget_keeps_complete_incumbent_and_reports_overrun(self):
        result = run_ig_scenario(self.path, replace(self.config, fixed_budget_s=1e-9), "5s", 1)
        self.assertTrue(result["completed"])
        self.assertGreater(result["budget_misses"], 0)
        self.assertFalse(result["budget_met"])
        self.assertIsNone(result["strict_cmax"])
        self.assertLess(result["budget_met_rate"], 1.0)
        self.assertTrue(all(r["iterations"] == 0 and r["cmax"] == r["sbase_cmax"] for r in result["events"]))

    def test_10x_never_invents_missing_or_invalid_timing(self):
        with self.assertRaises(ValueError):
            load_main_times(None, [], self.config)
        with self.assertRaises(ValueError):
            run_ig_scenario(self.path, self.config, "10x", 1, {})
        manager = _build_manager(str(self.path), self.config.baseline())
        fixture_times = dict.fromkeys(manager.pending_event_ids, 0.0123)
        result = run_ig_scenario(self.path, self.config, "10x", 1, fixture_times, iteration_limit=0)
        self.assertTrue(result["completed"])
        self.assertTrue(all(math.isclose(r["budget_s"], 0.123) for r in result["events"]))
        fixture_times[manager.pending_event_ids[0]] = float("nan")
        with self.assertRaises(ValueError):
            run_ig_scenario(self.path, self.config, "10x", 1, fixture_times)

    def test_failed_initialization_does_not_report_partial_cmax(self):
        with patch("dynamic_event_manager.DynamicEventManager.run_independent_baseline", side_effect=ValueError("fixture")):
            result = run_ig_scenario(self.path, self.config, "5s", 1, iteration_limit=0)
        self.assertFalse(result["completed"])
        self.assertIsNone(result["cmax"])
        self.assertEqual(result["committed_events"], 0)

    def test_incomplete_restarts_are_not_success_only_averaged(self):
        rows = [{"scenario_id": "fixture", "budget_tag": "5s", "stage": 1, "split": "test",
                 "completed": i != 2, "cmax": 10 if i != 2 else None,
                 "total_decision_s": 1, "total_search_s": .5, "budget_met": True,
                 "budget_misses": 0, "run_kind": "smoke"} for i in range(3)]
        summary = summarize(rows)[0]
        self.assertIsNone(summary["cmax_mean"])
        self.assertEqual(summary["completed_restarts"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

