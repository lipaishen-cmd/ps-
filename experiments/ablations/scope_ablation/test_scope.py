"""L7 regression: old-batch identity, scope boundaries and test split."""
import unittest
from types import SimpleNamespace

from experiments.ablations.scope_ablation.scope_metrics import event_perturbation
from experiments.ablations.scope_ablation.evaluate_scope import load_config, scope_parameters, select_test, build_manager
from data_models import Batch, MachineQueue, POOL_D


def solution(d1, d2=()):
    q1 = MachineQueue("D1", POOL_D, 6, batches=[Batch(o, uid, 1, "D1", start_time=start)
                                               for o, uid, start in d1])
    q2 = MachineQueue("D2", POOL_D, 6, batches=[Batch(o, uid, 1, "D2", start_time=start)
                                               for o, uid, start in d2])
    return SimpleNamespace(machine_queues={"D1": q1, "D2": q2})


class ScopeTests(unittest.TestCase):
    def test_new_insertion_shift_is_not_old_batch_move(self):
        before = solution([("A", 1, 5), ("B", 2, 10)])
        after = solution([("N", 3, 5), ("A", 1, 7), ("B", 2, 12)])
        result = event_perturbation(before, after, frozenset())
        self.assertEqual(result["moved_old_batches"], 0)
        self.assertEqual(result["cross_machine_migrations"], 0)
        self.assertEqual(result["mean_start_shift_min"], 2)

    def test_reordering_and_migration_count_once_per_old_batch(self):
        before = solution([("A", 1, 5), ("B", 2, 10), ("C", 3, 15)])
        reordered = solution([("B", 2, 10), ("A", 1, 5), ("C", 3, 15)])
        metrics = event_perturbation(before, reordered, frozenset())
        self.assertEqual(metrics["moved_old_batches"], 2)
        migrated = solution([("B", 2, 10), ("C", 3, 15)], [("A", 1, 5)])
        metrics = event_perturbation(before, migrated, frozenset())
        self.assertEqual(metrics["cross_machine_migrations"], 1)
        self.assertEqual(metrics["moved_old_batches"], 1)

    def test_resplit_replacement_is_explicit_not_imputed(self):
        before = solution([("A", 1, 5), ("B", 2, 10)])
        after = solution([("A", 1, 5), ("B", 4, 10), ("B", 5, 20)])
        metrics = event_perturbation(before, after, frozenset())
        self.assertEqual(metrics["moved_old_batches"], 1)
        self.assertEqual(metrics["replaced_old_batches"], 1)
        self.assertEqual(metrics["cross_machine_migrations"], 0)
        self.assertEqual(metrics["start_shift_observed_batches"], 1)

    def test_frozen_change_is_rejected(self):
        before = solution([("A", 1, 5), ("B", 2, 10)])
        after = solution([("A", 1, 6), ("B", 2, 10)])
        with self.assertRaises(ValueError):
            event_perturbation(before, after, frozenset({1}))

    def test_registered_split_and_scope_extremes(self):
        config, _ = load_config()
        for stage in (1, 2, 3):
            rows, _ = select_test(stage)
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(r["origin"] == "original" for r in rows), 2)
        item = select_test(1, ["scenario_test_000"])[0][0]
        zero = build_manager(item, "zero", config).trigger_next_event()
        self.assertEqual(len(zero.affected_uids), 0)
        full = build_manager(item, "full_tail", config).trigger_next_event()
        self.assertEqual(set(full.affected_uids), set(full.base_solution.get_all_active_uids()) - set(full.frozen_uids))
        self.assertEqual(scope_parameters("default", config), (8, 5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
