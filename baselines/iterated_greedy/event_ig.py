"""Event-level IG using the project's physical objects and atomic validator.

Search never changes the live schedule. Only a complete, validated best incumbent
is committed. F and K are excluded from destruction and all three neighborhoods.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "baselines", ROOT / "environment", ROOT / "agent"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from baseline_config import BaselineConfig
from baseline_runtime import _build_manager, _delivery_metrics
from data_models import Batch, allocate_uid
from decoder import full_decode
from partition_template_generator import generate_partition_templates
from solution_lifecycle import normalize_solution


@dataclass(frozen=True)
class IGConfig:
    schema: str = "pcb_event_ig_v1"
    restart_seeds: tuple = (1, 2, 3)
    budgets: tuple = ("10x", "5s")
    fixed_budget_s: float = 5.0
    main_time_multiplier: float = 10.0
    temperature_ratio: float = 0.01
    destroy_count: int = 2
    p_split: float = 0.2
    max_partition_templates: int = 16
    commit_reserve_s: float = 0.01
    k_seed: int = 5
    k_scope: int = 8
    delta_k: int = 4
    sheet_name: str = "Sheet1"
    cpu_threads: int = 1

    def __post_init__(self):
        if len(self.restart_seeds) != 3 or len(set(self.restart_seeds)) != 3:
            raise ValueError("事件级 IG requires three distinct restart seeds.")
        if not self.budgets or any(b not in {"10x", "5s"} for b in self.budgets):
            raise ValueError("Budget tags must be 10x or 5s.")
        for name in ("fixed_budget_s", "main_time_multiplier", "temperature_ratio"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not math.isfinite(self.p_split) or not 0.0 <= self.p_split <= 1.0:
            raise ValueError("p_split must lie in [0, 1].")
        if (not math.isfinite(self.commit_reserve_s) or self.commit_reserve_s < 0
                or self.destroy_count < 1 or self.cpu_threads < 1
                or not 1 <= self.max_partition_templates <= 16):
            raise ValueError("Invalid destruction, template, CPU or reserve setting.")

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path or ROOT / "configs/ig_config.json").read_text(encoding="utf-8-sig")))

    def baseline(self):
        return BaselineConfig(k_seed=self.k_seed, k_scope=self.k_scope, delta_k=self.delta_k,
                              max_partition_templates=self.max_partition_templates,
                              sheet_name=self.sheet_name, create_output_dirs=False)

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class SearchDeadline(Exception):
    """Discard an unfinished candidate and retain the last complete incumbent."""


class EventIG:
    def __init__(self, manager, context, config, rng, deadline):
        self.manager, self.context, self.config = manager, context, config
        self.rng, self.deadline = rng, deadline
        self.protected = set(context.frozen_uids) | set(context.stable_uids)
        self.tx = manager.begin_transaction(context)
        self.evaluations = 0
        self.invalid = 0
        self.attempts = {"destroy_repair": 0, "split": 0, "relocate": 0, "swap": 0}
        self.valid_neighbors = dict.fromkeys(self.attempts, 0)

    def check_time(self):
        if time.perf_counter() >= self.deadline:
            raise SearchDeadline

    def movable(self, solution):
        return [b for q in solution.machine_queues.values() for b in q.batches
                if b.u_id not in self.protected]

    def splittable_orders(self, solution):
        batches = [b for q in solution.machine_queues.values() for b in q.batches]
        blocked = {b.order_id for b in batches if b.u_id in self.protected}
        return sorted({b.order_id for b in batches} - blocked)

    def decode(self, candidate):
        self.check_time()
        normalize_solution(candidate)
        full_decode(candidate, self.context.orders, self.manager.machines_dict,
                    event_time=self.context.event_time, frozen_uids=self.context.frozen_uids)
        self.evaluations += 1
        self.check_time()
        return candidate

    def validate(self, candidate):
        self.check_time()
        errors = self.tx.validate(candidate)
        self.check_time()
        if errors:
            self.invalid += 1
            return False
        return True

    def slots(self, solution, batch):
        order = self.context.orders[batch.order_id]
        for mid in sorted(order.eligible_machines):
            if mid not in solution.machine_queues:
                continue
            machine = self.manager.machines_dict[mid]
            if machine.pool_type != self.manager.pool_type or batch.stack_count > machine.spindles:
                continue
            queue = solution.machine_queues[mid]
            first = max((i + 1 for i, b in enumerate(queue.batches)
                         if b.u_id in self.context.frozen_uids), default=0)
            for pos in range(first, len(queue.batches) + 1):
                yield mid, pos

    @staticmethod
    def remove(solution, batches):
        ids = {b.u_id for b in batches}
        for queue in solution.machine_queues.values():
            queue.batches[:] = [b for b in queue.batches if b.u_id not in ids]
        normalize_solution(solution)

    def repair(self, partial, removed):
        """Greedy insertion with full physical decoding, ties fixed by machine/slot."""
        for batch in removed:
            best, best_key = None, None
            for mid, pos in self.slots(partial, batch):
                self.check_time()
                trial = partial.copy()
                inserted = copy.copy(batch)
                inserted.machine_id = mid
                trial.machine_queues[mid].insert_at(pos, inserted)
                self.decode(trial)
                key = (float(trial.makespan), inserted.finish_time, mid, pos)
                if best_key is None or key < best_key:
                    best, best_key = trial, key
            if best is None:
                return None
            partial = best
        return partial

    def neighbor(self, solution, kind):
        self.check_time()
        self.attempts[kind] += 1
        movable = self.movable(solution)
        if not movable:
            return None
        trial = solution.copy()
        if kind == "destroy_repair":
            removed = self.rng.sample(movable, min(len(movable), self.config.destroy_count))
            self.remove(trial, removed)
            trial = self.repair(trial, removed)
        elif kind == "split":
            choices = self.splittable_orders(solution)
            if not choices:
                return None
            oid = self.rng.choice(choices)
            old = [b for b in movable if b.order_id == oid]
            templates = generate_partition_templates(self.context.orders[oid],
                max_templates=self.config.max_partition_templates, machines=self.manager.machines_dict)
            current = sorted(b.stack_count for b in old)
            templates = [t for t in templates if sorted(t) != current]
            if not templates:
                return None
            template = self.rng.choice(templates)
            removed = [Batch(oid, allocate_uid(), int(count), "") for count in template]
            self.remove(trial, old)
            trial = self.repair(trial, removed)
        elif kind == "relocate":
            batch = self.rng.choice(movable)
            self.remove(trial, [batch])
            choices = list(self.slots(trial, batch))
            if not choices:
                return None
            mid, pos = self.rng.choice(choices)
            inserted = copy.copy(batch)
            inserted.machine_id = mid
            trial.machine_queues[mid].insert_at(pos, inserted)
            self.decode(trial)
        elif kind == "swap":
            choices = [(mid, [i for i, b in enumerate(q.batches) if b.u_id not in self.protected])
                       for mid, q in trial.machine_queues.items()]
            choices = [(mid, indices) for mid, indices in choices if len(indices) >= 2]
            if not choices:
                return None
            mid, indices = self.rng.choice(choices)
            i, j = self.rng.sample(indices, 2)
            batches = trial.machine_queues[mid].batches
            batches[i], batches[j] = batches[j], batches[i]
            self.decode(trial)
        else:
            raise ValueError(kind)
        if trial is not None and self.validate(trial):
            self.valid_neighbors[kind] += 1
            return trial
        return None

    def search(self, initial, iteration_limit=None):
        """Standard single-path IG: probabilistic split vs d-batch destroy, then relocate & swap."""
        current, best = initial, initial
        temperature = max(1e-12, self.config.temperature_ratio * float(initial.makespan))
        iterations = accepted = uphill = 0
        try:
            while iteration_limit is None or iterations < iteration_limit:
                self.check_time()
                if self.rng.random() < self.config.p_split:
                    candidate = self.neighbor(current, "split")
                    if candidate is None:
                        candidate = self.neighbor(current, "destroy_repair")
                else:
                    candidate = self.neighbor(current, "destroy_repair")
                if candidate is None:
                    break
                if candidate.makespan < best.makespan:
                    best = candidate
                # Lightweight local search over relocate and swap only.
                kinds = ["relocate", "swap"]
                self.rng.shuffle(kinds)
                for kind in kinds:
                    local = self.neighbor(candidate, kind)
                    if local is not None and local.makespan < candidate.makespan:
                        candidate = local
                    # Keep every fully validated improvement even if a later move times out.
                    if candidate.makespan < best.makespan:
                        best = candidate
                delta = float(candidate.makespan - current.makespan)
                if delta <= 0 or self.rng.random() < math.exp(-delta / temperature):
                    current = candidate
                    accepted += 1
                    uphill += int(delta > 0)
                if candidate.makespan < best.makespan:
                    best = candidate
                iterations += 1
        except SearchDeadline:
            pass
        return best, {"iterations": iterations, "accepted": accepted, "uphill_accepted": uphill,
                      "temperature": temperature, "evaluations": self.evaluations,
                      "invalid_candidates": self.invalid, "attempts": self.attempts,
                      "valid_neighbors": self.valid_neighbors}


def run_ig_scenario(scenario_path, config, budget_tag, restart_seed, main_event_seconds=None,
                    iteration_limit=None):
    """Replay one independent restart; budgets apply to each entire event decision.

    Atomic shared operations cannot be preempted: record any actual deadline miss.
    iteration_limit is exclusively a smoke-test control and marks results accordingly.
    """
    if budget_tag not in {"10x", "5s"}:
        raise ValueError(budget_tag)
    if iteration_limit is not None and iteration_limit < 0:
        raise ValueError("iteration_limit must be nonnegative")
    started = time.perf_counter()
    manager = _build_manager(str(scenario_path), config.baseline())
    setup_s = time.perf_counter() - started
    if budget_tag == "10x":
        if not main_event_seconds or any(e not in main_event_seconds for e in manager.pending_event_ids):
            raise ValueError("10x requires measured main-method time for EVERY scenario event.")
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in main_event_seconds.values()):
            raise ValueError("Main-method event times must be positive and finite.")
    records, failure = [], None
    while manager.pending_event_ids:
        event_id = manager.pending_event_ids[0]
        budget = (config.fixed_budget_s if budget_tag == "5s" else
                  config.main_time_multiplier * float(main_event_seconds[event_id]))
        event_start = time.perf_counter()
        record = {"event_id": event_id, "budget_s": budget, "committed": False,
                  "initialization_s": 0.0, "search_s": 0.0, "validation_commit_s": 0.0}
        try:
            context = manager.trigger_next_event()
            record["trigger_scope_s"] = time.perf_counter() - event_start
            init_start = time.perf_counter()
            initial = manager.run_independent_baseline(context, max_templates=config.max_partition_templates)
            context = manager.active_context
            record["initialization_s"] = time.perf_counter() - init_start
            record.update(event_time=context.event_time, scope_k=context.k_scope,
                          scope_ratio=context.scope_ratio, sbase_cmax=float(initial.makespan))
            seed = int.from_bytes(hashlib.sha256(
                f"{Path(scenario_path).stem}|{restart_seed}|{event_id}".encode()).digest()[:8], "big")
            search_start = time.perf_counter()
            searcher = EventIG(manager, context, config, random.Random(seed),
                              event_start + budget - config.commit_reserve_s)
            best, statistics = searcher.search(initial, iteration_limit)
            record["search_s"] = time.perf_counter() - search_start
            record.update(statistics)
            commit_start = time.perf_counter()
            committed, solution, errors = manager.begin_transaction(context).commit(best)
            record["validation_commit_s"] = time.perf_counter() - commit_start
            record.update(committed=bool(committed), cmax=float(solution.makespan) if committed else None,
                          errors=errors)
            if not committed:
                failure = "; ".join(map(str, errors))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            record["errors"] = [failure]
        record["decision_s"] = time.perf_counter() - event_start
        record["budget_overrun_s"] = max(0.0, record["decision_s"] - budget)
        record["budget_met"] = record["decision_s"] <= budget
        records.append(record)
        if failure:
            manager.rollback_active()
            break
    completed = (failure is None and not manager.pending_event_ids and manager.active_context is None
                 and len(records) == len(manager.events) and all(r["committed"] for r in records))
    delivery = _delivery_metrics(manager.current_solution, manager.known_orders) if completed else (None, None, None)
    event_total = len(manager.events)
    budget_met_events = sum(bool(r.get("budget_met")) for r in records)
    budget_misses = sum(not bool(r.get("budget_met")) for r in records)
    budget_met = completed and budget_met_events == event_total
    cmax_val = float(manager.current_solution.makespan) if completed else None
    return {"scenario_id": Path(scenario_path).stem, "budget_tag": budget_tag,
            "restart_seed": restart_seed, "completed": completed,
            "cmax": cmax_val,
            "strict_cmax": cmax_val if budget_met else None,
            "total_search_s": sum(r["search_s"] for r in records),
            "total_decision_s": sum(r["decision_s"] for r in records),
            "setup_s": setup_s, "elapsed_s": time.perf_counter() - started,
            "event_count": event_total, "committed_events": sum(r["committed"] for r in records),
            "budget_met": budget_met,
            "budget_met_events": budget_met_events,
            "budget_met_rate": budget_met_events / event_total if event_total else 0.0,
            "budget_misses": budget_misses,
            "failure_reason": failure, "run_kind": "smoke" if iteration_limit is not None else "budgeted",
            "iteration_limit": iteration_limit, "config_sha256": config.digest(),
            "total_tardiness": delivery[0], "priority_weighted_tardiness": delivery[1],
            "tardy_order_count": delivery[2], "events": records}

