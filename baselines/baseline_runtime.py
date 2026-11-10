"""Shared, constraint-faithful runtime for the six approved rule baselines.

Every rule uses the active manifest, the same Pool-D event order, F/A/K
boundary, scope-expansion rule, transaction, physical decoder, and final
validator. No incomplete candidate is reported as a completed schedule.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_COMPARISON_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _COMPARISON_ROOT.parent
for _path in (
    _COMPARISON_ROOT,
    _COMPARISON_ROOT / "configs",
    _PROJECT_ROOT / "environment",
    _PROJECT_ROOT / "agent",
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from baseline_config import BaselineConfig
from data_models import DynamicInsertEvent, POOL_D, Solution
from decoder import full_decode
from dynamic_event_manager import DynamicEventManager, DynamicRescheduleContext
from dynamic_validator import load_dynamic_orders_and_events
from partition_template_generator import generate_partition_templates
from scope_selector import select_affected_domain
from solution_lifecycle import normalize_solution
from time_model import processing_time


RULE_NAMES = (
    "S_BASE",
    "FIFO",
    "SPT",
    "LPT",
    "EDD",
    "ECT",
)


@dataclass
class EventRecord:
    event_id: str
    event_time: float
    scope_k: int
    scope_ratio: float
    committed: bool
    action: Optional[str] = None
    action_index: Optional[int] = None
    reward: Optional[float] = None
    cmax_before: Optional[float] = None
    cmax_after: Optional[float] = None
    errors: Tuple[str, ...] = ()


@dataclass
class ScenarioResult:
    method: str
    scenario_path: str
    completed: bool
    final_cmax: Optional[float]
    elapsed_seconds: float
    event_count: int
    committed_events: int
    controlled_failures: int
    failure_reason: Optional[str]
    event_records: List[EventRecord]
    total_tardiness: Optional[float] = None
    priority_weighted_tardiness: Optional[float] = None
    tardy_order_count: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["scenario_path"] = os.path.abspath(self.scenario_path)
        return payload


def _build_manager(scenario_path: str, config: BaselineConfig) -> DynamicEventManager:
    initial_orders, events = load_dynamic_orders_and_events(scenario_path, config.sheet_name)
    return DynamicEventManager(
        pool_type=POOL_D,
        initial_orders=initial_orders,
        events=events,
        instance_name=config.sheet_name,
        k_scope=config.k_scope,
        k_seed=config.k_seed,
        delta_k=config.delta_k,
        formal_mode=config.formal_mode,
    )


def _unfrozen_old_count(context: DynamicRescheduleContext) -> int:
    return sum(
        1
        for queue in context.base_solution.machine_queues.values()
        for batch in queue.batches
        if batch.order_id not in context.new_order_ids and batch.u_id not in context.frozen_uids
    )


def _expand_scope(
    manager: DynamicEventManager,
    context: DynamicRescheduleContext,
    initial_k: int,
) -> Optional[DynamicRescheduleContext]:
    unfrozen = _unfrozen_old_count(context)
    if unfrozen <= 0:
        return None
    next_k = min(unfrozen, context.k_scope + context.delta_k)
    if next_k <= context.k_scope:
        return None
    scope = select_affected_domain(
        solution=context.base_solution,
        event=context.event,
        orders_dict=context.orders,
        machines_dict=manager.machines_dict,
        k_scope=next_k,
        k_seed=context.k_seed,
    )
    expanded = replace(
        context,
        k_scope=next_k,
        affected_uids=scope.affected_uids,
        stable_uids=scope.stable_uids,
        stable_signature=scope.stable_signature,
        scope_ratio=scope.scope_ratio,
    )
    manager.active_context = expanded
    return expanded


def _delivery_metrics(solution: Solution, orders: Dict[str, Any]) -> Tuple[float, float, int]:
    """Compute secondary due-date diagnostics without changing the Cmax objective."""
    completion: Dict[str, float] = {}
    for queue in solution.machine_queues.values():
        for batch in queue.batches:
            completion[batch.order_id] = max(
                completion.get(batch.order_id, 0.0),
                float(batch.finish_time),
            )
    priority_weight = {"高": 3.0, "中": 2.0, "普通": 1.0}
    total_tardiness = 0.0
    weighted_tardiness = 0.0
    tardy_count = 0
    for order_id, order in orders.items():
        raw_due = getattr(order, "due_date", None)
        if raw_due is None or order_id not in completion:
            continue
        tardiness = max(0.0, completion[order_id] - float(raw_due))
        total_tardiness += tardiness
        weighted_tardiness += priority_weight.get(str(getattr(order, "priority", "普通")), 1.0) * tardiness
        tardy_count += int(tardiness > 1e-9)
    return total_tardiness, weighted_tardiness, tardy_count


def _minimum_processing_time(manager: DynamicEventManager, context: DynamicRescheduleContext, batch) -> float:
    order = context.orders[batch.order_id]
    candidates = [
        processing_time(order, manager.machines_dict[machine_id])
        for machine_id in order.eligible_machines
        if machine_id in manager.machines_dict
        and manager.machines_dict[machine_id].pool_type == manager.pool_type
        and manager.machines_dict[machine_id].spindles >= batch.stack_count
    ]
    return min(candidates, default=float("inf"))


def _due_date(order, rule_name: str) -> float:
    """Return a finite due date and never turn missing EDD data into FIFO."""
    raw_due = getattr(order, "due_date", None)
    if raw_due is None:
        raise ValueError(f"规则 {rule_name} 要求订单 {order.order_id} 具有交期。")
    due = float(raw_due)
    if not math.isfinite(due) or due < 0.0:
        raise ValueError(f"规则 {rule_name} 收到订单 {order.order_id} 的非法交期 {raw_due!r}。")
    return due


def _rule_batch_key(
    rule_name: str,
    manager: DynamicEventManager,
    context: DynamicRescheduleContext,
    batch,
) -> Tuple[Any, ...]:
    order = context.orders[batch.order_id]
    release = float(getattr(order, "release_time", 0.0) or 0.0)
    stable_tail = (batch.order_id, -int(batch.stack_count), int(batch.u_id))
    if rule_name == "FIFO":
        return (release, *stable_tail)
    if rule_name == "SPT":
        return (_minimum_processing_time(manager, context, batch), release, *stable_tail)
    if rule_name == "LPT":
        return (-_minimum_processing_time(manager, context, batch), release, *stable_tail)
    if rule_name == "EDD":
        return (_due_date(order, rule_name), release, *stable_tail)
    if rule_name == "ECT":
        # ECT ranks machine-slot alternatives below and uses FIFO exposure order.
        return (release, *stable_tail)
    raise ValueError(f"未实现的规则 {rule_name!r}。")


def _rule_slot(
    rule_name: str,
    manager: DynamicEventManager,
    context: DynamicRescheduleContext,
    reallocation,
    batch,
    legal_slots: Sequence[Tuple[str, int]],
) -> Tuple[str, int]:
    """Rank legal slots with rule-specific keys and one common physical decoder."""
    if not legal_slots:
        raise ValueError(f"规则 {rule_name} 没有可用插槽。")
    base_cmax = float(reallocation.snapshot.makespan)
    rows: List[Tuple[Tuple[Any, ...], Tuple[str, int]]] = []
    for machine_id, slot_pos in legal_slots:
        trial = reallocation.snapshot.copy()
        trial_batch = copy.deepcopy(batch)
        trial_batch.machine_id = machine_id
        trial.machine_queues[machine_id].insert_at(int(slot_pos), trial_batch)
        try:
            normalize_solution(trial)
            full_decode(
                trial,
                context.orders,
                manager.machines_dict,
                event_time=context.event_time,
                frozen_uids=context.frozen_uids,
            )
        except Exception:
            continue
        placed = trial.active_batches.get(int(batch.u_id))
        if placed is None:
            continue
        common_tail = (
            float(trial.makespan) - base_cmax,
            float(trial.makespan),
            float(placed.finish_time),
            int(slot_pos),
            str(machine_id),
        )
        if rule_name == "ECT":
            score = (float(placed.finish_time), *common_tail)
        else:
            score = common_tail
        rows.append((score, (machine_id, int(slot_pos))))
    if not rows:
        raise ValueError(f"规则 {rule_name} 的合法插槽均无法通过统一解码。")
    return min(rows, key=lambda row: row[0])[1]


def _rule_template(rule_name: str, templates: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Use one capacity-efficient split convention for every pure rule."""
    if not templates:
        raise ValueError("规则调度没有合法拆批模板。")
    canonical = [tuple(int(part) for part in template) for template in templates]
    return min(canonical, key=lambda template: (len(template), -template[0], template))


def _build_rule_candidate(
    manager: DynamicEventManager,
    context: DynamicRescheduleContext,
    rule_name: str,
    max_templates: int,
) -> Tuple[Optional[Solution], Tuple[str, ...]]:
    """Build a pure dispatch-rule candidate without hidden repair or split bias."""
    if not isinstance(context.event, DynamicInsertEvent) or len(context.new_order_ids) != 1:
        return None, ("规则调度只接受当前正式合同中的单个动态插单事件。",)
    order = context.orders.get(context.new_order_ids[0])
    if order is None:
        return None, ("当前动态订单未出现在事件上下文中。",)
    templates = generate_partition_templates(
        order,
        max_templates=int(max_templates),
        machines=manager.machines_dict,
    )
    template = _rule_template(rule_name, templates)
    transaction = None
    try:
        transaction = manager.begin_reallocation(context=context, new_order=order, template=template)
        transaction.sub_batches.sort(
            key=lambda batch: _rule_batch_key(
                rule_name,
                manager,
                context,
                batch,
            )
        )
        while transaction.current_batch_idx < transaction.total_sub_batches:
            batch = transaction.get_current_batch()
            if batch is None:
                raise RuntimeError("规则事务的当前批次索引不一致。")
            legal_slots = transaction.get_valid_slots_for_batch(batch)
            machine_id, slot_pos = _rule_slot(
                rule_name,
                manager,
                context,
                transaction,
                batch,
                legal_slots,
            )
            if not transaction.step_assign(machine_id, slot_pos):
                raise RuntimeError(f"规则 {rule_name} 无法放置当前合法子批。")
        built, candidate, _ = transaction.commit()
        if not built:
            return None, ("规则候选未能通过统一解码。",)
        validator = manager.begin_transaction(context)
        errors = tuple(validator.validate(candidate))
        validator.rollback()
        return (candidate, ()) if not errors else (None, errors)
    except Exception as exc:
        return None, (str(exc),)
    finally:
        if transaction is not None and not transaction.is_committed:
            transaction.rollback()


def run_rule_scenario(
    scenario_path: str,
    rule_name: str,
    config: BaselineConfig,
) -> ScenarioResult:
    """Replay one approved rule using the shared event/transaction contract."""
    canonical_rule = str(rule_name).upper()
    if canonical_rule not in RULE_NAMES:
        raise ValueError(f"规则必须是 {RULE_NAMES} 之一，收到 {rule_name!r}。")

    started = time.perf_counter()
    manager = _build_manager(scenario_path, config)
    records: List[EventRecord] = []
    committed = 0
    failure: Optional[str] = None

    while manager.pending_event_ids:
        context = manager.trigger_next_event()
        initial_k = context.k_scope
        current_context = context
        committed_this_event = False
        errors: List[str] = []

        while True:
            try:
                if canonical_rule == "S_BASE":
                    candidate = manager.run_independent_baseline(
                        current_context,
                        max_templates=config.max_partition_templates,
                    )
                    current_context = manager.active_context
                    build_errors: Tuple[str, ...] = ()
                else:
                    candidate, build_errors = _build_rule_candidate(
                        manager,
                        current_context,
                        canonical_rule,
                        config.max_partition_templates,
                    )
                if candidate is not None:
                    transaction = manager.begin_transaction(current_context)
                    ok, solution, commit_errors = transaction.commit(candidate)
                    if ok:
                        records.append(
                            EventRecord(
                                event_id=current_context.event.event_id,
                                event_time=float(current_context.event_time),
                                scope_k=int(current_context.k_scope),
                                scope_ratio=float(current_context.scope_ratio),
                                committed=True,
                                action=canonical_rule,
                                cmax_before=float(current_context.base_solution.makespan),
                                cmax_after=float(solution.makespan),
                            )
                        )
                        committed += 1
                        committed_this_event = True
                        break
                    errors.extend(commit_errors)
                errors.extend(build_errors)
            except Exception as exc:
                errors.append(str(exc))

            expanded = _expand_scope(manager, current_context, initial_k)
            if expanded is None:
                break
            current_context = expanded

        if not committed_this_event:
            failure = "；".join(errors) if errors else "规则未能生成可提交的完整候选解。"
            records.append(
                EventRecord(
                    event_id=current_context.event.event_id,
                    event_time=float(current_context.event_time),
                    scope_k=int(current_context.k_scope),
                    scope_ratio=float(current_context.scope_ratio),
                    committed=False,
                    action=canonical_rule,
                    errors=tuple(errors),
                )
            )
            manager.rollback_active()
            break

    completed = failure is None and not manager.pending_event_ids
    delivery = _delivery_metrics(manager.current_solution, manager.known_orders) if completed else (None, None, None)
    return ScenarioResult(
        method=canonical_rule,
        scenario_path=scenario_path,
        completed=completed,
        final_cmax=float(manager.current_solution.makespan) if completed else None,
        elapsed_seconds=time.perf_counter() - started,
        event_count=len(manager.events),
        committed_events=committed,
        controlled_failures=0 if completed else 1,
        failure_reason=failure,
        event_records=records,
        total_tardiness=delivery[0],
        priority_weighted_tardiness=delivery[1],
        tardy_order_count=delivery[2],
    )
