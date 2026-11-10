"""
dynamic_two_agent_runner.py
===========================
把两个 Actor 接入 :mod:`dynamic_event_manager` 的在线决策适配器。

Agent1 (ManagerPolicy) 选择拆分模板，Agent2 (WorkerPolicy) 为模板中的每个
子批选择机器/插槽。策略输出只产生候选离散结构；统一解码器计算开工/完工时刻，
随后由 RescheduleTransaction 校验冻结边界和五大物理约束。
"""

from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "environment"))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_THIS_DIR, _ENVIRONMENT_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from data_models import Batch, DynamicInsertEvent, Order, Solution  # noqa: E402
from decoder import decode_machine  # noqa: E402
from dynamic_event_manager import (  # noqa: E402
    DynamicEventManager,
    DynamicRescheduleContext,
)
from graph_builder import PCBGraphBuilder  # noqa: E402
from manager_policy import ManagerPolicy  # noqa: E402
from worker_policy import WorkerPolicy  # noqa: E402
from batch_hgnn import PCBBatchHGNN  # noqa: E402
from critics import SharedValueCritic  # noqa: E402
from partition_template_generator import generate_partition_templates  # noqa: E402
from time_model import processing_time, setup_time  # noqa: E402
from reward import compute_macro_event_reward, compute_micro_step_reward  # noqa: E402


@dataclass
class DynamicDecision:
    """一次事件的双策略决策、候选排程及确定性可行性验证结果。"""

    event_id: str
    committed: bool
    order_id: Optional[str] = None
    template: Optional[Tuple[int, ...]] = None
    actor_candidate_solution: Optional[Solution] = None
    candidate_solution: Optional[Solution] = None
    errors: List[str] = field(default_factory=list)
    worker_actions: List[Tuple[str, int]] = field(default_factory=list)
    manager_log_prob: Optional[float] = None
    worker_log_probs: List[float] = field(default_factory=list)
    manager_g: Optional[torch.Tensor] = None
    manager_h_order: Optional[torch.Tensor] = None
    manager_templates: List[Tuple[int, ...]] = field(default_factory=list)
    manager_mask: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    manager_action_idx: Optional[int] = None
    manager_value: Optional[torch.Tensor] = None
    manager_records: List[dict] = field(default_factory=list)
    partition_plan: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    worker_records: List[dict] = field(default_factory=list)
    reward: float = 0.0
    cmax_before: float = 0.0
    cmax_after: Optional[float] = None
    affected_uids: List[int] = field(default_factory=list)
    response_time_seconds: float = 0.0
    initial_k_scope: int = 8
    effective_k_scope: int = 8
    expansion_count: int = 0
    scope_ratio: float = 0.0

    @property
    def final_solution(self) -> Optional[Solution]:
        return self.candidate_solution if self.candidate_solution is not None else self.actor_candidate_solution

    @property
    def trainable(self) -> bool:
        """只有 Actor 自主方案通过物理约束并原子提交时才写入正式 PPO 经验池。"""
        return self.committed and bool(self.worker_records)

    @property
    def raw_actor_feasibility(self) -> float:
        """Actor 自主方案可行性：1.0 表示物理合法并被采纳，0.0 表示非法被拒。"""
        return 1.0 if self.committed else 0.0

    @property
    def actor_cmax(self) -> Optional[float]:
        return float(self.candidate_solution.makespan) if self.candidate_solution is not None else (
            float(self.actor_candidate_solution.makespan) if self.actor_candidate_solution is not None else None
        )

    @property
    def raw_actor_makespan(self) -> Optional[float]:
        return self.actor_cmax

    @property
    def delivered_makespan(self) -> Optional[float]:
        return self.actor_cmax

    @property
    def adopted_source(self) -> str:
        return "actor" if self.committed else "none"


class DynamicTwoAgentRunner:
    """在线动态事件运行器；两个 Actor 决策，EventManager 执行确定性约束验证。"""

    def __init__(
        self,
        manager: DynamicEventManager,
        hgnn: PCBBatchHGNN,
        manager_policy: ManagerPolicy,
        worker_policy: WorkerPolicy,
        value_critic: Optional[SharedValueCritic] = None,
        device: torch.device = torch.device("cpu"),
        max_templates: int = 16,
        reward_mode: str = "autonomous_incremental",
        reward_scale: float = 10.0,
    ):
        self.manager = manager
        self.hgnn = hgnn.to(device)
        self.manager_policy = manager_policy.to(device)
        self.worker_policy = worker_policy.to(device)
        self.value_critic = None if value_critic is None else value_critic.to(device)
        self.device = device
        self.max_templates = max_templates
        self.reward_mode = reward_mode
        self.reward_scale = max(1e-6, float(reward_scale))

    def _compute_event_reward(
        self,
        committed: Solution,
        context: DynamicRescheduleContext,
    ) -> float:
        """
        不使用启发式参考解的双 Actor 强化学习奖励函数：
          - "autonomous_telescopic" / "scaled_relative" / "relative" (默认推荐):
            R_e = - (Cmax(committed) - Cmax(context.base_solution)) / kappa_R, kappa_R = 10.0
            基于事件前后最大完工时间差值，减少不同算例绝对工期尺度对奖励数值的影响。
          - "autonomous_scaled" / "scaled_absolute" / "absolute":
            R_e = - Cmax(committed) / kappa_R, kappa_R = 10.0
            直接最小化当前累计最大完工时间。
        """
        if self.reward_mode in ("scaled_absolute", "autonomous_scaled", "absolute"):
            raise ValueError(
                "Absolute reward modes were removed; reward.py defines the sole incremental contract."
            )
        return compute_macro_event_reward(
            committed_makespan=committed.makespan,
            base_makespan=context.base_solution.makespan,
            kappa_r=self.reward_scale,
        )

    @staticmethod
    def _priority_rank(order: Order) -> int:
        return {"高": 0, "中": 1, "普通": 2}.get(order.priority, 9)

    def _score_slots_by_target_completion(
        self,
        solution: Solution,
        batch: Batch,
        valid_slots: List[Tuple[str, int]],
        context: DynamicRescheduleContext,
    ) -> Dict[Tuple[str, int], float]:
        """无副作用试插每个合法槽位，返回目标机台的预测完工时刻。"""
        scores: Dict[Tuple[str, int], float] = {}
        for mid, pos in valid_slots:
            probe = solution.copy()
            probe_batch = copy.deepcopy(batch)
            probe_batch.machine_id = mid
            probe.machine_queues[mid].insert_at(pos, probe_batch)
            scores[(mid, pos)] = float(
                decode_machine(
                    queue=probe.machine_queues[mid],
                    orders_dict=context.orders,
                    machine=self.manager.machines_dict[mid],
                    event_time=context.event_time,
                    frozen_uids=context.frozen_uids,
                )
            )
        return scores

    def _build_graph_context(
        self,
        context: DynamicRescheduleContext,
        current_order: Order,
        solution_override: Optional[Solution] = None,
    ) -> Tuple[PCBGraphBuilder, int, torch.Tensor, torch.Tensor, torch.Tensor, Dict[Tuple[int, ...], int], Any]:
        graph_solution = solution_override if solution_override is not None else context.base_solution
        # 事件订单进入已知订单表后重建索引，避免沿用静态实例的过期图节点。
        active_order_ids = {
            batch.order_id
            for queue in graph_solution.machine_queues.values()
            for batch in queue.batches
        }
        orders = sorted(
            context.orders.values(),
            key=lambda o: (0 if o.order_id in active_order_ids else 1, o.order_id),
        )
        # 保证当前插单作为图中的 frontier；其余节点只提供全局上下文。
        orders = [o for o in orders if o.order_id != current_order.order_id] + [current_order]
        builder = PCBGraphBuilder(
            orders=orders,
            pool_type=self.manager.pool_type,
            machines_dict=self.manager.machines_dict,
            device=self.device,
        )
        frontier_idx = builder.order_to_idx[current_order.order_id]
        mask = self._template_mask(current_order, context)
        graph = builder.build_graph(
            solution=graph_solution,
            current_frontier_idx=frontier_idx,
            candidate_templates=list(mask),
            template_mask=mask,
            event_id=str(getattr(context, "event_id", "dynamic")),
            event_time=float(getattr(context, "event_time", 0.0)),
            scope_ratio=float(getattr(context, "scope_ratio", 0.0)),
        )
        with torch.no_grad():
            embeddings, global_embedding = self.hgnn(graph)
        order_embedding = embeddings["order"][frontier_idx: frontier_idx + 1]
        template_embeddings = embeddings["template"]
        return builder, frontier_idx, global_embedding, order_embedding, template_embeddings, mask, graph

    def _template_mask(self, order: Order, context: DynamicRescheduleContext) -> Dict[Tuple[int, ...], int]:
        templates = generate_partition_templates(
            order,
            max_templates=self.max_templates,
            machines=self.manager.machines_dict,
        )
        available_caps = [
            self.manager.machines_dict[mid].spindles
            for mid in order.eligible_machines
            if mid in self.manager.machines_dict
        ]
        max_available = max(available_caps, default=0)
        return {template: int(max_available > 0 and max(template) <= max_available) for template in templates}

    def _select_manager_action(
        self,
        g: torch.Tensor,
        h_order: torch.Tensor,
        h_templates: torch.Tensor,
        order: Order,
        mask: Dict[Tuple[int, ...], int],
        deterministic: bool,
        context: Optional[DynamicRescheduleContext] = None,
    ) -> Tuple[Tuple[int, ...], int, torch.Tensor]:
        templates = list(mask.keys())
        valid_templates = [template for template in templates if mask[template] == 1]
        if not valid_templates:
            raise RuntimeError(f"订单 {order.order_id} 没有可行动作模板。")
        dist, _ = self.manager_policy(
            g=g,
            h_order=h_order,
            h_templates=h_templates,
            templates=templates,
            template_mask=mask,
            order_W=order.W,
            order=order,
            context=context,
        )
        action_idx = torch.argmax(dist.probs) if deterministic else dist.sample()
        template = templates[action_idx.item()]
        return template, action_idx.item(), dist.log_prob(action_idx)

    def _repartitionable_orders(
        self,
        context: DynamicRescheduleContext,
        new_order: Order,
    ) -> List[Order]:
        """返回可整体重划分的订单；含冻结批次或稳定补集批次的旧订单保持原模板。"""
        by_order: Dict[str, List[int]] = {}
        for queue in context.base_solution.machine_queues.values():
            for batch in queue.batches:
                by_order.setdefault(batch.order_id, []).append(batch.u_id)

        candidates = [new_order]
        for oid, uids in by_order.items():
            if oid == new_order.order_id or not uids:
                continue
            if (
                all(uid not in context.frozen_uids and uid not in context.stable_uids for uid in uids)
                and oid in context.orders
            ):
                candidates.append(context.orders[oid])
        unique: Dict[str, Order] = {order.order_id: order for order in candidates}
        return sorted(
            unique.values(),
            key=lambda order: (
                0 if order.order_id == new_order.order_id else 1,
                self._priority_rank(order),
                float(order.release_time or 0.0),
                float(order.due_date) if order.due_date is not None else float("inf"),
                order.order_id,
            ),
        )

    def _select_split_option(
        self,
        context: DynamicRescheduleContext,
        order: Order,
        solution: Solution,
        deterministic: bool,
        option_index: int,
    ) -> Tuple[Tuple[int, ...], dict, PCBGraphBuilder, int]:
        builder, frontier_idx, g, h_order, h_templates, mask, graph = self._build_graph_context(
            context,
            order,
            solution_override=solution,
        )
        template, action_idx, log_prob = self._select_manager_action(
            g, h_order, h_templates, order, mask, deterministic, context=context
        )
        value = None
        if self.value_critic is not None:
            value = self.value_critic(g, h_order).detach().cpu()
        stored_mask = (
            self.manager_policy.get_last_joint_mask_dict()
            if hasattr(self.manager_policy, "get_last_joint_mask_dict")
            else dict(mask)
        )
        record = {
            "order_id": order.order_id,
            "option_index": option_index,
            "g": g.detach().cpu(),
            "h_order": h_order.detach().cpu(),
            "graph": graph,
            "frontier_idx": frontier_idx,
            "templates": list(mask.keys()),
            "mask": stored_mask,
            "order_W": order.W,
            "action_idx": action_idx,
            "log_prob": log_prob.detach().cpu(),
            "value": value,
            "tau": 0,
        }
        return template, record, builder, frontier_idx

    def _place_pending_batches(
        self,
        context: DynamicRescheduleContext,
        tx,
        builder: PCBGraphBuilder,
        frontier_idx: int,
        selected_templates: Dict[str, Tuple[int, ...]],
        deterministic: bool,
    ) -> Tuple[List[Tuple[str, int]], List[float], List[dict], List[str]]:
        errors: List[str] = []
        worker_actions: List[Tuple[str, int]] = []
        worker_log_probs: List[float] = []
        worker_records: List[dict] = []
        templates_cache: Dict[str, Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], int], Optional[Tuple[int, ...]]]] = {}

        while tx.current_batch_idx < tx.total_sub_batches:
            sub_idx = tx.current_batch_idx
            curr_b = tx.get_current_batch()
            if curr_b is None:
                break
            curr_order_id = curr_b.order_id
            curr_order_idx = builder.order_to_idx.get(curr_order_id, frontier_idx)
            if curr_order_id not in templates_cache:
                curr_order_obj = context.orders[curr_order_id]
                curr_mask = self._template_mask(curr_order_obj, context)
                templates_cache[curr_order_id] = (
                    list(curr_mask.keys()),
                    curr_mask,
                    selected_templates.get(curr_order_id),
                )
            curr_cands, curr_mask, curr_tpl = templates_cache[curr_order_id]

            graph = builder.build_graph(
                solution=tx.snapshot,
                current_frontier_idx=curr_order_idx,
                active_sub_batches=tx.sub_batches,
                current_sub_batch_idx=sub_idx,
                candidate_templates=curr_cands,
                template_mask=curr_mask,
                selected_template=curr_tpl,
                event_id=str(getattr(context, "event_id", "dynamic")),
                event_time=float(getattr(context, "event_time", 0.0)),
                scope_ratio=float(getattr(context, "scope_ratio", 0.0)),
            )
            with torch.inference_mode():
                embeddings, global_embedding = self.hgnn(graph)
            h_batch = embeddings["subbatch"][sub_idx: sub_idx + 1]
            h_order = embeddings["order"][curr_order_idx: curr_order_idx + 1]
            h_slots = embeddings["slot"]
            all_slot_keys = list(builder.slot_map.keys())
            valid_slots = [
                (mid, pos)
                for mid, pos in tx.get_valid_slots_for_batch(curr_b)
            ]
            if not valid_slots:
                errors.append(f"订单 {curr_order_id} 子批 {sub_idx} 没有可行机器/插槽。")
                return worker_actions, worker_log_probs, worker_records, errors

            c_before_machines: Dict[str, float] = {}
            for mid, queue in tx.snapshot.machine_queues.items():
                machine = self.manager.machines_dict.get(mid)
                if machine and machine.pool_type == self.manager.pool_type:
                    c_before_machines[mid] = decode_machine(
                        queue=queue,
                        orders_dict=context.orders,
                        machine=machine,
                        event_time=context.event_time,
                        frozen_uids=context.frozen_uids,
                    )
            worker_kwargs = {
                "h_batch": h_batch,
                "h_slots": h_slots,
                "slot_keys": all_slot_keys,
                "valid_slots": valid_slots,
            }
            if getattr(self.worker_policy, "requires_order_id", False):
                worker_kwargs["order_id"] = curr_order_id
            if getattr(self.worker_policy, "requires_slot_completion_times", False):
                worker_kwargs["slot_completion_times"] = self._score_slots_by_target_completion(
                    tx.snapshot, curr_b, valid_slots, context
                )
            try:
                dist, _ = self.worker_policy(**worker_kwargs)
            except Exception as exc:
                errors.append(f"Worker 决策执行异常: {exc}")
                return worker_actions, worker_log_probs, worker_records, errors
            action_idx = torch.argmax(dist.probs) if deterministic else dist.sample()
            chosen_slot = all_slot_keys[action_idx.item()]
            mean_c_before = sum(c_before_machines.values()) / max(1, len(c_before_machines))
            if not tx.step_assign(chosen_slot[0], chosen_slot[1]):
                errors.append(f"Worker 选择的插槽 {chosen_slot} 未通过事务前置校验。")
                return worker_actions, worker_log_probs, worker_records, errors

            target_machine = self.manager.machines_dict[chosen_slot[0]]
            c_after_mid = decode_machine(
                queue=tx.snapshot.machine_queues[chosen_slot[0]],
                orders_dict=context.orders,
                machine=target_machine,
                event_time=context.event_time,
                frozen_uids=context.frozen_uids,
            )
            delta_c_machine = c_after_mid - mean_c_before
            c_after_machines = dict(c_before_machines)
            c_after_machines[chosen_slot[0]] = c_after_mid
            curr_makespan = max(c_after_machines.values(), default=0.0)
            step_reward = compute_micro_step_reward(
                predicted_machine_cmax=c_after_mid,
                pool_mean_cmax=mean_c_before,
                current_actor_makespan=curr_makespan,
            )
            worker_actions.append(chosen_slot)
            worker_log_probs.append(float(dist.log_prob(action_idx).item()))
            value = None
            if self.value_critic is not None:
                value = self.value_critic(global_embedding, h_order, h_batch).detach().cpu()
            worker_records.append(
                {
                    "order_id": curr_order_id,
                    "g": global_embedding.detach().cpu(),
                    "h_order": h_order.detach().cpu(),
                    "graph": graph,
                    "sub_idx": sub_idx,
                    "order_idx": curr_order_idx,
                    "h_batch": h_batch.detach().cpu(),
                    "h_slots": h_slots.detach().cpu(),
                    "slot_keys": list(all_slot_keys),
                    "valid_slots": list(valid_slots),
                    "action_idx": action_idx.item(),
                    "log_prob": dist.log_prob(action_idx).detach().cpu(),
                    "value": value,
                    "load_penalty": float(max(0.0, delta_c_machine)),
                    "delta_c_machine": float(delta_c_machine),
                    "curr_makespan": float(curr_makespan),
                    "step_reward": float(step_reward),
                }
            )
        return worker_actions, worker_log_probs, worker_records, errors

    def _run_sequential_options(
        self,
        context: DynamicRescheduleContext,
        new_order: Order,
        deterministic: bool,
    ) -> Tuple[Optional[Solution], Dict[str, Tuple[int, ...]], List[dict], List[Tuple[str, int]], List[float], List[dict], List[str]]:
        tx = self.manager.begin_progressive_reallocation(context)
        plan: Dict[str, Tuple[int, ...]] = {}
        manager_records: List[dict] = []
        actions: List[Tuple[str, int]] = []
        log_probs: List[float] = []
        worker_records: List[dict] = []
        errors: List[str] = []
        last_builder: Optional[PCBGraphBuilder] = None
        last_frontier_idx = 0

        for option_index, order in enumerate(self._repartitionable_orders(context, new_order)):
            template, manager_record, builder, frontier_idx = self._select_split_option(
                context, order, tx.snapshot, deterministic, option_index
            )
            plan[order.order_id] = template
            tx.append_partition_option(order, template)
            option_start = len(worker_records)
            opt_actions, opt_log_probs, opt_records, opt_errors = self._place_pending_batches(
                context, tx, builder, frontier_idx, plan, deterministic
            )
            actions.extend(opt_actions)
            log_probs.extend(opt_log_probs)
            worker_records.extend(opt_records)
            errors.extend(opt_errors)
            if opt_errors:
                tx.rollback()
                return None, plan, manager_records, actions, log_probs, worker_records, errors
            tx.decode_partial()
            manager_record["tau"] = len(worker_records) - option_start
            if manager_record["tau"] <= 0:
                tx.rollback()
                errors.append(f"订单 {order.order_id} 的 Split Option 未产生 Placement 转移。")
                return None, plan, manager_records, actions, log_probs, worker_records, errors
            manager_records.append(manager_record)
            last_builder, last_frontier_idx = builder, frontier_idx

        _, tail_end = tx.append_remaining_fixed_batches()
        tail_count = tail_end - tx.current_batch_idx
        if tail_count > 0:
            if last_builder is None:
                last_builder, last_frontier_idx, *_ = self._build_graph_context(
                    context, new_order, solution_override=tx.snapshot
                )
            tail_actions, tail_log_probs, tail_records, tail_errors = self._place_pending_batches(
                context, tx, last_builder, last_frontier_idx, plan, deterministic
            )
            actions.extend(tail_actions)
            log_probs.extend(tail_log_probs)
            worker_records.extend(tail_records)
            errors.extend(tail_errors)
            if tail_errors:
                tx.rollback()
                return None, plan, manager_records, actions, log_probs, worker_records, errors
            tx.decode_partial()
            if manager_records:
                manager_records[-1]["tau"] += len(tail_records)

        committed, candidate, _ = tx.commit()
        if not committed:
            errors.append("渐进 OptionTransaction 解码或覆盖校验失败。")
            return None, plan, manager_records, actions, log_probs, worker_records, errors
        return candidate, plan, manager_records, actions, log_probs, worker_records, errors

    def run_next_event(
        self,
        deterministic: bool = True,
        context: Optional[DynamicRescheduleContext] = None,
    ) -> DynamicDecision:
        """由双 Actor 生成全部分批与机台—槽位动作的动态重调度决策。

        1. 由网络自主完成宏观分批模板与微观机台—插槽选择；
        2. 环境严守物理约束、冻结前缀、稳定补集拓扑签名与半主动时间解码；
        3. 仅当 Actor 自主方案通过原子事务验证时，方案被采纳并录入 PPO 训练池；
        4. 物理不可行时记录未提交结果并丢弃无效轨迹，不生成启发式替代调度方案。
        """
        if context is None:
            context = self.manager.trigger_next_event()
        start_event_t = time.perf_counter()
        decision = DynamicDecision(
            event_id=context.event.event_id,
            committed=False,
            order_id=context.event.order.order_id,
            cmax_before=float(context.base_solution.makespan),
            affected_uids=list(context.affected_uids),
            initial_k_scope=context.k_scope if context.k_scope < 900000 else len(context.affected_uids),
            effective_k_scope=context.k_scope if context.k_scope < 900000 else len(context.affected_uids),
            scope_ratio=getattr(context, "scope_ratio", 0.0),
        )

        # 动态插单事件：由 SplitPolicy (Agent1) 和 PlacementPolicy (Agent2) 联合自主决策
        # 严格接入参数规范 1.7 节与动态建模文档算法 1 第 7 步的确定性扩域循环 (ΔK=4)
        from scope_selector import select_affected_domain

        order = context.event.order
        initial_k = context.k_scope
        curr_k = context.k_scope
        expansion_count = 0
        curr_ctx = context
        total_unfrozen = len([
            b for q in curr_ctx.base_solution.machine_queues.values()
            for b in q.batches if b.u_id not in curr_ctx.frozen_uids
        ])

        while True:
            candidate = None
            try:
                (
                    candidate,
                    partition_plan,
                    manager_records,
                    actions,
                    worker_log_probs,
                    worker_records,
                    worker_errors,
                ) = self._run_sequential_options(curr_ctx, order, deterministic)
                decision.partition_plan = dict(partition_plan)
                decision.manager_records = manager_records
                new_record = next(record for record in manager_records if record["order_id"] == order.order_id)
                decision.template = partition_plan[order.order_id]
                decision.manager_action_idx = new_record["action_idx"]
                decision.manager_log_prob = float(new_record["log_prob"].item())
                decision.manager_g = new_record["g"]
                decision.manager_h_order = new_record["h_order"]
                decision.manager_templates = new_record["templates"]
                decision.manager_mask = new_record["mask"]
                decision.manager_value = new_record["value"]
                decision.worker_actions = actions
                decision.worker_log_probs = worker_log_probs
                decision.worker_records = worker_records
                decision.errors.extend(worker_errors)
            except Exception as exc:
                decision.errors.append(f"两个 Actor 自主生成候选解失败: {exc}")
                candidate = None

            if candidate is not None:
                decision.actor_candidate_solution = candidate.copy()
                # 严格经过原子事务的物理与拓扑约束检验
                tx = self.manager.begin_transaction(curr_ctx)
                ok, committed, errors = tx.commit(candidate)
                if ok:
                    decision.committed = True
                    decision.candidate_solution = committed.copy()
                    decision.cmax_after = float(committed.makespan)
                    decision.affected_uids = list(curr_ctx.affected_uids)
                    decision.reward = self._compute_event_reward(committed, curr_ctx)
                    decision.initial_k_scope = initial_k if initial_k < 900000 else len(curr_ctx.affected_uids)
                    decision.effective_k_scope = curr_k if curr_k < 900000 else len(curr_ctx.affected_uids)
                    decision.expansion_count = expansion_count
                    decision.scope_ratio = getattr(curr_ctx, "scope_ratio", 0.0)
                    decision.response_time_seconds = time.perf_counter() - start_event_t
                    return decision
                decision.errors.extend(errors)

            # 若当前受影响域无合法插槽或校验未通过，按 ΔK 扩大受影响域并重建图与动作集重试
            # 严格依据参数规范 1.1 节：作用域下界或非正扩张步长均禁用自动扩域。
            if initial_k == 0 or getattr(self.manager, "delta_k", 0) <= 0 or curr_k >= total_unfrozen:
                break
            expansion_count += 1
            curr_k = min(total_unfrozen, curr_k + self.manager.delta_k)
            new_ord_objs = [curr_ctx.orders[oid] for oid in curr_ctx.new_order_ids if oid in curr_ctx.orders]
            scope_res = select_affected_domain(
                solution=curr_ctx.base_solution,
                event_time=curr_ctx.event_time,
                frozen_uids=curr_ctx.frozen_uids,
                new_orders=new_ord_objs,
                orders_dict=curr_ctx.orders,
                machines_dict=self.manager.machines_dict,
                k_scope=curr_k,
                k_seed=curr_ctx.k_seed,
                enable_adjacency_closure=getattr(self.manager, "enable_adjacency_closure", True),
            )
            curr_ctx = DynamicRescheduleContext(
                event=curr_ctx.event,
                event_time=curr_ctx.event_time,
                base_solution=curr_ctx.base_solution,
                orders=curr_ctx.orders,
                freeze=curr_ctx.freeze,
                new_order_ids=curr_ctx.new_order_ids,
                k_scope=curr_k,
                k_seed=curr_ctx.k_seed,
                affected_uids=scope_res.affected_uids,
                stable_uids=scope_res.stable_uids,
                stable_signature=scope_res.stable_signature,
                scope_ratio=scope_res.scope_ratio,
            )
            self.manager.active_context = curr_ctx

        decision.initial_k_scope = initial_k if initial_k < 900000 else len(curr_ctx.affected_uids)
        decision.effective_k_scope = curr_k if curr_k < 900000 else len(curr_ctx.affected_uids)
        decision.expansion_count = expansion_count
        decision.affected_uids = list(curr_ctx.affected_uids)
        decision.scope_ratio = getattr(curr_ctx, "scope_ratio", 0.0)
        decision.response_time_seconds = time.perf_counter() - start_event_t
        return decision
