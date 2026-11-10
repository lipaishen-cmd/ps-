"""
scope_selector.py
=================
确定性受影响域范围选择器（Deterministic ScopeSelector，式 8–10）。

功能说明：
  1. 接收当前已解码排程 S^{e-1}、动态事件 e、冻结前缀 F_g^e 以及范围上界 K_scope；
  2. 计算既有未开始批次 U_{old,g}^e = B_g^{e-1} \\ F_g^e；
  3. 执行资源池隔离与机台资格过滤：仅保留与新订单共享至少一台合法机器的旧批次；
  4. 按三级字典序排序：
     - 键 1（机台拥堵度）：所属机台的未完工队列预测负载降序；
     - 键 2（时间临近性）：原计划开工时间距离事件时刻 t_e 升序；
     - 键 3（打破平局）：批次 u_id 升序；
  5. 截取前 K_scope 个种子批次，并在不超过 K_scope 的前提下加入同机队列紧邻闭包；
  6. 构造稳定补集 K_g^e = U_{old,g}^e \\ A_g^e 并锁定其稳定性签名 Sig^e(b)；
  7. 组合策略决策域 D_g^e = N_g^e ∪ A_g^e。

约束：
  - 范围选择器是确定性问题分解器，不包含可学习参数，不使用未来事件信息；
  - K_scope=0 与 K_scope>=|U_{old}| 分别定义作用域参数的下界和上界；
  - 稳定补集 K_g^e 与冻结前缀保持原有机台分配、批量与相对次序不变；开工与完工时刻由解码器按式 (16) 重新计算推进。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "environment"))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_THIS_DIR, _ENVIRONMENT_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_models import Batch, DynamicInsertEvent, Machine, Order, Solution


@dataclass(frozen=True)
class ScopeSelectionResult:
    """受影响域选择的确定性输出，包含三域划分与签名。"""

    as_of: float
    k_scope: int
    frozen_uids: frozenset[int]          # F_g^e: 冻结前缀
    affected_uids: frozenset[int]        # A_g^e: 确定性受影响域旧批次
    stable_uids: frozenset[int]          # K_g^e: 稳定补集 (U_old \ A_g^e)
    unfrozen_old_uids: frozenset[int]    # U_{old,g}^e: 全部既有未开始旧批次
    new_order_ids: Tuple[str, ...]       # 新到达订单编号
    stable_signature: Dict[int, Tuple]   # 稳定补集批次的刚性签名
    scope_ratio: float                   # |A_g^e| / |U_old,g^e|


def compute_stable_signature(solution: Solution, stable_uids: Iterable[int]) -> Dict[int, Tuple]:
    """
    计算稳定补集的刚性结构拓扑签名 (式 10 / 参数规范 1.6 节):
    Sig_K^e(b) = (m_b, o(b), q_b, ρ_{K,m}(b))
    其中 q_b ≡ b_ik 为子批叠板数，ρ_{K,m}(b) 为稳定补集子序列中的相对排名。
    稳定补集锁定机台归属、订单、批次大小及稳定批次间的相对先后拓扑序；
    其开工与完工绝对时间由统一解码器递推重算，不人为锁死原计划绝对时间。
    """
    wanted = set(stable_uids)
    sig: Dict[int, Tuple] = {}
    for mid, queue in solution.machine_queues.items():
        stable_in_queue = [b for b in queue.batches if b.u_id in wanted]
        for rel_pos, b in enumerate(stable_in_queue):
            sig[b.u_id] = (
                mid,
                b.order_id,
                b.stack_count,
                rel_pos,
            )
    return sig


def select_affected_domain(
    solution: Solution,
    event_time: float,
    frozen_uids: frozenset[int],
    new_orders: Sequence[Order],
    orders_dict: Dict[str, Order],
    machines_dict: Dict[str, Machine],
    k_scope: int = 8,
    k_seed: int = 5,
    enable_adjacency_closure: bool = True,
) -> ScopeSelectionResult:
    """
    确定性范围选择核心算法（式 8–10）。
    """
    t_e = float(event_time)
    k_limit = max(0, int(k_scope))
    s_limit = max(0, int(k_seed))

    # 1. 收集全部既有未开始批次 U_{old,g}^e
    unfrozen_batches: List[Batch] = []
    machine_unfrozen_batches: Dict[str, List[Batch]] = {m: [] for m in solution.machine_queues}
    machine_queue_positions: Dict[int, Tuple[str, int]] = {}

    for mid, queue in solution.machine_queues.items():
        for pos, b in enumerate(queue.batches):
            if b.u_id not in frozen_uids:
                unfrozen_batches.append(b)
                machine_unfrozen_batches[mid].append(b)
                machine_queue_positions[b.u_id] = (mid, pos)

    unfrozen_old_uids = frozenset(b.u_id for b in unfrozen_batches)

    # 作用域下界：k_scope == 0 时不选择任何旧批次。
    if k_limit == 0 or not unfrozen_batches:
        stable_uids = unfrozen_old_uids
        affected_uids = frozenset()
        stable_sig = compute_stable_signature(solution, stable_uids)
        new_ids = tuple(o.order_id for o in new_orders)
        return ScopeSelectionResult(
            as_of=t_e,
            k_scope=k_limit,
            frozen_uids=frozen_uids,
            affected_uids=affected_uids,
            stable_uids=stable_uids,
            unfrozen_old_uids=unfrozen_old_uids,
            new_order_ids=new_ids,
            stable_signature=stable_sig,
            scope_ratio=0.0,
        )

    # 2. 计算各机台综合预测剩余负载 L_m(t_e) = R_m(t_e) + \sum_{b \in Q_m^e} [P(o(b), m) + s_m(b^-, b)]
    machine_congestion: Dict[str, float] = {}
    for mid, m_batches in machine_unfrozen_batches.items():
        queue = solution.machine_queues.get(mid)
        m_obj = machines_dict.get(mid)

        # 2.1 计算活跃操作在事件时刻 t_e 的剩余占用时间 R_m(t_e)
        r_m = 0.0
        prev_order_id: Optional[str] = None
        if queue is not None:
            for b in queue.batches:
                if b.u_id in frozen_uids:
                    if float(b.finish_time) > t_e:
                        r_m = max(r_m, float(b.finish_time) - t_e)
                    prev_order_id = b.order_id

        # 2.2 累加未开始旧批次队列 Q_m^e 的加工工时与序列相关换型时间
        workload = r_m
        for b in m_batches:
            p_time = float(b.processing_time) if b.processing_time > 0 else 0.0
            if p_time <= 0.0 and m_obj is not None:
                b_order = orders_dict.get(b.order_id)
                if b_order is not None:
                    try:
                        from time_model import processing_time as calc_processing_time
                        p_time = float(calc_processing_time(b_order, m_obj))
                    except Exception:
                        p_time = 0.0
            s_time = 0.0
            if m_obj is not None:
                try:
                    from time_model import setup_time as calc_setup_time
                    s_time = calc_setup_time(
                        machine=m_obj,
                        prev_order_id=prev_order_id,
                        curr_order_id=b.order_id,
                        is_first_batch=(prev_order_id is None),
                    )
                except Exception:
                    s_time = float(m_obj.s_setup if prev_order_id != b.order_id else 0.0)
            workload += (p_time + s_time)
            prev_order_id = b.order_id

        machine_congestion[mid] = workload

    # 3. 确定新到达订单的可行机台并集
    new_eligible_machines: Set[str] = set()
    for o in new_orders:
        if o.pool_type != solution.pool_type:
            raise ValueError(
                f"新订单 {o.order_id} 的资源池 {o.pool_type} 与解资源池 {solution.pool_type} 不一致。"
            )
        new_eligible_machines.update(o.eligible_machines)

    # 4. 筛选与新订单共享机台资格且满足主轴容量可行性的候选旧批次及其他未冻结旧批次
    compatible_batches: List[Batch] = []
    other_batches: List[Batch] = []
    for b in unfrozen_batches:
        b_order = orders_dict.get(b.order_id)
        if b_order is None:
            continue
        # 严格检查新旧订单共享机台资格与主轴容量承载可行性
        if not new_eligible_machines:
            other_batches.append(b)
            continue
        shared_m = [
            m for m in b_order.eligible_machines
            if m in new_eligible_machines
            and machines_dict.get(m) is not None
            and machines_dict[m].spindles >= b.stack_count
        ]
        if shared_m:
            compatible_batches.append(b)
        else:
            other_batches.append(b)

    # 5. 三级字典序排序
    # 键 1: 机台拥堵度降序 (-congestion)
    # 键 2: 开工时间距事件时刻临近性升序 (|start_time - t_e|)
    # 键 3: 批次 UID 升序 (u_id)
    def sort_key(b: Batch):
        b_order = orders_dict.get(b.order_id)
        shared_m = [
            m for m in b_order.eligible_machines
            if m in new_eligible_machines
            and machines_dict.get(m) is not None
            and machines_dict[m].spindles >= b.stack_count
        ] if b_order and new_eligible_machines else []
        if shared_m:
            congestion = max(machine_congestion.get(m, 0.0) for m in shared_m)
        else:
            mid, _ = machine_queue_positions[b.u_id]
            congestion = machine_congestion.get(mid, 0.0)
        proximity = abs(float(b.start_time) - t_e) if float(b.start_time) > 0 else 0.0
        return (-congestion, proximity, b.u_id)

    compatible_batches.sort(key=sort_key)
    other_batches.sort(key=sort_key)

    # 6. 选取前 K_seed 个种子批次 (不超过 k_limit)
    # 严格仅从具备资格重叠的兼容批次中选取种子；若无兼容批次则种子为空，避免无关机台扰动
    seed_pool = compatible_batches
    seed_count = min(s_limit, k_limit, len(seed_pool))
    selected_uids: List[int] = [b.u_id for b in seed_pool[:seed_count]]
    selected_set: Set[int] = set(selected_uids)

    # 7. 队列邻接闭包扩展（在不超过 k_limit 预算下，优先合并同机队列紧邻的批次）
    if enable_adjacency_closure:
        if len(selected_set) < k_limit:
            frontier = list(selected_uids)
            frontier_idx = 0
            while frontier_idx < len(frontier) and len(selected_set) < k_limit:
                uid = frontier[frontier_idx]
                frontier_idx += 1
                if len(selected_set) >= k_limit:
                    break
                mid, pos = machine_queue_positions[uid]
                queue = solution.machine_queues[mid]
                # 尝试纳入前驱
                if pos > 0:
                    prev_b = queue.batches[pos - 1]
                    if prev_b.u_id in unfrozen_old_uids and prev_b.u_id not in selected_set:
                        selected_set.add(prev_b.u_id)
                        frontier.append(prev_b.u_id)
                        if len(selected_set) >= k_limit:
                            break
                # 尝试纳入后继
                if pos + 1 < len(queue.batches):
                    next_b = queue.batches[pos + 1]
                    if next_b.u_id in unfrozen_old_uids and next_b.u_id not in selected_set:
                        selected_set.add(next_b.u_id)
                        frontier.append(next_b.u_id)
                        if len(selected_set) >= k_limit:
                            break
    else:
        # 禁用邻接闭包时，仅依据全局字典序继续追加独立批次。
        for b in seed_pool[seed_count:] + other_batches:
            if len(selected_set) >= k_limit:
                break
            selected_set.add(b.u_id)

    # 8. 作用域上界对齐：K_scope >= |U_old,g^e| 时纳入全部未冻结旧批次。
    if k_scope >= len(unfrozen_old_uids):
        selected_set.update(unfrozen_old_uids)

    affected_uids = frozenset(selected_set)
    stable_uids = unfrozen_old_uids.difference(affected_uids)
    stable_sig = compute_stable_signature(solution, stable_uids)
    ratio = len(affected_uids) / max(1, len(unfrozen_old_uids))
    new_ids = tuple(o.order_id for o in new_orders)

    return ScopeSelectionResult(
        as_of=t_e,
        k_scope=k_limit,
        frozen_uids=frozen_uids,
        affected_uids=affected_uids,
        stable_uids=stable_uids,
        unfrozen_old_uids=unfrozen_old_uids,
        new_order_ids=new_ids,
        stable_signature=stable_sig,
        scope_ratio=ratio,
    )
