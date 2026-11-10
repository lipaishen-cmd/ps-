"""
solution_lifecycle.py
=====================
解结构生命周期、双向规范化映射、断言验证器与原子事务管理器
（规范第 8 章、第 9.8 节、第 9.14 节，TC-03、TC-05、TC-06、TC-08 核心）

功能说明：
  1. 解对象的标准化构建（create_empty_solution）与深克隆（clone_solution）。
  2. 规范化映射 Φ（normalize_solution）：保证机器队列主表示的标准字典序。
  3. 支配规约合并（repair_adjacent_batches）：合并同机相邻同订单且不超过主轴容量的批次。
  4. 五大物理断言验证器（validate_solution，只读，严禁修改解）：
     - A1 (TC-03): 需求精确守恒（∑ b.stack_count == W_i，零容差）。
     - A2 (TC-04): 机器资格合法性（m ∈ order.eligible_machines）。
     - A3 (TC-05): 主轴容量合法性（b.stack_count <= machine.spindles）。
     - A4 (TC-06): 批次 u_id 唯一性与出现性（单解内每个活跃 u_id 恰好出现一次）。
     - A5 (TC-10 & 时序): 跨池物理隔离与时间无重叠性。
  5. 动态拆分原子事务（AtomicTransaction，TC-08 核心）：
     - 支持事务级深拷贝快照、原子提交（Commit）与异常自动全量回滚（Rollback）。
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

# 动态引入第 1 层与同层模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = _THIS_DIR
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in [_ENVIRONMENT_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_models import (
    POOL_B,
    POOL_D,
    Batch,
    Machine,
    MachineQueue,
    Order,
    Solution,
)
from eligibility_builder import ALL_MACHINES, POOL_B_MACHINES, POOL_D_MACHINES
from time_model import processing_time, setup_time


def create_empty_solution(pool_type: str) -> Solution:
    """
    为指定资源池创建一个初始空解对象（初始化所有合法机床的空队列）。

    参数
    ----
    pool_type : str — POOL_D 或 POOL_B

    返回
    ----
    Solution 对象，各机器队列均为空。
    """
    if pool_type == POOL_D:
        machine_ids = POOL_D_MACHINES
    elif pool_type == POOL_B:
        machine_ids = POOL_B_MACHINES
    else:
        raise ValueError(f"未知资源池类型 '{pool_type}'，合法值为 POOL_D 或 POOL_B")

    queues: Dict[str, MachineQueue] = {}
    for mid in sorted(machine_ids):
        m = ALL_MACHINES[mid]
        queues[mid] = MachineQueue(
            machine_id=mid,
            pool_type=pool_type,
            spindles=m.spindles,
            batches=[],
            prefix_cache=[],
        )

    return Solution(
        pool_type=pool_type,
        machine_queues=queues,
        active_batches={},
        makespan=0.0,
        total_setup_time=0.0,
        spindle_utilization=0.0,
    )


def clone_solution(solution: Solution) -> Solution:
    """
    深拷贝解对象（创建完全隔离的独立内存副本，规范 9.14 节）。
    """
    return copy.deepcopy(solution)


def normalize_solution(solution: Solution) -> None:
    """
    双向规范化映射 Φ（规范 8.4 节）：
      1. 确保机器队列字典按机器编号升序排布；
      2. 重新同步解内的 active_batches 索引映射；
      3. 确保批次内部引用的 machine_id 与其所在队列一致。
    """
    # 1. 字典序排列机器队列
    sorted_queues: Dict[str, MachineQueue] = {}
    for mid in sorted(solution.machine_queues.keys()):
        sorted_queues[mid] = solution.machine_queues[mid]
    solution.machine_queues = sorted_queues

    # 2. 重新构建活跃批次索引并核对 machine_id
    active: Dict[int, Batch] = {}
    for mid, queue in solution.machine_queues.items():
        for b in queue.batches:
            b.machine_id = mid  # 强制同步归属机器
            active[b.u_id] = b
    solution.active_batches = active


def repair_adjacent_batches(
    solution: Solution,
    orders_dict: Dict[str, Order],
    machines_dict: Optional[Dict[str, Machine]] = None,
) -> bool:
    """
    支配规约合并修复（规范 8.4 节）：
    遍历各机队列，若发现相邻两个批次属于同一订单，且两者装载叠板单元数之和
    不大于该机器主轴数，则在不增加换型的前提下直接合并为单一批次，消除排列冗余。

    返回
    ----
    bool — 是否执行了至少一次批次合并
    """
    if machines_dict is None:
        machines_dict = ALL_MACHINES

    any_merged = False

    for mid, queue in solution.machine_queues.items():
        if len(queue.batches) < 2:
            continue

        spindles = machines_dict[mid].spindles
        new_batches: List[Batch] = []
        k = 0
        while k < len(queue.batches):
            curr_b = queue.batches[k]
            if k + 1 < len(queue.batches):
                next_b = queue.batches[k + 1]
                # 检查合并条件：同订单且合并后不超过机床主轴容量
                if (
                    curr_b.order_id == next_b.order_id
                    and curr_b.stack_count + next_b.stack_count <= spindles
                ):
                    # 确定性支配合并：保留前一批次的 u_id，累加叠板数
                    merged_batch = Batch(
                        order_id=curr_b.order_id,
                        u_id=curr_b.u_id,
                        stack_count=curr_b.stack_count + next_b.stack_count,
                        machine_id=mid,
                    )
                    new_batches.append(merged_batch)
                    any_merged = True
                    k += 2  # 跳过被合并的批次
                    continue

            new_batches.append(curr_b)
            k += 1

        queue.batches = new_batches

    if any_merged:
        # 更新索引并清空失效的前缀缓存
        normalize_solution(solution)
        for q in solution.machine_queues.values():
            q.prefix_cache = []

    return any_merged


def validate_solution(
    solution: Solution,
    orders_dict: Dict[str, Order],
    machines_dict: Optional[Dict[str, Machine]] = None,
    raise_exception: bool = True,
    verify_time_recurrence: bool = True,
    event_time: float = 0.0,
    frozen_uids: Optional[Iterable[int]] = None,
    release_times: Optional[Dict[str, float]] = None,
    require_decoded: bool = False,
) -> Tuple[bool, List[str]]:
    """
    五大物理断言严格验证器（规范 9.8 节，只读操作，严禁修改解状态）。

    断言检查清单：
      A1 (TC-03): 订单需求精确守恒（∑ b.stack_count == W_i，零容差）。
      A2 (TC-04): 机器资格合法性（m ∈ order.eligible_machines）。
      A3 (TC-05): 主轴容量与整数性合法性（b.stack_count ∈ Z+, b.stack_count <= machine.spindles）。
      A4 (TC-06): 批次 u_id 唯一性（每个活跃 u_id 在全部队列中恰好出现一次）。
      A5 (TC-10 & 时序): 跨池隔离，且已解码批次满足式 (16) 时间递推。

    参数
    ----
    solution        : Solution — 待校验的解
    orders_dict     : Dict[str, Order] — 本池订单字典
    machines_dict   : Optional[Dict[str, Machine]]
    raise_exception : bool — 若为 True，任一断言失败立即 raise ValueError
    verify_time_recurrence: bool — 若为 True，逐批校验工时公式与开工递推 (式 16)
    event_time      : float — 动态事件时刻 t_e（静态为 0.0）
    frozen_uids     : Optional[Iterable[int]] — 冻结批次 UID 集合（冻结批次跳过动态时间重算）
    release_times   : Optional[Dict[str, float]] — 订单释放时刻映射字典
    require_decoded : bool — 若为 True，解中所有活跃批次必须具有大于 0 的有效时间，未解码批次直接判定违规

    返回
    ----
    (is_valid: bool, errors: List[str])
    """
    if machines_dict is None:
        machines_dict = ALL_MACHINES

    errors: List[str] = []
    pool_type = solution.pool_type
    frozen_set: Optional[FrozenSet[int]] = (
        frozenset(int(uid) for uid in frozen_uids) if frozen_uids is not None else None
    )

    any_decoded = any(
        b.finish_time > 0.0 for q in solution.machine_queues.values() for b in q.batches
    )
    must_be_decoded = require_decoded or (verify_time_recurrence and any_decoded)

    # 1. 收集各订单在解中的实际叠板单元数与批次编号
    seen_uids: Set[int] = set()
    order_stack_sum: Dict[str, int] = {oid: 0 for oid in orders_dict.keys()}

    for mid, queue in solution.machine_queues.items():
        machine = machines_dict.get(mid)
        if machine is None:
            errors.append(f"解中包含未知机器 ID: '{mid}'")
            continue

        # A5-1: 检查机器与解所属资源池是否一致
        if machine.pool_type != pool_type:
            errors.append(
                f"TC-10 FAIL: 机器 {mid}（池={machine.pool_type}）出现在 {pool_type} 解中！"
            )

        prev_finish = 0.0
        for b_idx, b in enumerate(queue.batches):
            # A4: u_id 唯一性断言（TC-06）
            if b.u_id in seen_uids:
                errors.append(
                    f"TC-06 FAIL: 批次编号 u_id={b.u_id} 在解中重复出现！机器={mid}, 索引={b_idx}"
                )
            seen_uids.add(b.u_id)

            # 校验订单存在性
            order = orders_dict.get(b.order_id)
            if order is None:
                errors.append(
                    f"批次 u_id={b.u_id} 对应的订单 '{b.order_id}' 不在当前资源池订单字典中"
                )
                continue

            # 累加需求
            order_stack_sum[b.order_id] += b.stack_count

            # A2: 机器资格合法性断言（TC-04）
            if mid not in order.eligible_machines:
                errors.append(
                    f"TC-04 FAIL: 批次 u_id={b.u_id} (订单={b.order_id}) 被指派到不合格机器 {mid}！"
                    f"合格机器为 {order.eligible_machines}"
                )

            # A3: 主轴容量与整数性合法性断言（TC-05）
            if not isinstance(b.stack_count, int) or b.stack_count <= 0:
                errors.append(
                    f"批次 u_id={b.u_id} 的叠板单元数 stack_count={b.stack_count} 必须为正整数 (类型={type(b.stack_count).__name__})"
                )
            elif b.stack_count > machine.spindles:
                errors.append(
                    f"TC-05 FAIL: 批次 u_id={b.u_id} (订单={b.order_id}) 叠板数 {b.stack_count} "
                    f"超过机器 {mid} 的物理主轴数 {machine.spindles}！"
                )

            # A5-2: 时序递增性与工时模型递推验证
            if b.finish_time <= 0.0:
                if must_be_decoded:
                    errors.append(
                        f"时序缺失违规: 机器 {mid} 活跃批次 u_id={b.u_id} 未解码 (finish_time={b.finish_time})，"
                        "已解码候选解中所有活跃批次必须具有有效开工完工时间。"
                    )
                prev_finish = 0.0
                continue

            if b.finish_time > 0.0:
                if b.start_time < prev_finish - 1e-6:
                    errors.append(
                        f"时序冲突: 机器 {mid} 批次 u_id={b.u_id} 开工时间 {b.start_time:.2f} "
                        f"早于前一批次完工时间 {prev_finish:.2f}"
                    )
                if b.finish_time < b.start_time:
                    errors.append(
                        f"时序异常: 批次 u_id={b.u_id} 完工时间小于开工时间"
                    )
                # 独立逐批校验工时模型 p_im 与换型开工递推 (式 6, 7, 16)
                if verify_time_recurrence and order is not None:
                    p_expected = processing_time(order=order, machine=machine)
                    # 校验已解码批次的加工时间与工时模型计算结果一致
                    if abs(b.processing_time - p_expected) > 1e-4:
                        errors.append(
                            f"工时递推偏差: 机器 {mid} 批次 u_id={b.u_id} "
                            f"记录加工时间 {b.processing_time:.4f} 与工时公式计算值 {p_expected:.4f} 不符"
                        )
                    if abs(b.finish_time - (b.start_time + p_expected)) > 1e-4:
                        errors.append(
                            f"完工递推偏差: 机器 {mid} 批次 u_id={b.u_id} "
                            f"完工时间 {b.finish_time:.4f} != 开工 {b.start_time:.4f} + 工时 {p_expected:.4f}"
                        )
                    prev_oid = queue.batches[b_idx - 1].order_id if b_idx > 0 else None
                    s_expected = setup_time(
                        machine=machine,
                        prev_order_id=prev_oid,
                        curr_order_id=b.order_id,
                        is_first_batch=(b_idx == 0),
                    )
                    # 闭合式 (16) 递推验证：B_b^e = max{t_e, r_{o(b)}, C_c^e} + s_{cbm}
                    # 仅针对非冻结批次；冻结批次由专门的冻结签名校验锁死历史物理时刻
                    is_frozen = (frozen_set is not None and b.u_id in frozen_set)
                    if not is_frozen:
                        rel = 0.0
                        if release_times is not None and b.order_id in release_times:
                            rel = float(release_times[b.order_id])
                        elif hasattr(order, "release_time") and order.release_time is not None:
                            rel = float(order.release_time)
                        earliest_ready = max(prev_finish, float(event_time), rel)
                        expected_start = earliest_ready + s_expected
                        if abs(b.start_time - expected_start) > 1e-4:
                            errors.append(
                                f"换型与开工递推偏差 (式 16): 机器 {mid} 批次 u_id={b.u_id} "
                                f"开工时间 {b.start_time:.4f} != max(prev={prev_finish:.4f}, event={float(event_time):.4f}, rel={rel:.4f}) + 换型 {s_expected:.4f} = {expected_start:.4f}"
                            )
                prev_finish = b.finish_time

    # A1: 订单需求精确守恒断言（TC-03）
    for oid, order in orders_dict.items():
        actual_sum = order_stack_sum.get(oid, 0)
        expected_sum = order.W
        if actual_sum != expected_sum:
            errors.append(
                f"TC-03 FAIL: 订单 '{oid}' 需求守恒不成立！"
                f"实际批次叠板单元之和={actual_sum}, 理论需求 W_i={expected_sum} (差值={actual_sum - expected_sum})"
            )

    is_valid = (len(errors) == 0)
    if not is_valid and raise_exception:
        msg = f"解状态验证失败（共 {len(errors)} 项违规）：\n" + "\n".join(f"  * {e}" for e in errors)
        raise ValueError(msg)

    return is_valid, errors


class AtomicTransaction:
    """
    调度解原子事务上下文管理器（规范 9.14 节，TC-08 核心）。

    使用范例：
    ---------
    with AtomicTransaction(solution) as tx:
        # 在原解上执行多步试探操作（如 6 -> 2+2+2 动态拆分）
        ...
        if error_occurred:
            tx.rollback()  # 主动撤销回滚
        else:
            tx.commit()    # 提交改动
    # 若在 with 块内发生任何未捕获异常，将自动执行完整回滚！
    """

    def __init__(self, solution: Solution):
        self.target_solution = solution
        self._snapshot = clone_solution(solution)
        self.is_committed = False
        self.is_rolled_back = False

    def __enter__(self) -> "AtomicTransaction":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # 若发生未捕获异常且尚未提交，自动执行回滚
        if exc_type is not None and not self.is_committed:
            self.rollback()
            return False  # 允许异常继续向上冒泡
        elif not self.is_committed and not self.is_rolled_back:
            # 退出时既未 commit 也未 rollback，默认回滚
            self.rollback()
        return False

    def commit(self) -> None:
        """提交事务，改动正式生效"""
        if self.is_rolled_back:
            raise RuntimeError("无法提交已回滚的事务！")
        self.is_committed = True

    def rollback(self) -> None:
        """回滚事务，瞬间完全恢复进入事务前的完整快照（规范 TC-08）"""
        if self.is_committed:
            raise RuntimeError("无法回滚已提交的事务！")
        if self.is_rolled_back:
            return  # 避免重复回滚

        # 将快照数据完全写回原对象
        self.target_solution.pool_type = self._snapshot.pool_type
        self.target_solution.machine_queues = copy.deepcopy(self._snapshot.machine_queues)
        self.target_solution.active_batches = copy.deepcopy(self._snapshot.active_batches)
        self.target_solution.makespan = self._snapshot.makespan
        self.target_solution.total_setup_time = self._snapshot.total_setup_time
        self.target_solution.spindle_utilization = self._snapshot.spindle_utilization

        self.is_rolled_back = True
