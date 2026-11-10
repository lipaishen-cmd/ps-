"""
decoder.py
==========
统一全量最早开工解码器（Earliest-Start Decoder，式 6, 7, 16）

功能说明：
  1. 接收一个包含机器队列的调度解对象（Solution）。
  2. 严格按机器队列顺序执行最早开工解码（Earliest-Start）。
  3. 执行首批次强制初始换型（式 7 & 假设 2、10）。
  4. 严格调用唯一的 time_model 计算批次加工时间 P(i,m) 与换型时间 setup_time。
  5. 填充并更新每个批次的 start_time、processing_time、finish_time。
  6. 构建并填充各机器队列的完整前缀完工时间缓存（prefix_cache），供增量解码器复用。
  7. 闭合式 (16) 动态事件与订单到达约束：
     B_b^e = max{t_e, r_{o(b)}, C_c^e} + sigma_{cb}^m,  C_b^e = B_b^e + p_{o(b),m}
  8. 计算并更新解的最大完工时间（Makespan）、总换型时间、主轴利用率等核心指标。
  9. 同步更新 solution.active_batches 索引字典。

【时序规则与约束】
  - 禁止引入人工滞后或机器闲置时间，批次完成换型后立即开始加工（Earliest-Start）。
  - 禁止将批次装载量 B_batch 乘入加工时间。
  - 机器首批次必须计入机器特定换型时间（s0bm = Smchg）。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

# 动态引入第 1 层基础模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = _THIS_DIR
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in [_ENVIRONMENT_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_models import Batch, Machine, MachineQueue, Order, Solution
from eligibility_builder import ALL_MACHINES
from time_model import processing_time, setup_time


def decode_machine(
    queue: MachineQueue,
    orders_dict: Dict[str, Order],
    machine: Machine,
    event_time: float = 0.0,
    release_times: Optional[Dict[str, float]] = None,
    frozen_uids: Optional[Iterable[int]] = None,
) -> float:
    """
    对单台机器的作业队列执行全量最早开工递推（式 6, 7, 16）。

    参数
    ----
    queue         : MachineQueue — 目标机器队列
    orders_dict   : Dict[str, Order] — 订单字典（用于查询工件孔数、工艺等）
    machine       : Machine — 机器对象
    event_time    : float — 动态事件触发时刻 t_e（默认 0.0）
    release_times : Optional[Dict[str, float]] — 订单释放时间 r_{o(b)} 字典（默认 None）
    frozen_uids   : Optional[Iterable[int]] — 冻结批次 UID 集合（这些批次的工期与开工时间原样保留）

    返回
    ----
    该机器的最终完工时间（分钟，float）。
    同时就地修改批次的 start_time, processing_time, finish_time，
    并更新 queue.prefix_cache 缓存列表。
    """
    if queue.is_empty():
        queue.prefix_cache = []
        return 0.0

    current_time = 0.0
    prev_order_id: Optional[str] = None
    prefix_cache: List[float] = []
    frozen_set = frozenset(int(uid) for uid in frozen_uids) if frozen_uids is not None else None

    for k, batch in enumerate(queue.batches):
        is_first = (k == 0)

        # 1. 冻结判定：若批次属于明确的冻结集合，原样保持物理时序不变
        if frozen_set is not None and batch.u_id in frozen_set:
            prefix_cache.append(float(batch.finish_time))
            current_time = float(batch.finish_time)
            prev_order_id = batch.order_id
            continue

        # 未显式传入 frozen_uids 时，事件前已完工批次仍视为冻结
        if (
            frozen_set is None
            and event_time > 0.0
            and batch.finish_time > 0.0
            and batch.finish_time <= event_time + 1e-9
        ):
            prefix_cache.append(float(batch.finish_time))
            current_time = float(batch.finish_time)
            prev_order_id = batch.order_id
            continue

        # 2. 计算换型时间（首批强制初设，同订单为 0，跨订单换型）
        s = setup_time(
            machine=machine,
            prev_order_id=prev_order_id,
            curr_order_id=batch.order_id,
            is_first_batch=is_first,
        )

        # 3. 查询订单并调用工时模型计算加工时间
        order = orders_dict[batch.order_id]
        p = processing_time(order=order, machine=machine)

        # 4. 最早开工递推（闭合式 16：B_b^e = max{t_e, r_{o(b)}, C_c^e} + sigma_{cb}^m）
        rel = 0.0
        if release_times is not None and batch.order_id in release_times:
            rel = float(release_times[batch.order_id])
        elif hasattr(order, "release_time") and order.release_time is not None:
            rel = float(order.release_time)

        earliest_ready = max(current_time, float(event_time), rel)
        batch.start_time = earliest_ready + s
        batch.processing_time = p
        batch.finish_time = batch.start_time + p

        # 5. 推进机器时钟与前缀完工缓存
        current_time = batch.finish_time
        prefix_cache.append(current_time)

        # 6. 更新前驱订单编号
        prev_order_id = batch.order_id

    queue.prefix_cache = prefix_cache
    return current_time


def full_decode(
    solution: Solution,
    orders_dict: Dict[str, Order],
    machines_dict: Optional[Dict[str, Machine]] = None,
    event_time: float = 0.0,
    release_times: Optional[Dict[str, float]] = None,
    frozen_uids: Optional[Iterable[int]] = None,
) -> float:
    """
    对整个调度解执行全量最早开工解码（式 6, 7, 16）。

    参数
    ----
    solution      : Solution — 待解码的调度解
    orders_dict   : Dict[str, Order] — 订单注册字典
    machines_dict : Optional[Dict[str, Machine]] — 机器注册表，默认采用全局 ALL_MACHINES
    event_time    : float — 动态事件触发时刻 t_e（默认 0.0）
    release_times : Optional[Dict[str, float]] — 订单到达释放时刻字典（默认 None）

    返回
    ----
    当前资源池的最大完工时间 Makespan（float）。
    同时就地更新 solution 的 makespan、total_setup_time、spindle_utilization 及 active_batches。
    """
    if machines_dict is None:
        machines_dict = ALL_MACHINES

    total_setup_time = 0.0
    total_spindle_active_time = 0.0  # 主轴实际作业分钟数 (∑ b.stack_count * b.proc_time)
    total_spindles_in_pool = 0
    machine_finish_times: List[float] = []
    active_batches: Dict[int, Batch] = {}

    for machine_id, queue in solution.machine_queues.items():
        machine = machines_dict[machine_id]
        total_spindles_in_pool += machine.spindles

        # 单机全量解码（接收 event_time 与 release_times 与 frozen_uids）
        m_finish = decode_machine(
            queue,
            orders_dict,
            machine,
            event_time=event_time,
            release_times=release_times,
            frozen_uids=frozen_uids,
        )
        machine_finish_times.append(m_finish)

        # 收集该机上的批次统计量
        prev_order_id: Optional[str] = None
        for k, b in enumerate(queue.batches):
            is_first = (k == 0)
            s = setup_time(machine, prev_order_id, b.order_id, is_first)
            total_setup_time += s
            total_spindle_active_time += b.stack_count * b.processing_time
            prev_order_id = b.order_id

            # 注册活跃批次
            active_batches[b.u_id] = b

    # 计算资源池最大完工时间 Makespan
    makespan = max(machine_finish_times) if machine_finish_times else 0.0
    solution.makespan = makespan
    solution.total_setup_time = total_setup_time
    solution.active_batches = active_batches

    # 计算全池平均主轴利用率
    if makespan > 0.0 and total_spindles_in_pool > 0:
        total_capacity_time = total_spindles_in_pool * makespan
        solution.spindle_utilization = total_spindle_active_time / total_capacity_time
    else:
        solution.spindle_utilization = 0.0

    return makespan
