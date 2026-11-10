"""
atomic_transaction.py
=====================
调度原子事务管理器（规范第 10 章、第 18.2 节冻结项 3 & 4）

功能说明：
  1. 封装上层拆批 Option 及其衍生出的下层子批指派动作，提供事务级调度操作。
  2. 提供状态深克隆、试探插入、规范化校验与最早开工工期试算接口。
  3. 支持事务提交（Commit）与回滚（Rollback），避免产生不完整中间状态。
  4. 维护临时采样轨迹，未完成的尝试与无效解不写入 PPO Rollout Buffer。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "environment"))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in [_THIS_DIR, _ENVIRONMENT_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_models import Batch, Machine, MachineQueue, Order, Solution, allocate_uid
from eligibility_builder import ALL_MACHINES
from decoder import full_decode as decode
from solution_lifecycle import normalize_solution


class OptionTransaction:
    """
    单个上层拆批 Option 的调度原子事务。
    
    生命周期：
      __init__ -> step_assign (遍历 |p| 个子批) -> finalize (试算工期与校验) -> commit 或 rollback
    """

    def __init__(
        self,
        base_solution: Solution,
        order: Order,
        template: Tuple[int, ...],
        orders_dict: Dict[str, Order],
        machines_dict: Dict[str, Machine] = ALL_MACHINES,
        event_time: float = 0.0,
        frozen_uids: Optional[Iterable[int]] = None,
    ):
        self.base_solution = base_solution
        self.order = order
        self.template = template
        self.orders_dict = orders_dict
        self.machines_dict = machines_dict
        self.pool_type = base_solution.pool_type
        self.event_time = float(event_time)
        self.frozen_uids = frozenset(int(uid) for uid in frozen_uids) if frozen_uids is not None else frozenset()

        # 1. 解快照深克隆
        self.snapshot: Solution = base_solution.copy()

        # 2. 为模板中的每个子批预分配全局稳定递增 u_id
        self.sub_batches: List[Batch] = [
            Batch(
                order_id=order.order_id,
                u_id=allocate_uid(),
                stack_count=q,
                machine_id="",
            )
            for q in template
        ]

        # 3. 事务内临时轨迹（与 PPO 经验池严格物理隔离）
        self.temporary_worker_trajectory: List[dict] = []
        self.current_batch_idx = 0
        self.is_finalized = False
        self.is_committed = False
        self.is_rolled_back = False

    @property
    def total_sub_batches(self) -> int:
        return len(self.template)

    def get_current_batch(self) -> Optional[Batch]:
        if self.current_batch_idx < self.total_sub_batches:
            return self.sub_batches[self.current_batch_idx]
        return None

    def get_valid_slots_for_batch(self, batch: Batch) -> List[Tuple[str, int]]:
        """
        获取当前子批在当前解快照中的所有合法插入插槽 (machine_id, slot_pos)。
        
        硬判据（冻结项 3）：
          1. 机床与当前资源池同池；
          2. 机床在订单合格机床列表 eligible_machines 中；
          3. 机床主轴容量 C_m >= batch.stack_count；
          4. slot_pos in [0, len(queue)]。
        """
        valid_slots: List[Tuple[str, int]] = []
        for mid, queue in self.snapshot.machine_queues.items():
            mach = self.machines_dict.get(mid)
            if mach is None or mach.pool_type != self.pool_type:
                continue
            if mid not in self.order.eligible_machines:
                continue
            if mach.spindles < batch.stack_count:
                continue
            # 该机床合法插槽范围 [tail, len(queue.batches)]，不可插入冻结前缀内部
            tail = 0
            if self.frozen_uids:
                frozen_positions = [
                    idx for idx, b in enumerate(queue.batches) if b.u_id in self.frozen_uids
                ]
                tail = max(frozen_positions, default=-1) + 1
            for pos in range(tail, len(queue.batches) + 1):
                valid_slots.append((mid, pos))
        return valid_slots

    def step_assign(
        self,
        machine_id: str,
        slot_pos: int,
        step_meta: Optional[dict] = None,
    ) -> bool:
        """
        在解快照中试探插入当前子批。
        
        返回 True 表示插入格式合法，False 表示参数违规触发即时回滚。
        """
        if self.current_batch_idx >= self.total_sub_batches:
            return False

        curr_batch = self.sub_batches[self.current_batch_idx]
        mach = self.machines_dict.get(machine_id)
        if mach is None or mach.pool_type != self.pool_type:
            return False
        if machine_id not in self.order.eligible_machines:
            return False
        if mach.spindles < curr_batch.stack_count:
            return False

        queue = self.snapshot.machine_queues[machine_id]
        tail = 0
        if self.frozen_uids:
            frozen_positions = [
                idx for idx, b in enumerate(queue.batches) if b.u_id in self.frozen_uids
            ]
            tail = max(frozen_positions, default=-1) + 1
        if slot_pos < tail or slot_pos > len(queue.batches):
            return False

        # 赋值 machine_id 并插入队列指定插槽
        curr_batch.machine_id = machine_id
        queue.insert_at(slot_pos, curr_batch)

        # 记录临时微观转移
        transition = {
            "order_id": self.order.order_id,
            "sub_batch_idx": self.current_batch_idx,
            "u_id": curr_batch.u_id,
            "stack_count": curr_batch.stack_count,
            "machine_id": machine_id,
            "slot_pos": slot_pos,
        }
        if step_meta:
            transition.update(step_meta)
        self.temporary_worker_trajectory.append(transition)

        self.current_batch_idx += 1
        return True

    def validate_snapshot_integrity(self) -> Tuple[bool, List[str]]:
        """
        轻量级校验快照解的物理完整性（无副作用）：
          1. 主轴容量不超限；
          2. 机器资格满足；
          3. 活跃 u_id 唯一无重复；
          4. 资源池同池隔离。
        """
        errors = []
        seen_uids = set()
        for mid, queue in self.snapshot.machine_queues.items():
            mach = self.machines_dict.get(mid)
            if not mach:
                errors.append(f"未知机器: {mid}")
                continue
            for b in queue.batches:
                if b.u_id in seen_uids:
                    errors.append(f"重复批次 u_id: {b.u_id}")
                seen_uids.add(b.u_id)
                if b.stack_count > mach.spindles:
                    errors.append(f"批次 {b.u_id} 叠数 {b.stack_count} 超过机器 {mid} 容量 {mach.spindles}")
                b_order = self.orders_dict.get(b.order_id)
                if not b_order or mid not in b_order.eligible_machines:
                    errors.append(f"机器 {mid} 不具备加工订单 {b.order_id} 资格")
        return (len(errors) == 0, errors)

    def commit(self) -> Tuple[bool, Solution, List[dict]]:
        """
        原子提交事务。
        
        成功条件：
          1. 全部 |p| 个子批均已完成指派；
          2. 物理完整性断言 100% 通过；
          3. 规范化映射成功；
          4. 最早开工解码器无异常推导工期。
        
        返回：
          (success: bool, solution: Solution, committed_trajectory: List[dict])
        """
        if self.is_committed or self.is_rolled_back:
            raise RuntimeError("Transaction has already been finalized!")

        if self.current_batch_idx != self.total_sub_batches:
            self.rollback()
            return False, self.base_solution, []

        is_valid, errors = self.validate_snapshot_integrity()
        if not is_valid:
            self.rollback()
            return False, self.base_solution, []

        try:
            # 规范化映射与工期试算
            normalize_solution(self.snapshot)
            decode(
                self.snapshot,
                self.orders_dict,
                self.machines_dict,
                event_time=self.event_time,
                frozen_uids=self.frozen_uids,
            )
        except Exception:
            self.rollback()
            return False, self.base_solution, []

        # 全部通过，原子提交生效
        self.is_finalized = True
        self.is_committed = True
        committed_traj = list(self.temporary_worker_trajectory)
        return True, self.snapshot, committed_traj

    def rollback(self) -> Solution:
        """
        原子回滚事务：物理清空临时轨迹，彻底销毁快照，恢复事务前基准解。
        """
        self.is_finalized = True
        self.is_rolled_back = True
        self.temporary_worker_trajectory.clear()
        return self.base_solution
