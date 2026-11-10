"""
dynamic_event_manager.py
========================
事件驱动动态重调度基础层。

这个模块把“事件发生”和“策略作出决策”明确分开：

* :class:`DynamicEventManager` 只推进动态插单到达事件、维护当前解和冻结边界；
* :class:`RescheduleTransaction` 只允许对未冻结部分试排，并执行原子提交/回滚；
* 主训练/部署链路不构造、不读取、不回退到任何参考解；
* 文件末尾的独立基线构造器只供离线对比实验显式调用，主运行器不得调用。

两个 Actor 后续只需从 ``DynamicRescheduleContext`` 读取状态并返回候选解，
不能直接修改 ``current_solution``。因此该模块先把动态场景的安全边界固定下来，
再接入共享 Critic 与训练流程。
"""

from __future__ import annotations

import copy
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "environment"))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_THIS_DIR, _ENVIRONMENT_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_models import (  # noqa: E402
    Batch,
    DynamicInsertEvent,
    Machine,
    Order,
    POOL_B,
    POOL_D,
    Solution,
    allocate_uid,
)
from eligibility_builder import ALL_MACHINES  # noqa: E402
from partition_template_generator import generate_partition_templates  # noqa: E402
from solution_lifecycle import (  # noqa: E402
    create_empty_solution,
    normalize_solution,
    validate_solution,
)
from atomic_transaction import OptionTransaction  # noqa: E402
from decoder import full_decode
from time_model import setup_time, processing_time  # noqa: E402


class InfeasibleHeuristicError(RuntimeError):
    """独立离线启发式在当前约束与搜索空间内无法生成可行解。"""
    pass


DynamicEvent = DynamicInsertEvent


# 初始排程是事件前的确定性 SPT 调度结果，不属于任一学习策略的动作。
# 动态事件发生后的分批和机台—槽位决策仍全部由两个 Actor 生成；
# SPT 不进入动态状态、奖励、修复或失败回退链路。
INITIAL_SOLUTION_DISPATCH_RULE = "SPT"
INITIAL_SOLUTION_BUILDER_VERSION = "capacity-aware-spt-full-decoder-v2"
_INITIAL_SOLUTION_BLUEPRINT_CACHE: Dict[Tuple[object, ...], Solution] = {}


@dataclass(frozen=True)
class MakespanComparison:
    """候选解与参考/事件前解的 Cmax 比较，用于两个 Actor 的共同目标。"""

    baseline_cmax: float
    candidate_cmax: float

    @property
    def delta_cmax(self) -> float:
        return self.candidate_cmax - self.baseline_cmax

    @property
    def improved(self) -> bool:
        return self.delta_cmax < -1e-9


def compare_makespan(candidate: Solution, baseline: Solution) -> MakespanComparison:
    """返回 ΔCmax=Cmax(candidate)-Cmax(baseline)，不引入交期/拖期目标。"""
    return MakespanComparison(
        baseline_cmax=float(baseline.makespan),
        candidate_cmax=float(candidate.makespan),
    )


@dataclass(frozen=True)
class FreezeSnapshot:
    """事件时刻的冻结边界，严格采用 F_e=B_done ∪ B_processing ∪ B_setup_started。"""

    as_of: float
    done_uids: frozenset[int]
    processing_uids: frozenset[int]
    setup_started_uids: frozenset[int]
    frozen_uids: frozenset[int]


@dataclass
class DynamicRescheduleContext:
    """一次事件触发后交给两个 Actor 的只读语义上下文。"""

    event: DynamicEvent
    event_time: float
    base_solution: Solution
    orders: Dict[str, Order]
    freeze: FreezeSnapshot
    new_order_ids: Tuple[str, ...]
    k_scope: int = 8
    k_seed: int = 5
    enable_adjacency_closure: bool = True
    affected_uids: frozenset[int] = frozenset()
    stable_uids: frozenset[int] = frozenset()
    stable_signature: Dict[int, Tuple] = field(default_factory=dict)
    scope_ratio: float = 0.0

    @property
    def frozen_uids(self) -> frozenset[int]:
        return self.freeze.frozen_uids

    @property
    def movable_uids(self) -> frozenset[int]:
        if self.stable_uids or self.affected_uids or self.k_scope == 0:
            return self.affected_uids
        all_uids = frozenset(self.base_solution.get_all_active_uids())
        return all_uids.difference(self.freeze.frozen_uids)


def _require_time(value: object, field_name: str, event_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"事件 {event_id}: {field_name} 必须为有限非负数。") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"事件 {event_id}: {field_name} 必须为有限非负数。")
    return number


def _queue_end(queue) -> float:
    """从缓存或已解码批次取得机器当前队尾时刻。"""
    if queue.prefix_cache:
        return float(queue.prefix_cache[-1])
    return max((float(b.finish_time) for b in queue.batches), default=0.0)


def _batch_signature(solution: Solution, uids: Iterable[int]) -> Tuple[Dict[int, Tuple], List[str]]:
    """提取批次归属/物理量/已解码时刻，重复或缺失均报告。"""
    wanted = set(uids)
    found: Dict[int, Tuple] = {}
    errors: List[str] = []
    for mid, queue in solution.machine_queues.items():
        for batch in queue.batches:
            if batch.u_id not in wanted:
                continue
            sig = (
                mid,
                batch.order_id,
                batch.stack_count,
                float(batch.start_time),
                float(batch.finish_time),
                float(batch.processing_time),
            )
            if batch.u_id in found:
                errors.append(f"冻结批次 u_id={batch.u_id} 在候选解中重复出现。")
            else:
                found[batch.u_id] = sig
    missing = sorted(wanted.difference(found))
    errors.extend(f"冻结批次 u_id={uid} 在候选解中缺失。" for uid in missing)
    return found, errors


def compute_freeze_snapshot(
    solution: Solution,
    as_of: float,
    setup_started_uids: Iterable[int] = (),
    tolerance: float = 1e-9,
    machines_dict: Optional[Dict[str, Machine]] = None,
) -> FreezeSnapshot:
    """按事件时刻对已完成、加工中和已开始换型的批次建立冻结集合。"""
    t = _require_time(as_of, "as_of", "freeze")
    explicit_setup = frozenset(int(uid) for uid in setup_started_uids)
    done: Set[int] = set()
    processing: Set[int] = set()
    auto_setup: Set[int] = set()
    known_uids: Set[int] = set()

    m_dict = machines_dict if machines_dict is not None else ALL_MACHINES

    for mid, queue in solution.machine_queues.items():
        prev_oid: Optional[str] = None
        for k, batch in enumerate(queue.batches):
            known_uids.add(batch.u_id)
            start = float(batch.start_time)
            finish = float(batch.finish_time)
            # 未解码批次的默认 (0,0) 不得在 t>=0 时被误判为已完成。
            if finish > tolerance and finish <= t + tolerance:
                done.add(batch.u_id)
            elif finish > start + tolerance and start <= t + tolerance < finish:
                processing.add(batch.u_id)
            elif finish > start + tolerance and start > t + tolerance:
                # 规范要求：自动识别在 t_e 时刻已进入换型准备的批次 (B_b - s_m <= t_e < B_b)
                mach = m_dict.get(mid)
                if mach is not None:
                    s_m = setup_time(
                        machine=mach,
                        prev_order_id=prev_oid,
                        curr_order_id=batch.order_id,
                        is_first_batch=(k == 0),
                    )
                    if s_m > tolerance and (start - s_m) <= t + tolerance:
                        auto_setup.add(batch.u_id)
            prev_oid = batch.order_id

    unknown_setup = explicit_setup.difference(known_uids)
    if unknown_setup:
        raise ValueError(f"setup_started_uids 包含当前解不存在的批次: {sorted(unknown_setup)}")
    setup = explicit_setup | auto_setup
    frozen = frozenset(done | processing | setup)
    return FreezeSnapshot(
        as_of=t,
        done_uids=frozenset(done),
        processing_uids=frozenset(processing),
        setup_started_uids=frozenset(setup),
        frozen_uids=frozen,
    )


class RescheduleTransaction:
    """动态事件候选解的验证、原子提交与无痕回滚。"""

    def __init__(self, manager: "DynamicEventManager", context: DynamicRescheduleContext):
        self.manager = manager
        self.context = context
        self.base_solution = context.base_solution.copy()
        self._base_frozen_signature, self._base_signature_errors = _batch_signature(
            self.base_solution, context.frozen_uids
        )
        self.is_committed = False
        self.is_rolled_back = False

    def validate(self, candidate: Solution) -> List[str]:
        if self.is_committed or self.is_rolled_back:
            return ["事务已经结束，不能再次验证。"]
        if candidate is None:
            return ["候选解不能为空。"]
        errors = list(self._base_signature_errors)
        if candidate.pool_type != self.manager.pool_type:
            errors.append(
                f"候选解资源池 {candidate.pool_type} 与事件管理器 {self.manager.pool_type} 不一致。"
            )

        candidate = candidate.copy()
        normalize_solution(candidate)
        candidate_frozen_signature, sig_errors = _batch_signature(candidate, self.context.frozen_uids)
        errors.extend(sig_errors)
        for uid, expected in self._base_frozen_signature.items():
            actual = candidate_frozen_signature.get(uid)
            if actual is None:
                errors.append(f"冻结批次 u_id={uid} 在候选解中丢失；冻结集合必须原样保留。")
                continue
            if actual[:3] != expected[:3] or any(
                not math.isclose(actual[i], expected[i], rel_tol=0.0, abs_tol=1e-7)
                for i in (3, 4, 5)
            ):
                errors.append(
                    f"冻结批次 u_id={uid} 的机器、订单、叠数或时间被修改；"
                    "冻结集合只能原样保留。"
                )


        # 稳定补集刚性结构签名校验 (式 10): 所属机台、订单、叠数及机台内相对先后次序严格不变
        if self.context.stable_signature:
            from scope_selector import compute_stable_signature
            cand_stable_sig = compute_stable_signature(candidate, self.context.stable_uids)
            for uid, expected in self.context.stable_signature.items():
                actual = cand_stable_sig.get(uid)
                if actual is None:
                    errors.append(f"稳定补集批次 u_id={uid} 在候选解中丢失。")
                elif actual != expected:
                    errors.append(
                        f"稳定补集批次 u_id={uid} 结构签名被篡改：预期={expected}，实际={actual}。"
                    )

        # 任何非冻结批次的开工时间不得早于 max(event_time, order_release_time)
        for mid, queue in candidate.machine_queues.items():
            for batch in queue.batches:
                if batch.u_id not in self.context.frozen_uids:
                    order = self.context.orders.get(batch.order_id)
                    r_o = float(order.release_time) if order and hasattr(order, "release_time") and order.release_time is not None else 0.0
                    min_start = max(float(self.context.event_time), r_o)
                    if float(batch.start_time) < min_start - 1e-7:
                        errors.append(
                            f"非冻结批次 u_id={batch.u_id} 开工时间 {float(batch.start_time):.3f} "
                            f"早于事件或订单释放时刻 {min_start:.3f}。"
                        )

        valid, physical_errors = validate_solution(
            candidate,
            self.context.orders,
            self.manager.machines_dict,
            raise_exception=False,
            event_time=self.context.event_time,
            frozen_uids=self.context.frozen_uids,
            require_decoded=True,
        )
        if not valid:
            errors.extend(physical_errors)
        return errors

    def commit(self, candidate: Solution) -> Tuple[bool, Solution, List[str]]:
        errors = self.validate(candidate)
        if errors:
            self.rollback()
            return False, self.manager.current_solution, errors
        committed = candidate.copy()
        normalize_solution(committed)
        self.manager._commit_context(self.context, committed)
        self.is_committed = True
        return True, committed, []

    def rollback(self) -> Solution:
        if self.is_committed:
            raise RuntimeError("无法回滚已提交的动态事务。")
        self.is_rolled_back = True
        return self.manager.current_solution


class ReallocationTransaction:
    """仅将受影响域旧批次与新插单子批交给两个 Actor 的离散事务。"""

    def __init__(
        self,
        manager: "DynamicEventManager",
        context: DynamicRescheduleContext,
        staged_batches: Sequence[Batch],
        replaced_uids: Iterable[int] = (),
        progressive: bool = False,
    ):
        self.manager = manager
        self.context = context
        self.snapshot = context.base_solution.copy()
        self.frozen_uids = set(context.frozen_uids)
        # 稳定补集 context.stable_uids 在工作快照中保持机台与拓扑序不变，开工/完工时间由解码器重新计算
        self.stable_uids = set(self.context.stable_uids)
        # 深拷贝待放置批次，避免与 context.base_solution 共享可变对象引发引用污染
        self.progressive = bool(progressive)
        self.sub_batches: List[Batch] = [copy.deepcopy(batch) for batch in staged_batches]
        self.total_sub_batches = len(self.sub_batches)
        self.current_batch_idx = 0
        self.temporary_worker_trajectory: List[dict] = []
        self.is_committed = False
        self.is_rolled_back = False
        self.replaced_uids = set(int(uid) for uid in replaced_uids)

        existing_uids = {
            batch.u_id
            for queue in self.snapshot.machine_queues.values()
            for batch in queue.batches
        }
        movable_uids = set(self.context.movable_uids)
        self.original_movable_batches: Dict[int, Batch] = {
            batch.u_id: copy.deepcopy(batch)
            for queue in self.snapshot.machine_queues.values()
            for batch in queue.batches
            if batch.u_id in movable_uids
        }
        self._partitioned_order_ids: Set[str] = set()
        if not self.replaced_uids.issubset(movable_uids):
            raise ValueError("重排事务 replaced_uids 必须全部属于受影响域未冻结批次。")
        staged_existing_uids = {
            batch.u_id for batch in self.sub_batches if batch.u_id in existing_uids
        }
        expected_existing_uids = movable_uids.difference(self.replaced_uids)
        if not self.progressive and staged_existing_uids != expected_existing_uids:
            missing = sorted(expected_existing_uids.difference(staged_existing_uids))
            extra = sorted(staged_existing_uids.difference(expected_existing_uids))
            raise ValueError(
                f"重排事务必须完整覆盖受影响域批次；缺失={missing}，非法/重复={extra}。"
            )
        if len({batch.u_id for batch in self.sub_batches}) != self.total_sub_batches:
            raise ValueError("重排事务 staged_batches 存在重复 u_id。")

        # 仅从工作快照移除受影响域批次；冻结批次与稳定补集批次原样保留。
        removed: Set[int] = set()
        for queue in self.snapshot.machine_queues.values():
            kept: List[Batch] = []
            for batch in queue.batches:
                if batch.u_id in movable_uids:
                    removed.add(batch.u_id)
                else:
                    kept.append(batch)
            queue.batches = kept
        if removed != movable_uids:
            raise ValueError("重排事务移除受影响域批次时发现解状态不一致。")
        normalize_solution(self.snapshot)
        # 移除受影响域批次后，立即解码刷新快照的各机队列 prefix_cache 与稳定补集时间
        full_decode(
            self.snapshot,
            self.context.orders,
            self.manager.machines_dict,
            event_time=self.context.event_time,
            frozen_uids=self.context.frozen_uids,
        )

    @staticmethod
    def _validate_template(order: Order, template: Sequence[int]) -> Tuple[int, ...]:
        if (
            not template
            or any(isinstance(q, bool) or int(q) != q or int(q) <= 0 for q in template)
            or sum(int(q) for q in template) != order.W
        ):
            raise ValueError(f"订单 {order.order_id} 模板必须为正整数划分且精确守恒 W。")
        return tuple(int(q) for q in template)

    def append_partition_option(self, order: Order, template: Sequence[int]) -> Tuple[int, int]:
        """把一个 Split 的子批追加为紧邻 Placement 序列，返回半开区间 [start, end)。"""
        if not self.progressive:
            raise RuntimeError("只有渐进重排事务允许追加 Split Option。")
        if self.current_batch_idx != self.total_sub_batches:
            raise RuntimeError("上一 Split 的 Placement 尚未完成，不能选择下一个 Split。")
        if order.order_id in self._partitioned_order_ids:
            raise ValueError(f"订单 {order.order_id} 已选择过 Split 模板。")
        normalized = self._validate_template(order, template)
        order_uids = {
            batch.u_id
            for queue in self.context.base_solution.machine_queues.values()
            for batch in queue.batches
            if batch.order_id == order.order_id
        }
        if order_uids.intersection(self.context.frozen_uids):
            raise ValueError(f"订单 {order.order_id} 含冻结批次，不能整体重划分。")
        if order_uids.intersection(self.context.stable_uids):
            raise ValueError(f"订单 {order.order_id} 含稳定补集批次，不能整体重划分。")
        movable_order_uids = order_uids.intersection(self.context.movable_uids)
        if order_uids and movable_order_uids != order_uids:
            raise ValueError(f"订单 {order.order_id} 并非全部属于受影响域，不能整体重划分。")

        start = self.total_sub_batches
        self.replaced_uids.update(movable_order_uids)
        self.sub_batches.extend(
            Batch(order_id=order.order_id, u_id=allocate_uid(), stack_count=q, machine_id="")
            for q in normalized
        )
        self.total_sub_batches = len(self.sub_batches)
        self._partitioned_order_ids.add(order.order_id)
        return start, self.total_sub_batches

    def append_remaining_fixed_batches(self) -> Tuple[int, int]:
        """在最后一个 Split 后追加不可重分批的受影响旧批次尾段。"""
        if not self.progressive:
            raise RuntimeError("只有渐进重排事务允许追加固定模板尾段。")
        if self.current_batch_idx != self.total_sub_batches:
            raise RuntimeError("当前 Split 的 Placement 尚未完成。")
        already_staged = {batch.u_id for batch in self.sub_batches}
        remaining = [
            copy.deepcopy(batch)
            for uid, batch in self.original_movable_batches.items()
            if uid not in self.replaced_uids and uid not in already_staged
        ]
        priority_rank = {"高": 0, "中": 1, "普通": 2}

        def key(batch: Batch):
            order = self.context.orders[batch.order_id]
            raw_due = getattr(order, "due_date", None)
            return (
                priority_rank.get(str(getattr(order, "priority", "普通")), 9),
                float(getattr(order, "release_time", 0.0) or 0.0),
                float(raw_due) if raw_due is not None else float("inf"),
                order.order_id,
                -batch.stack_count,
                batch.u_id,
            )

        remaining.sort(key=key)
        start = self.total_sub_batches
        self.sub_batches.extend(remaining)
        self.total_sub_batches = len(self.sub_batches)
        return start, self.total_sub_batches

    def decode_partial(self) -> None:
        """在两个 Split 决策之间刷新完整临时排程，形成真实下一宏观状态。"""
        if self.current_batch_idx != self.total_sub_batches:
            raise RuntimeError("尚有未放置子批，不能形成下一宏观状态。")
        normalize_solution(self.snapshot)
        full_decode(
            self.snapshot,
            self.context.orders,
            self.manager.machines_dict,
            event_time=self.context.event_time,
            frozen_uids=self.context.frozen_uids,
        )

    def _validate_complete_coverage(self) -> None:
        movable_uids = set(self.context.movable_uids)
        existing_staged = {
            batch.u_id for batch in self.sub_batches if batch.u_id in self.original_movable_batches
        }
        expected = movable_uids.difference(self.replaced_uids)
        if existing_staged != expected:
            missing = sorted(expected.difference(existing_staged))
            extra = sorted(existing_staged.difference(expected))
            raise ValueError(f"渐进事务未完整覆盖受影响域；缺失={missing}，非法={extra}。")
        if len({batch.u_id for batch in self.sub_batches}) != len(self.sub_batches):
            raise ValueError("渐进事务 staged_batches 存在重复 u_id。")

    def get_current_batch(self) -> Optional[Batch]:
        if self.current_batch_idx >= self.total_sub_batches:
            return None
        return self.sub_batches[self.current_batch_idx]

    def _frozen_tail(self, machine_id: str) -> int:
        """返回机器队列中最后一个冻结批次之后的第一个可用插槽索引。"""
        queue = self.snapshot.machine_queues[machine_id]
        frozen_positions = [
            idx for idx, batch in enumerate(queue.batches) if batch.u_id in self.frozen_uids
        ]
        return max(frozen_positions, default=-1) + 1

    def get_valid_slots_for_batch(self, batch: Batch) -> List[Tuple[str, int]]:
        valid: List[Tuple[str, int]] = []
        order = self.context.orders.get(batch.order_id)
        if order is None:
            return valid
        for mid, queue in self.snapshot.machine_queues.items():
            machine = self.manager.machines_dict.get(mid)
            if machine is None or machine.pool_type != self.manager.pool_type:
                continue
            if mid not in order.eligible_machines or machine.spindles < batch.stack_count:
                continue
            # 严格依据参数规范 1.6/4.6 节：允许受影响批次进入冻结前缀之后的所有合法槽位，
            # 插入操作保持稳定补集批次间的相对先后顺序（拓扑不变），由解码器重新计算时间。
            tail = self._frozen_tail(mid)
            valid.extend((mid, pos) for pos in range(tail, len(queue.batches) + 1))
        return valid

    def step_assign(self, machine_id: str, slot_pos: int) -> bool:
        batch = self.get_current_batch()
        if batch is None:
            return False
        machine = self.manager.machines_dict.get(machine_id)
        if machine is None or machine.pool_type != self.manager.pool_type:
            return False
        order = self.context.orders.get(batch.order_id)
        if order is None or machine_id not in order.eligible_machines:
            return False
        if machine.spindles < batch.stack_count:
            return False
        queue = self.snapshot.machine_queues[machine_id]
        if slot_pos < self._frozen_tail(machine_id) or slot_pos > len(queue.batches):
            return False
        # 放置时同步更新批次的机器编号
        batch.machine_id = machine_id
        queue.insert_at(slot_pos, batch)
        self.temporary_worker_trajectory.append(
            {
                "order_id": batch.order_id,
                "u_id": batch.u_id,
                "stack_count": batch.stack_count,
                "machine_id": machine_id,
                "slot_pos": slot_pos,
                "sub_batch_idx": self.current_batch_idx,
            }
        )
        self.current_batch_idx += 1
        return True

    def commit(self) -> Tuple[bool, Solution, List[dict]]:
        if self.is_committed or self.is_rolled_back:
            raise RuntimeError("重排事务已经结束。")
        if self.current_batch_idx != self.total_sub_batches:
            self.rollback()
            return False, self.context.base_solution, []
        try:
            self._validate_complete_coverage()
            normalize_solution(self.snapshot)
            # 统一物理解码：仅真正冻结的批次传入 frozen_uids；稳定补集时间由式 (16) 重新递推
            full_decode(
                self.snapshot,
                self.context.orders,
                self.manager.machines_dict,
                event_time=self.context.event_time,
                frozen_uids=self.context.frozen_uids,
            )
        except Exception:
            self.rollback()
            return False, self.context.base_solution, []
        self.is_committed = True
        return True, self.snapshot, list(self.temporary_worker_trajectory)

    def rollback(self) -> Solution:
        self.is_rolled_back = True
        self.temporary_worker_trajectory.clear()
        return self.context.base_solution


class DynamicEventManager:
    """单资源池事件驱动管理器；策略网络只能通过事务提交候选解。

    ``formal_mode=True`` 限定当前正式动态实验为 ``DynamicInsertEvent``。
    设备故障仍保留为独立代码扩展与单元测试能力，但不能混入本轮
    ``PRE_FORMAL_ONLY_PROXY_DATA`` 的正式插单实验。
    """

    def __init__(
        self,
        pool_type: str,
        initial_orders: Sequence[Order],
        events: Sequence[DynamicEvent] = (),
        initial_solution: Optional[Solution] = None,
        machines_dict: Dict[str, Machine] = ALL_MACHINES,
        setup_started_uids: Iterable[int] = (),
        instance_name: Optional[str] = None,
        k_scope: int = 8,
        k_seed: int = 5,
        delta_k: int = 4,
        formal_mode: bool = False,
        enable_adjacency_closure: bool = True,
    ):
        if pool_type not in (POOL_D, POOL_B):
            raise ValueError(f"未知资源池类型 '{pool_type}'。")
        self.pool_type = pool_type
        self.machines_dict = dict(machines_dict)
        self.instance_name = instance_name
        self.k_scope = int(k_scope)
        self.k_seed = int(k_seed)
        self.delta_k = int(delta_k)
        self.formal_mode = bool(formal_mode)
        self.enable_adjacency_closure = bool(enable_adjacency_closure)
        self.initial_orders = list(initial_orders)
        self._validate_orders(self.initial_orders)
        self._validate_events(events)
        self.events: Tuple[DynamicEvent, ...] = tuple(
            sorted(events, key=lambda e: (self._event_time(e), self._event_id(e)))
        )
        self._event_by_id = {self._event_id(e): e for e in self.events}
        if len(self._event_by_id) != len(self.events):
            raise ValueError("动态事件 event_id 必须唯一。")
        self._initial_solution = initial_solution.copy() if initial_solution is not None else None
        self.setup_started_uids = frozenset(int(uid) for uid in setup_started_uids)
        self.current_solution: Solution = create_empty_solution(pool_type)
        self.current_time = 0.0
        self.pending_event_ids: List[str] = []
        self.triggered_event_ids: List[str] = []
        self.known_orders: Dict[str, Order] = {o.order_id: o for o in self.initial_orders}
        self.active_context: Optional[DynamicRescheduleContext] = None
        self.last_result: Optional[Tuple[str, bool, List[str]]] = None
        self.reset()

    def _validate_orders(self, orders: Sequence[Order]) -> None:
        ids: Set[str] = set()
        for order in orders:
            if order.order_id in ids:
                raise ValueError(f"订单号重复: {order.order_id}")
            ids.add(order.order_id)
            if order.pool_type != self.pool_type:
                raise ValueError(
                    f"订单 {order.order_id} 所属资源池 {order.pool_type} 与管理器 {self.pool_type} 不一致。"
                )

    def _validate_events(self, events: Sequence[DynamicEvent]) -> None:
        event_ids: Set[str] = set()
        order_ids = {o.order_id for o in self.initial_orders}
        for event in events:
            if not isinstance(event, DynamicInsertEvent):
                raise ValueError("动态调度环境严格仅支持 DynamicInsertEvent。")
            event_id = self._event_id(event)
            if event_id in event_ids:
                raise ValueError(f"动态事件 event_id 重复: {event_id}")
            event_ids.add(event_id)
            if self.instance_name is not None and event.instance_name != self.instance_name:
                raise ValueError(
                    f"事件 {event.event_id} 实例 {event.instance_name} 与管理器实例 {self.instance_name} 不一致。"
                )
            if event.order.order_id in order_ids:
                raise ValueError(f"动态订单号与已有订单冲突: {event.order.order_id}")
            if event.order.pool_type != self.pool_type:
                raise ValueError(f"动态订单 {event.order.order_id} 不属于资源池 {self.pool_type}。")
            order_ids.add(event.order.order_id)

    @staticmethod
    def _event_id(event: DynamicEvent) -> str:
        return event.event_id

    @staticmethod
    def _event_time(event: DynamicEvent) -> float:
        return float(event.actual_arrival_time)

    def _initial_solution_cache_key(self) -> Tuple[object, ...]:
        """返回只由静态物理输入决定的初始排程蓝图键。"""
        order_key = tuple(
            (
                str(order.order_id),
                str(order.process_type),
                int(order.Q),
                int(order.S),
                int(order.W),
                int(order.N),
                float(order.efficiency),
                int(order.H),
                tuple(str(mid) for mid in order.eligible_machines),
                float(getattr(order, "release_time", 0.0) or 0.0),
            )
            for order in sorted(self.initial_orders, key=lambda item: item.order_id)
        )
        machine_key = tuple(
            (
                str(mid),
                str(machine.pool_type),
                int(machine.spindles),
                str(machine.machine_type),
            )
            for mid, machine in sorted(self.machines_dict.items())
        )
        return (
            INITIAL_SOLUTION_BUILDER_VERSION,
            INITIAL_SOLUTION_DISPATCH_RULE,
            self.pool_type,
            order_key,
            machine_key,
        )

    @staticmethod
    def _batch_finish_time(solution: Solution, u_id: int) -> float:
        batch = solution.active_batches.get(u_id)
        if batch is None:
            raise RuntimeError(f"初始解候选中找不到临时批次 u_id={u_id}。")
        return float(batch.finish_time)

    def _construct_initial_solution_blueprint(self) -> Solution:
        """按容量感知 SPT 构造事件前静态排程蓝图。

        初始订单按其在合格机台上的最短单趟加工时间升序排列。处理当前
        订单时，不复用在线 SplitPolicy 的前 16 个模板，也不搜索模板：
        对每台合格机台，以 ``min(剩余叠板单元数, 机台主轴数)`` 形成
        一个队尾候选批次，调用正式 ``full_decode`` 后按
        ``(当前批次完工, Cmax, 批次容量降序, 机台)`` 选择。重复直至该
        订单需求守恒，再处理下一个 SPT 订单。这样既保持所有机台队列
        的 SPT 订单次序，又允许 D1--D3 的 6 主轴和 D4--D8 的 2 主轴
        同时承担初始负载。该过程只创建事件前起始状态，不产生神经网络
        轨迹，也不进入动态 Actor 的状态、奖励、修复或失败回退路径。
        """
        solution = create_empty_solution(self.pool_type)
        orders_dict = {order.order_id: order for order in self.initial_orders}
        def eligible_machine_ids(order: Order) -> List[str]:
            eligible = sorted(
                mid
                for mid in order.eligible_machines
                if mid in self.machines_dict
                and mid in solution.machine_queues
                and self.machines_dict[mid].pool_type == self.pool_type
            )
            if not eligible:
                raise RuntimeError(f"初始订单 {order.order_id} 无同池合格机台。")
            return eligible

        def spt_key(order: Order) -> Tuple[float, float, str]:
            eligible = eligible_machine_ids(order)
            eligible_times = [
                processing_time(order, self.machines_dict[mid])
                for mid in eligible
            ]
            return (
                float(min(eligible_times)),
                float(getattr(order, "release_time", 0.0) or 0.0),
                str(order.order_id),
            )

        next_temporary_uid = -1
        for order in sorted(self.initial_orders, key=spt_key):
            remaining = int(order.W)
            if remaining <= 0:
                raise RuntimeError(f"初始订单 {order.order_id} 的物理叠板单元数必须为正。")
            eligible = eligible_machine_ids(order)
            while remaining > 0:
                batch_uid = next_temporary_uid
                best_score: Optional[Tuple[float, float, int, str]] = None
                best_solution: Optional[Solution] = None
                best_quantity = 0
                for mid in eligible:
                    quantity = min(remaining, int(self.machines_dict[mid].spindles))
                    if quantity <= 0:
                        continue
                    candidate = copy.deepcopy(solution)
                    candidate.machine_queues[mid].append(
                        Batch(
                            order_id=order.order_id,
                            u_id=batch_uid,
                            stack_count=quantity,
                            machine_id=mid,
                        )
                    )
                    full_decode(candidate, orders_dict, self.machines_dict)
                    score = (
                        self._batch_finish_time(candidate, batch_uid),
                        float(candidate.makespan),
                        -int(quantity),
                        str(mid),
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_solution = candidate
                        best_quantity = quantity
                if best_solution is None or best_quantity <= 0:
                    raise RuntimeError(
                        f"容量感知 SPT 未能为初始订单 {order.order_id} 的剩余需求 "
                        f"{remaining} 生成合法队尾批次。"
                    )
                solution = best_solution
                remaining -= best_quantity
                next_temporary_uid -= 1

        next_blueprint_uid = 1
        for queue in solution.machine_queues.values():
            for batch in queue.batches:
                batch.u_id = next_blueprint_uid
                next_blueprint_uid += 1
        full_decode(solution, orders_dict, self.machines_dict)

        normalize_solution(solution)
        valid, errors = validate_solution(
            solution,
            orders_dict,
            self.machines_dict,
            raise_exception=False,
            require_decoded=True,
        )
        if not valid:
            raise RuntimeError(f"确定性初始解蓝图校验失败: {errors}")
        return solution

    def _materialize_initial_solution(self, blueprint: Solution) -> Solution:
        """复制缓存蓝图，为本次独立 episode 分配不可复用的正式 UID。"""
        solution = copy.deepcopy(blueprint)
        orders_dict = {order.order_id: order for order in self.initial_orders}
        for queue in solution.machine_queues.values():
            for batch in queue.batches:
                batch.u_id = allocate_uid()
        full_decode(solution, orders_dict, self.machines_dict)
        normalize_solution(solution)
        valid, errors = validate_solution(
            solution,
            orders_dict,
            self.machines_dict,
            raise_exception=False,
            require_decoded=True,
        )
        if not valid:
            raise RuntimeError(f"确定性初始解实例化校验失败: {errors}")
        return solution

    def _build_initial_solution(self) -> Solution:
        if not self.initial_orders:
            return create_empty_solution(self.pool_type)

        cache_key = self._initial_solution_cache_key()
        blueprint = _INITIAL_SOLUTION_BLUEPRINT_CACHE.get(cache_key)
        if blueprint is None:
            blueprint = self._construct_initial_solution_blueprint()
            _INITIAL_SOLUTION_BLUEPRINT_CACHE[cache_key] = copy.deepcopy(blueprint)
        return self._materialize_initial_solution(blueprint)

    def reset(self, initial_solution: Optional[Solution] = None, current_time: float = 0.0) -> dict:
        """恢复到初始静态解和未触发事件队列。"""
        self.current_time = _require_time(current_time, "current_time", "reset")
        # 每轮训练/评估都从静态订单集合重新开始；上一轮事件加入的动态订单
        # 不能参与本轮初始解校验，否则合法初始解会被误判为缺少动态批次。
        self.known_orders = {o.order_id: o for o in self.initial_orders}
        candidate = initial_solution.copy() if initial_solution is not None else (
            self._initial_solution.copy() if self._initial_solution is not None else self._build_initial_solution()
        )
        normalize_solution(candidate)
        valid, errors = validate_solution(candidate, self.known_orders, self.machines_dict, raise_exception=False)
        if not valid:
            raise ValueError("初始解未通过物理校验：" + "；".join(errors))
        self.current_solution = candidate
        self.pending_event_ids = [self._event_id(e) for e in self.events]
        self.triggered_event_ids = []
        self.active_context = None
        self.last_result = None
        # 重新核验显式 setup_started 边界。
        compute_freeze_snapshot(self.current_solution, self.current_time, self.setup_started_uids, machines_dict=self.machines_dict)
        return self.get_state()

    def trigger_next_event(self, event_id: Optional[str] = None) -> DynamicRescheduleContext:
        """推进到下一个实际事件时刻并生成冻结快照；不触碰当前解。"""
        if self.active_context is not None:
            raise RuntimeError("已有事件等待候选解提交或回滚。")
        if not self.pending_event_ids:
            raise StopIteration("没有待触发的动态事件。")
        selected_id = self.pending_event_ids[0] if event_id is None else event_id
        if selected_id not in self.pending_event_ids:
            raise ValueError(f"事件 {selected_id} 不在待触发队列中。")
        if selected_id != self.pending_event_ids[0]:
            raise ValueError("必须按实际事件时刻顺序触发事件，不能跳过更早事件。")
        event = self._event_by_id[selected_id]
        event_time = self._event_time(event)
        if event_time < self.current_time:
            raise ValueError("事件时刻不能早于当前仿真时刻。")
        self.current_time = event_time
        self.pending_event_ids.remove(selected_id)
        self.triggered_event_ids.append(selected_id)
        if isinstance(event, DynamicInsertEvent):
            self.known_orders[event.order.order_id] = event.order

        freeze = compute_freeze_snapshot(self.current_solution, self.current_time, self.setup_started_uids, machines_dict=self.machines_dict)

        scope_res = None
        if isinstance(event, DynamicInsertEvent) and self.k_scope >= 0:
            from scope_selector import select_affected_domain
            scope_res = select_affected_domain(
                solution=self.current_solution,
                event_time=self.current_time,
                frozen_uids=freeze.frozen_uids,
                new_orders=[event.order],
                orders_dict=self.known_orders,
                machines_dict=self.machines_dict,
                k_scope=self.k_scope,
                k_seed=self.k_seed,
                enable_adjacency_closure=self.enable_adjacency_closure,
            )

        context = DynamicRescheduleContext(
            event=event,
            event_time=self.current_time,
            base_solution=self.current_solution.copy(),
            orders=dict(self.known_orders),
            freeze=freeze,
            new_order_ids=(event.order.order_id,) if isinstance(event, DynamicInsertEvent) else (),
            k_scope=self.k_scope,
            k_seed=self.k_seed,
            affected_uids=scope_res.affected_uids if scope_res else frozenset(),
            stable_uids=scope_res.stable_uids if scope_res else frozenset(),
            stable_signature=scope_res.stable_signature if scope_res else {},
            scope_ratio=scope_res.scope_ratio if scope_res else 0.0,
        )
        self.active_context = context
        return context
        self.active_context = context
        return context

    def begin_transaction(self, context: Optional[DynamicRescheduleContext] = None) -> RescheduleTransaction:
        if context is None:
            context = self.active_context
        if context is None:
            raise RuntimeError("尚未触发事件。")
        if context is not self.active_context:
            raise ValueError("只能为当前活动事件创建事务。")
        return RescheduleTransaction(self, context)

    def begin_reallocation(
        self,
        context: Optional[DynamicRescheduleContext] = None,
        new_order: Optional[Order] = None,
        template: Sequence[int] = (),
        partition_plan: Optional[Dict[str, Sequence[int]]] = None,
    ) -> ReallocationTransaction:
        """创建“受影响域旧批次 + 新订单子批”的统一 Worker 重排事务。"""
        if context is None:
            context = self.active_context
        if context is None:
            raise RuntimeError("尚未触发事件。")
        if context is not self.active_context:
            raise ValueError("只能为当前活动事件创建重排事务。")
        if new_order is None:
            raise ValueError("重排事务必须提供新订单。")
        plan: Dict[str, Tuple[int, ...]] = dict(partition_plan or {})
        plan[new_order.order_id] = tuple(template)

        def _validate_template(order: Order, candidate_template: Sequence[int]) -> Tuple[int, ...]:
            if (
                not candidate_template
                or any(isinstance(q, bool) or int(q) != q or int(q) <= 0 for q in candidate_template)
                or sum(int(q) for q in candidate_template) != order.W
            ):
                raise ValueError(f"订单 {order.order_id} 模板必须为正整数划分且精确守恒 W。")
            return tuple(int(q) for q in candidate_template)

        plan = {
            oid: _validate_template(context.orders[oid], candidate_template)
            for oid, candidate_template in plan.items()
        }
        if new_order.order_id not in plan:
            raise ValueError("新订单模板必须为正整数划分且精确守恒 W。")
        batches_by_order: Dict[str, List[Batch]] = {}
        movable_old_uids = context.movable_uids
        existing_batches = [
            batch
            for queue in context.base_solution.machine_queues.values()
            for batch in queue.batches
            if batch.u_id in movable_old_uids
        ]
        movable_uids_by_order: Dict[str, Set[int]] = {}
        for batch in existing_batches:
            movable_uids_by_order.setdefault(batch.order_id, set()).add(batch.u_id)

        replaced_uids: Set[int] = set()
        for oid, candidate_template in plan.items():
            order = context.orders.get(oid)
            if order is None:
                raise ValueError(f"模板计划包含未知订单 {oid}。")
            order_uids = {
                batch.u_id
                for queue in context.base_solution.machine_queues.values()
                for batch in queue.batches
                if batch.order_id == oid
            }
            frozen_order_uids = order_uids.intersection(context.frozen_uids)
            if frozen_order_uids:
                raise ValueError(
                    f"订单 {oid} 存在冻结批次 {sorted(frozen_order_uids)}，不能整体重划分。"
                )
            stable_order_uids = order_uids.intersection(context.stable_uids)
            if stable_order_uids:
                raise ValueError(
                    f"订单 {oid} 存在稳定补集批次 {sorted(stable_order_uids)}，不能整体重划分。"
                )
            replaced_uids.update(movable_uids_by_order.get(oid, set()))
            batches_by_order[oid] = [
                Batch(order_id=oid, u_id=allocate_uid(), stack_count=stack_count, machine_id="")
                for stack_count in candidate_template
            ]

        # 决策域子批确定性放置顺序（严格对齐参数规范 3.2 节与算法 1 第 6 步）：
        # 1. 新订单子批先于受影响旧批次进入待放置序列；
        # 2. 订单间按业务优先级、释放时刻、交期 EDD、订单 ID 稳定排序；
        # 3. 同订单内按批次大小 q_b 降序；UID 升序决胜。
        staged_new_batches = [batch for oid in sorted(batches_by_order) for batch in batches_by_order[oid]]
        movable_batches = [batch for batch in existing_batches if batch.u_id not in replaced_uids]
        all_candidate_batches = staged_new_batches + movable_batches
        priority_rank = {"高": 0, "中": 1, "普通": 2}

        def subbatch_sort_key(batch: Batch):
            order = context.orders[batch.order_id]
            is_old = 0 if batch.order_id in context.new_order_ids else 1
            rel_time = float(getattr(order, "release_time", 0.0) or 0.0)
            raw_due = getattr(order, "due_date", None)
            due_date = float(raw_due) if raw_due is not None else float("inf")
            return (
                is_old,
                priority_rank.get(str(getattr(order, "priority", "普通")), 9),
                rel_time,
                due_date,
                order.order_id,
                -batch.stack_count,
                batch.u_id,
            )

        all_candidate_batches.sort(key=subbatch_sort_key)
        return ReallocationTransaction(
            self,
            context,
            all_candidate_batches,
            replaced_uids=replaced_uids,
        )

    def begin_progressive_reallocation(
        self,
        context: Optional[DynamicRescheduleContext] = None,
    ) -> ReallocationTransaction:
        """创建严格按 Split→Placement→下一 Split 推进的事件级渐进事务。"""
        if context is None:
            context = self.active_context
        if context is None:
            raise RuntimeError("尚未触发事件。")
        if context is not self.active_context:
            raise ValueError("只能为当前活动事件创建重排事务。")
        return ReallocationTransaction(
            self,
            context,
            staged_batches=(),
            replaced_uids=(),
            progressive=True,
        )

    def _commit_context(self, context: DynamicRescheduleContext, solution: Solution) -> None:
        if context is not self.active_context:
            raise RuntimeError("提交的事件上下文不是当前活动上下文。")
        self.current_solution = solution
        self.active_context = None
        self.last_result = (context.event.event_id, True, [])

    def rollback_active(self) -> None:
        """回滚候选解；事件已经发生，保持已触发状态，可重新开事务。"""
        if self.active_context is None:
            return
        event_id = self.active_context.event.event_id
        self.last_result = (event_id, False, ["候选解回滚，事件保留为已触发状态。"])

    def get_state(self) -> dict:
        """返回供状态编码器使用的稳定字典，不暴露可直接写入的内部解。"""
        freeze = compute_freeze_snapshot(self.current_solution, self.current_time, self.setup_started_uids, machines_dict=self.machines_dict)
        return {
            "pool_type": self.pool_type,
            "current_time": self.current_time,
            "order_ids": tuple(sorted(self.known_orders)),
            "pending_event_ids": tuple(self.pending_event_ids),
            "triggered_event_ids": tuple(self.triggered_event_ids),
            "frozen_uids": tuple(sorted(freeze.frozen_uids)),
            "done_uids": tuple(sorted(freeze.done_uids)),
            "processing_uids": tuple(sorted(freeze.processing_uids)),
            "setup_started_uids": tuple(sorted(freeze.setup_started_uids)),
            "active_event_id": self.active_context.event.event_id if self.active_context else None,
        }

    def build_independent_baseline_candidate(
        self,
        context: Optional[DynamicRescheduleContext] = None,
        max_templates: int = 16,
    ) -> Solution:
        """
        构造离线对比实验用的确定性基线候选；不得接入训练、状态、奖励或部署。
        1. 冻结前缀与稳定补集在物理机台队列上保持原样不变；
        2. 仅移出确定性受影响域中的旧批次；
        3. 对新到达订单采用贪心最大整批拆分确定分批模板；
        4. 全量决策域子批按 (新插单优先, 业务优先级, 释放时间, 交期, 订单ID, 批量降序, uid) 排序；
        5. 枚举各子批的全部合法插槽，按完整解码后的五元字典序
           (ΔCmax, Cmax, 当前子批完成时刻, 槽位, 机台索引) 确定性极小化放置；
        6. 调用统一物理解码器完成半主动时间递推，输出完整闭环候选解。
        """
        if context is None:
            context = self.active_context
        if context is None:
            raise RuntimeError("尚未触发事件，无法构造独立离线基线解。")

        candidate = context.base_solution.copy()
        affected_set = set(context.affected_uids)
        frozen_set = set(context.frozen_uids)

        # 1. 仅移出受影响域旧批次；冻结前缀与稳定补集原样保留在快照队列中
        removed_batches: List[Batch] = []
        for queue in candidate.machine_queues.values():
            kept: List[Batch] = []
            for b in queue.batches:
                if b.u_id in affected_set:
                    removed_batches.append(b)
                else:
                    kept.append(b)
            queue.batches = kept

        # 重新解码移出受影响批次后的解，获取精确的基线完工时间
        normalize_solution(candidate)
        full_decode(
            candidate,
            context.orders,
            self.machines_dict,
            event_time=context.event_time,
            frozen_uids=context.frozen_uids,
        )
        current_cmax = float(candidate.makespan)

        # 2. 为新到达订单通过贪心最大整批拆分选定基准分批模板
        priority_rank = {"高": 0, "中": 1, "普通": 2}
        new_orders = [context.orders[oid] for oid in context.new_order_ids if oid in context.orders]
        new_orders.sort(
            key=lambda o: (
                priority_rank.get(o.priority, 9),
                float(o.release_time) if hasattr(o, "release_time") and o.release_time is not None else 0.0,
                float(o.due_date) if hasattr(o, "due_date") and o.due_date is not None else 0.0,
                o.order_id,
            )
        )

        staged_new_batches: List[Batch] = []
        for order in new_orders:
            templates = generate_partition_templates(
                order,
                max_templates=max_templates,
                machines=self.machines_dict,
            )
            available_caps = [
                self.machines_dict[mid].spindles
                for mid in order.eligible_machines
                if mid in self.machines_dict
            ]
            max_cap = max(available_caps, default=0)
            valid_templates = [t for t in templates if max_cap > 0 and max(t) <= max_cap]
            if not valid_templates:
                raise InfeasibleHeuristicError(
                    f"订单 {order.order_id} 在可用机台主轴容量约束下没有合法的拆批模板。"
                )

            # 按规范 4.1 字典序：子批数升序, 首批容量降序, 模板内容字典序
            valid_templates.sort(key=lambda t: (len(t), -t[0], t))
            best_template = valid_templates[0]

            for stack_count in best_template:
                staged_new_batches.append(
                    Batch(
                        order_id=order.order_id,
                        u_id=allocate_uid(),
                        stack_count=stack_count,
                        machine_id="",
                    )
                )

        # 3. 收集受影响域旧批次
        staged_affected_batches: List[Batch] = []
        for b in removed_batches:
            staged_affected_batches.append(
                Batch(
                    order_id=b.order_id,
                    u_id=b.u_id,
                    stack_count=b.stack_count,
                    machine_id="",
                )
            )

        # 4. 全量决策域子批确定性排序（规范 3.2）
        def batch_decision_order(b: Batch):
            o = context.orders[b.order_id]
            is_new = 0 if b.order_id in context.new_order_ids else 1
            p_rank = priority_rank.get(o.priority, 9)
            rel = float(o.release_time) if hasattr(o, "release_time") and o.release_time is not None else 0.0
            due = float(o.due_date) if hasattr(o, "due_date") and o.due_date is not None else 0.0
            return (is_new, p_rank, rel, due, o.order_id, -b.stack_count, b.u_id)

        all_staged = staged_new_batches + staged_affected_batches
        all_staged.sort(key=batch_decision_order)

        # 机器有序索引映射（用于打破平局）
        machine_order = sorted(self.machines_dict.keys())
        machine_idx_map = {m: idx for idx, m in enumerate(machine_order)}

        # 5. 逐个子批枚举全部合法插槽，并按统一五元字典序选择。
        for batch in all_staged:
            order = context.orders[batch.order_id]
            eligible_mids = [
                mid for mid in order.eligible_machines
                if mid in self.machines_dict
                and self.machines_dict[mid].pool_type == self.pool_type
                and self.machines_dict[mid].spindles >= batch.stack_count
            ]
            if not eligible_mids:
                raise InfeasibleHeuristicError(f"离线基线子批 order={batch.order_id}, uid={batch.u_id} 没有可用合格机台。")

            valid_slots: List[Tuple[str, int]] = []
            for mid in eligible_mids:
                queue = candidate.machine_queues[mid]
                frozen_positions = [
                    idx for idx, queued in enumerate(queue.batches) if queued.u_id in frozen_set
                ]
                frozen_tail = max(frozen_positions, default=-1) + 1
                valid_slots.extend((mid, pos) for pos in range(frozen_tail, len(queue.batches) + 1))
            if not valid_slots:
                raise InfeasibleHeuristicError(f"离线基线子批 order={batch.order_id}, uid={batch.u_id} 没有可行机台插槽。")

            decoded_candidates: List[Tuple[Tuple[float, float, float, int, int], Solution]] = []
            for slot in valid_slots:
                target_mid, slot_l = slot
                trial = candidate.copy()
                trial_batch = copy.deepcopy(batch)
                trial_batch.machine_id = target_mid
                trial.machine_queues[target_mid].insert_at(slot_l, trial_batch)
                normalize_solution(trial)
                full_decode(
                    trial,
                    context.orders,
                    self.machines_dict,
                    event_time=context.event_time,
                    frozen_uids=context.frozen_uids,
                )
                placed = next(
                    queued
                    for queue in trial.machine_queues.values()
                    for queued in queue.batches
                    if queued.u_id == batch.u_id
                )
                score = (
                    float(trial.makespan) - current_cmax,
                    float(trial.makespan),
                    float(placed.finish_time),
                    int(slot_l),
                    int(machine_idx_map.get(target_mid, 999)),
                )
                decoded_candidates.append((score, trial))

            _, candidate = min(decoded_candidates, key=lambda item: item[0])
            current_cmax = float(candidate.makespan)

        normalize_solution(candidate)
        full_decode(
            candidate,
            context.orders,
            self.machines_dict,
            event_time=context.event_time,
            frozen_uids=context.frozen_uids,
        )
        return candidate

    def run_independent_baseline(
        self,
        context: Optional[DynamicRescheduleContext] = None,
        reference_builder: Optional[Callable[[DynamicRescheduleContext], Solution]] = None,
        max_templates: int = 16,
    ) -> Solution:
        """仅供离线对比实验显式调用；主算法禁止调用。"""
        if context is None:
            context = self.active_context
        if context is None:
            raise RuntimeError("尚未触发事件，无法运行独立离线基线。")
        if reference_builder is not None:
            candidate = reference_builder(context)
            tx = self.begin_transaction(context)
            errors = tx.validate(candidate)
            tx.rollback()
            if errors:
                raise ValueError("独立确定性基线未通过冻结签名或物理约束校验：" + "；".join(errors))
            reference = candidate.copy()
            normalize_solution(reference)
            return reference

        # 动态插单事件：支持以 self.delta_k 为步长的自适应受影响域扩展 (规范 1.7)
        from scope_selector import select_affected_domain

        curr_k = context.k_scope
        curr_ctx = context
        total_unfrozen = len([
            b for q in context.base_solution.machine_queues.values()
            for b in q.batches if b.u_id not in context.frozen_uids
        ])
        success = False
        last_errors: List[str] = []

        while True:
            try:
                candidate = self.build_independent_baseline_candidate(curr_ctx, max_templates=max_templates)
                tx = self.begin_transaction(curr_ctx)
                errors = tx.validate(candidate)
                tx.rollback()
                if not errors:
                    success = True
                    break
                last_errors = errors
            except Exception as exc:
                last_errors = [str(exc)]

            if curr_k >= total_unfrozen:
                break
            curr_k = min(total_unfrozen, curr_k + self.delta_k)
            new_ord_objs = [curr_ctx.orders[oid] for oid in curr_ctx.new_order_ids if oid in curr_ctx.orders]
            scope_res = select_affected_domain(
                solution=curr_ctx.base_solution,
                event_time=curr_ctx.event_time,
                frozen_uids=curr_ctx.frozen_uids,
                new_orders=new_ord_objs,
                orders_dict=curr_ctx.orders,
                machines_dict=self.machines_dict,
                k_scope=curr_k,
                k_seed=curr_ctx.k_seed,
                enable_adjacency_closure=self.enable_adjacency_closure,
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
            self.active_context = curr_ctx

        if not success:
            raise ValueError("独立确定性基线未找到可行后继排程：" + "；".join(last_errors))

        reference = candidate.copy()
        normalize_solution(reference)
        return reference


