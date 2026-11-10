"""
data_models.py
==============
统一核心数据对象定义（规范第 4 章、第 8 章、第 11.3 节）
所有模块必须通过此文件中的类型进行数据传递，禁止自行定义并行数据结构。

层级对应关系（规范 5.1 节）：
  第一层: PANEL 需求量 Q_i
  第二层: 最大叠片层数 ML_i，物理叠板单元数 S_i = ceil(Q_i / ML_i)
  第三层: 执行子批 b_ik（其 stack_count 为占用主轴数）
  第四层: 机床主轴容量 P_m

在当前数据结构中，Order.S 存储 ML_i，Order.W 存储 S_i；字段名不是
数学符号的重新定义，任何新增代码均应以这一映射解释。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 全局常量（规范 2.2 节，冻结不可修改）
# ---------------------------------------------------------------------------

POOL_D = "Pool_D"   # 普通钻孔池
POOL_B = "Pool_B"   # 专用背钻池

PROCESS_DRILL = "钻孔"   # 普通钻孔工步
PROCESS_BACKDRILL = "背钻"  # 背钻工步

# 机器固定主轴配置（规范 2.2 节）
# Pool D: 3 台 6 轴机 + 5 台 2 轴机
MACHINE_CONFIGS: List[Dict] = [
    # Pool D – 普通钻孔机
    {"machine_id": "D1", "pool_type": POOL_D, "spindles": 6, "machine_type": "6axis_drill"},
    {"machine_id": "D2", "pool_type": POOL_D, "spindles": 6, "machine_type": "6axis_drill"},
    {"machine_id": "D3", "pool_type": POOL_D, "spindles": 6, "machine_type": "6axis_drill"},
    {"machine_id": "D4", "pool_type": POOL_D, "spindles": 2, "machine_type": "2axis_drill"},
    {"machine_id": "D5", "pool_type": POOL_D, "spindles": 2, "machine_type": "2axis_drill"},
    {"machine_id": "D6", "pool_type": POOL_D, "spindles": 2, "machine_type": "2axis_drill"},
    {"machine_id": "D7", "pool_type": POOL_D, "spindles": 2, "machine_type": "2axis_drill"},
    {"machine_id": "D8", "pool_type": POOL_D, "spindles": 2, "machine_type": "2axis_drill"},
    # Pool B – 专用背钻机
    {"machine_id": "B1", "pool_type": POOL_B, "spindles": 2, "machine_type": "2axis_backdrill"},
    {"machine_id": "B2", "pool_type": POOL_B, "spindles": 2, "machine_type": "2axis_backdrill"},
]

# 工时参数（规范 5.3 & 5.4 节，冻结数值，禁止修改）
TIME_PARAMS: Dict[str, Dict] = {
    "2axis_drill": {
        "t_aux": 2.2,        # 辅助装夹时间 (min)
        "s_setup": 5.5,      # 跨订单换型时间 (min)
        "efficiency": 220.0, # 钻孔效率 (孔/min)
        "H": 1500,           # 折算换刀孔限
        "t_tool": 0.5,       # 单次换刀工时 (min) — 对应 Excel 中的 30S/次 = 0.5 min
    },
    "6axis_drill": {
        "t_aux": 6.6,
        "s_setup": 13.5,
        "efficiency": 220.0,
        "H": 1500,
        "t_tool": 0.5,       # 单次换刀工时 (min) — 对应 Excel 中的 30S/次 = 0.5 min
    },
    "2axis_backdrill": {
        "t_aux": 1.4,
        "s_setup": 12.0,
        "efficiency": 97.0,
        "H": 300,
        "t_tool": 0.5,
    },
}


# 默认折算换刀孔限（规范 5.3 & 5.4 节）
DEFAULT_H_MAP: Dict[str, int] = {
    PROCESS_DRILL: 1500,
    PROCESS_BACKDRILL: 300,
}

# 允许的钻削效率值（规范 3.4 节）
VALID_EFFICIENCIES: Set[float] = {220.0, 97.0}



# 动态排程常量定义
PRIORITY_HIGH = "高"
PRIORITY_MEDIUM = "中"
PRIORITY_NORMAL = "普通"
VALID_PRIORITIES = {PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_NORMAL}

STATUS_INITIAL = "初始"
STATUS_PENDING = "待到达"
STATUS_TRIGGERED = "已触发"
STATUS_SCHEDULED = "已排入"
STATUS_EXCEPTION = "异常"
VALID_STATUSES = {STATUS_INITIAL, STATUS_PENDING, STATUS_TRIGGERED, STATUS_SCHEDULED, STATUS_EXCEPTION}

UNFORECASTED_TAG = "未预报"


def _validate_positive_integer(value: object, field_name: str, order_id: str) -> int:
    """Validate a positive integer-like input without silently truncating it."""
    if isinstance(value, bool):
        raise ValueError(f"Order {order_id}: {field_name}={value!r} 必须为正整数。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Order {order_id}: {field_name}={value!r} 不是有效数值。") from exc
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValueError(f"Order {order_id}: {field_name}={value!r} 必须为正整数，不能截断小数、零或负数。")
    return int(number)


def _validate_nonnegative_time(value: object, field_name: str, order_id: str) -> float:
    """Validate a finite, non-negative time value."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Order {order_id}: {field_name}={value!r} 不是有效时间数值。") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Order {order_id}: {field_name}={value!r} 必须为有限非负数。")
    return number


def _same_optional_time(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """
    订单对象（规范 2.1 节、3.1 节）
    所有字段在 data_loader 完成导入后即为只读，算法不得修改。
    """
    order_id: str            # 订单编号，Sheet 内唯一
    process_type: str        # 生产工步："钻孔" 或 "背钻"
    Q: int                   # PANEL 需求量（客户下达板数）
    S: int                   # 最大叠片数（工艺给定）
    W: int                   # 物理叠板单元总数，= ceil(Q / S)（派生，不可修改）
    N: int                   # 单板总孔数
    efficiency: float        # 钻削效率 (孔/min)
    H: int                   # 折算换刀孔限
    eligible_machines: List[str] = field(default_factory=list)  # 合格机器 ID 列表
    raw_machine_grade: str = ""  # 原始字段 "使用的钻机等级/或制定机台"（保留追溯）

    # 动态排程字段（规范第 3 节）
    expected_arrival_time: Optional[float] = 0.0  # 预计到达时刻（分钟；None 表示未预报）
    actual_arrival_time: float = 0.0              # 实际到达时刻（分钟；初始订单默认为 0.0）
    release_time: float = 0.0                     # 允许加工释放时刻（分钟；初始订单默认为 0.0）
    priority: str = PRIORITY_NORMAL               # 优先级："高"、"中"、"普通"
    event_id: Optional[str] = None                # 动态插单事件ID（如 "INS-001"，初始订单为 None）
    instance_name: Optional[str] = None           # 关联调度实例名（如 "Sheet1"）
    status: str = STATUS_INITIAL                  # 状态："初始"、"待到达"、"已触发"、"已排入"、"异常"
    is_forecasted: bool = True                    # 是否已预报（False 表示突发未预报订单）
    due_date: Optional[float] = None              # 工序交付期（分钟；None 表示无明确交期或静态优化）

    def __post_init__(self):
        if not str(self.order_id).strip():
            raise ValueError("Order: 订单号不能为空。")
        if self.process_type not in (PROCESS_DRILL, PROCESS_BACKDRILL):
            raise ValueError(
                f"Order {self.order_id}: 未知生产工步 '{self.process_type}'，"
                f"合法值为 '{PROCESS_DRILL}' 或 '{PROCESS_BACKDRILL}'。"
            )

        # 工艺数量字段必须是正整数；不再接受 int(float(x)) 式静默截断。
        for field_name in ("Q", "S", "W", "N"):
            _validate_positive_integer(getattr(self, field_name), field_name, str(self.order_id))
        try:
            efficiency = float(self.efficiency)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Order {self.order_id}: 加工效率 {self.efficiency!r} 不是有效数值。") from exc
        if not math.isfinite(efficiency) or efficiency not in VALID_EFFICIENCIES:
            raise ValueError(
                f"Order {self.order_id}: 加工效率 {self.efficiency!r} 非法，必须为 {VALID_EFFICIENCIES} 之一。"
            )
        _validate_positive_integer(self.H, "H", str(self.order_id))

        # 强制校验 W 与数据关系 D1（规范 3.2 节）
        expected_W = math.ceil(self.Q / self.S)
        if self.W != expected_W:
            raise ValueError(
                f"Order {self.order_id}: 叠数 W={self.W} 与 ceil(Q/S)=ceil({self.Q}/{self.S})={expected_W} 不一致！"
                f"违反数据关系 D1。"
            )
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Order {self.order_id}: 优先级 '{self.priority}' 非法，必须为 {VALID_PRIORITIES} 之一。"
            )
        actual_time = _validate_nonnegative_time(self.actual_arrival_time, "实际到达时刻", str(self.order_id))
        release_time = _validate_nonnegative_time(self.release_time, "释放时刻", str(self.order_id))
        if release_time < actual_time:
            raise ValueError(
                f"Order {self.order_id}: 释放时刻 {self.release_time} 不能早于实际到达时刻 {self.actual_arrival_time}。"
            )
        if self.expected_arrival_time is not None:
            _validate_nonnegative_time(self.expected_arrival_time, "预计到达时刻", str(self.order_id))
        elif self.is_forecasted:
            raise ValueError(f"Order {self.order_id}: 未预报订单的预计到达时刻必须为 None，且 is_forecasted=False。")
        if self.expected_arrival_time is None and self.is_forecasted:
            raise ValueError(f"Order {self.order_id}: is_forecasted=True 时预计到达时刻不能为 None。")
        if self.event_id is not None and not str(self.event_id).strip():
            raise ValueError(f"Order {self.order_id}: event_id 不能为空字符串。")
        if self.instance_name is not None and not str(self.instance_name).strip():
            raise ValueError(f"Order {self.order_id}: instance_name 不能为空字符串。")
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Order {self.order_id}: 状态 '{self.status}' 非法，必须为 {VALID_STATUSES} 之一。"
            )
        if self.due_date is not None:
            _validate_nonnegative_time(self.due_date, "工序交付期", str(self.order_id))
            if self.due_date < self.actual_arrival_time:
                raise ValueError(
                    f"Order {self.order_id}: 交付期 {self.due_date} 不能早于实际到达时刻 {self.actual_arrival_time}。"
                )

    @property
    def pool_type(self) -> str:
        """根据生产工步返回所属资源池"""
        if self.process_type == PROCESS_DRILL:
            return POOL_D
        elif self.process_type == PROCESS_BACKDRILL:
            return POOL_B
        else:
            raise ValueError(f"Order {self.order_id}: 未知生产工步 '{self.process_type}'")


@dataclass
class DynamicInsertEvent:
    """
    动态插单到达事件对象
    在实际到达时刻 actual_arrival_time 触发动态重调度
    """
    event_id: str                                 # 事件唯一编号，如 "INS-001"
    instance_name: str                            # 静态实例名，如 "Sheet1"
    order: Order                                  # 携带完整工艺信息的订单对象
    expected_arrival_time: Optional[float]        # 预计到达时刻（分钟，未预报为 None）
    actual_arrival_time: float                    # 实际到达时刻（分钟）
    release_time: float                           # 允许加工释放时刻（分钟）
    priority: str                                 # 优先级
    status: str = STATUS_PENDING                  # 状态
    is_forecasted: bool = True                    # 是否已预报

    def __post_init__(self):
        if not str(self.event_id).strip():
            raise ValueError("DynamicInsertEvent: event_id 不能为空。")
        if not str(self.instance_name).strip():
            raise ValueError(f"DynamicInsertEvent {self.event_id}: instance_name 不能为空。")
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"DynamicInsertEvent {self.event_id}: 优先级 '{self.priority}' 非法。"
            )
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"DynamicInsertEvent {self.event_id}: 状态 '{self.status}' 非法。"
            )
        _validate_nonnegative_time(self.actual_arrival_time, "实际到达时刻", self.event_id)
        _validate_nonnegative_time(self.release_time, "释放时刻", self.event_id)
        if self.release_time < self.actual_arrival_time:
            raise ValueError(
                f"DynamicInsertEvent {self.event_id}: 释放时刻不能早于实际到达时刻。"
            )
        if self.expected_arrival_time is not None:
            _validate_nonnegative_time(self.expected_arrival_time, "预计到达时刻", self.event_id)
        if (self.expected_arrival_time is None) != (not self.is_forecasted):
            raise ValueError(
                f"DynamicInsertEvent {self.event_id}: expected_arrival_time 与 is_forecasted 不一致。"
            )

        # 事件是订单的不可变语义包装；两者字段不一致时立即拒绝，避免动态重调度漂移。
        if self.order.event_id != self.event_id or self.order.instance_name != self.instance_name:
            raise ValueError(f"DynamicInsertEvent {self.event_id}: 事件与订单的 ID/实例关联不一致。")
        if self.order.priority != self.priority or self.order.status != self.status:
            raise ValueError(f"DynamicInsertEvent {self.event_id}: 事件与订单的优先级/状态不一致。")
        if self.order.actual_arrival_time != self.actual_arrival_time or self.order.release_time != self.release_time:
            raise ValueError(f"DynamicInsertEvent {self.event_id}: 事件与订单的到达/释放时刻不一致。")
        if not _same_optional_time(self.order.expected_arrival_time, self.expected_arrival_time):
            raise ValueError(f"DynamicInsertEvent {self.event_id}: 事件与订单的预计到达时刻不一致。")
        if self.order.is_forecasted != self.is_forecasted:
            raise ValueError(f"DynamicInsertEvent {self.event_id}: 事件与订单的预报标记不一致。")

    @property
    def is_delayed(self) -> bool:
        """是否晚于预测到达"""
        return (
            self.is_forecasted
            and self.expected_arrival_time is not None
            and self.actual_arrival_time > self.expected_arrival_time
        )

    @property
    def is_early(self) -> bool:
        """是否早于预测到达"""
        return (
            self.is_forecasted
            and self.expected_arrival_time is not None
            and self.actual_arrival_time < self.expected_arrival_time
        )

    @property
    def is_unforecasted(self) -> bool:
        """是否为突发未预报插单"""
        return not self.is_forecasted or self.expected_arrival_time is None


@dataclass
class Machine:
    """
    机器对象（规范 2.2 节）
    """
    machine_id: str       # 机器编号，如 "D1", "B2"
    pool_type: str        # 所属资源池 POOL_D 或 POOL_B
    spindles: int         # 物理主轴数（2 或 6）
    machine_type: str     # 机型标识，用于查 TIME_PARAMS

    @property
    def t_aux(self) -> float:
        return TIME_PARAMS[self.machine_type]["t_aux"]

    @property
    def s_setup(self) -> float:
        return TIME_PARAMS[self.machine_type]["s_setup"]

    @property
    def default_efficiency(self) -> float:
        return TIME_PARAMS[self.machine_type]["efficiency"]

    @property
    def H(self) -> int:
        return TIME_PARAMS[self.machine_type]["H"]

    @property
    def t_tool(self) -> float:
        return TIME_PARAMS[self.machine_type]["t_tool"]


# 全局 u_id 分配器（线程安全级别：单线程使用，多线程需加锁）
_uid_counter: int = 0


def allocate_uid() -> int:
    """分配全局唯一稳定批次编号 u_id（规范 8.1 节：单调递增，不可复用）"""
    global _uid_counter
    _uid_counter += 1
    return _uid_counter


def reset_uid_counter(start: int = 0) -> None:
    """仅测试代码可调用，重置计数器"""
    global _uid_counter
    _uid_counter = start


@dataclass
class Batch:
    """
    执行批次对象（规范 8.2 节四元组染色体基因）
    基因 = (order_id, u_id, stack_count, machine_id)
    """
    order_id: str        # 所属订单
    u_id: int            # 永久稳定全局唯一编号（算法追踪用）
    stack_count: int     # 该批次叠板单元数（= 占用主轴数）
    machine_id: str      # 指派机器 ID

    # 以下为解码后填充的时间信息（规范 7.5 节）
    start_time: float = 0.0
    finish_time: float = 0.0
    processing_time: float = 0.0  # 由 time_model 计算，与 stack_count 无关

    def as_tuple(self) -> Tuple[str, int, int, str]:
        """返回染色体四元组 (order_id, u_id, stack_count, machine_id)"""
        return (self.order_id, self.u_id, self.stack_count, self.machine_id)


@dataclass
class MachineQueue:
    """
    机器有序加工队列（规范 8.3 节、11.3 节）
    """
    machine_id: str
    pool_type: str
    spindles: int
    batches: List[Batch] = field(default_factory=list)  # 按加工顺序排列
    prefix_cache: List[float] = field(default_factory=list)  # 前缀完工时间缓存（增量解码用）

    def is_empty(self) -> bool:
        return len(self.batches) == 0

    def append(self, batch: Batch) -> None:
        self.batches.append(batch)

    def insert_at(self, pos: int, batch: Batch) -> None:
        self.batches.insert(pos, batch)

    def remove(self, u_id: int) -> Optional[Batch]:
        for i, b in enumerate(self.batches):
            if b.u_id == u_id:
                return self.batches.pop(i)
        return None

    def get_batch_ids(self) -> List[int]:
        return [b.u_id for b in self.batches]

    def __len__(self) -> int:
        return len(self.batches)


@dataclass
class Solution:
    """
    调度解对象（规范 8.1~8.4 节、11.3 节）
    包含某一资源池内所有机器队列与活动批次的完整状态。
    """
    pool_type: str                            # POOL_D 或 POOL_B
    machine_queues: Dict[str, MachineQueue]   # machine_id -> MachineQueue
    active_batches: Dict[int, Batch] = field(default_factory=dict)  # u_id -> Batch

    # 解质量指标（由 decoder 填充）
    makespan: float = float("inf")
    total_setup_time: float = 0.0
    spindle_utilization: float = 0.0

    def get_all_active_uids(self) -> List[int]:
        """返回所有机器队列中的 u_id 有序列表"""
        uids = []
        for mq in self.machine_queues.values():
            uids.extend(mq.get_batch_ids())
        return uids

    def copy(self) -> "Solution":
        """深拷贝，用于动态重调度事务操作前的状态快照（规范 9.14 节原子事务隔离与回滚）"""
        import copy
        return copy.deepcopy(self)


@dataclass
class GlobalResult:
    """
    全局输出结果（规范 11.3 节）
    两池分别独立求解后汇总，全局 Cmax = max(Pool_D, Pool_B)
    """
    solution_D: Optional[Solution] = None
    solution_B: Optional[Solution] = None
    makespan_D: float = float("inf")
    makespan_B: float = float("inf")
    cpu_time: float = 0.0
    seed: int = 0
    valid: bool = False

    @property
    def global_makespan(self) -> float:
        """全局完工时间 = max(两池完工时间)（规范 7.1 节）"""
        return max(self.makespan_D, self.makespan_B)
