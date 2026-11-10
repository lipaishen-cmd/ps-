"""
eligibility_builder.py
======================
机器资源池划分与订单资格矩阵构建器（规范 2.2 节、3.1 节、TC-04A、TC-10）

功能说明：
  1. 定义全局标准机器列表（8 台普通钻机 + 2 台背钻机）。
  2. 根据订单的生产工步 + "使用的钻机等级/或制定机台"字段，
     为每个订单生成合格机器 ID 列表 eligible_machines。
  3. 检查双资源池独立约束：
     - 普通钻孔订单的合格机器只能来自 Pool D。
     - 背钻订单的合格机器只能来自 Pool B。
     - 任何跨池赋值均视为严重错误，立即抛出异常。

资格等级解析规则（规范 2.2 节注释、3.1 节字段说明）：
  "S/A/B/C" → 普通钻机全部合格（D1~D8 按工步过滤）
  "S/A/B"   → 含 6 轴机（D1~D3）+ 2 轴机（D4~D8）中 S/A/B 级
  "S/A"     → 只限 6 轴机（D1~D3）高端机型
  "S"       → 只限 D1（最高等级机）
  "指定机台Dx" → 精确锁定到指定机器 ID
  背钻 → 固定 B1, B2

【注意】当前 test_1.xlsx 和 test_2.xlsx 中的钻机等级字段均为 "S/A/B/C"，
即所有普通订单可在 Pool D 全部 8 台机上加工。本模块保留等级解析扩展性，
以备未来数据出现细粒度限制时使用。
"""

from __future__ import annotations

from typing import Dict, List, Set

from data_models import (
    MACHINE_CONFIGS,
    POOL_B,
    POOL_D,
    PROCESS_BACKDRILL,
    PROCESS_DRILL,
    Machine,
    Order,
)


# ---------------------------------------------------------------------------
# 全局机器配置字典（由 MACHINE_CONFIGS 初始化）
# ---------------------------------------------------------------------------

ALL_MACHINES: Dict[str, Machine] = {
    cfg["machine_id"]: Machine(**cfg) for cfg in MACHINE_CONFIGS
}

# 按资源池分组
POOL_D_MACHINES: List[str] = [
    mid for mid, m in ALL_MACHINES.items() if m.pool_type == POOL_D
]
POOL_B_MACHINES: List[str] = [
    mid for mid, m in ALL_MACHINES.items() if m.pool_type == POOL_B
]


# ---------------------------------------------------------------------------
# 资格等级解析
# ---------------------------------------------------------------------------

# 机器等级分档（仅适用于 Pool D 普通钻孔机）
# 等级越高（S > A > B > C），可用机器越少（高端订单限制在高等级机器上）
# 默认采用所有 Pool D 机器全部可用（S/A/B/C）
_GRADE_MAP: Dict[str, Set[str]] = {
    "S":       {"D1"},                         # 最高级，仅限 D1
    "S/A":     {"D1", "D2", "D3"},             # 6 轴机（高精度）
    "S/A/B":   {"D1", "D2", "D3", "D4", "D5", "D6"},  # 6 轴 + 部分 2 轴
    "S/A/B/C": set(POOL_D_MACHINES),           # 全部 Pool D 机器
}

ELIGIBILITY_POLICY_ID = "PCB_DYNAMIC_GRADE_TO_MACHINE_V20261022"
ELIGIBILITY_POLICY_VERSION = "1.0.0"


def eligibility_policy_descriptor() -> Dict[str, object]:
    """Return the frozen eligibility semantics required by the active contract.

    The external JSON policy file is hash-bound by the active dataset contract.
    The trainer compares it with this descriptor before a formal run, so a
    source-workbook hash alone can never silently reinterpret a raw grade.
    """
    return {
        "policy_id": ELIGIBILITY_POLICY_ID,
        "policy_version": ELIGIBILITY_POLICY_VERSION,
        "scope": "Current active Pool-D dynamic contract; Pool-B is retained as a separately isolated physical pool.",
        "drill_grade_to_machine": {
            grade: sorted(machines) for grade, machines in sorted(_GRADE_MAP.items())
        },
        "backdrill_eligible_machines": sorted(POOL_B_MACHINES),
        "exact_machine_assignment": "A single Pool-D machine ID or a comma-separated Pool-D machine-ID list is an exact eligibility restriction.",
        "unknown_or_empty_drill_grade": "REJECT",
        "provenance": {
            "source_field": "使用的钻机等级/或指定机台",
            "mapping_authority": "eligibility_builder.py::_GRADE_MAP synchronized with this frozen policy file",
            "confirmation_status": "CONFIRMED_FOR_ACTIVE_PRE_FORMAL_DYNAMIC_CONTRACT",
            "confirmed_on": "2026-10-22",
        },
    }


def _parse_drill_grade(grade_str: str, order_id: str, *, strict: bool = True) -> List[str]:
    """
    解析钻机等级字符串，返回合格机器 ID 列表（仅限 Pool D）。

    支持以下格式：
      - 等级组合字符串："S", "S/A", "S/A/B", "S/A/B/C"
      - 精确指定机台："D3", "D5"（直接使用机器 ID）
      - 多台指定（逗号分隔）："D3,D5"
    """
    grade_str = grade_str.strip()

    # 检查是否为精确机台指定（包含大写 D 后跟数字）
    if grade_str.startswith("D") or "," in grade_str:
        # 精确机台解析
        candidates = [g.strip() for g in grade_str.split(",")]
        valid = []
        for mid in candidates:
            if mid in ALL_MACHINES and ALL_MACHINES[mid].pool_type == POOL_D:
                valid.append(mid)
            else:
                raise ValueError(
                    f"Order {order_id}: 指定机台 '{mid}' 不在 Pool D 机器列表中，"
                    f"合法 Pool D 机器为 {POOL_D_MACHINES}"
                )
        return sorted(valid)

    # 等级字符串解析
    normalized = grade_str.upper().strip()
    # 标准化斜杠
    normalized = normalized.replace("／", "/")

    if normalized in _GRADE_MAP:
        return sorted(_GRADE_MAP[normalized])

    # 尝试宽松匹配（处理全角字符或空格）
    for key in _GRADE_MAP:
        if normalized.replace(" ", "") == key:
            return sorted(_GRADE_MAP[key])

    # 当前项目不再保留“未知等级默认全机可用”的静默放宽路径。该路径会把
    # 输入错误解释成更大的可行域，违反活动合同中的资格冻结要求。
    raise ValueError(
        f"Order {order_id}: 无法解析钻机等级字段 '{grade_str}'；"
        f"合法值为 {sorted(_GRADE_MAP)} 或 Pool D 指定机台。"
    )


def build_eligible_machines(order: Order, *, strict: bool = True) -> List[str]:
    """
    为单个订单生成合格机器 ID 列表（规范 2.2 节、3.1 节 TC-04A）。

    双资源池独立约束（TC-10）：
      - PROCESS_DRILL 订单 → 仅限 Pool D 机器（D1~D8）
      - PROCESS_BACKDRILL 订单 → 仅限 Pool B 机器（B1~B2）
      - 任何跨池结果均抛出异常

    参数
    ----
    order : Order — 已初始化的订单对象（pool_type 属性已由 process_type 派生）

    返回
    ----
    合格机器 ID 的有序列表
    """
    process_type = order.process_type

    if process_type == PROCESS_DRILL:
        # 普通钻孔：解析钻机等级
        grade = order.raw_machine_grade.strip()
        if not grade:
            raise ValueError(
                f"Order {order.order_id}: 普通钻孔订单的机器等级/指定机台不能为空。"
            )
        return _parse_drill_grade(grade, order.order_id, strict=strict)

    elif process_type == PROCESS_BACKDRILL:
        # 背钻：固定分配 Pool B 全部机器（B1, B2）
        # 背钻工艺表无机器等级字段，直接分配全部背钻机
        return sorted(POOL_B_MACHINES)

    else:
        raise ValueError(
            f"Order {order.order_id}: 未知生产工步 '{process_type}'，"
            f"合法值为 '{PROCESS_DRILL}' 或 '{PROCESS_BACKDRILL}'。"
        )


def build_eligibility_matrix(orders: List[Order], *, strict: bool = True) -> Dict[str, List[str]]:
    """
    为订单列表批量生成资格矩阵（规范 E5 约束的数据基础）。

    参数
    ----
    orders : List[Order]

    返回
    ----
    Dict[order_id -> List[machine_id]]
    """
    matrix: Dict[str, List[str]] = {}
    for order in orders:
        eligible = build_eligible_machines(order, strict=strict)
        order.eligible_machines = eligible  # 同步写入订单对象
        matrix[order.order_id] = eligible
    return matrix


def get_pool_machines(pool_type: str) -> List[str]:
    """返回指定资源池的全部机器 ID 列表"""
    if pool_type == POOL_D:
        return POOL_D_MACHINES
    elif pool_type == POOL_B:
        return POOL_B_MACHINES
    else:
        raise ValueError(f"未知资源池类型 '{pool_type}'，合法值为 '{POOL_D}' 或 '{POOL_B}'")


def validate_cross_pool_isolation(orders: List[Order]) -> None:
    """
    TC-10：验证双资源池独立约束。
    遍历所有订单，确保无跨池资格赋值。
    任何违规立即 raise ValueError。
    """
    for order in orders:
        expected_pool = order.pool_type
        for mid in order.eligible_machines:
            if mid not in ALL_MACHINES:
                raise ValueError(
                    f"TC-10 FAIL: Order {order.order_id} 的合格机器 '{mid}' 不在全局机器注册表中。"
                )
            actual_pool = ALL_MACHINES[mid].pool_type
            if actual_pool != expected_pool:
                raise ValueError(
                    f"TC-10 FAIL: 跨池资格赋值！"
                    f"Order {order.order_id}（工步={order.process_type}，所属池={expected_pool}）"
                    f"被分配到机器 {mid}（所属池={actual_pool}）。"
                    f"违反了双资源池独立约束（规范 2.2 节、假设 4）。"
                )
    # 全部通过
