"""
time_model.py
=============
PCB 钻孔车间工时与换型时间计算模型（规范第 5 章）

【计算规则与参数说明】
- 本模块提供统一的工时与换型时间计算接口，供各算法模块调用。
- 批次加工时间 P(i, m) 与批次叠板单元数 B_batch 无关（规范 5.5 节）。
- 批次加工时间按每趟加工计算，不与 B_batch 作乘法。
- 首批次计入初始换型时间（规范 5.6 节，假设 2 & 10）。
- 同订单连续批次换型时间为 0；跨订单换型时间为 s_setup（机型固定值）。
- 基础物理参数：t_aux（2.2, 6.6, 1.4）与 s_setup（5.5, 13.5, 12.0）。
"""

from __future__ import annotations

import math
from typing import Optional

from data_models import Machine, Order, TIME_PARAMS


def processing_time(order: Order, machine: Machine) -> float:
    """
    计算订单 i 在机器 m 上的单趟批次加工时间（规范 5.2 节唯一工时公式）。

    公式：P(i, m) = t_aux(m) + N_i / v(i, m) + ceil(N_i / H_i) * t_tool(m)

    参数
    ----
    order   : Order  — 订单对象（提供 N_i, H_i, efficiency）
    machine : Machine — 机器对象（提供 t_aux, t_tool）

    返回
    ----
    加工时间（分钟，float），与批次叠板单元数无关。

    注意
    ----
    - 效率 v(i, m) 使用订单对象中的 efficiency 字段（来自 Excel），
      不使用机器默认效率，以支持未来数据扩展（规范 3.4 节）。
    - 若订单效率与机型预期效率不一致，data_loader 已发出警告，
      本模块以订单实际值为准。
    """
    # 纯钻削时间（孔数 / 效率）
    t_drill = order.N / order.efficiency

    # 平均换刀时间：ceil(孔数 / 换刀孔限) * 单次换刀工时
    n_tool_changes = math.ceil(order.N / order.H)
    t_tool_total = n_tool_changes * machine.t_tool

    return machine.t_aux + t_drill + t_tool_total


def setup_time(
    machine: Machine,
    prev_order_id: Optional[str],
    curr_order_id: str,
    is_first_batch: bool,
) -> float:
    """
    计算当前批次的换型时间（规范 5.6 节）。

    规则（优先级从高到低）：
      1. 机器首批次（is_first_batch=True）：必须计入初始换型时间 s_setup(m)。
      2. 非首批次、跨订单（prev_order_id != curr_order_id）：计入 s_setup(m)。
      3. 非首批次、同订单（prev_order_id == curr_order_id）：换型时间 = 0。

    参数
    ----
    machine        : Machine — 机器对象（提供 s_setup）
    prev_order_id  : Optional[str] — 前一个批次的订单 ID，首批次时传 None
    curr_order_id  : str — 当前批次的订单 ID
    is_first_batch : bool — 是否为该机器的第一个批次

    返回
    ----
    换型时间（分钟，float）
    """
    if is_first_batch:
        # 首批次强制计入初始换型（规范假设 2 & 10）
        return machine.s_setup

    if prev_order_id is None:
        # 防御性处理：非首批次但前驱为 None，等价于首批次
        return machine.s_setup

    if prev_order_id == curr_order_id:
        # 同订单继承，换型为 0（规范假设 11）
        return 0.0
    else:
        # 跨订单换型（规范假设 12）
        return machine.s_setup


def batch_finish_time(
    order: Order,
    machine: Machine,
    start_time: float,
    prev_order_id: Optional[str],
    is_first_batch: bool,
) -> float:
    """
    计算批次的完工时间（便利函数）。

    完工时间 = 开工时间 + 换型时间 + 加工时间

    参数
    ----
    order          : Order
    machine        : Machine
    start_time     : float — 批次可以开始的最早时间（已考虑机器空闲与前驱完工）
    prev_order_id  : Optional[str] — 前一批次订单 ID
    is_first_batch : bool

    返回
    ----
    (setup_t, proc_t, finish_t) — 换型时间、加工时间、完工时间（均为分钟）
    """
    setup_t = setup_time(machine, prev_order_id, order.order_id, is_first_batch)
    proc_t = processing_time(order, machine)
    finish_t = start_time + setup_t + proc_t
    return setup_t, proc_t, finish_t


# ---------------------------------------------------------------------------
# 自验模块：直接运行时输出典型工时对照表（TC-02 参考基准）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_models import Machine, Order, MACHINE_CONFIGS, TIME_PARAMS

    # 构造典型机器对象
    machines = {cfg["machine_id"]: Machine(**cfg) for cfg in MACHINE_CONFIGS}

    # 构造测试订单（源自 test_1.xlsx Sheet1 第 1 行，孔数=14386, eff=220, H=1500）
    import math as _math

    test_order = Order(
        order_id="TEST_01",
        process_type="钻孔",
        Q=19, S=2, W=_math.ceil(19 / 2),
        N=14386, efficiency=220.0, H=1500,
    )

    print("=== time_model 自验输出（TC-02 基准参考） ===")
    print(f"订单 TEST_01: Q={test_order.Q}, S={test_order.S}, W={test_order.W}, N={test_order.N}")
    print()
    for mid, m in machines.items():
        # 只计算同池合格机器
        if test_order.process_type == "钻孔" and "B" in mid:
            continue
        p = processing_time(test_order, m)
        s_first = setup_time(m, None, test_order.order_id, is_first_batch=True)
        s_same = setup_time(m, test_order.order_id, test_order.order_id, is_first_batch=False)
        s_cross = setup_time(m, "OTHER", test_order.order_id, is_first_batch=False)
        print(
            f"  机器 {mid} ({m.machine_type}): "
            f"P={p:.4f}min | 首批换型={s_first:.1f}min | 同订单换型={s_same:.1f}min | 跨订单换型={s_cross:.1f}min"
        )
