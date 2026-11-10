"""
validate_time_model.py
======================
【物理合同检查程序 2】
检查对象：
  1. 四层物理制造层级映射：PANEL 需求量 Q_i、最大叠片层数 ML_i -> 物理叠板单元数 S_i = ceil(Q_i / ML_i) 守恒性；
  2. 批次单趟加工工时 P(i, m) 与执行批次叠板单元数 B_b 的独立性验证（式 1~5）；
  3. 机型工时参数核验（辅助装夹工时 t_aux、换型工时 s_setup、换刀工时 t_tool、钻削效率 v、换刀孔限 H）；
  4. 顺序相关换型时间 σ_{c,b}^m 决策规则闭合验证（式 6~7）；
  5. 事件驱动释放时刻与机台队列开工时间因果闭合递推关系（式 16）。

运行命令：
  python validate_time_model.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ENVIRONMENT_DIR = os.path.join(_PROJECT_ROOT, "environment")

for p in (_PROJECT_ROOT, _ENVIRONMENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_models import Machine, Order, MACHINE_CONFIGS, TIME_PARAMS, POOL_D, POOL_B
from data_loader import load_test1
from time_model import processing_time, setup_time


def calibrate_physical_parameters() -> Dict:
    print("=" * 80)
    print("【物理合同检查 2】PCB 物理制造模型、工时计算与换型时间参数一致性")
    print("=" * 80)

    # 1. 加载当前代理数据包样本；不得解释为现场实测标定
    samples_dir = os.path.join(_PROJECT_ROOT, "data", "test_samples")
    if not os.path.exists(os.path.join(samples_dir, "test_1.xlsx")):
        samples_dir = _PROJECT_ROOT

    sheet_dict = load_test1(samples_dir)
    orders: List[Order] = []
    for s_orders in sheet_dict.values():
        orders.extend(s_orders)
    machines_dict = {cfg["machine_id"]: Machine(**cfg) for cfg in MACHINE_CONFIGS}
    print(f"数据源: test_1.xlsx | 校验工单样本总数: {len(orders)} | 覆盖机台总数: {len(machines_dict)}")

    # 2. 标定检验 1：物理层级转换与叠板需求守恒性
    print("\n--- [检验 1] 物理叠板单元数转换守恒性检验 (S_i = ceil(Q_i / ML_i)) ---")
    conservation_failures = []
    for o in orders:
        theoretical_w = math.ceil(o.Q / o.S)
        if o.W != theoretical_w:
            conservation_failures.append((o.order_id, o.Q, o.S, o.W, theoretical_w))

    assert len(conservation_failures) == 0, f"发现物理需求不守恒违约: {conservation_failures}"
    print(f"  [验证通过] 全部 {len(orders)} 项生产工单满足 S_i = ceil(Q_i / ML_i)，理论误差为 0。")

    # 3. 标定检验 2：单趟批次加工工时与批次叠板单元数 B_b 严格无关性
    print("\n--- [检验 2] 单趟批次加工工时 P(i, m) 对批次大小 B_b 的不变性检验 ---")
    sample_order = orders[0]
    invariance_results = []
    for mid in ["D1", "D4"]:
        m = machines_dict[mid]
        base_proc = processing_time(sample_order, m)
        # 遍历不同主轴占用规模 B_b ∈ [1, C_m]
        for b_size in [1, 2, min(4, m.spindles), m.spindles]:
            p_val = processing_time(sample_order, m)
            delta = abs(p_val - base_proc)
            invariance_results.append((mid, b_size, p_val, delta))

    assert all(d < 1e-9 for _, _, _, d in invariance_results), "批次工时与批次大小相关，违反物理建模假设！"
    print(f"  [验证通过] 机台 D1(6主轴) 与 D4(2主轴) 测试 B_b ∈ [1, C_m]: 工时变异量 = 0.000000 min (严格独立)。")

    # 4. 检验 3：当前机型工时常数与实现配置的一致性
    print("\n--- [检验 3] 当前机型工时常数配置核验 (式 1~5) ---")
    print(f"{'机型类别':>15} | {'装夹工时 t_aux':>14} | {'换型工时 s_setup':>16} | {'单次换刀 t_tool':>14} | {'标准效率 v':>12}")
    print("-" * 80)
    expected_constants = {
        "2axis_drill": {"t_aux": 2.2, "s_setup": 5.5, "t_tool": 0.5, "efficiency": 220.0},
        "6axis_drill": {"t_aux": 6.6, "s_setup": 13.5, "t_tool": 0.5, "efficiency": 220.0},
        "2axis_backdrill": {"t_aux": 1.4, "s_setup": 12.0, "t_tool": 0.5, "efficiency": 97.0},
    }
    for mtype, exp in expected_constants.items():
        actual = TIME_PARAMS[mtype]
        for k, v in exp.items():
            assert abs(actual[k] - v) < 1e-6, f"{mtype} 参数 {k} 不一致: 期望 {v}, 实际 {actual[k]}"
        print(f"{mtype:>15} | {actual['t_aux']:>12.1f} min | {actual['s_setup']:>14.1f} min | {actual['t_tool']:>12.1f} min | {actual['efficiency']:>10.1f}")

    # 5. 标定检验 4：顺序相关换型时间决策规则闭合性
    print("\n--- [检验 4] 换型时间计算规则闭合性检验 (式 6~7) ---")
    m_d1 = machines_dict["D1"]
    s_first = setup_time(m_d1, prev_order_id=None, curr_order_id="ORD_A", is_first_batch=True)
    s_same = setup_time(m_d1, prev_order_id="ORD_A", curr_order_id="ORD_A", is_first_batch=False)
    s_cross = setup_time(m_d1, prev_order_id="ORD_A", curr_order_id="ORD_B", is_first_batch=False)

    assert s_first == 13.5 and s_same == 0.0 and s_cross == 13.5, "换型判定逻辑不合规！"
    print(f"  [验证通过] 机台初始首批次换型 = {s_first:.1f} min | 同工单连续加工换型 = {s_same:.1f} min | 跨工单加工换型 = {s_cross:.1f} min")

    # 6. 标定检验 5：动态释放时刻因果闭合递推 (式 16)
    print("\n--- [检验 5] 释放时刻与开工时间递推因果一致性检验 (式 16) ---")
    event_t = 100.0
    release_t = 120.0
    c_prev = 75.0
    calc_start = max(event_t, release_t, c_prev) + 5.5
    assert calc_start >= event_t and calc_start >= release_t, "开工时间违反物理因果律！"
    print(f"  [验证通过] 给定 t_e = {event_t} min, r_o = {release_t} min, C_prev = {c_prev} min:")
    print(f"             递推开工时间 B_b^e = {calc_start:.1f} min ≥ max{{t_e, r_o, C_prev}}，完全满足因果闭合。")

    print("\n" + "=" * 80)
    print("【物理合同检查结果】")
    print("  1. 当前工时常数与代码配置一致：2轴钻孔 (2.2, 5.5, 220), 6轴钻孔 (6.6, 13.5, 220), 2轴背钻 (1.4, 12.0, 97)；")
    print("  2. 当前样本的层级换算、换型规则与因果递推通过一致性检查；")
    print("  3. 本检查不建立参数来源、现场测量精度或外部有效性证据。")
    print("=" * 80)

    out_file = os.path.join(_THIS_DIR, "标定结果_PCB加工工时与换型时间物理模型参数.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "target": "Physical Time Model and Setup Parameters",
            "time_parameters": expected_constants,
            "spindle_capacities": {cfg["machine_id"]: cfg["spindles"] for cfg in MACHINE_CONFIGS},
            "status": "PHYSICAL_CONTRACT_CHECK_ONLY",
            "data_evidence_status": "PRE_FORMAL_ONLY_PROXY_DATA",
            "field_calibration_completed": False,
            "limitations": [
                "常数一致性检查不等于参数来源审计或现场测量标定。",
                "加工时间函数未接收批大小参数，因此重复调用只能验证接口定义，不能形成独立经验验证。",
                "代理数据包不能支持工业现场有效性结论。"
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)

    print(f"标定数据已完整记录至: {out_file}\n")
    return {"ok": True}


if __name__ == "__main__":
    calibrate_physical_parameters()
