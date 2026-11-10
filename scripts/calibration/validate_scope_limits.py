"""
validate_scope_limits.py
========================
【参数标定程序 1】
标定对象：
  1. 受影响域旧批次规模上限 K_scope（式 9）：检查作用域规模、扰动比例与选择器计算延迟；
  2. 初始候选子集规模 K_seed 与队列紧邻闭包扩展（Queue Adjacency Closure）参数配比；
  3. 参数域边界：K_scope = 0（不选择旧批次）与 K_scope >= |U_old|（纳入全部未冻结旧批次）。

运行命令：
  python validate_scope_limits.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ENVIRONMENT_DIR = os.path.join(_PROJECT_ROOT, "environment")
_AGENT_DIR = os.path.join(_PROJECT_ROOT, "agent")

for p in (_PROJECT_ROOT, _ENVIRONMENT_DIR, _AGENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_models import Batch, Machine, Order, Solution, MACHINE_CONFIGS, POOL_D
from dynamic_validator import load_dynamic_orders_and_events
from solution_lifecycle import create_empty_solution
from decoder import full_decode
from scope_selector import select_affected_domain


def construct_initial_schedule(orders: List[Order], machines_dict: Dict[str, Machine], pool_type: str) -> Solution:
    """构建确定性初始基准排程（基于需求守恒与容量约束）。"""
    sol = create_empty_solution(pool_type)
    orders_dict = {o.order_id: o for o in orders}
    m_keys = [m for m in machines_dict if "D" in m]
    m_idx = 0
    u_counter = 1

    for o in orders:
        if o.process_type != "钻孔":
            continue
        rem = o.W
        while rem > 0:
            b_size = min(2, rem)
            target_m = m_keys[m_idx % len(m_keys)]
            b = Batch(
                order_id=o.order_id,
                u_id=u_counter,
                stack_count=b_size,
                machine_id=target_m,
            )
            sol.machine_queues[target_m].batches.append(b)
            u_counter += 1
            rem -= b_size
            m_idx += 1

    ms = full_decode(sol, orders_dict=orders_dict)
    sol.makespan = ms
    return sol


def calibrate_scope_parameters() -> Dict:
    print("=" * 80)
    print("【参数标定 1】受影响域规模上限 K_scope 与初始子集规模 K_seed 灵敏度标定")
    print("=" * 80)

    scenario_path = os.path.join(_PROJECT_ROOT, "data", "dynamic_scenarios", "val", "scenario_val_000.xlsx")
    init_orders, dyn_events = load_dynamic_orders_and_events(scenario_path, "Sheet1")
    pool_type = POOL_D
    machines_dict = {cfg["machine_id"]: Machine(**cfg) for cfg in MACHINE_CONFIGS if cfg["pool_type"] == pool_type}
    orders_dict = {o.order_id: o for o in init_orders}
    for e in dyn_events:
        orders_dict[e.order.order_id] = e.order

    base_sol = construct_initial_schedule(init_orders, machines_dict, pool_type)
    init_ms = base_sol.makespan
    total_batches = sum(len(q.batches) for q in base_sol.machine_queues.values())
    print(f"输入实例: scenario_val_000.xlsx | 初始订单数: {len(init_orders)} | 批次总数: {total_batches} | 初始完工时间: {init_ms:.2f} min")

    event_0 = dyn_events[0]
    t_e = float(event_0.actual_arrival_time)
    print(f"动态事件: {event_0.event_id} | 到达时刻 t_e = {t_e:.2f} min | 新到达工单: {event_0.order.order_id} (W_i = {event_0.order.W})")

    frozen_uids = frozenset(b.u_id for q in base_sol.machine_queues.values() for b in q.batches if float(b.start_time) < t_e)
    unfrozen_total = total_batches - len(frozen_uids)
    print(f"时空划分: 冻结批次集合规模 |F_g^e| = {len(frozen_uids)} | 既有未开始批次规模 |U_old,g^e| = {unfrozen_total}")

    # 1. 扫描 K_scope 候选值
    k_scope_candidates = [0, 2, 5, 8, 10, 15, unfrozen_total]
    scan_results = []

    print("\n--- [实验 1] K_scope 离散谱系扫描评估 (固定 K_seed = 5) ---")
    print(f"{'K_scope':>8} | {'受影响批次数':>12} | {'受影响域占比':>12} | {'选择器时延(ms)':>14}")
    print("-" * 80)

    for k in k_scope_candidates:
        t0 = time.perf_counter()
        scope_res = select_affected_domain(
            solution=base_sol,
            event_time=t_e,
            frozen_uids=frozen_uids,
            new_orders=[event_0.order],
            orders_dict=orders_dict,
            machines_dict=machines_dict,
            k_scope=k,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_affected = len(scope_res.affected_uids)
        ratio = scope_res.scope_ratio

        scan_results.append({
            "k_scope": k,
            "k_seed": 5,
            "affected_count": n_affected,
            "scope_ratio": round(ratio, 4),
            "latency_ms": round(elapsed_ms, 3),
            "makespan": None,
            "improvement_pct": None,
            "evidence_scope": "scope_selector_structural_diagnostic_only"
        })

        k_repr = str(k) if k < unfrozen_total else f"{k}(全未冻结)"
        print(f"{k_repr:>8} | {n_affected:>12} | {ratio*100:>11.1f}% | {elapsed_ms:>14.3f}")

    # 2. 扫描 K_seed 候选值
    print("\n--- [实验 2] 初始子集规模 K_seed 灵敏度分析 (固定 K_scope = 8) ---")
    print(f"{'K_seed':>8} | {'初始种子批次':>12} | {'闭包扩展批次':>12} | {'最终受影响规模':>14} | {'结构特性评估'}")
    print("-" * 75)
    for s in [1, 3, 5, 8]:
        scope_res = select_affected_domain(
            solution=base_sol,
            event_time=t_e,
            frozen_uids=frozen_uids,
            new_orders=[event_0.order],
            orders_dict=orders_dict,
            machines_dict=machines_dict,
            k_scope=8,
            k_seed=s,
        )
        n_seed = min(s, len(scope_res.affected_uids))
        n_clos = len(scope_res.affected_uids) - n_seed
        eval_desc = "当前配置值" if s == 5 else ("种子覆盖较小，闭包扩展占比较高" if s < 5 else "种子覆盖较大，闭包扩展空间较小")
        print(f"{s:>8} | {n_seed:>12} | {n_clos:>12} | {len(scope_res.affected_uids):>14} | {eval_desc}")

    # 3. 结构诊断结论；不得将本脚本包装成性能标定或 Pareto 证据。
    print("\n" + "=" * 80)
    print("【受影响域结构诊断结果】")
    print("  1. 当前配置为 K_scope=8、K_seed=5、ΔK=4；本脚本只验证选择器作用域规模、占比与计算延迟；")
    print("  2. 本脚本不生成调度候选、不运行 Actor/基线，也不提供 Cmax、改善率或 Pareto 最优性证据；")
    print("  3. 正式参数敏感性仍须在固定 Manifest、冻结检查点和多场景/多种子协议下重新执行。")
    print("=" * 80)

    out_file = os.path.join(_THIS_DIR, "标定结果_受影响域规模上限K_scope与初始子集规模K_seed.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "target": "ScopeSelector Parameters",
            "configured_k_scope": 8,
            "configured_k_seed": 5,
            "configured_delta_k": 4,
            "experimental_evidence": scan_results,
            "status": "STRUCTURAL_DIAGNOSTIC_ONLY",
            "configured_values": {"k_scope": 8, "k_seed": 5, "delta_k": 4},
            "performance_sensitivity_completed": False,
            "limitations": "No schedule candidate or policy was evaluated; makespan and improvement are intentionally null.",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)

    print(f"标定数据已完整记录至: {out_file}\n")
    return {"ok": True, "status": "STRUCTURAL_DIAGNOSTIC_ONLY", "k_scope": 8, "k_seed": 5}


if __name__ == "__main__":
    calibrate_scope_parameters()
