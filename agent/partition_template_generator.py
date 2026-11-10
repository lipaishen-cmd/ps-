"""
partition_template_generator.py
===============================
规范可行整数划分拆批模板生成器（规范 8.2 节、18.2 节）

功能说明：
  根据工单需求量生成满足非递增排序与机床容量约束的候选拆批模板集合 P_i。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "environment"))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in [_THIS_DIR, _ENVIRONMENT_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_models import Machine, Order
from eligibility_builder import ALL_MACHINES


def get_order_max_spindles(
    order: Order,
    machines: Dict[str, Machine] = ALL_MACHINES,
) -> int:
    """获取该订单合格且可用机床中的最大物理主轴容量"""
    eligible_spindles = [
        machines[m_id].spindles
        for m_id in order.eligible_machines
        if m_id in machines
    ]
    if not eligible_spindles:
        return 0
    return max(eligible_spindles)


def _integer_partitions(n: int, max_val: int) -> List[Tuple[int, ...]]:
    """递归生成 n 的非增正整数划分，且单项不超过 max_val"""
    results: List[Tuple[int, ...]] = []

    def backtrack(remaining: int, current_max: int, current_p: List[int]):
        if remaining == 0:
            results.append(tuple(current_p))
            return
        limit = min(remaining, current_max)
        for val in range(limit, 0, -1):
            current_p.append(val)
            backtrack(remaining - val, val, current_p)
            current_p.pop()

    backtrack(n, max_val, [])
    return results


def generate_partition_templates(
    order: Order,
    max_templates: int = 16,
    machines: Dict[str, Machine] = ALL_MACHINES,
) -> List[Tuple[int, ...]]:
    """
    为订单生成规范可行拆分模板集合 P_i (对齐参数规范 3.1 节)。
    
    数学定义（公式 8-1 / 参数规范 3.1 节）：
      P_i = { p = (q_1, ..., q_{|p|}) | sum(q_r) = W_i, q_1 >= ... >= q_{|p|} >= 1, forall r, exists m in M_i, C_m >= q_r }
      
    超额截断规则（三级优先级）：
      1. 子批数量 n_k 升序（大批次优先，减少换型与装夹次数）；
      2. 首子批大小 b_{i1} 降序（主轴满载优先）；
      3. 序列数值字典序升序。
    """
    max_cap = get_order_max_spindles(order, machines)
    if max_cap <= 0:
        return []
    raw_partitions = _integer_partitions(order.W, max_cap)

    # 排序：严格按参数规范 3.1 节 (len(p), -p[0], p)
    def sort_key(p: Tuple[int, ...]):
        return (len(p), -p[0], p)

    sorted_partitions = sorted(raw_partitions, key=sort_key)

    # 截断保留前 max_templates 个核心模板
    return sorted_partitions[:max_templates]
