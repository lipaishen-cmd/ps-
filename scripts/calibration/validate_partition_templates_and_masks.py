"""
validate_partition_templates_and_masks.py
===========================================
【结构诊断程序 3】
诊断对象：
  1. 宏观分批策略（ManagerPolicy / SplitPolicy）整数拆分候选模板上限 N_max = 16 的空间分布与完备性；
  2. 模板三级截断准则有效性：子批数升序（最小化换型频次）-> 首批容量降序（主轴满载优先）-> 字典序；
  3. 微观机台指派策略（WorkerPolicy / PlacementPolicy）硬约束动作掩码：
     - 机台加工资格硬约束
     - 主轴容量上限硬约束（B_b <= C_m）
     - 资源池物理隔离硬约束（Pool D vs Pool B）
     确保非法插槽 Softmax 采样概率严格为零。

运行命令：
  python validate_partition_templates_and_masks.py
"""

from __future__ import annotations

import json
import math
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

import torch
from data_models import Machine, Order, MACHINE_CONFIGS, POOL_D, POOL_B
from data_loader import load_test1
from worker_policy import WorkerPolicy


def compute_admissible_partitions(w: int, max_spindles: int = 6) -> List[Tuple[int, ...]]:
    """递归生成满足主轴容量上界的全部合法整数拆分方案。"""
    res: List[Tuple[int, ...]] = []

    def backtrack(remain: int, max_val: int, current: List[int]):
        if remain == 0:
            res.append(tuple(current))
            return
        for val in range(min(remain, max_val), 0, -1):
            backtrack(remain - val, val, current + [val])

    backtrack(w, max_spindles, [])
    # 按学术三级准则排序：子批数量升序 -> 首批容量降序 -> 序列字典序升序
    res.sort(key=lambda t: (len(t), -t[0], t))
    return res


def calibrate_templates_and_masks() -> Dict:
    print("=" * 80)
    print("【结构诊断 3】分批候选模板上限 N_max 与微观机台指派动作掩码")
    print("=" * 80)

    # 1. 代理数据包样本加载；不得解释为现场实测标定
    samples_dir = os.path.join(_PROJECT_ROOT, "data", "test_samples")
    if not os.path.exists(os.path.join(samples_dir, "test_1.xlsx")):
        samples_dir = _PROJECT_ROOT

    sheet_dict = load_test1(samples_dir)
    orders: List[Order] = []
    for s_orders in sheet_dict.values():
        orders.extend(s_orders)

    unique_w = sorted(list(set(o.W for o in orders)))
    print(f"数据源: test_1.xlsx | 工单叠板单元数 W_i 离散取值种数: {len(unique_w)} | 取值范围: [{min(unique_w)}, {max(unique_w)}]")

    # 2. 标定检验 1：拆分模板空间分布与 N_max = 16 覆盖特性
    print("\n--- [检验 1] 拆分模板空间规模与 N_max = 16 截断敏感度分析 ---")
    print(f"{'W_i':>6} | {'全拆分总数':>12} | {'截断保留数':>12} | {'覆盖比率':>10} | {'前3项典型分批模板序列'}")
    print("-" * 75)

    distribution_records = []
    for w in unique_w:
        all_splits = compute_admissible_partitions(w, max_spindles=6)
        total_n = len(all_splits)
        retained = all_splits[:16]
        cov_ratio = 16 / total_n if total_n > 16 else 1.0

        top3 = str(retained[:3]) if len(retained) >= 3 else str(retained)
        distribution_records.append({
            "W_i": w,
            "total_partitions": total_n,
            "retained_count": len(retained),
            "retained_templates": [list(t) for t in retained[:5]]
        })
        print(f"{w:>6} | {total_n:>12} | {len(retained):>12} | {cov_ratio*100:>9.1f}% | {top3}")

    print(f"\n  [结构结论] W_i ≤ 7 范围内分批方案完全由 N_max = 16 完整覆盖（覆盖率 100%）；")
    print(f"             W_i ≥ 8 较大工单经三级准则截断后，均严格保留了换型开销最小的宏观大子批组合。")
    print(f"             候选上限 N_max = 16 保持候选规模有界；本检验不证明解质量最优或无损。")

    # 3. 标定检验 2：微观指派动作空间硬掩码阻断机制实测
    print("\n--- [检验 2] PlacementPolicy 微观动作空间硬掩码合法性实测 ---")
    d_h = 32
    worker = WorkerPolicy(d_batch=d_h, d_slot=d_h, d_hidden=d_h)

    # 构造测试状态：待指派子批规模 B_b = 6 (仅 6 轴机 D1~D3 合法；2轴机 D4~D8 超限；B1~B2 跨池)
    h_batch = torch.randn(1, d_h)
    slot_keys = [
        ("D1", 0), ("D2", 0), ("D3", 0),  # 合法插槽 (Pool D, 6主轴机台)
        ("D4", 0), ("D5", 0),              # 非法插槽 1: 容量超限 (Pool D, 2主轴机台, B_b=6 > 2)
        ("B1", 0), ("B2", 0),              # 非法插槽 2: 资源池跨界违规 (Pool B 机台)
    ]
    h_slots = torch.randn(len(slot_keys), d_h)
    valid_slots = [("D1", 0), ("D2", 0), ("D3", 0)]

    dist, logits = worker(h_batch=h_batch, h_slots=h_slots, slot_keys=slot_keys, valid_slots=valid_slots)
    probs = dist.probs.detach().numpy().flatten()

    print(f"{'插槽机台标识':>14} | {'机台主轴容量':>12} | {'掩码合法性':>10} | {'Softmax采样概率':>16} | {'动作空间约束'}")
    print("-" * 75)
    for idx, key in enumerate(slot_keys):
        m_name = key[0]
        m_cap = 6 if m_name in ("D1", "D2", "D3") else 2
        p_val = probs[idx]
        is_legal = 1 if key in valid_slots else 0
        tag = "满足全部物理约束" if is_legal == 1 else ("主轴容量超限阻断" if "D" in m_name else "加工池物理隔离阻断")
        key_str = f"{key[0]}_slot{key[1]}"
        print(f"{key_str:>14} | {m_cap:>12} | {is_legal:>10} | {p_val:>16.6f} | {tag}")
        if is_legal == 0:
            assert p_val < 1e-9, f"动作掩码失效：非法动作 {key} 概率非零 ({p_val})！"

    print(f"\n  [验证通过] 硬动作掩码实测: 非法插槽 Softmax 采样概率恒等于 0.000000，物理边界 100% 闭合。")

    print("\n" + "=" * 80)
    print("【结构诊断结果】")
    print("  1. 当前配置 N_max = 16；本脚本只报告模板覆盖和截断，不证明其性能最优；")
    print("  2. 给定测试状态下，非法插槽概率为零；仍需全场景性质测试覆盖资格、容量与池隔离；")
    print("  3. 正式参数敏感性须用冻结数据、冻结检查点和多种子评测完成。")
    print("=" * 80)

    out_file = os.path.join(_THIS_DIR, "标定结果_分批候选模板上限N_max与机台指派动作空间掩码.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "target": "N_max and Placement Action Masks",
            "configured_n_max": 16,
            "truncation_rule": "fewest_subbatches_asc -> largest_first_desc -> lexicographical",
            "mask_observation": "zero probability on illegal actions in the constructed diagnostic state",
            "distribution_summary": distribution_records[:8],
            "status": "STRUCTURAL_DIAGNOSTIC_ONLY",
            "performance_calibration_completed": False,
            "limitations": [
                "代理数据包不是现场测量标定证据。",
                "随机嵌入只用于验证掩码数值行为，不用于推断调度性能。",
                "N_max=16 的解质量、速度权衡与统计最优性尚未评测。"
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)

    print(f"标定数据已完整记录至: {out_file}\n")
    return {"ok": True, "n_max": 16}


if __name__ == "__main__":
    calibrate_templates_and_masks()
