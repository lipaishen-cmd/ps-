"""
ablation_common.py
==================
单 Actor 架构消融实验的基础评估与统计模块。
内部依赖模块：不作为 PyCharm 直接运行入口。
对齐《PCB_Dynamic_Model_Unified.docx》第 10.4 节与第 10.5 节规范。

功能说明：
  1. 路径解析与物理环境、统一解码及异构图强化学习模块绑定；
  2. 标准三组 18/6/6 数据集场景解析（在 val 验证集或 test 测试集上评估）；
  3. 双维度指标收集（调度效率 Cmax + 车间扰动度指标：真性旧批次迁移数、跨机迁移数、队列逆序数、开工偏差 Delta S、延误损失）；
  4. 消除幸存者偏差的透明统计（完整可行率、受控失败数、配对有效样本统计、95% 置信区间）；
  5. 顶刊标准 LaTeX / Markdown 三线表生成与逐事件遥测（event_records.jsonl）序列化。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. 严格路径解析与底层模块注册
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]

ABLATION_RESULTS_ROOT = PROJECT_ROOT / "result" / "消融实验"
C2_RESULTS_DIR = ABLATION_RESULTS_ROOT

# 主方法与唯一消融方法标识
PROPOSED_METHOD_ID = "AD-HG-DAPPO"
C2_PRIMARY_ABLATION_ID = "AD-HG-SAPPO"  # 唯一消融：层级双策略 PPO 到统一单策略 PPO


def ensure_ablation_result_tree() -> None:
    """创建主结果根目录和唯一单 Actor 消融结果目录。"""
    for result_dir in (ABLATION_RESULTS_ROOT, C2_RESULTS_DIR):
        result_dir.mkdir(parents=True, exist_ok=True)


for rel_path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "agent",
):
    path_str = str(rel_path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_models import POOL_D, Batch, DynamicInsertEvent, Order, Solution
from dynamic_event_manager import DynamicEventManager, DynamicRescheduleContext
from dynamic_two_agent_runner import DynamicDecision, DynamicTwoAgentRunner
from dynamic_validator import load_dynamic_orders_and_events
from hppo_trainer import HPPOTrainer


def resolve_independent_stage_protocol(
    stage: int,
    seed: int,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """返回与主方法完全一致的独立 Stage 训练/选模协议。

    每个 Stage 都从随机初始化开始，绝不继承上一 Stage 权重。训练预算取自
    每个独立 Stage 的 6 个场景各 300 episode 配额；每完成一轮 s1..s6 后更新一次 PPO。
    验证窗口严格复用主训练器的固定两场景协议。最后一个返回值是规模升序循环
    得到的场景索引序列，长度严格等于该 Stage 的 1800 episode 正式预算。
    """
    if stage not in (1, 2, 3):
        raise ValueError(f"独立训练 stage 必须为 1..3，收到 {stage!r}。")

    from dynamic_multi_scenario_trainer import (
        STAGE_VALIDATION_FILENAMES,
        compute_stage_only_quotas,
    )

    train_scenarios = resolve_evaluation_scenarios(split="train", stage=stage)
    quotas = compute_stage_only_quotas(stage_index=stage)
    if len(train_scenarios) != len(quotas):
        raise RuntimeError(
            f"Stage {stage} 训练场景数与正式配额数不一致："
            f"scenarios={len(train_scenarios)}, quotas={len(quotas)}。"
        )

    val_by_name = {
        Path(path).name: path
        for path in resolve_evaluation_scenarios(split="val", stage=stage)
    }
    expected_val_names = STAGE_VALIDATION_FILENAMES[stage]
    missing_val = [name for name in expected_val_names if name not in val_by_name]
    if missing_val:
        raise FileNotFoundError(
            f"Stage {stage} 固定验证窗口缺少场景: {missing_val}。"
        )
    val_scenarios = [val_by_name[name] for name in expected_val_names]

    # 严格按算例从小到大依次循环 (s1..s6)；更新节奏由训练器每 6 场景执行一次。
    schedule = [i % len(train_scenarios) for i in range(sum(quotas))]

    if len(schedule) != sum(quotas):
        raise RuntimeError("独立 Stage 训练调度长度与正式配额总和不一致。")
    return train_scenarios, val_scenarios, quotas, schedule


# ---------------------------------------------------------------------------
# 2. 评估数据结构定义
# ---------------------------------------------------------------------------
@dataclass
class EventEvaluationRecord:
    """单个动态事件决策级追踪遥测记录 (Event-level raw Actor telemetry)."""
    ablation_id: str                     # 消融变体唯一标识
    scenario_name: str                   # 场景名称 (如 scenario_val_000)
    event_id: str                        # 动态事件 ID
    event_type: str                      # 事件类型 (如 ARRIVE / COMPLETE)
    committed: bool                      # 该事件是否成功通过原子提交
    initial_k: int                       # 初始设定的作用域 K_scope
    effective_k: int                     # 经过自适应扩域重试后的实际生效 K_scope
    expansion_count: int                 # 触发扩域重试次数
    scope_ratio: float                   # 受影响域批次比例 |A| / |U_old|
    affected_uids_count: int             # 受影响域旧批次数量
    response_time_seconds: float         # 双 Actor 决策与解码耗时 (秒)
    seed: int = 1                        # 随机种子（正式统计逐一运行 Seed 1–5）
    cmax_before: float = 0.0             # 事件发生前的 Cmax (min)
    cmax_after: Optional[float] = None   # 决策提交后的 Cmax (min)
    failure_reason: Optional[str] = None # 失败原因 (若未提交)


@dataclass
class ScenarioEvaluationRecord:
    """单个动态场景在指定消融配置下的真实评测记录。"""
    ablation_id: str                     # 消融变体唯一标识
    category: str                        # 固定为唯一架构消融类别
    scenario_name: str                   # 场景名称 (如 scenario_val_002)
    scenario_path: str                   # 场景文件完整路径
    seed: int                            # 随机种子
    completed: bool                      # raw Actor 自主决策是否全部通过原子提交完成
    initial_cmax: float                  # 事件前初始方案的 Cmax
    final_cmax: Optional[float]          # 最终完成方案的 Cmax (若失败则为 None)
    cmax_delta: Optional[float]          # final_cmax - initial_cmax
    elapsed_seconds: float               # 场景求解端到端耗时 (秒)
    total_events: int                    # 总事件数
    committed_events: int                # 成功原子提交的事件数
    controlled_failures: int             # 受控失败数 (1 if not completed else 0)
    # 车间动态稳定性扰动指标 (式 10 与第 10.5 节规范)
    moved_old_batches: int               # 发生机台迁移或顺序变化的既有旧批次数（消除插入平移误差）
    machine_reassigned_batches: int      # 跨机台物理搬移的旧批次数
    queue_order_inversions: int          # 同一机台队列内原本前后的两个旧批次发生相对次序倒置的对数
    mean_start_time_deviation: float     # 旧批次开工时刻平均绝对偏差 Delta S (min)
    total_delay_loss: float              # 旧批次正向延误时刻累积量 (min)
    # 作用域与动态扩域诊断记录
    initial_k_scope: int                 # 初始设定的 K_scope
    effective_k_scope_mean: float        # 考虑 ΔK 扩域后的实际平均作用域大小
    expansion_count_total: int           # 触发确定性扩域 (ΔK) 的总次数
    actual_affected_ratio: float         # 实际进入受影响域的批次比例 |A| / |U_old|
    failure_details: Optional[str] = None


@dataclass
class AblationSummaryStatistics:
    """消除幸存者偏差的变体汇总统计量。"""
    ablation_id: str
    category: str
    total_evaluated_scenarios: int
    completed_scenarios: int
    raw_feasibility_rate: float          # 完整可行率: completed / total
    controlled_failure_total: int
    # 完工时间统计 (仅在完成场景上统计，且显式披露有效样本数)
    valid_cmax_count: int
    mean_cmax: Optional[float]
    std_cmax: Optional[float]
    median_cmax: Optional[float]
    iqr_cmax: Optional[float]
    ci95_cmax: Optional[Tuple[float, float]]
    # 稳定性与效率指标
    mean_moved_batches: float
    mean_machine_reassignments: float
    mean_queue_inversions: float
    mean_start_time_deviation: float
    mean_total_delay_loss: float
    mean_elapsed_seconds: float
    mean_effective_k_scope: float
    total_expansions: int


# ---------------------------------------------------------------------------
# 3. 场景加载与环境验证
# ---------------------------------------------------------------------------
def resolve_evaluation_scenarios(
    split: str = "val",
    stage: Optional[int] = None,
    custom_scenario: Optional[str] = None,
) -> List[str]:
    """解析待评测的场景文件，严格绑定活动数据 Manifest。"""
    if custom_scenario:
        p = Path(custom_scenario).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"指定的算例文件不存在: {p}")
        return [str(p)]

    from dynamic_multi_scenario_trainer import load_active_dataset_contract

    contract = load_active_dataset_contract()
    manifest = contract["manifest"]
    dataset_dir = Path(str(contract["resolved_dataset_dir"]))
    records = list(manifest["splits"].get(split, []))
    if not records:
        raise ValueError(f"不支持的 split: {split} (必须是 val / test / train)")

    if split == "train":
        records.sort(
            key=lambda r: (
                int(r["initial_orders"]),
                int(r["dynamic_events"]),
                str(r["scenario_id"]),
            )
        )
        if stage is not None:
            if stage not in (1, 2, 3):
                raise ValueError(f"无效的 stage: {stage} (必须为 1..3)")
            start = (stage - 1) * 6
            records = records[start : start + 6]
    elif split in ("val", "test"):
        if stage is not None:
            if stage not in (1, 2, 3):
                raise ValueError(f"无效的 stage: {stage} (必须为 1..3)")
            start = (stage - 1) * 2
            records = records[start : start + 2]

    files = [str((dataset_dir / str(r["filepath"])).resolve()) for r in records]
    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"活动场景文件缺失: {missing}")
    return files


def load_hppo_trainer(
    checkpoint_path: str | Path,
    initial_orders: Sequence[Order],
    device: torch.device = torch.device("cpu"),
) -> HPPOTrainer:
    """加载已训练的 AD-HG-DAPPO 检查点，确保 load_optimizer=False。"""
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"检查点不存在: {ckpt}")

    trainer = HPPOTrainer(orders=initial_orders, pool_type=POOL_D, device=device)
    payload = trainer.load_checkpoint(str(ckpt), load_optimizer=False)
    trainer.checkpoint_payload = payload
    return trainer


# ---------------------------------------------------------------------------
# 4. 车间动态稳定性扰动指标计算 (消除插入平移误差)
# ---------------------------------------------------------------------------
def compute_perturbation_metrics(
    initial_solution: Solution,
    final_solution: Solution,
    initial_uids: Set[int],
) -> Tuple[int, int, int, float, float]:
    """
    严格计算相对初始基准方案的扰动度（Shop-floor Nervousness）：
      1. moved_batches: 发生跨机迁移，或在同机旧批次投影序列中相对次序发生变化的既有旧批次数（消除新插入批次导致的全局索引平移虚高）；
      2. reassigned_batches: 发生跨机迁移的旧批次数；
      3. inversions: 同机队列内原本前后的两个旧批次发生相对次序倒置的对数；
      4. mean_start_time_deviation: 旧批次开工时刻平均绝对偏差 Delta S (min)；
      5. total_delay_loss: 旧批次正向延误时刻累积量 (min)。
    """
    if not initial_solution or not final_solution or not initial_uids:
        return 0, 0, 0, 0.0, 0.0

    # 1. 提取旧批次在初始解中的机台归属、同机排位投影与开工时刻
    init_map: Dict[int, Tuple[str, int, float]] = {}
    init_proj: Dict[str, List[int]] = {}
    for mid, q in initial_solution.machine_queues.items():
        init_proj[mid] = []
        for b in q.batches:
            if b.u_id in initial_uids:
                init_map[b.u_id] = (mid, len(init_proj[mid]), float(b.start_time if b.start_time is not None else 0.0))
                init_proj[mid].append(b.u_id)

    # 2. 提取旧批次在最终解中的机台归属、同机排位投影与开工时刻
    final_map: Dict[int, Tuple[str, int, float]] = {}
    final_proj: Dict[str, List[int]] = {}
    for mid, q in final_solution.machine_queues.items():
        final_proj[mid] = []
        for b in q.batches:
            if b.u_id in initial_uids:
                final_map[b.u_id] = (mid, len(final_proj[mid]), float(b.start_time if b.start_time is not None else 0.0))
                final_proj[mid].append(b.u_id)

    common_uids = set(init_map.keys()).intersection(final_map.keys())
    if not common_uids:
        return 0, 0, 0, 0.0, 0.0

    reassigned = 0
    moved = 0
    start_devs: List[float] = []
    delays: List[float] = []

    for uid in common_uids:
        m_i, p_i_proj, s_i = init_map[uid]
        m_f, p_f_proj, s_f = final_map[uid]

        dev = abs(s_f - s_i)
        start_devs.append(dev)
        delays.append(max(0.0, s_f - s_i))

        if m_i != m_f:
            reassigned += 1
            moved += 1
        else:
            # 仅在旧批次自身序列中比较相对排位，排除新插入批次导致的挤占位移
            if p_i_proj != p_f_proj:
                moved += 1

    # 3. 统计同机且均保留在该机台上的旧批次逆序对数
    inversions = 0
    uid_list = sorted(common_uids)
    for i in range(len(uid_list)):
        u1 = uid_list[i]
        m1_i, p1_i, _ = init_map[u1]
        m1_f, p1_f, _ = final_map[u1]
        for j in range(i + 1, len(uid_list)):
            u2 = uid_list[j]
            m2_i, p2_i, _ = init_map[u2]
            m2_f, p2_f, _ = final_map[u2]
            if m1_i == m2_i and m1_f == m2_f and m1_i == m1_f:
                if (p1_i < p2_i) != (p1_f < p2_f):
                    inversions += 1

    mean_dev = float(np.mean(start_devs)) if start_devs else 0.0
    total_delay = float(np.sum(delays)) if delays else 0.0
    return moved, reassigned, inversions, mean_dev, total_delay


# ---------------------------------------------------------------------------
# 5. 消除幸存者偏差的汇总统计函数
# ---------------------------------------------------------------------------
def compute_summary_statistics(
    records: List[ScenarioEvaluationRecord],
) -> AblationSummaryStatistics:
    """计算单个变体的严格汇总统计量，包含中位数、IQR 与 95% 置信区间。"""
    if not records:
        raise ValueError("记录列表为空，无法计算统计量。")

    ablation_id = records[0].ablation_id
    category = records[0].category
    total_cnt = len(records)
    completed_recs = [r for r in records if r.completed and r.final_cmax is not None]
    comp_cnt = len(completed_recs)
    feas_rate = comp_cnt / max(1, total_cnt)
    failures = sum(r.controlled_failures for r in records)

    cmax_vals = [r.final_cmax for r in completed_recs]
    if cmax_vals:
        mean_cmax = float(np.mean(cmax_vals))
        std_cmax = float(np.std(cmax_vals, ddof=1)) if len(cmax_vals) > 1 else 0.0
        median_cmax = float(np.median(cmax_vals))
        q75, q25 = np.percentile(cmax_vals, [75, 25])
        iqr_cmax = float(q75 - q25)
        if len(cmax_vals) > 1:
            sem = std_cmax / math.sqrt(len(cmax_vals))
            ci95 = (mean_cmax - 1.96 * sem, mean_cmax + 1.96 * sem)
        else:
            ci95 = (mean_cmax, mean_cmax)
    else:
        mean_cmax = std_cmax = median_cmax = iqr_cmax = ci95 = None

    mean_moved = float(np.mean([r.moved_old_batches for r in completed_recs])) if completed_recs else 0.0
    mean_reassigned = float(np.mean([r.machine_reassigned_batches for r in completed_recs])) if completed_recs else 0.0
    mean_invs = float(np.mean([r.queue_order_inversions for r in completed_recs])) if completed_recs else 0.0
    mean_dev = float(np.mean([r.mean_start_time_deviation for r in completed_recs])) if completed_recs else 0.0
    mean_delay = float(np.mean([r.total_delay_loss for r in completed_recs])) if completed_recs else 0.0

    mean_time = float(np.mean([r.elapsed_seconds for r in records]))
    mean_eff_k = float(np.mean([r.effective_k_scope_mean for r in records]))
    tot_exp = sum(r.expansion_count_total for r in records)

    return AblationSummaryStatistics(
        ablation_id=ablation_id,
        category=category,
        total_evaluated_scenarios=total_cnt,
        completed_scenarios=comp_cnt,
        raw_feasibility_rate=feas_rate,
        controlled_failure_total=failures,
        valid_cmax_count=len(cmax_vals),
        mean_cmax=mean_cmax,
        std_cmax=std_cmax,
        median_cmax=median_cmax,
        iqr_cmax=iqr_cmax,
        ci95_cmax=ci95,
        mean_moved_batches=mean_moved,
        mean_machine_reassignments=mean_reassigned,
        mean_queue_inversions=mean_invs,
        mean_start_time_deviation=mean_dev,
        mean_total_delay_loss=mean_delay,
        mean_elapsed_seconds=mean_time,
        mean_effective_k_scope=mean_eff_k,
        total_expansions=tot_exp,
    )


# ---------------------------------------------------------------------------
# 6. 事件级遥测记录序列化支持
# ---------------------------------------------------------------------------
def save_event_records(records: List[EventEvaluationRecord], output_path: str | Path) -> None:
    """将逐事件细粒度决策遥测持久化为 JSON Lines 文件。"""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 7. 学术三线表生成器 (Markdown & LaTeX)
# ---------------------------------------------------------------------------
def format_ablation_markdown_table(
    summaries: List[AblationSummaryStatistics],
    anchor_id: str = PROPOSED_METHOD_ID,
) -> str:
    """生成包含可行率、有效样本数与扰动指标的学术级 Markdown 表格。"""
    lines = [
        "| 变体名称 (Variant) | 维度分类 | 完整可行率 (%) | 有效/总场景数 | 完工时间 $C_{\\max}$ (min) | 相对增量 (Gap %) | 扰动批次数 | 跨机迁移数 | 逆序数 | 开工偏差 $\\Delta S$ (min) | 响应时间 (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    anchor_cmax: Optional[float] = None
    for s in summaries:
        if s.ablation_id == anchor_id and s.mean_cmax is not None:
            anchor_cmax = s.mean_cmax
            break

    for s in summaries:
        is_anchor = (s.ablation_id == anchor_id)
        feas_str = f"{s.raw_feasibility_rate * 100.0:.1f}%"
        count_str = f"{s.completed_scenarios}/{s.total_evaluated_scenarios}"
        if s.mean_cmax is not None:
            cmax_str = f"{s.mean_cmax:.2f} ± {s.std_cmax:.2f}" if s.std_cmax else f"{s.mean_cmax:.2f}"
            if anchor_cmax is not None and not is_anchor:
                gap = ((s.mean_cmax - anchor_cmax) / anchor_cmax) * 100.0
                gap_str = f"{gap:+.2f}%"
            else:
                gap_str = "—"
        else:
            cmax_str = "N/A (无完整解)"
            gap_str = "N/A"

        moved = f"{s.mean_moved_batches:.1f}"
        reassigned = f"{s.mean_machine_reassignments:.1f}"
        invs = f"{s.mean_queue_inversions:.1f}"
        dev = f"{s.mean_start_time_deviation:.2f}"
        time_str = f"{s.mean_elapsed_seconds:.2f}"

        prefix = "**" if is_anchor else ""
        suffix = "**" if is_anchor else ""
        lines.append(
            f"| {prefix}{s.ablation_id}{suffix} | {s.category} | {feas_str} | {count_str} | {cmax_str} | {gap_str} | {moved} | {reassigned} | {invs} | {dev} | {time_str} |"
        )

    return "\n".join(lines)


def generate_ablation_latex_table(
    summaries: List[AblationSummaryStatistics],
    anchor_id: str = PROPOSED_METHOD_ID,
) -> str:
    """生成符合顶刊规范 (IEEE Transactions / Elsevier) 的综合消融实验 LaTeX 三线表。"""
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation Study Results on the Representative Large-Scale Dynamic PCB Benchmarks (Stage 3)}",
        r"\label{tab:master_ablation_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Variant Configuration} & \textbf{Feasible (\%)} & \textbf{Valid / Total} & \textbf{Makespan $C_{\max}$ (min)} & \textbf{Gap (\%)} & \textbf{Moved Batches} & \textbf{Inversions} & \textbf{$\Delta S$ (min)} & \textbf{Time (s)} \\",
        r"\midrule",
    ]

    anchor_cmax: Optional[float] = None
    for s in summaries:
        if s.ablation_id == anchor_id and s.mean_cmax is not None:
            anchor_cmax = s.mean_cmax
            break

    last_cat = ""
    for s in summaries:
        cat = s.category
        name = s.ablation_id
        feas_str = f"{s.raw_feasibility_rate * 100.0:.1f}\\%"
        cnt_str = f"{s.completed_scenarios} / {s.total_evaluated_scenarios}"

        if s.mean_cmax is not None:
            cmax_val = s.mean_cmax
            cmax_std = s.std_cmax if s.std_cmax else 0.0
            cmax_str = f"{cmax_val:.2f} $\\pm$ {cmax_std:.2f}"
            gap_str = f"{((cmax_val - anchor_cmax) / anchor_cmax * 100.0):+.2f}\\%" if (anchor_cmax and name != anchor_id) else "---"
        else:
            cmax_str = "N/A"
            gap_str = "N/A"

        moved = f"{s.mean_moved_batches:.1f}"
        invs = f"{s.mean_queue_inversions:.1f}"
        dev = f"{s.mean_start_time_deviation:.2f}"
        elapsed = f"{s.mean_elapsed_seconds:.2f}"

        if cat != last_cat and last_cat != "":
            latex_lines.append(r"\midrule")
        last_cat = cat

        clean_cat = cat.replace("&", r"\&")
        clean_name = name.replace("_", r"\_").replace("&", r"\&")
        if name == anchor_id:
            latex_lines.append(
                f"\\textbf{{{clean_cat}}} & \\textbf{{{clean_name}}} & \\textbf{{{feas_str}}} & \\textbf{{{cnt_str}}} & \\textbf{{{cmax_str}}} & \\textbf{{{gap_str}}} & \\textbf{{{moved}}} & \\textbf{{{invs}}} & \\textbf{{{dev}}} & \\textbf{{{elapsed}}} \\\\"
            )
        else:
            latex_lines.append(
                f"{clean_cat} & {clean_name} & {feas_str} & {cnt_str} & {cmax_str} & {gap_str} & {moved} & {invs} & {dev} & {elapsed} \\\\"
            )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(latex_lines)
