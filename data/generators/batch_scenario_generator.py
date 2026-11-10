"""
batch_scenario_generator.py
===========================
批量动态调度场景生成器（支持训练集/验证集/测试集体系化生成）

功能说明：
  1. 读取源 Excel 文件（test_1.xlsx / test_2.xlsx 等），不修改源文件。
  2. 多维度受控生成：
     - 插单时刻：支持相对初始 Cmax 比例（如 0.08~0.75）或绝对时间区间；
       支持均匀分布 (uniform)、泊松/指数间隔到达 (poisson)、脉冲突发到达 (burst)。
     - 插单规模：支持轻量 (1~3单)、中量 (4~7单)、重量 (8~12单) 等离散档位或自定义范围；
       工艺参数真实符合正整数要求与物理关系 D1: W = ceil(Q / S)。
     - 优先级分布：高 / 中 / 普通，支持自定义权重分布（均衡/高急/常态）。
     - 预报误差形态：准时 (on-time)、早到 (early)、晚到 (delayed)、突发未预报 (unforecasted)。
  3. 双资源池隔离与工步匹配：普通钻孔订单限制 Pool D，背钻订单限制 Pool B。
  4. 自动全量运行 dynamic_validator.validate_dynamic_excel 自检，确保 100% 合规。
  5. 批量生成 train / val / test 数据集与 dataset_manifest.json 元数据清单。
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# 将生成器目录、项目根目录与环境模块加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ENVIRONMENT_DIR = os.path.join(_PROJECT_ROOT, "environment")
for _p in (_THIS_DIR, _PROJECT_ROOT, _ENVIRONMENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_models import (
    DEFAULT_H_MAP,
    POOL_B,
    POOL_D,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_NORMAL,
    PROCESS_BACKDRILL,
    PROCESS_DRILL,
    STATUS_PENDING,
    UNFORECASTED_TAG,
    VALID_EFFICIENCIES,
    VALID_PRIORITIES,
    Order,
)
from dynamic_validator import (
    DYNAMIC_REQUIRED_FIELDS,
    INITIAL_REQUIRED_FIELDS,
    ValidationError,
    validate_dynamic_excel,
)
from eligibility_builder import (
    ALL_MACHINES,
    POOL_B_MACHINES,
    POOL_D_MACHINES,
    build_eligibility_matrix,
    validate_cross_pool_isolation,
)


# ---------------------------------------------------------------------------
# 样式定义常量
# ---------------------------------------------------------------------------
_FONT_HEADER = Font(name="Microsoft YaHei", size=11, bold=True, color="1F497D")
_FONT_DATA = Font(name="Microsoft YaHei", size=10)
_FILL_INITIAL_NEW = PatternFill(start_color="EDF2F8", end_color="EDF2F8", fill_type="solid")
_FILL_DYN_HEADER = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
_ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=False)
_THIN_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)

DYNAMIC_SHEET_NAME = "动态插单"

DYNAMIC_HEADERS = [
    "实例名",
    "事件ID",
    "订单",
    "生产工步",
    "PANEL数量",
    "加工效率（孔/min)",
    "最大叠片数",
    "单PNL总孔数",
    "使用的钻机等级/或制定机台",
    "叠数",
    "预计到达时刻",
    "实际到达时刻",
    "释放时刻",
    "优先级",
    "状态",
]


# ---------------------------------------------------------------------------
# 配置类定义
# ---------------------------------------------------------------------------

@dataclass
class ForecastErrorConfig:
    """预报误差生成配置"""
    unforecasted_prob: float = 0.20   # 突发未预报概率
    delay_prob: float = 0.30          # 晚到概率 (actual > expected)
    early_prob: float = 0.20          # 早到概率 (actual < expected)
    ontime_prob: float = 0.30         # 准时概率 (actual == expected)
    # 提前/晚到相对实际到达时刻的误差比例区间 (e.g. 0.05 ~ 0.25)
    error_ratio_range: Tuple[float, float] = (0.05, 0.25)
    # 最小绝对误差 (分钟)
    min_error_minutes: float = 5.0
    # 最大绝对误差 (分钟)
    max_error_minutes: float = 60.0

    def __post_init__(self):
        total = self.unforecasted_prob + self.delay_prob + self.early_prob + self.ontime_prob
        if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"预报概率之和必须为 1.0，当前为 {total}")
        if self.error_ratio_range[0] < 0 or self.error_ratio_range[1] < self.error_ratio_range[0]:
            raise ValueError(f"error_ratio_range 非法: {self.error_ratio_range}")


@dataclass
class ScenarioConfig:
    """单个场景的生成配置"""
    scenario_id: str = "scenario_default"
    seed: int = 42
    # 目标实例名列表（默认空，表示源工作簿中所有 Sheet 开头的实例）
    target_sheets: Optional[List[str]] = None
    # 每个实例生成的插单数量范围 (min_count, max_count)
    insertions_per_sheet: Tuple[int, int] = (3, 6)
    # 插单时刻模式: 'relative' (基于初始 Cmax 比例) 或 'absolute' (绝对分钟)
    arrival_mode: str = "relative"
    # relative 模式下的相对完工时间到达区间 [min_ratio, max_ratio]
    arrival_ratio_range: Tuple[float, float] = (0.08, 0.75)
    # absolute 模式下的到达区间 (分钟)
    arrival_time_range: Tuple[float, float] = (50.0, 1500.0)
    # 到达时间分布模式: 'uniform', 'poisson', 'burst'
    arrival_pattern: str = "uniform"
    # 优先级采样权重字典: 高 / 中 / 普通
    priority_weights: Dict[str, float] = field(
        default_factory=lambda: {
            PRIORITY_HIGH: 0.25,
            PRIORITY_MEDIUM: 0.35,
            PRIORITY_NORMAL: 0.40,
        }
    )
    # 预报误差配置
    forecast_config: ForecastErrorConfig = field(default_factory=ForecastErrorConfig)
    # 释放延时范围 (实际到达后到释放的准备延时，分钟)
    prep_delay_range: Tuple[float, float] = (0.0, 0.0)
    # 钻孔机台等级默认值
    default_drill_grade: str = "S/A/B/C"

    def __post_init__(self):
        if self.insertions_per_sheet[0] < 1 or self.insertions_per_sheet[1] < self.insertions_per_sheet[0]:
            raise ValueError(f"insertions_per_sheet 非法: {self.insertions_per_sheet}")
        for p in self.priority_weights:
            if p not in VALID_PRIORITIES:
                raise ValueError(f"未知优先级: {p}")


# ---------------------------------------------------------------------------
# 估算辅助函数
# ---------------------------------------------------------------------------

def estimate_sheet_makespan(ws: openpyxl.worksheet.worksheet.Worksheet) -> float:
    """
    快速、稳健地估算单个静态 Sheet 的初始 makespan 基准。
    基于总工时 / 资源池主轴数 + 基础换型开销，提供真实生产环境时间尺度。
    """
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 2:
        return 1000.0

    total_proc_work = 0.0
    spindle_count = 28  # 3*6 + 5*2 = 28 轴 (Pool D)

    for row in rows[1:]:
        if all(c is None or str(c).strip() == "" for c in row):
            continue
        try:
            q = float(row[2])
            eff = float(row[3])
            s = float(row[4])
            n = float(row[5])
            w = math.ceil(q / s)
            # 单批加工工时约等于 N / eff (分)
            t_work = (n / eff) * w
            total_proc_work += t_work
        except Exception:
            continue

    # 平均负载完工时间 + 预留 20% 换型/辅助开销
    est_cmax = (total_proc_work / spindle_count) * 1.2 + 60.0
    return max(300.0, est_cmax)


# ---------------------------------------------------------------------------
# 核心生成器类
# ---------------------------------------------------------------------------

class DynamicScenarioGenerator:
    """动态场景生成引擎"""

    def __init__(self, src_excel: str):
        self.src_excel = os.path.abspath(src_excel)
        if not os.path.exists(self.src_excel):
            raise FileNotFoundError(f"源 Excel 文件不存在: '{self.src_excel}'")

    def _generate_order_process_data(
        self,
        rng: random.Random,
        process_type: str,
        empirical_pool: List[Dict[str, Any]],
        default_drill_grade: str,
    ) -> Dict[str, Any]:
        """
        生成一条符合物理规范的订单工艺数据。
        如果存在经验池，优先从经验池分布中抽样并施加微扰；否则按物理规范生成。
        严格确保 W = ceil(Q / S)。
        """
        if empirical_pool:
            base = rng.choice(empirical_pool)
            # 允许在经验值附近微扰
            q = max(5, int(base["Q"] + rng.choice([-3, -2, -1, 0, 1, 2, 3])))
            s = int(base["S"])
            n = max(1000, int(base["N"] + rng.randint(-1500, 1500)))
            eff = float(base["efficiency"])
            grade = base["raw_machine_grade"] or default_drill_grade
        else:
            if process_type == PROCESS_DRILL:
                q = rng.randint(8, 30)
                s = rng.choice([1, 2, 3, 5])
                n = rng.randint(3000, 50000)
                eff = 220.0
                grade = default_drill_grade
            else:
                q = rng.randint(6, 20)
                s = rng.choice([1, 2, 3])
                n = rng.randint(1500, 20000)
                eff = 97.0
                grade = ""

        w = math.ceil(q / s)
        return {
            "process_type": process_type,
            "Q": q,
            "efficiency": eff,
            "S": s,
            "N": n,
            "raw_machine_grade": grade,
            "W_excel": w,
        }

    def _sample_arrival_times(
        self,
        rng: random.Random,
        count: int,
        cmax_est: float,
        config: ScenarioConfig,
    ) -> List[float]:
        """按配置生成到达时刻列表"""
        if config.arrival_mode == "relative":
            min_t = cmax_est * config.arrival_ratio_range[0]
            max_t = cmax_est * config.arrival_ratio_range[1]
        else:
            min_t = config.arrival_time_range[0]
            max_t = config.arrival_time_range[1]

        min_t = max(10.0, min_t)
        max_t = max(min_t + 10.0, max_t)

        pattern = config.arrival_pattern.lower()
        if pattern == "uniform":
            times = [rng.uniform(min_t, max_t) for _ in range(count)]
        elif pattern == "poisson":
            # 间歇到达：指数间隔
            span = max_t - min_t
            mean_gap = span / (count + 1)
            cur = min_t
            times = []
            for _ in range(count):
                cur += rng.expovariate(1.0 / mean_gap)
                cur = min(cur, max_t)
                times.append(cur)
        elif pattern == "burst":
            # 脉冲突发：聚集成 1~2 个突发时间团
            clusters = [rng.uniform(min_t, max_t) for _ in range(max(1, count // 3))]
            times = []
            for _ in range(count):
                center = rng.choice(clusters)
                t = max(0.0, center + rng.uniform(-15.0, 15.0))
                times.append(t)
        else:
            times = [rng.uniform(min_t, max_t) for _ in range(count)]

        # 保留 1 位小数并排序
        times = [round(t, 1) for t in times]
        times.sort()
        return times

    def _sample_forecast(
        self,
        rng: random.Random,
        actual_time: float,
        cfg: ForecastErrorConfig,
    ) -> Tuple[Optional[float], bool, str]:
        """
        生成单条事件的预报到达时间与预报类型。
        返回: (expected_arrival_time, is_forecasted, forecast_type)
        """
        r = rng.random()
        p_unfc = cfg.unforecasted_prob
        p_ontime = p_unfc + cfg.ontime_prob
        p_early = p_ontime + cfg.early_prob

        if r < p_unfc:
            # 突发未预报
            return None, False, "unforecasted"
        elif r < p_ontime:
            # 准时
            return actual_time, True, "on_time"
        elif r < p_early:
            # 早到: actual < expected, 即 expected = actual + delta
            ratio = rng.uniform(cfg.error_ratio_range[0], cfg.error_ratio_range[1])
            delta = max(cfg.min_error_minutes, min(cfg.max_error_minutes, actual_time * ratio))
            expected = round(actual_time + delta, 1)
            return expected, True, "early"
        else:
            # 晚到: actual > expected, 即 expected = actual - delta
            ratio = rng.uniform(cfg.error_ratio_range[0], cfg.error_ratio_range[1])
            delta = max(cfg.min_error_minutes, min(cfg.max_error_minutes, actual_time * ratio))
            expected = max(0.0, round(actual_time - delta, 1))
            return expected, True, "delayed"

    def _sample_priority(self, rng: random.Random, weights: Dict[str, float]) -> str:
        """按权重随机抽取优先级"""
        items = list(weights.items())
        labels = [item[0] for item in items]
        w = [item[1] for item in items]
        return rng.choices(labels, weights=w, k=1)[0]

    def generate_scenario(
        self,
        dst_excel: str,
        config: Optional[ScenarioConfig] = None,
    ) -> Dict[str, Any]:
        """
        基于配置生成符合校验规范的动态调度 Excel 文件。
        不修改或覆盖源 Excel 文件。
        """
        if config is None:
            config = ScenarioConfig()

        dst_abs = os.path.abspath(dst_excel)
        if dst_abs == self.src_excel:
            raise ValueError(f"目标文件路径禁止与源文件相同，防止覆盖源文件: '{dst_abs}'")

        rng = random.Random(config.seed)

        # 读取源工作簿（只读方式）
        src_wb = openpyxl.load_workbook(self.src_excel, data_only=True)
        dst_wb = openpyxl.Workbook()
        dst_wb.remove(dst_wb.active)  # 移除默认创建的空 Sheet

        all_sheets = src_wb.sheetnames
        order_sheets = [s for s in all_sheets if s.lower().startswith("sheet")]

        if config.target_sheets:
            active_target_sheets = [s for s in order_sheets if s in config.target_sheets]
            if not active_target_sheets:
                raise ValueError(
                    f"配置的 target_sheets {config.target_sheets} 中没有匹配的订单 Sheet。"
                )
        else:
            active_target_sheets = order_sheets

        initial_orders_by_sheet: Dict[str, List[int]] = {}
        empirical_pool_by_sheet: Dict[str, List[Dict[str, Any]]] = {}
        sheet_pool_type: Dict[str, str] = {}
        sheet_cmax_est: Dict[str, float] = {}

        # 1. 复制原 Sheet 并增加“预计到达时刻”与“优先级”列
        for sheet_name in all_sheets:
            src_ws = src_wb[sheet_name]
            dst_ws = dst_wb.create_sheet(title=sheet_name)
            is_order_sheet = sheet_name.lower().startswith("sheet")

            existing_order_ids: List[int] = []
            empirical_pool: List[Dict[str, Any]] = []
            proc_types_seen: Set[str] = set()

            for r_idx, row in enumerate(src_ws.iter_rows(values_only=True), start=1):
                for c_idx, val in enumerate(row, start=1):
                    cell = dst_ws.cell(row=r_idx, column=c_idx, value=val)
                    cell.font = _FONT_DATA
                    cell.alignment = _ALIGN_CENTER
                    cell.border = _THIN_BORDER
                    if r_idx == 1:
                        cell.font = _FONT_HEADER

                if is_order_sheet:
                    if r_idx == 1:
                        c9 = dst_ws.cell(row=1, column=9, value="预计到达时刻")
                        c10 = dst_ws.cell(row=1, column=10, value="优先级")
                        for c in (c9, c10):
                            c.font = _FONT_HEADER
                            c.fill = _FILL_INITIAL_NEW
                            c.alignment = _ALIGN_CENTER
                            c.border = _THIN_BORDER
                    else:
                        order_val = row[0] if len(row) > 0 else None
                        if order_val is not None and str(order_val).strip() != "":
                            try:
                                oid_num = int(float(order_val))
                                existing_order_ids.append(oid_num)
                            except Exception:
                                pass

                            # 记录经验参数供抽样参考
                            try:
                                pt = str(row[1]).strip()
                                proc_types_seen.add(pt)
                                empirical_pool.append({
                                    "process_type": pt,
                                    "Q": int(float(row[2])),
                                    "efficiency": float(row[3]),
                                    "S": int(float(row[4])),
                                    "N": int(float(row[5])),
                                    "raw_machine_grade": str(row[6]).strip() if len(row) > 6 and row[6] else "",
                                })
                            except Exception:
                                pass

                            c9 = dst_ws.cell(row=r_idx, column=9, value=0)
                            c10 = dst_ws.cell(row=r_idx, column=10, value="普通")
                            for c in (c9, c10):
                                c.font = _FONT_DATA
                                c.alignment = _ALIGN_CENTER
                                c.border = _THIN_BORDER

            # 调整列宽
            for col in dst_ws.columns:
                col_letter = get_column_letter(col[0].column)
                max_len = max((len(str(c.value or "")) for c in col), default=0)
                dst_ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

            if is_order_sheet:
                initial_orders_by_sheet[sheet_name] = existing_order_ids
                empirical_pool_by_sheet[sheet_name] = empirical_pool
                # 判定该 sheet 的主要工步
                is_backdrill = any(p in ("背钻", "backdrill") for p in proc_types_seen)
                sheet_pool_type[sheet_name] = PROCESS_BACKDRILL if is_backdrill else PROCESS_DRILL
                sheet_cmax_est[sheet_name] = estimate_sheet_makespan(src_ws)

        # 2. 生成“动态插单”Sheet
        dyn_ws = dst_wb.create_sheet(title=DYNAMIC_SHEET_NAME)
        for c_idx, h_name in enumerate(DYNAMIC_HEADERS, start=1):
            cell = dyn_ws.cell(row=1, column=c_idx, value=h_name)
            cell.font = _FONT_HEADER
            cell.fill = _FILL_DYN_HEADER
            cell.alignment = _ALIGN_CENTER
            cell.border = _THIN_BORDER

        all_dynamic_rows: List[List[Any]] = []
        event_counter = 1
        scenario_events_summary: List[Dict[str, Any]] = []

        for sname in active_target_sheets:
            existing_ids = initial_orders_by_sheet.get(sname, [])
            start_order_id = max(existing_ids, default=0) + 101

            count = rng.randint(config.insertions_per_sheet[0], config.insertions_per_sheet[1])
            cmax = sheet_cmax_est.get(sname, 1500.0)
            arrival_times = self._sample_arrival_times(rng, count, cmax, config)
            ptype = sheet_pool_type.get(sname, PROCESS_DRILL)
            pool = empirical_pool_by_sheet.get(sname, [])

            for i, act_time in enumerate(arrival_times):
                ev_id = f"INS-{event_counter:03d}"
                order_id = start_order_id + i
                proc_data = self._generate_order_process_data(
                    rng, ptype, pool, config.default_drill_grade
                )
                exp_time, is_forecasted, f_type = self._sample_forecast(
                    rng, act_time, config.forecast_config
                )
                prep_delay = rng.uniform(config.prep_delay_range[0], config.prep_delay_range[1])
                rel_time = round(act_time + prep_delay, 1)
                priority = self._sample_priority(rng, config.priority_weights)
                status = STATUS_PENDING

                row = [
                    sname,
                    ev_id,
                    order_id,
                    proc_data["process_type"],
                    proc_data["Q"],
                    proc_data["efficiency"],
                    proc_data["S"],
                    proc_data["N"],
                    proc_data["raw_machine_grade"],
                    proc_data["W_excel"],
                    exp_time if is_forecasted else UNFORECASTED_TAG,
                    act_time,
                    rel_time,
                    priority,
                    status,
                ]
                all_dynamic_rows.append(row)
                event_counter += 1

                scenario_events_summary.append({
                    "event_id": ev_id,
                    "instance": sname,
                    "order_id": order_id,
                    "process_type": proc_data["process_type"],
                    "Q": proc_data["Q"],
                    "S": proc_data["S"],
                    "W": proc_data["W_excel"],
                    "N": proc_data["N"],
                    "priority": priority,
                    "is_forecasted": is_forecasted,
                    "forecast_type": f_type,
                    "expected_arrival_time": exp_time,
                    "actual_arrival_time": act_time,
                    "release_time": rel_time,
                })

        # 按 (actual_arrival_time, event_id) 稳定排序写入
        all_dynamic_rows.sort(key=lambda r: (r[11], r[1]))

        for r_idx, r_data in enumerate(all_dynamic_rows, start=2):
            for c_idx, val in enumerate(r_data, start=1):
                cell = dyn_ws.cell(row=r_idx, column=c_idx, value=val)
                cell.font = _FONT_DATA
                cell.alignment = _ALIGN_CENTER
                cell.border = _THIN_BORDER

        # 调整动态插单列宽
        for col in dyn_ws.columns:
            col_letter = get_column_letter(col[0].column)
            max_len = max((len(str(c.value or "")) for c in col), default=0)
            dyn_ws.column_dimensions[col_letter].width = max(max_len + 3, 14)

        # 确保输出目录存在
        parent_dir = os.path.dirname(dst_abs)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        dst_wb.save(dst_abs)

        # 3. 严格校验自检（不通过立即抛异常）
        validation = validate_dynamic_excel(dst_abs)

        return {
            "scenario_id": config.scenario_id,
            "filepath": dst_abs,
            "seed": config.seed,
            "target_sheets": active_target_sheets,
            "total_initial_orders": validation["total_initial_orders"],
            "total_dynamic_events": len(all_dynamic_rows),
            "validation_status": validation["status"],
            "events": scenario_events_summary,
        }


# ---------------------------------------------------------------------------
# 数据集批量生成接口
# ---------------------------------------------------------------------------

# The formal dynamic experiment is fixed to V20261027.  These values prevent
# accidental recreation of the retired 10/3/5 split when this generic helper is
# used to make a candidate dataset.
V20261027_DEFAULT_SPLIT_COUNTS = (18, 6, 6)
_ACTIVE_FORMAL_DATASET_DIR = os.path.normcase(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dynamic_scenarios"))
)


def _reject_active_formal_output_dir(output_dir: str) -> None:
    """Keep the hash-bound V20261027 dataset immutable to generic generation."""
    resolved = os.path.normcase(os.path.normpath(os.path.abspath(output_dir)))
    if resolved == _ACTIVE_FORMAL_DATASET_DIR:
        raise ValueError(
            "data/dynamic_scenarios is the hash-bound active V20261027 formal dataset. "
            "The generic generator may only write an explicitly named candidate directory; "
            "it must not replace the active formal dataset."
        )


def generate_dynamic_dataset(
    src_excel: str,
    output_dir: str,
    train_count: int = V20261027_DEFAULT_SPLIT_COUNTS[0],
    val_count: int = V20261027_DEFAULT_SPLIT_COUNTS[1],
    test_count: int = V20261027_DEFAULT_SPLIT_COUNTS[2],
    base_seed: int = 1000,
    target_sheets: Optional[List[str]] = None,
    default_drill_grade: str = "S/A/B/C",
) -> Dict[str, Any]:
    """
    批量生成候选动态调度数据集，并在根目录输出 dataset_manifest.json。

    默认数量沿用当前 V20261027 的 18/6/6 规模，但本函数的通用
    随机场景逻辑不是 V20261027 的可替代重建入口。它拒绝写入活动
    ``data/dynamic_scenarios``；正式 V20261027 三组数据只能由
    ``build_three_group_dynamic_dataset.py`` 重建，并由哈希绑定清单和
    工作簿读取、验证与训练。
    """
    _reject_active_formal_output_dir(output_dir)
    generator = DynamicScenarioGenerator(src_excel)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    splits_spec = [
        ("train", train_count),
        ("val", val_count),
        ("test", test_count),
    ]

    manifest: Dict[str, Any] = {
        "dataset_name": "PCB_Dynamic_Scheduling_Benchmark",
        "created_at": datetime.datetime.now().isoformat(),
        "src_excel": os.path.abspath(src_excel),
        "base_seed": base_seed,
        "splits": {},
        "total_scenarios": 0,
    }

    current_seed = base_seed
    total_generated = 0

    # 预设不同规模档位以增加多样性
    scale_presets = [
        (1, 3),   # 轻量
        (3, 6),   # 中量
        (6, 10),  # 重量
    ]

    # 预设不同到达模式
    pattern_presets = ["uniform", "poisson", "burst"]

    # 预设不同优先级侧重点
    priority_presets = [
        # 均衡
        {PRIORITY_HIGH: 0.25, PRIORITY_MEDIUM: 0.35, PRIORITY_NORMAL: 0.40},
        # 紧迫（高优先级多）
        {PRIORITY_HIGH: 0.50, PRIORITY_MEDIUM: 0.30, PRIORITY_NORMAL: 0.20},
        # 常态（普通多）
        {PRIORITY_HIGH: 0.10, PRIORITY_MEDIUM: 0.20, PRIORITY_NORMAL: 0.70},
    ]

    for split_name, count in splits_spec:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        split_records: List[Dict[str, Any]] = []

        for idx in range(count):
            scenario_id = f"{split_name}_{idx:03d}"
            dst_file = os.path.join(split_dir, f"scenario_{scenario_id}.xlsx")

            # 轮换组合不同参数维度
            scale_range = scale_presets[idx % len(scale_presets)]
            pattern = pattern_presets[idx % len(pattern_presets)]
            pri_weights = priority_presets[idx % len(priority_presets)]

            # 训练集采用更宽的预测误差探索；测试集涵盖所有四种边界
            cfg = ScenarioConfig(
                scenario_id=scenario_id,
                seed=current_seed,
                target_sheets=target_sheets,
                insertions_per_sheet=scale_range,
                arrival_mode="relative",
                arrival_ratio_range=(0.08, 0.75),
                arrival_pattern=pattern,
                priority_weights=pri_weights,
                forecast_config=ForecastErrorConfig(
                    unforecasted_prob=0.25,
                    delay_prob=0.30,
                    early_prob=0.20,
                    ontime_prob=0.25,
                    error_ratio_range=(0.05, 0.30),
                ),
                prep_delay_range=(0.0, 3.0),
                default_drill_grade=default_drill_grade,
            )

            record = generator.generate_scenario(dst_file, cfg)
            # 记录相对路径以保证跨机器可移植性
            record_copy = dict(record)
            record_copy["filepath"] = os.path.relpath(record["filepath"], output_dir)
            split_records.append(record_copy)

            current_seed += 1
            total_generated += 1

        manifest["splits"][split_name] = split_records

    manifest["total_scenarios"] = total_generated
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"成功批量生成数据集，共 {total_generated} 个场景文件，元数据已写入: {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the candidate-only CLI and expose its defaults to regression tests."""
    parser = argparse.ArgumentParser(description="PCB 批量动态场景生成器")
    parser.add_argument("--src", default="test_1.xlsx", help="源 Excel 文件路径，默认 test_1.xlsx")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="候选数据集输出目录；禁止指定活动 V20261027 目录 data/dynamic_scenarios",
    )
    parser.add_argument("--train-count", type=int, default=V20261027_DEFAULT_SPLIT_COUNTS[0], help="训练集场景数量，默认 V20261027 的 18")
    parser.add_argument("--val-count", type=int, default=V20261027_DEFAULT_SPLIT_COUNTS[1], help="验证集场景数量，默认 V20261027 的 6")
    parser.add_argument("--test-count", type=int, default=V20261027_DEFAULT_SPLIT_COUNTS[2], help="测试集场景数量，默认 V20261027 的 6")
    parser.add_argument("--seed", type=int, default=1000, help="起始随机种子")
    parser.add_argument("--target-sheet", default=None, help="可选：限定单实例名，如 Sheet1")
    return parser


def main():
    parser = build_argument_parser()

    args = parser.parse_args()

    targets = [args.target_sheet] if args.target_sheet else None
    generate_dynamic_dataset(
        src_excel=args.src,
        output_dir=args.out_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        base_seed=args.seed,
        target_sheets=targets,
    )


if __name__ == "__main__":
    main()
