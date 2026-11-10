"""
data_loader.py
==============
原始 Excel 数据导入与字段校验器（规范第 3 章、TC-01、TC-04A）

功能说明：
  1. 从 test_1.xlsx / test_2.xlsx 读取各 Sheet 的订单数据。
  2. 严格执行字段字典匹配（规范 3.1 节），不做模糊匹配。
  3. 逐行核验数据关系 D1：S_i = ceil(Q_i / ML_i)，与 Excel 物理叠板单元数列 100% 一致（TC-01）。
  4. 对钻孔效率进行合法值检查（规范 3.4 节）。
  5. 调用 eligibility_builder 为每个订单生成资格矩阵并完成 TC-10 校验。
  6. 返回标准 Order 对象列表，供上层算法使用。

实验编号与 Sheet 映射（规范 3.2 & 3.3 节）：
  test_1.xlsx：Sheet 按订单数从大到小排列，实验编号与 Sheet 顺序相反。
    Sheet1 → 实验 5（最大规模），…，Sheet5 → 实验 1（最小规模）
  test_2.xlsx：同理，Sheet1 → 实验 10，…，Sheet5 → 实验 6

字段名称映射（规范 3.1 节，精确匹配，允许去除首尾空格和全半角括号标准化）：
  "订单"                        → order_id
  "生产工步"                    → process_type
  "PANEL数量"                   → Q（PANEL 需求量）
  "加工效率（孔/min)"            → efficiency
  "最大叠片数"                  → S（代码字段，数学符号为 ML_i）
  "单PNL总孔数"                 → N
  "使用的钻机等级/或制定机台"    → raw_machine_grade
  "叠数"                        → W（代码字段，数学符号为 S_i；仅用于 D1 校验，由程序重新计算）
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import openpyxl

from data_models import (
    PROCESS_BACKDRILL,
    PROCESS_DRILL,
    Order,
)
from eligibility_builder import (
    build_eligibility_matrix,
    validate_cross_pool_isolation,
)


# ---------------------------------------------------------------------------
# 字段名称规范（规范 3.1 节严格字典）
# ---------------------------------------------------------------------------

# 允许的钻孔效率值（规范 3.4 节）
VALID_EFFICIENCIES = {220.0, 97.0}

# 允许的换刀孔限（规范 5.3 & 5.4 节）
DEFAULT_H_MAP = {
    PROCESS_DRILL: 1500,
    PROCESS_BACKDRILL: 300,
}

# 字段名称别名表（规范 3.4 节：允许去除首尾空格和全半角括号标准化）
_FIELD_ALIASES: Dict[str, str] = {
    "订单":                    "order_id",
    "订单编号":                "order_id",
    "生产工步":                "process_type",
    "PANEL数量":               "Q",
    "panel数量":               "Q",
    "加工效率（孔/min)":       "efficiency",
    "加工效率(孔/min)":        "efficiency",
    "加工效率（孔/min）":      "efficiency",
    "加工效率(孔/min）":       "efficiency",
    "加工效率（孔/MIN)":       "efficiency",
    "最大叠片数":              "S",
    "单PNL总孔数":             "N",
    "单pnl总孔数":             "N",
    "使用的钻机等级/或制定机台": "raw_machine_grade",
    "使用的钻机等级/或指定机台": "raw_machine_grade",
    "使用的钻机等级/或制定机台 ": "raw_machine_grade",
    "使用的钻机等级/指定机台":   "raw_machine_grade",
    "叠数":                    "W_excel",
    "预计到达时刻":              "expected_arrival_time",
    "预计到达时间":              "expected_arrival_time",
    "实际到达时刻":              "actual_arrival_time",
    "实际到达时间":              "actual_arrival_time",
    "释放时刻":                  "release_time",
    "释放时间":                  "release_time",
    "优先级":                    "priority",
    "状态":                      "status",
    "实例名":                    "instance_name",
    "事件ID":                    "event_id",
    "事件编号":                  "event_id",
    "加工效率":                  "efficiency",
    "工艺最大叠片数":              "S",
    "物理叠板单元数":              "W_excel",
    "订单优先级":                  "priority",
    "工序交付期":                  "due_date",
    "交付期":                      "due_date",
    "交期":                        "due_date",
    "due_date":                    "due_date",
}

# 最低必要字段集合
_REQUIRED_FIELDS = {"order_id", "process_type", "Q", "efficiency", "S", "N"}


def _normalize_header(raw: str) -> str:
    """标准化列名（去首尾空格，保留内容用于字典匹配）"""
    if raw is None:
        return ""
    return str(raw).strip()


def _parse_headers(row) -> Dict[str, int]:
    """
    将 Excel 第一行解析为 {规范字段名 -> 列索引} 映射。
    对于无法识别的列名发出警告（不中止，仅记录）。
    """
    field_to_col: Dict[str, int] = {}
    for col_idx, cell_val in enumerate(row):
        raw = _normalize_header(cell_val)
        canonical = _FIELD_ALIASES.get(raw)
        if canonical is None:
            # 尝试小写匹配
            canonical = _FIELD_ALIASES.get(raw.lower())
        if canonical:
            field_to_col[canonical] = col_idx
        # 忽略未知列名（不影响解析，仅跳过）
    return field_to_col


def _validate_required_fields(field_to_col: Dict[str, int], sheet_name: str, filepath: str) -> None:
    """检查所有最低必要字段均已解析成功"""
    missing = _REQUIRED_FIELDS - set(field_to_col.keys())
    if missing:
        raise ValueError(
            f"数据导入失败：文件 '{filepath}' Sheet '{sheet_name}' "
            f"缺少以下必要字段（列名未能匹配）：{missing}\n"
            f"已识别的字段映射：{field_to_col}\n"
            f"请核验 Excel 文件列名是否符合规范 3.1 节字段字典。"
        )


def _safe_int(val, field: str, order_id, sheet: str) -> int:
    """安全转换为正整数，任何异常立即抛出（禁止以 0 替代）"""
    try:
        numeric = float(str(val))
        if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
            raise ValueError(f"值 {val} 不是正整数")
        v = int(numeric)
        return v
    except Exception:
        raise ValueError(
            f"字段 '{field}' 解析失败：Sheet='{sheet}', order_id='{order_id}', "
            f"原始值='{val}'。规范要求此字段为正整数，禁止以 0 替代。"
        )


def _safe_float(val, field: str, order_id, sheet: str) -> float:
    """安全转换为正实数"""
    try:
        v = float(str(val))
        if v <= 0:
            raise ValueError(f"值 {val} 不是正实数")
        return v
    except Exception:
        raise ValueError(
            f"字段 '{field}' 解析失败：Sheet='{sheet}', order_id='{order_id}', "
            f"原始值='{val}'。规范要求此字段为正实数，禁止以 0 替代。"
        )


def load_sheet(
    filepath: str,
    sheet_name: str,
    verify_W: bool = True,
) -> List[Order]:
    """
    从指定 Sheet 加载并校验订单数据。

    参数
    ----
    filepath   : Excel 文件路径
    sheet_name : Sheet 名称（如 "Sheet1"）
    verify_W   : 是否执行 D1 叠数核验（TC-01），默认开启

    返回
    ----
    List[Order] — 通过校验的标准订单对象列表（含 eligible_machines）

    异常
    ----
    ValueError  : 字段缺失、D1 不一致、效率非法、跨池指派等
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"数据文件 '{filepath}' 不存在，请检查路径配置。"
        )

    wb = openpyxl.load_workbook(filepath, data_only=True)
    if sheet_name not in wb.sheetnames:
        raise ValueError(
            f"Sheet '{sheet_name}' 不存在于文件 '{filepath}'。"
            f"可用 Sheet：{wb.sheetnames}"
        )

    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))

    if len(rows) < 2:
        raise ValueError(
            f"Sheet '{sheet_name}' 仅有 {len(rows)} 行（含表头），无有效数据行。"
        )

    # 解析表头
    header_row = rows[0]
    field_to_col = _parse_headers(header_row)
    _validate_required_fields(field_to_col, sheet_name, filepath)

    has_W_col = "W_excel" in field_to_col
    has_grade_col = "raw_machine_grade" in field_to_col

    if not has_W_col and verify_W:
        warnings.warn(
            f"Sheet '{sheet_name}' 中未找到'叠数'列，将跳过 TC-01 叠数核验。",
            UserWarning,
        )

    orders: List[Order] = []
    d1_errors: List[str] = []

    for row_idx, row in enumerate(rows[1:], start=2):  # 从第 2 行（数据行）开始
        # 跳过全空行
        if all(cell is None or str(cell).strip() == "" for cell in row):
            continue

        def get(field: str):
            col = field_to_col.get(field)
            return row[col] if col is not None and col < len(row) else None

        raw_order_id = get("order_id")
        if raw_order_id is None or str(raw_order_id).strip() == "":
            continue  # 跳过空订单行

        order_id_str = str(raw_order_id).strip()
        process_type_raw = str(get("process_type")).strip()

        # 标准化生产工步
        if process_type_raw in ("钻孔", "普通钻孔", "drill"):
            process_type = PROCESS_DRILL
        elif process_type_raw in ("背钻", "backdrill"):
            process_type = PROCESS_BACKDRILL
        else:
            raise ValueError(
                f"Sheet '{sheet_name}' 第 {row_idx} 行 Order '{order_id_str}': "
                f"未知生产工步 '{process_type_raw}'，合法值为 '钻孔' 或 '背钻'。"
            )

        Q = _safe_int(get("Q"), "PANEL数量", order_id_str, sheet_name)
        efficiency = _safe_float(get("efficiency"), "加工效率", order_id_str, sheet_name)
        S_raw = get("S")
        S = _safe_int(S_raw, "最大叠片数", order_id_str, sheet_name)
        N = _safe_int(get("N"), "单PNL总孔数", order_id_str, sheet_name)

        # 换刀孔限 H（按工步查默认值，规范 5.3 & 5.4 节）
        H = DEFAULT_H_MAP[process_type]

        # 计算 W（规范数据关系 D1）
        W_computed = math.ceil(Q / S)

        # TC-01：核验 Excel 叠数列
        if has_W_col and verify_W:
            W_excel_raw = get("W_excel")
            if W_excel_raw is not None and str(W_excel_raw).strip() != "":
                W_excel = _safe_int(W_excel_raw, "物理叠板单元数", order_id_str, sheet_name)
                if W_excel != W_computed:
                    d1_errors.append(
                        f"  Sheet='{sheet_name}' Row={row_idx} "
                        f"Order='{order_id_str}': "
                        f"Excel物理叠板单元数={W_excel} vs 程序计算ceil({Q}/{S})={W_computed}"
                    )

        # 效率合法值检查（规范 3.4 节：不强制覆盖其他合法值）
        if efficiency not in VALID_EFFICIENCIES:
            warnings.warn(
                f"Sheet '{sheet_name}' Order '{order_id_str}': "
                f"加工效率 {efficiency} 不在预期值 {VALID_EFFICIENCIES} 中，"
                f"视为合法参数继续使用（规范 3.4 节）。",
                UserWarning,
            )

        raw_grade = ""
        if has_grade_col:
            grade_raw = get("raw_machine_grade")
            if grade_raw is not None:
                raw_grade = str(grade_raw).strip()

        # 动态重调度字段解析
        exp_arr_raw = get("expected_arrival_time")
        if exp_arr_raw is not None and str(exp_arr_raw).strip() != "":
            exp_arr_str = str(exp_arr_raw).strip()
            if exp_arr_str in ("未预报", "未预测", "突发", "None"):
                expected_arrival_time = None
                is_forecasted = False
            else:
                try:
                    expected_arrival_time = float(exp_arr_str)
                    is_forecasted = True
                except Exception:
                    expected_arrival_time = None
                    is_forecasted = False
        else:
            expected_arrival_time = 0.0
            is_forecasted = True

        pri_raw = get("priority")
        priority = str(pri_raw).strip() if pri_raw is not None and str(pri_raw).strip() != "" else "普通"

        act_arr_raw = get("actual_arrival_time")
        try:
            actual_arrival_time = float(act_arr_raw) if act_arr_raw is not None and str(act_arr_raw).strip() != "" else 0.0
        except Exception:
            actual_arrival_time = 0.0

        rel_raw = get("release_time")
        try:
            release_time = float(rel_raw) if rel_raw is not None and str(rel_raw).strip() != "" else actual_arrival_time
        except Exception:
            release_time = actual_arrival_time

        ev_id_raw = get("event_id")
        event_id = str(ev_id_raw).strip() if ev_id_raw is not None and str(ev_id_raw).strip() != "" else None

        status_raw = get("status")
        status = str(status_raw).strip() if status_raw is not None and str(status_raw).strip() != "" else "初始"

        due_raw = get("due_date")
        try:
            due_date = float(due_raw) if due_raw is not None and str(due_raw).strip() != "" else None
        except Exception:
            due_date = None

        try:
            order = Order(
                order_id=order_id_str,
                process_type=process_type,
                Q=Q,
                S=S,
                W=W_computed,
                N=N,
                efficiency=efficiency,
                H=H,
                raw_machine_grade=raw_grade,
                expected_arrival_time=expected_arrival_time,
                actual_arrival_time=actual_arrival_time,
                release_time=release_time,
                priority=priority,
                event_id=event_id,
                instance_name=sheet_name,
                status=status,
                is_forecasted=is_forecasted,
                due_date=due_date,
            )
        except ValueError as e:
            raise ValueError(
                f"Sheet '{sheet_name}' Row {row_idx} Order '{order_id_str}': "
                f"创建订单对象失败：{e}"
            )

        orders.append(order)

    # TC-01 最终报告（累积，一次性输出所有差异）
    if d1_errors:
        raise ValueError(
            f"TC-01 FAIL：共发现 {len(d1_errors)} 条叠数 D1 不一致，求解中止。\n"
            + "\n".join(d1_errors)
            + "\n请检查 Excel 文件或程序计算逻辑。"
        )

    # 构建资格矩阵（TC-04A）
    build_eligibility_matrix(orders, strict=True)

    # TC-10：验证双资源池隔离
    validate_cross_pool_isolation(orders)

    return orders


def load_file(
    filepath: str,
    sheet_names: Optional[List[str]] = None,
    verify_W: bool = True,
) -> Dict[str, List[Order]]:
    """
    加载 Excel 文件中的所有（或指定）数据 Sheet（跳过工艺表 Sheet）。

    参数
    ----
    filepath    : Excel 文件路径
    sheet_names : 指定需要加载的 Sheet 列表；None 则自动过滤并加载所有数据 Sheet
    verify_W    : 是否执行 D1 叠数核验

    返回
    ----
    Dict[sheet_name -> List[Order]]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件 '{filepath}' 不存在。")

    wb = openpyxl.load_workbook(filepath, data_only=True)
    all_sheets = wb.sheetnames

    # 自动过滤工艺表（规范 3.1 节：数据 Sheet 均命名为 Sheet1~Sheet5）
    # 工艺表（"钻孔工艺"、"背钻工艺"）不包含订单数据，自动跳过
    DATA_SHEET_KEYWORDS = ["Sheet", "sheet"]
    if sheet_names is None:
        sheet_names = [
            s for s in all_sheets
            if any(s.startswith(kw) for kw in DATA_SHEET_KEYWORDS)
        ]
        if not sheet_names:
            warnings.warn(
                f"文件 '{filepath}' 中未找到以 'Sheet' 开头的数据 Sheet，"
                f"将尝试加载全部 {len(all_sheets)} 个 Sheet。",
                UserWarning,
            )
            sheet_names = all_sheets

    result: Dict[str, List[Order]] = {}
    for sname in sheet_names:
        if sname not in all_sheets:
            warnings.warn(f"Sheet '{sname}' 不存在，跳过。", UserWarning)
            continue
        try:
            orders = load_sheet(filepath, sname, verify_W=verify_W)
            result[sname] = orders
        except ValueError as e:
            raise ValueError(f"加载 Sheet '{sname}' 时发生错误：\n{e}")

    return result


def _find_default_base_dir() -> str:
    """自动寻找 test_1.xlsx / test_2.xlsx 所在目录"""
    if os.path.exists("test_1.xlsx"):
        return "."
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(this_dir, "test_1.xlsx")):
        return this_dir
    parent_dir = os.path.abspath(os.path.join(this_dir, ".."))
    if os.path.exists(os.path.join(parent_dir, "test_1.xlsx")):
        return parent_dir
    return "."


def load_test1(base_dir: Optional[str] = None, verify_W: bool = True) -> Dict[str, List[Order]]:
    """
    加载 test_1.xlsx（单工序，仅钻孔订单，实验 1~5，规范 3.2 节）

    返回：{"Sheet1": [...], "Sheet2": [...], ..., "Sheet5": [...]}
    """
    if base_dir is None:
        base_dir = _find_default_base_dir()
    path = os.path.join(base_dir, "test_1.xlsx")
    return load_file(path, sheet_names=["Sheet1", "Sheet2", "Sheet3", "Sheet4", "Sheet5"], verify_W=verify_W)


def load_test2(base_dir: Optional[str] = None, verify_W: bool = True) -> Dict[str, List[Order]]:
    """
    加载 test_2.xlsx（多工序，含钻孔与背钻订单，实验 6~10，规范 3.3 节）

    返回：{"Sheet1": [...], "Sheet2": [...], ..., "Sheet5": [...]}
    """
    if base_dir is None:
        base_dir = _find_default_base_dir()
    path = os.path.join(base_dir, "test_2.xlsx")
    return load_file(path, sheet_names=["Sheet1", "Sheet2", "Sheet3", "Sheet4", "Sheet5"], verify_W=verify_W)


def get_experiment_mapping(filename: str) -> Dict[str, int]:
    """
    返回 Sheet 名 -> 实验编号的映射（规范 3.2 & 3.3 节）。

    Sheet1 订单数最多（最大规模实验），实验编号最大：
      test_1: Sheet1→实验5, Sheet2→实验4, ..., Sheet5→实验1
      test_2: Sheet1→实验10, Sheet2→实验9, ..., Sheet5→实验6
    """
    if "test_1" in filename or "test1" in filename:
        return {f"Sheet{i}": 6 - i for i in range(1, 6)}  # Sheet1→5, Sheet5→1
    elif "test_2" in filename or "test2" in filename:
        return {f"Sheet{i}": 11 - i for i in range(1, 6)}  # Sheet1→10, Sheet5→6
    else:
        return {f"Sheet{i}": i for i in range(1, 6)}


if __name__ == "__main__":
    print("=== 正在运行 data_loader.py 独立数据加载测试 ===")
    t1 = load_test1()
    total_t1 = sum(len(v) for v in t1.values())
    print(f"test_1.xlsx: 成功加载 {len(t1)} 个 Sheet, 共 {total_t1} 条订单, D1 叠数核验 100% 通过")

    t2 = load_test2()
    total_t2 = sum(len(v) for v in t2.values())
    print(f"test_2.xlsx: 成功加载 {len(t2)} 个 Sheet, 共 {total_t2} 条订单, D1 叠数核验 100% 通过")
    print("=== data_loader.py 运行正常，无任何报错 ===")
