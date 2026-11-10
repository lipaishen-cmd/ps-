"""
dynamic_validator.py
====================
动态插单与优先级数据校验器与加载器（TC-DYN-01 ~ TC-DYN-06）

本模块定义动态调度场景下的数据校验规范：
  - TC-DYN-01 表头与字段完整性校验：核验初始表 10 列、插单表 15 列标准字段
  - TC-DYN-02 优先级合法性校验：严格限定在 {"高", "中", "普通"}
  - TC-DYN-03 到达事件与时序因果校验：初始订单 arrival=0；动态插单预计到达支持数值或"未预报"；
              实际到达 >= 0；释放时刻 >= 实际到达时刻
  - TC-DYN-04 工艺参数与物理关系 D1 校验：W = ceil(Q / S) 零容差一致
  - TC-DYN-05 唯一性与关联性校验：事件ID全局唯一、实例名存在、同一实例内订单号不重
  - TC-DYN-06 机台资格与双资源池隔离：普通钻孔限定 Pool D，背钻限定 Pool B，禁止跨池
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import openpyxl

from data_models import (
    DEFAULT_H_MAP,
    DynamicInsertEvent,
    Order,
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
    VALID_STATUSES,
)
from eligibility_builder import build_eligibility_matrix, validate_cross_pool_isolation


# ---------------------------------------------------------------------------
# 规范字段字典定义
# ---------------------------------------------------------------------------

INITIAL_REQUIRED_FIELDS = {
    "order_id",
    "process_type",
    "Q",
    "efficiency",
    "S",
    "N",
    "W_excel",
    "raw_machine_grade",
    "expected_arrival_time",
    "priority",
    "due_date",
}

DYNAMIC_REQUIRED_FIELDS = {
    "instance_name",
    "event_id",
    "order_id",
    "process_type",
    "Q",
    "efficiency",
    "S",
    "N",
    "raw_machine_grade",
    "W_excel",
    "expected_arrival_time",
    "actual_arrival_time",
    "release_time",
    "priority",
    "due_date",
    "status",
}

DYNAMIC_HEADER_ALIASES: Dict[str, str] = {
    # 实例名 / 事件ID
    "实例名":                    "instance_name",
    "实例":                      "instance_name",
    "instance":                  "instance_name",
    "sheet名":                   "instance_name",
    "事件id":                    "event_id",
    "事件编号":                  "event_id",
    "event_id":                  "event_id",
    # 8 个工艺字段
    "订单":                      "order_id",
    "订单编号":                  "order_id",
    "生产工步":                  "process_type",
    "工步":                      "process_type",
    "panel数量":                 "Q",
    "加工效率（孔/min)":         "efficiency",
    "加工效率(孔/min)":          "efficiency",
    "加工效率（孔/min）":        "efficiency",
    "加工效率":                  "efficiency",
    "最大叠片数":                "S",
    "单pnl总孔数":               "N",
    "使用的钻机等级/或制定机台":   "raw_machine_grade",
    "使用的钻机等级/或指定机台":   "raw_machine_grade",
    "使用的钻机等级/指定机台":     "raw_machine_grade",
    "钻机等级":                  "raw_machine_grade",
    "叠数":                      "W_excel",
    # 动态排程特有字段
    "预计到达时刻":              "expected_arrival_time",
    "预计到达时间":              "expected_arrival_time",
    "实际到达时刻":              "actual_arrival_time",
    "实际到达时间":              "actual_arrival_time",
    "释放时刻":                  "release_time",
    "释放时间":                  "release_time",
    "优先级":                    "priority",
    "交期":                      "due_date",
    "工序交付期":                "due_date",
    "交付期":                    "due_date",
    "due_date":                  "due_date",
    "状态":                      "status",
}


def _norm(raw: Any) -> str:
    """标准化单元格字符串：去前后空格"""
    return "" if raw is None else str(raw).strip()


def _parse_positive_int(raw: Any, field_name: str, location: str) -> int:
    """解析正整数；拒绝小数、零、负数、空值和非有限数，禁止静默截断。"""
    text = _norm(raw)
    if not text:
        raise ValueError(f"{location}: 字段 '{field_name}' 不能为空，必须为正整数。")
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: 字段 '{field_name}'={text!r} 不是有效数值。") from exc
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValueError(
            f"{location}: 字段 '{field_name}'={text!r} 必须为正整数，不能截断小数、零或负数。"
        )
    return int(number)


def _parse_nonnegative_float(raw: Any, field_name: str, location: str) -> float:
    """解析有限非负数，用于到达/释放时刻。"""
    text = _norm(raw)
    if not text:
        raise ValueError(f"{location}: 字段 '{field_name}' 不能为空，必须为有限非负数。")
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: 字段 '{field_name}'={text!r} 不是有效数值。") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{location}: 字段 '{field_name}'={text!r} 必须为有限非负数。")
    return number


def _parse_efficiency(raw: Any, location: str) -> float:
    """解析并核验工艺效率白名单。"""
    text = _norm(raw)
    if not text:
        raise ValueError(f"{location}: 加工效率不能为空。")
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: 加工效率={text!r} 不是有效数值。") from exc
    if not math.isfinite(number) or number not in VALID_EFFICIENCIES:
        raise ValueError(
            f"{location}: 加工效率={text!r} 非法，必须为 {sorted(VALID_EFFICIENCIES)} 之一。"
        )
    return number


def parse_row_headers(header_cells) -> Dict[str, int]:
    """解析表头单元格行，返回 {规范字段名: 列索引}"""
    field_to_col: Dict[str, int] = {}
    for col_idx, cell in enumerate(header_cells):
        val = _norm(cell if not hasattr(cell, "value") else cell.value)
        canonical = DYNAMIC_HEADER_ALIASES.get(val) or DYNAMIC_HEADER_ALIASES.get(val.lower())
        if canonical:
            field_to_col[canonical] = col_idx
    return field_to_col


# ---------------------------------------------------------------------------
# 核心校验规则实现
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """动态数据校验不通过时抛出的结构化异常"""
    pass


def validate_dynamic_excel(filepath: str) -> Dict[str, Any]:
    """
    全量校验包含动态插单与优先级的 Excel 文件。
    严格执行规范 TC-DYN-01 ~ TC-DYN-06。

    若校验失败，抛出 ValidationError 并列出所有违规项；
    若校验成功，返回通过摘要信息字典。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: '{filepath}'")

    wb = openpyxl.load_workbook(filepath, data_only=True)
    all_sheets = wb.sheetnames
    errors: List[str] = []

    # 1. 检查“动态插单”Sheet 是否存在
    if "动态插单" not in all_sheets:
        errors.append("TC-DYN-01 FAIL: 工作簿中缺少必须的 '动态插单' 工作表。")

    # 2. 校验初始订单 Sheet (如 Sheet1~Sheet5)
    order_sheets = [s for s in all_sheets if s.lower().startswith("sheet")]
    if not order_sheets:
        errors.append("TC-DYN-01 FAIL: 工作簿中未发现任何以 'Sheet' 开头的订单实例表。")

    initial_orders_by_instance: Dict[str, List[Order]] = {}
    sheet_order_ids: Dict[str, Set[str]] = {}

    for sname in order_sheets:
        ws = wb[sname]
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 2:
            errors.append(f"TC-DYN-01 FAIL: 实例 '{sname}' 行数不足（仅有 {len(rows)} 行），无有效数据。")
            continue

        f2c = parse_row_headers(rows[0])
        # 检查初始订单表头
        missing_init = INITIAL_REQUIRED_FIELDS - set(f2c.keys())
        if missing_init:
            errors.append(
                f"TC-DYN-01 FAIL: 实例 '{sname}' 缺少必要列: {missing_init}。"
                f"初始表必须包含原8列 + '预计到达时刻' + '优先级'。"
            )
            continue

        inst_orders: List[Order] = []
        ids_seen: Set[str] = set()

        for r_idx, row in enumerate(rows[1:], start=2):
            if all(c is None or str(c).strip() == "" for c in row):
                continue
            oid_raw = _norm(row[f2c["order_id"]])
            if not oid_raw:
                errors.append(
                    f"TC-DYN-05 FAIL: 实例 '{sname}' 第 {r_idx} 行订单号为空。"
                )
                continue

            # TC-DYN-05: 实例内订单号唯一
            if oid_raw in ids_seen:
                errors.append(f"TC-DYN-05 FAIL: 实例 '{sname}' 第 {r_idx} 行订单号 '{oid_raw}' 重复。")
            ids_seen.add(oid_raw)

            # TC-DYN-02: 优先级校验
            pri = _norm(row[f2c["priority"]])
            if pri not in VALID_PRIORITIES:
                errors.append(
                    f"TC-DYN-02 FAIL: 实例 '{sname}' Row {r_idx} 订单 '{oid_raw}': "
                    f"优先级 '{pri}' 非法，必须为 {VALID_PRIORITIES} 之一。"
                )

            # TC-DYN-03: 初始订单预计到达时刻必须为 0
            arr_val = row[f2c["expected_arrival_time"]]
            try:
                arr_num = float(arr_val)
                if arr_num != 0.0:
                    errors.append(
                        f"TC-DYN-03 FAIL: 实例 '{sname}' Row {r_idx} 订单 '{oid_raw}': "
                        f"初始订单预计到达时刻必须为 0，当前为 {arr_num}。"
                    )
            except Exception:
                errors.append(
                    f"TC-DYN-03 FAIL: 实例 '{sname}' Row {r_idx} 订单 '{oid_raw}': "
                    f"预计到达时刻 '{arr_val}' 无法解析为数值 0。"
                )

            # 工艺字段与 D1 校验
            try:
                location = f"实例 '{sname}' Row {r_idx} 订单 '{oid_raw}'"
                q = _parse_positive_int(row[f2c["Q"]], "Q", location)
                s = _parse_positive_int(row[f2c["S"]], "S", location)
                n = _parse_positive_int(row[f2c["N"]], "N", location)
                eff = _parse_efficiency(row[f2c["efficiency"]], location)
                w_excel = _parse_positive_int(row[f2c["W_excel"]], "W", location)
                w_calc = math.ceil(q / s)
                if w_excel != w_calc:
                    errors.append(
                        f"TC-DYN-04 FAIL: 实例 '{sname}' Row {r_idx} 订单 '{oid_raw}': "
                        f"D1 叠数违规: Excel 叠数={w_excel} != ceil({q}/{s})={w_calc}。"
                    )
                ptype = _norm(row[f2c["process_type"]])
                if ptype in ("钻孔", "普通钻孔", "drill"):
                    ptype_std = PROCESS_DRILL
                elif ptype in ("背钻", "backdrill"):
                    ptype_std = PROCESS_BACKDRILL
                else:
                    errors.append(f"TC-DYN-06 FAIL: 实例 '{sname}' Row {r_idx} 订单 '{oid_raw}': 未知工步 '{ptype}'。")
                    continue

                raw_grade = _norm(row[f2c["raw_machine_grade"]])

                # TC-DYN-03: 交期校验
                due_val = row[f2c["due_date"]]
                due_time = 0.0
                try:
                    due_time = _parse_nonnegative_float(
                        due_val,
                        "交期",
                        location,
                    )
                except Exception:
                    errors.append(
                        f"TC-DYN-03 FAIL: 实例 '{sname}' Row {r_idx} 订单 '{oid_raw}': "
                        f"交期 '{due_val}' 无法解析为有效非负数值。"
                    )

                order_obj = Order(
                    order_id=oid_raw,
                    process_type=ptype_std,
                    Q=q,
                    S=s,
                    W=w_calc,
                    N=n,
                    efficiency=eff,
                    H=DEFAULT_H_MAP[ptype_std],
                    raw_machine_grade=raw_grade,
                    expected_arrival_time=0.0,
                    actual_arrival_time=0.0,
                    release_time=0.0,
                    priority=pri if pri in VALID_PRIORITIES else PRIORITY_NORMAL,
                    due_date=due_time,
                    instance_name=sname,
                    status="初始",
                    is_forecasted=True,
                )
                inst_orders.append(order_obj)
            except Exception as ex:
                errors.append(f"TC-DYN-04 FAIL: 实例 '{sname}' Row {r_idx} 解析工艺参数失败: {ex}")

        sheet_order_ids[sname] = ids_seen
        initial_orders_by_instance[sname] = inst_orders

        # 构建资格矩阵并校验隔离性
        if inst_orders:
            try:
                build_eligibility_matrix(inst_orders, strict=True)
                validate_cross_pool_isolation(inst_orders)
            except Exception as ex:
                errors.append(f"TC-DYN-06 FAIL: 实例 '{sname}' 机器资格/跨池校验失败: {ex}")

    # 3. 校验“动态插单”表
    dynamic_events: List[DynamicInsertEvent] = []
    if "动态插单" in all_sheets:
        ws_dyn = wb["动态插单"]
        dyn_rows = list(ws_dyn.iter_rows(values_only=True))
        if len(dyn_rows) < 2:
            errors.append("TC-DYN-01 FAIL: '动态插单' 工作表无有效数据行。")
        else:
            d_f2c = parse_row_headers(dyn_rows[0])
            missing_dyn = DYNAMIC_REQUIRED_FIELDS - set(d_f2c.keys())
            if missing_dyn:
                errors.append(
                    f"TC-DYN-01 FAIL: '动态插单' 表缺少必要列: {missing_dyn}。"
                    f"动态插单表必须包含全部 15 列标准工艺与事件字段。"
                )
            else:
                event_ids_seen: Set[str] = set()

                for r_idx, row in enumerate(dyn_rows[1:], start=2):
                    if all(c is None or str(c).strip() == "" for c in row):
                        continue

                    inst_name = _norm(row[d_f2c["instance_name"]])
                    ev_id = _norm(row[d_f2c["event_id"]])
                    oid_raw = _norm(row[d_f2c["order_id"]])

                    # TC-DYN-05: 实例名关联性
                    if not inst_name:
                        errors.append(f"TC-DYN-05 FAIL: 动态插单 Row {r_idx}: 实例名为空。")
                    if inst_name not in order_sheets:
                        errors.append(
                            f"TC-DYN-05 FAIL: 动态插单 Row {r_idx}: 实例名 '{inst_name}' 不存在于工作簿数据 Sheet 中。"
                        )

                    # TC-DYN-05: 事件ID唯一性
                    if not ev_id:
                        errors.append(f"TC-DYN-05 FAIL: 动态插单 Row {r_idx}: 事件ID为空。")
                    elif ev_id in event_ids_seen:
                        errors.append(f"TC-DYN-05 FAIL: 动态插单 Row {r_idx}: 事件ID '{ev_id}' 重复。")
                    if ev_id:
                        event_ids_seen.add(ev_id)

                    if not oid_raw:
                        errors.append(f"TC-DYN-05 FAIL: 动态插单 Row {r_idx}: 订单号为空。")

                    # TC-DYN-05: 插单订单号不能与初始订单号冲突，也不能与同一实例内其他插单冲突
                    if oid_raw and inst_name in sheet_order_ids:
                        if oid_raw in sheet_order_ids[inst_name]:
                            errors.append(
                                f"TC-DYN-05 FAIL: 动态插单 Row {r_idx}: 订单号 '{oid_raw}' 已存在于实例 '{inst_name}' 中。"
                            )
                        sheet_order_ids[inst_name].add(oid_raw)

                    # TC-DYN-02: 优先级
                    pri = _norm(row[d_f2c["priority"]])
                    if pri not in VALID_PRIORITIES:
                        errors.append(
                            f"TC-DYN-02 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"优先级 '{pri}' 非法，必须为 {VALID_PRIORITIES} 之一。"
                        )

                    # TC-DYN-03: 到达时刻与释放时刻校验
                    exp_val = row[d_f2c["expected_arrival_time"]]
                    exp_str = _norm(exp_val)
                    is_forecasted = True
                    exp_time: Optional[float] = None

                    if exp_str == UNFORECASTED_TAG:
                        is_forecasted = False
                        exp_time = None
                    elif not exp_str:
                        errors.append(
                            f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"预计到达时刻不能为空；未预报订单必须明确填写 '{UNFORECASTED_TAG}'。"
                        )
                        is_forecasted = False
                        exp_time = None
                    elif exp_str in ("未预测", "突发", "None"):
                        errors.append(
                            f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"预计到达标记 '{exp_str}' 不规范，未预报订单只能填写 '{UNFORECASTED_TAG}'。"
                        )
                        is_forecasted = False
                        exp_time = None
                    else:
                        try:
                            exp_time = _parse_nonnegative_float(
                                exp_str,
                                "预计到达时刻",
                                f"动态插单 Row {r_idx} 事件 '{ev_id}'",
                            )
                        except Exception:
                            errors.append(
                                f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                                f"预计到达时刻 '{exp_str}' 非法，必须为非负数值或 '{UNFORECASTED_TAG}'。"
                            )

                    act_val = row[d_f2c["actual_arrival_time"]]
                    act_time = -1.0
                    try:
                        act_time = _parse_nonnegative_float(
                            act_val,
                            "实际到达时刻",
                            f"动态插单 Row {r_idx} 事件 '{ev_id}'",
                        )
                    except Exception:
                        errors.append(
                            f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"实际到达时刻 '{act_val}' 无法解析为数值。"
                        )

                    rel_val = row[d_f2c["release_time"]]
                    rel_time = -1.0
                    try:
                        rel_time = _parse_nonnegative_float(
                            rel_val,
                            "释放时刻",
                            f"动态插单 Row {r_idx} 事件 '{ev_id}'",
                        )
                    except Exception:
                        errors.append(
                            f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"释放时刻 '{rel_val}' 无法解析为数值。"
                        )

                    # 释放时刻 >= 实际到达时刻
                    if act_time >= 0 and rel_time >= 0 and rel_time < act_time:
                        errors.append(
                            f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"释放时刻 ({rel_time}) 严禁早于实际到达时刻 ({act_time})！"
                        )

                    # TC-DYN-03: 交期校验
                    due_val = row[d_f2c["due_date"]]
                    due_time = -1.0
                    try:
                        due_time = _parse_nonnegative_float(
                            due_val,
                            "交期",
                            f"动态插单 Row {r_idx} 事件 '{ev_id}'",
                        )
                    except Exception:
                        errors.append(
                            f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"交期 '{due_val}' 无法解析为数值。"
                        )

                    # 交期 >= 实际到达时刻
                    if act_time >= 0 and due_time >= 0 and due_time < act_time:
                        errors.append(
                            f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"交期 ({due_time}) 严禁早于实际到达时刻 ({act_time})！"
                        )

                    # 状态检查
                    status_raw = _norm(row[d_f2c["status"]])
                    if status_raw not in VALID_STATUSES:
                        errors.append(
                            f"TC-DYN-03 FAIL: 动态插单 Row {r_idx} 事件 '{ev_id}': "
                            f"状态 '{status_raw}' 非法，必须明确填写 {VALID_STATUSES} 之一。"
                        )

                    # 工艺字段与 D1 校验
                    try:
                        location = f"动态插单 Row {r_idx} 订单 '{oid_raw}'"
                        q = _parse_positive_int(row[d_f2c["Q"]], "Q", location)
                        s = _parse_positive_int(row[d_f2c["S"]], "S", location)
                        n = _parse_positive_int(row[d_f2c["N"]], "N", location)
                        eff = _parse_efficiency(row[d_f2c["efficiency"]], location)
                        w_excel = _parse_positive_int(row[d_f2c["W_excel"]], "W", location)
                        w_calc = math.ceil(q / s)
                        if w_excel != w_calc:
                            errors.append(
                                f"TC-DYN-04 FAIL: 动态插单 Row {r_idx} 订单 '{oid_raw}': "
                                f"D1 叠数违规: Excel 叠数={w_excel} != ceil({q}/{s})={w_calc}。"
                            )
                        ptype = _norm(row[d_f2c["process_type"]])
                        if ptype in ("钻孔", "普通钻孔", "drill"):
                            ptype_std = PROCESS_DRILL
                        elif ptype in ("背钻", "backdrill"):
                            ptype_std = PROCESS_BACKDRILL
                        else:
                            errors.append(f"TC-DYN-06 FAIL: 动态插单 Row {r_idx} 订单 '{oid_raw}': 未知工步 '{ptype}'。")
                            continue

                        raw_grade = _norm(row[d_f2c["raw_machine_grade"]])

                        order_obj = Order(
                            order_id=oid_raw,
                            process_type=ptype_std,
                            Q=q,
                            S=s,
                            W=w_calc,
                            N=n,
                            efficiency=eff,
                            H=DEFAULT_H_MAP[ptype_std],
                            raw_machine_grade=raw_grade,
                            expected_arrival_time=exp_time,
                            actual_arrival_time=act_time if act_time >= 0 else 0.0,
                            release_time=rel_time if rel_time >= 0 else 0.0,
                            priority=pri if pri in VALID_PRIORITIES else PRIORITY_NORMAL,
                            due_date=due_time,
                            event_id=ev_id,
                            instance_name=inst_name,
                            status=status_raw or STATUS_PENDING,
                            is_forecasted=is_forecasted,
                        )

                        # 构建该插单的合格机台
                        build_eligibility_matrix([order_obj], strict=True)
                        validate_cross_pool_isolation([order_obj])

                        event = DynamicInsertEvent(
                            event_id=ev_id,
                            instance_name=inst_name,
                            order=order_obj,
                            expected_arrival_time=exp_time,
                            actual_arrival_time=act_time,
                            release_time=rel_time,
                            priority=pri,
                            status=status_raw or STATUS_PENDING,
                            is_forecasted=is_forecasted,
                        )
                        dynamic_events.append(event)
                    except Exception as ex:
                        errors.append(f"TC-DYN-04/06 FAIL: 动态插单 Row {r_idx} 解析工艺/机台失败: {ex}")

    # 如果有任何违规，汇总抛出异常
    if errors:
        msg = f"动态 Excel 数据校验失败，共发现 {len(errors)} 项违规：\n" + "\n".join(errors)
        raise ValidationError(msg)

    total_initial = sum(len(orders) for orders in initial_orders_by_instance.values())
    summary = {
        "status": "PASS",
        "order_sheets": order_sheets,
        "total_initial_orders": total_initial,
        "initial_orders_by_instance": {k: len(v) for k, v in initial_orders_by_instance.items()},
        "total_dynamic_events": len(dynamic_events),
        "events": [
            {
                "event_id": e.event_id,
                "instance": e.instance_name,
                "order_id": e.order.order_id,
                "is_forecasted": e.is_forecasted,
                "expected": e.expected_arrival_time,
                "actual": e.actual_arrival_time,
                "release": e.release_time,
                "priority": e.priority,
            }
            for e in dynamic_events
        ],
    }
    return summary


# ---------------------------------------------------------------------------
# 高级加载接口
# ---------------------------------------------------------------------------

def load_dynamic_orders_and_events(
    filepath: str,
    instance_name: str,
) -> Tuple[List[Order], List[DynamicInsertEvent]]:
    """
    针对指定静态实例（例如 'Sheet1'），同时加载其初始订单和对应的动态插单事件列表。
    返回：(初始订单列表, 按实际到达时刻升序排序的动态插单事件列表)
    """
    validate_dynamic_excel(filepath)

    wb = openpyxl.load_workbook(filepath, data_only=True)
    if instance_name not in wb.sheetnames:
        raise ValueError(f"指定实例 '{instance_name}' 不存在于 '{filepath}' 中。")

    # 1. 加载初始订单
    ws = wb[instance_name]
    rows = list(ws.iter_rows(values_only=True))
    f2c = parse_row_headers(rows[0])

    initial_orders: List[Order] = []
    for row in rows[1:]:
        if all(c is None or str(c).strip() == "" for c in row):
            continue
        oid = _norm(row[f2c["order_id"]])
        if not oid:
            continue
        ptype = _norm(row[f2c["process_type"]])
        ptype_std = PROCESS_BACKDRILL if ptype in ("背钻", "backdrill") else PROCESS_DRILL
        location = f"实例 '{instance_name}' 订单 '{oid}'"
        q = _parse_positive_int(row[f2c["Q"]], "Q", location)
        s = _parse_positive_int(row[f2c["S"]], "S", location)
        n = _parse_positive_int(row[f2c["N"]], "N", location)
        eff = _parse_efficiency(row[f2c["efficiency"]], location)
        col_grade = f2c.get("raw_machine_grade")
        raw_grade = _norm(row[col_grade]) if col_grade is not None and col_grade < len(row) else ""
        col_pri = f2c.get("priority")
        pri = _norm(row[col_pri]) if col_pri is not None and col_pri < len(row) else PRIORITY_NORMAL
        col_due = f2c.get("due_date")
        due_val = _parse_nonnegative_float(row[col_due], "交期", location) if col_due is not None and col_due < len(row) and row[col_due] not in (None, "") else None

        order = Order(
            order_id=oid,
            process_type=ptype_std,
            Q=q,
            S=s,
            W=math.ceil(q / s),
            N=n,
            efficiency=eff,
            H=DEFAULT_H_MAP[ptype_std],
            raw_machine_grade=raw_grade,
            expected_arrival_time=0.0,
            actual_arrival_time=0.0,
            release_time=0.0,
            priority=pri if pri in VALID_PRIORITIES else PRIORITY_NORMAL,
            due_date=due_val,
            instance_name=instance_name,
            status="初始",
            is_forecasted=True,
        )
        initial_orders.append(order)

    build_eligibility_matrix(initial_orders, strict=True)
    validate_cross_pool_isolation(initial_orders)

    # 2. 加载该实例的动态插单事件
    events: List[DynamicInsertEvent] = []
    if "动态插单" in wb.sheetnames:
        ws_dyn = wb["动态插单"]
        drows = list(ws_dyn.iter_rows(values_only=True))
        d_f2c = parse_row_headers(drows[0])
        for row in drows[1:]:
            if all(c is None or str(c).strip() == "" for c in row):
                continue
            inst = _norm(row[d_f2c["instance_name"]])
            if inst != instance_name:
                continue

            ev_id = _norm(row[d_f2c["event_id"]])
            oid = _norm(row[d_f2c["order_id"]])
            ptype = _norm(row[d_f2c["process_type"]])
            ptype_std = PROCESS_BACKDRILL if ptype in ("背钻", "backdrill") else PROCESS_DRILL
            location = f"动态插单事件 '{ev_id}' 订单 '{oid}'"
            q = _parse_positive_int(row[d_f2c["Q"]], "Q", location)
            s = _parse_positive_int(row[d_f2c["S"]], "S", location)
            n = _parse_positive_int(row[d_f2c["N"]], "N", location)
            eff = _parse_efficiency(row[d_f2c["efficiency"]], location)
            raw_grade = _norm(row[d_f2c["raw_machine_grade"]])

            exp_str = _norm(row[d_f2c["expected_arrival_time"]])
            is_forecasted = exp_str != UNFORECASTED_TAG
            exp_time = _parse_nonnegative_float(exp_str, "预计到达时刻", location) if is_forecasted else None

            act_time = float(row[d_f2c["actual_arrival_time"]])
            rel_time = float(row[d_f2c["release_time"]])
            pri = _norm(row[d_f2c["priority"]])
            st = _norm(row[d_f2c["status"]])
            col_due = d_f2c.get("due_date")
            due_val = _parse_nonnegative_float(row[col_due], "交期", location) if col_due is not None and col_due < len(row) and row[col_due] not in (None, "") else None

            ord_obj = Order(
                order_id=oid,
                process_type=ptype_std,
                Q=q,
                S=s,
                W=math.ceil(q / s),
                N=n,
                efficiency=eff,
                H=DEFAULT_H_MAP[ptype_std],
                raw_machine_grade=raw_grade,
                expected_arrival_time=exp_time,
                actual_arrival_time=act_time,
                release_time=rel_time,
                priority=pri if pri in VALID_PRIORITIES else PRIORITY_NORMAL,
                due_date=due_val,
                event_id=ev_id,
                instance_name=instance_name,
                status=st,
                is_forecasted=is_forecasted,
            )
            build_eligibility_matrix([ord_obj], strict=True)
            validate_cross_pool_isolation([ord_obj])

            ev = DynamicInsertEvent(
                event_id=ev_id,
                instance_name=instance_name,
                order=ord_obj,
                expected_arrival_time=exp_time,
                actual_arrival_time=act_time,
                release_time=rel_time,
                priority=pri,
                status=st,
                is_forecasted=is_forecasted,
            )
            events.append(ev)

    # 按照实际到达时刻升序排序（重调度事件驱动流）
    events.sort(key=lambda x: (x.actual_arrival_time, x.event_id))
    return initial_orders, events
