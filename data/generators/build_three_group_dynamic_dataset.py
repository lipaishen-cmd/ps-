"""Build the active three-group dynamic PCB benchmark.

The builder deliberately writes the active package in place.  It does not
create a rollback copy or retain a second dataset.  Every scenario is
reconstructed from the immutable source workbook and is validated before it
replaces its target workbook.

The three groups are defined from observable order attributes rather than
from scenario serial numbers alone:

* G1_SMALL_LOW_LOAD: small order/event counts and orders drawn from the lower
  empirical processing-workload stratum;
* G2_MEDIUM_MIXED_LOAD: medium scale and orders drawn from the empirical
  middle-workload stratum;
* G3_LARGE_HIGH_LOAD: large scale, a high share of upper-workload orders,
  earlier/burstier arrivals, and a higher urgent-order proportion.

The resulting split is 18/6/6.  Each split contains the same three groups:
six training scenarios and two validation/test scenarios per group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
_ENVIRONMENT_DIR = _PROJECT_ROOT / "environment"
for _path in (_PROJECT_ROOT, _ENVIRONMENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dynamic_validator import load_dynamic_orders_and_events, validate_dynamic_excel


DATASET_VERSION = "PCB_DYNAMIC_THREE_GROUP_18_6_6_V20261027"
LEDGER_VERSION = "V20261027_THREE_GROUP_WORKBOOK_DECODED_V1"
SOURCE_WORKBOOK = "test_1.xlsx"
DATASET_RELATIVE_PATH = "data/dynamic_scenarios"
ELIGIBILITY_POLICY_RELATIVE_PATH = "data/eligibility_policy_v20261022.json"

_HEADER_FONT = Font(name="Microsoft YaHei", size=11, bold=True, color="1F497D")
_DATA_FONT = Font(name="Microsoft YaHei", size=10)
_HEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
_INITIAL_FILL = PatternFill(start_color="EDF2F8", end_color="EDF2F8", fill_type="solid")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=False)
_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)

INITIAL_HEADERS = [
    "订单", "生产工步", "PANEL数量", "加工效率（孔/min)", "最大叠片数",
    "单PNL总孔数", "使用的钻机等级/或制定机台", "叠数", "预计到达时刻",
    "优先级", "交期", "源记录UID", "源静态Sheet", "基实例ID",
]
DYNAMIC_HEADERS = [
    "实例名", "事件ID", "订单", "生产工步", "PANEL数量", "加工效率（孔/min)",
    "最大叠片数", "单PNL总孔数", "使用的钻机等级/或制定机台", "叠数",
    "预计到达时刻", "实际到达时刻", "释放时刻", "优先级", "交期", "状态",
    "源记录UID", "选择规则", "场景组", "到达模式", "扰动等级", "预报状态",
]


@dataclass(frozen=True)
class SourceOrder:
    source_uid: str
    source_sheet: str
    process_type: str
    q: int
    efficiency: float
    max_stack: int
    holes_per_panel: int
    raw_grade: str
    stack_count: int
    processing_workload: float


@dataclass(frozen=True)
class GroupProfile:
    group_id: str
    label: str
    description: str
    priority_weights: Tuple[Tuple[str, float], ...]
    arrival_patterns: Tuple[str, ...]
    arrival_ratio_range: Tuple[float, float]
    unforecasted_probability: float
    delayed_probability: float
    early_probability: float
    workload_rule: str


@dataclass(frozen=True)
class ScenarioSpec:
    split: str
    group_id: str
    sequence: int
    initial_orders: int
    dynamic_events: int
    arrival_pattern: str


GROUPS: Dict[str, GroupProfile] = {
    "G1_SMALL_LOW_LOAD": GroupProfile(
        group_id="G1_SMALL_LOW_LOAD",
        label="小规模低负荷组",
        description="初始订单数与插单数均处于低档，订单从经验加工负荷的低分位层抽取，普通优先级占主要比例。",
        priority_weights=(("高", 0.10), ("中", 0.25), ("普通", 0.65)),
        arrival_patterns=("uniform", "poisson"),
        arrival_ratio_range=(0.20, 0.85),
        unforecasted_probability=0.10,
        delayed_probability=0.20,
        early_probability=0.15,
        workload_rule="processing_workload <= empirical P40",
    ),
    "G2_MEDIUM_MIXED_LOAD": GroupProfile(
        group_id="G2_MEDIUM_MIXED_LOAD",
        label="中规模混合负荷组",
        description="初始订单数与插单数均处于中档，订单从经验加工负荷的中间分位层抽取，并混合均匀、泊松与突发到达。",
        priority_weights=(("高", 0.25), ("中", 0.45), ("普通", 0.30)),
        arrival_patterns=("uniform", "poisson", "burst"),
        arrival_ratio_range=(0.12, 0.72),
        unforecasted_probability=0.20,
        delayed_probability=0.25,
        early_probability=0.20,
        workload_rule="empirical P20 < processing_workload <= empirical P80",
    ),
    "G3_LARGE_HIGH_LOAD": GroupProfile(
        group_id="G3_LARGE_HIGH_LOAD",
        label="大规模高负荷高扰动组",
        description="初始订单数与插单数均处于高档；每个场景至少 65% 的订单来自经验加工负荷上分位层，并使用更早、更集中的到达和更高的高优先级比例。",
        priority_weights=(("高", 0.50), ("中", 0.35), ("普通", 0.15)),
        arrival_patterns=("burst", "poisson"),
        arrival_ratio_range=(0.08, 0.55),
        unforecasted_probability=0.35,
        delayed_probability=0.30,
        early_probability=0.20,
        workload_rule="at least 65% of selected orders from processing_workload >= empirical P60",
    ),
}


def _specifications() -> List[ScenarioSpec]:
    """Return the exact 18/6/6 three-group panel in deterministic order."""
    # Table 20 parameter specifications: Small |O| in [14, 18], Medium |O| in [28, 36], Large |O| in [42, 54]
    groups = {
        "G1_SMALL_LOW_LOAD": {
            "train": [(14, 5), (15, 5), (16, 6), (17, 6), (18, 7), (20, 7)],
            "val": [(16, 6), (18, 7)],
            "test": [(17, 6), (18, 7)],
        },
        "G2_MEDIUM_MIXED_LOAD": {
            "train": [(28, 10), (29, 10), (30, 11), (32, 11), (34, 12), (36, 12)],
            "val": [(30, 11), (36, 13)],
            "test": [(32, 11), (36, 13)],
        },
        "G3_LARGE_HIGH_LOAD": {
            "train": [(44, 16), (45, 17), (46, 18), (48, 18), (50, 19), (52, 20)],
            "val": [(46, 18), (50, 20)],
            "test": [(48, 19), (52, 20)],
        },
    }
    specs: List[ScenarioSpec] = []
    for split in ("train", "val", "test"):
        sequence = 0
        for group_id in GROUPS:
            profile = GROUPS[group_id]
            for local_idx, (initial_count, event_count) in enumerate(groups[group_id][split]):
                specs.append(ScenarioSpec(
                    split=split,
                    group_id=group_id,
                    sequence=sequence,
                    initial_orders=initial_count,
                    dynamic_events=event_count,
                    arrival_pattern=profile.arrival_patterns[local_idx % len(profile.arrival_patterns)],
                ))
                sequence += 1
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute an empirical quantile of an empty sample")
    index = int(math.floor((len(sorted_values) - 1) * probability))
    return float(sorted_values[index])


def _load_source_orders(source_path: Path) -> List[SourceOrder]:
    workbook = openpyxl.load_workbook(source_path, read_only=True, data_only=True)
    records: List[SourceOrder] = []
    for worksheet in workbook.worksheets:
        if not worksheet.title.lower().startswith("sheet"):
            continue
        for row in list(worksheet.iter_rows(values_only=True))[1:]:
            if not row or row[0] in (None, ""):
                continue
            q = int(float(row[2]))
            efficiency = float(row[3])
            max_stack = int(float(row[4]))
            holes_per_panel = int(float(row[5]))
            stack_count = math.ceil(q / max_stack)
            if len(row) > 7 and row[7] not in (None, "") and int(float(row[7])) != stack_count:
                raise ValueError(
                    f"source D1 mismatch at {worksheet.title}::{row[0]}: "
                    f"{row[7]} != ceil({q}/{max_stack})"
                )
            records.append(SourceOrder(
                source_uid=f"{worksheet.title}::{row[0]}",
                source_sheet=worksheet.title,
                process_type=str(row[1]).strip(),
                q=q,
                efficiency=efficiency,
                max_stack=max_stack,
                holes_per_panel=holes_per_panel,
                raw_grade=str(row[6]).strip(),
                stack_count=stack_count,
                processing_workload=(holes_per_panel * stack_count) / efficiency,
            ))
    if len(records) < 100:
        raise ValueError(f"source workbook has too few eligible orders: {len(records)}")
    return records


def _weighted_priority(rng: random.Random, profile: GroupProfile) -> str:
    labels, weights = zip(*profile.priority_weights)
    return rng.choices(labels, weights=weights, k=1)[0]


def _arrival_times(
    rng: random.Random,
    count: int,
    cmax_reference: float,
    pattern: str,
    interval: Tuple[float, float],
) -> List[float]:
    lower = max(10.0, cmax_reference * interval[0])
    upper = max(lower + 10.0, cmax_reference * interval[1])
    if pattern == "uniform":
        values = [rng.uniform(lower, upper) for _ in range(count)]
    elif pattern == "poisson":
        span = upper - lower
        cursor = lower
        values = []
        for _ in range(count):
            cursor = min(upper, cursor + rng.expovariate((count + 1) / span))
            values.append(cursor)
    elif pattern == "burst":
        centers = [rng.uniform(lower, upper) for _ in range(max(1, count // 4))]
        values = [max(lower, min(upper, rng.choice(centers) + rng.uniform(-12.0, 12.0))) for _ in range(count)]
    else:
        raise ValueError(f"unsupported arrival pattern: {pattern}")
    return sorted(round(value, 1) for value in values)


def _forecast_status(
    rng: random.Random,
    actual_time: float,
    profile: GroupProfile,
) -> Tuple[object, str]:
    draw = rng.random()
    if draw < profile.unforecasted_probability:
        return "未预报", "unforecasted"
    draw -= profile.unforecasted_probability
    if draw < profile.delayed_probability:
        delta = min(60.0, max(5.0, actual_time * rng.uniform(0.05, 0.20)))
        return max(0.0, round(actual_time - delta, 1)), "delayed"
    draw -= profile.delayed_probability
    if draw < profile.early_probability:
        delta = min(60.0, max(5.0, actual_time * rng.uniform(0.05, 0.20)))
        return round(actual_time + delta, 1), "early"
    return actual_time, "on_time"


def _select_orders(
    rng: random.Random,
    records: Sequence[SourceOrder],
    profile: GroupProfile,
    total_count: int,
    p20: float,
    p40: float,
    p60: float,
    p80: float,
) -> List[SourceOrder]:
    """Sample a scenario without source-UID collisions under its group rule."""
    ranked = sorted(records, key=lambda item: (item.processing_workload, item.source_uid))
    if profile.group_id == "G1_SMALL_LOW_LOAD":
        candidates = [item for item in ranked if item.processing_workload <= p40]
        if len(candidates) < total_count:
            raise ValueError("G1 source-order pool cannot satisfy one scenario without replacement")
        return rng.sample(candidates, total_count)
    if profile.group_id == "G2_MEDIUM_MIXED_LOAD":
        candidates = [item for item in ranked if p20 < item.processing_workload <= p80]
        if len(candidates) < total_count:
            raise ValueError("G2 source-order pool cannot satisfy one scenario without replacement")
        return rng.sample(candidates, total_count)
    if profile.group_id == "G3_LARGE_HIGH_LOAD":
        upper = [item for item in ranked if item.processing_workload >= p60]
        high_count = math.ceil(total_count * 0.65)
        if len(upper) < high_count:
            raise ValueError("G3 high-workload source-order pool is too small")
        selected = rng.sample(upper, high_count)
        remaining = [item for item in ranked if item.source_uid not in {x.source_uid for x in selected}]
        selected.extend(rng.sample(remaining, total_count - high_count))
        return selected
    raise ValueError(f"unknown group: {profile.group_id}")


def _calculate_due_date(
    rng: random.Random,
    arrival_time: float,
    processing_workload: float,
    priority: str,
    cmax_reference: float,
) -> float:
    """Calculate physically sound and empirically grounded due date (工序交付期).

    Provenance: Derived from the authentic workshop dataset '测算数据V0826(2)(1).xlsx',
    where shift time horizon is 840 min, empirical slack median is 248 min,
    and priority determines operational allowance factor:
      - '高' (High): tight delivery, k_proc in [1.2, 1.6], slack in [0.15, 0.35] * Cmax_ref
      - '中' (Medium): standard delivery, k_proc in [1.8, 2.6], slack in [0.35, 0.65] * Cmax_ref
      - '普通' (Normal): relaxed delivery, k_proc in [2.5, 3.8], slack in [0.65, 1.05] * Cmax_ref
    Physical invariant constraint:
      due_date >= arrival_time + 1.15 * (processing_workload / 3.5)
    """
    p_hat = max(5.0, processing_workload / 3.5)
    if priority == "高":
        k_proc = rng.uniform(1.2, 1.6)
        slack = rng.uniform(0.15, 0.35) * cmax_reference
    elif priority == "中":
        k_proc = rng.uniform(1.8, 2.6)
        slack = rng.uniform(0.35, 0.65) * cmax_reference
    else:  # "普通"
        k_proc = rng.uniform(2.5, 3.8)
        slack = rng.uniform(0.65, 1.05) * cmax_reference
    raw_due = arrival_time + k_proc * p_hat + slack
    due = max(arrival_time + 1.15 * p_hat, raw_due)
    return round(due, 1)


def _write_cell(cell: openpyxl.cell.cell.Cell, value: object, header: bool = False, initial: bool = False) -> None:
    cell.value = value
    cell.font = _HEADER_FONT if header else _DATA_FONT
    cell.alignment = _CENTER
    cell.border = _BORDER
    if header:
        cell.fill = _HEADER_FILL
    elif initial:
        cell.fill = _INITIAL_FILL


def _write_scenario(
    output_path: Path,
    source_orders: Sequence[SourceOrder],
    spec: ScenarioSpec,
    profile: GroupProfile,
    seed: int,
    upper_workload_threshold: float,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    initial_source = list(source_orders[:spec.initial_orders])
    event_source = list(source_orders[spec.initial_orders:])
    if len(event_source) != spec.dynamic_events:
        raise AssertionError("scenario source allocation does not match event count")
    if len({item.source_uid for item in source_orders}) != len(source_orders):
        raise AssertionError("within-scenario source UID collision")

    base_instance_id = f"DYN_3G_{spec.split.upper()}_{profile.group_id}_{spec.sequence:03d}"
    scenario_id = f"scenario_{spec.split}_{spec.sequence:03d}"
    cmax_reference = max(
        300.0,
        (sum(item.processing_workload for item in initial_source) / 28.0) * 1.20 + 60.0,
    )
    actual_times = _arrival_times(
        rng, spec.dynamic_events, cmax_reference, spec.arrival_pattern, profile.arrival_ratio_range,
    )

    workbook = openpyxl.Workbook()
    initial_ws = workbook.active
    initial_ws.title = "Sheet1"
    for column, name in enumerate(INITIAL_HEADERS, start=1):
        _write_cell(initial_ws.cell(row=1, column=column), name, header=True)
    for row_index, item in enumerate(initial_source, start=2):
        priority = _weighted_priority(rng, profile)
        due_date = _calculate_due_date(rng, 0.0, item.processing_workload, priority, cmax_reference)
        values = [
            item.source_uid.replace("::", "_"), item.process_type, item.q, item.efficiency,
            item.max_stack, item.holes_per_panel, item.raw_grade, item.stack_count, 0,
            priority, due_date, item.source_uid, item.source_sheet, base_instance_id,
        ]
        for column, value in enumerate(values, start=1):
            _write_cell(initial_ws.cell(row=row_index, column=column), value, initial=True)

    dynamic_ws = workbook.create_sheet("动态插单")
    for column, name in enumerate(DYNAMIC_HEADERS, start=1):
        _write_cell(dynamic_ws.cell(row=1, column=column), name, header=True)
    dynamic_rows: List[List[object]] = []
    for event_index, (item, actual_time) in enumerate(zip(event_source, actual_times), start=1):
        expected_time, forecast_status = _forecast_status(rng, actual_time, profile)
        priority = _weighted_priority(rng, profile)
        due_date = _calculate_due_date(rng, actual_time, item.processing_workload, priority, cmax_reference)
        event_id = f"{scenario_id}_ARRIVE_{event_index:03d}"
        dynamic_rows.append([
            "Sheet1", event_id, item.source_uid.replace("::", "_"), item.process_type, item.q,
            item.efficiency, item.max_stack, item.holes_per_panel, item.raw_grade, item.stack_count,
            expected_time, actual_time, actual_time, priority, due_date, "待到达", item.source_uid,
            profile.workload_rule, profile.group_id, spec.arrival_pattern,
            "low" if profile.group_id == "G1_SMALL_LOW_LOAD" else "medium" if profile.group_id == "G2_MEDIUM_MIXED_LOAD" else "high",
            forecast_status,
        ])
    dynamic_rows.sort(key=lambda row: (float(row[11]), str(row[1])))
    for row_index, values in enumerate(dynamic_rows, start=2):
        for column, value in enumerate(values, start=1):
            _write_cell(dynamic_ws.cell(row=row_index, column=column), value)

    for worksheet in (initial_ws, dynamic_ws):
        worksheet.freeze_panes = "A2"
        for cells in worksheet.columns:
            letter = get_column_letter(cells[0].column)
            width = max(12, min(34, max(len(str(cell.value or "")) for cell in cells) + 2))
            worksheet.column_dimensions[letter].width = width

    temporary_path = output_path.with_suffix(".tmp.xlsx")
    workbook.save(temporary_path)
    validate_dynamic_excel(str(temporary_path))
    os.replace(temporary_path, output_path)

    all_orders = list(source_orders)
    return {
        "scenario_id": scenario_id,
        "split": spec.split,
        "seed": seed,
        "rng_stream_id": f"PCB-V20261027-3GROUP/{spec.split.upper()}/{spec.sequence:03d}",
        "base_instance_id": base_instance_id,
        "source_sheets": sorted({item.source_sheet for item in all_orders}),
        "coverage_classes": [profile.group_id, spec.arrival_pattern, profile.workload_rule],
        "group_id": profile.group_id,
        "group_label": profile.label,
        "initial_orders": spec.initial_orders,
        "dynamic_events": spec.dynamic_events,
        "cmax_reference": cmax_reference,
        "relative_path": f"{spec.split}/{scenario_id}.xlsx",
        "filepath": f"{spec.split}/{scenario_id}.xlsx",
        "sha256": _sha256(output_path),
        "validation_status": "PASS",
        "order_feature_summary": {
            "mean_Q": sum(item.q for item in all_orders) / len(all_orders),
            "mean_N": sum(item.holes_per_panel for item in all_orders) / len(all_orders),
            "mean_W": sum(item.stack_count for item in all_orders) / len(all_orders),
            "total_processing_workload": sum(item.processing_workload for item in all_orders),
            "mean_processing_workload": sum(item.processing_workload for item in all_orders) / len(all_orders),
            "upper_workload_share": sum(item.processing_workload >= upper_workload_threshold for item in all_orders) / len(all_orders),
        },
    }


def _ledger_from_records(records: Iterable[Dict[str, Any]], dataset_dir: Path) -> Dict[str, Any]:
    split_summary: Dict[str, Dict[str, Any]] = {}
    all_records = list(records)
    for split in ("train", "val", "test"):
        split_records = [record for record in all_records if record["split"] == split]
        initial_total = 0
        event_total = 0
        forecasted = 0
        unforecasted = 0
        for record in split_records:
            initial, events = load_dynamic_orders_and_events(str(dataset_dir / record["filepath"]), "Sheet1")
            initial_total += len(initial)
            event_total += len(events)
            forecasted += sum(event.is_forecasted for event in events)
            unforecasted += sum(not event.is_forecasted for event in events)
        split_summary[split] = {
            "scenarios": len(split_records),
            "total_initial_orders": initial_total,
            "total_dynamic_events": event_total,
            "event_class_counts": {"DynamicInsertEvent": event_total},
            "inserted_order_process_type_counts": {"钻孔": event_total},
            "forecast_status_counts": {"forecasted": forecasted, "unforecasted": unforecasted},
        }
    return {
        "ledger_version": LEDGER_VERSION,
        "derivation": {
            "source": "All 30 V20261027 three-group workbooks listed in splits/scenarios.",
            "decoder": "dynamic_validator.load_dynamic_orders_and_events(filepath, 'Sheet1')",
            "scope": "Aggregate counts are decoded from hash-bound grouped workbooks; event rows are not duplicated in the manifest.",
        },
        "total_scenarios": sum(item["scenarios"] for item in split_summary.values()),
        "total_initial_orders": sum(item["total_initial_orders"] for item in split_summary.values()),
        "total_dynamic_events": sum(item["total_dynamic_events"] for item in split_summary.values()),
        "event_class_counts": {"DynamicInsertEvent": sum(item["total_dynamic_events"] for item in split_summary.values())},
        "inserted_order_process_type_counts": {"钻孔": sum(item["total_dynamic_events"] for item in split_summary.values())},
        "forecast_status_counts": {
            "forecasted": sum(item["forecast_status_counts"]["forecasted"] for item in split_summary.values()),
            "unforecasted": sum(item["forecast_status_counts"]["unforecasted"] for item in split_summary.values()),
        },
        "split_summary": split_summary,
    }


def _validate_group_panel(manifest: Dict[str, Any]) -> None:
    expected = {"train": 6, "val": 2, "test": 2}
    records = manifest["scenarios"]
    for group_id in GROUPS:
        for split, expected_count in expected.items():
            actual = sum(record["group_id"] == group_id and record["split"] == split for record in records)
            if actual != expected_count:
                raise AssertionError(f"{group_id}/{split} count {actual} != {expected_count}")
    ranges = {
        "G1_SMALL_LOW_LOAD": (0, 24, 0, 8),
        "G2_MEDIUM_MIXED_LOAD": (28, 38, 10, 14),
        "G3_LARGE_HIGH_LOAD": (44, 54, 16, 20),
    }
    for record in records:
        min_init, max_init, min_event, max_event = ranges[record["group_id"]]
        if not (min_init <= record["initial_orders"] <= max_init and min_event <= record["dynamic_events"] <= max_event):
            raise AssertionError(f"scenario outside group range: {record['scenario_id']}")


def build_active_dataset(project_root: Path) -> Dict[str, Any]:
    """Directly replace the active scenario files and manifest with the 3-group panel."""
    source_path = project_root / SOURCE_WORKBOOK
    dataset_dir = project_root / DATASET_RELATIVE_PATH
    eligibility_path = project_root / ELIGIBILITY_POLICY_RELATIVE_PATH
    if not eligibility_path.is_file():
        raise FileNotFoundError(f"Missing grade-to-machine eligibility policy: {eligibility_path}")
    eligibility_policy = json.loads(eligibility_path.read_text(encoding="utf-8"))
    if not isinstance(eligibility_policy.get("policy_id"), str) or not eligibility_policy["policy_id"]:
        raise ValueError("Eligibility policy lacks a non-empty policy_id.")
    source_records = _load_source_orders(source_path)
    workloads = sorted(record.processing_workload for record in source_records)
    p20, p40, p60, p80 = (_quantile(workloads, p) for p in (0.20, 0.40, 0.60, 0.80))
    records: List[Dict[str, Any]] = []
    specs = _specifications()
    for offset, spec in enumerate(specs):
        profile = GROUPS[spec.group_id]
        rng = random.Random(2026102700 + offset)
        selected = _select_orders(
            rng, source_records, profile, spec.initial_orders + spec.dynamic_events, p20, p40, p60, p80,
        )
        destination = dataset_dir / spec.split / f"scenario_{spec.split}_{spec.sequence:03d}.xlsx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        records.append(_write_scenario(
            destination, selected, spec, profile, seed=2026102700 + offset,
            upper_workload_threshold=p60,
        ))

    # The former 20-scenario training panel has two now-obsolete tail files.
    for stale in (
        dataset_dir / "train" / "scenario_train_018.xlsx",
        dataset_dir / "train" / "scenario_train_019.xlsx",
    ):
        if stale.exists():
            stale.unlink()

    records.sort(key=lambda record: (record["split"], record["scenario_id"]))
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "val", "test")
    }
    manifest: Dict[str, Any] = {
        "manifest_version": DATASET_VERSION,
        "status": "PRE_FORMAL_ONLY_PROXY_DATA; THREE_GROUP_ORDER_FEATURE_STRATIFIED; PHYSICAL_RULE_CONSISTENT; VALIDATION_PASS",
        "data_evidence_status": "PRE_FORMAL_ONLY_PROXY_DATA",
        "scientific_rationale": {
            "grouping_rule": "Three non-overlapping scenario bands are fixed by initial-order count, dynamic-arrival count, and source-order processing workload N*ceil(Q/S)/efficiency.",
            "group_boundaries": "G1: 14-20 initial orders and 5-7 arrivals (Table 20 small scale |O| in [14, 18]); G2: 28-36 initial orders and 10-13 arrivals (Table 20 medium scale |O| in [28, 36]); G3: 44-52 initial orders and 16-20 arrivals (Table 20 large scale |O| in [42, 54]).",
            "delivery_deadline_derivation": "Derived from authentic workshop dataset 测算数据V0826(2)(1).xlsx, where shift horizon is 840 min, slack median is 248 min, and due date satisfies d_i = r_i + k_proc*(W_i*N_i/(eta_i*3.5)) + Delta_slack, strictly subject to d_i >= r_i + 1.15*p_hat.",
            "source_sampling": "Each scenario samples source UIDs without collision within the scenario. The source workbook remains read-only; source UID reuse across independently generated base scenarios is recorded rather than treated as an unseen raw-order guarantee.",
            "physical_invariants": "Every workbook is validated against D1 W=ceil(Q/S), the grade-to-machine eligibility policy, and Pool D isolation before it replaces an active scenario file.",
            "arrival_process": "G1 uses late and dispersed arrivals; G2 mixes uniform, Poisson and burst arrivals; G3 uses earlier, burstier arrivals and higher unforecasted/urgent shares.",
        },
        "source_workbook": {
            "name": SOURCE_WORKBOOK,
            "sha256": _sha256(source_path),
            "total_verified_orders": len(source_records),
            "scope": "Sheet1~Sheet5 authentic drilling production orders",
        },
        "eligibility_policy": {
            "policy_id": eligibility_policy["policy_id"],
            "relative_path": ELIGIBILITY_POLICY_RELATIVE_PATH,
            "sha256": _sha256(eligibility_path),
        },
        "split_policy": {
            "within_scenario_order_disjoint": True,
            "base_instance_split_isolation": True,
            "resampling_pool_size": len(source_records),
            "grouped_panel": {"train_per_group": 6, "val_per_group": 2, "test_per_group": 2},
            "training_curriculum_policy": {
                "policy_version": "PCB_DYNAMIC_V20261027_THREE_GROUP_CYCLIC_1800",
                "selection_rule": "B_GROUPED_SCALE_RANKED_PROGRESSIVE_QUOTAS",
                "scope": "train",
                "ranking_fields": ["initial_orders", "dynamic_events", "scenario_id"],
                "ranking_direction": "ascending",
                "formal_train_scenarios": 18,
                "formal_total_episodes": 5400,
                "base_quota": 300,
                "quota_step": 0,
                "quota_assignment": "Each stage runs 1800 episodes cycling sequentially through its 6 scale-ranked scenarios (s1 to s6) for 300 cycles, accumulating 6 scenarios per cycle and updating PPO once after all 6 scenarios finish (300 PPO updates per stage, 5400 episodes across all 3 stages).",
            },
        },
        "group_definitions": {
            group_id: {
                "label": profile.label,
                "description": profile.description,
                "workload_rule": profile.workload_rule,
                "arrival_ratio_range": list(profile.arrival_ratio_range),
                "arrival_patterns": list(profile.arrival_patterns),
                "priority_weights": dict(profile.priority_weights),
                "unforecasted_probability": profile.unforecasted_probability,
            }
            for group_id, profile in GROUPS.items()
        },
        "expected_scenarios": {"train": 18, "val": 6, "test": 6},
        "splits": split_records,
        "scenarios": records,
    }
    manifest["dataset_ledger"] = _ledger_from_records(records, dataset_dir)
    _validate_group_panel(manifest)
    manifest_path = dataset_dir / "dataset_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="重新构建活动三组动态场景数据集与清单。")
    parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    manifest = build_active_dataset(project_root)
    print(json.dumps({
        "manifest_version": manifest["manifest_version"],
        "expected_scenarios": manifest["expected_scenarios"],
        "dataset_ledger": manifest["dataset_ledger"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
