"""Unified, result-only evaluator for the main method and six rule baselines.

The evaluator never runs a solver and never substitutes a failed schedule. It
reads frozen long-form CSV/JSON artifacts, normalises method identifiers, checks
panel coverage, and delegates statistical aggregation to baseline_statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baseline_statistics import aggregate_reports, parse_seed_spec
from configs.experiment_protocol import FORMAL_SEED_SPEC


METHOD_FAMILIES: Dict[str, tuple[str, ...]] = {
    "main": ("AD-HG-DAPPO",),
    "ablation": ("AD-HG-SAPPO",),
    "rules": (
        "S_BASE", "FIFO", "SPT", "LPT", "EDD", "ECT",
    ),
}

KNOWN_METHODS = frozenset(method for methods in METHOD_FAMILIES.values() for method in methods)
REQUIRED_COLUMNS = ("method_id", "seed", "split", "scenario", "completed", "final_cmax")


def canonical_method(value: Any) -> str:
    method = str(value).strip().upper()
    if method not in KNOWN_METHODS:
        raise ValueError(f"未知 method_id={value!r}；允许值为 {sorted(KNOWN_METHODS)}")
    return method


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"无法解析布尔值 {value!r}")


def _normalise_record(row: Mapping[str, Any], source: Path) -> Dict[str, Any]:
    missing = [name for name in REQUIRED_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"{source} 缺少字段 {missing}")
    completed = _as_bool(row["completed"])
    raw_cmax = row.get("final_cmax")
    final_cmax = None if raw_cmax in (None, "", "null", "None") else float(raw_cmax)
    if completed and final_cmax is None:
        raise ValueError(f"{source}: completed=true 但 final_cmax 为空")
    if not completed and final_cmax is not None:
        raise ValueError(f"{source}: 失败记录不得填充 final_cmax")
    return {
        "method": canonical_method(row["method_id"]),
        "seed": int(row["seed"]),
        "split": str(row["split"]).strip().lower(),
        "stage": None if row.get("stage") in (None, "", "full", "None") else int(row["stage"]),
        "scenario_path": str(row["scenario"]),
        "completed": completed,
        "final_cmax": final_cmax,
        "elapsed_seconds": float(row.get("elapsed_seconds") or 0.0),
        "controlled_failures": int(row.get("controlled_failures") or (0 if completed else 1)),
        "checkpoint_type": str(row.get("checkpoint_type") or "not_applicable"),
        "source_artifact": str(source.resolve()),
    }


def _json_rows(payload: Any, source: Path) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, list):
        yield from payload
        return
    if not isinstance(payload, dict):
        raise ValueError(f"{source} 顶层必须是对象或记录数组")
    if isinstance(payload.get("records"), list):
        yield from payload["records"]
        return
    if isinstance(payload.get("detailed_records"), list):
        yield from payload["detailed_records"]
        return
    raise ValueError(f"{source} 未找到 records 或 detailed_records 数组")


def load_records(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = list(_json_rows(payload, path))
        else:
            raise ValueError(f"只接受 CSV/JSON，收到 {path}")
        records.extend(_normalise_record(row, path) for row in rows)
    if not records:
        raise ValueError("输入结果中没有记录")
    return records


def evaluate(
    records: Sequence[Dict[str, Any]],
    *,
    reference_method: str,
    required_seeds: Sequence[int],
) -> Dict[str, Any]:
    splits = {row["split"] for row in records}
    stages = {row["stage"] for row in records}
    if len(splits) != 1 or len(stages) != 1:
        raise ValueError(f"一次统一评估只能包含同一 split/stage；收到 split={splits}, stage={stages}")

    grouped: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[row["seed"]][row["method"]].append(row)

    reports: List[Dict[str, Any]] = []
    for seed, methods in sorted(grouped.items()):
        scenario_files = sorted(
            {Path(row["scenario_path"]).name for rows in methods.values() for row in rows}
        )
        checkpoint_types = {
            method: next((row["checkpoint_type"] for row in rows), "not_applicable")
            for method, rows in methods.items()
        }
        reports.append({
            "seed": seed,
            "scenario_files": scenario_files,
            "details": {
                method: sorted(rows, key=lambda row: Path(row["scenario_path"]).name)
                for method, rows in methods.items()
            },
            "checkpoint_types": checkpoint_types,
        })

    observed_methods = sorted({row["method"] for row in records})
    aggregate = aggregate_reports(
        reports,
        reference_method=canonical_method(reference_method),
        evaluation_split=next(iter(splits)),
        evaluation_stage=next(iter(stages)),
        required_seeds=required_seeds,
    )
    aggregate.update({
        "schema": "pcb_dynamic_six_rule_evaluator_v2",
        "method_families": METHOD_FAMILIES,
        "observed_methods": observed_methods,
        "missing_approved_methods": sorted(KNOWN_METHODS.difference(observed_methods)),
        "result_policy": {
            "shared_physical_decoder_required": True,
            "failed_cmax_imputation": False,
            "coverage_reported_before_conditional_cmax": True,
            "rule_count": len(METHOD_FAMILIES["rules"]),
        },
    })
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="PCB 动态调度全方法统一结果评估器")
    parser.add_argument("--input", action="append", required=True, help="长表 CSV/JSON；可重复传入")
    parser.add_argument("--output", required=True, help="统一 JSON 报告输出路径")
    parser.add_argument("--reference", default="AD-HG-DAPPO")
    parser.add_argument("--required-seeds", default=FORMAL_SEED_SPEC)
    args = parser.parse_args()

    records = load_records([Path(value) for value in args.input])
    report = evaluate(
        records,
        reference_method=args.reference,
        required_seeds=parse_seed_spec(args.required_seeds),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] 统一评估报告: {output.resolve()}")
    print(f"[COVERAGE] 已读取 {len(report['observed_methods'])}/{len(KNOWN_METHODS)} 个批准方法。")


if __name__ == "__main__":
    main()
