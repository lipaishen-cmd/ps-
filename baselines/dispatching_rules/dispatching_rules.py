"""Independent six-rule replays under the shared dynamic scheduling contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


THIS_DIR = Path(__file__).resolve().parent
COMPARISON_ROOT = THIS_DIR.parent
PROJECT_ROOT = COMPARISON_ROOT.parent

# Keep standalone CLI execution and IDE resolution aligned. The evaluation
# runners import comparison utilities and the environment/agent modules by their
# established top-level names; active training launchers do not import this
# offline comparison entry point.
IMPORT_ROOTS = (
    COMPARISON_ROOT,
    PROJECT_ROOT,
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "agent",
)
for path in IMPORT_ROOTS:
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_config import BaselineConfig, resolve_evaluation_files
from baseline_runtime import RULE_NAMES, run_rule_scenario


def solve_scenario_with_dispatching_rule(
    scenario_path: str,
    rule_name: str = "FIFO",
    *,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """Run one approved rule with the shared split, exact slot score, scope and repair policy."""
    return run_rule_scenario(
        scenario_path,
        rule_name,
        config or BaselineConfig(),
    ).as_dict()


def run_rule_suite(
    *,
    rules: Iterable[str],
    split: str,
    stage: Optional[int] = None,
    scenario_path: Optional[str] = None,
    export_report: bool = True,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """Replay the six approved rules and retain incomplete outcomes as incomplete."""
    active_config = config or BaselineConfig()
    files = [str(Path(scenario_path).resolve())] if scenario_path else resolve_evaluation_files(
        active_config,
        split,
        stage=stage,
    )
    canonical_rules = [str(rule).upper() for rule in rules]
    if any(rule not in RULE_NAMES for rule in canonical_rules):
        raise ValueError(f"规则只允许 {RULE_NAMES}，收到 {canonical_rules}。")

    results: Dict[str, List[Dict[str, Any]]] = {}
    summary: List[Dict[str, Any]] = []
    for rule in canonical_rules:
        records = [
            solve_scenario_with_dispatching_rule(path, rule, config=active_config)
            for path in files
        ]
        results[rule] = records
        complete_cmax = [
            float(record["final_cmax"])
            for record in records
            if record["completed"] and record["final_cmax"] is not None
        ]
        complete_records = [record for record in records if record["completed"]]
        total_tardiness = [float(record["total_tardiness"]) for record in complete_records]
        weighted_tardiness = [
            float(record["priority_weighted_tardiness"])
            for record in complete_records
        ]
        tardy_counts = [int(record["tardy_order_count"]) for record in complete_records]
        summary.append(
            {
                "method": rule,
                "scenario_count": len(records),
                "complete_count": len(complete_cmax),
                "complete_rate": len(complete_cmax) / max(1, len(records)),
                "mean_cmax_complete": (
                    sum(complete_cmax) / len(complete_cmax) if complete_cmax else None
                ),
                "mean_total_tardiness_complete": (
                    sum(total_tardiness) / len(total_tardiness) if total_tardiness else None
                ),
                "mean_priority_weighted_tardiness_complete": (
                    sum(weighted_tardiness) / len(weighted_tardiness) if weighted_tardiness else None
                ),
                "mean_tardy_order_count_complete": (
                    sum(tardy_counts) / len(tardy_counts) if tardy_counts else None
                ),
                "controlled_failures": sum(
                    int(record["controlled_failures"]) for record in records
                ),
            }
        )

    payload: Dict[str, Any] = {
        "schema": "six_rule_replay_v1",
        "split": split,
        "stage": stage,
        "scenario_files": files,
        "summary": summary,
        "details": results,
        "evidence_boundary": (
            "本报表只记录同一事件、范围、解码和校验合同下的离线回放；"
            "不构成算法收敛、统计显著性或现场有效性结论。"
        ),
    }
    if export_report:
        scope = f"{split}_stage{stage}" if stage is not None else split
        output = Path(active_config.artifacts_dir) / "rules" / scope / "rule_replay.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["report_path"] = str(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="六种规则调度独立回放入口")
    parser.add_argument("--rule", choices=("all", *RULE_NAMES), default="all")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    if args.stage is not None and args.split != "train" and args.scenario is None:
        parser.error("--stage 只用于 manifest 的 train 场景子集；验证/测试请省略。")

    rules = list(RULE_NAMES) if args.rule == "all" else [args.rule]
    report = run_rule_suite(
        rules=rules,
        split=args.split,
        stage=args.stage,
        scenario_path=args.scenario,
        export_report=not args.no_export,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if "report_path" in report:
        print(report["report_path"])


if __name__ == "__main__":
    main()
