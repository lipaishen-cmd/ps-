"""Multi-seed completion-aware paired baseline statistics.

Features:
    - Primary reporting of completion counts and completion rates.
    - Conditional Cmax statistics only over fully completed instances.
    - Paired Wilcoxon signed-rank test on independent seed averages over jointly complete scenarios.
    - Holm-Bonferroni correction for family-wise error rate control.
    - Matched-pairs rank-biserial effect size calculation.
    - Clear distinction between stochastic learned models and deterministic dispatching rules.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from configs.experiment_protocol import FORMAL_SEEDS, FORMAL_SEED_COUNT


def parse_seed_spec(specification: str) -> List[int]:
    """Parse seed ranges without silently accepting duplicates or zero."""
    text = str(specification).strip()
    if not text:
        raise ValueError("种子格式不能为空。")
    seeds: List[int] = []
    seen = set()
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise ValueError(f"非法种子区间格式: {part}")
            start, end = int(bounds[0].strip()), int(bounds[1].strip())
            if start > end or start <= 0:
                raise ValueError(f"非法种子区间范围: {part}")
            range_seeds = list(range(start, end + 1))
        else:
            val = int(part)
            if val <= 0:
                raise ValueError(f"种子必须为正整数: {val}")
            range_seeds = [val]
        for s in range_seeds:
            if s in seen:
                raise ValueError(f"种子重复: {s}")
            seen.add(s)
            seeds.append(s)
    if not seeds:
        raise ValueError("至少需要一个种子。")
    return seeds


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down adjustment to a sequence of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = p_val * (m - rank)
        adj = min(1.0, max(running_max, adj))
        running_max = adj
        adjusted[orig_idx] = float(adj)
    return adjusted


def _rank_biserial(differences: Sequence[float]) -> float:
    """Compute matched-pairs rank-biserial correlation."""
    diffs = np.asarray(differences, dtype=np.float64)
    nonzero = diffs[diffs != 0.0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    w_pos = np.sum(ranks[nonzero > 0.0])
    w_neg = np.sum(ranks[nonzero < 0.0])
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total > 0 else 0.0


def _attempt_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for report in reports:
        seed = int(report.get("seed", 0))
        details = report.get("details", {})
        for method, records in details.items():
            for rec in records:
                scenario = Path(str(rec.get("scenario_path", rec.get("scenario", "")))).name
                rows.append({
                    "seed": seed,
                    "method": str(method),
                    "scenario": scenario,
                    "completed": bool(rec.get("completed", False)),
                    "final_cmax": rec.get("final_cmax"),
                    "elapsed_seconds": float(rec.get("elapsed_seconds", 0.0)),
                    "controlled_failures": int(rec.get("controlled_failures", 0)),
                })
    return rows


def aggregate_reports(
    reports: List[Dict[str, Any]],
    reference_method: str = "S_BASE",
    min_completion_rate: float = 1.0,
    min_completed_scenarios: Optional[int] = None,
    evaluation_split: Optional[str] = None,
    evaluation_stage: Any = "unspecified",
    required_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Aggregate attempts without imputing Cmax for controlled failures."""
    if not reports:
        raise ValueError("报告列表不能为空。")

    all_methods = sorted({m for r in reports for m in r.get("details", {}).keys()})
    if reference_method not in all_methods and "S_BASE" in all_methods:
        reference_method = "S_BASE"

    seed_list = sorted({int(r.get("seed", 0)) for r in reports})
    total_seeds = len(seed_list)
    deterministic_rules = {
        "S_BASE", "FIFO", "SPT", "LPT", "EDD", "ECT",
    }

    # Determine expected scenarios per seed
    def scenario_count(report: Dict[str, Any]) -> int:
        scenarios = report.get("scenario_files")
        if scenarios:
            return len(scenarios)
        details = report.get("details", {})
        if details:
            first_key = next(iter(details))
            return len(details[first_key])
        return 0

    expected_scenarios = max((scenario_count(r) for r in reports), default=0)
    expected_attempts_per_method = expected_scenarios * total_seeds

    def scenario_panel(report: Dict[str, Any]) -> Tuple[str, ...]:
        scenarios = report.get("scenario_files")
        if scenarios:
            return tuple(Path(str(path)).name for path in scenarios)
        details = report.get("details", {})
        if not details:
            return ()
        first_key = next(iter(details))
        return tuple(
            Path(str(rec.get("scenario_path", rec.get("scenario", "")))).name
            for rec in details[first_key]
        )

    panels = [scenario_panel(report) for report in reports]
    panel_consistent = bool(panels) and bool(panels[0]) and all(panel == panels[0] for panel in panels)

    method_summary = []
    for method in all_methods:
        attempts = [
            rec
            for r in reports
            for rec in r.get("details", {}).get(method, [])
        ]
        attempt_count = len(attempts)
        missing_checkpoint_seeds = sum(
            1 for r in reports if method in r.get("skipped", {})
        )
        complete_records = [rec for rec in attempts if rec.get("completed") and rec.get("final_cmax") is not None]
        complete_count = len(complete_records)
        complete_cmax = [float(rec["final_cmax"]) for rec in complete_records]
        failures = sum(int(rec.get("controlled_failures", 0)) for rec in attempts)
        elapsed = [float(rec.get("elapsed_seconds", 0.0)) for rec in attempts]

        coverage = float(attempt_count / expected_attempts_per_method) if expected_attempts_per_method > 0 else 0.0
        complete_rate = float(complete_count / attempt_count) if attempt_count > 0 else 0.0

        q25 = float(np.percentile(complete_cmax, 25)) if complete_cmax else None
        q75 = float(np.percentile(complete_cmax, 75)) if complete_cmax else None
        iqr = float(q75 - q25) if (q25 is not None and q75 is not None) else None

        ci95 = None
        if len(complete_cmax) >= 2:
            mean_c = float(np.mean(complete_cmax))
            sem_c = float(stats.sem(complete_cmax))
            if sem_c > 0:
                ci_bounds = stats.t.interval(0.95, df=len(complete_cmax) - 1, loc=mean_c, scale=sem_c)
                ci95 = [float(ci_bounds[0]), float(ci_bounds[1])]
            else:
                ci95 = [mean_c, mean_c]

        complete_panel_seed_means: List[float] = []
        for report in reports:
            seed_records = list(report.get("details", {}).get(method, []))
            seed_complete = [
                float(rec["final_cmax"])
                for rec in seed_records
                if rec.get("completed") and rec.get("final_cmax") is not None
            ]
            if expected_scenarios > 0 and len(seed_complete) == expected_scenarios:
                complete_panel_seed_means.append(float(np.mean(seed_complete)))

        seed_ci95 = None
        if method not in deterministic_rules and len(complete_panel_seed_means) >= 2:
            seed_mean = float(np.mean(complete_panel_seed_means))
            seed_sem = float(stats.sem(complete_panel_seed_means))
            if seed_sem > 0:
                bounds = stats.t.interval(
                    0.95,
                    df=len(complete_panel_seed_means) - 1,
                    loc=seed_mean,
                    scale=seed_sem,
                )
                seed_ci95 = [float(bounds[0]), float(bounds[1])]
            else:
                seed_ci95 = [seed_mean, seed_mean]

        method_summary.append({
            "method": method,
            "attempt_count": attempt_count,
            "expected_attempt_count": expected_attempts_per_method,
            "evaluation_coverage": coverage,
            "missing_checkpoint_seed_count": missing_checkpoint_seeds,
            "complete_count": complete_count,
            "complete_rate": complete_rate,
            "mean_cmax_complete": float(np.mean(complete_cmax)) if complete_cmax else None,
            "std_cmax_complete": float(np.std(complete_cmax, ddof=1)) if len(complete_cmax) > 1 else 0.0,
            "median_cmax_complete": float(np.median(complete_cmax)) if complete_cmax else None,
            "q25_cmax_complete": q25,
            "q75_cmax_complete": q75,
            "iqr_cmax_complete": iqr,
            "ci95_cmax_complete": ci95,
            "complete_panel_seed_count": len(complete_panel_seed_means),
            "mean_seed_cmax_complete_panel": (
                float(np.mean(complete_panel_seed_means)) if complete_panel_seed_means else None
            ),
            "ci95_seed_mean_complete_panel": seed_ci95,
            "ci95_seed_mean_unit": (
                "independent_seed_mean_over_full_test_panel"
                if method not in deterministic_rules
                else "not_applicable_deterministic_method"
            ),
            "controlled_failures": failures,
            "mean_response_seconds": float(np.mean(elapsed)) if elapsed else 0.0,
        })

    # Paired test against reference_method
    paired_tests = []
    unadjusted_p_values = []
    p_indices = []

    for method in all_methods:
        if method == reference_method:
            continue

        # Collect paired scenarios across seeds
        paired_seed_diffs: List[float] = []
        scenario_pair_deltas: List[float] = []
        scenarios_per_seed: List[int] = []
        total_paired_complete = 0
        excluded_unpaired = 0

        for r in reports:
            seed = int(r.get("seed", 0))
            ref_recs = {
                Path(str(rec.get("scenario_path", rec.get("scenario", "")))).name: rec
                for rec in r.get("details", {}).get(reference_method, [])
            }
            cand_recs = {
                Path(str(rec.get("scenario_path", rec.get("scenario", "")))).name: rec
                for rec in r.get("details", {}).get(method, [])
            }
            common_scenarios = sorted(set(ref_recs.keys()) & set(cand_recs.keys()))
            jointly_complete_diffs = []

            for sc in common_scenarios:
                ref_item = ref_recs[sc]
                cand_item = cand_recs[sc]
                if (
                    ref_item.get("completed")
                    and cand_item.get("completed")
                    and ref_item.get("final_cmax") is not None
                    and cand_item.get("final_cmax") is not None
                ):
                    delta = float(cand_item["final_cmax"]) - float(ref_item["final_cmax"])
                    jointly_complete_diffs.append(delta)
                    scenario_pair_deltas.append(delta)
                    total_paired_complete += 1
                else:
                    excluded_unpaired += 1

            if jointly_complete_diffs:
                paired_seed_diffs.append(float(np.mean(jointly_complete_diffs)))
                scenarios_per_seed.append(len(jointly_complete_diffs))

        paired_seed_count = len(paired_seed_diffs)
        is_learned = method not in deterministic_rules
        can_infer = is_learned and paired_seed_count >= 2

        # Wilcoxon test
        if can_infer and len(set(paired_seed_diffs)) > 1:
            try:
                stat_res = stats.wilcoxon(
                    paired_seed_diffs,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                p_val = float(stat_res.pvalue)
            except Exception:
                p_val = 1.0
        else:
            p_val = 1.0

        rb_eff = _rank_biserial(paired_seed_diffs)

        mean_diff = float(np.mean(paired_seed_diffs)) if paired_seed_diffs else None
        median_diff = float(np.median(paired_seed_diffs)) if paired_seed_diffs else None
        scenario_mean_diff = float(np.mean(scenario_pair_deltas)) if scenario_pair_deltas else None

        delta_iqr = None
        delta_ci95 = None
        if len(paired_seed_diffs) >= 2:
            delta_q25 = float(np.percentile(paired_seed_diffs, 25))
            delta_q75 = float(np.percentile(paired_seed_diffs, 75))
            delta_iqr = float(delta_q75 - delta_q25)
            sem_d = float(stats.sem(paired_seed_diffs))
            if sem_d > 0:
                ci_d = stats.t.interval(0.95, df=len(paired_seed_diffs) - 1, loc=mean_diff, scale=sem_d)
                delta_ci95 = [float(ci_d[0]), float(ci_d[1])]
            else:
                delta_ci95 = [mean_diff, mean_diff]

        test_entry = {
            "method": method,
            "reference_method": reference_method,
            "paired_complete_count": total_paired_complete,
            "scenario_pair_count": len(scenario_pair_deltas),
            "paired_seed_count": paired_seed_count,
            "effective_paired_seeds": paired_seed_count,
            "shared_attempt_count": total_paired_complete + excluded_unpaired,
            "excluded_unpaired_or_incomplete": excluded_unpaired,
            "min_scenarios_per_paired_seed": min(scenarios_per_seed, default=0),
            "max_scenarios_per_paired_seed": max(scenarios_per_seed, default=0),
            "mean_paired_delta_cmax": mean_diff,
            "median_paired_delta_cmax": median_diff,
            "iqr_paired_delta_cmax": delta_iqr,
            "ci95_paired_delta_cmax": delta_ci95,
            "mean_scenario_delta_cmax": scenario_mean_diff,
            "wilcoxon_two_sided_p": p_val if can_infer else None,
            "holm_adjusted_p": p_val if can_infer else None,
            "paired_rank_biserial": rb_eff,
            "effect_direction": "method_minus_reference",
            "eligible_for_seed_level_inference": can_infer,
            "inference_unit": (
                "independent_seed_mean_over_jointly_complete_scenarios"
                if is_learned
                else "descriptive_fixed_scenario_pairs"
            ),
            "minimum_independent_seeds_satisfied": paired_seed_count >= FORMAL_SEED_COUNT,
        }
        if can_infer:
            unadjusted_p_values.append(p_val)
            p_indices.append(len(paired_tests))
        paired_tests.append(test_entry)

    # Adjust p-values with Holm-Bonferroni
    if unadjusted_p_values:
        adjusted = _holm_adjust(unadjusted_p_values)
        for idx, adj_p in zip(p_indices, adjusted):
            paired_tests[idx]["holm_adjusted_p"] = adj_p

    coverage_complete = all(row["evaluation_coverage"] == 1.0 for row in method_summary)
    learned_methods = [m for m in all_methods if m not in deterministic_rules]
    qualified_checkpoints_only = bool(learned_methods) and all(
        report.get("checkpoint_types", {}).get(method) == "best"
        for report in reports
        for method in learned_methods
    )
    learned_paired_counts = [pt["paired_seed_count"] for pt in paired_tests if pt["method"] in learned_methods]
    min_paired_seeds = min(learned_paired_counts) if learned_paired_counts else 0

    # Strict feasibility and panel completeness checks
    learned_complete_rates = [
        row["complete_rate"] for row in method_summary if row["method"] in learned_methods
    ]
    min_learned_complete_rate = min(learned_complete_rates) if learned_complete_rates else 0.0

    learned_min_scenarios = [
        pt["min_scenarios_per_paired_seed"] for pt in paired_tests if pt["method"] in learned_methods
    ]
    min_scenarios_achieved = min(learned_min_scenarios) if learned_min_scenarios else 0
    req_comp_rate = float(min_completion_rate)
    if min_completed_scenarios is not None:
        required_min_scenarios_per_seed = int(min_completed_scenarios)
    else:
        required_min_scenarios_per_seed = max(1, math.ceil(expected_scenarios * req_comp_rate)) if expected_scenarios > 0 else 1

    formal_context_satisfied = (
        str(evaluation_split).lower() == "test" and evaluation_stage is None
    )
    required_seed_list = sorted(
        {int(seed) for seed in (required_seeds if required_seeds is not None else FORMAL_SEEDS)}
    )
    seed_protocol_satisfied = seed_list == required_seed_list
    seeds_satisfied = seed_protocol_satisfied and min_paired_seeds >= len(required_seed_list)
    completion_rate_satisfied = min_learned_complete_rate >= req_comp_rate
    scenarios_per_seed_satisfied = min_scenarios_achieved >= required_min_scenarios_per_seed

    formal_ready = (
        coverage_complete
        and panel_consistent
        and formal_context_satisfied
        and qualified_checkpoints_only
        and seeds_satisfied
        and completion_rate_satisfied
        and scenarios_per_seed_satisfied
        and len(learned_methods) > 0
    )
    status_label = "FORMAL_EVALUATION_READY" if formal_ready else "RESEARCH_PROTOTYPE / PRE-FORMAL-COMPARISON"

    formal_readiness_blockers = []
    if not formal_context_satisfied:
        formal_readiness_blockers.append("Formal inference requires split=test and stage=full")
    if not seed_protocol_satisfied:
        formal_readiness_blockers.append(
            f"Formal seed panel mismatch: observed={seed_list}, required={required_seed_list}"
        )
    if not coverage_complete:
        formal_readiness_blockers.append("Missing checkpoints across evaluation seeds")
    if not panel_consistent:
        formal_readiness_blockers.append("Evaluation scenario panel or order differs across seeds")
    if not qualified_checkpoints_only:
        formal_readiness_blockers.append(
            "Formal inference requires checkpoint_best.pt for every learned method and seed"
        )
    if seed_protocol_satisfied and min_paired_seeds < len(required_seed_list):
        formal_readiness_blockers.append(
            f"Insufficient paired learned-method seeds: {min_paired_seeds} < {len(required_seed_list)}"
        )
    if min_learned_complete_rate < req_comp_rate:
        formal_readiness_blockers.append(f"Learned method completion rate below {req_comp_rate*100:.1f}% threshold ({min_learned_complete_rate*100:.1f}%)")
    if not scenarios_per_seed_satisfied:
        formal_readiness_blockers.append(f"Per-seed completed scenario count below panel threshold ({min_scenarios_achieved} < {required_min_scenarios_per_seed})")

    return {
        "schema": "baseline_multi_seed_statistics_v3",
        "status_label": status_label,
        "seed_count": total_seeds,
        "minimum_seed_requirement": FORMAL_SEED_COUNT,
        "required_seed_panel": required_seed_list,
        "observed_seed_panel": seed_list,
        "formal_seed_protocol_satisfied": seed_protocol_satisfied,
        "formal_context_satisfied": formal_context_satisfied,
        "scenario_panel_consistent": panel_consistent,
        "qualified_checkpoints_only": qualified_checkpoints_only,
        "expected_scenarios_per_seed": expected_scenarios,
        "configured_min_completion_rate": req_comp_rate,
        "configured_min_completed_scenarios": required_min_scenarios_per_seed,
        "checkpoint_coverage_complete": coverage_complete,
        "minimum_independent_seeds_satisfied": seeds_satisfied,
        "min_paired_seeds_for_learned_methods": min_paired_seeds,
        "min_learned_complete_rate": min_learned_complete_rate,
        "min_scenarios_per_paired_seed": min_scenarios_achieved,
        "formal_ready_for_paired_inference": formal_ready,
        "formal_readiness_blockers": formal_readiness_blockers,
        "reference_method": reference_method,
        "method_summary": method_summary,
        "paired_tests": paired_tests,
        "interpretation": {
            "completion_primary": "All attempts report completion rate before conditional Cmax.",
            "cmax_is_conditional_on_complete_solution": True,
            "failed_runs_receive_no_imputed_cmax": True,
            "missing_checkpoints_are_coverage_gaps_not_scheduling_failures": True,
            "paired_test": "two-sided Wilcoxon signed-rank",
            "paired_test_unit": "one mean paired Cmax difference per independent seed",
            "confidence_interval_unit": "independent seed mean over the complete fixed panel",
            "within_seed_aggregation": "mean over jointly completed fixed scenarios",
            "deterministic_rules": "descriptive only; seed repetitions are not inferential units",
            "multiple_comparison": "Holm",
            "effect_size": "matched-pairs rank-biserial correlation",
        },
    }
