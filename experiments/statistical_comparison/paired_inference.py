"""Paired test-set inference: Wilcoxon/Holm and Friedman/Nemenyi ranks."""
from __future__ import annotations

import math
import statistics

import numpy as np
from scipy import stats

from .prepare_panel import MAIN, METHODS


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1.0, (len(order) - rank) * p_values[index]))
        adjusted[index] = maximum
    return adjusted


def compare_main(panel: list[dict], alpha: float = .05):
    if len(panel) != 36 or len({r["scenario_id"] for r in panel}) != 36:
        raise ValueError("Paired inference requires 36 unique test scenarios")
    if any(r.get(method) is None for r in panel for method in METHODS):
        raise ValueError("Paired inference requires a complete method-by-scenario panel")
    comparisons = []
    for opponent in METHODS:
        if opponent == MAIN:
            continue
        # Round only the subtraction, avoiding artificial Wilcoxon rank differences
        # from floating point subtraction of the same minute-valued Cmax.
        differences = np.round(np.asarray([r[MAIN] - r[opponent] for r in panel], dtype=float), 9)
        wins = int(np.sum(differences < 0))
        ties = int(np.sum(differences == 0))
        losses = int(np.sum(differences > 0))
        if ties == 36:
            statistic, p_raw = 0.0, 1.0
        else:
            test = stats.wilcoxon(differences, zero_method="wilcox",
                                  alternative="two-sided", method="auto")
            statistic, p_raw = float(test.statistic), float(test.pvalue)
        if not math.isfinite(p_raw):
            raise ValueError(f"Non-finite Wilcoxon p for {opponent}")
        comparisons.append({"opponent": opponent, "paired_scenarios": 36,
            "wins": wins, "ties": ties, "losses": losses,
            "win_tie_loss": f"{wins}/{ties}/{losses}",
            "wilcoxon_w": statistic, "p_raw": p_raw,
            "median_main_minus_opponent_min": float(np.median(differences))})
    adjusted = holm_adjust([r["p_raw"] for r in comparisons])
    for row, p in zip(comparisons, adjusted):
        row["p_holm"] = p
        row["significant_0_05"] = p < alpha
    return comparisons


def friedman_nemenyi(panel: list[dict], alpha: float = .05):
    if len(panel) != 36 or any(r.get(method) is None for r in panel for method in METHODS):
        raise ValueError("Friedman requires complete 36-by-method panel")
    matrix = np.asarray([[r[method] for method in METHODS] for r in panel], dtype=float)
    if matrix.shape != (36, len(METHODS)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid Friedman panel")
    ranks = np.asarray([stats.rankdata(row, method="average") for row in matrix])
    mean_ranks = ranks.mean(axis=0)
    if np.all(np.ptp(matrix, axis=1) == 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
        statistic, p_value = float(test.statistic), float(test.pvalue)
    if not math.isfinite(p_value):
        raise ValueError("Non-finite Friedman p")
    k, n = len(METHODS), 36
    q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2))
    cd = q * math.sqrt(k * (k + 1) / (6 * n))
    return ({"scenarios": n, "methods": k, "statistic": statistic,
             "p_value": p_value, "alpha": alpha,
             "nemenyi_q": q, "critical_difference": cd,
             "reject_omnibus": p_value < alpha},
            [{"method_id": method, "average_rank": float(rank)}
             for method, rank in zip(METHODS, mean_ranks)])


def stage_descriptives(panel: list[dict]):
    rows = []
    for stage in (1, 2, 3):
        stage_rows = [r for r in panel if r["stage"] == stage]
        for method in METHODS:
            values = [r[method] for r in stage_rows if r[method] is not None]
            rows.append({"stage": stage, "method_id": method, "n": len(values),
                "mean_cmax": statistics.mean(values) if values else None,
                "median_cmax": statistics.median(values) if values else None,
                "sd_cmax": statistics.stdev(values) if len(values) > 1 else None})
    return rows
