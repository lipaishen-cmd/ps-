"""Generate all manuscript tables and inline numbers from a single authoritative CSV.

Enforces L1 (split separation), L2 (exact data-consistent counts), L3 (single
source of truth; removes undefined cross-stage column), L5 (steady convergence
indicators when val_curve.csv is available), and L10 (deployable best single
rule selected strictly on train+val as primary reference, with Min Rule labeled
as post-hoc envelope).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
for directory in (THIS_DIR, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from audit_anomalies import load_l3_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "null"):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def _fmt(val: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    if val is None:
        return "N/A"
    if signed:
        return f"{val:+.{decimals}f}"
    return f"{val:.{decimals}f}"


def win_tie_loss(
    proposed_vals: Sequence[Optional[float]],
    ref_vals: Sequence[Optional[float]],
    tie_tol_min: float = 0.1,
) -> Tuple[int, int, int]:
    """Count (wins, ties, losses) where |proposed - ref| <= tie_tol_min is a tie."""
    wins = ties = losses = 0
    for p, r in zip(proposed_vals, ref_vals):
        if p is None or r is None:
            continue
        diff = float(p) - float(r)
        if abs(diff) <= tie_tol_min:
            ties += 1
        elif diff < 0.0:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def wilcoxon_p(
    proposed_vals: Sequence[Optional[float]],
    ref_vals: Sequence[Optional[float]],
) -> Optional[float]:
    pairs = [
        float(p) - float(r)
        for p, r in zip(proposed_vals, ref_vals)
        if p is not None and r is not None
    ]
    if not pairs:
        return None
    diffs = np.round(np.asarray(pairs, dtype=float), 9)
    if np.all(diffs == 0.0):
        return 1.0
    res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(res.pvalue)


def holm_adjust(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    valid_indices = [i for i, p in enumerate(p_values) if p is not None]
    adjusted: List[Optional[float]] = [None] * len(p_values)
    if not valid_indices:
        return adjusted
    ordered = sorted(valid_indices, key=lambda i: float(p_values[i]))  # type: ignore[arg-type]
    running_max = 0.0
    m = len(ordered)
    for rank, idx in enumerate(ordered):
        raw_p = float(p_values[idx])  # type: ignore[arg-type]
        running_max = max(running_max, min(1.0, (m - rank) * raw_p))
        adjusted[idx] = running_max
    return adjusted


def _compute_val_convergence(val_curve_path: Optional[Path]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Compute steady convergence indicators per (stage, method) from val_curve.csv."""
    if val_curve_path is None or not val_curve_path.is_file():
        return {}
    by_run: Dict[Tuple[int, str, int], List[Tuple[int, float]]] = defaultdict(list)
    with val_curve_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            val = _to_float(row.get("val_mean_cmax"))
            feas = _to_float(row.get("val_feasible")) or 0.0
            if val is not None and feas >= 1.0 - 1e-9:
                by_run[(int(row["stage"]), str(row["method"]), int(row["seed"]))].append(
                    (int(row["episode"]), val)
                )

    per_group: Dict[Tuple[int, str], List[Dict[str, Optional[float]]]] = defaultdict(list)
    for (stage, method, _seed), pts in sorted(by_run.items()):
        pts.sort(key=lambda t: t[0])
        best_ep, best_val = min(pts, key=lambda t: (t[1], t[0]))
        final_ep, final_val = pts[-1]
        thresh = final_val * 1.01
        ep_1pct = None
        for idx in range(len(pts)):
            if all(v <= thresh for _, v in pts[idx:]):
                ep_1pct = float(pts[idx][0])
                break
        ep_span = max(1.0, float(pts[-1][0] - pts[0][0]))
        min_c = min(v for _, v in pts)
        max_c = max(v for _, v in pts)
        c_span = max_c - min_c
        if len(pts) == 1 or c_span <= 1e-12:
            auc = 0.0
        else:
            auc = sum(
                0.5 * (((v1 - min_c) / c_span) + ((v2 - min_c) / c_span)) * ((e2 - e1) / ep_span)
                for (e1, v1), (e2, v2) in zip(pts[:-1], pts[1:])
            )
        per_group[(stage, method)].append({
            "best_val_cmax": best_val,
            "best_val_episode": float(best_ep),
            "final_val_cmax": final_val,
            "final_val_episode": float(final_ep),
            "episodes_to_within_1pct_final": ep_1pct,
            "normalized_val_auc": auc,
        })

    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for key, runs in per_group.items():
        summary: Dict[str, Any] = {"seed_count": len(runs)}
        for k in (
            "best_val_cmax",
            "best_val_episode",
            "final_val_cmax",
            "episodes_to_within_1pct_final",
            "normalized_val_auc",
        ):
            vals = [float(r[k]) for r in runs if r[k] is not None]
            summary[f"{k}_mean"] = statistics.mean(vals) if vals else None
            summary[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else None
        out[key] = summary
    return out


def _normalize_wide_rows(csv_path: Path, rules: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    for r in raw_rows:
        stage_str = str(r["stage"])
        stage_num = int(r["stage_num"]) if r.get("stage_num") else int(stage_str.replace("Stage", "").strip())
        split_raw = str(r["split"]).strip()
        # Distinguish main test (T0 / T-ID / test) from factor_test (T-F)
        is_main_test = split_raw in ("T0", "T-ID", "test")
        is_factor_test = split_raw in ("T-F", "factor_test")
        split_group = "test" if is_main_test else ("factor_test" if is_factor_test else split_raw)

        prop = _to_float(r.get("proposed_cmax"))
        abla = _to_float(r.get("ablation_cmax"))
        rule_vals = {rule: _to_float(r.get(rule)) for rule in rules}
        valid_rules = [(rule_vals[rule], idx, rule) for idx, rule in enumerate(rules) if rule_vals[rule] is not None]
        if valid_rules:
            min_cmax, _, min_name = min(valid_rules, key=lambda t: (t[0], t[1]))
        else:
            min_cmax = _to_float(r.get("min_rule_cmax"))
            min_name = str(r.get("min_rule_name", ""))

        rows.append({
            "scenario_id": str(r["scenario_id"]),
            "stage": f"Stage {stage_num}",
            "stage_num": stage_num,
            "split_raw": split_raw,
            "split_group": split_group,
            "origin": str(r.get("origin", "")),
            "initial_orders": int(r["initial_orders"]),
            "dynamic_events": int(r["dynamic_events"]),
            "arrival": str(r.get("arrival", "")),
            "unforecast_ratio": _to_float(r.get("unforecast_ratio")),
            "id_or_ood": str(r.get("id_or_ood", "ID")) or "ID",
            "seed_count": int(r.get("seed_count") or 1),
            "proposed_cmax": prop,
            "ablation_cmax": abla,
            "min_rule_name": min_name,
            "min_rule_cmax": min_cmax,
            **rule_vals,
        })

    # Recompute deployable best rule strictly on train + val per stage (L10)
    deployable_by_stage: Dict[int, str] = {}
    for stage_num in sorted({row["stage_num"] for row in rows}):
        tv_rows = [row for row in rows if row["stage_num"] == stage_num and row["split_group"] in ("train", "val")]
        candidates: List[Tuple[float, int, str]] = []
        for idx, rule in enumerate(rules):
            vals = [row[rule] for row in tv_rows]
            if vals and all(v is not None for v in vals):
                candidates.append((statistics.mean(float(v) for v in vals), idx, rule))
        if not candidates:
            raise RuntimeError(f"Stage {stage_num} lacks complete rule results on train+val.")
        candidates.sort(key=lambda t: (t[0], t[1]))
        deployable_by_stage[stage_num] = candidates[0][2]

    for row in rows:
        dep_name = deployable_by_stage[row["stage_num"]]
        dep_cmax = row[dep_name]
        prop = row["proposed_cmax"]
        min_c = row["min_rule_cmax"]
        abla = row["ablation_cmax"]
        s_base = row["S_BASE"]
        row["deployable_rule_name"] = dep_name
        row["deployable_rule_cmax"] = dep_cmax
        row["imp_vs_deployable_pct"] = ((dep_cmax - prop) / dep_cmax * 100.0) if (prop is not None and dep_cmax) else None
        row["imp_vs_min_rule_pct"] = ((min_c - prop) / min_c * 100.0) if (prop is not None and min_c) else None
        row["imp_vs_s_base_pct"] = ((s_base - prop) / s_base * 100.0) if (prop is not None and s_base) else None
        row["imp_vs_ablation_pct"] = ((abla - prop) / abla * 100.0) if (prop is not None and abla) else None
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _to_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def _mean_safe(vals: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None]
    return statistics.mean(clean) if clean else None


def _std_safe(vals: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None]
    if len(clean) > 1:
        return statistics.stdev(clean)
    return 0.0 if len(clean) == 1 else None


def _ratio_imp(prop_mean: Optional[float], ref_mean: Optional[float]) -> Optional[float]:
    if prop_mean is None or ref_mean is None or ref_mean == 0.0:
        return None
    return (ref_mean - prop_mean) / ref_mean * 100.0


def generate_all_tables_from_csv(
    csv_path: Optional[str] = None,
    config_path: Optional[str] = None,
    val_curve_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = 1,
) -> Dict[str, Any]:
    config, config_sha = load_l3_config(config_path)
    rules: List[str] = list(config["rules"])
    tie_tol = float(config["tie_tolerance_min"])
    dec = int(config["display_decimals"])

    auth_csv = Path(csv_path) if csv_path else (ROOT / config["authoritative_csv_path"].format(seed=seed))
    if not auth_csv.is_file():
        raise FileNotFoundError(f"Authoritative CSV not found: {auth_csv}")
    csv_sha = _sha256(auth_csv)

    out_dir = Path(output_dir) if output_dir else (ROOT / config["output_root"] / f"seed_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    vc_path = Path(val_curve_path) if val_curve_path else (out_dir / "val_curve.csv")
    val_conv = _compute_val_convergence(vc_path if vc_path.is_file() else None)

    rows = _normalize_wide_rows(auth_csv, rules)
    main_rows = [r for r in rows if r["split_group"] in ("train", "val", "test")]
    factor_rows = [r for r in rows if r["split_group"] == "factor_test"]
    stages = sorted({r["stage_num"] for r in main_rows})
    deployable_by_stage = {
        s: next(r["deployable_rule_name"] for r in main_rows if r["stage_num"] == s)
        for s in stages
    }

    # Table 23a: Full per-scenario comparison + separated train/val/test subtotals (L1+L10)
    t23a_rows: List[Dict[str, Any]] = []
    for stage_num in stages:
        for split_key in ("train", "val", "test"):
            sub = [r for r in main_rows if r["stage_num"] == stage_num and r["split_group"] == split_key]
            if not sub:
                continue
            for r in sub:
                dep_c = r["deployable_rule_cmax"]
                min_c = r["min_rule_cmax"]
                prop = r["proposed_cmax"]
                t23a_rows.append({
                    "stage": r["stage"],
                    "scenario_id": r["scenario_id"],
                    "split": r["split_raw"],
                    "orders": r["initial_orders"],
                    "events": r["dynamic_events"],
                    "AD-HG-DAPPO": round(prop, dec) if prop is not None else None,
                    "AD-HG-SAPPO": round(r["ablation_cmax"], dec) if r["ablation_cmax"] is not None else None,
                    "deployable_rule": f"{_fmt(dep_c, dec)} ({r['deployable_rule_name']})",
                    "imp_vs_deployable_pct": round(r["imp_vs_deployable_pct"], 2) if r["imp_vs_deployable_pct"] is not None else None,
                    "min_rule_posthoc": f"{_fmt(min_c, dec)} ({r['min_rule_name']})",
                    "imp_vs_min_rule_pct": round(r["imp_vs_min_rule_pct"], 2) if r["imp_vs_min_rule_pct"] is not None else None,
                    **{rule: round(r[rule], dec) if r[rule] is not None else None for rule in rules},
                })
            # Split subtotal row (never mixed across splits)
            p_mean = _mean_safe([r["proposed_cmax"] for r in sub])
            a_mean = _mean_safe([r["ablation_cmax"] for r in sub])
            dep_mean = _mean_safe([r["deployable_rule_cmax"] for r in sub])
            min_mean = _mean_safe([r["min_rule_cmax"] for r in sub])
            wtl_dep = win_tie_loss([r["proposed_cmax"] for r in sub], [r["deployable_rule_cmax"] for r in sub], tie_tol)
            wtl_min = win_tie_loss([r["proposed_cmax"] for r in sub], [r["min_rule_cmax"] for r in sub], tie_tol)
            split_label = {"train": "训练集均值", "val": "验证集均值(选模)", "test": "测试集均值(泛化)"}[split_key]
            t23a_rows.append({
                "stage": f"Stage {stage_num}",
                "scenario_id": f"{split_label} (n={len(sub)})",
                "split": split_key,
                "orders": "-",
                "events": "-",
                "AD-HG-DAPPO": round(p_mean, dec) if p_mean is not None else None,
                "AD-HG-SAPPO": round(a_mean, dec) if a_mean is not None else None,
                "deployable_rule": f"{_fmt(dep_mean, dec)} ({deployable_by_stage[stage_num]}, W/T/L={wtl_dep[0]}/{wtl_dep[1]}/{wtl_dep[2]})",
                "imp_vs_deployable_pct": round(_ratio_imp(p_mean, dep_mean), 2) if _ratio_imp(p_mean, dep_mean) is not None else None,
                "min_rule_posthoc": f"{_fmt(min_mean, dec)} (事后包络, W/T/L={wtl_min[0]}/{wtl_min[1]}/{wtl_min[2]})",
                "imp_vs_min_rule_pct": round(_ratio_imp(p_mean, min_mean), 2) if _ratio_imp(p_mean, min_mean) is not None else None,
                **{rule: round(_mean_safe([r[rule] for r in sub]), dec) for rule in rules},
            })

    _write_csv(out_dir / "table_23a.csv", t23a_rows)
    t23a_headers = ["阶段", "算例 ID", "划分", "单", "事件", "主方法", "单策略消融", "可部署最佳规则", "相对可部署改善(%)", "事后包络 Min Rule(不可部署)", "相对包络改善(%)", *rules]
    t23a_md = _to_markdown_table(
        t23a_headers,
        [
            [
                r["stage"], r["scenario_id"], r["split"], str(r["orders"]), str(r["events"]),
                _fmt(r["AD-HG-DAPPO"], dec), _fmt(r["AD-HG-SAPPO"], dec),
                r["deployable_rule"], _fmt(r["imp_vs_deployable_pct"], 2, signed=True),
                r["min_rule_posthoc"], _fmt(r["imp_vs_min_rule_pct"], 2, signed=True),
                *[_fmt(r[rule], dec) for rule in rules],
            ]
            for r in t23a_rows
        ],
    )
    (out_dir / "table_23a.md").write_text(
        "# 表 23a（L1/L3/L10 重构版）：按数据划分分列的逐算例对比总表\n\n"
        "> 注：训练集为样本内表现，验证集用于选模，**泛化结论仅依据测试集**；Min Rule 为事后包络（不可部署）。\n\n"
        + t23a_md + "\n",
        encoding="utf-8",
    )

    # Table 23b: Dual-Actor vs Same-Stage Single-Actor (no cross-stage column!)
    t23b_rows: List[Dict[str, Any]] = []
    for stage_num in stages:
        for split_key in ("train", "val", "test"):
            sub = [r for r in main_rows if r["stage_num"] == stage_num and r["split_group"] == split_key]
            for r in sub:
                p, a = r["proposed_cmax"], r["ablation_cmax"]
                diff = (p - a) if (p is not None and a is not None) else None
                imp = ((a - p) / a * 100.0) if (p is not None and a) else None
                t23b_rows.append({
                    "stage": r["stage"],
                    "scenario_id": r["scenario_id"],
                    "split": r["split_raw"],
                    "AD-HG-DAPPO": round(p, dec) if p is not None else None,
                    "AD-HG-SAPPO": round(a, dec) if a is not None else None,
                    "diff_min": round(diff, dec) if diff is not None else None,
                    "imp_pct": round(imp, 2) if imp is not None else None,
                })
            p_mean = _mean_safe([r["proposed_cmax"] for r in sub])
            a_mean = _mean_safe([r["ablation_cmax"] for r in sub])
            wtl = win_tie_loss([r["proposed_cmax"] for r in sub], [r["ablation_cmax"] for r in sub], tie_tol)
            t23b_rows.append({
                "stage": f"Stage {stage_num}",
                "scenario_id": f"{split_key} 均值 (n={len(sub)}, W/T/L={wtl[0]}/{wtl[1]}/{wtl[2]})",
                "split": split_key,
                "AD-HG-DAPPO": round(p_mean, dec) if p_mean is not None else None,
                "AD-HG-SAPPO": round(a_mean, dec) if a_mean is not None else None,
                "diff_min": round(p_mean - a_mean, dec) if (p_mean is not None and a_mean is not None) else None,
                "imp_pct": round(_ratio_imp(p_mean, a_mean), 2) if _ratio_imp(p_mean, a_mean) is not None else None,
            })

    _write_csv(out_dir / "table_23b.csv", t23b_rows)
    t23b_md = _to_markdown_table(
        ["阶段", "算例 ID", "划分", "AD-HG-DAPPO (min)", "同阶段 AD-HG-SAPPO (min)", "工期差 (min)", "相对改善 (%)"],
        [
            [
                r["stage"], r["scenario_id"], r["split"],
                _fmt(r["AD-HG-DAPPO"], dec), _fmt(r["AD-HG-SAPPO"], dec),
                _fmt(r["diff_min"], dec, signed=True), _fmt(r["imp_pct"], 2, signed=True),
            ]
            for r in t23b_rows
        ],
    )
    (out_dir / "table_23b.md").write_text(
        "# 表 23b（L3 重构版）：分层双 Actor 与同阶段单策略消融逐算例对比\n\n"
        "> 仅保留同阶段严格隔离训练的 `AD-HG-SAPPO` 单策略消融对比。\n\n"
        + t23b_md + "\n",
        encoding="utf-8",
    )

    # Table 35: Separated train / val / test comparison by Stage + Pooled Test
    methods_t35: List[Tuple[str, str]] = [
        ("AD-HG-DAPPO", "proposed_cmax"),
        ("AD-HG-SAPPO", "ablation_cmax"),
        ("DEPLOYABLE_BEST_RULE", "deployable_rule_cmax"),
        ("Min Rule (事后包络, 不可部署)", "min_rule_cmax"),
        *[(rule, rule) for rule in rules],
    ]
    t35_rows: List[Dict[str, Any]] = []
    pooled_test = [r for r in main_rows if r["split_group"] == "test"]
    pooled_p_vals: List[Optional[float]] = []
    for label, col in methods_t35:
        row_dict: Dict[str, Any] = {"method": label}
        for s in stages:
            for sp in ("train", "val", "test"):
                sub = [r for r in main_rows if r["stage_num"] == s and r["split_group"] == sp]
                vals = [r[col] for r in sub]
                row_dict[f"S{s}_{sp}_mean"] = round(_mean_safe(vals), dec) if _mean_safe(vals) is not None else None
                row_dict[f"S{s}_{sp}_std"] = round(_std_safe(vals), dec) if _std_safe(vals) is not None else None
        if label == "AD-HG-DAPPO":
            row_dict["test_wtl_main_vs_opp"] = "-"
            row_dict["test_p_raw"] = None
            pooled_p_vals.append(None)
        else:
            wtl = win_tie_loss([r["proposed_cmax"] for r in pooled_test], [r[col] for r in pooled_test], tie_tol)
            p_raw = wilcoxon_p([r["proposed_cmax"] for r in pooled_test], [r[col] for r in pooled_test])
            row_dict["test_wtl_main_vs_opp"] = f"{wtl[0]}/{wtl[1]}/{wtl[2]}"
            row_dict["test_p_raw"] = round(p_raw, 6) if p_raw is not None else None
            pooled_p_vals.append(p_raw)
        t35_rows.append(row_dict)

    # Holm adjust across the 10 opponents (excluding Min Rule post-hoc or including? Follow L9: 6 rules + deployable + SAPPO; keep Min Rule unadjusted / separate)
    holm_targets = [
        i for i, r in enumerate(t35_rows)
        if r["method"] not in ("AD-HG-DAPPO", "Min Rule (事后包络, 不可部署)")
    ]
    adj_subset = holm_adjust([pooled_p_vals[i] for i in holm_targets])
    for idx, p_h in zip(holm_targets, adj_subset):
        t35_rows[idx]["test_p_holm"] = round(p_h, 6) if p_h is not None else None
    for row_dict in t35_rows:
        row_dict.setdefault("test_p_holm", None)

    _write_csv(out_dir / "table_35.csv", t35_rows)
    t35_headers = [
        "方法", "S1 训练(6)", "S1 验证(2)", "S1 测试",
        "S2 训练(6)", "S2 验证(2)", "S2 测试",
        "S3 训练(6)", "S3 验证(2)", "S3 测试",
        f"合并测试(n={len(pooled_test)}) W/T/L", "Wilcoxon p_raw", "Holm p_holm",
    ]
    t35_md = _to_markdown_table(
        t35_headers,
        [
            [
                r["method"],
                _fmt(r.get("S1_train_mean"), dec), _fmt(r.get("S1_val_mean"), dec), _fmt(r.get("S1_test_mean"), dec),
                _fmt(r.get("S2_train_mean"), dec), _fmt(r.get("S2_val_mean"), dec), _fmt(r.get("S2_test_mean"), dec),
                _fmt(r.get("S3_train_mean"), dec), _fmt(r.get("S3_val_mean"), dec), _fmt(r.get("S3_test_mean"), dec),
                r["test_wtl_main_vs_opp"],
                f"{r['test_p_raw']:.4f}" if r["test_p_raw"] is not None else "-",
                f"{r['test_p_holm']:.4f}" if r["test_p_holm"] is not None else "-",
            ]
            for r in t35_rows
        ],
    )
    (out_dir / "table_35.md").write_text(
        "# 表 35（L1/L9/L10 重构版）：各阶段按训练/验证/测试分栏对比与合并测试统计检验\n\n"
        + t35_md + "\n",
        encoding="utf-8",
    )

    # Table 36: DAPPO vs SAPPO summary by Stage & Split + Convergence Indicators
    t36_rows: List[Dict[str, Any]] = []
    for s in stages:
        conv_d = val_conv.get((s, "AD-HG-DAPPO"), {})
        conv_s = val_conv.get((s, "AD-HG-SAPPO"), {})
        for sp in ("train", "val", "test"):
            sub = [r for r in main_rows if r["stage_num"] == s and r["split_group"] == sp]
            p_m = _mean_safe([r["proposed_cmax"] for r in sub])
            a_m = _mean_safe([r["ablation_cmax"] for r in sub])
            wtl = win_tie_loss([r["proposed_cmax"] for r in sub], [r["ablation_cmax"] for r in sub], tie_tol)
            t36_rows.append({
                "stage": f"Stage {s}",
                "split": sp,
                "n": len(sub),
                "AD-HG-DAPPO_mean": round(p_m, dec) if p_m is not None else None,
                "AD-HG-SAPPO_mean": round(a_m, dec) if a_m is not None else None,
                "saving_min": round(a_m - p_m, dec) if (p_m is not None and a_m is not None) else None,
                "imp_pct": round(_ratio_imp(p_m, a_m), 2) if _ratio_imp(p_m, a_m) is not None else None,
                "win_tie_loss": f"{wtl[0]}/{wtl[1]}/{wtl[2]}",
                "dappo_ep_within_1pct": conv_d.get("episodes_to_within_1pct_final_mean") if sp == "val" else None,
                "sappo_ep_within_1pct": conv_s.get("episodes_to_within_1pct_final_mean") if sp == "val" else None,
                "dappo_norm_auc": round(conv_d["normalized_val_auc_mean"], 4) if (sp == "val" and conv_d.get("normalized_val_auc_mean") is not None) else None,
                "sappo_norm_auc": round(conv_s["normalized_val_auc_mean"], 4) if (sp == "val" and conv_s.get("normalized_val_auc_mean") is not None) else None,
            })

    _write_csv(out_dir / "table_36.csv", t36_rows)
    t36_md = _to_markdown_table(
        ["阶段", "划分", "样本数", "主方法均值", "单策略消融均值", "工期节约(min)", "改善(%)", "胜/平/负", "DAPPO 1%收敛轮次", "SAPPO 1%收敛轮次", "DAPPO AUC", "SAPPO AUC"],
        [
            [
                r["stage"], r["split"], str(r["n"]),
                _fmt(r["AD-HG-DAPPO_mean"], dec), _fmt(r["AD-HG-SAPPO_mean"], dec),
                _fmt(r["saving_min"], dec, signed=True), _fmt(r["imp_pct"], 2, signed=True),
                r["win_tie_loss"],
                _fmt(r["dappo_ep_within_1pct"], 1) if r["dappo_ep_within_1pct"] is not None else "-",
                _fmt(r["sappo_ep_within_1pct"], 1) if r["sappo_ep_within_1pct"] is not None else "-",
                _fmt(r["dappo_norm_auc"], 4) if r["dappo_norm_auc"] is not None else "-",
                _fmt(r["sappo_norm_auc"], 4) if r["sappo_norm_auc"] is not None else "-",
            ]
            for r in t36_rows
        ],
    )
    (out_dir / "table_36.md").write_text(
        "# 表 36（L2/L3/L5 重构版）：分层双 Actor 与单策略消融分划分均值及稳健收敛指标\n\n"
        + t36_md + "\n",
        encoding="utf-8",
    )

    # Table 37: Validation (model selection) & Test (generalization) per-scenario
    t37_rows: List[Dict[str, Any]] = []
    for r in main_rows:
        if r["split_group"] not in ("val", "test"):
            continue
        t37_rows.append({
            "scenario_id": r["scenario_id"],
            "split": r["split_raw"],
            "role_note": "选模验证(不入泛化结论)" if r["split_group"] == "val" else "独立测试(泛化)",
            "stage": r["stage"],
            "scale": f"{r['initial_orders']}+{r['dynamic_events']}",
            "S_BASE": round(r["S_BASE"], dec) if r["S_BASE"] is not None else None,
            "deployable_rule": f"{_fmt(r['deployable_rule_cmax'], dec)} ({r['deployable_rule_name']})",
            "min_rule_posthoc": f"{_fmt(r['min_rule_cmax'], dec)} ({r['min_rule_name']})",
            "AD-HG-SAPPO": round(r["ablation_cmax"], dec) if r["ablation_cmax"] is not None else None,
            "AD-HG-DAPPO": round(r["proposed_cmax"], dec) if r["proposed_cmax"] is not None else None,
            "imp_vs_deployable_pct": round(r["imp_vs_deployable_pct"], 2) if r["imp_vs_deployable_pct"] is not None else None,
            "imp_vs_min_rule_pct": round(r["imp_vs_min_rule_pct"], 2) if r["imp_vs_min_rule_pct"] is not None else None,
        })

    _write_csv(out_dir / "table_37.csv", t37_rows)
    t37_md = _to_markdown_table(
        ["算例 ID", "划分", "用途定位", "阶段", "规模", "S_BASE", "可部署最佳规则", "事后包络 Min Rule", "单策略消融", "本文主方法", "相对可部署改善(%)", "相对包络改善(%)"],
        [
            [
                r["scenario_id"], r["split"], r["role_note"], r["stage"], r["scale"],
                _fmt(r["S_BASE"], dec), r["deployable_rule"], r["min_rule_posthoc"],
                _fmt(r["AD-HG-SAPPO"], dec), _fmt(r["AD-HG-DAPPO"], dec),
                _fmt(r["imp_vs_deployable_pct"], 2, signed=True),
                _fmt(r["imp_vs_min_rule_pct"], 2, signed=True),
            ]
            for r in t37_rows
        ],
    )
    (out_dir / "table_37.md").write_text(
        "# 表 37（L1/L3/L10 重构版）：验证集（仅选模）与测试集（泛化）逐算例实测表\n\n"
        + t37_md + "\n",
        encoding="utf-8",
    )

    # Table 38: Stage/Split summary + Factor Grid (separated ID vs OOD)
    t38_rows: List[Dict[str, Any]] = []
    for s in stages:
        for sp in ("train", "val", "test"):
            sub = [r for r in main_rows if r["stage_num"] == s and r["split_group"] == sp]
            if not sub:
                continue
            p_m = _mean_safe([r["proposed_cmax"] for r in sub])
            dep_m = _mean_safe([r["deployable_rule_cmax"] for r in sub])
            min_m = _mean_safe([r["min_rule_cmax"] for r in sub])
            sb_m = _mean_safe([r["S_BASE"] for r in sub])
            t38_rows.append({
                "group_type": "阶段×划分",
                "level": f"Stage {s} / {sp} (n={len(sub)})",
                "id_or_ood": "ID",
                "proposed_mean": round(p_m, dec) if p_m is not None else None,
                "deployable_rule_mean": round(dep_m, dec) if dep_m is not None else None,
                "min_rule_mean_posthoc": round(min_m, dec) if min_m is not None else None,
                "imp_vs_deployable_pct": round(_ratio_imp(p_m, dep_m), 2) if _ratio_imp(p_m, dep_m) is not None else None,
                "imp_vs_min_rule_pct": round(_ratio_imp(p_m, min_m), 2) if _ratio_imp(p_m, min_m) is not None else None,
                "imp_vs_s_base_pct": round(_ratio_imp(p_m, sb_m), 2) if _ratio_imp(p_m, sb_m) is not None else None,
                "saving_vs_deployable_min": round(dep_m - p_m, dec) if (p_m is not None and dep_m is not None) else None,
            })

    # Arrival breakdown ONLY on test set (never mixing val with test)
    for arr in ("uniform", "poisson", "burst"):
        sub = [r for r in pooled_test if r["arrival"] == arr]
        if not sub:
            continue
        p_m = _mean_safe([r["proposed_cmax"] for r in sub])
        dep_m = _mean_safe([r["deployable_rule_cmax"] for r in sub])
        min_m = _mean_safe([r["min_rule_cmax"] for r in sub])
        sb_m = _mean_safe([r["S_BASE"] for r in sub])
        t38_rows.append({
            "group_type": "到达模式(仅测试集)",
            "level": f"{arr} (n={len(sub)})",
            "id_or_ood": "ID",
            "proposed_mean": round(p_m, dec) if p_m is not None else None,
            "deployable_rule_mean": round(dep_m, dec) if dep_m is not None else None,
            "min_rule_mean_posthoc": round(min_m, dec) if min_m is not None else None,
            "imp_vs_deployable_pct": round(_ratio_imp(p_m, dep_m), 2) if _ratio_imp(p_m, dep_m) is not None else None,
            "imp_vs_min_rule_pct": round(_ratio_imp(p_m, min_m), 2) if _ratio_imp(p_m, min_m) is not None else None,
            "imp_vs_s_base_pct": round(_ratio_imp(p_m, sb_m), 2) if _ratio_imp(p_m, sb_m) is not None else None,
            "saving_vs_deployable_min": round(dep_m - p_m, dec) if (p_m is not None and dep_m is not None) else None,
        })

    if factor_rows:
        for id_label in ("ID", "OOD"):
            for arr in ("uniform", "poisson", "burst"):
                sub = [r for r in factor_rows if r["id_or_ood"] == id_label and r["arrival"] == arr]
                if not sub:
                    continue
                p_m = _mean_safe([r["proposed_cmax"] for r in sub])
                dep_m = _mean_safe([r["deployable_rule_cmax"] for r in sub])
                min_m = _mean_safe([r["min_rule_cmax"] for r in sub])
                sb_m = _mean_safe([r["S_BASE"] for r in sub])
                t38_rows.append({
                    "group_type": f"因子网格 T-F ({id_label})",
                    "level": f"{arr} (n={len(sub)})",
                    "id_or_ood": id_label,
                    "proposed_mean": round(p_m, dec) if p_m is not None else None,
                    "deployable_rule_mean": round(dep_m, dec) if dep_m is not None else None,
                    "min_rule_mean_posthoc": round(min_m, dec) if min_m is not None else None,
                    "imp_vs_deployable_pct": round(_ratio_imp(p_m, dep_m), 2) if _ratio_imp(p_m, dep_m) is not None else None,
                    "imp_vs_min_rule_pct": round(_ratio_imp(p_m, min_m), 2) if _ratio_imp(p_m, min_m) is not None else None,
                    "imp_vs_s_base_pct": round(_ratio_imp(p_m, sb_m), 2) if _ratio_imp(p_m, sb_m) is not None else None,
                    "saving_vs_deployable_min": round(dep_m - p_m, dec) if (p_m is not None and dep_m is not None) else None,
                })

    _write_csv(out_dir / "table_38.csv", t38_rows)
    t38_md = _to_markdown_table(
        ["分组维度", "水平(样本数)", "分布标签", "主方法均值", "可部署规则均值", "事后包络均值", "相对可部署改善(%)", "相对包络改善(%)", "相对 S_BASE 改善(%)", "节约工时(min)"],
        [
            [
                r["group_type"], r["level"], r["id_or_ood"],
                _fmt(r["proposed_mean"], dec), _fmt(r["deployable_rule_mean"], dec),
                _fmt(r["min_rule_mean_posthoc"], dec),
                _fmt(r["imp_vs_deployable_pct"], 2, signed=True),
                _fmt(r["imp_vs_min_rule_pct"], 2, signed=True),
                _fmt(r["imp_vs_s_base_pct"], 2, signed=True),
                _fmt(r["saving_vs_deployable_min"], dec, signed=True),
            ]
            for r in t38_rows
        ],
    )
    (out_dir / "table_38.md").write_text(
        "# 表 38（L1/L4/L10 重构版）：按数据划分与分布内外严格分列的汇总表\n\n"
        + t38_md + "\n",
        encoding="utf-8",
    )

    # Table 39: Appendix H Full Scenario Registry
    t39_rows: List[Dict[str, Any]] = [
        {
            "scenario_id": r["scenario_id"],
            "stage_split": f"S{r['stage_num']}/{r['split_raw']}",
            "scale": f"{r['initial_orders']}+{r['dynamic_events']}",
            "arrival": r["arrival"],
            "id_or_ood": r["id_or_ood"],
            "AD-HG-DAPPO": round(r["proposed_cmax"], dec) if r["proposed_cmax"] is not None else None,
            "AD-HG-SAPPO": round(r["ablation_cmax"], dec) if r["ablation_cmax"] is not None else None,
            "deployable_rule": f"{_fmt(r['deployable_rule_cmax'], dec)} ({r['deployable_rule_name']})",
            "min_rule_posthoc": f"{_fmt(r['min_rule_cmax'], dec)} ({r['min_rule_name']})",
            "imp_vs_deployable_pct": round(r["imp_vs_deployable_pct"], 2) if r["imp_vs_deployable_pct"] is not None else None,
            **{rule: round(r[rule], dec) if r[rule] is not None else None for rule in rules},
        }
        for r in rows
    ]
    _write_csv(out_dir / "table_39.csv", t39_rows)
    t39_md = _to_markdown_table(
        ["算例 ID", "阶段/划分", "单+事件", "到达模式", "分布", "主方法", "消融", "可部署规则", "事后包络", "相对可部署改善(%)", *rules],
        [
            [
                r["scenario_id"], r["stage_split"], r["scale"], r["arrival"], r["id_or_ood"],
                _fmt(r["AD-HG-DAPPO"], dec), _fmt(r["AD-HG-SAPPO"], dec),
                r["deployable_rule"], r["min_rule_posthoc"],
                _fmt(r["imp_vs_deployable_pct"], 2, signed=True),
                *[_fmt(r[rule], dec) for rule in rules],
            ]
            for r in t39_rows
        ],
    )
    (out_dir / "table_39.md").write_text(
        "# 表 39（附录 H 自动生成版）：全场景参数与实测完工时间总表\n\n"
        + t39_md + "\n",
        encoding="utf-8",
    )

    # Authoritative inline numbers JSON
    pooled_train = [r for r in main_rows if r["split_group"] == "train"]
    pooled_val = [r for r in main_rows if r["split_group"] == "val"]
    s3_rows = [r for r in main_rows if r["stage_num"] == 3]

    worst_test = min(
        (r for r in pooled_test if r["imp_vs_min_rule_pct"] is not None),
        key=lambda r: float(r["imp_vs_min_rule_pct"]),
    )
    inline_numbers = {
        "schema": "l3_manuscript_inline_numbers_v1",
        "generated_at": datetime.now().isoformat(),
        "source_csv": str(auth_csv),
        "source_csv_sha256": csv_sha,
        "config_sha256": config_sha,
        "tie_tolerance_min": tie_tol,
        "deployable_rule_by_stage": {f"S{k}": v for k, v in deployable_by_stage.items()},
        "per_stage_test_ratio_imp_vs_deployable_pct": {
            f"S{s}": round(
                _ratio_imp(
                    _mean_safe([r["proposed_cmax"] for r in pooled_test if r["stage_num"] == s]),
                    _mean_safe([r["deployable_rule_cmax"] for r in pooled_test if r["stage_num"] == s]),
                ),
                2,
            )
            for s in stages
        },
        "per_stage_test_ratio_imp_vs_min_rule_pct": {
            f"S{s}": round(
                _ratio_imp(
                    _mean_safe([r["proposed_cmax"] for r in pooled_test if r["stage_num"] == s]),
                    _mean_safe([r["min_rule_cmax"] for r in pooled_test if r["stage_num"] == s]),
                ),
                2,
            )
            for s in stages
        },
        "pooled_test_per_case_mean_imp_vs_deployable_pct": round(_mean_safe([r["imp_vs_deployable_pct"] for r in pooled_test]), 2),
        "pooled_test_per_case_mean_imp_vs_min_rule_pct": round(_mean_safe([r["imp_vs_min_rule_pct"] for r in pooled_test]), 2),
        "pooled_train_per_case_mean_imp_vs_min_rule_pct": round(_mean_safe([r["imp_vs_min_rule_pct"] for r in pooled_train]), 2),
        "pooled_val_per_case_mean_imp_vs_min_rule_pct": round(_mean_safe([r["imp_vs_min_rule_pct"] for r in pooled_val]), 2),
        "win_tie_loss": {
            "all_main_vs_s_base": list(win_tie_loss([r["proposed_cmax"] for r in main_rows], [r["S_BASE"] for r in main_rows], tie_tol)),
            "all_main_vs_min_rule": list(win_tie_loss([r["proposed_cmax"] for r in main_rows], [r["min_rule_cmax"] for r in main_rows], tie_tol)),
            "all_main_vs_deployable_rule": list(win_tie_loss([r["proposed_cmax"] for r in main_rows], [r["deployable_rule_cmax"] for r in main_rows], tie_tol)),
            "s3_main_vs_sappo": list(win_tie_loss([r["proposed_cmax"] for r in s3_rows], [r["ablation_cmax"] for r in s3_rows], tie_tol)),
            "pooled_train_vs_min_rule": list(win_tie_loss([r["proposed_cmax"] for r in pooled_train], [r["min_rule_cmax"] for r in pooled_train], tie_tol)),
            "pooled_val_vs_min_rule": list(win_tie_loss([r["proposed_cmax"] for r in pooled_val], [r["min_rule_cmax"] for r in pooled_val], tie_tol)),
            "pooled_test_vs_min_rule": list(win_tie_loss([r["proposed_cmax"] for r in pooled_test], [r["min_rule_cmax"] for r in pooled_test], tie_tol)),
            "pooled_test_vs_deployable_rule": list(win_tie_loss([r["proposed_cmax"] for r in pooled_test], [r["deployable_rule_cmax"] for r in pooled_test], tie_tol)),
            "pooled_train_vs_sappo": list(win_tie_loss([r["proposed_cmax"] for r in pooled_train], [r["ablation_cmax"] for r in pooled_train], tie_tol)),
            "pooled_val_vs_sappo": list(win_tie_loss([r["proposed_cmax"] for r in pooled_val], [r["ablation_cmax"] for r in pooled_val], tie_tol)),
            "pooled_test_vs_sappo": list(win_tie_loss([r["proposed_cmax"] for r in pooled_test], [r["ablation_cmax"] for r in pooled_test], tie_tol)),
        },
        "pooled_test_wilcoxon_p_raw": {
            "vs_deployable_rule": wilcoxon_p([r["proposed_cmax"] for r in pooled_test], [r["deployable_rule_cmax"] for r in pooled_test]),
            "vs_min_rule": wilcoxon_p([r["proposed_cmax"] for r in pooled_test], [r["min_rule_cmax"] for r in pooled_test]),
            "vs_sappo": wilcoxon_p([r["proposed_cmax"] for r in pooled_test], [r["ablation_cmax"] for r in pooled_test]),
            "vs_s_base": wilcoxon_p([r["proposed_cmax"] for r in pooled_test], [r["S_BASE"] for r in pooled_test]),
        },
        "worst_test_scenario": {
            "scenario_id": worst_test["scenario_id"],
            "proposed_cmax": round(worst_test["proposed_cmax"], dec),
            "min_rule_name": worst_test["min_rule_name"],
            "min_rule_cmax": round(worst_test["min_rule_cmax"], dec),
            "imp_vs_min_rule_pct": round(worst_test["imp_vs_min_rule_pct"], 2),
        },
    }
    (out_dir / "manuscript_inline_numbers.json").write_text(
        json.dumps(inline_numbers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    master_md = "\n\n".join([
        f"# L3 单一 CSV 自动出表总览\n\n- 权威数据源：`{auth_csv}` (SHA-256: `{csv_sha[:16]}...`)\n- 生成时间：{inline_numbers['generated_at']}\n- 各阶段可部署最佳规则（仅由 train+val 选定）：`{inline_numbers['deployable_rule_by_stage']}`",
        (out_dir / "table_35.md").read_text(encoding="utf-8"),
        (out_dir / "table_36.md").read_text(encoding="utf-8"),
        (out_dir / "table_37.md").read_text(encoding="utf-8"),
        (out_dir / "table_38.md").read_text(encoding="utf-8"),
        (out_dir / "table_23b.md").read_text(encoding="utf-8"),
        (out_dir / "table_23a.md").read_text(encoding="utf-8"),
        (out_dir / "table_39.md").read_text(encoding="utf-8"),
    ])
    (out_dir / "tables_master_report.md").write_text(master_md, encoding="utf-8")

    return inline_numbers


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="Explicit path to a single authoritative comparison CSV")
    parser.add_argument("--config", default=None, help="Path to configs/paper_tables.json")
    parser.add_argument("--val-curve", default=None, help="Optional path to val_curve.csv")
    parser.add_argument("--output", default=None, help="Output root or explicit directory")
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds, e.g. 1 or 1,2,3")
    parser.add_argument("--skip-missing-seeds", action="store_true", help="Skip seeds whose authoritative CSV is missing")
    args = parser.parse_args(argv)

    if args.csv:
        summary = generate_all_tables_from_csv(
            csv_path=args.csv,
            config_path=args.config,
            val_curve_path=args.val_curve,
            output_dir=args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    config, _ = load_l3_config(args.config)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    all_summaries: Dict[str, Any] = {}
    for seed in seeds:
        auth_csv = ROOT / config["authoritative_csv_path"].format(seed=seed)
        if not auth_csv.is_file():
            if args.skip_missing_seeds:
                print(f"[Skip seed_{seed}] Authoritative CSV missing: {auth_csv}")
                continue
            raise FileNotFoundError(f"Authoritative CSV missing for seed_{seed}: {auth_csv}")
        seed_out = str(Path(args.output) / f"seed_{seed}") if args.output else None
        all_summaries[f"seed_{seed}"] = generate_all_tables_from_csv(
            config_path=args.config,
            val_curve_path=args.val_curve,
            output_dir=seed_out,
            seed=seed,
        )
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
