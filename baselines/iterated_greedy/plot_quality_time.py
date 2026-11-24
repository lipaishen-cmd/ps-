"""事件级 IG test-set quality/time plot & strict budget-compliance table."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _positive(value, label):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _is_budget_met(row):
    val = str(row.get("budget_met", "")).strip().lower()
    if val in ("true", "false"):
        return val == "true"
    return int(row.get("budget_misses", 0)) == 0


def _validate_inputs(ig_rows, main_rows, reference_rows, expected_scenarios=36):
    references = {}
    selected_rules = {}
    valid_rules = {"S_BASE", "FIFO", "SPT", "LPT", "EDD", "ECT"}
    for row in reference_rows:
        sid = row["scenario_id"]
        if sid in references or row["selection_split"] != "train_val":
            raise ValueError("Reference rule must be uniquely selected on training+validation only")
        stage = int(row["stage"])
        rule = row["rule_id"]
        if stage not in (1, 2, 3) or rule not in valid_rules:
            raise ValueError("Invalid stage/rule in deployable reference")
        if stage in selected_rules and selected_rules[stage] != rule:
            raise ValueError("Deployable rule must be fixed per stage, not selected per test scenario")
        selected_rules[stage] = rule
        references[sid] = row
    if len(references) != expected_scenarios:
        raise ValueError(f"Expected {expected_scenarios} test reference scenarios, got {len(references)}")
    if expected_scenarios == 36 and {stage: sum(int(r["stage"]) == stage for r in references.values())
                                     for stage in (1, 2, 3)} != {1: 12, 2: 12, 3: 12}:
        raise ValueError("Expected 12 test scenarios in each of stages 1, 2 and 3")
    main = {}
    for row in main_rows:
        sid = row["scenario_id"]
        if sid in main or row["completed"].lower() != "true":
            raise ValueError("Main method requires one completed replay per scenario")
        main[sid] = row
    if set(main) != set(references):
        raise ValueError("Main and reference scenario coverage differs")
    ig = defaultdict(list)
    for row in ig_rows:
        if row["run_kind"] != "budgeted" or row["split"] != "test":
            raise ValueError("事件级 IG plot requires actual budgeted test runs")
        if row["completed"].lower() != "true":
            raise ValueError("Incomplete IG restart must be reported, not plotted as complete")
        if row["budget_tag"] not in {"10x", "5s"}:
            raise ValueError("Unknown IG budget")
        ig[row["scenario_id"], row["budget_tag"]].append(row)
    if set(ig) != {(sid, tag) for sid in references for tag in ("10x", "5s")}:
        raise ValueError("Need two IG budgets for every test scenario")
    for sid, ref in references.items():
        scenario_hash = ref["scenario_sha256"]
        if main[sid]["scenario_sha256"] != scenario_hash:
            raise ValueError(f"Main/reference hash mismatch: {sid}")
        if int(main[sid]["stage"]) != int(ref["stage"]) or main[sid]["split"] != "test":
            raise ValueError(f"Main/reference stage or role mismatch: {sid}")
        if int(ig[sid, "10x"][0]["event_count"]) != int(ig[sid, "5s"][0]["event_count"]) or int(main[sid]["event_count"]) != int(ig[sid, "5s"][0]["event_count"]):
            raise ValueError(f"Event counts differ across IG budgets: {sid}")
    return references, main, ig


def _summarize_subset(references, main, ig, stage_filter=None):
    sub_refs = {sid: ref for sid, ref in references.items()
                if stage_filter is None or int(ref["stage"]) == stage_filter}
    points = []
    for method in ("AD-HG-DAPPO", "IG-10x", "IG-5s"):
        strict_qualities, raw_qualities = [], []
        strict_cmax_vals, raw_cmax_vals = [], []
        total_time, total_events, misses, compliant_restarts, total_restarts = 0.0, 0, 0, 0, 0
        for sid, ref in sorted(sub_refs.items()):
            reference_cmax = _positive(ref["reference_cmax"], "reference Cmax")
            scenario_hash = ref["scenario_sha256"]
            if method == "AD-HG-DAPPO":
                runs = [main[sid]]
                event_count = int(runs[0]["event_count"])
                scenario_cmax = _positive(runs[0]["cmax"], "main Cmax")
                scenario_time = _positive(runs[0]["total_decision_s"], "main time")
                total_restarts += 1
                compliant_restarts += 1
                strict_qualities.append(100.0 * (reference_cmax - scenario_cmax) / reference_cmax)
                raw_qualities.append(strict_qualities[-1])
                strict_cmax_vals.append(scenario_cmax)
                raw_cmax_vals.append(scenario_cmax)
            else:
                runs = ig[sid, method.removeprefix("IG-")]
                if (len(runs) != 3 or {r["restart"] for r in runs} != {"1", "2", "3"}
                        or len({r["event_count"] for r in runs}) != 1):
                    raise ValueError(f"Need exactly 3 independent IG restarts: {sid}/{method}")
                if any(r["scenario_sha256"] != scenario_hash for r in runs):
                    raise ValueError(f"IG/reference hash mismatch: {sid}")
                if any(int(r["stage"]) != int(ref["stage"]) for r in runs):
                    raise ValueError(f"IG/reference stage mismatch: {sid}")
                event_count = sum(int(r["event_count"]) for r in runs)
                run_misses = sum(int(r["budget_misses"]) for r in runs)
                misses += run_misses
                scenario_time = sum(_positive(r["total_decision_s"], "IG time") for r in runs)
                raw_cmax = statistics.mean(_positive(r["cmax"], "IG Cmax") for r in runs)
                raw_cmax_vals.append(raw_cmax)
                raw_qualities.append(100.0 * (reference_cmax - raw_cmax) / reference_cmax)

                compliant_runs = [r for r in runs if _is_budget_met(r)]
                total_restarts += len(runs)
                compliant_restarts += len(compliant_runs)
                # Strict budget compliance: timed-out restarts cannot count as valid within-budget solutions.
                if len(compliant_runs) == 3:
                    strict_cmax = statistics.mean(_positive(r["cmax"], "IG strict Cmax") for r in compliant_runs)
                    strict_cmax_vals.append(strict_cmax)
                    strict_qualities.append(100.0 * (reference_cmax - strict_cmax) / reference_cmax)
            if event_count <= 0:
                raise ValueError(f"No events: {sid}")
            total_time += scenario_time
            total_events += event_count
        met_events = total_events - misses
        points.append({
            "stage": f"Stage {stage_filter}" if stage_filter is not None else "All",
            "method": method,
            "scenario_count": len(sub_refs),
            "strict_compliant_scenarios": len(strict_qualities),
            "scenario_budget_met_rate": len(strict_qualities) / len(sub_refs),
            "compliant_restarts": compliant_restarts,
            "total_restarts": total_restarts,
            "event_budget_met_rate": met_events / total_events,
            "mean_relative_improvement_pct": statistics.mean(strict_qualities) if strict_qualities else None,
            "raw_mean_relative_improvement_pct": statistics.mean(raw_qualities),
            "strict_mean_cmax": statistics.mean(strict_cmax_vals) if strict_cmax_vals else None,
            "raw_mean_cmax": statistics.mean(raw_cmax_vals),
            "mean_event_decision_s": total_time / total_events,
            "event_observations": total_events,
            "budget_met_events": met_events,
            "budget_misses": misses,
        })
    return points


def quality_time_points(ig_rows, main_rows, reference_rows, expected_scenarios=36):
    """Equal scenario weight for strict quality; event-weighted mean for decision time."""
    references, main, ig = _validate_inputs(ig_rows, main_rows, reference_rows, expected_scenarios)
    return _summarize_subset(references, main, ig, stage_filter=None)


def stage_quality_time_rows(ig_rows, main_rows, reference_rows, expected_scenarios=36):
    """Per-stage + overall strict budget compliance summary table."""
    references, main, ig = _validate_inputs(ig_rows, main_rows, reference_rows, expected_scenarios)
    rows = []
    for stage in sorted({int(r["stage"]) for r in references.values()}):
        rows.extend(_summarize_subset(references, main, ig, stage_filter=stage))
    rows.extend(_summarize_subset(references, main, ig, stage_filter=None))
    return rows


def _fmt_num(val, decimals=2, signed=False):
    if val is None:
        return "N/A (0 达标)"
    return f"{val:+.{decimals}f}" if signed else f"{val:.{decimals}f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ig-runs", required=True, help="Combined actual test IG runs, two budgets x 3 restarts")
    p.add_argument("--main-runs", required=True, help="Actual trained-checkpoint main_runs.csv")
    p.add_argument("--reference-csv", required=True,
                   help="scenario_id,scenario_sha256,reference_cmax,selection_split=train_val")
    p.add_argument("--output", required=True, help="New PDF/SVG/PNG file")
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    ig_rows = read(args.ig_runs)
    main_rows = read(args.main_runs)
    ref_rows = read(args.reference_csv)
    points = quality_time_points(ig_rows, main_rows, ref_rows)
    table_rows = stage_quality_time_rows(ig_rows, main_rows, ref_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax_qt, ax_comp) = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)
    colors = {"AD-HG-DAPPO": "#155e75", "IG-10x": "#d97706", "IG-5s": "#b91c1c"}

    # Panel (a): Quality-Time Tradeoff (strict within-budget solutions vs raw fallback)
    for point in points:
        name = point["method"]
        met_pct = 100.0 * point["event_budget_met_rate"]
        strict_n = point["strict_compliant_scenarios"]
        total_n = point["scenario_count"]
        raw_y = point["raw_mean_relative_improvement_pct"]
        strict_y = point["mean_relative_improvement_pct"]
        x = point["mean_event_decision_s"]
        dx, ha = (-8, "right") if x > 4.0 else (7, "left")
        if strict_y is not None:
            label = f"{name} (budget_met={met_pct:.1f}%, {strict_n}/{total_n} valid)"
            ax_qt.scatter(x, strict_y, s=105, color=colors[name], zorder=4, label=label)
            ax_qt.annotate(f"{name}\n({met_pct:.1f}% met, {strict_n}/{total_n})",
                           (x, strict_y), xytext=(dx, 6), textcoords="offset points", ha=ha, fontsize=8)
            if strict_n < total_n:
                ax_qt.scatter(x, raw_y, s=90, facecolors="none", edgecolors=colors[name],
                              linewidths=1.6, zorder=3, label=f"{name} raw fallback (incl. timeout)")
                ax_qt.plot([x, x], [strict_y, raw_y], linestyle=":", color=colors[name], alpha=0.8)
                ax_qt.annotate(f"{name} raw", (x, raw_y), xytext=(dx, -11),
                               textcoords="offset points", ha=ha, fontsize=7.5, color=colors[name])
        else:
            ax_qt.scatter(x, raw_y, s=90, facecolors="none", edgecolors=colors[name],
                          linewidths=1.6, zorder=3,
                          label=f"{name} (0/{total_n} strict, raw fallback only)")
    ax_qt.axhline(0.0, color="#64748b", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_qt.set(xlabel="Mean decision time per event (s)",
              ylabel="Mean Cmax improvement vs deployable rule (%)",
              title=f"(a) Strict Within-Budget Quality–Time (n={points[0]['scenario_count']})")
    ax_qt.set_xlim(-0.2, 5.5)
    ax_qt.margins(y=0.16)
    ax_qt.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax_qt.grid(alpha=0.25)
    ax_qt.legend(fontsize=7.3, loc="center right", bbox_to_anchor=(0.98, 0.48))

    # Panel (b): Event-level Real-Time Budget Compliance Rate (budget_met %)
    stages = ["Stage 1", "Stage 2", "Stage 3", "All"]
    methods = ["AD-HG-DAPPO", "IG-10x", "IG-5s"]
    lookup = {(r["stage"], r["method"]): 100.0 * r["event_budget_met_rate"] for r in table_rows}
    bar_w = 0.26
    xs = list(range(len(stages)))
    for idx, m in enumerate(methods):
        ys = [lookup[(st, m)] for st in stages]
        positions = [x + (idx - 1) * bar_w for x in xs]
        bars = ax_comp.bar(positions, ys, width=bar_w, color=colors[m], alpha=0.9, label=m)
        for bar, y_val in zip(bars, ys):
            ax_comp.text(bar.get_x() + bar.get_width() / 2.0, y_val + 1.5,
                         f"{y_val:.0f}%" if y_val >= 99.95 else f"{y_val:.1f}%",
                         ha="center", va="bottom", fontsize=7.3)
    ax_comp.set_xticks(xs)
    ax_comp.set_xticklabels(stages)
    ax_comp.set_ylim(0, 114)
    ax_comp.set(ylabel="Event-level budget_met rate (%)",
                title="(b) Real-Time Budget Compliance Rate (budget_met)")
    ax_comp.grid(axis="y", alpha=0.25)
    ax_comp.legend(fontsize=8, loc="lower left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)

    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    table_csv = out.with_name(out.stem + "_table.csv")
    with table_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    md_lines = [
        "# 事件级 IG 与主算法实时达标率及质量—时间对比表",
        "",
        "> 严格预算合规口径：**超时事件的解不能算作预算内达成的有效解**；只有全部事件满足 `decision_s <= budget` 的重启与场景才计入 `严格预算内 Cmax / 改善(%)`。",
        "",
        "| 阶段 | 方法 | 场景数 | 严格达标场景 | 实时达标率 (`budget_met`) | 严格预算内 Cmax | 严格预算内改善(%) | 物理全量 Cmax(含超时回退) | 物理全量改善(%) | 单事件均值耗时(ms) | 超时事件/总事件 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in table_rows:
        md_lines.append(
            f"| {r['stage']} | {r['method']} | {r['scenario_count']} | "
            f"{r['strict_compliant_scenarios']}/{r['scenario_count']} ({100*r['scenario_budget_met_rate']:.1f}%) | "
            f"**{100*r['event_budget_met_rate']:.2f}%** | "
            f"{_fmt_num(r['strict_mean_cmax'], 2)} | {_fmt_num(r['mean_relative_improvement_pct'], 2, signed=True)} | "
            f"{_fmt_num(r['raw_mean_cmax'], 2)} | {_fmt_num(r['raw_mean_relative_improvement_pct'], 2, signed=True)} | "
            f"{1000*r['mean_event_decision_s']:.2f} | {r['budget_misses']}/{r['event_observations']} |"
        )
    out.with_name(out.stem + "_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
