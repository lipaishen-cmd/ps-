"""Plot L11 (DAPPO-Expand) vs AD-HG-DAPPO quality and Split-stage graph overhead."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _parse_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "null"):
        return None
    return float(value)


def _aggregate_for_plot(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Aggregate completed scenario means per (stage, method_id)."""
    by_stage_method_sid: Dict[tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("completed", "")).strip().lower() != "true":
            continue
        stage = int(row["stage"])
        method = str(row["method_id"])
        sid = str(row["scenario_id"])
        by_stage_method_sid[(stage, method, sid)].append(row)

    grouped: Dict[tuple[int, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (stage, method, _sid), s_rows in sorted(by_stage_method_sid.items()):
        cmax_vals = [v for v in (_parse_float(r.get("final_cmax")) for r in s_rows) if v is not None]
        gb_vals = [v for v in (_parse_float(r.get("mean_split_graph_build_ms")) for r in s_rows) if v is not None]
        hf_vals = [v for v in (_parse_float(r.get("mean_split_hgnn_forward_ms")) for r in s_rows) if v is not None]
        sub_vals = [v for v in (_parse_float(r.get("mean_split_subbatch_nodes")) for r in s_rows) if v is not None]
        edge_vals = [v for v in (_parse_float(r.get("mean_split_total_edges")) for r in s_rows) if v is not None]
        if cmax_vals:
            grouped[(stage, method)]["cmax"].append(statistics.mean(cmax_vals))
        if gb_vals:
            grouped[(stage, method)]["split_gb_ms"].append(statistics.mean(gb_vals))
        if hf_vals:
            grouped[(stage, method)]["split_hf_ms"].append(statistics.mean(hf_vals))
        if gb_vals and hf_vals and len(gb_vals) == len(hf_vals):
            grouped[(stage, method)]["split_total_ms"].append(
                statistics.mean(a + b for a, b in zip(gb_vals, hf_vals))
            )
        if sub_vals:
            grouped[(stage, method)]["split_subbatch_nodes"].append(statistics.mean(sub_vals))
        if edge_vals:
            grouped[(stage, method)]["split_total_edges"].append(statistics.mean(edge_vals))

    result: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (stage, method), metrics in grouped.items():
        stats_dict: Dict[str, float] = {}
        for key, vals in metrics.items():
            stats_dict[f"{key}_mean"] = statistics.mean(vals) if vals else 0.0
            stats_dict[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats_dict["n_scenarios"] = float(len(metrics.get("cmax", [])))
        result[stage][method] = stats_dict
    return result


def generate_plot(expand_eval_paths: Sequence[Path], output_path: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(output_path)
    rows: List[Dict[str, Any]] = []
    for path in expand_eval_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("No rows found in --expand-eval CSV file(s).")

    agg = _aggregate_for_plot(rows)
    stages = sorted(agg.keys())
    if not stages:
        raise ValueError("No completed rows available to plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    x = np.arange(len(stages))
    width = 0.35

    main_id = "AD-HG-DAPPO"
    exp_id = "DAPPO-Expand"

    # Left panel: Stage-wise Cmax comparison
    main_cmax = [agg[s].get(main_id, {}).get("cmax_mean", 0.0) for s in stages]
    main_std = [agg[s].get(main_id, {}).get("cmax_std", 0.0) for s in stages]
    exp_cmax = [agg[s].get(exp_id, {}).get("cmax_mean", 0.0) for s in stages]
    exp_std = [agg[s].get(exp_id, {}).get("cmax_std", 0.0) for s in stages]

    bars_m = ax_left.bar(
        x - width / 2, main_cmax, width, yerr=main_std, capsize=4,
        color="#1f4e79", edgecolor="black", linewidth=0.6,
        label=f"{main_id} (On-Demand)",
    )
    bars_e = ax_left.bar(
        x + width / 2, exp_cmax, width, yerr=exp_std, capsize=4,
        color="#c55a11", edgecolor="black", linewidth=0.6,
        label=f"{exp_id} (Pre-Expanded)",
    )
    for idx, s in enumerate(stages):
        m_val = main_cmax[idx]
        e_val = exp_cmax[idx]
        if m_val > 0 and e_val > 0:
            gap_pct = (e_val - m_val) / m_val * 100.0
            top_y = max(m_val + main_std[idx], e_val + exp_std[idx])
            ax_left.text(
                x[idx], top_y * 1.03, f"Δ={gap_pct:+.2f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"Stage {s}\n(n={int(agg[s].get(main_id, {}).get('n_scenarios', 0))})" for s in stages])
    ax_left.set_ylabel("Makespan $C_{\\max}$ (min)")
    ax_left.set_title("(a) Scheduling Quality Comparison ($C_{\\max}$)")
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.legend(loc="best", fontsize=8.5)
    del bars_m, bars_e

    # Right panel: Split-stage graph build + HGNN latency & node/edge bloat
    main_gb = np.array([agg[s].get(main_id, {}).get("split_gb_ms_mean", 0.0) for s in stages])
    main_hf = np.array([agg[s].get(main_id, {}).get("split_hf_ms_mean", 0.0) for s in stages])
    exp_gb = np.array([agg[s].get(exp_id, {}).get("split_gb_ms_mean", 0.0) for s in stages])
    exp_hf = np.array([agg[s].get(exp_id, {}).get("split_hf_ms_mean", 0.0) for s in stages])

    ax_right.bar(
        x - width / 2, main_gb, width, color="#2e75b6", edgecolor="black", linewidth=0.6,
        label=f"{main_id}: Graph Build",
    )
    ax_right.bar(
        x - width / 2, main_hf, width, bottom=main_gb, color="#9dc3e6", edgecolor="black", linewidth=0.6,
        label=f"{main_id}: HGNN Forward",
    )
    ax_right.bar(
        x + width / 2, exp_gb, width, color="#c55a11", edgecolor="black", linewidth=0.6,
        label=f"{exp_id}: Graph Build",
    )
    ax_right.bar(
        x + width / 2, exp_hf, width, bottom=exp_gb, color="#f4b183", edgecolor="black", linewidth=0.6,
        label=f"{exp_id}: HGNN Forward",
    )

    for idx, s in enumerate(stages):
        m_nodes = agg[s].get(main_id, {}).get("split_subbatch_nodes_mean", 0.0)
        e_nodes = agg[s].get(exp_id, {}).get("split_subbatch_nodes_mean", 0.0)
        m_edges = agg[s].get(main_id, {}).get("split_total_edges_mean", 0.0)
        e_edges = agg[s].get(exp_id, {}).get("split_total_edges_mean", 0.0)
        ax_right.text(
            x[idx] - width / 2, (main_gb[idx] + main_hf[idx]) * 1.03 + 0.02,
            f"sub={m_nodes:.0f}\nE={m_edges:.0f}",
            ha="center", va="bottom", fontsize=7.5, color="#1f4e79",
        )
        ax_right.text(
            x[idx] + width / 2, (exp_gb[idx] + exp_hf[idx]) * 1.03 + 0.02,
            f"sub={e_nodes:.0f}\nE={e_edges:.0f}",
            ha="center", va="bottom", fontsize=7.5, color="#833c0c",
        )

    max_lat = max(
        float(np.max(main_gb + main_hf)) if len(main_gb) else 0.0,
        float(np.max(exp_gb + exp_hf)) if len(exp_gb) else 0.0,
    )
    ax_right.set_ylim(0.0, max(1.0, max_lat * 1.32))
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"Stage {s}" for s in stages])
    ax_right.set_ylabel("Split-Stage Graph + HGNN Time per Event (ms)")
    ax_right.set_title("(b) Split-Stage Graph Bloat & Latency Overhead")
    ax_right.grid(axis="y", alpha=0.25)
    ax_right.legend(loc="upper left", fontsize=7.8, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expand-eval", required=True, action="append",
        help="Path to expand_eval.csv; repeatable if stages were evaluated separately.",
    )
    parser.add_argument("--output", required=True, help="New output SVG/PNG/PDF figure path.")
    args = parser.parse_args(argv)
    out = generate_plot([Path(p) for p in args.expand_eval], Path(args.output))
    print(out.resolve())


if __name__ == "__main__":
    main()
