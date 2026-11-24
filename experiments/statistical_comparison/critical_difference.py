"""Draw a Nemenyi critical-difference diagram from complete test-panel ranks."""
from __future__ import annotations

from pathlib import Path


def _maximal_groups(sorted_ranks: list[tuple[str, float]], cd: float):
    intervals = []
    for start in range(len(sorted_ranks)):
        end = start
        while end + 1 < len(sorted_ranks) and sorted_ranks[end + 1][1] - sorted_ranks[start][1] <= cd:
            end += 1
        if end > start:
            intervals.append((start, end))
    return [(a, b) for a, b in intervals if not any(
        c <= a and b <= d and (c, d) != (a, b) for c, d in intervals)]


def draw_cd(ranks: list[dict], test: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    ordered = sorted(((r["method_id"], float(r["average_rank"])) for r in ranks),
                     key=lambda x: (x[1], x[0]))
    k = len(ordered)
    if k != int(test["methods"]):
        raise ValueError("CD rank count mismatch")
    cd = float(test["critical_difference"])
    groups = _maximal_groups(ordered, cd)
    fig, ax = plt.subplots(figsize=(12, 6.7), constrained_layout=True)
    for index, (method, rank) in enumerate(ordered):
        y = k - index
        ax.plot([1, rank], [y, y], color="#D8DFE8", linewidth=.9, zorder=1)
        ax.scatter([rank], [y], s=35, color="#175A91" if method == "AD-HG-DAPPO" else "#6A7480", zorder=3)
        ax.text(rank + .16, y, f"{method}  ({rank:.2f})", va="center", fontsize=8.5,
                fontweight="bold" if method == "AD-HG-DAPPO" else "normal")
    for level, (first, last) in enumerate(groups):
        y = k + 1.25 + level * .28
        ax.plot([ordered[first][1], ordered[last][1]], [y, y], color="#2B6F65",
                linewidth=3, solid_capstyle="round")
    ax.plot([1, 1 + cd], [k + 1.25 + max(1, len(groups)) * .28] * 2,
            color="#B74F2A", linewidth=3)
    ax.text(1, k + 1.7 + max(1, len(groups)) * .28, f"Nemenyi CD = {cd:.3f}",
            fontsize=10, color="#B74F2A")
    ax.set_xlim(.75, k + 3.7)
    ax.set_ylim(.4, k + 2.5 + max(1, len(groups)) * .28)
    ax.set_xticks(range(1, k + 1))
    ax.set_yticks([])
    ax.set_xlabel("Average rank across 36 paired test scenarios (lower is better)")
    ax.set_title("Friedman ranks and Nemenyi critical difference")
    ax.text(.99, .01, f"Friedman p = {test['p_value']:.4g}; alpha = {test['alpha']:.2g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output
