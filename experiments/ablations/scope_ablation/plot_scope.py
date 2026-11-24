"""L7 scatter: final Cmax versus old batches moved per event, by stage."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-eval", required=True, action="append",
                        help="One scope_eval.csv; repeat for separately run stages")
    parser.add_argument("--output", required=True, help="New SVG/PDF/PNG file")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for source in args.scope_eval:
        with Path(source).open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows or any(row["split"] != "test" for row in rows):
        raise ValueError("L7 figure requires test-scenario scope_eval.csv")
    keys = [(r["scenario_id"], r["method_id"], r["scope_variant"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate scenario/method/scope result")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    colors = {"zero": "#2563eb", "default": "#d97706", "full_tail": "#15803d"}
    symbols = {"AD-HG-DAPPO": "o", "S_BASE": "x"}
    for stage, ax in zip((1, 2, 3), axes):
        stage_rows = [r for r in rows if int(r["stage"]) == stage]
        n = len({r["scenario_id"] for r in stage_rows})
        for method, marker in symbols.items():
            for variant, color in colors.items():
                group = [r for r in stage_rows if r["method_id"] == method and r["scope_variant"] == variant]
                valid = [r for r in group if r["completed"].lower() == "true"]
                if valid:
                    ax.scatter([float(r["mean_moved_old_per_event"]) for r in valid],
                               [float(r["final_cmax"]) for r in valid],
                               color=color, marker=marker, s=39, alpha=.75,
                               label=f"{method} / {variant} ({len(valid)}/{len(group)})")
        ax.set(title=f"Stage {stage} — {n} test scenarios", xlabel="Old batches moved per event")
        ax.grid(alpha=.22)
        if stage == 1:
            ax.set_ylabel("Final Cmax (min)")
        if stage_rows:
            ax.legend(fontsize=6.8, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
