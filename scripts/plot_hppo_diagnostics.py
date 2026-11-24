"""Render audit-friendly HPPO curves from one persisted training JSON log.

The script never writes into the input run directory.  It reads a saved JSON
log and writes PNG/PDF figures plus an index JSON to an explicitly selected
output directory. Only records explicitly marked ``ppo_updated`` are eligible
for PPO/value-loss curves, and validation curves use ``raw_actor_cmax`` only.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_COLORS = {
    "raw": "#0072B2",
    "manager": "#D55E00",
    "worker": "#CC79A7",
    "cmax": "#0072B2",
    "reward": "#E69F00",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _safe_stem(value: str) -> str:
    stem = Path(value).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "scenario"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _save_pair(fig: plt.Figure, out_dir: Path, stem: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _update_records(history: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return optimizer steps explicitly marked by the current log contract."""
    records: List[Dict[str, Any]] = []
    for row in history:
        loss = row.get("loss")
        if bool(row.get("ppo_updated")) and isinstance(loss, dict) and loss:
            records.append(row)
    return records


def _plot_value_losses(history: Sequence[Dict[str, Any]], out_dir: Path) -> Tuple[List[Path], int]:
    updates = _update_records(history)
    episodes: List[int] = []
    manager_losses: List[float] = []
    worker_losses: List[float] = []
    for row in updates:
        loss = row["loss"]
        m_loss = _as_float(loss.get("m_val_loss"))
        w_loss = _as_float(loss.get("w_val_loss"))
        if m_loss is None and w_loss is None:
            continue
        episodes.append(int(row.get("episode", len(episodes) + 1)))
        manager_losses.append(float("nan") if m_loss is None else m_loss)
        worker_losses.append(float("nan") if w_loss is None else w_loss)

    if not episodes:
        return [], 0
    fig, axis = plt.subplots(figsize=(6.6, 3.6))
    axis.plot(episodes, manager_losses, color=_COLORS["manager"], linewidth=1.8, marker="o", markersize=3, label="Manager value loss")
    axis.plot(episodes, worker_losses, color=_COLORS["worker"], linewidth=1.8, marker="s", markersize=3, label="Worker value loss")
    axis.set_xlabel("Training episode of PPO update")
    axis.set_ylabel("Value loss")
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return _save_pair(fig, out_dir, "value_losses_by_ppo_update"), len(episodes)


def _plot_training_scenarios(history: Sequence[Dict[str, Any]], out_dir: Path) -> List[Path]:
    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in history:
        filename = row.get("train_file")
        if isinstance(filename, str) and filename:
            by_scenario[filename].append(row)

    written: List[Path] = []
    for filename in sorted(by_scenario):
        rows = by_scenario[filename]
        episode = [int(row.get("episode", idx + 1)) for idx, row in enumerate(rows)]
        cmax = [_as_float(row.get("final_cmax")) for row in rows]
        reward = [_as_float(row.get("mean_reward")) for row in rows]
        if not any(value is not None for value in cmax):
            continue

        fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.6, 4.7), sharex=True)
        top.plot(episode, cmax, color=_COLORS["cmax"], linewidth=1.5)
        top.set_ylabel("Cmax (min)")
        top.grid(alpha=0.25, linewidth=0.6)
        bottom.plot(episode, reward, color=_COLORS["reward"], linewidth=1.35)
        bottom.set_xlabel("Global training episode")
        bottom.set_ylabel("Mean event reward")
        bottom.grid(alpha=0.25, linewidth=0.6)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.98, hspace=0.14)
        written.extend(_save_pair(fig, out_dir, f"train_{_safe_stem(filename)}"))
    return written


def _plot_validation_scenarios(validation_history: Sequence[Dict[str, Any]], out_dir: Path) -> List[Path]:
    by_scenario: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for record in validation_history:
        episode = int(record.get("episode", 0))
        details = record.get("details", [])
        if not isinstance(details, list):
            continue
        for detail in details:
            if isinstance(detail, dict) and isinstance(detail.get("file"), str):
                by_scenario[detail["file"]].append((episode, detail))

    written: List[Path] = []
    for filename in sorted(by_scenario):
        points = by_scenario[filename]
        episodes = [point[0] for point in points]
        raw = [_as_float(detail.get("raw_actor_cmax")) for _, detail in points]
        if not any(value is not None for value in raw):
            continue
        fig, axis = plt.subplots(figsize=(6.6, 3.6))
        axis.plot(episodes, raw, color=_COLORS["raw"], linewidth=1.8, marker="o", markersize=3, label="Raw Actor")
        axis.set_xlabel("Validation episode")
        axis.set_ylabel("Cmax (min)")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(frameon=False, ncol=1)
        fig.tight_layout()
        written.extend(_save_pair(fig, out_dir, f"validation_{_safe_stem(filename)}"))
    return written


def generate_figures(log_path: Path, out_dir: Path) -> Dict[str, Any]:
    """Create only per-scenario and value-loss diagnostics from one run log."""
    with log_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("训练日志根节点必须是 JSON 对象。")
    history = report.get("training_history", [])
    validation_history = report.get("validation_history", [])
    if not isinstance(history, list) or not isinstance(validation_history, list):
        raise ValueError("训练日志缺少列表型 training_history 或 validation_history。")

    _configure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    value_files, update_count = _plot_value_losses(history, out_dir)
    train_files = _plot_training_scenarios(history, out_dir)
    validation_files = _plot_validation_scenarios(validation_history, out_dir)
    manifest = {
        "input_log": str(log_path.resolve()),
        "training_records": len(history),
        "validation_records": len(validation_history),
        "ppo_update_points_plotted": update_count,
        "value_loss_figures": [path.name for path in value_files],
        "per_training_scenario_figures": [path.name for path in train_files],
        "per_validation_scenario_figures": [path.name for path in validation_files],
        "notes": [
            "PPO/value-loss figures contain actual optimizer updates only.",
            "Validation figures use the current raw_actor_cmax field only.",
        ],
    }
    manifest_path = out_dir / "figure_index.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    manifest["index"] = manifest_path.name
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-scenario and PPO value-loss HPPO diagnostics.")
    parser.add_argument("--log-json", required=True, help="Persisted multi_scenario_train_log.json to read only")
    parser.add_argument("--out-dir", required=True, help="New or existing directory for figures and figure_index.json")
    args = parser.parse_args()
    manifest = generate_figures(Path(args.log_json), Path(args.out_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
