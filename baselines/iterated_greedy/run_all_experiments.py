"""Batch runner for ALL 5s IG experiments across Stage 1, Stage 2, and Stage 3.

- Stage 1: 12 test scenarios with '5s' budget (3 restarts each)
- Stage 2: 12 test scenarios with '5s' budget (3 restarts each)
- Stage 3: 12 test scenarios with '5s' budget (3 restarts each)
- Total: 36 test scenarios x 3 restarts = 108 runs (100% 5s budget).
- Automatically merges all results into combined CSV files upon completion.
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.iterated_greedy.run_ig import main as run_ig_main, write_csv


def merge_csvs(csv_paths: list[Path], output_path: Path):
    all_rows = []
    for p in csv_paths:
        if p.is_file():
            with p.open(encoding="utf-8-sig", newline="") as stream:
                all_rows.extend(list(csv.DictReader(stream)))
    if all_rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_path, all_rows)
        print(f"Merged {len(all_rows)} rows into {output_path}", flush=True)


def run_all_5s():
    start_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = ROOT / "result/对比算法/ig" / f"all_5s_experiments_{timestamp}"
    base_out.mkdir(parents=True, exist_ok=False)

    s1_out = base_out / "stage1_5s"
    s2_out = base_out / "stage2_5s"
    s3_out = base_out / "stage3_5s"

    stages_plan = [
        ("Stage 1 (5s)", ["--stage", "1", "--split", "test", "--budget", "5s", "--output", str(s1_out)]),
        ("Stage 2 (5s)", ["--stage", "2", "--split", "test", "--budget", "5s", "--output", str(s2_out)]),
        ("Stage 3 (5s)", ["--stage", "3", "--split", "test", "--budget", "5s", "--output", str(s3_out)]),
    ]

    for label, argv in stages_plan:
        print(f"\n=======================================================", flush=True)
        print(f" [START] Running {label} IG experiments...", flush=True)
        print(f" Arguments: {' '.join(argv)}", flush=True)
        print(f"=======================================================\n", flush=True)
        code = run_ig_main(argv)
        if code != 0:
            print(f"[WARNING] {label} finished with exit code {code}", flush=True)

    # Merge results
    print(f"\n=======================================================", flush=True)
    print(f" [MERGE] Merging all 5s stage results...", flush=True)
    print(f"=======================================================\n", flush=True)

    merge_csvs([s1_out / "ig_runs.csv", s2_out / "ig_runs.csv", s3_out / "ig_runs.csv"],
               base_out / "all_stages_ig_runs.csv")
    merge_csvs([s1_out / "ig_summary.csv", s2_out / "ig_summary.csv", s3_out / "ig_summary.csv"],
               base_out / "all_stages_ig_summary.csv")
    merge_csvs([s1_out / "ig_events.csv", s2_out / "ig_events.csv", s3_out / "ig_events.csv"],
               base_out / "all_stages_ig_events.csv")

    elapsed_m = (time.perf_counter() - start_time) / 60.0
    print(f"\n[DONE] All 5s IG experiments completed in {elapsed_m:.2f} minutes!", flush=True)
    print(f"Master artifacts: {base_out}", flush=True)


if __name__ == "__main__":
    run_all_5s()
