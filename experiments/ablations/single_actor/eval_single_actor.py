"""Evaluate the proposed method against the sole single-Actor ablation."""

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (THIS_DIR, EXPERIMENT_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from single_actor_eval import execute_c2_ablation
from configs.seed_choice import DEFAULT_FORMAL_SEED, FORMAL_SEEDS, apply_selected_seed

SELECTED_SEED = 1


if __name__ == "__main__":
    apply_selected_seed(SELECTED_SEED)
    parser = argparse.ArgumentParser(
        description="评测唯一 C2 消融：AD-HG-DAPPO vs. AD-HG-SAPPO"
    )
    parser.add_argument("--stage", default="1", choices=("1", "2", "3", "all"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--seed", type=int, choices=FORMAL_SEEDS, default=DEFAULT_FORMAL_SEED
    )
    parser.add_argument("--proposed-checkpoint", default=None)
    parser.add_argument("--single-actor-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.stage == "all" and any(
        (args.proposed_checkpoint, args.single_actor_checkpoint, args.output_dir)
    ):
        parser.error("--stage all 不能与单一路径覆盖参数同时使用。")

    stages = (1, 2, 3) if args.stage == "all" else (int(args.stage),)
    for stage in stages:
        proposed_checkpoint = args.proposed_checkpoint or (
            PROJECT_ROOT
            / "result"
            / "主算法"
            / f"stage{stage}"
            / f"seed_{args.seed}"
            / f"stage{stage}_best.pt"
        )
        single_checkpoint = args.single_actor_checkpoint or (
            PROJECT_ROOT
            / "result"
            / "消融实验"
            / f"stage{stage}"
            / f"seed_{args.seed}"
            / f"stage{stage}_best.pt"
        )
        execute_c2_ablation(
            proposed_checkpoint=str(proposed_checkpoint),
            single_actor_checkpoint=str(single_checkpoint),
            split=args.split,
            stage=stage,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir,
        )
