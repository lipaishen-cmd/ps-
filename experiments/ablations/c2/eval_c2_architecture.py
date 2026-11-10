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

from ablation_c2_architecture import execute_c2_ablation
from configs.experiment_protocol import DEFAULT_FORMAL_SEED, FORMAL_SEEDS, validate_formal_seed

# ===== 这里改 seed：1、2、3、4、5 任选一个（必须与训练时的 Seed 一致）=====
SELECTED_SEED = 1
# =========================================================================


def _apply_selected_seed() -> int:
    """命令行 --seed 优先；否则把 SELECTED_SEED 注入命令行。"""
    argv = sys.argv[1:]
    cli_raw = None
    for index, arg in enumerate(argv):
        if arg == "--seed" and index + 1 < len(argv):
            cli_raw = argv[index + 1]
        elif arg.startswith("--seed="):
            cli_raw = arg.split("=", 1)[1]

    try:
        if cli_raw is None:
            chosen = validate_formal_seed(SELECTED_SEED)
            sys.argv = [sys.argv[0], *argv, "--seed", str(chosen)]
            print(f"[种子选择] 使用脚本顶部 SELECTED_SEED = {chosen}（正式范围 {list(FORMAL_SEEDS)}）。")
        else:
            try:
                cli_value = int(cli_raw)
            except ValueError as exc:
                raise ValueError(f"--seed 的数值无法解析：{cli_raw!r}。") from exc
            chosen = validate_formal_seed(cli_value)
            print(f"[种子选择] 使用命令行指定的 Seed {chosen}（覆盖脚本顶部 SELECTED_SEED）。")
    except ValueError as exc:
        raise SystemExit(
            f"[种子选择] {exc}\n"
            f"请把本文件顶部的 SELECTED_SEED 改成 {list(FORMAL_SEEDS)} 之一。"
        ) from exc
    return chosen


if __name__ == "__main__":
    _apply_selected_seed()
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
