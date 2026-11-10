"""PCB 动态重调度双 Actor PPO 分阶段独立训练启动入口（默认全局启用 GPU 运算）。

用法：
    python train.py --stage 1 --seed 1 # 独立训练 Stage 1（小规模低负荷组：6 个算例循环 300 次，共 1800 轮，s1~s6 循环更新一次 PPO，共 300 次更新）
    python train.py --stage 2 --seed 1 # 独立训练 Stage 2（中规模混合负荷组：6 个算例循环 300 次，共 1800 轮，s1~s6 循环更新一次 PPO，共 300 次更新）
    python train.py --stage 3 --seed 1 # 独立训练 Stage 3（大规模高负荷组：6 个算例循环 300 次，共 1800 轮，s1~s6 循环更新一次 PPO，共 300 次更新）

或者快捷位置参数：
    python train.py 1
    python train.py 2
    python train.py 3

也可以直接运行各阶段专用脚本（在 IDE 中右键直接 Run 即可，默认使用 GPU）：
    python train_stage1.py
    python train_stage2.py
    python train_stage3.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 注册项目路径与核心求解器层
_PROJECT_ROOT = Path(__file__).resolve().parent
_AGENT_DIR = _PROJECT_ROOT / "agent"
_ENVIRONMENT_DIR = _PROJECT_ROOT / "environment"

for _p in (str(_PROJECT_ROOT), str(_AGENT_DIR), str(_ENVIRONMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from agent.dynamic_multi_scenario_trainer import main as trainer_main
from configs.experiment_protocol import DEFAULT_FORMAL_SEED, FORMAL_SEEDS


def _ensure_gpu_flag(args_list: list[str]) -> list[str]:
    """确保参数列表中明确包含 GPU/device 设置与 1800 轮循环更新参数。"""
    res = list(args_list)
    has_device = any(arg == "--device" or arg.startswith("--device=") for arg in res)
    if not has_device:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        res.extend(["--device", target_device])
    has_rollout = any(arg == "--rollout-scenarios" or arg.startswith("--rollout-scenarios=") for arg in res)
    if not has_rollout:
        res.extend(["--rollout-scenarios", "6"])
    return res


def run_stage(stage: int, extra_args: list[str] | None = None) -> None:
    """运行指定阶段的独立训练，默认启用 GPU。"""
    if stage not in (1, 2, 3):
        raise ValueError(f"阶段编号必须为 1, 2 或 3，收到: {stage}")

    cmd_args = ["--stage-only", str(stage)]
    if extra_args:
        cmd_args.extend(extra_args)

    has_seed = any(arg == "--seed" or arg.startswith("--seed=") for arg in cmd_args)
    if not has_seed:
        cmd_args.extend(["--seed", str(DEFAULT_FORMAL_SEED)])

    cmd_args = _ensure_gpu_flag(cmd_args)

    if torch.cuda.is_available():
        print(f"[GPU加速已开启] 训练运算设备: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[警告] 未检测到可用 GPU，当前将降级在 CPU 上运行。")

    orig_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + cmd_args
        trainer_main()
    finally:
        sys.argv = orig_argv


def main() -> None:
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="PCB 动态重调度双 Actor PPO 分阶段独立训练入口（默认 GPU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 优先识别首个位置参数阶段编号 (1, 2, 3)，避免 parse_known_args 误吞后续选项的数值
    raw_args = list(sys.argv[1:])
    pos_stage = None
    if raw_args and raw_args[0] in ("1", "2", "3"):
        pos_stage = int(raw_args.pop(0))

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=pos_stage,
        help="阶段编号 (1, 2 或 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_dev,
        help=f"训练计算设备 (默认: {default_dev})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=FORMAL_SEEDS,
        default=DEFAULT_FORMAL_SEED,
        help=f"正式随机种子（{list(FORMAL_SEEDS)}，默认 {DEFAULT_FORMAL_SEED}）",
    )
    parser.add_argument("--fresh-start", action="store_true", help="强制从第 1 轮重新训练（忽略历史断点）")
    parser.add_argument("--resume", action="store_true", help="从历史断点恢复训练")

    args, unknown = parser.parse_known_args(raw_args)

    chosen_stage = args.stage
    if chosen_stage is None:
        parser.print_help()
        print("\n[错误] 请指定要独立训练的阶段编号 (1, 2 或 3)！")
        print("例如：")
        print("    python train.py --stage 1")
        print("    python train.py 1")
        print("    python train_stage1.py\n")
        sys.exit(1)

    forward_args: list[str] = ["--device", args.device]
    forward_args.extend(["--seed", str(args.seed)])
    if args.fresh_start:
        forward_args.append("--fresh-start")
    if args.resume:
        forward_args.append("--resume")
    forward_args.extend(unknown)

    run_stage(stage=chosen_stage, extra_args=forward_args)


if __name__ == "__main__":
    main()
