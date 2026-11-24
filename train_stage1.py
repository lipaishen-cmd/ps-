"""PCB 动态重调度双 Actor PPO: Stage 1 独立训练脚本。

规模特征：小规模低负荷组（14~20 工单，6 个算例循环 300 次，共 1800 轮，s1~s6 完成一轮循环更新一次 PPO（rollout_scenarios = 6，共 300 次 PPO 更新））
验证集：val_000, val_001
输出模型：result/主算法/stage1/seed_S/stage1_best.pt

切换随机种子：只改下面 SELECTED_SEED 这一个数字（1-5），然后在 PyCharm / VS Code 里右键 Run。
命令行 --seed 依然可用且优先级更高：python train_stage1.py --seed 3 --fresh-start
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.seed_choice import apply_selected_seed
from train import run_stage

SELECTED_SEED = 1


if __name__ == "__main__":
    apply_selected_seed(SELECTED_SEED)
    run_stage(stage=1, extra_args=sys.argv[1:])
