"""ATTENTION-DDQN-PCB：Stage 3 独立训练入口（大规模高负荷高扰动组）。

切换随机种子：只改下面 SELECTED_SEED 这一个数字（1-5），然后右键 Run。
命令行 --seed 依然可用且优先级更高：python baselines/external_rl/iddqn/train_stage3.py --seed 3
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.external_rl.stage_launcher import run_stage_script
SELECTED_SEED = 1


if __name__ == "__main__":
    run_stage_script(fixed_stage=3, fixed_method="iddqn", selected_seed=SELECTED_SEED)
