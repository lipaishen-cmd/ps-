"""DAPPO-Expand (L11 C2 消融)：Stage 2 独立训练入口。

正式预算为 6 个中规模混合负荷场景各 300 episode，共 1800 episode、
300 次 PPO 更新。可在 IDE 中直接运行本文件。

切换随机种子：只改下面 SELECTED_SEED 这一个数字（1-5），然后右键 Run。
命令行 --seed 依然可用且优先级更高：python train_stage2.py --seed 2
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
for _p in (str(_THIS_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.seed_choice import apply_selected_seed
from train_dappo_expand import main
SELECTED_SEED = 1


if __name__ == "__main__":
    apply_selected_seed(SELECTED_SEED)
    main(fixed_stage=2)
