"""外部对比 RL 算法：Stage 1 独立训练入口（小规模低负荷组，默认 1800 轮匹配预算 + GPU）。

切换算法：改下面 SELECTED_METHOD（"am_a2c" / "transformer_a3c" / "etdqn" / "iddqn" / "all"）。
切换种子：改下面 SELECTED_SEED（1-5），然后直接在 PyCharm / VS Code 右键 Run。
命令行参数依然可用且优先级更高：python baselines/external_rl/train_stage1.py --method etdqn --seed 2
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.external_rl.stage_launcher import run_stage_script
SELECTED_METHOD = "am_a2c"  # 可选: "am_a2c", "transformer_a3c", "etdqn", "iddqn", "all"
SELECTED_SEED = 1           # 可选: 1, 2, 3, 4, 5


if __name__ == "__main__":
    run_stage_script(
        fixed_stage=1,
        selected_seed=SELECTED_SEED,
        selected_method=SELECTED_METHOD,
    )
