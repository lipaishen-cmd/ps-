"""外部对比 RL 算法测试集评估入口。"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.external_rl import METHODS
from baselines.external_rl.run import main as run_main

# 评估配置
SELECTED_METHOD = "all"   # 可选: "am_a2c", "transformer_a3c", "etdqn", "iddqn", "all"
SELECTED_STAGE = 1        # 可选: 1, 2, 3
SELECTED_SEED = 1         # 可选: 1, 2, 3, 4, 5
SELECTED_SPLIT = "test"   # 可选: "val", "test"


if __name__ == "__main__":
    targets = tuple(METHODS.keys()) if SELECTED_METHOD == "all" else (SELECTED_METHOD,)
    for method in targets:
        run_main([
            "--method", method,
            "--mode", "evaluate",
            "--stage", str(SELECTED_STAGE),
            "--seed", str(SELECTED_SEED),
            "--split", SELECTED_SPLIT,
            "--fresh-start",
        ])
