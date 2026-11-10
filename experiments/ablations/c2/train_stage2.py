"""AD-HG-SAPPO 消融：Stage 2 独立训练入口。

正式预算为 6 个中规模混合负荷场景各 300 episode，共 1800 episode、
300 次 PPO 更新。可在 IDE 中直接运行本文件。

切换随机种子：只改下面 SELECTED_SEED 这一个数字（1-5），然后右键 Run。
命令行 --seed 依然可用且优先级更高：python train_stage2.py --seed 3
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
for _p in (str(_THIS_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.experiment_protocol import FORMAL_SEEDS, validate_formal_seed
from train_single_actor import main

# ===== 这里改 seed：1、2、3、4、5 任选一个（必须与主算法同 Stage 一致） =====
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
    main(fixed_stage=2)
