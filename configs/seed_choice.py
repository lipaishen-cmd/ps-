"""随机种子选择与校验模块（对应全项目统一正式实验协议）。"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from .experiment_protocol import (
    DEFAULT_FORMAL_SEED,
    FORMAL_SEED_COUNT,
    FORMAL_SEED_SPEC,
    FORMAL_SEEDS,
    validate_formal_seed,
)


def apply_selected_seed(
    selected_seed: int,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """命令行 ``--seed`` 优先；否则将脚本顶部的 ``SELECTED_SEED`` 注入 ``sys.argv``。"""
    args_list = list(sys.argv[1:] if argv is None else argv)
    cli_raw: Optional[str] = None
    for index, arg in enumerate(args_list):
        if arg == "--seed" and index + 1 < len(args_list):
            cli_raw = args_list[index + 1]
        elif arg.startswith("--seed="):
            cli_raw = arg.split("=", 1)[1]

    try:
        if cli_raw is None:
            chosen = validate_formal_seed(selected_seed)
            sys.argv = [sys.argv[0], *args_list, "--seed", str(chosen)]
            print(
                f"[种子选择] 使用脚本顶部 SELECTED_SEED = {chosen}（正式范围 {list(FORMAL_SEEDS)}）。"
            )
        else:
            try:
                cli_value = int(cli_raw)
            except ValueError as exc:
                raise ValueError(f"--seed 的数值无法解析：{cli_raw!r}。") from exc
            chosen = validate_formal_seed(cli_value)
            print(
                f"[种子选择] 使用命令行指定的 Seed {chosen}（覆盖脚本顶部 SELECTED_SEED）。"
            )
    except ValueError as exc:
        raise SystemExit(
            f"[种子选择] {exc}\n"
            f"请把本文件顶部的 SELECTED_SEED 改成 {list(FORMAL_SEEDS)} 之一。"
        ) from exc
    return chosen


__all__ = [
    "DEFAULT_FORMAL_SEED",
    "FORMAL_SEED_COUNT",
    "FORMAL_SEED_SPEC",
    "FORMAL_SEEDS",
    "apply_selected_seed",
    "validate_formal_seed",
]
