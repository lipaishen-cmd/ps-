"""全项目统一正式实验协议。"""

from __future__ import annotations

from typing import Final

FORMAL_SEEDS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)
DEFAULT_FORMAL_SEED: Final[int] = FORMAL_SEEDS[0]
FORMAL_SEED_SPEC: Final[str] = "1-5"
FORMAL_SEED_COUNT: Final[int] = len(FORMAL_SEEDS)


def validate_formal_seed(seed: int) -> int:
    value = int(seed)
    if value not in FORMAL_SEEDS:
        raise ValueError(
            f"正式实验 seed 必须为 {list(FORMAL_SEEDS)} 之一，收到 {seed!r}。"
        )
    return value
