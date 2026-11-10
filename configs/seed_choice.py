"""随机种子选择与校验模块（对应全项目统一正式实验协议）。"""

from __future__ import annotations

from .experiment_protocol import (
    DEFAULT_FORMAL_SEED,
    FORMAL_SEED_COUNT,
    FORMAL_SEED_SPEC,
    FORMAL_SEEDS,
    validate_formal_seed,
)

__all__ = [
    "DEFAULT_FORMAL_SEED",
    "FORMAL_SEED_COUNT",
    "FORMAL_SEED_SPEC",
    "FORMAL_SEEDS",
    "validate_formal_seed",
]
