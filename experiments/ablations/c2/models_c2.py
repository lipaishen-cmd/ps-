"""Public model exports for the only C2 architecture ablation."""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
_ENVIRONMENT_DIR = _PROJECT_ROOT / "environment"

for _p in (str(_THIS_DIR), str(_PROJECT_ROOT), str(_AGENT_DIR), str(_ENVIRONMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from single_actor_policy import (
    PLACEMENT_ACTION,
    SPLIT_ACTION,
    SingleActorPolicy,
    matched_hidden_dim,
    reference_dual_actor_parameter_count,
)

__all__ = [
    "SPLIT_ACTION",
    "PLACEMENT_ACTION",
    "SingleActorPolicy",
    "matched_hidden_dim",
    "reference_dual_actor_parameter_count",
]
