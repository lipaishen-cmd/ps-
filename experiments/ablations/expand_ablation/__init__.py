"""L11 (DAPPO-Expand) C2 ablation: pre-expanding all candidate templates vs on-demand graph construction."""

from __future__ import annotations

from .dappo_expand_trainer import (
    DAPPO_EXPAND_CODE_ID,
    DAPPO_EXPAND_GRAPH_MODE,
    DAPPO_EXPAND_GRAPH_SCHEMA_VERSION,
    DAPPO_EXPAND_METHOD_ID,
    DAPPOExpandTrainer,
    ExpandedTwoAgentRunner,
    InstrumentedTwoAgentRunner,
)
from .expanded_graph_builder import PreExpandedPCBGraphBuilder, TimedOnDemandPCBGraphBuilder

__all__ = [
    "DAPPO_EXPAND_CODE_ID",
    "DAPPO_EXPAND_GRAPH_MODE",
    "DAPPO_EXPAND_GRAPH_SCHEMA_VERSION",
    "DAPPO_EXPAND_METHOD_ID",
    "DAPPOExpandTrainer",
    "ExpandedTwoAgentRunner",
    "InstrumentedTwoAgentRunner",
    "PreExpandedPCBGraphBuilder",
    "TimedOnDemandPCBGraphBuilder",
]
