"""PCB 钻孔车间两级批调度强化学习模块。"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "PCBBatchHGNN": (".batch_hgnn", "PCBBatchHGNN"),
    "SharedValueCritic": (".critics", "SharedValueCritic"),
    "DynamicEventManager": (".dynamic_event_manager", "DynamicEventManager"),
    "DynamicMultiScenarioTrainer": (".dynamic_multi_scenario_trainer", "DynamicMultiScenarioTrainer"),
    "DynamicTwoAgentRunner": (".dynamic_two_agent_runner", "DynamicTwoAgentRunner"),
    "PCBGraphBuilder": (".graph_builder", "PCBGraphBuilder"),
    "HPPOTrainer": (".hppo_trainer", "HPPOTrainer"),
    "ManagerPolicy": (".manager_policy", "ManagerPolicy"),
    "WorkerPolicy": (".worker_policy", "WorkerPolicy"),
    "ManagerRolloutBuffer": (".rollout_buffer", "ManagerRolloutBuffer"),
    "WorkerRolloutBuffer": (".rollout_buffer", "WorkerRolloutBuffer"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = importlib.import_module(module_name, package=__name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
