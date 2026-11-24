"""Unified transactional rollout buffer for the single-Actor ablation."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class SingleActorRolloutBuffer:
    """Store split and placement decisions in one chronological trajectory."""

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reset()

    def reset(self) -> None:
        self.transitions: List[Dict[str, Any]] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []

    def add(self, transition: Dict[str, Any]) -> None:
        required = {"decision_type", "action_idx", "reward", "done", "log_prob", "value"}
        missing = required.difference(transition)
        if missing:
            raise ValueError(f"SingleActor transition is missing fields: {sorted(missing)}")
        item = dict(transition)
        item["reward"] = float(item["reward"])
        item["done"] = bool(item["done"])
        for key in ("g", "h_order", "h_batch", "h_slots", "log_prob", "value"):
            value = item.get(key)
            if isinstance(value, torch.Tensor):
                item[key] = value.detach().cpu()
        self.transitions.append(item)

    def add_terminal_reward(self, reward: float) -> None:
        if not self.transitions:
            raise RuntimeError("Cannot assign terminal reward to an empty rollout buffer.")
        self.transitions[-1]["reward"] += float(reward)

    def mark_last_done(self) -> None:
        if self.transitions:
            self.transitions[-1]["done"] = True

    def truncate(self, size: int) -> None:
        """显式恢复/测试工具；事件失败不会删除此前已提交的合法转移。"""
        size = int(size)
        if size < 0 or size > len(self.transitions):
            raise ValueError(f"Invalid SingleActor buffer truncate size: {size}")
        del self.transitions[size:]
        self.returns = []
        self.advantages = []

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        n = len(self.transitions)
        self.returns = [0.0] * n
        self.advantages = [0.0] * n
        gae = 0.0
        for index in reversed(range(n)):
            item = self.transitions[index]
            done = bool(item["done"])
            current_value = float(torch.as_tensor(item["value"]).reshape(-1)[0].item())
            if done:
                next_value = 0.0
            elif index == n - 1:
                next_value = float(last_value)
            else:
                next_value = float(
                    torch.as_tensor(self.transitions[index + 1]["value"]).reshape(-1)[0].item()
                )
            delta = item["reward"] + (0.0 if done else self.gamma * next_value) - current_value
            gae = delta + (0.0 if done else self.gamma * self.gae_lambda * gae)
            self.advantages[index] = gae
            self.returns[index] = gae + current_value

    def decision_type_counts(self) -> Dict[str, int]:
        """Return auditable split/placement counts without changing sampling weights."""
        counts = {"split": 0, "placement": 0}
        for item in self.transitions:
            decision_type = str(item["decision_type"])
            if decision_type not in counts:
                raise ValueError(f"Unsupported SingleActor decision type: {decision_type!r}")
            counts[decision_type] += 1
        return counts

    def __len__(self) -> int:
        return len(self.transitions)

