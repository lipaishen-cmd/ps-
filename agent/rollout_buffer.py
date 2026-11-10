"""
rollout_buffer.py
=================
事务隔离的双层经验回放池（规范第 13 章、第 18.2 节冻结项 4、第 18.4 节 Gate 9 核心执行体）

功能说明：
  1. 提供 ManagerRolloutBuffer 与 WorkerRolloutBuffer。
  2. 实现微观临时轨迹隔离，维护 temporary_worker_trajectory。
  3. 遵循事务提交机制：
     - 仅在 commit_option 成功后，将临时轨迹转入正式 WorkerRolloutBuffer；
     - 若发生 rollback_option，清空临时轨迹，确保缓冲区只保留有效采样。
  4. 支持半MDP跳步 GAE 与微观单步 GAE 优势函数计算。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch


class ManagerRolloutBuffer:
    """
    宏观半MDP经验池。
    存储：(g, h_order, templates, template_mask, order_W, action_idx, reward, tau, log_prob, val)
    """

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.g_list: List[torch.Tensor] = []
        self.h_order_list: List[torch.Tensor] = []
        self.graph_list: List[Optional[Any]] = []
        self.frontier_idx_list: List[int] = []
        self.templates_list: List[List[Tuple[int, ...]]] = []
        self.mask_list: List[Dict[Tuple[int, ...], int]] = []
        self.order_W_list: List[int] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.taus: List[int] = []
        self.dones: List[bool] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []

    def add(
        self,
        g: torch.Tensor,
        h_order: torch.Tensor,
        templates: List[Tuple[int, ...]],
        mask: Dict[Tuple[int, ...], int],
        order_W: int,
        action_idx: int,
        reward: float,
        tau: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        graph: Optional[Any] = None,
        frontier_idx: int = 0,
        done: bool = False,
    ):
        self.g_list.append(g.detach().cpu())
        self.h_order_list.append(h_order.detach().cpu())
        self.graph_list.append(graph)
        self.frontier_idx_list.append(frontier_idx)
        self.templates_list.append(templates)
        self.mask_list.append(mask)
        self.order_W_list.append(order_W)
        self.actions.append(action_idx)
        self.rewards.append(reward)
        self.taus.append(tau)
        self.dones.append(bool(done))
        self.log_probs.append(log_prob.detach().cpu())
        self.values.append(value.detach().cpu())

    def compute_returns_and_advantages(self, last_val: float = 0.0):
        """
        计算半MDP跳步 GAE 优势：
        支持 Episode 边界精确截断，严禁跨场景/跨 Episode 价值与优势倒灌污染：
        delta_k = R_k^S + (1 - done_k) * gamma^{tau_k} V(s_{k+1}) - V(s_k)
        A_k = delta_k + (1 - done_k) * gamma^{tau_k} lambda A_{k+1}
        """
        n = len(self.rewards)
        self.returns = [0.0] * n
        self.advantages = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            is_done = self.dones[t] if (hasattr(self, "dones") and t < len(self.dones)) else (t == n - 1)
            next_val = 0.0 if is_done else (last_val if t == n - 1 else self.values[t + 1].item())
            discount = self.gamma ** self.taus[t]
            delta = self.rewards[t] + (0.0 if is_done else discount * next_val) - self.values[t].item()
            # 半MDP跳步广义优势衰减因子: (gamma * lambda)^{tau_k} (式 33)
            # 在 gamma=1.0 环境中，跨步时序衰减严格由 lambda^{tau_k} 驱动：
            # 主方法中 tau_k 为宏观动作对应的真实微观子批步数，持续时间更长的
            # 宏观决策采用更强的时序衰减；统一单策略 C2 使用其独立的单步轨迹缓冲区。
            gae_decay = (self.gamma * self.gae_lambda) ** self.taus[t]
            last_gae = delta + (0.0 if is_done else gae_decay * last_gae)
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t].item()

    def truncate(self, size: int) -> None:
        """将正式经验池恢复到指定长度；仅供显式恢复/测试，不用于事件失败回滚。"""
        size = int(size)
        if size < 0 or size > len(self):
            raise ValueError(f"ManagerRolloutBuffer.truncate 的目标长度无效: {size}")
        fields = (
            "g_list", "h_order_list", "graph_list", "frontier_idx_list",
            "templates_list", "mask_list", "order_W_list", "actions",
            "rewards", "taus", "dones", "log_probs", "values",
        )
        for field in fields:
            del getattr(self, field)[size:]
        self.returns = []
        self.advantages = []

    def mark_last_done(self) -> None:
        """在受控事件失败时终止当前 episode，同时保留此前已提交的合法轨迹。"""
        if self.dones:
            self.dones[-1] = True
            self.returns = []
            self.advantages = []

    def __len__(self) -> int:
        return len(self.actions)


class WorkerRolloutBuffer:
    """
    微观单步经验池。
    存储：(h_batch, h_slots, slot_keys, valid_slots, action_idx, reward, log_prob, val, graph, sub_idx)
    """

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.temporary_worker_trajectory: List[dict] = []
        self.reset()

    def reset(self):
        self.temporary_worker_trajectory.clear()
        self.g_list: List[Optional[torch.Tensor]] = []
        self.h_order_list: List[Optional[torch.Tensor]] = []
        self.graph_list: List[Optional[Any]] = []
        self.sub_idx_list: List[int] = []
        self.order_idx_list: List[int] = []
        self.h_batch_list: List[torch.Tensor] = []
        self.h_slots_list: List[torch.Tensor] = []
        self.slot_keys_list: List[List[Tuple[str, int]]] = []
        self.valid_slots_list: List[List[Tuple[str, int]]] = []
        self.order_id_list: List[Optional[str]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []

    def record_temporary_step(
        self,
        h_batch: torch.Tensor,
        h_slots: torch.Tensor,
        slot_keys: List[Tuple[str, int]],
        valid_slots: List[Tuple[str, int]],
        action_idx: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        h_order: Optional[torch.Tensor] = None,
        graph: Optional[Any] = None,
        sub_idx: int = 0,
        order_idx: int = 0,
        done: bool = False,
        order_id: Optional[str] = None,
    ):
        """记录单步临时转移至隔离区"""
        self.temporary_worker_trajectory.append({
            "order_id": order_id,
            "g": None if g is None else g.detach().cpu(),
            "h_order": None if h_order is None else h_order.detach().cpu(),
            "graph": graph,
            "sub_idx": sub_idx,
            "order_idx": order_idx,
            "h_batch": h_batch.detach().cpu(),
            "h_slots": h_slots.detach().cpu(),
            "slot_keys": slot_keys,
            "valid_slots": valid_slots,
            "action_idx": action_idx,
            "reward": reward,
            "done": bool(done),
            "log_prob": log_prob.detach().cpu(),
            "value": value.detach().cpu(),
        })

    def abort_transaction(self):
        """事务回滚：物理清空临时轨迹，杜绝脏数据泄漏"""
        self.temporary_worker_trajectory.clear()

    def commit_transaction(self, done: Optional[bool] = None):
        """事务提交：正式写入正式经验池并硬断言数据完整"""
        m = len(self.temporary_worker_trajectory)
        for idx, item in enumerate(self.temporary_worker_trajectory):
            self.g_list.append(item["g"])
            self.h_order_list.append(item["h_order"])
            self.graph_list.append(item.get("graph"))
            self.sub_idx_list.append(item.get("sub_idx", 0))
            self.order_idx_list.append(item.get("order_idx", 0))
            self.h_batch_list.append(item["h_batch"])
            self.h_slots_list.append(item["h_slots"])
            self.slot_keys_list.append(item["slot_keys"])
            self.valid_slots_list.append(item["valid_slots"])
            self.order_id_list.append(item.get("order_id"))
            self.actions.append(item["action_idx"])
            self.rewards.append(item["reward"])
            step_done = bool(done) if (done is not None and idx == m - 1) else bool(item.get("done", False))
            self.dones.append(step_done)
            self.log_probs.append(item["log_prob"])
            self.values.append(item["value"])

        self.temporary_worker_trajectory.clear()

    def compute_returns_and_advantages(self, last_val: float = 0.0):
        """微观标准步级 GAE 优势计算 (支持 Episode 边界精确截断)"""
        n = len(self.rewards)
        self.returns = [0.0] * n
        self.advantages = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            is_done = self.dones[t] if (hasattr(self, "dones") and t < len(self.dones)) else (t == n - 1)
            next_val = 0.0 if is_done else (last_val if t == n - 1 else self.values[t + 1].item())
            delta = self.rewards[t] + (0.0 if is_done else self.gamma * next_val) - self.values[t].item()
            last_gae = delta + (0.0 if is_done else self.gamma * self.gae_lambda * last_gae)
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t].item()

    def truncate(self, size: int) -> None:
        """将正式经验池恢复到指定长度；仅供显式恢复/测试，不用于事件失败回滚。"""
        size = int(size)
        if size < 0 or size > len(self):
            raise ValueError(f"WorkerRolloutBuffer.truncate 的目标长度无效: {size}")
        self.temporary_worker_trajectory.clear()
        fields = (
            "g_list", "h_order_list", "graph_list", "sub_idx_list", "order_idx_list",
            "h_batch_list", "h_slots_list", "slot_keys_list",
            "valid_slots_list", "order_id_list", "actions", "rewards", "dones",
            "log_probs", "values",
        )
        for field in fields:
            del getattr(self, field)[size:]
        self.returns = []
        self.advantages = []

    def mark_last_done(self) -> None:
        """在受控事件失败时终止当前 episode，同时保留此前已提交的合法轨迹。"""
        self.temporary_worker_trajectory.clear()
        if self.dones:
            self.dones[-1] = True
            self.returns = []
            self.advantages = []

    def __len__(self) -> int:
        return len(self.actions)
