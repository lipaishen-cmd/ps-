"""
worker_policy.py
================
下层微观子批插槽决策策略网络 WorkerPolicy（规范第 9 章、第 18.4 节 Gate 8 核心执行体）

功能说明：
  1. 基于待排子批表征 h_b 与全部候选插槽表征 h_l 计算匹配评分。
  2. 施加微观动作掩码 M^A：屏蔽机型不兼容或越界的非法 (m, l) 插槽。
  3. 通过 Softmax 输出微观指派概率分布 pi^A(a_t | s_t^A)。
  4. 确保非法插槽动作的选择概率置为 0.0。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkerPolicy(nn.Module):
    """
    Worker 机器-插槽决策策略网络。
    """

    def __init__(
        self,
        d_batch: int = 64,
        d_slot: int = 64,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.matcher = nn.Sequential(
            nn.Linear(d_batch + d_slot, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        h_batch: torch.Tensor,                                # [1, d_batch]
        h_slots: torch.Tensor,                                # [N_slots, d_slot]
        slot_keys: List[Tuple[str, int]],                     # 每一个插槽对应的 (machine_id, slot_pos)
        valid_slots: List[Tuple[str, int]],                   # 合法动作列表
        order_id: Optional[str] = None,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        前向计算微观动作概率分布与掩码后 logits。
        
        ``order_id`` 是微观策略的可选轨迹标识。标准 WorkerPolicy 的动作只由
        当前子批、插槽表征和硬掩码决定，因此不将该标识作为网络输入；保留此
        关键字参数仅用于保留轨迹中的订单标识，不进入策略网络输入。

        返回：
          (dist: Categorical, masked_logits: [N_slots])
        """
        device = h_batch.device
        n_slots = h_slots.size(0)
        if n_slots == 0:
            raise ValueError("No slots provided to WorkerPolicy!")

        b_exp = h_batch.expand(n_slots, -1)
        pairs = torch.cat([b_exp, h_slots], dim=-1)
        raw_logits = self.matcher(pairs).squeeze(-1) # [N_slots]

        # 施加微观硬掩码
        valid_set = set(valid_slots)
        mask_list = [(key in valid_set) for key in slot_keys]
        mask_tensor = torch.tensor(mask_list, dtype=torch.bool, device=device)

        masked_logits = torch.where(mask_tensor, raw_logits, torch.tensor(-1e9, device=device))

        dist = torch.distributions.Categorical(logits=masked_logits)
        return dist, masked_logits
