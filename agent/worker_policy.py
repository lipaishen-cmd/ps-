"""下层子批插槽指派策略网络 WorkerPolicy，输出满足约束的候选插槽概率分布。"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkerPolicy(nn.Module):
    """
    Worker 机器-插槽决策策略网络（双塔投影相加 + Tanh 激活，与 Manager/SingleActor 结构对齐）。
    """

    def __init__(
        self,
        d_batch: int = 64,
        d_slot: int = 64,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.batch_proj = nn.Linear(d_batch, d_hidden)
        self.slot_proj = nn.Linear(d_slot, d_hidden)
        self.head = nn.Sequential(
            nn.Tanh(),
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

        if h_batch.is_inference():
            h_batch = h_batch.clone()
        if h_slots.is_inference():
            h_slots = h_slots.clone()

        batch_emb = self.batch_proj(h_batch).expand(n_slots, -1)
        slot_emb = self.slot_proj(h_slots)
        raw_logits = self.head(batch_emb + slot_emb).squeeze(-1) # [N_slots]

        # 施加微观硬掩码
        valid_set = set(valid_slots)
        mask_list = [(key in valid_set) for key in slot_keys]
        mask_tensor = torch.tensor(mask_list, dtype=torch.bool, device=device)

        masked_logits = torch.where(mask_tensor, raw_logits, torch.tensor(-1e9, device=device))

        dist = torch.distributions.Categorical(logits=masked_logits)
        return dist, masked_logits
