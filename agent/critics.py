"""
critics.py
==========
双时间尺度状态价值 Critic 网络（规范第 12 章、第 18.3 节第七阶段）

功能说明：
  1. SharedValueCritic: 用统一定长输入同时评估宏观/微观状态价值。
  2. 两个 Actor 仍分开训练；共享 Critic 接收宏观长程和微观步级回报监督。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedValueCritic(nn.Module):
    """
    Manager/Worker 共用的状态价值网络。

    两个 Actor 仍完全独立；共享 Critic 通过统一的定长输入工作：
    Manager 时刻没有当前子批，``h_batch`` 用零向量补齐；Worker 时刻传入
    当前子批嵌入。这样价值网络参数只实例化一份，同时保留两种状态语义。
    """

    def __init__(
        self,
        d_g: int = 128,
        d_order: int = 64,
        d_batch: int = 64,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.d_batch = d_batch
        self.net = nn.Sequential(
            nn.Linear(d_g + d_order + d_batch, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        g: torch.Tensor,
        h_order: torch.Tensor,
        h_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        ``h_batch=None`` 表示 Manager 宏观决策时刻；否则表示 Worker 微观时刻。
        ``g``/``h_order`` 支持形状 ``[B, d]``，单样本也保持批维度。
        """
        if h_batch is None:
            h_batch = torch.zeros(
                (g.shape[0], self.d_batch),
                dtype=g.dtype,
                device=g.device,
            )
        if h_batch.ndim == 1:
            h_batch = h_batch.unsqueeze(0)
        inp = torch.cat([g, h_order, h_batch], dim=-1)
        return self.net(inp)
