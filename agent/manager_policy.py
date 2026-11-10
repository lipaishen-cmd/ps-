"""
manager_policy.py
=================
上层宏观拆批决策策略网络 ManagerPolicy（规范第 8 章、第 18.4 节 Gate 8 核心执行体）

功能说明：
  1. 基于 HGNN 全局图表征 g 与当前待排工单表征 h_i，计算候选划分模板的评分。
  2. 施加动作掩码 M^S：将不可行模板的 logit 置为负无穷。
  3. 通过 Softmax 输出概率分布 pi^S(p | s^S)。
  4. 确保被掩码模板的选择概率置为 0.0。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from template_encoding import TEMPLATE_FEATURE_DIM, encode_templates


class ManagerPolicy(nn.Module):
    """
    Manager 拆批决策策略网络。
    """

    def __init__(
        self,
        d_g: int = 128,      # 全局上下文 g 维度
        d_order: int = 64,   # 订单嵌入维度
        d_template: int = TEMPLATE_FEATURE_DIM,
        d_template_node: int = 64,
        d_hidden: int = 64,
    ):
        super().__init__()
        # 双塔投影结构：消除 96 维上下文对 5 维模板特征的过度稀释
        self.ctx_proj = nn.Linear(d_g + d_order, d_hidden)
        self.tpl_proj = nn.Linear(d_template, d_hidden)
        self.tpl_node_proj = nn.Linear(d_template_node, d_hidden)
        self.head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def extract_template_features(
        self,
        templates: List[Tuple[int, ...]],
        order_W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        提取无损模板矩阵 [N_templates, 10]：六维计数直方图与四维辅助统计量。
        """
        del order_W  # demand conservation is validated by the graph/template generator
        return encode_templates(templates, device=device)

    def forward(
        self,
        g: torch.Tensor,                                  # [1, d_g]
        h_order: torch.Tensor,                            # [1, d_order]
        h_templates: torch.Tensor,                        # [N_templates, d_template_node]
        templates: List[Tuple[int, ...]],                 # 候选模板列表
        template_mask: Dict[Tuple[int, ...], int],        # 动作硬掩码 {p: 1/0}
        order_W: int,
        **kwargs,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        前向计算动作概率分布与掩码后 logits。
        
        返回：
          (dist: Categorical, masked_logits: [N_templates])
        """
        device = g.device
        n_templates = len(templates)
        if n_templates == 0:
            raise ValueError("No templates provided to ManagerPolicy!")

        if h_templates.shape[0] != n_templates:
            raise ValueError(
                f"Template-node count {h_templates.shape[0]} != action count {n_templates}."
            )
        tpl_feats = self.extract_template_features(templates, order_W, device)

        # 双塔隐空间投影：上下文与候选模板各占 50% 权重
        ctx = torch.cat([g, h_order], dim=-1) # [1, d_g + d_order]
        ctx_emb = self.ctx_proj(ctx).expand(n_templates, -1) # [N_t, d_hidden]
        tpl_emb = self.tpl_proj(tpl_feats) + self.tpl_node_proj(h_templates)

        raw_logits = self.head(ctx_emb + tpl_emb).squeeze(-1) # [N_templates]

        # 施加硬掩码
        mask_tensor = torch.tensor(
            [template_mask.get(t, 0) for t in templates],
            dtype=torch.bool,
            device=device,
        )
        masked_logits = torch.where(mask_tensor, raw_logits, torch.tensor(-1e9, device=device))

        dist = torch.distributions.Categorical(logits=masked_logits)
        return dist, masked_logits
