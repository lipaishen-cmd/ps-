"""
hppo_trainer.py
===============
双时间尺度层级近端策略优化训练器 HPPO（规范第 14 章、第 18.3 节第九阶段，Gate 10 核心执行体）

功能说明：
  1. 统一管理 HGNN 骨干特征提取器、Manager/Worker 两个独立策略网络与共享 Critic 价值网络。
  2. 驱动 DynamicEventManager 执行动态插单轨迹采样，维护事务隔离的双 RolloutBuffer。
  3. 执行半MDP跳步 GAE 与微观单步 GAE 优势计算。
  4. 联合反向传播更新两级 PPO 裁剪损失、价值函数损失与动作熵正则。
  5. 提供模型权重保存、加载与训练收敛性监控接口。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "environment"))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in [_THIS_DIR, _ENVIRONMENT_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_models import Order
from graph_builder import GRAPH_SCHEMA_VERSION, PCBGraphBuilder
from template_encoding import TEMPLATE_ENCODING_VERSION, TEMPLATE_FEATURE_DIM
from reward import REWARD_CONTRACT_VERSION
from batch_hgnn import PCBBatchHGNN
from manager_policy import ManagerPolicy
from worker_policy import WorkerPolicy
from critics import SharedValueCritic
from rollout_buffer import ManagerRolloutBuffer, WorkerRolloutBuffer
from dynamic_event_manager import (
    DynamicEventManager,
    INITIAL_SOLUTION_BUILDER_VERSION,
    INITIAL_SOLUTION_DISPATCH_RULE,
)
from dynamic_two_agent_runner import DynamicDecision, DynamicTwoAgentRunner

# The sole formal dynamic-experiment protocol is V20261027_THREE_GROUP. Keep
# these defaults in one place so direct HPPO use cannot silently diverge from
# the grouped multi-scenario entry point.
FORMAL_DYNAMIC_PROTOCOL_VERSION = "V20261027_THREE_GROUP"
FORMAL_DYNAMIC_SPLIT_LR = 3e-4
FORMAL_DYNAMIC_PLACEMENT_LR = 1.5e-4
# Backward-compatible name used by C2 for its unchanged shared Actor learning rate.
FORMAL_DYNAMIC_LR = FORMAL_DYNAMIC_SPLIT_LR
FORMAL_DYNAMIC_REWARD_SCALE = 10.0
FORMAL_DYNAMIC_C_ENT = 0.0015
DUAL_ACTOR_LR_CONTRACT_VERSION = "split3e-4-placement1p5e-4-v20261106"
DUAL_ACTOR_SEMANTICS_VERSION = "dual-actor-sequential-template10-placementlr-v20261106"
MAIN_CHECKPOINT_FORMAT_VERSION = 7


class HPPOTrainer:
    """
    双尺度 HPPO 联合训练器。
    """

    method_id = "AD-HG-DAPPO"
    code_id = "dual_actor_hppo"
    dual_actor_semantics_version = DUAL_ACTOR_SEMANTICS_VERSION

    def __init__(
        self,
        orders: List[Order],
        pool_type: str,
        d_h: int = 32,
        lr: float = FORMAL_DYNAMIC_LR,
        placement_lr: float = FORMAL_DYNAMIC_PLACEMENT_LR,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.30,
        c_val: float = 0.25,
        c_ent: float = FORMAL_DYNAMIC_C_ENT,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        device: torch.device = torch.device("cpu"),
        reward_mode: str = "scaled_relative",
        target_kl: float = 0.05,
        max_kl: float = 0.15,
    ):
        self.orders = list(orders)
        self.pool_type = pool_type
        self.device = device
        self.d_h = d_h
        self.lr = float(lr)
        self.worker_lr = float(placement_lr)
        if self.lr <= 0.0 or self.worker_lr <= 0.0:
            raise ValueError("Split/Placement learning rates must both be positive.")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c_val = c_val
        self.c_ent = c_ent
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = max(1, int(ppo_epochs))
        self.reward_mode = str(reward_mode)
        self.target_kl = float(target_kl)
        self.max_kl = float(max_kl)
        self.graph_schema_version = GRAPH_SCHEMA_VERSION

        # 1. 异构图构建器
        self.graph_builder = PCBGraphBuilder(orders=self.orders, pool_type=pool_type, device=device)

        # 2. 神经网络体系
        self.hgnn = PCBBatchHGNN(d_h=d_h).to(device)
        self.manager_policy = ManagerPolicy(
            d_g=2 * d_h,
            d_order=d_h,
            d_template=TEMPLATE_FEATURE_DIM,
            d_template_node=d_h,
            d_hidden=d_h,
        ).to(device)
        self.worker_policy = WorkerPolicy(d_batch=d_h, d_slot=d_h, d_hidden=d_h).to(device)
        # 两个 Actor 独立实例化；价值网络只保留一个共享参数实例。
        self.shared_critic = SharedValueCritic(
            d_g=2 * d_h,
            d_order=d_h,
            d_batch=d_h,
            d_hidden=d_h,
        ).to(device)
        # 两组角色名必须引用同一策略对象，不能产生重复参数。
        self.manager_critic = self.shared_critic
        self.worker_critic = self.shared_critic

        # 3. 经验池
        self.manager_buffer = ManagerRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
        self.worker_buffer = WorkerRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)

        # 4. 优化器：两个 Actor 必须拥有互不共享状态的独立 Adam。
        # HGNN 是共享编码器而非 Actor，使用第三个独立 Adam 接收两类策略损失；
        # 共享 Critic 使用第四个独立 Adam。任何参数只属于一个优化器。
        self.encoder_params = list(self.hgnn.parameters())
        self.actor_params_split = list(self.manager_policy.parameters())
        self.actor_params_placement = list(self.worker_policy.parameters())
        self.actor_params_manager = self.actor_params_split
        self.actor_params_worker = self.actor_params_placement
        self.actor_params = self.actor_params_split + self.actor_params_placement
        self.critic_params = list(self.shared_critic.parameters())
        self.all_params = self.encoder_params + self.actor_params + self.critic_params

        self.optimizer_split = optim.Adam(self.actor_params_split, lr=self.lr)
        self.optimizer_placement = optim.Adam(self.actor_params_placement, lr=self.worker_lr)
        self.optimizer_encoder = optim.Adam(self.encoder_params, lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic_params, lr=self.lr)

    def collect_dynamic_episode(
        self,
        manager: DynamicEventManager,
        deterministic: bool = False,
        reward_scale: float = FORMAL_DYNAMIC_REWARD_SCALE,
        clear_buffer: bool = True,
        record_rollout: bool = True,
    ) -> List[DynamicDecision]:
        """驱动事件队列的在线插单/故障轨迹采样。

        每个事件的候选解必须经过 EventManager 原子事务验证；只有 Actor 候选
        提交时才写入两级 RolloutBuffer；失败候选没有任何替代解或经验写入。

        ``record_rollout=False`` 是验证/部署推理的强隔离开关：该调用既不会
        清空既有训练经验池，也不会向 Manager / Worker buffer 追加任何轨迹。
        调用方仍可获得完整的事件决策与最终物理解，用于计算指标。这个显式
        开关避免评估路径因默认参数而污染下一次 PPO 更新。
        """
        runner = DynamicTwoAgentRunner(
            manager=manager,
            hgnn=self.hgnn,
            manager_policy=self.manager_policy,
            worker_policy=self.worker_policy,
            value_critic=self.shared_critic,
            device=self.device,
            reward_mode=self.reward_mode,
            reward_scale=reward_scale,
        )
        # 评估必须对正在累积的训练 rollout 完全只读。即使调用方误保留了
        # clear_buffer=True，也不能让一次 record_rollout=False 的调用改变它。
        if clear_buffer and record_rollout:
            self.manager_buffer.reset()
            self.worker_buffer.reset()
        decisions: List[DynamicDecision] = []
        while manager.pending_event_ids:
            decision = runner.run_next_event(
                deterministic=deterministic,
            )
            decisions.append(decision)
            is_last_event = (len(manager.pending_event_ids) == 0)
            if record_rollout and decision.trainable:
                # 只有 Actor 候选经 EventManager 原子提交后，才转移到正式经验池。
                # DynamicTwoAgentRunner 已按 reward_scale 完成一次且仅一次归一化。
                scaled_reward = float(decision.reward)
                n_wr = len(decision.worker_records)
                # Worker 单步微观奖励严格依据式 (31) 与参数规范 5.1 节：
                # 使用每步临时放置后实时解码的局部增量 ΔC_m 与当前状态 makespan
                for idx, record in enumerate(decision.worker_records):
                    if "step_reward" not in record:
                        raise RuntimeError("Formal Placement transition is missing reward.py step_reward.")
                    step_reward = float(record["step_reward"])
                    step_done = is_last_event and (idx == n_wr - 1)
                    self.worker_buffer.record_temporary_step(
                        h_batch=record["h_batch"],
                        h_slots=record["h_slots"],
                        slot_keys=record["slot_keys"],
                        valid_slots=record["valid_slots"],
                        action_idx=record["action_idx"],
                        reward=step_reward,
                        log_prob=record["log_prob"],
                        value=record["value"],
                        g=record["g"],
                        h_order=record["h_order"],
                        graph=record.get("graph"),
                        sub_idx=record.get("sub_idx", 0),
                        order_idx=record.get("order_idx", 0),
                        done=step_done,
                        order_id=record.get("order_id"),
                    )
                self.worker_buffer.commit_transaction(done=is_last_event)
                n_mr = len(decision.manager_records)
                for idx_m, manager_record in enumerate(decision.manager_records):
                    m_done = is_last_event and (idx_m == n_mr - 1)
                    m_reward = scaled_reward if idx_m == n_mr - 1 else 0.0
                    m_tau = int(manager_record.get("tau", 0))
                    if m_tau <= 0:
                        raise RuntimeError(
                            f"Manager Option {manager_record.get('option_index', idx_m)} 缺少真实正持续时间 tau。"
                        )
                    self.manager_buffer.add(
                        g=manager_record["g"],
                        h_order=manager_record["h_order"],
                        templates=manager_record["templates"],
                        mask=manager_record["mask"],
                        order_W=manager_record["order_W"],
                        action_idx=manager_record["action_idx"],
                        reward=m_reward,
                        tau=m_tau,
                        log_prob=manager_record["log_prob"],
                        value=manager_record["value"],
                        graph=manager_record.get("graph"),
                        frontier_idx=manager_record.get("frontier_idx", 0),
                        done=m_done,
                    )
            if not decision.committed or manager.active_context is not None:
                # 只回滚当前失败事件：事件前物理解保持不变，当前事件的临时
                # 候选/临时轨迹不进入经验池；此前已原子提交事件的合法轨迹保留。
                # 未解决事件不能被因果跳过，因此 episode 在此处终止，并把
                # 最后一个已提交转移标记为 done，避免 GAE 跨越失败边界。
                manager.rollback_active()
                self.worker_buffer.abort_transaction()
                if record_rollout:
                    if len(self.manager_buffer) > 0:
                        self.manager_buffer.mark_last_done()
                    if len(self.worker_buffer) > 0:
                        self.worker_buffer.mark_last_done()
                break
        return decisions

    def _current_manager_critic_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        global_rows: List[torch.Tensor] = []
        order_rows: List[torch.Tensor] = []
        with torch.no_grad():
            for graph, frontier_idx in zip(
                self.manager_buffer.graph_list, self.manager_buffer.frontier_idx_list
            ):
                if graph is None:
                    raise RuntimeError("Formal Split transition is missing its source graph.")
                embeddings, global_embedding = self.hgnn(graph.to(self.device))
                global_rows.append(global_embedding.detach())
                order_rows.append(
                    embeddings["order"][frontier_idx : frontier_idx + 1].detach()
                )
        return torch.cat(global_rows, dim=0), torch.cat(order_rows, dim=0)

    def _current_worker_critic_inputs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_rows: List[torch.Tensor] = []
        order_rows: List[torch.Tensor] = []
        batch_rows: List[torch.Tensor] = []
        with torch.no_grad():
            for graph, order_idx, sub_idx in zip(
                self.worker_buffer.graph_list,
                self.worker_buffer.order_idx_list,
                self.worker_buffer.sub_idx_list,
            ):
                if graph is None:
                    raise RuntimeError("Formal Placement transition is missing its source graph.")
                embeddings, global_embedding = self.hgnn(graph.to(self.device))
                global_rows.append(global_embedding.detach())
                order_rows.append(embeddings["order"][order_idx : order_idx + 1].detach())
                batch_rows.append(embeddings["subbatch"][sub_idx : sub_idx + 1].detach())
        return (
            torch.cat(global_rows, dim=0),
            torch.cat(order_rows, dim=0),
            torch.cat(batch_rows, dim=0),
        )

    def update_policies(self) -> Dict[str, float]:
        """
        执行双尺度 HPPO 参数联合梯度更新。
        特性：
          1. 端到端可微异构图神经网络反传 (End-to-End HGNN Backprop)：
             在策略前向时通过图拓扑实时前向重计算，打通 Policy 梯度至 self.hgnn 的 71 个参数张量，
             使策略损失梯度能够更新 HGNN 参数；
          2. 两个 Actor 与共享模块完全分离更新：SplitPolicy、PlacementPolicy、共享 HGNN 编码器
             和共享 Critic 分别拥有独立 Adam；两类策略损失各自更新对应 Actor 与共享编码器，
             Critic 输入显式使用 .detach()，价值损失不更新两个 Actor 或 HGNN；
          3. 标准多 Epoch PPO 迭代与探针级诊断指标。
        """
        n_m = len(self.manager_buffer)
        n_w = len(self.worker_buffer)
        if n_m == 0 and n_w == 0:
            return {}

        # ------------------------------------------------------------------
        # 1. 预处理：优势函数标准化与张量准备
        # ------------------------------------------------------------------
        if n_m > 0:
            self.manager_buffer.compute_returns_and_advantages(last_val=0.0)
            m_advs = torch.tensor(self.manager_buffer.advantages, dtype=torch.float32, device=self.device)
            m_rets = torch.tensor(self.manager_buffer.returns, dtype=torch.float32, device=self.device)
            if len(m_advs) > 1:
                m_advs = (m_advs - m_advs.mean()) / (m_advs.std() + 1e-8)

            m_actions = torch.tensor(self.manager_buffer.actions, dtype=torch.long, device=self.device)
            old_log_probs_m = torch.stack([lp.to(self.device) for lp in self.manager_buffer.log_probs])

        if n_w > 0:
            self.worker_buffer.compute_returns_and_advantages(last_val=0.0)
            w_advs = torch.tensor(self.worker_buffer.advantages, dtype=torch.float32, device=self.device)
            w_rets = torch.tensor(self.worker_buffer.returns, dtype=torch.float32, device=self.device)
            if len(w_advs) > 1:
                w_advs = (w_advs - w_advs.mean()) / (w_advs.std() + 1e-8)

            w_actions = torch.tensor(self.worker_buffer.actions, dtype=torch.long, device=self.device)
            old_log_probs_w = torch.stack([lp.to(self.device) for lp in self.worker_buffer.log_probs])

        # ------------------------------------------------------------------
        # 2. 多 Epoch Mini-Batch PPO 联合更新循环 (对齐顶刊 SGD 范式)
        # ------------------------------------------------------------------
        last_metrics: Dict[str, float] = {}
        m_batch_size = max(4, min(64, n_m)) if n_m > 0 else 1
        w_batch_size = max(8, min(64, n_w)) if n_w > 0 else 1

        early_stop_triggered = False
        for epoch in range(self.ppo_epochs):
            if early_stop_triggered:
                break
            loss_m_policy_epoch = 0.0
            loss_m_ent_epoch = 0.0
            loss_m_val_epoch = 0.0
            m_ratios: List[float] = []
            m_clipped_flags: List[float] = []
            m_approx_kls: List[float] = []
            m_entropy_coeffs: List[float] = []
            grad_norms_actor: List[float] = []
            grad_norms_critic: List[float] = []

            loss_w_policy_epoch = 0.0
            loss_w_ent_epoch = 0.0
            loss_w_val_epoch = 0.0
            w_ratios: List[float] = []
            w_clipped_flags: List[float] = []
            w_approx_kls: List[float] = []
            w_entropy_coeffs: List[float] = []

            # --- (A) 宏观 Manager Mini-Batch SGD 更新 ---
            if n_m > 0:
                m_perm = torch.randperm(n_m)
                for start_idx in range(0, n_m, m_batch_size):
                    batch_idxes = m_perm[start_idx : start_idx + m_batch_size]
                    mb_size = len(batch_idxes)
                    loss_mb_policy = torch.tensor(0.0, device=self.device)
                    loss_mb_ent = torch.tensor(0.0, device=self.device)

                    for i_idx in batch_idxes:
                        i = i_idx.item()
                        graph_m = self.manager_buffer.graph_list[i] if (hasattr(self.manager_buffer, "graph_list") and i < len(self.manager_buffer.graph_list)) else None
                        if graph_m is None:
                            raise RuntimeError("Formal Split transition is missing its source graph.")
                        emb_m, g_i = self.hgnn(graph_m.to(self.device))
                        f_idx = self.manager_buffer.frontier_idx_list[i]
                        ho_i = emb_m["order"][f_idx: f_idx + 1]
                        ht_i = emb_m["template"]

                        templates = self.manager_buffer.templates_list[i]
                        mask = self.manager_buffer.mask_list[i]
                        order_W = self.manager_buffer.order_W_list[i]
                        act_idx = m_actions[i]
                        old_lp = old_log_probs_m[i]

                        dist_m, _ = self.manager_policy(
                            g_i, ho_i, ht_i, templates, mask, order_W
                        )
                        new_lp = dist_m.log_prob(act_idx)
                        entropy = dist_m.entropy()

                        ratio = torch.exp(new_lp - old_lp)
                        if not torch.isfinite(ratio):
                            raise FloatingPointError("Non-finite Split PPO probability ratio.")
                        surr1 = ratio * m_advs[i]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * m_advs[i]

                        loss_mb_policy = loss_mb_policy + (-torch.min(surr1, surr2))
                        loss_mb_ent = loss_mb_ent + (-entropy)

                        with torch.no_grad():
                            r_val = ratio.item()
                            m_ratios.append(r_val)
                            is_clip = float(r_val < (1.0 - self.clip_eps) or r_val > (1.0 + self.clip_eps))
                            m_clipped_flags.append(is_clip)
                            log_ratio = (old_lp - new_lp).item()
                            m_approx_kls.append(0.5 * (log_ratio ** 2))

                    loss_mb_policy = loss_mb_policy / mb_size
                    loss_mb_ent = loss_mb_ent / mb_size
                    # Manager Policy 动作熵下限自适应保护 (H < 1.0 时动态增强)
                    manager_ent_val = -loss_mb_ent.item()
                    if manager_ent_val < 1.0:
                        adaptive_m_c_ent = max(self.c_ent, self.c_ent * (1.0 / max(1e-4, manager_ent_val)), 0.003 * (1.0 - manager_ent_val))
                    else:
                        adaptive_m_c_ent = self.c_ent
                    m_entropy_coeffs.append(float(adaptive_m_c_ent))
                    loss_mb_actor = loss_mb_policy + adaptive_m_c_ent * loss_mb_ent

                    # 触发 Mini-Batch 优化步进
                    if loss_mb_actor.requires_grad:
                        self.optimizer_split.zero_grad()
                        self.optimizer_encoder.zero_grad()
                        loss_mb_actor.backward()
                        norm_a = nn.utils.clip_grad_norm_(
                            self.actor_params_split + self.encoder_params,
                            self.max_grad_norm,
                        )
                        norm_val = float(norm_a.item() if hasattr(norm_a, "item") else norm_a)
                        grad_norms_actor.append(norm_val)
                        if torch.isfinite(torch.as_tensor(norm_a)):
                            self.optimizer_split.step()
                            self.optimizer_encoder.step()
                        else:
                            self.optimizer_split.zero_grad()
                            self.optimizer_encoder.zero_grad()

                    loss_m_policy_epoch += loss_mb_policy.item() * (mb_size / n_m)
                    loss_m_ent_epoch += loss_mb_ent.item() * (mb_size / n_m)

                    # 批次内 Manager KL 散度实时检测与信任域熔断保护
                    if len(m_approx_kls) >= mb_size:
                        recent_batch_m_kl = float(sum(m_approx_kls[-mb_size:]) / mb_size)
                        if recent_batch_m_kl > self.max_kl:
                            early_stop_triggered = True
                            break

                # Manager Critic 单独更新
                g_m_current, ho_m_current = self._current_manager_critic_inputs()
                new_val_m = self.manager_critic(
                    g_m_current.detach(), ho_m_current.detach()
                ).squeeze(-1)
                loss_m_val = F.smooth_l1_loss(new_val_m, m_rets, beta=1.0)
                loss_m_val_epoch = loss_m_val.item()
                if loss_m_val.requires_grad:
                    self.optimizer_critic.zero_grad()
                    (self.c_val * loss_m_val).backward()
                    norm_c = nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                    norm_c_val = float(norm_c.item() if hasattr(norm_c, "item") else norm_c)
                    grad_norms_critic.append(norm_c_val)
                    if torch.isfinite(torch.as_tensor(norm_c)):
                        self.optimizer_critic.step()
                    else:
                        self.optimizer_critic.zero_grad()

            # --- (B) 微观 Worker Mini-Batch SGD 更新 ---
            has_learnable_worker = len(getattr(self, "actor_params_worker", [])) > 0
            if n_w > 0 and has_learnable_worker:
                w_perm = torch.randperm(n_w)
                for start_idx in range(0, n_w, w_batch_size):
                    batch_idxes = w_perm[start_idx : start_idx + w_batch_size]
                    wb_size = len(batch_idxes)
                    loss_wb_policy = torch.tensor(0.0, device=self.device)
                    loss_wb_ent = torch.tensor(0.0, device=self.device)

                    for i_idx in batch_idxes:
                        i = i_idx.item()
                        graph_w = self.worker_buffer.graph_list[i] if (hasattr(self.worker_buffer, "graph_list") and i < len(self.worker_buffer.graph_list)) else None
                        if graph_w is None:
                            raise RuntimeError("Formal Placement transition is missing its source graph.")
                        emb_w, _ = self.hgnn(graph_w.to(self.device))
                        s_idx = self.worker_buffer.sub_idx_list[i]
                        hb_i = emb_w["subbatch"][s_idx: s_idx + 1]
                        hs_i = emb_w["slot"]

                        slot_keys = self.worker_buffer.slot_keys_list[i]
                        valid_slots = self.worker_buffer.valid_slots_list[i]
                        w_oid = (
                            self.worker_buffer.order_id_list[i]
                            if hasattr(self.worker_buffer, "order_id_list") and i < len(self.worker_buffer.order_id_list)
                            else None
                        )
                        act_idx = w_actions[i]
                        old_lp = old_log_probs_w[i]

                        dist_w, _ = self.worker_policy(hb_i, hs_i, slot_keys, valid_slots, order_id=w_oid)
                        new_lp = dist_w.log_prob(act_idx)
                        entropy = dist_w.entropy()

                        ratio = torch.exp(new_lp - old_lp)
                        if not torch.isfinite(ratio):
                            raise FloatingPointError("Non-finite Placement PPO probability ratio.")
                        surr1 = ratio * w_advs[i]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * w_advs[i]

                        loss_wb_policy = loss_wb_policy + (-torch.min(surr1, surr2))
                        loss_wb_ent = loss_wb_ent + (-entropy)

                        with torch.no_grad():
                            r_val = ratio.item()
                            w_ratios.append(r_val)
                            is_clip = float(r_val < (1.0 - self.clip_eps) or r_val > (1.0 + self.clip_eps))
                            w_clipped_flags.append(is_clip)
                            log_ratio = (old_lp - new_lp).item()
                            w_approx_kls.append(0.5 * (log_ratio ** 2))

                    loss_wb_policy = loss_wb_policy / wb_size
                    loss_wb_ent = loss_wb_ent / wb_size
                    # 引入 Worker Policy 熵下限自适应保护：
                    # 动态监控 Worker Entropy，若 H < 1.5，提供高刚性惩罚推力，强行阻止匹配网络过早陷入局部死锁与确定性硬选择
                    worker_ent_val = -loss_wb_ent.item()
                    if worker_ent_val < 1.5:
                        adaptive_c_ent = max(self.c_ent, self.c_ent * (1.5 / max(1e-4, worker_ent_val)), 0.005 * (1.5 - worker_ent_val))
                    else:
                        adaptive_c_ent = self.c_ent
                    w_entropy_coeffs.append(float(adaptive_c_ent))
                    loss_wb_actor = loss_wb_policy + adaptive_c_ent * loss_wb_ent

                    # 触发 Mini-Batch 优化步进
                    if loss_wb_actor.requires_grad:
                        self.optimizer_placement.zero_grad()
                        self.optimizer_encoder.zero_grad()
                        loss_wb_actor.backward()
                        norm_a = nn.utils.clip_grad_norm_(
                            self.actor_params_placement + self.encoder_params,
                            self.max_grad_norm,
                        )
                        norm_val = float(norm_a.item() if hasattr(norm_a, "item") else norm_a)
                        grad_norms_actor.append(norm_val)
                        if torch.isfinite(torch.as_tensor(norm_a)):
                            self.optimizer_placement.step()
                            self.optimizer_encoder.step()
                        else:
                            self.optimizer_placement.zero_grad()
                            self.optimizer_encoder.zero_grad()

                    loss_w_policy_epoch += loss_wb_policy.item() * (wb_size / n_w)
                    loss_w_ent_epoch += loss_wb_ent.item() * (wb_size / n_w)

                    # 批次内 Worker KL 散度实时检测与信任域熔断保护
                    if len(w_approx_kls) >= wb_size:
                        recent_batch_w_kl = float(sum(w_approx_kls[-wb_size:]) / wb_size)
                        if recent_batch_w_kl > self.max_kl:
                            early_stop_triggered = True
                            break

                # Worker Critic 单独更新
                if n_w > 0:
                    g_w_current, ho_w_current, hb_w_current = self._current_worker_critic_inputs()
                    new_val_w = self.shared_critic(
                        g_w_current.detach(), ho_w_current.detach(), hb_w_current.detach()
                    ).squeeze(-1)
                    loss_w_val = F.smooth_l1_loss(new_val_w, w_rets, beta=1.0)
                    loss_w_val_epoch = loss_w_val.item()
                    if loss_w_val.requires_grad:
                        self.optimizer_critic.zero_grad()
                        (self.c_val * loss_w_val).backward()
                        norm_c = nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                        norm_c_val = float(norm_c.item() if hasattr(norm_c, "item") else norm_c)
                        grad_norms_critic.append(norm_c_val)
                        if torch.isfinite(torch.as_tensor(norm_c)):
                            self.optimizer_critic.step()
                        else:
                            self.optimizer_critic.zero_grad()

            # 汇总最新 Epoch 诊断数据
            # PPO KL Early-Stopping 信任域保护检测（含 target_kl 偏离控制与 max_kl 熔断）
            cur_epoch_m_kl = float(sum(m_approx_kls) / len(m_approx_kls)) if m_approx_kls else 0.0
            cur_epoch_w_kl = float(sum(w_approx_kls) / len(w_approx_kls)) if w_approx_kls else 0.0
            if (
                cur_epoch_w_kl > self.max_kl
                or cur_epoch_m_kl > self.max_kl
                or (self.target_kl is not None and (cur_epoch_w_kl > 1.5 * self.target_kl or cur_epoch_m_kl > 1.5 * self.target_kl))
            ):
                early_stop_triggered = True

            mean_gn_actor = float(sum(grad_norms_actor) / len(grad_norms_actor)) if grad_norms_actor else 0.0
            mean_gn_critic = float(sum(grad_norms_critic) / len(grad_norms_critic)) if grad_norms_critic else 0.0
            tot_loss_epoch = loss_m_policy_epoch + loss_w_policy_epoch + self.c_val * (loss_m_val_epoch + loss_w_val_epoch)
            last_metrics = {
                "total_loss": tot_loss_epoch,
                "m_policy_loss": loss_m_policy_epoch,
                "m_val_loss": loss_m_val_epoch,
                "w_policy_loss": loss_w_policy_epoch,
                "w_val_loss": loss_w_val_epoch,
                "m_entropy": (-loss_m_ent_epoch) if n_m > 0 else 0.0,
                "w_entropy": (-loss_w_ent_epoch) if n_w > 0 else 0.0,
                "m_clip_fraction": float(sum(m_clipped_flags) / len(m_clipped_flags)) if m_clipped_flags else 0.0,
                "w_clip_fraction": float(sum(w_clipped_flags) / len(w_clipped_flags)) if w_clipped_flags else 0.0,
                "m_ratio_mean": float(sum(m_ratios) / len(m_ratios)) if m_ratios else 1.0,
                "w_ratio_mean": float(sum(w_ratios) / len(w_ratios)) if w_ratios else 1.0,
                "m_approx_kl": cur_epoch_m_kl,
                "w_approx_kl": cur_epoch_w_kl,
                "grad_norm_actor": mean_gn_actor,
                "grad_norm_critic": mean_gn_critic,
                "grad_norm": mean_gn_actor + mean_gn_critic,
                "c_ent": float(self.c_ent),
                "m_entropy_coeff_effective": float(sum(m_entropy_coeffs) / len(m_entropy_coeffs)) if m_entropy_coeffs else float(self.c_ent),
                "w_entropy_coeff_effective": float(sum(w_entropy_coeffs) / len(w_entropy_coeffs)) if w_entropy_coeffs else float(self.c_ent),
                "ppo_epochs": float(self.ppo_epochs),
                "ppo_epochs_completed": float(epoch + 1),
                "kl_early_stopped": float(1.0 if early_stop_triggered else 0.0),
            }
            if early_stop_triggered:
                break

        # 更新完成后物理重置经验池，准备累积下一批轨迹
        self.manager_buffer.reset()
        self.worker_buffer.reset()

        return last_metrics

    def step_entropy_anneal(self, factor: float = 0.9997, min_ent: float = 0.0005):
        """动作熵正则化系数指数退火，平滑释放策略梯度（保持 min_ent 探索底线）"""
        self.c_ent = max(min_ent, self.c_ent * factor)

    def save_checkpoint(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        """保存 HGNN、两个 Actor、共享 Critic、双优化器状态以及可选的训练进度元数据。"""
        filepath = os.path.abspath(filepath)
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        external_meta = dict(extra_meta or {})
        authoritative_identity = {
            "method_id": self.method_id,
            "code_id": self.code_id,
            "dual_actor_semantics_version": DUAL_ACTOR_SEMANTICS_VERSION,
            "dual_actor_lr_contract_version": DUAL_ACTOR_LR_CONTRACT_VERSION,
            "split_learning_rate": self.lr,
            "placement_learning_rate": self.worker_lr,
            "encoder_learning_rate": self.lr,
            "critic_learning_rate": self.lr,
        }
        for key, expected in authoritative_identity.items():
            if key in external_meta and external_meta[key] != expected:
                raise ValueError(
                    f"extra_meta cannot override main-method identity field {key!r}: "
                    f"received {external_meta[key]!r}, expected {expected!r}."
                )
        payload = {
            "format_version": MAIN_CHECKPOINT_FORMAT_VERSION,
            "split_learning_rate": self.lr,
            "placement_learning_rate": self.worker_lr,
            "encoder_learning_rate": self.lr,
            "critic_learning_rate": self.lr,
            "initial_solution_builder_version": INITIAL_SOLUTION_BUILDER_VERSION,
            "initial_solution_dispatch_rule": INITIAL_SOLUTION_DISPATCH_RULE,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "template_encoding_version": TEMPLATE_ENCODING_VERSION,
            "reward_contract_version": REWARD_CONTRACT_VERSION,
            "pool_type": self.pool_type,
            "hgnn": self.hgnn.state_dict(),
            "manager_policy": self.manager_policy.state_dict(),
            "worker_policy": self.worker_policy.state_dict(),
            "shared_critic": self.shared_critic.state_dict(),
            "optimizer_split": self.optimizer_split.state_dict(),
            "optimizer_placement": self.optimizer_placement.state_dict(),
            "optimizer_encoder": self.optimizer_encoder.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
        }
        payload.update(external_meta)
        payload.update(authoritative_identity)
        for attempt in range(3):
            try:
                torch.save(payload, filepath)
                break
            except (PermissionError, OSError) as exc:
                if attempt == 2:
                    print(f"[警告] 模型断点写入受系统或OneDrive临时锁定影响 ({exc})，已安全跳过本次写入以防中断训练。")
                else:
                    time.sleep(0.15)
        return filepath

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, object]:
        """加载训练权重；推理时可将 ``load_optimizer`` 设为 False。"""
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型权重不存在: {filepath}")
        try:
            payload = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:  # 支持不接受 weights_only 参数的 PyTorch 版本
            payload = torch.load(filepath, map_location=self.device)
        if not isinstance(payload, dict):
            raise ValueError("模型权重文件格式错误：顶层必须为字典。")
        if payload.get("method_id") != self.method_id:
            raise ValueError("Checkpoint 不是当前 AD-HG-DAPPO 主算法权重。")
        if payload.get("code_id") != self.code_id:
            raise ValueError("Checkpoint code_id 与当前双 Actor 实现不一致。")
        if payload.get("dual_actor_semantics_version") != DUAL_ACTOR_SEMANTICS_VERSION:
            raise ValueError(
                "Checkpoint 不具备当前双 Actor 顺序 Option 语义；旧的单 Actor 优化器/先选多次 Split "
                "检查点不得恢复，请从随机初始化重新训练。"
            )
        if payload.get("dual_actor_lr_contract_version") != DUAL_ACTOR_LR_CONTRACT_VERSION:
            raise ValueError(
                "Checkpoint learning-rate contract is obsolete; the current main method requires "
                "independent Split=3e-4 and Placement=1.5e-4 learning rates. Retrain from scratch."
            )
        if payload.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
            raise ValueError(
                "Checkpoint 与当前六节点订单上下文 Graph Schema 不兼容；"
                f"当前要求 {GRAPH_SCHEMA_VERSION}，请从随机初始化重新训练。"
            )
        if payload.get("format_version") != MAIN_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Checkpoint format is obsolete; old checkpoints are forbidden.")
        expected_learning_rates = {
            "split_learning_rate": self.lr,
            "placement_learning_rate": self.worker_lr,
            "encoder_learning_rate": self.lr,
            "critic_learning_rate": self.lr,
        }
        mismatched_learning_rates = {
            key: (payload.get(key), expected)
            for key, expected in expected_learning_rates.items()
            if payload.get(key) != expected
        }
        if mismatched_learning_rates:
            raise ValueError(
                "Checkpoint learning rates do not match the instantiated trainer: "
                f"{mismatched_learning_rates}. Retrain from scratch or use the matching configuration."
            )
        if payload.get("initial_solution_builder_version") != INITIAL_SOLUTION_BUILDER_VERSION:
            raise ValueError(
                "Checkpoint initial-solution contract is obsolete; "
                "only the current SPT initializer is allowed. Retrain from scratch."
            )
        if payload.get("initial_solution_dispatch_rule") != INITIAL_SOLUTION_DISPATCH_RULE:
            raise ValueError(
                "Checkpoint initial dispatch rule is obsolete; only SPT is allowed."
            )
        if payload.get("template_encoding_version") != TEMPLATE_ENCODING_VERSION:
            raise ValueError("Checkpoint template encoding is obsolete; retrain from scratch.")
        if payload.get("reward_contract_version") != REWARD_CONTRACT_VERSION:
            raise ValueError("Checkpoint reward contract is obsolete; retrain from scratch.")
        required = (
            "hgnn",
            "manager_policy",
            "worker_policy",
            "shared_critic",
            "optimizer_split",
            "optimizer_placement",
            "optimizer_encoder",
            "optimizer_critic",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"模型权重缺少字段: {missing}")

        # 变体结构必须由对应消融组的加载器预先实例化；核心训练器仅加载主算法结构。
        self.hgnn.load_state_dict(payload["hgnn"])
        self.manager_policy.load_state_dict(payload["manager_policy"])
        self.worker_policy.load_state_dict(payload["worker_policy"])
        self.shared_critic.load_state_dict(payload["shared_critic"])
        if load_optimizer:
            self.optimizer_split.load_state_dict(payload["optimizer_split"])
            self.optimizer_placement.load_state_dict(payload["optimizer_placement"])
            self.optimizer_encoder.load_state_dict(payload["optimizer_encoder"])
            self.optimizer_critic.load_state_dict(payload["optimizer_critic"])
        return payload

    def set_lr_multiplier(self, multiplier: float) -> None:
        """根据阶段跃迁平滑阻尼与 Warmup 乘子动态调整优化器学习率。"""
        mult = max(0.01, min(1.0, float(multiplier)))
        cur_manager_lr = self.lr * mult
        cur_worker_lr = self.worker_lr * mult
        cur_critic_lr = self.lr * mult

        for param_group in self.optimizer_split.param_groups:
            param_group["lr"] = cur_manager_lr
        for param_group in self.optimizer_placement.param_groups:
            param_group["lr"] = cur_worker_lr
        for param_group in self.optimizer_encoder.param_groups:
            param_group["lr"] = cur_manager_lr
        for param_group in self.optimizer_critic.param_groups:
            param_group["lr"] = cur_critic_lr

