"""Executable single-Actor PPO ablation with the main method's environment."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
_ENVIRONMENT_DIR = _PROJECT_ROOT / "environment"

for _p in (str(_THIS_DIR), str(_PROJECT_ROOT), str(_AGENT_DIR), str(_ENVIRONMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from batch_hgnn import PCBBatchHGNN
from critics import SharedValueCritic
from data_models import Order
from dynamic_event_manager import (
    DynamicEventManager,
    INITIAL_SOLUTION_BUILDER_VERSION,
    INITIAL_SOLUTION_DISPATCH_RULE,
)
from dynamic_two_agent_runner import DynamicDecision, DynamicTwoAgentRunner
from graph_builder import GRAPH_SCHEMA_VERSION, PCBGraphBuilder
from template_encoding import TEMPLATE_ENCODING_VERSION, TEMPLATE_FEATURE_DIM
from reward import REWARD_CONTRACT_VERSION
from hppo_trainer import (
    FORMAL_DYNAMIC_C_ENT,
    FORMAL_DYNAMIC_LR,
    FORMAL_DYNAMIC_PROTOCOL_VERSION,
    FORMAL_DYNAMIC_REWARD_SCALE,
)

from single_actor_buffer import SingleActorRolloutBuffer
from single_actor_policy import PLACEMENT_ACTION, SPLIT_ACTION, SingleActorPolicy


SINGLE_ACTOR_METHOD_ID = "AD-HG-SAPPO"
SINGLE_ACTOR_CODE_ID = "c2_single_actor_ppo"
SINGLE_ACTOR_ABLATION_SCOPE = "hierarchical_dual_policy_to_unified_single_policy"
SINGLE_ACTOR_CREDIT_ASSIGNMENT = "single_step_gae_on_unified_chronological_trajectory"
SINGLE_ACTOR_EVENT_REWARD_ASSIGNMENT = "last_placement_transition"
SINGLE_ACTOR_OBJECTIVE_WEIGHTING = "transition_frequency_weighted"
SINGLE_ACTOR_LR_SEMANTICS = "one_shared_actor_optimizer_uniform_lr_no_action_reweighting"
SINGLE_ACTOR_ENTROPY_REGULARIZATION = "type_conditioned_manager_worker_thresholds_frequency_weighted"
SPLIT_ENTROPY_TARGET = 1.0
PLACEMENT_ENTROPY_TARGET = 1.5


class SingleActorPPOTrainer:
    """PPO trainer containing one Actor, one buffer and one Actor optimizer."""

    method_id = SINGLE_ACTOR_METHOD_ID
    code_id = SINGLE_ACTOR_CODE_ID

    def __init__(
        self,
        orders: List[Order],
        pool_type: str,
        d_h: int = 32,
        lr: float = FORMAL_DYNAMIC_LR,
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
        self.d_h = int(d_h)
        self.lr = float(lr)
        self.worker_lr = self.lr
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.c_val = float(c_val)
        self.c_ent = float(c_ent)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = max(1, int(ppo_epochs))
        self.reward_mode = str(reward_mode)
        self.target_kl = float(target_kl)
        self.max_kl = float(max_kl)
        self.graph_schema_version = GRAPH_SCHEMA_VERSION

        self.graph_builder = PCBGraphBuilder(
            orders=self.orders, pool_type=pool_type, device=device
        )
        self.hgnn = PCBBatchHGNN(d_h=self.d_h).to(device)
        self.single_actor_policy = SingleActorPolicy(d_h=self.d_h).to(device)
        self.shared_critic = SharedValueCritic(
            d_g=2 * self.d_h,
            d_order=self.d_h,
            d_batch=self.d_h,
            d_hidden=self.d_h,
        ).to(device)

        # Compatibility aliases point to the same objects; no second Actor or
        # second Critic is instantiated.
        self.manager_policy = self.single_actor_policy
        self.worker_policy = self.single_actor_policy
        self.manager_critic = self.shared_critic
        self.worker_critic = self.shared_critic

        self.rollout_buffer = SingleActorRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
        self.manager_buffer = self.rollout_buffer
        self.worker_buffer = self.rollout_buffer

        self.actor_params = list(self.hgnn.parameters()) + list(
            self.single_actor_policy.parameters()
        )
        self.critic_params = list(self.shared_critic.parameters())
        self.optimizer_actor = optim.Adam(self.actor_params, lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic_params, lr=self.lr)
        self.optimizer = self.optimizer_actor
        self.checkpoint_contract: Dict[str, Any] = {}

    def adaptive_entropy_coefficient(
        self,
        decision_type: str,
        entropy_value: float,
    ) -> float:
        """Return the main-method-matched entropy coefficient for one action type.

        The coefficient is action-type conditioned, but the final entropy term
        remains divided by the full mixed mini-batch size.  Hence split and
        placement transitions retain their empirical frequencies: no type
        resampling and no surrogate-loss reweighting are introduced.
        """
        entropy = float(entropy_value)
        if decision_type == SPLIT_ACTION:
            if entropy >= SPLIT_ENTROPY_TARGET:
                return float(self.c_ent)
            return float(
                max(
                    self.c_ent,
                    self.c_ent * (SPLIT_ENTROPY_TARGET / max(1.0e-4, entropy)),
                    0.003 * (SPLIT_ENTROPY_TARGET - entropy),
                )
            )
        if decision_type == PLACEMENT_ACTION:
            if entropy >= PLACEMENT_ENTROPY_TARGET:
                return float(self.c_ent)
            return float(
                max(
                    self.c_ent,
                    self.c_ent * (PLACEMENT_ENTROPY_TARGET / max(1.0e-4, entropy)),
                    0.005 * (PLACEMENT_ENTROPY_TARGET - entropy),
                )
            )
        raise ValueError(f"Unsupported decision_type for entropy protection: {decision_type!r}")

    def _add_manager_record(self, record: Dict[str, Any]) -> None:
        self.rollout_buffer.add(
            {
                "decision_type": SPLIT_ACTION,
                "g": record["g"],
                "h_order": record["h_order"],
                "graph": record.get("graph"),
                "frontier_idx": int(record.get("frontier_idx", 0)),
                "templates": list(record["templates"]),
                "template_mask": dict(record["mask"]),
                "order_W": int(record["order_W"]),
                "action_idx": int(record["action_idx"]),
                "reward": 0.0,
                "done": False,
                "log_prob": record["log_prob"],
                "value": record["value"],
            }
        )

    def _add_worker_record(self, record: Dict[str, Any]) -> None:
        self.rollout_buffer.add(
            {
                "decision_type": PLACEMENT_ACTION,
                "g": record["g"],
                "h_order": record["h_order"],
                "h_batch": record["h_batch"],
                "h_slots": record["h_slots"],
                "graph": record.get("graph"),
                "sub_idx": int(record.get("sub_idx", 0)),
                "order_idx": int(record.get("order_idx", 0)),
                "slot_keys": list(record["slot_keys"]),
                "valid_slots": list(record["valid_slots"]),
                "action_idx": int(record["action_idx"]),
                "reward": float(record["step_reward"]),
                "done": False,
                "log_prob": record["log_prob"],
                "value": record["value"],
            }
        )

    def collect_dynamic_episode(
        self,
        manager: DynamicEventManager,
        deterministic: bool = False,
        reward_scale: float = FORMAL_DYNAMIC_REWARD_SCALE,
        clear_buffer: bool = True,
        record_rollout: bool = True,
    ) -> List[DynamicDecision]:
        """Collect one scenario using the main runner and a single policy object."""
        if clear_buffer and record_rollout:
            self.rollout_buffer.reset()
        runner = DynamicTwoAgentRunner(
            manager=manager,
            hgnn=self.hgnn,
            manager_policy=self.single_actor_policy,
            worker_policy=self.single_actor_policy,
            value_critic=self.shared_critic,
            device=self.device,
            reward_mode=self.reward_mode,
            reward_scale=reward_scale,
        )
        if runner.manager_policy is not runner.worker_policy:
            raise AssertionError("Single-Actor ablation created more than one policy instance.")

        decisions: List[DynamicDecision] = []
        while manager.pending_event_ids:
            decision = runner.run_next_event(deterministic=deterministic)
            decisions.append(decision)
            if record_rollout and decision.trainable:
                event_start = len(self.rollout_buffer)
                for record in decision.manager_records:
                    self._add_manager_record(record)
                for record in decision.worker_records:
                    self._add_worker_record(record)
                if len(self.rollout_buffer) == event_start:
                    raise RuntimeError("A trainable event produced no SingleActor transitions.")
                # The event-level makespan reward is assigned once to the last
                # physical action; preceding split decisions receive it through GAE.
                self.rollout_buffer.add_terminal_reward(float(decision.reward))
                if not manager.pending_event_ids:
                    self.rollout_buffer.mark_last_done()

            if not decision.committed or manager.active_context is not None:
                # 与主方法保持同一事件失败语义：只回滚当前事件，保留此前
                # 已提交的合法轨迹；episode 在未解决事件处终止。
                manager.rollback_active()
                if record_rollout:
                    if len(self.rollout_buffer) > 0:
                        self.rollout_buffer.mark_last_done()
                break
        return decisions

    def _recompute_distribution(self, item: Dict[str, Any]):
        graph = item.get("graph")
        if graph is None:
            raise RuntimeError("Formal C2 transition is missing its source graph.")

        embeddings, global_embedding = self.hgnn(graph.to(self.device))
        if item["decision_type"] == SPLIT_ACTION:
            frontier_idx = int(item["frontier_idx"])
            h_order = embeddings["order"][frontier_idx : frontier_idx + 1]
            h_templates = embeddings["template"]
            return self.single_actor_policy.split_distribution(
                global_embedding,
                h_order,
                item["templates"],
                item["template_mask"],
                item["order_W"],
                h_templates=h_templates,
            )
        sub_idx = int(item["sub_idx"])
        h_batch = embeddings["subbatch"][sub_idx : sub_idx + 1]
        return self.single_actor_policy.placement_distribution(
            h_batch,
            embeddings["slot"],
            item["slot_keys"],
            item["valid_slots"],
        )

    def _current_critic_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_rows: List[torch.Tensor] = []
        order_rows: List[torch.Tensor] = []
        batch_rows: List[torch.Tensor] = []
        with torch.no_grad():
            for item in self.rollout_buffer.transitions:
                graph = item.get("graph")
                if graph is None:
                    raise RuntimeError("Formal C2 transition is missing its source graph.")
                embeddings, global_embedding = self.hgnn(graph.to(self.device))
                order_idx = int(
                    item.get("frontier_idx", 0)
                    if item["decision_type"] == SPLIT_ACTION
                    else item.get("order_idx", 0)
                )
                global_rows.append(global_embedding.detach())
                order_rows.append(embeddings["order"][order_idx : order_idx + 1].detach())
                if item["decision_type"] == PLACEMENT_ACTION:
                    sub_idx = int(item["sub_idx"])
                    batch_rows.append(embeddings["subbatch"][sub_idx : sub_idx + 1].detach())
                else:
                    batch_rows.append(torch.zeros((1, self.d_h), device=self.device))
        return (
            torch.cat(global_rows, dim=0),
            torch.cat(order_rows, dim=0),
            torch.cat(batch_rows, dim=0),
        )

    def update_policies(self) -> Dict[str, float]:
        """Run one transition-weighted PPO objective and report type-wise diagnostics."""
        n = len(self.rollout_buffer)
        if n == 0:
            return {}
        self.rollout_buffer.compute_returns_and_advantages(last_value=0.0)
        raw_advantages = torch.tensor(
            self.rollout_buffer.advantages, dtype=torch.float32, device=self.device
        )
        advantages = raw_advantages.clone()
        returns = torch.tensor(
            self.rollout_buffer.returns, dtype=torch.float32, device=self.device
        )
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        old_log_probs = torch.stack(
            [torch.as_tensor(t["log_prob"]).to(self.device).reshape(()) for t in self.rollout_buffer.transitions]
        )
        type_counts = self.rollout_buffer.decision_type_counts()
        type_indices = {
            decision_type: [
                index
                for index, item in enumerate(self.rollout_buffer.transitions)
                if item["decision_type"] == decision_type
            ]
            for decision_type in (SPLIT_ACTION, PLACEMENT_ACTION)
        }

        def _type_tensor_stat(values: torch.Tensor, decision_type: str, stat: str) -> float:
            indices = type_indices[decision_type]
            if not indices:
                return 0.0
            selected = values[torch.tensor(indices, dtype=torch.long, device=self.device)]
            if stat == "mean":
                return float(selected.mean().item())
            if stat == "std":
                return float(selected.std(unbiased=False).item())
            raise ValueError(f"Unsupported statistic: {stat}")

        batch_size = max(8, min(64, n))
        last_metrics: Dict[str, float] = {}
        stopped = False
        actor_optimizer_steps = 0
        critic_optimizer_steps = 0
        mini_batches_processed = 0
        adaptive_entropy_values = {
            SPLIT_ACTION: [],
            PLACEMENT_ACTION: [],
        }
        type_diagnostics = {
            SPLIT_ACTION: {
                "evaluations": 0.0,
                "policy_loss_sum": 0.0,
                "entropy_sum": 0.0,
                "kl_sum": 0.0,
                "clip_sum": 0.0,
                "ratio_sum": 0.0,
            },
            PLACEMENT_ACTION: {
                "evaluations": 0.0,
                "policy_loss_sum": 0.0,
                "entropy_sum": 0.0,
                "kl_sum": 0.0,
                "clip_sum": 0.0,
                "ratio_sum": 0.0,
            },
        }
        for epoch in range(self.ppo_epochs):
            policy_total = 0.0
            entropy_total = 0.0
            ratios: List[float] = []
            clipped: List[float] = []
            kls: List[float] = []
            actor_norms: List[float] = []
            critic_norms: List[float] = []

            permutation = torch.randperm(n)
            for start in range(0, n, batch_size):
                indices = permutation[start : start + batch_size]
                policy_loss = torch.tensor(0.0, device=self.device)
                entropy_loss_by_type = {
                    SPLIT_ACTION: torch.tensor(0.0, device=self.device),
                    PLACEMENT_ACTION: torch.tensor(0.0, device=self.device),
                }
                entropy_count_by_type = {
                    SPLIT_ACTION: 0,
                    PLACEMENT_ACTION: 0,
                }
                for tensor_index in indices:
                    index = int(tensor_index.item())
                    item = self.rollout_buffer.transitions[index]
                    distribution, _ = self._recompute_distribution(item)
                    action = torch.tensor(item["action_idx"], dtype=torch.long, device=self.device)
                    new_log_prob = distribution.log_prob(action)
                    ratio = torch.exp(new_log_prob - old_log_probs[index])
                    if not torch.isfinite(ratio):
                        raise FloatingPointError("Non-finite SingleActor PPO probability ratio.")
                    surrogate_1 = ratio * advantages[index]
                    surrogate_2 = torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    ) * advantages[index]
                    sample_policy_loss = -torch.min(surrogate_1, surrogate_2)
                    sample_entropy = distribution.entropy()
                    policy_loss = policy_loss + sample_policy_loss
                    decision_type = item["decision_type"]
                    entropy_loss_by_type[decision_type] = (
                        entropy_loss_by_type[decision_type] - sample_entropy
                    )
                    entropy_count_by_type[decision_type] += 1
                    ratio_value = float(ratio.detach().item())
                    kl_value = float(
                        0.5 * (old_log_probs[index] - new_log_prob).pow(2).detach().item()
                    )
                    clipped_value = float(
                        ratio_value < 1.0 - self.clip_eps
                        or ratio_value > 1.0 + self.clip_eps
                    )
                    ratios.append(ratio_value)
                    clipped.append(clipped_value)
                    kls.append(kl_value)
                    diagnostics = type_diagnostics[decision_type]
                    diagnostics["evaluations"] += 1.0
                    diagnostics["policy_loss_sum"] += float(sample_policy_loss.detach().item())
                    diagnostics["entropy_sum"] += float(sample_entropy.detach().item())
                    diagnostics["kl_sum"] += kl_value
                    diagnostics["clip_sum"] += clipped_value
                    diagnostics["ratio_sum"] += ratio_value

                size = max(1, len(indices))
                policy_loss = policy_loss / size
                entropy_loss = sum(entropy_loss_by_type.values()) / size
                entropy_regularizer = torch.tensor(0.0, device=self.device)
                for decision_type in (SPLIT_ACTION, PLACEMENT_ACTION):
                    type_count = entropy_count_by_type[decision_type]
                    if type_count <= 0:
                        continue
                    type_entropy_value = float(
                        (
                            -entropy_loss_by_type[decision_type] / type_count
                        ).detach().item()
                    )
                    adaptive_entropy = self.adaptive_entropy_coefficient(
                        decision_type,
                        type_entropy_value,
                    )
                    adaptive_entropy_values[decision_type].append(adaptive_entropy)
                    entropy_regularizer = entropy_regularizer + (
                        adaptive_entropy
                        * entropy_loss_by_type[decision_type]
                        / size
                    )
                actor_loss = policy_loss + entropy_regularizer
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                actor_norm = nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                if torch.isfinite(torch.as_tensor(actor_norm)):
                    self.optimizer_actor.step()
                    actor_optimizer_steps += 1
                actor_norms.append(float(torch.as_tensor(actor_norm).item()))
                mini_batches_processed += 1
                policy_total += float(policy_loss.item()) * (size / n)
                entropy_total += float(entropy_loss.item()) * (size / n)

                recent_kl = float(sum(kls[-size:]) / size)
                if recent_kl > self.max_kl:
                    stopped = True
                    break

            g_batch, h_order_batch, h_batch = self._current_critic_inputs()
            values = self.shared_critic(
                g_batch.detach(), h_order_batch.detach(), h_batch.detach()
            ).squeeze(-1)
            value_loss = F.smooth_l1_loss(values, returns, beta=1.0)
            self.optimizer_critic.zero_grad()
            (self.c_val * value_loss).backward()
            critic_norm = nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
            if torch.isfinite(torch.as_tensor(critic_norm)):
                self.optimizer_critic.step()
                critic_optimizer_steps += 1
            critic_norms.append(float(torch.as_tensor(critic_norm).item()))

            mean_kl = float(sum(kls) / len(kls)) if kls else 0.0
            if self.target_kl is not None and mean_kl > 1.5 * self.target_kl:
                stopped = True
            mean_entropy = -entropy_total
            processed_transition_evaluations = sum(
                values["evaluations"] for values in type_diagnostics.values()
            )

            def _type_mean(decision_type: str, key: str) -> float:
                values = type_diagnostics[decision_type]
                return float(values[key] / max(1.0, values["evaluations"]))

            last_metrics = {
                "total_loss": policy_total + self.c_val * float(value_loss.item()),
                "single_policy_loss": policy_total,
                "single_value_loss": float(value_loss.item()),
                "single_entropy": mean_entropy,
                "single_clip_fraction": float(sum(clipped) / len(clipped)) if clipped else 0.0,
                "single_ratio_mean": float(sum(ratios) / len(ratios)) if ratios else 1.0,
                "single_approx_kl": mean_kl,
                "grad_norm_actor": float(sum(actor_norms) / len(actor_norms)) if actor_norms else 0.0,
                "grad_norm_critic": float(sum(critic_norms) / len(critic_norms)) if critic_norms else 0.0,
                "c_ent": float(self.c_ent),
                "split_entropy_coeff_effective": float(
                    sum(adaptive_entropy_values[SPLIT_ACTION])
                    / max(1, len(adaptive_entropy_values[SPLIT_ACTION]))
                ),
                "placement_entropy_coeff_effective": float(
                    sum(adaptive_entropy_values[PLACEMENT_ACTION])
                    / max(1, len(adaptive_entropy_values[PLACEMENT_ACTION]))
                ),
                "ppo_epochs": float(self.ppo_epochs),
                "ppo_epochs_completed": float(epoch + 1),
                "kl_early_stopped": float(stopped),
                "num_transitions": float(n),
                "split_transition_count": float(type_counts[SPLIT_ACTION]),
                "placement_transition_count": float(type_counts[PLACEMENT_ACTION]),
                "split_transition_ratio": float(type_counts[SPLIT_ACTION] / n),
                "placement_transition_ratio": float(type_counts[PLACEMENT_ACTION] / n),
                "split_advantage_raw_mean": _type_tensor_stat(
                    raw_advantages, SPLIT_ACTION, "mean"
                ),
                "split_advantage_raw_std": _type_tensor_stat(
                    raw_advantages, SPLIT_ACTION, "std"
                ),
                "placement_advantage_raw_mean": _type_tensor_stat(
                    raw_advantages, PLACEMENT_ACTION, "mean"
                ),
                "placement_advantage_raw_std": _type_tensor_stat(
                    raw_advantages, PLACEMENT_ACTION, "std"
                ),
                "split_advantage_normalized_mean": _type_tensor_stat(
                    advantages, SPLIT_ACTION, "mean"
                ),
                "split_advantage_normalized_std": _type_tensor_stat(
                    advantages, SPLIT_ACTION, "std"
                ),
                "placement_advantage_normalized_mean": _type_tensor_stat(
                    advantages, PLACEMENT_ACTION, "mean"
                ),
                "placement_advantage_normalized_std": _type_tensor_stat(
                    advantages, PLACEMENT_ACTION, "std"
                ),
                "split_policy_loss_diagnostic": _type_mean(
                    SPLIT_ACTION, "policy_loss_sum"
                ),
                "placement_policy_loss_diagnostic": _type_mean(
                    PLACEMENT_ACTION, "policy_loss_sum"
                ),
                "split_entropy_diagnostic": _type_mean(SPLIT_ACTION, "entropy_sum"),
                "placement_entropy_diagnostic": _type_mean(
                    PLACEMENT_ACTION, "entropy_sum"
                ),
                "split_approx_kl_diagnostic": _type_mean(SPLIT_ACTION, "kl_sum"),
                "placement_approx_kl_diagnostic": _type_mean(
                    PLACEMENT_ACTION, "kl_sum"
                ),
                "split_clip_fraction_diagnostic": _type_mean(
                    SPLIT_ACTION, "clip_sum"
                ),
                "placement_clip_fraction_diagnostic": _type_mean(
                    PLACEMENT_ACTION, "clip_sum"
                ),
                "split_ratio_mean_diagnostic": _type_mean(SPLIT_ACTION, "ratio_sum"),
                "placement_ratio_mean_diagnostic": _type_mean(
                    PLACEMENT_ACTION, "ratio_sum"
                ),
                "actor_optimizer_steps": float(actor_optimizer_steps),
                "critic_optimizer_steps": float(critic_optimizer_steps),
                "mini_batches_processed": float(mini_batches_processed),
                "processed_transition_evaluations": float(
                    processed_transition_evaluations
                ),
                # Compatibility keys for the unmodified experiment logger.
                "m_policy_loss": policy_total,
                "m_val_loss": float(value_loss.item()),
                "w_policy_loss": 0.0,
                "w_val_loss": 0.0,
                "m_entropy": mean_entropy,
                "w_entropy": mean_entropy,
            }
            if stopped:
                break

        self.rollout_buffer.reset()
        return last_metrics

    def step_entropy_anneal(self, factor: float = 0.9997, min_ent: float = 0.0005):
        self.c_ent = max(float(min_ent), self.c_ent * float(factor))

    def set_lr_multiplier(self, multiplier: float) -> None:
        value = float(multiplier)
        for group in self.optimizer_actor.param_groups:
            group["lr"] = self.lr * value
        for group in self.optimizer_critic.param_groups:
            group["lr"] = self.lr * value

    def save_checkpoint(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        filepath = os.path.abspath(filepath)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        contract_meta = dict(self.checkpoint_contract or {})
        external_meta = dict(extra_meta or {})
        authoritative_identity = {
            "method_id": self.method_id,
            "code_id": self.code_id,
            "ablation_scope": SINGLE_ACTOR_ABLATION_SCOPE,
            "credit_assignment": SINGLE_ACTOR_CREDIT_ASSIGNMENT,
            "event_reward_assignment": SINGLE_ACTOR_EVENT_REWARD_ASSIGNMENT,
            "objective_weighting": SINGLE_ACTOR_OBJECTIVE_WEIGHTING,
        }
        for source_name, source in (
            ("checkpoint_contract", contract_meta),
            ("extra_meta", external_meta),
        ):
            for key, expected in authoritative_identity.items():
                if key in source and source[key] != expected:
                    raise ValueError(
                        f"{source_name} cannot override C2 identity field {key!r}: "
                        f"received {source[key]!r}, expected {expected!r}."
                    )
        protected_contract_fields = (
            "stage",
            "seed",
            "protocol_version",
            "graph_schema_version",
            "dataset_manifest_sha256",
            "active_package_sha256",
        )
        for key in protected_contract_fields:
            if (
                key in contract_meta
                and key in external_meta
                and contract_meta[key] != external_meta[key]
            ):
                raise ValueError(
                    f"extra_meta conflicts with checkpoint_contract field {key!r}: "
                    f"{external_meta[key]!r} != {contract_meta[key]!r}."
                )
        payload: Dict[str, Any] = {
            "format_version": 4,
            "initial_solution_builder_version": INITIAL_SOLUTION_BUILDER_VERSION,
            "initial_solution_dispatch_rule": INITIAL_SOLUTION_DISPATCH_RULE,
            "protocol_version": FORMAL_DYNAMIC_PROTOCOL_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "template_encoding_version": TEMPLATE_ENCODING_VERSION,
            "template_feature_dim": TEMPLATE_FEATURE_DIM,
            "reward_contract_version": REWARD_CONTRACT_VERSION,
            "pool_type": self.pool_type,
            "d_h": self.d_h,
            "actor_learning_rate": self.lr,
            "actor_lr_semantics": SINGLE_ACTOR_LR_SEMANTICS,
            "entropy_regularization": SINGLE_ACTOR_ENTROPY_REGULARIZATION,
            "split_entropy_target": SPLIT_ENTROPY_TARGET,
            "placement_entropy_target": PLACEMENT_ENTROPY_TARGET,
            "single_actor_hidden": self.single_actor_policy.d_hidden,
            "single_actor_parameter_count": sum(
                p.numel() for p in self.single_actor_policy.parameters()
            ),
            "reference_dual_actor_parameter_count": self.single_actor_policy.reference_actor_parameters,
            "parameter_match_error": self.single_actor_policy.parameter_match_error,
            "hgnn": self.hgnn.state_dict(),
            "single_actor_policy": self.single_actor_policy.state_dict(),
            "shared_critic": self.shared_critic.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
        }
        payload.update(external_meta)
        payload.update(contract_meta)
        payload.update(authoritative_identity)
        for attempt in range(3):
            try:
                torch.save(payload, filepath)
                break
            except (PermissionError, OSError):
                if attempt == 2:
                    raise
                time.sleep(0.15)
        return filepath

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        filepath = os.path.abspath(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"SingleActor checkpoint does not exist: {filepath}")
        try:
            payload = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(filepath, map_location=self.device)
        if payload.get("method_id") != SINGLE_ACTOR_METHOD_ID:
            raise ValueError("Checkpoint is not an AD-HG-SAPPO single-Actor checkpoint.")
        if payload.get("code_id") != SINGLE_ACTOR_CODE_ID:
            raise ValueError("Checkpoint code_id is not the current C2 implementation.")
        if payload.get("ablation_scope") != SINGLE_ACTOR_ABLATION_SCOPE:
            raise ValueError("Checkpoint does not implement the current C2 architecture scope.")
        if payload.get("actor_lr_semantics") != SINGLE_ACTOR_LR_SEMANTICS:
            raise ValueError("Checkpoint does not use the current shared-Actor LR semantics.")
        if payload.get("entropy_regularization") != SINGLE_ACTOR_ENTROPY_REGULARIZATION:
            raise ValueError("Checkpoint does not use the current type-conditioned entropy contract.")
        if payload.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
            raise ValueError("Checkpoint graph schema is incompatible with the current project.")
        if payload.get("template_encoding_version") != TEMPLATE_ENCODING_VERSION:
            raise ValueError(
                f"Checkpoint template_encoding_version {payload.get('template_encoding_version')!r} "
                f"is incompatible with current {TEMPLATE_ENCODING_VERSION!r}. Old checkpoints with feature collisions are prohibited."
            )
        if payload.get("format_version") != 4:
            raise ValueError("C2 checkpoint format is obsolete; old checkpoints are forbidden.")
        if payload.get("initial_solution_builder_version") != INITIAL_SOLUTION_BUILDER_VERSION:
            raise ValueError(
                "C2 checkpoint initial-solution contract is obsolete; retrain from scratch."
            )
        if payload.get("initial_solution_dispatch_rule") != INITIAL_SOLUTION_DISPATCH_RULE:
            raise ValueError(
                "C2 checkpoint initial dispatch rule is obsolete; only SPT is allowed."
            )
        if payload.get("reward_contract_version") not in (None, REWARD_CONTRACT_VERSION):
            raise ValueError("C2 checkpoint reward contract is obsolete.")
        required = ("hgnn", "single_actor_policy", "shared_critic")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"SingleActor checkpoint is missing fields: {missing}")
        self.hgnn.load_state_dict(payload["hgnn"])
        self.single_actor_policy.load_state_dict(payload["single_actor_policy"])
        self.shared_critic.load_state_dict(payload["shared_critic"])
        if load_optimizer:
            if payload.get("optimizer_actor") is not None:
                self.optimizer_actor.load_state_dict(payload["optimizer_actor"])
            if payload.get("optimizer_critic") is not None:
                self.optimizer_critic.load_state_dict(payload["optimizer_critic"])
        return payload

