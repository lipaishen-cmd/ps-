"""Runner and HPPO trainer for the L11 (DAPPO-Expand) C2 ablation.

``DAPPO-Expand`` preserves the exact dual-Actor architecture, parameter count,
four independent Adam optimizers, and semi-Markov/single-step GAE updates of
``AD-HG-DAPPO``, while replacing on-demand progressive graph construction with
``PreExpandedPCBGraphBuilder`` (pre-expanding all candidate templates' subbatch
nodes during the Split stage).
"""

from __future__ import annotations

import functools
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
for _path in (
    _THIS_DIR,
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "agent"),
    os.path.join(_PROJECT_ROOT, "environment"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch

from data_models import Order, Solution
from dynamic_event_manager import (
    DynamicEventManager,
    DynamicRescheduleContext,
    INITIAL_SOLUTION_BUILDER_VERSION,
    INITIAL_SOLUTION_DISPATCH_RULE,
)
from dynamic_two_agent_runner import DynamicDecision, DynamicTwoAgentRunner
from expanded_graph_builder import PreExpandedPCBGraphBuilder, TimedOnDemandPCBGraphBuilder
from graph_builder import PCBGraphBuilder
from hppo_trainer import (
    DUAL_ACTOR_LR_CONTRACT_VERSION,
    DUAL_ACTOR_SEMANTICS_VERSION,
    FORMAL_DYNAMIC_C_ENT,
    FORMAL_DYNAMIC_LR,
    FORMAL_DYNAMIC_PLACEMENT_LR,
    FORMAL_DYNAMIC_REWARD_SCALE,
    HPPOTrainer,
    MAIN_CHECKPOINT_FORMAT_VERSION,
)
from reward import REWARD_CONTRACT_VERSION
from template_encoding import TEMPLATE_ENCODING_VERSION

DAPPO_EXPAND_METHOD_ID = "DAPPO-Expand"
DAPPO_EXPAND_CODE_ID = "dappo_expand"
DAPPO_EXPAND_GRAPH_MODE = "pre_expand_all_candidate_templates"
DAPPO_EXPAND_GRAPH_SCHEMA_VERSION = "six-node-20-directed-template10-preexpanded-v20261118"


class InstrumentedTwoAgentRunner(DynamicTwoAgentRunner):
    """Dual-Actor runner with pluggable graph builder and graph/HGNN telemetry."""

    def __init__(
        self,
        *args,
        graph_builder_cls: Type[PCBGraphBuilder] = TimedOnDemandPCBGraphBuilder,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.graph_builder_cls = graph_builder_cls
        self._event_telemetry: Dict[str, Any] = self._empty_telemetry()

    @staticmethod
    def _empty_telemetry() -> Dict[str, Any]:
        return {
            "split_graph_calls": 0,
            "placement_graph_calls": 0,
            "split_graph_build_ms": 0.0,
            "placement_graph_build_ms": 0.0,
            "split_hgnn_forward_ms": 0.0,
            "placement_hgnn_forward_ms": 0.0,
            "split_subbatch_nodes_list": [],
            "split_total_edges_list": [],
            "placement_subbatch_nodes_list": [],
            "placement_total_edges_list": [],
        }

    def _instrument_builder(self, builder: PCBGraphBuilder) -> None:
        orig_build_graph = builder.build_graph

        @functools.wraps(orig_build_graph)
        def wrapped_build_graph(*args, **kwargs):
            t0 = time.perf_counter()
            data = orig_build_graph(*args, **kwargs)
            build_ms = getattr(data, "graph_build_ms", (time.perf_counter() - t0) * 1000.0)
            n_sub = int(getattr(data, "num_subbatch_nodes", data["subbatch"].x.size(0)))
            n_edges = int(
                getattr(
                    data,
                    "num_total_edges",
                    sum(data[et].edge_index.size(1) for et in data.edge_types),
                )
            )
            is_placement = bool(kwargs.get("active_sub_batches")) or (
                len(args) >= 3 and bool(args[2])
            )
            if is_placement:
                self._event_telemetry["placement_graph_calls"] += 1
                self._event_telemetry["placement_graph_build_ms"] += float(build_ms)
                self._event_telemetry["placement_subbatch_nodes_list"].append(n_sub)
                self._event_telemetry["placement_total_edges_list"].append(n_edges)
            else:
                self._event_telemetry["split_graph_calls"] += 1
                self._event_telemetry["split_graph_build_ms"] += float(build_ms)
                self._event_telemetry["split_subbatch_nodes_list"].append(n_sub)
                self._event_telemetry["split_total_edges_list"].append(n_edges)
            return data

        builder.build_graph = wrapped_build_graph  # type: ignore[method-assign]

    def _build_graph_context(
        self,
        context: DynamicRescheduleContext,
        current_order: Order,
        solution_override: Optional[Solution] = None,
    ) -> Tuple[PCBGraphBuilder, int, torch.Tensor, torch.Tensor, torch.Tensor, Dict[Tuple[int, ...], int], Any]:
        graph_solution = solution_override if solution_override is not None else context.base_solution
        active_order_ids = {
            batch.order_id
            for queue in graph_solution.machine_queues.values()
            for batch in queue.batches
        }
        orders = sorted(
            context.orders.values(),
            key=lambda o: (0 if o.order_id in active_order_ids else 1, o.order_id),
        )
        orders = [o for o in orders if o.order_id != current_order.order_id] + [current_order]
        builder = self.graph_builder_cls(
            orders=orders,
            pool_type=self.manager.pool_type,
            machines_dict=self.manager.machines_dict,
            device=self.device,
        )
        self._instrument_builder(builder)
        frontier_idx = builder.order_to_idx[current_order.order_id]
        mask = self._template_mask(current_order, context)
        graph = builder.build_graph(
            solution=graph_solution,
            current_frontier_idx=frontier_idx,
            candidate_templates=list(mask),
            template_mask=mask,
            event_id=str(getattr(context, "event_id", "dynamic")),
            event_time=float(getattr(context, "event_time", 0.0)),
            scope_ratio=float(getattr(context, "scope_ratio", 0.0)),
        )
        with torch.no_grad():
            embeddings, global_embedding = self.hgnn(graph)
        order_embedding = embeddings["order"][frontier_idx: frontier_idx + 1]
        template_embeddings = embeddings["template"]
        return builder, frontier_idx, global_embedding, order_embedding, template_embeddings, mask, graph

    def run_next_event(
        self,
        deterministic: bool = True,
        context: Optional[DynamicRescheduleContext] = None,
    ) -> DynamicDecision:
        self._event_telemetry = self._empty_telemetry()
        hgnn_timer: Dict[str, Any] = {"start": 0.0, "stage": "split"}

        def _pre_hook(_module, inputs):
            hgnn_timer["start"] = time.perf_counter()
            data = inputs[0] if inputs else None
            is_place = False
            if data is not None and hasattr(data, "event_context") and data.event_context is not None:
                is_place = float(data.event_context[0, 2].item()) >= 0.5
            hgnn_timer["stage"] = "placement" if is_place else "split"

        def _post_hook(_module, _inputs, _outputs):
            dt_ms = (time.perf_counter() - hgnn_timer["start"]) * 1000.0
            if hgnn_timer["stage"] == "placement":
                self._event_telemetry["placement_hgnn_forward_ms"] += dt_ms
            else:
                self._event_telemetry["split_hgnn_forward_ms"] += dt_ms

        pre_handle = self.hgnn.register_forward_pre_hook(_pre_hook)
        post_handle = self.hgnn.register_forward_hook(_post_hook)
        try:
            decision = super().run_next_event(deterministic=deterministic, context=context)
        finally:
            pre_handle.remove()
            post_handle.remove()

        tel = self._event_telemetry
        split_nodes = tel["split_subbatch_nodes_list"]
        split_edges = tel["split_total_edges_list"]
        place_nodes = tel["placement_subbatch_nodes_list"]
        place_edges = tel["placement_total_edges_list"]
        decision.graph_telemetry = {  # type: ignore[attr-defined]
            "graph_construction_mode": getattr(self.graph_builder_cls, "__name__", "unknown"),
            "split_graph_calls": tel["split_graph_calls"],
            "placement_graph_calls": tel["placement_graph_calls"],
            "split_graph_build_ms": tel["split_graph_build_ms"],
            "placement_graph_build_ms": tel["placement_graph_build_ms"],
            "event_graph_build_ms": tel["split_graph_build_ms"] + tel["placement_graph_build_ms"],
            "split_hgnn_forward_ms": tel["split_hgnn_forward_ms"],
            "placement_hgnn_forward_ms": tel["placement_hgnn_forward_ms"],
            "event_hgnn_forward_ms": tel["split_hgnn_forward_ms"] + tel["placement_hgnn_forward_ms"],
            "split_subbatch_nodes_mean": sum(split_nodes) / len(split_nodes) if split_nodes else 0.0,
            "split_subbatch_nodes_max": max(split_nodes) if split_nodes else 0,
            "split_total_edges_mean": sum(split_edges) / len(split_edges) if split_edges else 0.0,
            "split_total_edges_max": max(split_edges) if split_edges else 0,
            "placement_subbatch_nodes_mean": sum(place_nodes) / len(place_nodes) if place_nodes else 0.0,
            "placement_total_edges_mean": sum(place_edges) / len(place_edges) if place_edges else 0.0,
        }
        return decision


class ExpandedTwoAgentRunner(InstrumentedTwoAgentRunner):
    """Dual-Actor runner fixed to pre-expanded candidate templates in Split stage."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("graph_builder_cls", PreExpandedPCBGraphBuilder)
        super().__init__(*args, **kwargs)


class DAPPOExpandTrainer(HPPOTrainer):
    """L11 C2 ablation trainer: identical to HPPOTrainer except PreExpandedPCBGraphBuilder."""

    method_id = DAPPO_EXPAND_METHOD_ID
    code_id = DAPPO_EXPAND_CODE_ID
    graph_construction_mode = DAPPO_EXPAND_GRAPH_MODE

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
    ) -> None:
        super().__init__(
            orders=orders,
            pool_type=pool_type,
            d_h=d_h,
            lr=lr,
            placement_lr=placement_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            c_val=c_val,
            c_ent=c_ent,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            device=device,
            reward_mode=reward_mode,
            target_kl=target_kl,
            max_kl=max_kl,
        )
        self.graph_schema_version = DAPPO_EXPAND_GRAPH_SCHEMA_VERSION
        self.graph_builder = PreExpandedPCBGraphBuilder(
            orders=self.orders, pool_type=pool_type, device=device
        )

    def collect_dynamic_episode(
        self,
        manager: DynamicEventManager,
        deterministic: bool = False,
        reward_scale: float = FORMAL_DYNAMIC_REWARD_SCALE,
        clear_buffer: bool = True,
        record_rollout: bool = True,
    ) -> List[DynamicDecision]:
        runner = ExpandedTwoAgentRunner(
            manager=manager,
            hgnn=self.hgnn,
            manager_policy=self.manager_policy,
            worker_policy=self.worker_policy,
            value_critic=self.shared_critic,
            device=self.device,
            reward_mode=self.reward_mode,
            reward_scale=reward_scale,
        )
        if clear_buffer and record_rollout:
            self.manager_buffer.reset()
            self.worker_buffer.reset()
        decisions: List[DynamicDecision] = []
        while manager.pending_event_ids:
            decision = runner.run_next_event(deterministic=deterministic)
            decisions.append(decision)
            is_last_event = len(manager.pending_event_ids) == 0
            if record_rollout and decision.trainable:
                scaled_reward = float(decision.reward)
                n_wr = len(decision.worker_records)
                for idx, record in enumerate(decision.worker_records):
                    if "step_reward" not in record:
                        raise RuntimeError("Placement transition is missing reward.py step_reward.")
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
                self.worker_buffer.commit_transaction(
                    done=is_last_event,
                    terminal_reward=scaled_reward,
                )
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
                manager.rollback_active()
                self.worker_buffer.abort_transaction()
                if record_rollout:
                    if len(self.manager_buffer) > 0:
                        self.manager_buffer.mark_last_done()
                    if len(self.worker_buffer) > 0:
                        self.worker_buffer.mark_last_done()
                break
        return decisions

    def save_checkpoint(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        filepath = os.path.abspath(filepath)
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        external_meta = dict(extra_meta or {})
        authoritative_identity = {
            "method_id": self.method_id,
            "code_id": self.code_id,
            "graph_construction_mode": self.graph_construction_mode,
            "graph_schema_version": self.graph_schema_version,
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
                    f"extra_meta cannot override DAPPO-Expand identity field {key!r}: "
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
            "graph_schema_version": self.graph_schema_version,
            "graph_construction_mode": self.graph_construction_mode,
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
                    print(f"[警告] DAPPO-Expand 断点写入受限 ({exc})，已跳过本次写入。")
                else:
                    time.sleep(0.15)
        return filepath

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, object]:
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型权重不存在: {filepath}")
        try:
            payload = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(filepath, map_location=self.device)
        if not isinstance(payload, dict):
            raise ValueError("模型权重文件格式错误：顶层必须为字典。")
        if payload.get("method_id") != self.method_id:
            raise ValueError("Checkpoint 不是 L11 DAPPO-Expand 消融权重。")
        if payload.get("code_id") != self.code_id:
            raise ValueError("Checkpoint code_id 与 DAPPO-Expand 实现不一致。")
        if payload.get("graph_construction_mode") != self.graph_construction_mode:
            raise ValueError(
                "Checkpoint 构图模式不是 pre_expand_all_candidate_templates，拒绝加载。"
            )
        if payload.get("graph_schema_version") != self.graph_schema_version:
            raise ValueError(
                f"Checkpoint graph_schema_version 不匹配：期望 {self.graph_schema_version}。"
            )
        if payload.get("dual_actor_semantics_version") != DUAL_ACTOR_SEMANTICS_VERSION:
            raise ValueError("Checkpoint dual_actor_semantics_version 不匹配。")
        if payload.get("dual_actor_lr_contract_version") != DUAL_ACTOR_LR_CONTRACT_VERSION:
            raise ValueError("Checkpoint dual_actor_lr_contract_version 不匹配。")
        if payload.get("format_version") != MAIN_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Checkpoint format_version 不匹配。")
        expected_learning_rates = {
            "split_learning_rate": self.lr,
            "placement_learning_rate": self.worker_lr,
            "encoder_learning_rate": self.lr,
            "critic_learning_rate": self.lr,
        }
        mismatched = {
            key: (payload.get(key), expected)
            for key, expected in expected_learning_rates.items()
            if payload.get(key) != expected
        }
        if mismatched:
            raise ValueError(f"Checkpoint 学习率不匹配: {mismatched}")
        if payload.get("initial_solution_builder_version") != INITIAL_SOLUTION_BUILDER_VERSION:
            raise ValueError("Checkpoint initial_solution_builder_version 不匹配。")
        if payload.get("initial_solution_dispatch_rule") != INITIAL_SOLUTION_DISPATCH_RULE:
            raise ValueError("Checkpoint initial_solution_dispatch_rule 不匹配。")
        if payload.get("template_encoding_version") != TEMPLATE_ENCODING_VERSION:
            raise ValueError("Checkpoint template_encoding_version 不匹配。")
        if payload.get("reward_contract_version") != REWARD_CONTRACT_VERSION:
            raise ValueError("Checkpoint reward_contract_version 不匹配。")

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
