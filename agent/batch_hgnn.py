"""PCB-BatchHGNN over the authority-locked six-node graph schema."""

from __future__ import annotations

import os
from typing import Dict, FrozenSet, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter


def _configure_small_graph_cpu_runtime() -> None:
    """Bound CPU parallelism for the small-graph HGNN workload.

    The six-node-type graphs used by this project are small, while each
    decision can invoke many scatter and layer-normalisation operations.
    Letting the BLAS/OpenMP runtime use its default thread pool causes severe
    oversubscription on Windows and was observed to turn a single validation
    scene into an unbounded-looking run.  Keep the safe default at one thread,
    allow an explicit override for controlled benchmarking, and leave CUDA
    execution unchanged.
    """
    raw_threads = os.environ.get("PCB_TORCH_NUM_THREADS", "1")
    try:
        threads = max(1, min(32, int(raw_threads)))
    except ValueError:
        threads = 1
    torch.set_num_threads(threads)
    try:
        # This can only be set before inter-op work starts.  A later import in
        # an already-running process must not turn a valid run into an error.
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


_configure_small_graph_cpu_runtime()


class PCBBatchHGNN(nn.Module):
    """Encode pool, order, template, subbatch, machine and slot nodes.

    The final ``pool`` representation is the sole global state readout. The
    model never creates an event node and accepts zero-length template or
    subbatch stores without changing its metadata contract.
    """

    def __init__(
        self,
        d_pool: int = 6,
        d_order: int = 17,
        d_template: int = 10,
        d_subbatch: int = 5,
        d_machine: int = 10,
        d_slot: int = 5,
        d_h: int = 64,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.proj_pool = self._projection(d_pool, d_h)
        self.proj_order = self._projection(d_order, d_h)
        self.proj_template = self._projection(d_template, d_h)
        self.proj_subbatch = self._projection(d_subbatch, d_h)
        self.proj_machine = self._projection(d_machine, d_h)
        self.proj_slot = self._projection(d_slot, d_h)

        self.w_order_template = nn.Linear(d_h, d_h)
        self.w_template_order = nn.Linear(d_h, d_h)
        self.w_order_subbatch = nn.Linear(2 * d_h, d_h)
        self.w_subbatch_order = nn.Linear(d_h, d_h)
        self.w_subbatch_precedence = nn.Linear(d_h, d_h)
        self.w_machine_attention = nn.Linear(d_h, d_h)
        self.w_subbatch_attention = nn.Linear(d_h, d_h)
        self.attn_subbatch_machine = nn.Parameter(torch.empty(1, d_h))
        self.w_subbatch_machine = nn.Linear(d_h, d_h)
        self.w_machine_slot = nn.Linear(2 * d_h, d_h)
        self.w_template_subbatch = nn.Linear(d_h, d_h)
        self.w_subbatch_template = nn.Linear(d_h, d_h)
        self.w_order_machine = nn.Linear(d_h, d_h)
        self.w_machine_order = nn.Linear(d_h, d_h)
        self.w_subbatch_slot = nn.Linear(2 * d_h, d_h)
        self.w_order_pool = nn.Linear(d_h, d_h)
        self.w_machine_pool = nn.Linear(d_h, d_h)
        self.w_template_pool = nn.Linear(d_h, d_h)

        self.ln_order_template = nn.LayerNorm(d_h)
        self.ln_template = nn.LayerNorm(d_h)
        self.ln_template_subbatch = nn.LayerNorm(d_h)
        self.ln_order_subbatch = nn.LayerNorm(d_h)
        self.ln_subbatch = nn.LayerNorm(d_h)
        self.ln_machine = nn.LayerNorm(d_h)
        self.ln_machine_order = nn.LayerNorm(d_h)
        self.ln_slot = nn.LayerNorm(d_h)
        self.ln_slot_subbatch = nn.LayerNorm(d_h)
        self.ln_pool = nn.LayerNorm(d_h)
        self.proj_event_context = nn.Sequential(nn.Linear(5, d_h), nn.LayerNorm(d_h), nn.ReLU())
        self.pool_readout = nn.Sequential(nn.Linear(d_h, 2 * d_h), nn.LayerNorm(2 * d_h), nn.ReLU())
        self.mlp_match = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        nn.init.xavier_uniform_(self.attn_subbatch_machine)
        self.last_consumed_edge_types: FrozenSet[Tuple[str, str, str]] = frozenset()

    PROPAGATED_EDGE_TYPES: FrozenSet[Tuple[str, str, str]] = frozenset({
        ("pool", "contains", "order"), ("order", "in_pool", "pool"),
        ("pool", "contains", "machine"), ("machine", "in_pool", "pool"),
        ("order", "has_candidate", "template"), ("template", "candidate_for", "order"),
        ("order", "contains", "subbatch"), ("subbatch", "belongs_to", "order"),
        ("template", "instantiates", "subbatch"), ("subbatch", "instantiated_from", "template"),
        ("order", "eligible_for", "machine"), ("machine", "eligible_order", "order"),
        ("subbatch", "fits_machine", "machine"), ("machine", "fits_subbatch", "subbatch"),
        ("machine", "has_slot", "slot"), ("slot", "hosted_by", "machine"),
        ("subbatch", "precedes", "subbatch"), ("subbatch", "follows", "subbatch"),
        ("subbatch", "can_insert", "slot"), ("slot", "can_receive", "subbatch"),
    })

    @staticmethod
    def _projection(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU())

    @staticmethod
    def _has_edges(data: HeteroData, edge_type: Tuple[str, str, str]) -> bool:
        return edge_type in data.edge_types and data[edge_type].edge_index.size(1) > 0

    def forward(self, data: HeteroData) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Return all six node embeddings and the pool-derived global readout."""
        required = {"pool", "order", "template", "subbatch", "machine", "slot"}
        missing = required.difference(data.node_types)
        if missing:
            raise ValueError(f"HGNN requires the six-node graph schema; missing {sorted(missing)}.")

        h_pool = self.proj_pool(data["pool"].x)
        if hasattr(data, "event_context") and data.event_context is not None:
            ctx = data.event_context.to(h_pool.device)
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            h_pool = h_pool + self.proj_event_context(ctx)
        h_order = self.proj_order(data["order"].x)
        h_template = self.proj_template(data["template"].x)
        h_subbatch = self.proj_subbatch(data["subbatch"].x)
        h_machine = self.proj_machine(data["machine"].x)
        h_slot = self.proj_slot(data["slot"].x)
        consumed: Set[Tuple[str, str, str]] = set()

        def use(edge_type: Tuple[str, str, str]):
            consumed.add(edge_type)
            return data[edge_type].edge_index

        edge_po = ("pool", "contains", "order")
        source, target = use(edge_po)
        if source.numel():
            h_order = self.ln_order_subbatch(
                h_order + scatter(
                    F.relu(self.w_order_pool(h_pool[source])), target, dim=0,
                    dim_size=h_order.size(0), reduce="mean",
                )
            )

        edge_pm = ("pool", "contains", "machine")
        source, target = use(edge_pm)
        if source.numel():
            h_machine = self.ln_machine(
                h_machine + scatter(
                    F.relu(self.w_machine_pool(h_pool[source])), target, dim=0,
                    dim_size=h_machine.size(0), reduce="mean",
                )
            )

        edge_ot = ("order", "has_candidate", "template")
        reverse_ot = ("template", "candidate_for", "order")
        source, target = use(edge_ot)
        reverse_source, reverse_target = use(reverse_ot)
        if self._has_edges(data, edge_ot):
            template_message = scatter(
                F.relu(self.w_order_template(h_order[source])), target, dim=0,
                dim_size=h_template.size(0), reduce="mean",
            )
            h_template = self.ln_template(h_template + template_message)
            order_message = scatter(
                F.relu(self.w_template_order(h_template[reverse_source])), reverse_target, dim=0,
                dim_size=h_order.size(0), reduce="mean",
            )
            h_order = self.ln_order_template(h_order + order_message)

        edge_os = ("order", "contains", "subbatch")
        reverse_os = ("subbatch", "belongs_to", "order")
        source, target = use(edge_os)
        reverse_source, reverse_target = use(reverse_os)
        if self._has_edges(data, edge_os):
            subbatch_message = scatter(
                F.relu(self.w_order_subbatch(torch.cat((h_order[source], h_subbatch[target]), dim=-1))),
                target, dim=0, dim_size=h_subbatch.size(0), reduce="mean",
            )
            h_subbatch = self.ln_subbatch(h_subbatch + subbatch_message)
            order_message = scatter(
                F.relu(self.w_subbatch_order(h_subbatch[reverse_source])), reverse_target, dim=0,
                dim_size=h_order.size(0), reduce="mean",
            )
            h_order = self.ln_order_subbatch(h_order + order_message)

        edge_ss = ("subbatch", "precedes", "subbatch")
        reverse_ss = ("subbatch", "follows", "subbatch")
        source, target = use(edge_ss)
        reverse_source, reverse_target = use(reverse_ss)
        if self._has_edges(data, edge_ss):
            precedence_message = scatter(
                F.relu(self.w_subbatch_precedence(h_subbatch[source])), target, dim=0,
                dim_size=h_subbatch.size(0), reduce="mean",
            )
            if reverse_source.numel():
                precedence_message = precedence_message + scatter(
                    F.relu(self.w_subbatch_precedence(h_subbatch[reverse_source])), reverse_target,
                    dim=0, dim_size=h_subbatch.size(0), reduce="mean",
                )
            h_subbatch = self.ln_subbatch(h_subbatch + precedence_message)

        edge_sm = ("subbatch", "fits_machine", "machine")
        reverse_sm = ("machine", "fits_subbatch", "subbatch")
        source, target = use(edge_sm)
        reverse_source, reverse_target = use(reverse_sm)
        if self._has_edges(data, edge_sm):
            attention = F.leaky_relu(
                self.w_machine_attention(h_machine[target]) + self.w_subbatch_attention(h_subbatch[source])
            ).mul(self.attn_subbatch_machine).sum(dim=-1, keepdim=True)
            attention = torch.exp(torch.clamp(attention, min=-10.0, max=10.0))
            denominator = scatter(
                attention, target, dim=0, dim_size=h_machine.size(0), reduce="sum",
            ).clamp_min(1e-8)
            machine_message = scatter(
                F.relu(self.w_subbatch_machine(h_subbatch[source])) * (attention / denominator[target]),
                target, dim=0, dim_size=h_machine.size(0), reduce="sum",
            )
            h_machine = self.ln_machine(h_machine + machine_message)
            if reverse_source.numel():
                h_subbatch = self.ln_subbatch(
                    h_subbatch + scatter(
                        F.relu(self.w_subbatch_machine(h_machine[reverse_source])), reverse_target,
                        dim=0, dim_size=h_subbatch.size(0), reduce="mean",
                    )
                )

        edge_ms = ("machine", "has_slot", "slot")
        reverse_ms = ("slot", "hosted_by", "machine")
        source, target = use(edge_ms)
        reverse_source, reverse_target = use(reverse_ms)
        if self._has_edges(data, edge_ms):
            slot_message = scatter(
                F.relu(self.w_machine_slot(torch.cat((h_machine[source], h_slot[target]), dim=-1))),
                target, dim=0, dim_size=h_slot.size(0), reduce="mean",
            )
            h_slot = self.ln_slot(h_slot + slot_message)
            if reverse_source.numel():
                h_machine = self.ln_machine(
                    h_machine + scatter(
                        F.relu(self.w_machine_slot(torch.cat(
                            (h_machine[reverse_target], h_slot[reverse_source]), dim=-1
                        ))), reverse_target, dim=0, dim_size=h_machine.size(0), reduce="mean",
                    )
                )

        edge_ts = ("template", "instantiates", "subbatch")
        reverse_ts = ("subbatch", "instantiated_from", "template")
        source, target = use(edge_ts)
        reverse_source, reverse_target = use(reverse_ts)
        if self._has_edges(data, edge_ts):
            subbatch_message = scatter(
                F.relu(self.w_template_subbatch(h_template[source])), target, dim=0,
                dim_size=h_subbatch.size(0), reduce="mean",
            )
            h_subbatch = self.ln_subbatch(h_subbatch + subbatch_message)
            template_message = scatter(
                F.relu(self.w_subbatch_template(h_subbatch[reverse_source])), reverse_target, dim=0,
                dim_size=h_template.size(0), reduce="mean",
            )
            h_template = self.ln_template_subbatch(h_template + template_message)

        edge_om = ("order", "eligible_for", "machine")
        reverse_om = ("machine", "eligible_order", "order")
        source, target = use(edge_om)
        reverse_source, reverse_target = use(reverse_om)
        if self._has_edges(data, edge_om):
            machine_message = scatter(
                F.relu(self.w_order_machine(h_order[source])), target, dim=0,
                dim_size=h_machine.size(0), reduce="mean",
            )
            h_machine = self.ln_machine_order(h_machine + machine_message)
            order_message = scatter(
                F.relu(self.w_machine_order(h_machine[reverse_source])), reverse_target, dim=0,
                dim_size=h_order.size(0), reduce="mean",
            )
            h_order = self.ln_order_subbatch(h_order + order_message)

        edge_bslot = ("subbatch", "can_insert", "slot")
        reverse_bslot = ("slot", "can_receive", "subbatch")
        source, target = use(edge_bslot)
        reverse_source, reverse_target = use(reverse_bslot)
        if self._has_edges(data, edge_bslot):
            slot_message = scatter(
                F.relu(self.w_subbatch_slot(torch.cat((h_subbatch[source], h_slot[target]), dim=-1))),
                target, dim=0, dim_size=h_slot.size(0), reduce="mean",
            )
            h_slot = self.ln_slot_subbatch(h_slot + slot_message)
            if reverse_source.numel():
                h_subbatch = self.ln_subbatch(
                    h_subbatch + scatter(
                        F.relu(self.w_subbatch_slot(torch.cat(
                            (h_subbatch[reverse_target], h_slot[reverse_source]), dim=-1
                        ))), reverse_target, dim=0, dim_size=h_subbatch.size(0), reduce="mean",
                    )
                )

        # The single pool node gathers all pool-local state; no order/machine
        # mean/max concatenation is used as a competing global readout.
        pool_message = torch.zeros_like(h_pool)
        for edge_type, embedding, projector in (
            (("order", "in_pool", "pool"), h_order, self.w_order_pool),
            (("machine", "in_pool", "pool"), h_machine, self.w_machine_pool),
        ):
            use(edge_type)
            if self._has_edges(data, edge_type):
                source, target = data[edge_type].edge_index
                pool_message += scatter(
                    projector(embedding[source]), target, dim=0,
                    dim_size=h_pool.size(0), reduce="mean",
                )
        forward_ot = ("order", "has_candidate", "template")
        if self._has_edges(data, forward_ot):
            order_idx, template_idx = data[forward_ot].edge_index
            template_to_order = scatter(
                self.w_template_pool(h_template[template_idx]), order_idx, dim=0,
                dim_size=h_order.size(0), reduce="mean",
            )
            source, target = data[("order", "in_pool", "pool")].edge_index
            pool_message += scatter(
                template_to_order[source], target, dim=0,
                dim_size=h_pool.size(0), reduce="mean",
            )
        h_pool = self.ln_pool(h_pool + pool_message)
        global_readout = self.pool_readout(h_pool)
        self.last_consumed_edge_types = frozenset(consumed)
        if self.last_consumed_edge_types != self.PROPAGATED_EDGE_TYPES:
            missing = self.PROPAGATED_EDGE_TYPES.difference(self.last_consumed_edge_types)
            raise RuntimeError(f"HGNN relation propagation is incomplete: missing {sorted(missing)}")

        return {
            "pool": h_pool,
            "order": h_order,
            "template": h_template,
            "subbatch": h_subbatch,
            "machine": h_machine,
            "slot": h_slot,
        }, global_readout

    def compute_worker_match_scores(self, subbatch_emb: torch.Tensor, slot_emb: torch.Tensor) -> torch.Tensor:
        """Score the current subbatch against every slot before the hard mask."""
        if subbatch_emb.size(0) != 1:
            raise ValueError("Worker matching expects exactly one current subbatch embedding.")
        pair = torch.cat((subbatch_emb.expand(slot_emb.size(0), -1), slot_emb), dim=-1)
        return self.mlp_match(pair).sum(dim=-1)
