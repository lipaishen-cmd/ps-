"""Pre-expanded heterogeneous graph builder for the L11 (DAPPO-Expand) C2 ablation.

Unlike the main method's on-demand progressive graph construction
(``PCBGraphBuilder``), where Split-stage graphs keep the ``subbatch`` node store
empty until ``SplitPolicy`` picks one template, ``PreExpandedPCBGraphBuilder``
pre-expands hypothetical ``subbatch`` nodes and their incident edges for *all*
candidate partition templates of the current order during the Split stage.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
for _path in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "agent"), os.path.join(_PROJECT_ROOT, "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch
from torch_geometric.data import HeteroData

from data_models import Batch, Order, Solution
from graph_builder import (
    NORM_MAX_SPINDLES,
    NORM_WORKLOAD_HOURS,
    PCBGraphBuilder,
    Template,
)


def attach_graph_telemetry(
    data: HeteroData,
    graph_build_ms: float,
    mode: str,
) -> HeteroData:
    """Attach graph size and build latency telemetry onto a HeteroData object."""
    data.graph_build_ms = float(graph_build_ms)
    data.num_subbatch_nodes = int(data["subbatch"].x.size(0))
    data.num_total_edges = int(sum(data[et].edge_index.size(1) for et in data.edge_types))
    data.graph_construction_mode = str(mode)
    return data


class TimedOnDemandPCBGraphBuilder(PCBGraphBuilder):
    """On-demand PCBGraphBuilder with identical build-time telemetry for fair comparison."""

    def build_graph(
        self,
        solution: Solution,
        current_frontier_idx: int,
        active_sub_batches: Optional[Sequence[Batch]] = None,
        current_sub_batch_idx: int = 0,
        candidate_templates: Optional[Sequence[Template]] = None,
        template_mask: Optional[Mapping[Template, int]] = None,
        selected_template: Optional[Template] = None,
        event_id: Optional[str] = None,
        event_time: float = 0.0,
        scope_ratio: float = 0.0,
    ) -> HeteroData:
        t0 = time.perf_counter()
        data = super().build_graph(
            solution=solution,
            current_frontier_idx=current_frontier_idx,
            active_sub_batches=active_sub_batches,
            current_sub_batch_idx=current_sub_batch_idx,
            candidate_templates=candidate_templates,
            template_mask=template_mask,
            selected_template=selected_template,
            event_id=event_id,
            event_time=event_time,
            scope_ratio=scope_ratio,
        )
        return attach_graph_telemetry(
            data,
            graph_build_ms=(time.perf_counter() - t0) * 1000.0,
            mode="on_demand_progressive",
        )


class PreExpandedPCBGraphBuilder(PCBGraphBuilder):
    """L11 C2 ablation: pre-expand all candidate templates' subbatch nodes in Split stage."""

    def __init__(
        self,
        *args,
        expand_unselected_in_placement: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.expand_unselected_in_placement = bool(expand_unselected_in_placement)

    def _append_pre_expanded_templates(
        self,
        data: HeteroData,
        solution: Solution,
        current_frontier_idx: int,
        current_order: Order,
        templates: Sequence[Template],
        event_id: Optional[str],
    ) -> None:
        stable_event_id = str(event_id or f"static:{self.pool_type}:{current_frontier_idx}")
        order_workload = (
            current_order.N / max(1.0, current_order.efficiency)
        ) / NORM_WORKLOAD_HOURS

        subbatch_rows: List[List[float]] = []
        template_ids: List[int] = []
        stable_uids: List[Tuple[str, str, int, int]] = []

        order_subbatch_edges: List[Tuple[int, int]] = []
        template_subbatch_edges: List[Tuple[int, int]] = []
        subbatch_machine_edges: List[Tuple[int, int]] = []
        subbatch_slot_edges: List[Tuple[int, int]] = []
        precedence_edges: List[Tuple[int, int]] = []

        for tpl_idx, tpl in enumerate(templates):
            tpl_len = max(1.0, float(len(tpl)))
            prev_subbatch_idx: Optional[int] = None
            for part_idx, raw_stack in enumerate(tpl):
                stack_count = int(raw_stack)
                subbatch_idx = len(subbatch_rows)
                subbatch_rows.append(
                    [
                        stack_count / NORM_MAX_SPINDLES,
                        part_idx / tpl_len,
                        stack_count / NORM_MAX_SPINDLES,
                        order_workload,
                        0.0,
                    ]
                )
                template_ids.append(tpl_idx)
                stable_uids.append((stable_event_id, current_order.order_id, tpl_idx, part_idx))

                order_subbatch_edges.append((current_frontier_idx, subbatch_idx))
                template_subbatch_edges.append((tpl_idx, subbatch_idx))
                if prev_subbatch_idx is not None:
                    precedence_edges.append((prev_subbatch_idx, subbatch_idx))
                prev_subbatch_idx = subbatch_idx

                for machine_id in current_order.eligible_machines:
                    machine = self.machines_dict.get(machine_id)
                    if machine is None or machine.spindles < stack_count:
                        continue
                    subbatch_machine_edges.append((subbatch_idx, self.machine_to_idx[machine_id]))
                    q_len = len(solution.machine_queues[machine_id].batches)
                    for position in range(q_len + 1):
                        subbatch_slot_edges.append((subbatch_idx, self.slot_map[(machine_id, position)]))

        data["subbatch"].x = torch.tensor(
            subbatch_rows, dtype=torch.float32, device=self.device
        ).reshape((len(subbatch_rows), self.SUBBATCH_FEATURE_DIM))
        data["subbatch"].template_id = torch.tensor(
            template_ids, dtype=torch.long, device=self.device
        )
        data["subbatch"].stable_uid = stable_uids

        self._set_relation_pair(data, "order_subbatch", order_subbatch_edges)
        self._set_relation_pair(data, "template_subbatch", template_subbatch_edges)
        self._set_relation_pair(data, "subbatch_machine", subbatch_machine_edges)
        self._set_relation_pair(data, "subbatch_precedence", precedence_edges)
        self._set_relation_pair(data, "subbatch_slot", subbatch_slot_edges)

    def build_graph(
        self,
        solution: Solution,
        current_frontier_idx: int,
        active_sub_batches: Optional[Sequence[Batch]] = None,
        current_sub_batch_idx: int = 0,
        candidate_templates: Optional[Sequence[Template]] = None,
        template_mask: Optional[Mapping[Template, int]] = None,
        selected_template: Optional[Template] = None,
        event_id: Optional[str] = None,
        event_time: float = 0.0,
        scope_ratio: float = 0.0,
    ) -> HeteroData:
        """Build the 6-node HeteroData graph, pre-expanding all templates in Split stage."""
        t0 = time.perf_counter()
        data = super().build_graph(
            solution=solution,
            current_frontier_idx=current_frontier_idx,
            active_sub_batches=active_sub_batches,
            current_sub_batch_idx=current_sub_batch_idx,
            candidate_templates=candidate_templates,
            template_mask=template_mask,
            selected_template=selected_template,
            event_id=event_id,
            event_time=event_time,
            scope_ratio=scope_ratio,
        )

        templates = list(candidate_templates or ())
        current_order = (
            self.orders[current_frontier_idx]
            if current_frontier_idx < len(self.orders)
            else None
        )
        is_split_stage = not bool(active_sub_batches)
        if is_split_stage and current_order is not None and templates:
            self._append_pre_expanded_templates(
                data=data,
                solution=solution,
                current_frontier_idx=current_frontier_idx,
                current_order=current_order,
                templates=templates,
                event_id=event_id,
            )

        if tuple(data.node_types) != self.NODE_TYPES or set(data.edge_types) != set(self.EDGE_TYPES):
            raise RuntimeError("Pre-expanded graph schema drifted from the authority-locked metadata.")

        return attach_graph_telemetry(
            data,
            graph_build_ms=(time.perf_counter() - t0) * 1000.0,
            mode="pre_expand_all_candidate_templates",
        )
