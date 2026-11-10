"""
PCB 钻孔车间异构图构建器（包含六类节点）。

图模式固定为 ``pool/order/template/subbatch/machine/slot``。事件是图级
``event_context``，而不是节点；Pool D 和 Pool B 始终在各自独立的图中构建。
在 Split 阶段，template 是与 SplitPolicy 动作索引一一对应的节点，subbatch
store 保持零长度；在 Placement 阶段，只有已选模板实例化的子批与其相关关系
被激活。所有 node store 及正反向 relation store 在两个阶段均显式存在。
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENVIRONMENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "environment"))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _path in (_THIS_DIR, _ENVIRONMENT_DIR, _ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch
from torch_geometric.data import HeteroData

from data_models import Batch, Machine, Order, POOL_D, Solution
from eligibility_builder import ALL_MACHINES
from time_model import TIME_PARAMS, setup_time
from template_encoding import TEMPLATE_ENCODING_VERSION, TEMPLATE_FEATURE_DIM, encode_template


NodeType = str
EdgeType = Tuple[NodeType, str, NodeType]
Template = Tuple[int, ...]
GRAPH_SCHEMA_VERSION = "six-node-20-directed-template10-v20261105"

# ===========================================================================
# 物理特征归一化参考常数（基于车间物理参数上界）
# ===========================================================================
NORM_STACK_COUNT_MAX = 5.0      # 叠板数最大上限 S_max
NORM_TOOL_CHANGES_MAX = 10.0    # 换刀次数归一化基准
NORM_WORKLOAD_HOURS = 100.0     # 工单钻孔工时基准 (分钟归一化)
NORM_MAX_SPINDLES = 6.0         # 六轴钻机最大主轴数 S_spindle_max
NORM_SIX_SPINDLE_COUNT = 3.0    # 6轴钻机台数基准 (D1~D3)
NORM_TWO_SPINDLE_COUNT = 5.0    # 2轴钻机台数基准 (D4~D8)
NORM_TEMPLATE_LEN_MAX = 10.0    # 模板最大长度 (最大拆分块数)
NORM_TOTAL_MACHINES_MAX = 10.0  # 全车间总机器台数 (8钻+2背钻)


class PCBGraphBuilder:
    """构建六类节点 HeteroData 异构图数据结构。"""

    NODE_TYPES: Tuple[NodeType, ...] = (
        "pool", "order", "template", "subbatch", "machine", "slot",
    )
    POOL_FEATURE_DIM = 6
    # 10 physical + 4 causal business/context + 3 frontier indicators.
    ORDER_FEATURE_DIM = 17
    TEMPLATE_FEATURE_DIM = TEMPLATE_FEATURE_DIM
    SUBBATCH_FEATURE_DIM = 5
    MACHINE_FEATURE_DIM = 10
    SLOT_FEATURE_DIM = 5

    # Each forward relation has a named, explicit reverse relation. The names
    # are part of the frozen runtime schema; callers must not infer relation
    # types from the decision stage.
    RELATION_PAIRS: Mapping[str, Tuple[EdgeType, EdgeType]] = {
        "pool_order": (("pool", "contains", "order"), ("order", "in_pool", "pool")),
        "pool_machine": (("pool", "contains", "machine"), ("machine", "in_pool", "pool")),
        "order_template": (("order", "has_candidate", "template"), ("template", "candidate_for", "order")),
        "order_subbatch": (("order", "contains", "subbatch"), ("subbatch", "belongs_to", "order")),
        "template_subbatch": (("template", "instantiates", "subbatch"), ("subbatch", "instantiated_from", "template")),
        "order_machine": (("order", "eligible_for", "machine"), ("machine", "eligible_order", "order")),
        "subbatch_machine": (("subbatch", "fits_machine", "machine"), ("machine", "fits_subbatch", "subbatch")),
        "machine_slot": (("machine", "has_slot", "slot"), ("slot", "hosted_by", "machine")),
        "subbatch_precedence": (("subbatch", "precedes", "subbatch"), ("subbatch", "follows", "subbatch")),
        "subbatch_slot": (("subbatch", "can_insert", "slot"), ("slot", "can_receive", "subbatch")),
    }
    EDGE_TYPES: Tuple[EdgeType, ...] = tuple(
        edge for pair in RELATION_PAIRS.values() for edge in pair
    )

    def __init__(
        self,
        orders: List[Order],
        pool_type: str,
        machines_dict: Dict[str, Machine] = ALL_MACHINES,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.orders = list(orders)
        self.pool_type = pool_type
        incompatible = [order.order_id for order in self.orders if order.pool_type != pool_type]
        if incompatible:
            raise ValueError(
                f"Graph for pool {pool_type} received cross-pool orders: {incompatible}."
            )
        self.orders_dict = {order.order_id: order for order in self.orders}
        self.machines_dict = {
            machine_id: machine
            for machine_id, machine in machines_dict.items()
            if machine.pool_type == pool_type
        }
        if not self.machines_dict:
            raise ValueError(f"No machines registered for pool {pool_type}.")
        self.device = device
        self.order_ids = [order.order_id for order in self.orders]
        self.order_to_idx = {order_id: index for index, order_id in enumerate(self.order_ids)}
        self.machine_ids = sorted(self.machines_dict)
        self.machine_to_idx = {machine_id: index for index, machine_id in enumerate(self.machine_ids)}
        self.max_W = max((order.W for order in self.orders), default=1)
        self.max_Q = max((order.Q for order in self.orders), default=1)
        self.max_N = max((order.N for order in self.orders), default=1)
        total_work = sum((order.N / order.efficiency) * order.W for order in self.orders)
        total_spindles = sum(machine.spindles for machine in self.machines_dict.values())
        self.LB_g = max(1.0, total_work / max(1, total_spindles))
        max_release = max((float(order.release_time or 0.0) for order in self.orders), default=0.0)
        max_due = max((float(order.due_date or 0.0) for order in self.orders), default=0.0)
        self.time_scale = max(1.0, self.LB_g, max_release, max_due)
        self.slot_map: Dict[Tuple[str, int], int] = {}

        # -------------------------------------------------------------
        # 高性能 GPU 向量化缓存：静态边索引与静态工单物理特征预计算
        # -------------------------------------------------------------
        self._cached_static_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # 1. pool_order (100% 静态)
        fwd_po = self._edge_tensor([(0, index) for index in range(len(self.orders))])
        self._cached_static_edges["pool_order"] = (fwd_po, fwd_po[[1, 0]] if fwd_po.numel() > 0 else self._empty_edges())
        # 2. pool_machine (100% 静态)
        fwd_pm = self._edge_tensor([(0, index) for index in range(len(self.machine_ids))])
        self._cached_static_edges["pool_machine"] = (fwd_pm, fwd_pm[[1, 0]] if fwd_pm.numel() > 0 else self._empty_edges())
        # 3. order_machine (100% 静态机台加工资格关系)
        om_list = [
            (self.order_to_idx[order.order_id], self.machine_to_idx[machine_id])
            for order in self.orders
            for machine_id in order.eligible_machines
            if machine_id in self.machine_to_idx
        ]
        fwd_om = self._edge_tensor(om_list)
        self._cached_static_edges["order_machine"] = (fwd_om, fwd_om[[1, 0]] if fwd_om.numel() > 0 else self._empty_edges())

        # 4. 静态工单物理特征底表。业务动态特征在 build_graph 中按
        #    当前 event_time 因果计算，不能预编码未来事件状态。
        base_order_features: List[List[float]] = []
        for order in self.orders:
            eligible = [machine_id for machine_id in order.eligible_machines if machine_id in self.machines_dict]
            base_order_features.append([
                order.W / max(1.0, float(self.max_W)),
                order.Q / max(1.0, float(self.max_Q)),
                order.S / NORM_STACK_COUNT_MAX,
                order.N / max(1.0, float(self.max_N)),
                (order.N / max(1.0, order.efficiency)) / NORM_WORKLOAD_HOURS,
                math.ceil(order.N / max(1, order.H)) / NORM_TOOL_CHANGES_MAX,
                ((order.N / order.efficiency) * (order.W / NORM_MAX_SPINDLES)) / self.LB_g,
                len(eligible) / max(1, len(self.machines_dict)),
                sum(self.machines_dict[m].spindles == 6 for m in eligible) / NORM_SIX_SPINDLE_COUNT,
                sum(self.machines_dict[m].spindles == 2 for m in eligible) / NORM_TWO_SPINDLE_COUNT,
            ])
        self._static_order_features_base = torch.tensor(
            base_order_features, dtype=torch.float32, device=self.device
        )
        self._order_indices = torch.arange(len(self.orders), device=self.device)

    def _empty_edges(self) -> torch.Tensor:
        return torch.empty((2, 0), dtype=torch.long, device=self.device)

    def _edge_tensor(self, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
        if not edges:
            return self._empty_edges()
        return torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()

    def _set_relation_pair(
        self,
        data: HeteroData,
        name: str,
        forward_edges: Optional[Sequence[Tuple[int, int]]],
    ) -> None:
        forward, reverse = self.RELATION_PAIRS[name]
        if forward_edges is None and name in self._cached_static_edges:
            data[forward].edge_index, data[reverse].edge_index = self._cached_static_edges[name]
            return
        edge_tensor = self._edge_tensor(forward_edges or ())
        data[forward].edge_index = edge_tensor
        data[reverse].edge_index = edge_tensor[[1, 0]] if edge_tensor.numel() > 0 else self._empty_edges()

    def _template_features(
        self,
        templates: Sequence[Template],
        current_order: Optional[Order],
        template_mask: Mapping[Template, int],
    ) -> List[List[float]]:
        if current_order is None:
            return []
        order_w = max(1, current_order.W)
        features: List[List[float]] = []
        for template in templates:
            if not template or any(int(part) <= 0 for part in template):
                raise ValueError(f"Invalid partition template: {template!r}")
            features.append(encode_template(template))
            if sum(template) != order_w:
                raise ValueError(
                    f"Template {template} does not conserve current order {current_order.order_id} demand {order_w}."
                )
        return features

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
        """Create a fixed-schema graph for one resource-pool decision state.

        ``candidate_templates`` must be the exact ordered action list supplied to
        SplitPolicy. Its index is persisted as ``template_id`` on the matching
        node. ``selected_template`` only activates template--subbatch edges; it
        never pre-materialises subbatches for unselected candidates.
        """
        if solution.pool_type != self.pool_type:
            raise ValueError(
                f"Solution pool {solution.pool_type} cannot be encoded by {self.pool_type} graph builder."
            )
        if not 0 <= current_frontier_idx <= len(self.orders):
            raise IndexError(f"Invalid frontier index {current_frontier_idx} for {len(self.orders)} orders.")

        data = HeteroData()
        templates = list(candidate_templates or ())
        if len(set(templates)) != len(templates):
            raise ValueError("Candidate template list contains duplicate action handles.")
        mask = dict(template_mask or {})
        if selected_template is not None and selected_template not in templates:
            raise ValueError("Selected template is absent from candidate template action handles.")
        selected_template_id = templates.index(selected_template) if selected_template is not None else -1
        current_order = self.orders[current_frontier_idx] if current_frontier_idx < len(self.orders) else None

        # Graph-level state is deliberately not a node store.
        placement_stage = 1.0 if active_sub_batches else 0.0
        data.event_context = torch.tensor(
            [[1.0 if self.pool_type == POOL_D else 0.0,
              current_frontier_idx / max(1, len(self.orders)),
              placement_stage,
              max(0.0, float(event_time)) / self.time_scale,
              float(scope_ratio)]],
            dtype=torch.float32,
            device=self.device,
        )

        # pool: exactly one node and the only global readout anchor.
        total_spindles = sum(machine.spindles for machine in self.machines_dict.values())
        queue_lengths = [len(solution.machine_queues[machine_id].batches) for machine_id in self.machine_ids]
        queue_ends = [
            solution.machine_queues[machine_id].prefix_cache[-1]
            for machine_id in self.machine_ids
            if solution.machine_queues[machine_id].prefix_cache
        ]
        data["pool"].x = torch.tensor([[
            1.0 if self.pool_type == POOL_D else 0.0,
            len(self.machine_ids) / NORM_TOTAL_MACHINES_MAX,
            total_spindles / 28.0,
            sum(queue_lengths) / max(1.0, 50.0 * len(self.machine_ids)),
            (max(queue_ends) if queue_ends else 0.0) / self.LB_g,
            current_frontier_idx / max(1, len(self.orders)),
        ]], dtype=torch.float32, device=self.device)

        # order nodes: current/released information only. The four business
        # features consume the completed data contract without changing the
        # pure Cmax objective: urgency, release, remaining due slack, forecast.
        priority_feature = torch.tensor(
            [[{"高": 1.0, "中": 0.5, "普通": 0.0}.get(str(order.priority), 0.0)] for order in self.orders],
            dtype=torch.float32,
            device=self.device,
        )
        release_feature = torch.tensor(
            [[max(0.0, float(order.release_time or 0.0)) / self.time_scale] for order in self.orders],
            dtype=torch.float32,
            device=self.device,
        ).clamp_(0.0, 2.0)
        slack_feature = torch.tensor(
            [[max(0.0, float(order.due_date or event_time) - float(event_time)) / self.time_scale] for order in self.orders],
            dtype=torch.float32,
            device=self.device,
        ).clamp_(0.0, 2.0)
        forecast_feature = torch.tensor(
            [[1.0 if bool(getattr(order, "is_forecasted", True)) else 0.0] for order in self.orders],
            dtype=torch.float32,
            device=self.device,
        )
        is_past = (self._order_indices < current_frontier_idx).float().unsqueeze(-1)
        is_curr = (self._order_indices == current_frontier_idx).float().unsqueeze(-1)
        is_future = (self._order_indices > current_frontier_idx).float().unsqueeze(-1)
        data["order"].x = torch.cat(
            [
                self._static_order_features_base,
                priority_feature,
                release_feature,
                slack_feature,
                forecast_feature,
                is_past,
                is_curr,
                is_future,
            ],
            dim=-1,
        )
        if data["order"].x.size(-1) != self.ORDER_FEATURE_DIM:
            raise RuntimeError(
                f"Order feature schema mismatch: {data['order'].x.size(-1)} != {self.ORDER_FEATURE_DIM}."
            )

        # template: always present, but only the exposed order receives action nodes.
        template_features = self._template_features(templates, current_order, mask)
        data["template"].x = torch.tensor(
            template_features or [], dtype=torch.float32, device=self.device
        ).reshape((len(template_features), self.TEMPLATE_FEATURE_DIM))
        data["template"].template_id = torch.arange(
            len(templates), dtype=torch.long, device=self.device
        )
        data["template"].action_mask = torch.tensor(
            [int(mask.get(template, 0)) for template in templates],
            dtype=torch.long,
            device=self.device,
        )
        # Register the empty store before machine/slot creation so PyG metadata
        # remains in the authority-defined node-type order in both stages.
        data["subbatch"].x = torch.empty(
            (0, self.SUBBATCH_FEATURE_DIM), dtype=torch.float32, device=self.device
        )

        # machine nodes
        max_machine_time = max(queue_ends or [1.0])
        machine_features: List[List[float]] = []
        for machine_id in self.machine_ids:
            machine = self.machines_dict[machine_id]
            queue = solution.machine_queues[machine_id]
            machine_time = queue.prefix_cache[-1] if queue.prefix_cache else 0.0
            queue_len = len(queue.batches)
            frontier_same = float(
                bool(queue_len and current_order is not None and queue.batches[-1].order_id == current_order.order_id)
            )
            m_params = TIME_PARAMS.get(machine.machine_type, {})
            t_aux = m_params.get("t_aux", 6.6 if machine.spindles == 6 else 2.2)
            s_setup = m_params.get("s_setup", 13.5 if machine.spindles == 6 else 5.5)
            machine_features.append([
                machine.spindles / NORM_MAX_SPINDLES,
                t_aux / 10.0,
                s_setup / 20.0,
                machine_time / max(1.0, max_machine_time),
                queue_len / 50.0,
                float(queue_len > 0),
                float(queue_len == 0),
                frontier_same,
                sum(batch.stack_count for batch in queue.batches) / max(1.0, machine.spindles * max(1, queue_len)),
                (machine_time - self.LB_g) / self.LB_g,
            ])
        data["machine"].x = torch.tensor(machine_features, dtype=torch.float32, device=self.device)

        # slot nodes, one per admissible insertion position.
        subbatches = list(active_sub_batches or ())
        target_order_id = None
        if subbatches and 0 <= current_sub_batch_idx < len(subbatches):
            target_order_id = subbatches[current_sub_batch_idx].order_id
        elif current_order is not None:
            target_order_id = current_order.order_id

        self.slot_map = {}
        slot_features: List[List[float]] = []
        machine_slot_edges: List[Tuple[int, int]] = []
        for machine_idx, machine_id in enumerate(self.machine_ids):
            machine = self.machines_dict.get(machine_id)
            queue = solution.machine_queues[machine_id]
            for position in range(len(queue.batches) + 1):
                slot_index = len(slot_features)
                self.slot_map[(machine_id, position)] = slot_index

                pred_oid = queue.batches[position - 1].order_id if position > 0 else None
                succ_oid = queue.batches[position].order_id if position < len(queue.batches) else None
                pred_same = float(bool(pred_oid is not None and target_order_id is not None and pred_oid == target_order_id))

                # 精确计算在该插槽插入当前批次导致的换型时间增量 Δs = s(pred, curr) + s(curr, succ) - s(pred, succ)
                if machine is not None and target_order_id is not None:
                    s_before = (
                        setup_time(machine, pred_oid, succ_oid, is_first_batch=(position == 0 and len(queue.batches) > 0))
                        if succ_oid is not None else 0.0
                    )
                    s_pb = setup_time(machine, pred_oid, target_order_id, is_first_batch=(position == 0))
                    s_bs = setup_time(machine, target_order_id, succ_oid, is_first_batch=False) if succ_oid is not None else 0.0
                    delta_s = max(0.0, (s_pb + s_bs) - s_before)
                else:
                    delta_s = 0.0

                ready_time = (
                    queue.prefix_cache[position - 1]
                    if (position > 0 and queue.prefix_cache and len(queue.prefix_cache) >= position)
                    else 0.0
                )
                slot_features.append([
                    position / max(1.0, float(len(queue.batches) + 1)),
                    float(position == len(queue.batches)),
                    pred_same,
                    delta_s / 20.0,
                    ready_time / max(1.0, max_machine_time),
                ])
                machine_slot_edges.append((machine_idx, slot_index))
        data["slot"].x = torch.tensor(slot_features, dtype=torch.float32, device=self.device)

        data["subbatch"].x = torch.tensor(
            [[
                batch.stack_count / NORM_MAX_SPINDLES,
                index / max(1.0, float(len(subbatches))),
                batch.stack_count / NORM_MAX_SPINDLES,
                (self.orders_dict[batch.order_id].N / max(1.0, self.orders_dict[batch.order_id].efficiency)) / NORM_WORKLOAD_HOURS,
                float(index == current_sub_batch_idx),
            ] for index, batch in enumerate(subbatches)],
            dtype=torch.float32,
            device=self.device,
        ).reshape((len(subbatches), self.SUBBATCH_FEATURE_DIM))
        data["subbatch"].template_id = torch.tensor(
            [selected_template_id if batch.order_id == getattr(current_order, "order_id", None) else -1 for batch in subbatches],
            dtype=torch.long,
            device=self.device,
        )
        stable_event_id = str(event_id or f"static:{self.pool_type}:{current_frontier_idx}")
        data["subbatch"].stable_uid = [
            (stable_event_id, batch.order_id, selected_template_id, rank)
            for rank, batch in enumerate(subbatches)
        ]

        # Fixed relations (100% 静态边张量直接复用，零内存分配)
        self._set_relation_pair(data, "pool_order", None)
        self._set_relation_pair(data, "pool_machine", None)
        self._set_relation_pair(
            data,
            "order_template",
            [] if current_order is None else [(current_frontier_idx, index) for index in range(len(templates))],
        )
        self._set_relation_pair(data, "order_machine", None)
        self._set_relation_pair(data, "machine_slot", machine_slot_edges)

        order_subbatch_edges: List[Tuple[int, int]] = []
        template_subbatch_edges: List[Tuple[int, int]] = []
        subbatch_machine_edges: List[Tuple[int, int]] = []
        subbatch_slot_edges: List[Tuple[int, int]] = []
        assigned_positions: Dict[str, List[Tuple[int, int]]] = {}
        for subbatch_idx, batch in enumerate(subbatches):
            order = self.orders_dict.get(batch.order_id)
            if order is None:
                raise ValueError(f"Subbatch {batch.u_id} references unknown order {batch.order_id!r}.")
            order_subbatch_edges.append((self.order_to_idx[order.order_id], subbatch_idx))
            if selected_template_id >= 0 and batch.order_id == getattr(current_order, "order_id", None):
                template_subbatch_edges.append((selected_template_id, subbatch_idx))

            # A committed subbatch remains a node, but only current/unclaimed
            # subbatches expose capacity and insertion action edges.
            assigned_queue_and_pos = None
            if batch.machine_id and batch.machine_id in solution.machine_queues:
                for idx, queued in enumerate(solution.machine_queues[batch.machine_id].batches):
                    if queued.u_id == batch.u_id:
                        assigned_queue_and_pos = (batch.machine_id, idx)
                        break
            if assigned_queue_and_pos is None:
                for mid, q in solution.machine_queues.items():
                    for idx, queued in enumerate(q.batches):
                        if queued.u_id == batch.u_id:
                            assigned_queue_and_pos = (mid, idx)
                            batch.machine_id = mid
                            break
                    if assigned_queue_and_pos is not None:
                        break

            if assigned_queue_and_pos is not None:
                assigned_mid, assigned_idx = assigned_queue_and_pos
                assigned_positions.setdefault(assigned_mid, []).append((assigned_idx, subbatch_idx))
                continue
            for machine_id in order.eligible_machines:
                machine = self.machines_dict.get(machine_id)
                if machine is None or machine.spindles < batch.stack_count:
                    continue
                subbatch_machine_edges.append((subbatch_idx, self.machine_to_idx[machine_id]))
                for position in range(len(solution.machine_queues[machine_id].batches) + 1):
                    subbatch_slot_edges.append((subbatch_idx, self.slot_map[(machine_id, position)]))

        precedence_edges: List[Tuple[int, int]] = []
        for positions in assigned_positions.values():
            ordered = [subbatch_idx for _, subbatch_idx in sorted(positions)]
            precedence_edges.extend(zip(ordered, ordered[1:]))
        self._set_relation_pair(data, "order_subbatch", order_subbatch_edges)
        self._set_relation_pair(data, "template_subbatch", template_subbatch_edges)
        self._set_relation_pair(data, "subbatch_machine", subbatch_machine_edges)
        self._set_relation_pair(data, "subbatch_precedence", precedence_edges)
        self._set_relation_pair(data, "subbatch_slot", subbatch_slot_edges)

        # This assertion catches accidental schema drift at its source.
        if tuple(data.node_types) != self.NODE_TYPES or set(data.edge_types) != set(self.EDGE_TYPES):
            raise RuntimeError("Graph schema construction drifted from the authority-locked metadata.")
        return data
