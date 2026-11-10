"""Single-Actor policy used by the only architecture ablation.

The module deliberately preserves the two physical action domains of the main
method (partition templates and machine-slot pairs) while using one parameter
set and one scoring head for both domains.  It is therefore a single policy,
not two policies hidden behind a wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
_ENVIRONMENT_DIR = _PROJECT_ROOT / "environment"

for _p in (str(_THIS_DIR), str(_PROJECT_ROOT), str(_AGENT_DIR), str(_ENVIRONMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from manager_policy import ManagerPolicy
from worker_policy import WorkerPolicy
from template_encoding import TEMPLATE_FEATURE_DIM, encode_templates


SPLIT_ACTION = "split"
PLACEMENT_ACTION = "placement"


def reference_dual_actor_parameter_count(d_h: int) -> int:
    """Return the trainable parameter count of the main method's two Actors."""
    manager = ManagerPolicy(
        d_g=2 * d_h,
        d_order=d_h,
        d_template=TEMPLATE_FEATURE_DIM,
        d_template_node=d_h,
        d_hidden=d_h,
    )
    worker = WorkerPolicy(d_batch=d_h, d_slot=d_h, d_hidden=d_h)
    return sum(p.numel() for p in manager.parameters()) + sum(
        p.numel() for p in worker.parameters()
    )


def matched_hidden_dim(d_h: int) -> int:
    """Choose the hidden width that best matches the two-Actor parameter budget."""
    target = reference_dual_actor_parameter_count(d_h)
    best_width = 8
    best_gap = float("inf")
    for width in range(8, max(64, 4 * d_h) + 1):
        # state projection + candidate projection + common scoring MLP
        count = (4 * d_h + 2) * width + width
        count += (TEMPLATE_FEATURE_DIM + d_h + 2) * width + width
        count += width * width + width
        count += width + 1
        gap = abs(count - target)
        if gap < best_gap:
            best_gap = gap
            best_width = width
    return best_width


class SingleActorPolicy(nn.Module):
    """One policy network over typed partition and placement candidates.

    A single context projection, candidate projection and scoring head are
    shared by both decision types.  Missing state components are filled with
    zeros because the main PlacementPolicy does not consume global/order
    embeddings and the main SplitPolicy does not consume a current subbatch.
    """

    def __init__(self, d_h: int = 32, d_hidden: Optional[int] = None):
        super().__init__()
        self.d_h = int(d_h)
        self.d_g = 2 * self.d_h
        self.d_hidden = int(d_hidden or matched_hidden_dim(self.d_h))
        self.reference_actor_parameters = reference_dual_actor_parameter_count(self.d_h)

        # state = [g, h_order, h_batch, one_hot(decision_type)]
        self.state_projection = nn.Linear(4 * self.d_h + 2, self.d_hidden)
        # candidate = [template_features, h_slot/h_template, one_hot(decision_type)]
        self.candidate_projection = nn.Linear(TEMPLATE_FEATURE_DIM + self.d_h + 2, self.d_hidden)
        self.scorer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, 1),
        )

        actual = sum(p.numel() for p in self.parameters())
        self.parameter_match_error = abs(actual - self.reference_actor_parameters) / max(
            1, self.reference_actor_parameters
        )
        if self.parameter_match_error > 0.05:
            raise ValueError(
                "SingleActorPolicy parameter budget differs from the main two-Actor "
                f"budget by {self.parameter_match_error:.2%}, exceeding 5%."
            )

    @staticmethod
    def extract_template_features(
        templates: Sequence[Tuple[int, ...]],
        order_W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Use the exact 10-dim template encoding without collision."""
        del order_W
        return encode_templates(templates, device=device)

    def _distribution(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        feasible_mask: torch.Tensor,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        if state.ndim != 2 or state.shape[0] != 1:
            raise ValueError(f"SingleActor state must have shape [1, d], got {tuple(state.shape)}.")
        if candidates.ndim != 2 or candidates.shape[0] == 0:
            raise ValueError("SingleActor candidate set must be a non-empty matrix.")
        feasible_mask = feasible_mask.to(device=state.device, dtype=torch.bool)
        if feasible_mask.numel() != candidates.shape[0]:
            raise ValueError("Candidate count and feasibility-mask length are inconsistent.")
        if not bool(feasible_mask.any().item()):
            raise RuntimeError("The current SingleActor decision has no feasible candidate.")

        state_embedding = self.state_projection(state).expand(candidates.shape[0], -1)
        candidate_embedding = self.candidate_projection(candidates)
        raw_logits = self.scorer(state_embedding + candidate_embedding).squeeze(-1)
        masked_logits = torch.where(
            feasible_mask,
            raw_logits,
            torch.full_like(raw_logits, -1.0e9),
        )
        return torch.distributions.Categorical(logits=masked_logits), masked_logits

    def split_distribution(
        self,
        g: torch.Tensor,
        h_order: torch.Tensor,
        templates: Sequence[Tuple[int, ...]],
        template_mask: Dict[Tuple[int, ...], int],
        order_W: int,
        h_templates: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        device = g.device
        type_code = torch.tensor([[1.0, 0.0]], dtype=g.dtype, device=device)
        zero_batch = torch.zeros((1, self.d_h), dtype=g.dtype, device=device)
        state = torch.cat([g, h_order, zero_batch, type_code], dim=-1)

        template_features = self.extract_template_features(templates, order_W, device)
        if h_templates.shape != (len(templates), self.d_h):
            raise ValueError("Split template embeddings do not match the candidate set.")
        node_features = h_templates.to(device)
        candidate_type = type_code.expand(len(templates), -1)
        candidates = torch.cat([template_features, node_features, candidate_type], dim=-1)
        mask = torch.tensor(
            [bool(template_mask.get(template, 0)) for template in templates],
            dtype=torch.bool,
            device=device,
        )
        return self._distribution(state, candidates, mask)

    def placement_distribution(
        self,
        h_batch: torch.Tensor,
        h_slots: torch.Tensor,
        slot_keys: Sequence[Tuple[str, int]],
        valid_slots: Sequence[Tuple[str, int]],
        **kwargs,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        device = h_batch.device
        type_code = torch.tensor([[0.0, 1.0]], dtype=h_batch.dtype, device=device)
        zero_global = torch.zeros((1, self.d_g), dtype=h_batch.dtype, device=device)
        zero_order = torch.zeros((1, self.d_h), dtype=h_batch.dtype, device=device)
        state = torch.cat([zero_global, zero_order, h_batch, type_code], dim=-1)

        zero_templates = torch.zeros(
            (h_slots.shape[0], TEMPLATE_FEATURE_DIM), dtype=h_slots.dtype, device=device
        )
        candidate_type = type_code.expand(h_slots.shape[0], -1)
        candidates = torch.cat([zero_templates, h_slots, candidate_type], dim=-1)
        valid_set = set(valid_slots)
        mask = torch.tensor(
            [slot in valid_set for slot in slot_keys],
            dtype=torch.bool,
            device=device,
        )
        return self._distribution(state, candidates, mask)

    def forward(self, *args, **kwargs):
        """Support the existing runner interface without creating policy wrappers."""
        if "g" in kwargs:
            return self.split_distribution(
                kwargs["g"],
                kwargs["h_order"],
                kwargs["templates"],
                kwargs["template_mask"],
                kwargs["order_W"],
                h_templates=kwargs["h_templates"],
            )
        if "h_batch" in kwargs:
            return self.placement_distribution(
                kwargs["h_batch"],
                kwargs["h_slots"],
                kwargs["slot_keys"],
                kwargs["valid_slots"],
            )
        if len(args) >= 6 and args[0].shape[-1] == self.d_g:
            return self.split_distribution(
                args[0], args[1], args[3], args[4], args[5], h_templates=args[2]
            )
        if len(args) >= 4:
            return self.placement_distribution(args[0], args[1], args[2], args[3])
        raise TypeError("Cannot infer SingleActor decision type from the supplied arguments.")
