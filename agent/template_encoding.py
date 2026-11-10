"""Single source of truth for exact partition-template features."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


Template = Tuple[int, ...]
TEMPLATE_FEATURE_DIM = 10
TEMPLATE_ENCODING_VERSION = "partition-histogram10-v20261105"


def encode_template(template: Sequence[int]) -> List[float]:
    """Encode a partition without collisions for parts in ``1..6``.

    The first six coordinates are the exact count histogram ``c_1..c_6``.
    The remaining four coordinates retain useful aggregate statistics:
    ``len(p)``, ``max(p)/6``, ``min(p)/6`` and ``mean(p)/6``.
    Feasibility is deliberately not embedded here; it remains a hard mask.
    """
    parts = tuple(int(part) for part in template)
    if not parts:
        raise ValueError("Partition template cannot be empty.")
    if any(part < 1 or part > 6 for part in parts):
        raise ValueError(f"Template parts must lie in 1..6, got {parts!r}.")
    histogram = [float(parts.count(value)) for value in range(1, 7)]
    return histogram + [
        float(len(parts)),
        float(max(parts)) / 6.0,
        float(min(parts)) / 6.0,
        (float(sum(parts)) / float(len(parts))) / 6.0,
    ]


def encode_templates(
    templates: Sequence[Template],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    features = [encode_template(template) for template in templates]
    return torch.tensor(features, dtype=dtype, device=device).reshape(
        len(features), TEMPLATE_FEATURE_DIM
    )
