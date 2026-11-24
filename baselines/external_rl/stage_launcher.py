"""IDE-friendly stage training launcher for external RL baselines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.seed_choice import apply_selected_seed
from baselines.external_rl import METHODS
from baselines.external_rl.run import main as run_main


def _extract_cli_method(args: List[str]) -> tuple[Optional[str], List[str]]:
    method: Optional[str] = None
    remaining: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--method" and i + 1 < len(args):
            method = args[i + 1]
            i += 2
        elif arg.startswith("--method="):
            method = arg.split("=", 1)[1]
            i += 1
        else:
            remaining.append(arg)
            i += 1
    return method, remaining


def run_stage_script(
    *,
    fixed_stage: int,
    selected_seed: int,
    fixed_method: Optional[str] = None,
    selected_method: str = "am_a2c",
) -> None:
    """Apply SELECTED_SEED (CLI --seed wins) and run Stage training for target method(s)."""
    if fixed_stage not in (1, 2, 3):
        raise ValueError(f"stage 必须为 1, 2 或 3，收到: {fixed_stage!r}")

    apply_selected_seed(selected_seed)
    cli_args = list(sys.argv[1:])
    cli_method, passthrough = _extract_cli_method(cli_args)

    raw_target = fixed_method if fixed_method is not None else (cli_method or selected_method)
    if raw_target == "all":
        targets: Iterable[str] = tuple(METHODS.keys())
    else:
        if raw_target not in METHODS:
            raise ValueError(
                f"未知算法 {raw_target!r}；可选 {list(METHODS.keys())} 或 'all'。"
            )
        targets = (raw_target,)

    has_mode = any(a == "--mode" or a.startswith("--mode=") for a in passthrough)
    has_stage = any(a == "--stage" or a.startswith("--stage=") for a in passthrough)

    for method in targets:
        argv = ["--method", method]
        if not has_mode:
            argv.extend(["--mode", "train"])
        if not has_stage:
            argv.extend(["--stage", str(fixed_stage)])
        argv.extend(passthrough)
        run_main(argv)
