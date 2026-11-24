"""Train the sole C2 ablation: AD-HG-SAPPO single-Actor PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[2]
ENVIRONMENT_DIR = PROJECT_ROOT / "environment"
AGENT_DIR = PROJECT_ROOT / "agent"
for path in (THIS_DIR, EXPERIMENT_ROOT, PROJECT_ROOT, ENVIRONMENT_DIR, AGENT_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from data_models import POOL_D
from dynamic_event_manager import (
    INITIAL_SOLUTION_BUILDER_VERSION,
    INITIAL_SOLUTION_DISPATCH_RULE,
)
from dynamic_multi_scenario_trainer import DynamicMultiScenarioTrainer
from dynamic_validator import load_dynamic_orders_and_events
from hppo_trainer import (
    FORMAL_DYNAMIC_C_ENT,
    FORMAL_DYNAMIC_LR,
    FORMAL_DYNAMIC_PROTOCOL_VERSION,
    FORMAL_DYNAMIC_REWARD_SCALE,
)
from single_actor_ppo_trainer import (
    SINGLE_ACTOR_ABLATION_SCOPE,
    SINGLE_ACTOR_CODE_ID,
    SINGLE_ACTOR_CREDIT_ASSIGNMENT,
    SINGLE_ACTOR_ENTROPY_REGULARIZATION,
    SINGLE_ACTOR_EVENT_REWARD_ASSIGNMENT,
    SINGLE_ACTOR_LR_SEMANTICS,
    SINGLE_ACTOR_METHOD_ID,
    SINGLE_ACTOR_OBJECTIVE_WEIGHTING,
    PLACEMENT_ENTROPY_TARGET,
    SPLIT_ENTROPY_TARGET,
    SingleActorPPOTrainer,
)
from configs.experiment_protocol import DEFAULT_FORMAL_SEED, FORMAL_SEEDS, validate_formal_seed

C2_RESULTS_DIR = PROJECT_ROOT / "result" / "消融实验"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_single_actor_run_paths(stage: int, seed: int) -> Dict[str, Path]:
    """Resolve the non-overwriting result layout shared by all C2 entry points."""
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2 or 3; received {stage!r}.")
    seed = validate_formal_seed(seed)
    run_dir = C2_RESULTS_DIR / f"stage{stage}" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "best_checkpoint": run_dir / f"stage{stage}_best.pt",
        "latest_checkpoint": run_dir / f"stage{stage}_latest.pt",
        "log": run_dir / f"stage{stage}_train_log.json",
        "contract": run_dir / "experiment_contract.json",
        "tensorboard": run_dir / "tensorboard",
    }


def resolve_single_actor_start_mode(
    paths: Dict[str, Path],
    *,
    resume: bool = False,
    fresh_start: bool = False,
) -> tuple[bool, str]:
    """Resolve a safe C2 start mode without silently mixing result generations."""
    if resume and fresh_start:
        raise ValueError("--resume 与 --fresh-start 不能同时使用。")

    has_latest = paths["latest_checkpoint"].is_file()
    has_log = paths["log"].is_file()
    has_any_result = any(
        (
            paths["best_checkpoint"].exists(),
            has_latest,
            has_log,
            paths["contract"].exists(),
            paths["tensorboard"].exists(),
        )
    )

    if fresh_start:
        return False, "fresh_start"
    if resume:
        if not (has_latest and has_log):
            raise FileNotFoundError(
                "C2 续训需要成对的 latest checkpoint 与训练日志："
                f"latest={has_latest}, log={has_log}。"
            )
        return True, "resume"
    if has_latest and has_log:
        return True, "auto_resume"
    if has_any_result:
        raise RuntimeError(
            "检测到孤立或残缺的 C2 训练产物。为防止混合新旧结果，"
            "请恢复成对的日志与 latest checkpoint，或显式使用 --fresh-start。"
        )
    return False, "new"


class SingleActorMultiScenarioTrainer(DynamicMultiScenarioTrainer):
    """Reuse the main experiment's data, curriculum, validation and logging."""

    def __init__(self, *args, checkpoint_in: Optional[str] = None, **kwargs):
        super().__init__(*args, checkpoint_in=None, **kwargs)
        sample_orders, _ = load_dynamic_orders_and_events(self.train_files[0], self.sheet_name)
        self.trainer = SingleActorPPOTrainer(
            orders=sample_orders,
            pool_type=self.pool_type,
            d_h=int(kwargs.get("d_h", 32)),
            lr=float(kwargs.get("lr", FORMAL_DYNAMIC_LR)),
            ppo_epochs=int(kwargs.get("ppo_epochs", 4)),
            c_ent=float(kwargs.get("c_ent", FORMAL_DYNAMIC_C_ENT)),
            device=self.device,
            reward_mode=str(kwargs.get("reward_mode", "scaled_relative")),
            target_kl=float(kwargs.get("target_kl", 0.05)),
            max_kl=float(kwargs.get("max_kl", 0.15)),
        )
        self.checkpoint_payload = None
        if checkpoint_in:
            self.checkpoint_payload = self.trainer.load_checkpoint(
                checkpoint_in, load_optimizer=True
            )


def train_single_actor_stage(
    stage: int,
    seed: int = 1,
    episodes: Optional[int] = None,
    lr: float = FORMAL_DYNAMIC_LR,
    eval_interval: int = 10,
    rollout_scenarios: int = 6,
    ppo_epochs: int = 4,
    device: str = "cuda",
    resume: bool = False,
    fresh_start: bool = False,
) -> Dict[str, Any]:
    """Train one independent Stage under the main experiment's protocol."""
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2 or 3; received {stage!r}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    paths = resolve_single_actor_run_paths(stage, seed)
    run_dir = paths["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = paths["best_checkpoint"]
    latest_checkpoint = paths["latest_checkpoint"]
    log_path = paths["log"]
    metadata_path = paths["contract"]

    effective_resume, start_mode = resolve_single_actor_start_mode(
        paths,
        resume=resume,
        fresh_start=fresh_start,
    )
    if start_mode == "fresh_start":
        print(f"[C2 启动模式] Stage {stage}, seed={seed} 显式从随机初始化开始。")
    elif start_mode == "auto_resume":
        print(f"[C2 启动模式] Stage {stage}, seed={seed} 检测到成对断点，自动续训。")
    elif start_mode == "resume":
        print(f"[C2 启动模式] Stage {stage}, seed={seed} 按 --resume 续训。")
    else:
        print(f"[C2 启动模式] Stage {stage}, seed={seed} 未发现历史结果，全新训练。")

    selected_device = torch.device(
        device if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    checkpoint_in = (
        str(latest_checkpoint)
        if effective_resume and latest_checkpoint.is_file()
        else None
    )
    driver = SingleActorMultiScenarioTrainer(
        pool_type=POOL_D,
        d_h=32,
        lr=lr,
        ppo_epochs=ppo_epochs,
        reward_scale=FORMAL_DYNAMIC_REWARD_SCALE,
        checkpoint_in=checkpoint_in,
        device=selected_device,
        stage_only=stage,
        rollout_scenarios=rollout_scenarios,
        reward_mode="scaled_relative",
        target_kl=0.05,
        max_kl=0.15,
        c_ent=FORMAL_DYNAMIC_C_ENT,
    )
    contract = {
        "method_id": SINGLE_ACTOR_METHOD_ID,
        "code_id": SINGLE_ACTOR_CODE_ID,
        "only_ablation": True,
        "ablation_scope": SINGLE_ACTOR_ABLATION_SCOPE,
        "credit_assignment": SINGLE_ACTOR_CREDIT_ASSIGNMENT,
        "event_reward_assignment": SINGLE_ACTOR_EVENT_REWARD_ASSIGNMENT,
        "objective_weighting": SINGLE_ACTOR_OBJECTIVE_WEIGHTING,
        "actor_lr_semantics": SINGLE_ACTOR_LR_SEMANTICS,
        "actor_learning_rate": lr,
        "entropy_regularization": SINGLE_ACTOR_ENTROPY_REGULARIZATION,
        "split_entropy_target": SPLIT_ENTROPY_TARGET,
        "placement_entropy_target": PLACEMENT_ENTROPY_TARGET,
        "protocol_version": FORMAL_DYNAMIC_PROTOCOL_VERSION,
        "initial_solution_builder_version": INITIAL_SOLUTION_BUILDER_VERSION,
        "initial_solution_dispatch_rule": INITIAL_SOLUTION_DISPATCH_RULE,
        "stage": stage,
        "seed": seed,
        "train_scenarios": [Path(path).name for path in driver.train_files],
        "validation_scenarios": [Path(path).name for path in driver.val_files],
        "formal_episode_budget": sum(driver.stage_quota_override or []),
        "requested_episode_budget": episodes,
        "rollout_scenarios": rollout_scenarios,
        "ppo_epochs": ppo_epochs,
        "lr": lr,
        "reward_scale": FORMAL_DYNAMIC_REWARD_SCALE,
        "parameter_match_error": driver.trainer.single_actor_policy.parameter_match_error,
        "single_actor_parameter_count": sum(
            p.numel() for p in driver.trainer.single_actor_policy.parameters()
        ),
        "reference_dual_actor_parameter_count": (
            driver.trainer.single_actor_policy.reference_actor_parameters
        ),
        "actor_optimizer_count": 1,
        "critic_optimizer_count": 1,
    }
    manifest_path = PROJECT_ROOT / "data" / "dynamic_scenarios" / "dataset_manifest.json"
    active_package_path = PROJECT_ROOT / "data" / "active_experiment_package.json"
    contract["dataset_manifest_path"] = str(manifest_path.relative_to(PROJECT_ROOT))
    contract["dataset_manifest_sha256"] = _sha256(manifest_path)
    contract["active_package_path"] = str(active_package_path.relative_to(PROJECT_ROOT))
    contract["active_package_sha256"] = _sha256(active_package_path)
    driver.trainer.checkpoint_contract = dict(contract)
    metadata_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    result = driver.train(
        total_episodes=episodes,
        episodes_per_scenario=300,
        eval_interval=eval_interval,
        checkpoint_out=str(best_checkpoint),
        latest_checkpoint_out=str(latest_checkpoint),
        log_out=str(log_path),
        tensorboard_dir=str(paths["tensorboard"]),
        curriculum_mode="round_robin",
        resume=effective_resume,
        seed=seed,
    )
    return {
        "result": result,
        "best_checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "log": str(log_path),
        "contract": str(metadata_path),
    }


def load_single_actor_trainer(
    checkpoint_path: str | Path,
    orders,
    device: torch.device = torch.device("cpu"),
) -> SingleActorPPOTrainer:
    trainer = SingleActorPPOTrainer(orders=list(orders), pool_type=POOL_D, device=device)
    payload = trainer.load_checkpoint(
        str(Path(checkpoint_path).resolve()), load_optimizer=False
    )
    trainer.checkpoint_payload = payload
    return trainer


def main(argv: Optional[list[str]] = None, fixed_stage: Optional[int] = None) -> None:
    """Run one fixed Stage entry point or the general ``--stage`` entry point."""
    if fixed_stage is not None and fixed_stage not in (1, 2, 3):
        raise ValueError(f"fixed_stage must be 1, 2 or 3; received {fixed_stage!r}.")

    parser = argparse.ArgumentParser(
        description="训练唯一 C2 消融：单 Actor、单轨迹、单 PPO 目标（AD-HG-SAPPO）"
    )
    if fixed_stage is None:
        parser.add_argument("--stage", default="1", choices=("1", "2", "3", "all"))
    parser.add_argument(
        "--seed",
        type=int,
        choices=FORMAL_SEEDS,
        default=DEFAULT_FORMAL_SEED,
        help=f"正式随机种子（{list(FORMAL_SEEDS)}，默认 {DEFAULT_FORMAL_SEED}）",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="仅用于调试的预算覆盖；省略时使用每 Stage 1800 episode 正式预算。",
    )
    parser.add_argument("--lr", type=float, default=FORMAL_DYNAMIC_LR)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--rollout-scenarios", type=int, default=6)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    start_mode = parser.add_mutually_exclusive_group()
    start_mode.add_argument("--resume", action="store_true")
    start_mode.add_argument(
        "--fresh-start",
        action="store_true",
        help="强制从随机初始化开始，不加载已有 C2 断点。",
    )
    args = parser.parse_args(argv)

    if fixed_stage is not None:
        stages = (fixed_stage,)
    else:
        stages = (1, 2, 3) if args.stage == "all" else (int(args.stage),)
    for selected_stage in stages:
        print(
            f"[C2/{SINGLE_ACTOR_CODE_ID}] Stage {selected_stage}, seed={args.seed}, "
            f"device={args.device}"
        )
        train_single_actor_stage(
            stage=selected_stage,
            seed=args.seed,
            episodes=args.episodes,
            lr=args.lr,
            eval_interval=args.eval_interval,
            rollout_scenarios=args.rollout_scenarios,
            ppo_epochs=args.ppo_epochs,
            device=args.device,
            resume=args.resume,
            fresh_start=args.fresh_start,
        )


if __name__ == "__main__":
    main()

