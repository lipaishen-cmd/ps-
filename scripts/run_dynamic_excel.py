"""
run_dynamic_excel.py
====================
从动态 Excel 直接运行单资源池的双智能体在线重调度。

输入文件只读：脚本不会修改订单表；如需保存结果，使用 ``--json-out``
显式指定一个新的 JSON 文件。动态插单由“动态插单”工作表提供，事件按
实际到达时刻排序；设备故障仍可通过 ``DynamicEventManager`` API 注入。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_ENVIRONMENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "environment"))
_AGENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "agent"))
for _p in (_THIS_DIR, _ENVIRONMENT_DIR, _AGENT_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from data_models import POOL_B, POOL_D  # noqa: E402
from dynamic_validator import load_dynamic_orders_and_events, validate_dynamic_excel  # noqa: E402
from dynamic_event_manager import DynamicEventManager  # noqa: E402
from hppo_trainer import HPPOTrainer  # noqa: E402


DEBUG_DYNAMIC_SCOPE = "DEBUG_ONLY_NON_FORMAL_DYNAMIC_EXECUTION"
DEBUG_DYNAMIC_ARTIFACT_ROOT = os.path.join(_ROOT_DIR, "debug_artifacts", "dynamic")


def _require_debug_artifact_path(path: str, label: str) -> str:
    """Reject any attempt to write a single-Excel debug artifact as formal output."""
    candidate = os.path.abspath(path)
    root = os.path.abspath(DEBUG_DYNAMIC_ARTIFACT_ROOT)
    try:
        inside_root = os.path.commonpath([root, candidate]) == root
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(f"{label} 只能写入 DEBUG_ONLY 目录 {root}，拒绝 {candidate}。")
    return candidate


def _solution_cmax(solution: Optional[Any]) -> Optional[float]:
    return None if solution is None else float(solution.makespan)


def run_dynamic_instance(
    filepath: str,
    instance_name: str,
    pool_type: Optional[str] = None,
    d_h: int = 32,
    deterministic: bool = True,
    checkpoint_in: Optional[str] = None,
    train_episodes: int = 0,
    checkpoint_out: Optional[str] = None,
    debug_only: bool = False,
) -> Dict[str, Any]:
    """运行一个单 Excel 调试实例并返回可序列化报告。

    此入口不读取 V20261027 Manifest，也不具备正式训练或正式评测资格；
    调用方必须明确声明 ``debug_only=True``。
    """
    if debug_only is not True:
        raise ValueError(
            "run_dynamic_excel.py 是 DEBUG_ONLY 单算例工具。"
            "正式 V20261027 训练仅可使用 dynamic_multi_scenario_trainer.py；"
            "正式评测必须使用冻结检查点、活动 Manifest 与统一结果评估合同。"
        )
    if checkpoint_out:
        checkpoint_out = _require_debug_artifact_path(checkpoint_out, "DEBUG_ONLY checkpoint")
    filepath = os.path.abspath(filepath)
    validation = validate_dynamic_excel(filepath)
    initial_orders, insertion_events = load_dynamic_orders_and_events(filepath, instance_name)
    if not initial_orders:
        raise ValueError(f"实例 {instance_name!r} 没有初始订单。")

    inferred_pools = {order.pool_type for order in initial_orders}
    if pool_type is None:
        if len(inferred_pools) != 1:
            raise ValueError(
                f"实例 {instance_name!r} 同时包含多个资源池 {sorted(inferred_pools)}；"
                "请分别以 --pool-type D/B 运行。"
            )
        pool_type = next(iter(inferred_pools))
    if pool_type not in (POOL_D, POOL_B):
        raise ValueError(f"pool_type 必须为 {POOL_D!r} 或 {POOL_B!r}。")
    if inferred_pools != {pool_type}:
        raise ValueError(
            f"实例 {instance_name!r} 的初始订单资源池为 {sorted(inferred_pools)}，"
            f"与指定 pool_type={pool_type!r} 不一致。"
        )
    if any(event.order.pool_type != pool_type for event in insertion_events):
        raise ValueError("动态插单包含跨资源池订单；请按资源池拆分实例后分别运行。")

    manager = DynamicEventManager(
        pool_type=pool_type,
        initial_orders=initial_orders,
        events=insertion_events,
        instance_name=instance_name,
    )
    trainer = HPPOTrainer(
        orders=initial_orders,
        pool_type=pool_type,
        d_h=d_h,
        device=torch.device("cpu"),
    )
    if checkpoint_in:
        checkpoint = trainer.load_checkpoint(checkpoint_in, load_optimizer=train_episodes > 0)
        saved_pool = checkpoint.get("pool_type")
        if saved_pool is not None and saved_pool != pool_type:
            raise ValueError(
                f"模型权重资源池 {saved_pool!r} 与当前实例资源池 {pool_type!r} 不一致。"
            )

    if train_episodes < 0:
        raise ValueError("train_episodes 不能为负数。")
    training_logs = []
    for episode_idx in range(train_episodes):
        manager.reset()
        train_decisions = trainer.collect_dynamic_episode(
            manager,
            deterministic=False,
        )
        loss = trainer.update_policies()
        training_logs.append(
            {
                "episode": episode_idx + 1,
                "committed_events": sum(int(decision.committed) for decision in train_decisions),
                "trainable_events": sum(int(decision.trainable) for decision in train_decisions),
                "manager_samples": len(trainer.manager_buffer),
                "worker_samples": len(trainer.worker_buffer),
                "loss": loss,
            }
        )

    # 训练完成后恢复同一初始状态，再做一次独立执行评估。
    manager.reset()
    decisions = trainer.collect_dynamic_episode(
        manager,
        deterministic=deterministic,
    )
    if checkpoint_out:
        trainer.save_checkpoint(checkpoint_out)

    event_rows = []
    for decision in decisions:
        event_rows.append(
            {
                "event_id": decision.event_id,
                "order_id": decision.order_id,
                "committed": bool(decision.committed),
                "candidate_cmax": _solution_cmax(decision.candidate_solution),
                "actor_candidate_cmax": _solution_cmax(decision.actor_candidate_solution),
                "raw_actor_feasibility": float(decision.raw_actor_feasibility),
                "reward": float(decision.reward),
                "partition_plan": {oid: list(tpl) for oid, tpl in decision.partition_plan.items()},
                "worker_action_count": len(decision.worker_actions),
                "errors": list(decision.errors),
            }
        )

    return {
        "execution_scope": DEBUG_DYNAMIC_SCOPE,
        "formal_dynamic_eligible": False,
        "formal_dynamic_restriction": (
            "单 Excel 调试结果不得写入 V20261027 正式结果目录，"
            "不得作为正式训练、正式评测、收敛或统计结论。"
        ),
        "input_file": filepath,
        "instance_name": instance_name,
        "pool_type": pool_type,
        "validation_status": validation["status"],
        "initial_order_count": len(initial_orders),
        "dynamic_event_count": len(insertion_events),
        "processed_event_count": len(decisions),
        "events": event_rows,
        "final_cmax": float(manager.current_solution.makespan),
        "all_events_committed": all(decision.committed for decision in decisions),
        "pending_event_ids": list(manager.pending_event_ids),
        "training": training_logs,
        "checkpoint_in": None if checkpoint_in is None else os.path.abspath(checkpoint_in),
        "checkpoint_out": None if checkpoint_out is None else os.path.abspath(checkpoint_out),
        "decision_mode": "autonomous_dual_actor_no_anchor",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 PCB 动态插单双智能体重调度（仅 DEBUG_ONLY）")
    parser.add_argument("--debug-only", action="store_true", help="确认本次为非正式单算例调试")
    parser.add_argument("--xlsx", required=True, help="动态 Excel 文件路径")
    parser.add_argument("--sheet", required=True, help="实例工作表名，如 Sheet1")
    parser.add_argument("--pool-type", choices=(POOL_D, POOL_B), help="资源池；省略时从初始订单自动推断")
    parser.add_argument("--d-h", type=int, default=32, help="HGNN 隐藏维度，默认 32")
    parser.add_argument("--sample", action="store_true", help="采样动作；默认使用确定性 argmax")
    parser.add_argument("--checkpoint-in", help="可选：加载已有 HPPO 权重")
    parser.add_argument("--train-episodes", type=int, default=0, help="先进行动态训练的轮数，默认 0")
    parser.add_argument("--checkpoint-out", help="可选：训练/执行后保存 HPPO 权重")
    parser.add_argument("--json-out", help="可选：另存 JSON 报告，不覆盖输入 Excel")
    args = parser.parse_args()

    if not args.debug_only:
        parser.error(
            "此入口只允许 --debug-only。正式 V20261027 训练/评测不得通过单 Excel 脚本执行。"
        )
    if args.checkpoint_out:
        args.checkpoint_out = _require_debug_artifact_path(args.checkpoint_out, "DEBUG_ONLY checkpoint")
    if args.json_out:
        args.json_out = _require_debug_artifact_path(args.json_out, "DEBUG_ONLY JSON 报告")

    report = run_dynamic_instance(
        filepath=args.xlsx,
        instance_name=args.sheet,
        pool_type=args.pool_type,
        d_h=args.d_h,
        deterministic=not args.sample,
        checkpoint_in=args.checkpoint_in,
        train_episodes=args.train_episodes,
        checkpoint_out=args.checkpoint_out,
        debug_only=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        output_path = args.json_out
        if os.path.abspath(args.xlsx) == output_path:
            raise ValueError("JSON 报告路径不能覆盖输入 Excel。")
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"已写出 JSON 报告: {output_path}")


if __name__ == "__main__":
    main()
