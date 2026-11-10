"""核对当前 HPPO 配置合同，不声称完成超参数性能标定。

本脚本只验证代码默认值、双时间尺度学习率和奖励缩放实现是否与当前
参数规范一致。超参数优劣、收敛稳定性与统计显著性必须由冻结协议下的
多场景、多种子对照实验给出，不能由随机张量、伪优势或手写评价标签推断。
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
LAYER5_DIR = PROJECT_ROOT / "agent"
for path in (PROJECT_ROOT, LAYER5_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hppo_trainer import HPPOTrainer
from reward import compute_macro_event_reward


EXPECTED = {
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "clip_eps": 0.30,
    "c_val": 0.25,
    "c_ent": 0.0015,
    "max_grad_norm": 0.5,
    "ppo_epochs": 4,
    "target_kl": 0.05,
    "max_kl": 0.15,
    "split_lr": 3e-4,
    "placement_lr": 1.5e-4,
    "kappa_R": 10.0,
}


def _default(signature: inspect.Signature, name: str) -> Any:
    value = signature.parameters[name].default
    if value is inspect.Parameter.empty:
        raise AssertionError(f"{name} 没有默认值，无法完成配置合同核对。")
    return value


def validate_hppo_configuration_contract() -> Dict[str, Any]:
    trainer_sig = inspect.signature(HPPOTrainer.__init__)
    reward_sig = inspect.signature(compute_macro_event_reward)
    observed = {
        key: _default(trainer_sig, key)
        for key in (
            "gamma",
            "gae_lambda",
            "clip_eps",
            "c_val",
            "c_ent",
            "max_grad_norm",
            "ppo_epochs",
            "target_kl",
            "max_kl",
        )
    }
    observed["kappa_R"] = _default(reward_sig, "kappa_r")
    observed["split_lr"] = _default(trainer_sig, "lr")
    observed["placement_lr"] = _default(trainer_sig, "placement_lr")

    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED.items()
        if observed.get(key) != expected
    }
    if mismatches:
        raise AssertionError(f"HPPO 配置合同不一致：{mismatches}")

    return {
        "ok": True,
        "status": "CONFIGURATION_CONTRACT_CHECK_ONLY",
        "configured_parameters": EXPECTED,
        "observed_defaults": observed,
        "performance_calibration_completed": False,
        "limitations": [
            "未比较候选超参数的训练曲线、测试集 Cmax、方差或显著性。",
            "未据随机张量、伪优势或手写标签推断最优学习率、KL 稳定性或收敛速度。",
            "正式结论仍需固定 Manifest、冻结评测协议和多场景多种子实验。",
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> int:
    result = validate_hppo_configuration_contract()
    out_file = THIS_DIR / "标定结果_HPPO超参数与奖励函数缩放因子.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"配置合同诊断已记录至：{out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
