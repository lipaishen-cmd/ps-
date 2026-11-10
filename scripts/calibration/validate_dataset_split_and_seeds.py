"""Validate the active dynamic-data contract and formal five-seed requirement.

This is a protocol validator only. It never fabricates makespan samples,
effect sizes, confidence intervals, or significance claims. Real statistics
may be computed only after the five isolated formal runs have completed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from configs.experiment_protocol import FORMAL_SEEDS
_DATA_DIR = _PROJECT_ROOT / "data"
_ACTIVE_CONTRACT = _DATA_DIR / "active_experiment_package.json"
_MULTI_SEED_REGISTRY = _PROJECT_ROOT / "result" / "formal_multiseed" / "formal_multiseed_registry.json"
_FORMAL_SEEDS = list(FORMAL_SEEDS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_experimental_protocol() -> Dict[str, Any]:
    """Validate metadata only; do not interpret protocol as an experiment result."""
    if not _ACTIVE_CONTRACT.is_file():
        raise FileNotFoundError(f"找不到活动数据合同：{_ACTIVE_CONTRACT}")
    contract = json.loads(_ACTIVE_CONTRACT.read_text(encoding="utf-8"))
    expected = {"train": 18, "val": 6, "test": 6}
    if contract.get("expected_scenarios") != expected:
        raise ValueError("活动合同不满足当前三组 18/6/6 场景协议。")
    if contract.get("data_evidence_status") != "PRE_FORMAL_ONLY_PROXY_DATA":
        raise ValueError("活动合同必须保留 PRE_FORMAL_ONLY_PROXY_DATA 证据边界。")

    manifest = _PROJECT_ROOT / contract["dataset_relative_path"] / contract["manifest_filename"]
    if _sha256(manifest) != contract.get("manifest_sha256"):
        raise ValueError("活动 Manifest SHA-256 与活动合同不一致。")
    policy = _PROJECT_ROOT / contract["eligibility_policy"]["relative_path"]
    if _sha256(policy) != contract["eligibility_policy"].get("sha256"):
        raise ValueError("资格策略 SHA-256 与活动合同不一致。")

    status = "PROTOCOL_LOCKED_RESULTS_NOT_RUN"
    completed_seeds = []
    if _MULTI_SEED_REGISTRY.is_file():
        registry = json.loads(_MULTI_SEED_REGISTRY.read_text(encoding="utf-8"))
        completed_seeds = sorted(
            int(record["seed"])
            for record in registry.get("runs", [])
            if record.get("status") == "COMPLETED"
        )
        if completed_seeds == _FORMAL_SEEDS:
            status = "5_SEED_RUNS_COMPLETED_STATISTICAL_ANALYSIS_REQUIRED"

    return {
        "ok": True,
        "status": status,
        "data_evidence_status": contract["data_evidence_status"],
        "expected_scenarios": expected,
        "formal_seed_requirement": _FORMAL_SEEDS,
        "completed_seeds": completed_seeds,
        "manifest_sha256": contract["manifest_sha256"],
        "eligibility_policy_sha256": contract["eligibility_policy"]["sha256"],
        "note": "本结果仅验证协议与已登记运行；未生成或推断任何算法性能、显著性或现场有效性结论。",
    }


def main() -> int:
    result = validate_experimental_protocol()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
