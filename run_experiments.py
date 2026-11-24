"""全项目统一实验调度入口。

用法：
  1. 在配置区设置 SEEDS、STAGES、START_MODE 与 RUN_STEPS 后直接运行。
  2. 可通过命令行参数覆盖配置，例如：
       python run_experiments.py --dry-run
       python run_experiments.py --seeds 1,2,3 --only train_main,eval_single_actor
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_protocol import validate_formal_seed

# 实验配置
SEEDS: List[int] = [1]                  # 正式种子范围 [1, 2, 3, 4, 5]
STAGES: List[int] = [1, 2, 3]           # 训练/评测阶段 [1, 2, 3]
DEVICE: str = "cuda"                    # "cuda" 或 "cpu"
START_MODE: str = "auto"                # "auto" | "resume" | "fresh"

RUN_STEPS: Dict[str, bool] = {
    # 训练阶段
    "train_main": True,                 # 主算法 AD-HG-DAPPO（Stage 1~3）
    "train_single_actor": True,         # 单 Actor 架构消融 AD-HG-SAPPO
    "train_expand_ablation": False,     # 全模板预展开消融 DAPPO-Expand
    # 基线对比
    "run_rules": True,                  # 六种调度规则（train/val/test 全划分）
    "calibrate_and_run_ig": False,      # 事件级 IG（先校准主算法耗时，再跑 10x/5s）
    # 消融评测与绘图
    "eval_single_actor": True,          # 主算法 vs 单 Actor 消融评测
    "eval_scope_ablation": True,        # 受影响域 K_scope 消融评测与散点图
    "eval_expand_ablation": False,      # 全模板预展开消融评测与对比图
    # 出表、统计检验与诊断
    "build_tables": True,               # 全量 CSV 与论文三线表生成
    "run_stats": False,                 # 配对非参数检验（需主算法 + 规则 + IG 完整结果）
    "report_runtime": False,            # 决策/训练耗时汇总报告
    "plot_diagnostics": True,           # 训练损失与验证收敛曲线绘制
}

RESULT_ROOT = PROJECT_ROOT / "result"


def _latest_match(base_dir: Path, pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching ``pattern`` under ``base_dir``."""
    if not base_dir.is_dir():
        return None
    matches = sorted(
        base_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _run_cmd(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    label: str = "",
) -> None:
    """Execute one subprocess under ``PROJECT_ROOT`` or print it in dry-run mode."""
    printable = " ".join(str(part) for part in cmd)
    prefix = f"[DRY-RUN] {label}: " if dry_run else f"[RUN] {label}: "
    print(f"{prefix}{printable}", flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    proc = subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"步骤 {label!r} 执行失败（exit code={proc.returncode}）：{printable}")


def _mode_flags(start_mode: str) -> List[str]:
    if start_mode == "fresh":
        return ["--fresh-start"]
    if start_mode == "resume":
        return ["--resume"]
    if start_mode == "auto":
        return []
    raise ValueError(f"不支持的 START_MODE={start_mode!r}，可选 'auto' / 'resume' / 'fresh'。")


def _step_train_main(
    py: str, seeds: Sequence[int], stages: Sequence[int], device: str, start_mode: str, dry_run: bool
) -> None:
    flags = _mode_flags(start_mode)
    for seed in seeds:
        for stage in stages:
            _run_cmd(
                [py, "train.py", "--stage", str(stage), "--seed", str(seed), "--device", device, *flags],
                dry_run=dry_run,
                label=f"train_main/stage{stage}/seed_{seed}",
            )


def _step_train_single_actor(
    py: str, seeds: Sequence[int], stages: Sequence[int], device: str, start_mode: str, dry_run: bool
) -> None:
    flags = _mode_flags(start_mode)
    script = str(PROJECT_ROOT / "experiments" / "ablations" / "single_actor" / "train_single_actor.py")
    for seed in seeds:
        for stage in stages:
            _run_cmd(
                [py, script, "--stage", str(stage), "--seed", str(seed), "--device", device, *flags],
                dry_run=dry_run,
                label=f"train_single_actor/stage{stage}/seed_{seed}",
            )


def _step_train_expand(
    py: str, seeds: Sequence[int], stages: Sequence[int], device: str, start_mode: str, dry_run: bool
) -> None:
    flags = _mode_flags(start_mode)
    script = str(PROJECT_ROOT / "experiments" / "ablations" / "expand_ablation" / "train_dappo_expand.py")
    for seed in seeds:
        for stage in stages:
            _run_cmd(
                [py, script, "--stage", str(stage), "--seed", str(seed), "--device", device, *flags],
                dry_run=dry_run,
                label=f"train_expand_ablation/stage{stage}/seed_{seed}",
            )


def _step_run_rules(py: str, stages: Sequence[int], dry_run: bool) -> None:
    script = str(PROJECT_ROOT / "experiments" / "runtime" / "decision_training_runtime.py")
    for split in ("train", "val", "test"):
        cmd = [py, script, "rules", "--split", split]
        if list(stages) != [1, 2, 3]:
            for stage in stages:
                _run_cmd([*cmd, "--stage", str(stage)], dry_run=dry_run, label=f"run_rules/{split}/stage{stage}")
        else:
            _run_cmd(cmd, dry_run=dry_run, label=f"run_rules/{split}")


def _step_calibrate_and_run_ig(
    py: str, seeds: Sequence[int], stages: Sequence[int], dry_run: bool
) -> None:
    cal_script = str(PROJECT_ROOT / "baselines" / "iterated_greedy" / "calibrate_main.py")
    ig_script = str(PROJECT_ROOT / "baselines" / "iterated_greedy" / "run_ig.py")
    seed = int(seeds[0])
    ig_root = RESULT_ROOT / "对比算法" / "ig"
    for stage in stages:
        ckpt = RESULT_ROOT / "主算法" / f"stage{stage}" / f"seed_{seed}" / f"stage{stage}_best.pt"
        if not dry_run and not ckpt.is_file():
            raise FileNotFoundError(f"calibrate_and_run_ig 缺少主算法检查点：{ckpt}")
        _run_cmd(
            [py, cal_script, "--checkpoint", str(ckpt), "--split", "test", "--stage", str(stage)],
            dry_run=dry_run,
            label=f"calibrate_main/stage{stage}",
        )
        if dry_run:
            timing_arg = f"<latest_main_timing_stage{stage}/main_event_times.csv>"
        else:
            timing_csv = _latest_match(ig_root, f"main_timing_stage{stage}_*/main_event_times.csv")
            if timing_csv is None:
                raise FileNotFoundError(f"未找到 Stage {stage} 校准生成的 main_event_times.csv")
            timing_arg = str(timing_csv)
        _run_cmd(
            [py, ig_script, "--split", "test", "--stage", str(stage), "--budget", "both", "--main-timing", timing_arg],
            dry_run=dry_run,
            label=f"run_ig/stage{stage}",
        )


def _step_eval_single_actor(
    py: str, seeds: Sequence[int], stages: Sequence[int], device: str, dry_run: bool
) -> None:
    script = str(PROJECT_ROOT / "experiments" / "ablations" / "single_actor" / "eval_single_actor.py")
    for seed in seeds:
        for stage in stages:
            _run_cmd(
                [py, script, "--stage", str(stage), "--split", "test", "--seed", str(seed), "--device", device],
                dry_run=dry_run,
                label=f"eval_single_actor/stage{stage}/seed_{seed}",
            )


def _step_eval_scope(py: str, stages: Sequence[int], dry_run: bool) -> None:
    eval_script = str(PROJECT_ROOT / "experiments" / "ablations" / "scope_ablation" / "evaluate_scope.py")
    plot_script = str(PROJECT_ROOT / "experiments" / "ablations" / "scope_ablation" / "plot_scope.py")
    scope_root = RESULT_ROOT / "消融实验" / "受影响范围"
    if list(stages) == [1, 2, 3]:
        _run_cmd([py, eval_script, "--scope", "all", "--method", "both"], dry_run=dry_run, label="eval_scope_ablation")
        if dry_run:
            _run_cmd(
                [py, plot_script, "--scope-eval", "<latest_scope_eval.csv>", "--output", "<latest_scope_dir/scope_scatter.png>"],
                dry_run=True,
                label="plot_scope_ablation",
            )
            return
        latest_csv = _latest_match(scope_root, "scope_*/scope_eval.csv")
        if latest_csv is not None:
            out_img = latest_csv.parent / "scope_scatter.png"
            if not out_img.exists():
                _run_cmd([py, plot_script, "--scope-eval", str(latest_csv), "--output", str(out_img)], dry_run=False, label="plot_scope_ablation")
    else:
        for stage in stages:
            _run_cmd(
                [py, eval_script, "--stage", str(stage), "--scope", "all", "--method", "both"],
                dry_run=dry_run,
                label=f"eval_scope_ablation/stage{stage}",
            )


def _step_eval_expand(
    py: str, seeds: Sequence[int], stages: Sequence[int], device: str, dry_run: bool
) -> None:
    eval_script = str(PROJECT_ROOT / "experiments" / "ablations" / "expand_ablation" / "evaluate_expand.py")
    plot_script = str(PROJECT_ROOT / "experiments" / "ablations" / "expand_ablation" / "plot_expand.py")
    seeds_str = ",".join(str(s) for s in seeds)
    stage_arg = "all" if list(stages) == [1, 2, 3] else str(stages[0])
    _run_cmd(
        [py, eval_script, "--stage", stage_arg, "--split", "test", "--seeds", seeds_str, "--device", device],
        dry_run=dry_run,
        label="eval_expand_ablation",
    )
    if dry_run:
        _run_cmd(
            [py, plot_script, "--expand-eval", "<latest_expand_eval.csv>", "--output", "<latest_expand_dir/expand_comparison.png>"],
            dry_run=True,
            label="plot_expand_ablation",
        )
        return
    latest_csv = _latest_match(RESULT_ROOT / "消融实验" / "全模板预展开" / "评测", "expand_*/expand_eval.csv")
    if latest_csv is not None:
        out_img = latest_csv.parent / "expand_comparison.png"
        if not out_img.exists():
            _run_cmd([py, plot_script, "--expand-eval", str(latest_csv), "--output", str(out_img)], dry_run=False, label="plot_expand_ablation")


def _step_build_tables(py: str, seeds: Sequence[int], dry_run: bool) -> None:
    csv_script = str(PROJECT_ROOT / "experiments" / "table_generation" / "build_authoritative_csv.py")
    tab_script = str(PROJECT_ROOT / "experiments" / "table_generation" / "generate_paper_tables.py")
    seeds_str = ",".join(str(s) for s in seeds)
    _run_cmd([py, csv_script, "--seeds", seeds_str, "--include-l4", "--skip-missing-seeds"], dry_run=dry_run, label="build_authoritative_csv")
    _run_cmd([py, tab_script, "--seeds", seeds_str, "--skip-missing-seeds"], dry_run=dry_run, label="generate_paper_tables")


def _step_run_stats(py: str, seeds: Sequence[int], dry_run: bool) -> None:
    script = str(PROJECT_ROOT / "experiments" / "statistical_comparison" / "run_statistical_comparison.py")
    seeds_str = ",".join(str(s) for s in seeds)
    if dry_run:
        _run_cmd(
            [py, script, "--input", "<test_eval.csv / ig_runs.csv / rule_scenario_results.csv>", "--seeds", seeds_str],
            dry_run=True,
            label="run_stats",
        )
        return
    inputs: List[str] = []
    for seed in seeds:
        eval_csv = RESULT_ROOT / "数据核查与自动出表" / f"seed_{seed}" / "test_eval.csv"
        if eval_csv.is_file():
            inputs.extend(["--input", str(eval_csv)])
    for ig_csv in sorted((RESULT_ROOT / "对比算法" / "ig").glob("ig_*/ig_runs.csv")):
        inputs.extend(["--input", str(ig_csv)])
    for rule_csv in (
        sorted((RESULT_ROOT / "耗时与效率" / "规则耗时").glob("**/rule_scenario_results.csv"))
        + sorted((RESULT_ROOT / "耗时实验" / "决策与训练耗时").glob("rules_*/rule_scenario_results.csv"))
    ):
        inputs.extend(["--input", str(rule_csv)])
    if not inputs:
        raise FileNotFoundError("run_stats 未找到任何可统计的 test_eval.csv / ig_runs.csv / rule_scenario_results.csv。")
    _run_cmd([py, script, *inputs, "--seeds", seeds_str], dry_run=False, label="run_stats")


def _step_report_runtime(py: str, dry_run: bool) -> None:
    script = str(PROJECT_ROOT / "experiments" / "runtime" / "decision_training_runtime.py")
    if dry_run:
        _run_cmd([py, script, "report", "--event-file", "<event_times.csv...>"], dry_run=True, label="report_runtime")
        return
    event_args: List[str] = []
    for p in sorted((RESULT_ROOT / "对比算法" / "ig").glob("main_timing_*/main_event_times.csv")):
        event_args.extend(["--event-file", str(p)])
    for p in sorted((RESULT_ROOT / "对比算法" / "ig").glob("ig_*/ig_events.csv")):
        event_args.extend(["--event-file", str(p)])
    for p in (
        sorted((RESULT_ROOT / "耗时与效率" / "规则耗时").glob("**/rule_event_times.csv"))
        + sorted((RESULT_ROOT / "耗时实验" / "决策与训练耗时").glob("rules_*/rule_event_times.csv"))
    ):
        event_args.extend(["--event-file", str(p)])
    for p in sorted((RESULT_ROOT / "消融实验" / "受影响范围").glob("scope_*/scope_events.csv")):
        event_args.extend(["--event-file", str(p)])
    if not event_args:
        raise FileNotFoundError("report_runtime 未找到任何耗时事件记录 CSV。")
    _run_cmd([py, script, "report", *event_args], dry_run=False, label="report_runtime")


def _step_plot_diagnostics(
    py: str, seeds: Sequence[int], stages: Sequence[int], dry_run: bool
) -> None:
    script = str(PROJECT_ROOT / "scripts" / "plot_hppo_diagnostics.py")
    for label_dir, base in (("主算法", RESULT_ROOT / "主算法"), ("单Actor消融", RESULT_ROOT / "消融实验")):
        for seed in seeds:
            for stage in stages:
                log_json = base / f"stage{stage}" / f"seed_{seed}" / f"stage{stage}_train_log.json"
                if dry_run or log_json.is_file():
                    out_dir = RESULT_ROOT / "科研图表" / label_dir / f"stage{stage}" / f"seed_{seed}"
                    _run_cmd(
                        [py, script, "--log-json", str(log_json), "--out-dir", str(out_dir)],
                        dry_run=dry_run,
                        label=f"plot_diagnostics/{label_dir}/stage{stage}/seed_{seed}",
                    )


def run_pipeline(
    *,
    seeds: Sequence[int],
    stages: Sequence[int],
    device: str,
    start_mode: str,
    steps: Dict[str, bool],
    dry_run: bool = False,
) -> None:
    """Execute enabled experiment stages in dependency order."""
    valid_seeds = [validate_formal_seed(s) for s in seeds]
    valid_stages = [int(s) for s in stages]
    if any(s not in (1, 2, 3) for s in valid_stages):
        raise ValueError(f"STAGES 必须为 1/2/3 的子集，收到：{stages}")

    py = sys.executable
    if steps.get("train_main"):
        _step_train_main(py, valid_seeds, valid_stages, device, start_mode, dry_run)
    if steps.get("train_single_actor"):
        _step_train_single_actor(py, valid_seeds, valid_stages, device, start_mode, dry_run)
    if steps.get("train_expand_ablation"):
        _step_train_expand(py, valid_seeds, valid_stages, device, start_mode, dry_run)
    if steps.get("run_rules"):
        _step_run_rules(py, valid_stages, dry_run)
    if steps.get("calibrate_and_run_ig"):
        _step_calibrate_and_run_ig(py, valid_seeds, valid_stages, dry_run)
    if steps.get("eval_single_actor"):
        _step_eval_single_actor(py, valid_seeds, valid_stages, device, dry_run)
    if steps.get("eval_scope_ablation"):
        _step_eval_scope(py, valid_stages, dry_run)
    if steps.get("eval_expand_ablation"):
        _step_eval_expand(py, valid_seeds, valid_stages, device, dry_run)
    if steps.get("build_tables"):
        _step_build_tables(py, valid_seeds, dry_run)
    if steps.get("run_stats"):
        _step_run_stats(py, valid_seeds, dry_run)
    if steps.get("report_runtime"):
        _step_report_runtime(py, dry_run)
    if steps.get("plot_diagnostics"):
        _step_plot_diagnostics(py, valid_seeds, valid_stages, dry_run)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=None, help="逗号分隔随机种子，如 1 或 1,2,3")
    parser.add_argument("--stages", default=None, help="逗号分隔阶段号，如 1,2,3")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--start-mode", choices=("auto", "resume", "fresh"), default=None)
    parser.add_argument(
        "--only",
        default=None,
        help=f"仅执行指定步骤（逗号分隔），可选：{','.join(RUN_STEPS.keys())}",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印执行计划与命令，不实际运行")
    args = parser.parse_args(argv)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()] if args.seeds else list(SEEDS)
    stages = [int(x.strip()) for x in args.stages.split(",") if x.strip()] if args.stages else list(STAGES)
    device = args.device or DEVICE
    start_mode = args.start_mode or START_MODE

    steps = dict(RUN_STEPS)
    if args.only:
        chosen = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = [x for x in chosen if x not in steps]
        if unknown:
            parser.error(f"未知步骤名：{unknown}；可选值为 {list(steps.keys())}")
        steps = {key: (key in chosen) for key in steps}

    run_pipeline(
        seeds=seeds,
        stages=stages,
        device=device,
        start_mode=start_mode,
        steps=steps,
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
