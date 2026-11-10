"""
run_validation_suite.py
=======================
【全系统参数诊断统一执行入口】
功能说明：
  1. 自动化顺次调度当前 5 项配置/结构/协议诊断子程序；
  2. 捕获各子程序的执行状态码与输出日志；
  3. 整合诊断执行报告；
  4. 明确区分“脚本可执行/合同一致”与“性能、统计、现场有效性已验证”。

运行命令：
  python run_validation_suite.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS_IN_ORDER = [
    ("程序 1 (受影响域)", "validate_scope_limits.py"),
    ("程序 2 (物理合同)", "validate_time_model.py"),
    ("程序 3 (分批与掩码)", "validate_partition_templates_and_masks.py"),
    ("程序 4 (PPO配置)", "validate_hppo_and_reward_config.py"),
    ("程序 5 (数据与种子)", "validate_dataset_split_and_seeds.py"),
]


def run_all_calibrations():
    print("=" * 85)
    print("      PCB 钻孔车间事件驱动动态重调度配置、结构与协议诊断总程序")
    print("=" * 85)
    print(f"执行工作区: {_THIS_DIR}")
    print(f"待执行诊断程序总数: {len(SCRIPTS_IN_ORDER)}")
    print("=" * 85 + "\n")

    python_exe = sys.executable
    total_start = time.perf_counter()
    summary_results = []

    for idx, (label, script_name) in enumerate(SCRIPTS_IN_ORDER, 1):
        script_path = os.path.join(_THIS_DIR, script_name)
        assert os.path.exists(script_path), f"找不到子程序: {script_path}"

        print(f">>> [{idx}/{len(SCRIPTS_IN_ORDER)}] 启动调度: {label} -> {script_name}")
        t_sub_start = time.perf_counter()
        proc = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        sub_elapsed = time.perf_counter() - t_sub_start

        is_success = (proc.returncode == 0)
        summary_results.append({
            "step": idx,
            "label": label,
            "script": script_name,
            "success": is_success,
            "returncode": proc.returncode,
            "elapsed_seconds": round(sub_elapsed, 3),
        })

        if is_success:
            print(f"    [PASS] 诊断脚本执行完成 (耗时: {sub_elapsed:.2f}s)\n")
        else:
            print(f"    [FAIL] 标定异常退出 (返回码: {proc.returncode})")
            print("--- 错误日志片段 ---")
            print(proc.stderr[-500:] if proc.stderr else proc.stdout[-500:])
            print("-" * 50 + "\n")

    total_elapsed = time.perf_counter() - total_start
    all_passed = all(r["success"] for r in summary_results)

    # 输出综合看板
    print("=" * 85)
    print("                 全系统配置、结构与协议诊断执行汇总")
    print("=" * 85)
    print(f"{'序号':>4} | {'标定模块':>16} | {'标定程序脚本名称':>45} | {'耗时(s)':>8} | {'状态'}")
    print("-" * 85)
    for r in summary_results:
        status_str = "[PASS 成功]" if r["success"] else "[FAIL 失败]"
        print(f"{r['step']:>4} | {r['label']:>16} | {r['script']:>45} | {r['elapsed_seconds']:>8.2f} | {status_str}")

    print("-" * 85)
    passed_count = sum(1 for r in summary_results if r["success"])
    print(f"总计耗时: {total_elapsed:.2f} 秒 | 诊断脚本执行成功: {passed_count}/{len(summary_results)}")
    print("=" * 85)

    if all_passed:
        print("\n【证据边界】当前诊断脚本均可执行；这只证明所检查的配置、结构或协议合同未触发错误。")
        print("              它不证明超参数最优、算法收敛、显著优于基线、现场参数有效或论文实验闭环。")
    else:
        print("\n【警告】部分参数标定程序未能通过，请审查对应子程序日志！")

    # 汇总落盘
    summary_file = os.path.join(_THIS_DIR, "全体系核心参数标定与确定性验证总报告.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "title": "Configuration, Structural and Protocol Diagnostic Report",
            "evidence_status": "STRUCTURAL_DIAGNOSTIC_SUITE_ONLY",
            "all_passed": all_passed,
            "performance_calibration_completed": False,
            "field_validation_completed": False,
            "note": "all_passed only means every listed diagnostic script exited successfully; it is not a performance or publication-readiness claim",
            "total_elapsed_seconds": round(total_elapsed, 3),
            "modules_executed": summary_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)

    print(f"\n标定总纲报告已落盘至: {summary_file}\n")
    return all_passed


if __name__ == "__main__":
    success = run_all_calibrations()
    sys.exit(0 if success else 1)
