"""
reward.py
=========
双时间尺度分层差分奖励实现
对应当前方法文档《PCB.docx》式 (30)、(31) 及参数规范第 5.1 节。

核心功能：
  1. 宏观事件级差分奖励 (式 30):
       R_e^S = - [C_max(S_e^{actor}) - C_max(S^{e-1})] / kappa_R,  kappa_R = 10.0
       基于车间自身前后状态差分构建。
  2. 微观放置单步差分奖励 (式 31):
       r_{e,k}^P = -10.0 * Delta C_m(k) / max(100.0, C_max(S_{e,k}^{curr}))
       其中 Delta C_m(k) = C_hat_m^{(k)} - C_bar_g^{(k-1)}，提供密集局部反馈。
"""

from __future__ import annotations

import math


REWARD_CONTRACT_VERSION = "pool-mean-micro-incremental-macro-v20261105"


# =========================================================================
# 状态差分奖励计算函数（式 30 与式 31）
# =========================================================================

def compute_macro_event_reward(
    committed_makespan: float,
    base_makespan: float,
    kappa_r: float = 10.0,
) -> float:
    """
    宏观事件级自主奖励 (式 30 / 参数规范 5.1 节):
      R_e^S = - [C_max(S_e^{actor}) - C_max(S^{e-1})] / kappa_R
    完全基于车间在事件前后两个已提交排程的状态差分，不使用外部启发式基线。
    """
    values = (committed_makespan, base_makespan, kappa_r)
    if not all(math.isfinite(float(value)) for value in values) or float(kappa_r) <= 0.0:
        raise ValueError("Macro reward inputs must be finite and kappa_r must be positive.")
    delta_cmax = float(committed_makespan) - float(base_makespan)
    return -float(delta_cmax) / float(kappa_r)


def compute_micro_step_reward(
    predicted_machine_cmax: float,
    pool_mean_cmax: float,
    current_actor_makespan: float,
    scale: float = 10.0,
    min_norm: float = 100.0,
) -> float:
    """
    微观放置单步自主差分奖励 (式 31 / 参数规范 5.1 节):
      r_{e,k}^P = -10.0 * Delta C_m(k) / max(100.0, C_max(S_{e,k}^{curr}))
    其中 Delta C_m(k) = C_hat_m^{(k)} - C_bar_g^{(k-1)}。
    施加相对池均值偏晚完工的负反馈，归一化量只取当前 Actor 临时状态。
    """
    values = (predicted_machine_cmax, pool_mean_cmax, current_actor_makespan, scale, min_norm)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Micro reward inputs must be finite.")
    if float(scale) <= 0.0 or float(min_norm) <= 0.0:
        raise ValueError("Micro reward scale and min_norm must be positive.")
    delta_cm = float(predicted_machine_cmax) - float(pool_mean_cmax)
    denom = max(float(min_norm), float(current_actor_makespan))
    return -float(scale) * delta_cm / denom
