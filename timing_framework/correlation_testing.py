"""
相关性检验法模块
================

实现《择时因子的择时框架》中的"相关性检验法"，
使用信息系数（IC）分析对择时因子的预测能力进行定量评估。

择时因子的 IC 与截面因子的 IC 有何不同？
-----------------------------------------
* **截面因子**：在每个时间截面，计算 N 只股票的因子向量与收益向量之间的相关系数。
* **择时因子**：每个时间点只有一只资产、一个因子值，
  因此 IC 定义为因子值时间序列与后续收益率时间序列之间的相关系数。

本模块在滚动窗口上计算 IC，得到随时间变化的 IC 序列，
以反映因子预测能力在不同市场环境下的演变。

核心指标
--------
* IC Mean（IC均值）   ：滚动 IC 的平均值，为正表示因子方向正确。
* IC Std（IC标准差）  ：IC 值的波动性，越小说明因子越稳定。
* ICIR                ：IC均值 / IC标准差，衡量因子的信息比率。
                        择时因子参考：ICIR > 0.5 偏好，> 1.0 优秀。
* IC > 0 比例         ：IC 序列中正值的占比。
* T 检验              ：检验 IC 序列均值是否显著偏离零（H₀: IC_mean = 0）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# 结果容器
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ICTestResult:
    """
    单个预测周期的完整 IC 分析结果。

    属性
    ----
    period : int
        预测周期（单位：期数）。
    ic_mean : float
        滚动 IC 序列的均值。
    ic_std : float
        滚动 IC 序列的标准差。
    icir : float
        信息比率 = ic_mean / ic_std。
    ic_positive_ratio : float
        IC > 0 的窗口占比。
    ic_series : pd.Series
        完整的滚动 IC 时间序列。
    t_statistic : float
        H₀: IC_mean = 0 的 t 统计量。
    p_value : float
        双侧 p 值。
    is_significant : bool
        p < 0.05 时为 True。
    method : str
        相关系数计算方法：'pearson' 或 'spearman'。
    """

    period: int
    ic_mean: float
    ic_std: float
    icir: float
    ic_positive_ratio: float
    ic_series: pd.Series
    t_statistic: float
    p_value: float
    is_significant: bool
    method: str = "pearson"

    def summary(self) -> str:
        # 评级标签
        icir_label = (
            "优秀 (>1.0)"
            if abs(self.icir) > 1.0
            else "良好 (>0.5)"
            if abs(self.icir) > 0.5
            else "偏弱"
        )
        lines = [
            f"┌─── IC 分析  (预测 {self.period}d，{self.method}) ─────────────┐",
            f"│ IC 均值          : {self.ic_mean:>+8.4f}",
            f"│ IC 标准差        : {self.ic_std:>8.4f}",
            f"│ ICIR             : {self.icir:>+8.4f}  [{icir_label}]",
            f"│ IC > 0 占比      : {self.ic_positive_ratio:>7.2%}",
            f"│ T 统计量         : {self.t_statistic:>+8.4f}",
            f"│ P 值             : {self.p_value:>8.4f}",
            f"│ 显著性 (α=5%)    : {'✓ 显著' if self.is_significant else '✗ 不显著':>5}",
            "└────────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ICTestResult(period={self.period}, ic_mean={self.ic_mean:+.4f}, "
            f"icir={self.icir:+.4f}, p={self.p_value:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 核心计算函数
# ══════════════════════════════════════════════════════════════════════════════


def calculate_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    method: str = "pearson",
) -> float:
    """
    计算因子与未来收益率之间的单个 IC 值。

    两个序列按索引对齐后删除 NaN，再计算相关系数。

    参数
    ----
    factor : pd.Series
        t 时刻的因子值。
    forward_returns : pd.Series
        t 时刻的未来收益率（已提前移位，即 ``returns.shift(-n)``）。
    method : str
        'pearson'（线性相关）或 'spearman'（秩相关，对异常值更鲁棒）。

    返回
    ----
    float
        IC 值（取值范围 [-1, +1]），数据不足时返回 NaN。
    """
    df = pd.DataFrame({"f": factor, "r": forward_returns}).dropna()
    if len(df) < 3:
        return np.nan

    if method == "spearman":
        ic, _ = stats.spearmanr(df["f"], df["r"])
    else:
        ic, _ = stats.pearsonr(df["f"], df["r"])

    return float(ic)


def calculate_rolling_ic(
    factor: pd.Series,
    returns: pd.Series,
    forward_period: int = 1,
    rolling_window: int = 60,
    method: str = "pearson",
) -> pd.Series:
    """
    计算滚动 IC 时间序列。

    在 t 时刻，使用最近 `rolling_window` 个（因子值，未来收益率）对计算 IC，
    从而观察因子预测能力随时间的演变。

    参数
    ----
    factor : pd.Series
        因子序列。
    returns : pd.Series
        日收益率序列（未移位）。
    forward_period : int
        预测周期（天数）。
    rolling_window : int
        每次 IC 计算使用的历史观测数量。
    method : str
        'pearson' 或 'spearman'。

    返回
    ----
    pd.Series
        滚动 IC 序列，前 `rolling_window` 个值为 NaN。
    """
    fwd_returns = returns.shift(-forward_period)
    df = pd.DataFrame({"f": factor, "r": fwd_returns})

    ic_values = pd.Series(index=factor.index, dtype=float)

    for i in range(rolling_window, len(df)):
        window_data = df.iloc[i - rolling_window : i].dropna()
        # 数据量不足窗口的四分之一时跳过，避免噪声过大
        if len(window_data) < max(5, rolling_window // 4):
            continue

        if method == "spearman":
            ic, _ = stats.spearmanr(window_data["f"], window_data["r"])
        else:
            ic, _ = stats.pearsonr(window_data["f"], window_data["r"])

        ic_values.iloc[i] = ic

    return ic_values


# ══════════════════════════════════════════════════════════════════════════════
# 高层封装类
# ══════════════════════════════════════════════════════════════════════════════


class CorrelationTester:
    """
    基于 IC 的相关性检验器。

    参数
    ----
    method : str
        相关系数计算方法——'pearson'（默认）或 'spearman'。
        Spearman 为秩相关，对异常值更稳健。
    """

    def __init__(self, method: str = "pearson") -> None:
        if method not in ("pearson", "spearman"):
            raise ValueError("method 必须为 'pearson' 或 'spearman'")
        self.method = method

    # ──────────────────────────────────────────────────────────────── #

    def run_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
        forward_period: int = 1,
        rolling_window: Optional[int] = None,
    ) -> ICTestResult:
        """
        对单个预测周期运行 IC 分析。

        参数
        ----
        factor : pd.Series
            因子序列（与 ``returns`` 索引对齐）。
        returns : pd.Series
            日收益率序列（内部自动完成移位）。
        forward_period : int
            预测周期（天数）。
        rolling_window : int 或 None
            滚动 IC 窗口。默认 min(60, n_obs/6)，最小 20。

        返回
        ----
        ICTestResult
        """
        n_obs = factor.dropna().shape[0]
        if rolling_window is None:
            rolling_window = max(20, min(60, n_obs // 6))

        ic_series = calculate_rolling_ic(
            factor, returns, forward_period, rolling_window, self.method
        )
        ic_clean = ic_series.dropna()

        if len(ic_clean) < 3:
            # 备用方案：计算全样本单一 IC 值
            fwd = returns.shift(-forward_period)
            single_ic = calculate_ic(factor, fwd, self.method)
            return ICTestResult(
                period=forward_period,
                ic_mean=single_ic,
                ic_std=np.nan,
                icir=np.nan,
                ic_positive_ratio=float(single_ic > 0) if not np.isnan(single_ic) else np.nan,
                ic_series=ic_series,
                t_statistic=np.nan,
                p_value=1.0,
                is_significant=False,
                method=self.method,
            )

        ic_mean = float(ic_clean.mean())
        ic_std = float(ic_clean.std(ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_pos_ratio = float((ic_clean > 0).mean())

        # 单样本 t 检验：H₀: IC均值 = 0
        t_stat, p_val = stats.ttest_1samp(ic_clean, 0.0)

        return ICTestResult(
            period=forward_period,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_positive_ratio=ic_pos_ratio,
            ic_series=ic_series,
            t_statistic=float(t_stat),
            p_value=float(p_val),
            is_significant=(float(p_val) < 0.05),
            method=self.method,
        )

    def run_multi_period(
        self,
        factor: pd.Series,
        returns: pd.Series,
        periods: List[int] = [1, 5, 10, 20],
        rolling_window: Optional[int] = None,
    ) -> Dict[int, ICTestResult]:
        """
        对多个预测周期同时运行 IC 分析。

        参数
        ----
        factor : pd.Series
            因子序列。
        returns : pd.Series
            日收益率序列。
        periods : list of int
            待检验的预测周期列表（如 [1, 5, 10, 20]）。
        rolling_window : int 或 None
            公共滚动窗口大小；若为 None，则各周期自动选择。

        返回
        ----
        dict
            {周期: ICTestResult}
        """
        return {
            p: self.run_test(factor, returns, p, rolling_window) for p in periods
        }
