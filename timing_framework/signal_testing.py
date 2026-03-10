"""
信号检验法模块
==============

实现《择时因子的择时框架》中的"信号检验法"（信号检验法），
对择时因子的有效性进行系统性评估。

检验流程
--------
第一步 —— 信号生成：
    将连续的因子值转化为离散的交易信号，支持三种生成方式：

    * 阈值法（Threshold）：
        因子 > 上阈值 → 多头信号（+1）
        因子 < 下阈值 → 空头信号（-1）

    * 均线法（Moving Average）：
        因子 > 因子自身的 N 日均线 → 多头信号（+1）
        因子 < 因子自身的 N 日均线 → 空头信号（-1）

    * 极值法（Percentile）：
        因子 ≤ 历史低分位数 → 多头信号（+1）
        因子 ≥ 历史高分位数 → 空头信号（-1）

第二步 —— 信号评估：
    对照实际收益率，衡量信号质量。核心指标：

    * 胜率（Win Rate）        ：方向正确的信号占比
    * 预期收益（Avg Return）  ：多/空信号期间的平均收益率
    * 盈亏比（P/L Ratio）     ：平均盈利 / 平均亏损的绝对值
    * T 检验显著性            ：检验多/空信号期间收益率是否存在显著差异

约定
----
信号取值：+1 = 做多，-1 = 做空，0 = 空仓（不操作）。
收益率需提前做移位对齐（shift(-n)）或传入已移位的序列。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# 结果容器
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SignalTestResult:
    """
    单次信号检验的完整指标集合。

    所有收益率指标均以原始小数形式表示（非百分比）。

    属性
    ----
    n_long : int               多头信号次数
    long_win_rate : float      多头胜率  P(return > 0 | 多头信号)
    long_avg_return : float    多头期间平均收益率（期望 > 0）
    long_avg_profit : float    多头盈利交易的平均收益
    long_avg_loss : float      多头亏损交易的平均亏损
    long_pl_ratio : float      多头盈亏比 |avg_profit / avg_loss|
    n_short : int              空头信号次数
    short_win_rate : float     空头胜率  P(return < 0 | 空头信号)
    short_avg_return : float   空头期间平均收益率（期望 < 0）
    short_avg_profit : float   空头盈利交易的平均收益
    short_avg_loss : float     空头亏损交易的平均亏损
    short_pl_ratio : float     空头盈亏比
    overall_win_rate : float   综合胜率 = (正确多头 + 正确空头) / 总信号数
    long_short_return_spread : float  多头均收益 − 空头均收益（期望 > 0）
    t_statistic : float        Welch t 检验统计量（多头 vs 空头收益率）
    p_value : float            双侧 p 值
    is_significant : bool      p < 0.05 则为显著
    long_coverage : float      多头信号占全部交易日的比例
    short_coverage : float     空头信号占全部交易日的比例
    """

    # --- 多头信号指标 ---
    n_long: int
    long_win_rate: float
    long_avg_return: float
    long_avg_profit: float
    long_avg_loss: float
    long_pl_ratio: float

    # --- 空头信号指标 ---
    n_short: int
    short_win_rate: float
    short_avg_return: float
    short_avg_profit: float
    short_avg_loss: float
    short_pl_ratio: float

    # --- 综合指标 ---
    overall_win_rate: float
    long_short_return_spread: float

    # --- 统计显著性 ---
    t_statistic: float
    p_value: float
    is_significant: bool

    # --- 信号覆盖率 ---
    long_coverage: float
    short_coverage: float

    def summary(self) -> str:
        lines = [
            "┌─── 信号检验结果 ──────────────────────────────────────────┐",
            f"│ 多头信号数 : {self.n_long:>5}  ({self.long_coverage:>5.1%} 的时间)",
            f"│   胜   率  : {self.long_win_rate:>7.2%}",
            f"│   均收益率 : {self.long_avg_return:>+9.4f}",
            f"│   盈亏比   : {self.long_pl_ratio:>7.2f}",
            f"│ 空头信号数 : {self.n_short:>5}  ({self.short_coverage:>5.1%} 的时间)",
            f"│   胜   率  : {self.short_win_rate:>7.2%}",
            f"│   均收益率 : {self.short_avg_return:>+9.4f}",
            f"│   盈亏比   : {self.short_pl_ratio:>7.2f}",
            f"│ 综合胜率        : {self.overall_win_rate:>6.2%}",
            f"│ 多空收益差      : {self.long_short_return_spread:>+9.4f}",
            f"│ T 统计量        : {self.t_statistic:>+8.4f}",
            f"│ P 值            : {self.p_value:>8.4f}",
            f"│ 显著性 (α=5%)   : {'✓ 显著' if self.is_significant else '✗ 不显著':>5}",
            "└────────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SignalTestResult(胜率={self.overall_win_rate:.2%}, "
            f"多空差={self.long_short_return_spread:+.4f}, "
            f"p={self.p_value:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 信号生成函数
# ══════════════════════════════════════════════════════════════════════════════


def generate_threshold_signals(
    factor: pd.Series,
    upper: float = 1.0,
    lower: float = -1.0,
) -> pd.Series:
    """
    阈值法信号生成。

    因子超过上阈值时市场偏强 → 多头（+1）；
    因子低于下阈值时市场偏弱 → 空头（-1）；
    介于两者之间 → 沿袭上一日。

    对于标准化因子（均值≈0，标准差≈1），推荐 upper=1.0、lower=-1.0，
    可覆盖约 16% 的极端值区域。

    参数
    ----
    factor : pd.Series
        因子序列（建议经预处理标准化后使用）。
    upper : float
        超过此值发出多头信号（+1）。
    lower : float
        低于此值发出空头信号（-1）。

    返回
    ----
    pd.Series
        整数信号序列：+1、0 或 -1。
    """
    sigma_ = factor.std()
    mean_ = factor.mean()
    signals = pd.Series(np.nan, index=factor.index, dtype=int)
    signals[factor > upper * sigma_ + mean_] = 1
    signals[factor < lower * sigma_ + mean_] = -1
    signals.fillna(method="ffill", inplace=True)
    return signals


def generate_ma_signals(factor: pd.Series, window: int = 20) -> pd.Series:
    """
    均线法信号生成。

    因子高于自身 N 日均线（处于上升趋势）→ 多头（+1）；
    因子低于自身 N 日均线（处于下降趋势）→ 空头（-1）。

    参数
    ----
    factor : pd.Series
        因子序列。
    window : int
        均线计算的回望窗口。

    返回
    ----
    pd.Series
        整数信号序列：+1、0 或 -1。前 window//2 个值为 0（预热期）。
    """
    ma = factor.rolling(window, min_periods=window // 2).mean()
    signals = pd.Series(0, index=factor.index, dtype=int)
    signals[factor > ma] = 1
    signals[factor < ma] = -1
    # 预热期设为 0，避免初始不稳定信号
    signals.iloc[: window // 2] = 0
    return signals


def generate_percentile_signals(
    factor: pd.Series,
    lower_pct: float = 0.20,
    upper_pct: float = 0.80,
    window: Optional[int] = None,
) -> pd.Series:
    """
    极值法（历史分位数）信号生成。

    因子值 ≤ 历史低分位数（历史偏低/超卖）→ 多头（+1）；
    因子值 ≥ 历史高分位数（历史偏高/超买）→ 空头（-1）。

    参数
    ----
    factor : pd.Series
        因子序列。
    lower_pct : float
        多头信号的下分位数阈值（如 0.20 表示历史最低 20%）。
    upper_pct : float
        空头信号的上分位数阈值（如 0.80 表示历史最高 20%）。
    window : int 或 None
        滚动窗口大小。若为 None，则使用全样本分位数
        （存在未来函数，仅用于研究，实盘请务必设置 window）。

    返回
    ----
    pd.Series
        整数信号序列：+1、0 或 -1。
    """
    signals = pd.Series(0, index=factor.index, dtype=int)

    if window is not None:
        # 滚动分位数：严格使用历史数据，无未来函数
        for i in range(window, len(factor)):
            subset = factor.iloc[i - window : i]
            low_val = subset.quantile(lower_pct)
            high_val = subset.quantile(upper_pct)
            val = factor.iloc[i]
            if val <= low_val:
                signals.iloc[i] = 1
            elif val >= high_val:
                signals.iloc[i] = -1
    else:
        # 全样本分位数（有未来函数，仅用于研究）
        low_val = factor.quantile(lower_pct)
        high_val = factor.quantile(upper_pct)
        signals[factor <= low_val] = 1
        signals[factor >= high_val] = -1

    return signals


def generate_zero_signals(factor: pd.Series) -> pd.Series:
    """
    零值法信号生成。

    因子值小于 均值（标准化下为0） 时 → 空仓、空头（-1）；
    因子值大于 均值（标准化下为0） 时 → 多头（+1）。
    因子值等于 均值（标准化下为0） 时 → 沿袭上一日。

    参数
    ----
    factor : pd.Series
        因子序列。

    返回
    ----
    pd.Series
        整数信号序列：+1、0、-1 。
    """
    mean_ = factor.mean()
    signals = pd.Series(-1, index=factor.index, dtype=int)
    signals[factor > mean_] = 1
    signals[factor == mean_] = np.nan
    signals.fillna(method='ffill', inplace=True)
    return signals



def generate_diff_zero_signals(factor: pd.Series) -> pd.Series:
    """
    零值法（因子差）信号生成。

    因子值小于等于 0 时 → 空仓（-1）；
    因子值大于 0 时 → 多头（+1）。

    参数
    ----
    factor : pd.Series
        因子序列。

    返回
    ----
    pd.Series
        整数信号序列：+1、0 。
    """
    factor = factor.diff()
    signals = pd.Series(-1, index=factor.index, dtype=int)
    signals[factor > 0] = 1
    return signals


def generate_ma_diff_zero_signals(
    factor: pd.Series,
    short_window: int = 20,
    long_window: int = 60,
) -> pd.Series:
    """
    长短均线差值法信号生成。

    计算因子自身的短期均线与长期均线之差，再将差值送入零值法：
        差值 > 差值序列均值 → 多头（+1）；
        差值 < 差值序列均值 → 空头（-1）。

    参数
    ----
    factor       : pd.Series  因子序列。
    short_window : int        短均线窗口（交易日）。
    long_window  : int        长均线窗口（交易日），须 > short_window。

    返回
    ----
    pd.Series  整数信号序列：+1 或 -1。
    """
    short_ma = factor.rolling(short_window, min_periods=short_window // 2).mean()
    long_ma  = factor.rolling(long_window,  min_periods=long_window  // 2).mean()
    diff     = short_ma - long_ma
    return generate_zero_signals(diff)





def _side_metrics(
    side_returns: pd.Series, correct_positive: bool
) -> Tuple[float, float, float, float, float]:
    """
    计算单侧（多头或空头）的胜率、平均收益、平均盈利、平均亏损和盈亏比。

    参数
    ----
    side_returns : pd.Series
        该方向信号期间的实际收益率序列。
    correct_positive : bool
        True 表示多头方向（正收益 = 正确）；
        False 表示空头方向（负收益 = 正确）。

    返回
    ----
    tuple
        (胜率, 平均收益, 平均盈利, 平均亏损, 盈亏比)
    """
    n = len(side_returns)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_return = side_returns.mean()

    if correct_positive:
        wins = side_returns[side_returns > 0]
        losses = side_returns[side_returns <= 0]
    else:
        wins = side_returns[side_returns < 0]
        losses = side_returns[side_returns >= 0]

    win_rate = len(wins) / n
    avg_profit = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    if avg_loss != 0 and avg_profit != 0:
        pl_ratio = abs(avg_profit / avg_loss)
    elif avg_loss == 0:
        pl_ratio = float("inf")  # 从未亏损
    else:
        pl_ratio = 0.0

    return win_rate, avg_return, avg_profit, avg_loss, pl_ratio


def evaluate_signals(
    signals: pd.Series, forward_returns: pd.Series
) -> SignalTestResult:
    """
    对照未来实际收益率，全面评估信号质量。

    通过索引对齐 `signals` 和 `forward_returns`，过滤掉空仓期（signal == 0），
    计算完整的信号质量指标。

    参数
    ----
    signals : pd.Series
        离散信号序列（+1 多头、-1 空头、0 空仓）。
        t 时刻的信号表示从 t 到 t+1（或 t+n）的持仓方向。
    forward_returns : pd.Series
        持仓期间的实际收益率。对于 1 日预测，应传入 ``returns.shift(-1)``。

    返回
    ----
    SignalTestResult
        完整的信号评估指标。
    """
    # 对齐并过滤空仓期
    df = pd.DataFrame({"signal": signals, "ret": forward_returns}).dropna()
    df = df[df["signal"] != 0]

    n_total_periods = len(signals)
    long_df = df[df["signal"] == 1]
    short_df = df[df["signal"] == -1]

    long_rets = long_df["ret"]
    short_rets = short_df["ret"]

    # 分别计算多/空两侧指标
    l_wr, l_avg, l_profit, l_loss, l_pl = _side_metrics(long_rets, True)
    s_wr, s_avg, s_profit, s_loss, s_pl = _side_metrics(short_rets, False)

    # 综合胜率：正确多头 + 正确空头
    n_long = len(long_rets)
    n_short = len(short_rets)
    correct_longs = (long_rets > 0).sum() if n_long > 0 else 0
    correct_shorts = (short_rets < 0).sum() if n_short > 0 else 0
    total_signals = n_long + n_short
    overall_wr = (correct_longs + correct_shorts) / total_signals if total_signals > 0 else 0.0

    # Welch t 检验：H0: 多头均收益 == 空头均收益
    if n_long > 1 and n_short > 1:
        t_stat, p_val = stats.ttest_ind(long_rets, short_rets, equal_var=False)
    else:
        t_stat, p_val = 0.0, 1.0

    # 信号覆盖率
    long_cov = n_long / n_total_periods if n_total_periods > 0 else 0.0
    short_cov = n_short / n_total_periods if n_total_periods > 0 else 0.0

    return SignalTestResult(
        n_long=n_long,
        long_win_rate=l_wr,
        long_avg_return=l_avg,
        long_avg_profit=l_profit,
        long_avg_loss=l_loss,
        long_pl_ratio=l_pl,
        n_short=n_short,
        short_win_rate=s_wr,
        short_avg_return=s_avg,
        short_avg_profit=s_profit,
        short_avg_loss=s_loss,
        short_pl_ratio=s_pl,
        overall_win_rate=overall_wr,
        long_short_return_spread=l_avg - s_avg,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        is_significant=(float(p_val) < 0.05),
        long_coverage=long_cov,
        short_coverage=short_cov,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 信号方法完整结果容器（含IS/OOS）
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SignalMethodResult:
    """
    单种信号检验方法的完整结果（含全样本、样本内和样本外三段）。

    属性
    ----
    full       : SignalTestResult  全样本检验结果
    signal     : pd.Series         全样本信号序列（+1/0/-1）
    insample   : SignalTestResult  样本内检验结果（可选）
    outsample  : SignalTestResult  样本外检验结果（可选）
    split_date : pd.Timestamp      IS/OOS 分割日期（可选）
    """

    full: SignalTestResult
    signal: pd.Series
    insample: Optional[SignalTestResult] = None
    outsample: Optional[SignalTestResult] = None
    split_date: Optional["pd.Timestamp"] = None


# ══════════════════════════════════════════════════════════════════════════════
# 高层封装类
# ══════════════════════════════════════════════════════════════════════════════


class SignalTester:
    """
    信号检验器，依次运行三种信号生成方式并对各自进行评估。

    参数
    ----
    forward_period : int
        因子预测的未来周期数。类内部自动完成收益率移位对齐。
    """

    def __init__(self, forward_period: int = 1) -> None:
        self.forward_period = forward_period

    # ──────────────────────────────────────────────────────────────── #

    def run_threshold_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
        upper: float = 1.0,
        lower: float = -1.0,
    ) -> SignalTestResult:
        """阈值法信号检验。"""
        fwd = returns.shift(-self.forward_period)
        signals = generate_threshold_signals(factor, upper, lower)
        return [evaluate_signals(signals, fwd),signals]

    def run_ma_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
        window: int = 20,
    ) -> SignalTestResult:
        """均线法信号检验。"""
        fwd = returns.shift(-self.forward_period)
        signals = generate_ma_signals(factor, window)
        return [evaluate_signals(signals, fwd),signals]

    def run_percentile_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
        lower_pct: float = 0.20,
        upper_pct: float = 0.80,
        window: Optional[int] = None,
    ) -> SignalTestResult:
        """极值法（历史分位数）信号检验。"""
        fwd = returns.shift(-self.forward_period)
        signals = generate_percentile_signals(factor, lower_pct, upper_pct, window)
        return [evaluate_signals(signals, fwd),signals]

    def run_zero_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
    ) -> SignalTestResult:
        """零值法信号检验。"""
        fwd = returns.shift(-self.forward_period)
        signals = generate_zero_signals(factor)
        return [evaluate_signals(signals, fwd),signals]

    def run_diff_zero_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
    ) -> SignalTestResult:
        """零值法（因子差）信号检验。"""
        fwd = returns.shift(-self.forward_period)
        signals = generate_diff_zero_signals(factor)
        return [evaluate_signals(signals, fwd),signals]

    def run_ma_diff_zero_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
        short_window: int = 20,
        long_window: int = 60,
    ) -> SignalTestResult:
        """长短均线差值法信号检验（差值送入零值法）。"""
        fwd = returns.shift(-self.forward_period)
        signals = generate_ma_diff_zero_signals(factor, short_window, long_window)
        return [evaluate_signals(signals, fwd), signals]


    def run_all(
        self,
        factor: pd.Series,
        returns: pd.Series,
        threshold_upper: float = 1.0,
        threshold_lower: float = -1.0,
        ma_window: int = 20,
        pct_lower: float = 0.20,
        pct_upper: float = 0.80,
        pct_window: Optional[int] = None,
        test_ratio: float = 0.30,
        ma_short: int = 20,
        ma_long: int = 60,
    ) -> Dict[str, SignalMethodResult]:
        """
        同时运行所有信号检验方法，返回 SignalMethodResult 字典。

        每种方法均包含全样本、样本内（IS）和样本外（OOS）三段评估结果。
        IS/OOS 切分点由 test_ratio 控制（最后 test_ratio 比例为样本外）。

        返回
        ----
        dict
            键：'threshold'、'moving_average'、'percentile'、'zero'、
            'diff_zero'、'MA250_diff_zero'；
            值：SignalMethodResult（含 .full / .signal / .insample / .outsample / .split_date）。
        """
        # ── 确定 IS/OOS 切分点 ─────────────────────────────────────────
        valid = factor.dropna()
        n = len(valid)
        split_date = None
        is_factor = oos_factor = is_returns = oos_returns = None

        if n >= 10 and 0.0 < test_ratio < 1.0:
            split_idx = int(n * (1.0 - test_ratio))
            if 0 < split_idx < n:
                split_date  = valid.index[split_idx]
                is_factor   = factor[:split_date]
                oos_factor  = factor[split_date:]
                is_returns  = returns[:split_date]
                oos_returns = returns[split_date:]

        # ── 内部辅助：对子集安全地运行某种方法 ────────────────────────
        def _try_is_oos(run_fn_is, run_fn_oos):
            """返回 (is_result, oos_result)，任一失败则返回 None。"""
            if split_date is None:
                return None, None
            try:
                is_r  = run_fn_is()[0]
                oos_r = run_fn_oos()[0]
                return is_r, oos_r
            except Exception:
                return None, None

        # ── 逐方法计算 ──────────────────────────────────────────────────
        results: Dict[str, SignalMethodResult] = {}

        # 阈值法
        full_thr = self.run_threshold_test(factor, returns, threshold_upper, threshold_lower)
        is_thr, oos_thr = _try_is_oos(
            lambda: self.run_threshold_test(is_factor, is_returns, threshold_upper, threshold_lower),
            lambda: self.run_threshold_test(oos_factor, oos_returns, threshold_upper, threshold_lower),
        )
        results["threshold"] = SignalMethodResult(
            full=full_thr[0], signal=full_thr[1],
            insample=is_thr, outsample=oos_thr, split_date=split_date,
        )

        # 均线法
        full_ma = self.run_ma_test(factor, returns, ma_window)
        is_ma, oos_ma = _try_is_oos(
            lambda: self.run_ma_test(is_factor, is_returns, ma_window),
            lambda: self.run_ma_test(oos_factor, oos_returns, ma_window),
        )
        results["moving_average"] = SignalMethodResult(
            full=full_ma[0], signal=full_ma[1],
            insample=is_ma, outsample=oos_ma, split_date=split_date,
        )

        # 极值法
        full_pct = self.run_percentile_test(factor, returns, pct_lower, pct_upper, pct_window)
        is_pct, oos_pct = _try_is_oos(
            lambda: self.run_percentile_test(is_factor, is_returns, pct_lower, pct_upper, pct_window),
            lambda: self.run_percentile_test(oos_factor, oos_returns, pct_lower, pct_upper, pct_window),
        )
        results["percentile"] = SignalMethodResult(
            full=full_pct[0], signal=full_pct[1],
            insample=is_pct, outsample=oos_pct, split_date=split_date,
        )

        # 零值法
        full_z = self.run_zero_test(factor, returns)
        is_z, oos_z = _try_is_oos(
            lambda: self.run_zero_test(is_factor, is_returns),
            lambda: self.run_zero_test(oos_factor, oos_returns),
        )
        results["zero"] = SignalMethodResult(
            full=full_z[0], signal=full_z[1],
            insample=is_z, outsample=oos_z, split_date=split_date,
        )

        # 差分零值法
        full_dz = self.run_diff_zero_test(factor, returns)
        is_dz, oos_dz = _try_is_oos(
            lambda: self.run_diff_zero_test(is_factor, is_returns),
            lambda: self.run_diff_zero_test(oos_factor, oos_returns),
        )
        results["diff_zero"] = SignalMethodResult(
            full=full_dz[0], signal=full_dz[1],
            insample=is_dz, outsample=oos_dz, split_date=split_date,
        )

        # 250日移动平均差分零值法
        full_ma250 = self.run_diff_zero_test(factor.rolling(250).mean(), returns)
        is_ma250, oos_ma250 = _try_is_oos(
            lambda: self.run_diff_zero_test(is_factor.rolling(250).mean(), is_returns),
            lambda: self.run_diff_zero_test(oos_factor.rolling(250).mean(), oos_returns),
        )
        results["MA250_diff_zero"] = SignalMethodResult(
            full=full_ma250[0], signal=full_ma250[1],
            insample=is_ma250, outsample=oos_ma250, split_date=split_date,
        )

        # 长短均线差值法
        full_mad = self.run_ma_diff_zero_test(factor, returns, ma_short, ma_long)
        is_mad, oos_mad = _try_is_oos(
            lambda: self.run_ma_diff_zero_test(is_factor, is_returns, ma_short, ma_long),
            lambda: self.run_ma_diff_zero_test(oos_factor, oos_returns, ma_short, ma_long),
        )
        results["ma_diff_zero"] = SignalMethodResult(
            full=full_mad[0], signal=full_mad[1],
            insample=is_mad, outsample=oos_mad, split_date=split_date,
        )

        return results

