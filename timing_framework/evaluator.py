"""
综合评价模块
============

提供核心类 :class:`TimingFactorEvaluator`，整合完整的择时因子评估流程，
并生成综合评分和可视化图表。

评估流程
--------
1. 因子预处理  —— 可选 MAD 去极值 + 滚动 Z-score 标准化
2. 信号检验    —— 阈值/均线/极值三种方法，计算胜率和盈亏比
3. IC 检验     —— 多周期滚动 IC 及 ICIR
4. 回归检验    —— OLS 因子系数显著性及滚动稳定性
5. 稳健性评估  —— 样本内外分割 + 市场状态分组

综合评分
--------
各维度评分范围 0~1，加权合成综合评分。
权重：IC 30%，信号 30%，回归 20%，稳健性 20%。

等级划分：
    A（优秀）：综合评分 ≥ 0.75
    B（良好）：综合评分 ≥ 0.60
    C（一般）：综合评分 ≥ 0.45
    D（较差）：综合评分 ≥ 0.30
    F（无效）：综合评分  < 0.30
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .correlation_testing import CorrelationTester, ICTestResult
from .preprocessing import FactorPreprocessor
from .regression_testing import RegressionTester, RegressionResult
from .robustness import InSampleOutSampleResult, RobustnessTester
from .signal_testing import SignalTester, SignalTestResult


# ══════════════════════════════════════════════════════════════════════════════
# 评分容器
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FactorScore:
    """
    择时因子综合评分结果。

    各维度评分均在 [0, 1] 范围内。

    属性
    ----
    ic_score : float
        基于 |ICIR| 的得分。ICIR ≥ 1.0 → 满分 1.0；线性内插。
    signal_score : float
        基于胜率和统计显著性的得分。
    regression_score : float
        基于 β 显著性和 R² 的得分。
    robustness_score : float
        基于样本外 IC 衰减和符号一致性的得分。
    composite_score : float
        加权综合评分（IC 30% + 信号 30% + 回归 20% + 稳健性 20%）。
    grade : str
        字母等级：A / B / C / D / F。
    """

    ic_score: float
    signal_score: float
    regression_score: float
    robustness_score: float
    composite_score: float
    grade: str

    def summary(self) -> str:
        bar_width = 20

        def _bar(v: float) -> str:
            """生成进度条字符串。"""
            filled = int(round(v * bar_width))
            return "█" * filled + "░" * (bar_width - filled)

        lines = [
            "┌─── 因子综合评分 ────────────────────────────────────────────┐",
            f"│ IC 评分          {_bar(self.ic_score)} {self.ic_score:.2f}",
            f"│ 信号评分         {_bar(self.signal_score)} {self.signal_score:.2f}",
            f"│ 回归评分         {_bar(self.regression_score)} {self.regression_score:.2f}",
            f"│ 稳健性评分       {_bar(self.robustness_score)} {self.robustness_score:.2f}",
            f"│ ─────────────────────────────────────────────────────────── │",
            f"│ 综合评分         {_bar(self.composite_score)} {self.composite_score:.2f}  等级: {self.grade}",
            "└────────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FactorScore(综合={self.composite_score:.2f}, 等级={self.grade})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 主评估器
# ══════════════════════════════════════════════════════════════════════════════


class TimingFactorEvaluator:
    """
    择时因子综合评价框架的核心类。

    将预处理、信号检验、IC/ICIR 分析、OLS 回归和稳健性评估
    整合为统一的评估工作流。

    参数
    ----
    factor_name : str
        因子名称（用于报告标题和图表标注）。
    forward_period : int
        预测周期（交易日数），即 F(t) 预测 R(t+n) 中的 n。
    ic_method : str
        IC 计算方法：'pearson'（默认）或 'spearman'。

    示例
    ----
    基本用法::

        evaluator = TimingFactorEvaluator('MA动量', forward_period=1)
        evaluator.evaluate(factor_series, returns_series, prices=price_series)
        evaluator.report()
        fig = evaluator.plot()
        fig.savefig('factor_eval.png', dpi=100, bbox_inches='tight')
    """

    # 各维度评分权重
    _SCORE_WEIGHTS = {"ic": 0.30, "signal": 0.30, "regression": 0.20, "robustness": 0.20}

    def __init__(
        self,
        factor_name: str = "因子",
        forward_period: int = 1,
        ic_method: str = "pearson",
    ) -> None:
        self.factor_name = factor_name
        self.forward_period = forward_period
        self.ic_method = ic_method

        # 各子模块实例
        self._preprocessor = FactorPreprocessor()
        self._signal_tester = SignalTester(forward_period)
        self._ic_tester = CorrelationTester(ic_method)
        self._reg_tester = RegressionTester()
        self._robustness_tester = RobustnessTester(forward_period)

        # 结果存储（由 evaluate() 填充）
        self._factor: Optional[pd.Series] = None
        self._returns: Optional[pd.Series] = None
        self._prices: Optional[pd.Series] = None
        self._signal_results: Optional[Dict[str, SignalTestResult]] = None
        self._ic_results: Optional[Dict[int, ICTestResult]] = None
        self._reg_result: Optional[RegressionResult] = None
        self._robustness_result: Optional[InSampleOutSampleResult] = None
        self._regime_results: Optional[Dict] = None
        self._rolling_reg: Optional[pd.DataFrame] = None

    # ────────────────────────────────────────────────────────────────── #
    #  主评估方法                                                          #
    # ────────────────────────────────────────────────────────────────── #

    def evaluate(
        self,
        factor: pd.Series,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        preprocess: bool = True,
        rolling_window: int = 252,
        run_robustness: bool = True,
        run_rolling_regression: bool = False,
        signal_kwargs: Optional[dict] = None,
        ic_periods: Optional[List[int]] = None,
    ) -> "TimingFactorEvaluator":
        """
        运行完整的因子评估流程。

        参数
        ----
        factor : pd.Series
            原始因子序列，需与 ``returns`` 共享日期时间索引。
        returns : pd.Series
            日收益率序列（如 ``prices.pct_change()``）。
        prices : pd.Series 或 None
            收盘价序列（市场状态检验所需；可选）。
        preprocess : bool
            是否执行 MAD 去极值 + 滚动 Z-score 标准化。
        rolling_window : int
            滚动预处理的回望窗口（交易日数）。
        run_robustness : bool
            是否运行样本内外检验和市场状态检验。
        run_rolling_regression : bool
            是否运行（耗时较长的）滚动 OLS 回归。
        signal_kwargs : dict 或 None
            覆盖 :meth:`SignalTester.run_all` 的默认参数。
            可用键：threshold_upper, threshold_lower, ma_window, pct_lower, pct_upper。
        ic_periods : list of int 或 None
            IC 分析的预测周期列表，默认 [1, 5, 10, 20]。

        返回
        ----
        self
            支持链式调用。
        """
        self._returns = returns
        self._prices = prices

        # ─── 1. 因子预处理 ─────────────────────────────────────────────
        if preprocess:
            self._factor = FactorPreprocessor.preprocess(
                factor,
                winsorize=True,
                standardize=True,
                rolling_window=rolling_window,
            )
        else:
            self._factor = factor.copy()

        # ─── 2. 信号检验 ────────────────────────────────────────────────
        self._signal_results = self._signal_tester.run_all(
            self._factor, returns, **(signal_kwargs or {})
        )

        # ─── 3. IC / ICIR 检验 ─────────────────────────────────────────
        _ic_periods = ic_periods or [1, 5, 10, 20]
        self._ic_results = self._ic_tester.run_multi_period(
            self._factor, returns, periods=_ic_periods
        )

        # ─── 4. 回归检验 ────────────────────────────────────────────────
        try:
            self._reg_result = self._reg_tester.run_regression(
                self._factor, returns, forward_period=self.forward_period
            )
        except Exception as exc:
            warnings.warn(f"回归失败：{exc}", stacklevel=2)
            self._reg_result = None

        # ─── 5. 稳健性评估 ─────────────────────────────────────────────
        if run_robustness:
            try:
                self._robustness_result = (
                    self._robustness_tester.insample_outsample_test(
                        self._factor, returns
                    )
                )
            except Exception as exc:
                warnings.warn(f"样本内外检验失败：{exc}", stacklevel=2)

            if prices is not None:
                try:
                    self._regime_results = (
                        self._robustness_tester.market_regime_test(
                            self._factor, returns, prices
                        )
                    )
                except Exception as exc:
                    warnings.warn(f"市场状态检验失败：{exc}", stacklevel=2)

        # ─── 6. 可选：滚动回归 ─────────────────────────────────────────
        if run_rolling_regression:
            try:
                self._rolling_reg = self._reg_tester.rolling_regression(
                    self._factor, returns, forward_period=self.forward_period
                )
            except Exception as exc:
                warnings.warn(f"滚动回归失败：{exc}", stacklevel=2)

        return self

    # ────────────────────────────────────────────────────────────────── #
    #  综合评分                                                            #
    # ────────────────────────────────────────────────────────────────── #

    def score(self) -> FactorScore:
        """
        计算因子综合质量评分。

        返回
        ----
        FactorScore
        """
        # ── IC 评分：|ICIR| → [0, 1] ────────────────────────────────
        ic_score = 0.0
        if self._ic_results:
            ic_res = self._ic_results.get(self.forward_period)
            if ic_res and not np.isnan(ic_res.icir):
                # 线性映射：|ICIR| 在 [0, 1.0] 区间映射到 [0, 1] 并截断
                ic_score = min(1.0, abs(ic_res.icir))

        # ── 信号评分：胜率 + 显著性加分 ────────────────────────────
        signal_score = 0.0
        if self._signal_results:
            thr = self._signal_results.get("threshold")
            if thr:
                # 胜率在 [0.50, 0.70] 区间线性映射到 [0, 1]
                wr_score = min(1.0, max(0.0, (thr.overall_win_rate - 0.50) / 0.20))
                sig_bonus = 0.20 if thr.is_significant else 0.0
                signal_score = min(1.0, wr_score + sig_bonus)

        # ── 回归评分：β 显著性 + R² 加分 ───────────────────────────
        regression_score = 0.0
        if self._reg_result:
            base = 0.50 if self._reg_result.is_significant else 0.10
            r2_bonus = min(0.50, self._reg_result.r_squared * 10)
            regression_score = min(1.0, base + r2_bonus)

        # ── 稳健性评分：OOS 同向 + 衰减程度 ────────────────────────
        robustness_score = 0.5  # 未运行检验时的默认分
        if self._robustness_result:
            same_sign = float(self._robustness_result.is_robust)
            deg = abs(self._robustness_result.ic_degradation)
            # 衰减越小分越高
            deg_score = max(0.0, 1.0 - deg)
            robustness_score = 0.50 * same_sign + 0.50 * deg_score

        # ── 综合加权 ────────────────────────────────────────────────
        w = self._SCORE_WEIGHTS
        composite = (
            w["ic"] * ic_score
            + w["signal"] * signal_score
            + w["regression"] * regression_score
            + w["robustness"] * robustness_score
        )

        grade = (
            "A" if composite >= 0.75
            else "B" if composite >= 0.60
            else "C" if composite >= 0.45
            else "D" if composite >= 0.30
            else "F"
        )

        return FactorScore(
            ic_score=ic_score,
            signal_score=signal_score,
            regression_score=regression_score,
            robustness_score=robustness_score,
            composite_score=composite,
            grade=grade,
        )

    # ────────────────────────────────────────────────────────────────── #
    #  文字报告                                                            #
    # ────────────────────────────────────────────────────────────────── #

    def report(self) -> None:
        """向标准输出打印完整的因子评估报告。"""
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  择时因子评估：{self.factor_name}")
        print(f"  预测周期：{self.forward_period} 日 | IC 方法：{self.ic_method}")
        print(f"{sep}\n")

        # 信号检验
        print("── 1. 信号检验法 ───────────────────────────────────────────")
        for method_name, result in (self._signal_results or {}).items():
            label_map = {
                "threshold": "阈值法",
                "moving_average": "均线法",
                "percentile": "极值法",
            }
            label = label_map.get(method_name, method_name.upper())
            print(f"\n  [{label}]")
            print(result.summary())

        # IC 检验
        print("\n── 2. 相关性检验法（IC / ICIR）─────────────────────────────")
        for period, ic_res in (self._ic_results or {}).items():
            print(f"\n{ic_res.summary()}")

        # 回归检验
        print("\n── 3. 回归模型法 ───────────────────────────────────────────")
        if self._reg_result:
            print(self._reg_result.summary())
        else:
            print("  （未获得回归结果）")

        # 稳健性评估
        print("\n── 4. 稳健性评估 ───────────────────────────────────────────")
        if self._robustness_result:
            print(self._robustness_result.summary())
        else:
            print("  （未运行稳健性检验）")

        if self._regime_results:
            print("\n  市场状态分组：")
            regime_label = {"bull": "牛市", "bear": "熊市", "sideways": "震荡市"}
            for regime, ic_r in self._regime_results.items():
                label = regime_label.get(regime, regime)
                print(
                    f"    {label:<5}：IC={ic_r.ic_mean:>+.4f}  "
                    f"ICIR={ic_r.icir:>+.4f}  "
                    f"显著={'✓' if ic_r.is_significant else '✗'}"
                )

        # 综合评分
        print(f"\n── 5. 综合评分 ─────────────────────────────────────────────")
        print(self.score().summary())
        print(f"\n{sep}\n")

    # ────────────────────────────────────────────────────────────────── #
    #  可视化                                                              #
    # ────────────────────────────────────────────────────────────────── #

    def plot(self, figsize: tuple = (16, 13)) -> plt.Figure:
        """
        生成包含 8 个子图的综合评估图表。

        面板布局：
            第 0 行：因子时间序列（占 2 列） | 滚动 IC 序列
            第 1 行：信号平均收益率 | 胜率对比 | 多周期 IC/ICIR
            第 2 行：收益分布直方图 | 因子-收益散点图 | 综合评分条形图

        参数
        ----
        figsize : tuple
            图表尺寸（英寸）。

        返回
        ----
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
        fig.suptitle(
            f"择时因子评估：{self.factor_name}  "
            f"（预测 {self.forward_period}日）",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        # ── 第 0 行 ────────────────────────────────────────────────────
        ax00 = fig.add_subplot(gs[0, :2])   # 因子时间序列（宽图）
        self._plot_factor_series(ax00)

        ax02 = fig.add_subplot(gs[0, 2])    # 滚动 IC 序列
        self._plot_rolling_ic(ax02)

        # ── 第 1 行 ────────────────────────────────────────────────────
        ax10 = fig.add_subplot(gs[1, 0])    # 信号平均收益率
        self._plot_avg_returns(ax10)

        ax11 = fig.add_subplot(gs[1, 1])    # 三种方法胜率对比
        self._plot_win_rates(ax11)

        ax12 = fig.add_subplot(gs[1, 2])    # 多周期 IC & ICIR
        self._plot_ic_periods(ax12)

        # ── 第 2 行 ────────────────────────────────────────────────────
        ax20 = fig.add_subplot(gs[2, 0])    # 按信号分组的收益分布
        self._plot_return_distribution(ax20)

        ax21 = fig.add_subplot(gs[2, 1])    # 因子值 vs 未来收益散点图
        self._plot_scatter(ax21)

        ax22 = fig.add_subplot(gs[2, 2])    # 综合评分条形图
        self._plot_score_bars(ax22)

        return fig

    # ────────────────────────────────────────────────────────────────── #
    #  绘图辅助方法                                                        #
    # ────────────────────────────────────────────────────────────────── #

    def _plot_factor_series(self, ax: plt.Axes) -> None:
        """绘制因子时间序列，标注 ±1σ 阈值区域。"""
        if self._factor is None:
            return
        self._factor.plot(ax=ax, color="#2196F3", linewidth=0.8, alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.axhline(1.0, color="#F44336", linewidth=0.6, linestyle=":", alpha=0.7,
                   label="±1σ 阈值")
        ax.axhline(-1.0, color="#4CAF50", linewidth=0.6, linestyle=":", alpha=0.7)
        # 高亮超过阈值的区域
        ax.fill_between(self._factor.index, 1.0, self._factor,
                        where=self._factor > 1.0, alpha=0.15, color="#F44336")
        ax.fill_between(self._factor.index, -1.0, self._factor,
                        where=self._factor < -1.0, alpha=0.15, color="#4CAF50")
        ax.set_title("因子值（预处理后）", fontsize=10)
        ax.set_ylabel("标准化因子值")
        ax.legend(fontsize=7)

    def _plot_rolling_ic(self, ax: plt.Axes) -> None:
        """绘制滚动 IC 时间序列。"""
        if not self._ic_results or self.forward_period not in self._ic_results:
            ax.set_title("滚动 IC（数据不足）")
            return
        ic_series = self._ic_results[self.forward_period].ic_series.dropna()
        ic_series.plot(ax=ax, color="#FF9800", linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ic_mean = self._ic_results[self.forward_period].ic_mean
        ax.axhline(ic_mean, color="#2196F3", linewidth=1.0, linestyle="--",
                   label=f"均值={ic_mean:+.3f}")
        ax.set_title(f"滚动 IC（预测 {self.forward_period}日）", fontsize=10)
        ax.set_ylabel("IC")
        ax.legend(fontsize=7)

    def _plot_avg_returns(self, ax: plt.Axes) -> None:
        """绘制多空信号期间的平均收益率柱形图。"""
        if not self._signal_results or "threshold" not in self._signal_results:
            return
        r = self._signal_results["threshold"]
        vals = [r.long_avg_return * 100, r.short_avg_return * 100]
        colors = ["#4CAF50" if v > 0 else "#F44336" for v in vals]
        bars = ax.bar(["多头信号", "空头信号"], vals, color=colors,
                      alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.005 if v >= 0 else -0.010),
                    f"{v:+.3f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.set_title("各信号平均收益率（阈值法）", fontsize=10)
        ax.set_ylabel("平均收益率 (%)")

    def _plot_win_rates(self, ax: plt.Axes) -> None:
        """绘制三种信号方法的综合胜率对比柱形图。"""
        if not self._signal_results:
            return
        methods = list(self._signal_results.keys())
        method_labels = {"threshold": "阈值法", "moving_average": "均线法", "percentile": "极值法"}
        labels = [method_labels.get(m, m) for m in methods]
        win_rates = [self._signal_results[m].overall_win_rate * 100 for m in methods]
        colors = ["#4CAF50" if wr > 50 else "#F44336" for wr in win_rates]
        ax.bar(range(len(methods)), win_rates, color=colors, alpha=0.8,
               edgecolor="black", linewidth=0.5)
        ax.axhline(50, color="#F44336", linewidth=1.2, linestyle="--",
                   label="50% 基准线")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("三种方法胜率对比", fontsize=10)
        ax.set_ylabel("综合胜率 (%)")
        ax.set_ylim(35, 75)
        ax.legend(fontsize=7)

    def _plot_ic_periods(self, ax: plt.Axes) -> None:
        """绘制多预测周期的 IC 均值与 ICIR 对比图。"""
        if not self._ic_results:
            return
        periods = list(self._ic_results.keys())
        ic_means = [self._ic_results[p].ic_mean for p in periods]
        icirs = [self._ic_results[p].icir for p in periods]
        x = np.arange(len(periods))
        # IC 均值用左轴，ICIR 用右轴
        ax.bar(x - 0.2, ic_means, width=0.35, label="IC 均值", alpha=0.8,
               color="#2196F3", edgecolor="black", linewidth=0.5)
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, icirs, width=0.35, label="ICIR", alpha=0.8,
                color="#FF9800", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}日" for p in periods])
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_title("各预测周期 IC & ICIR", fontsize=10)
        ax.set_ylabel("IC 均值", color="#2196F3")
        ax2.set_ylabel("ICIR", color="#FF9800")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)

    def _plot_return_distribution(self, ax: plt.Axes) -> None:
        """绘制多头/空头/中性信号分组的收益分布直方图。"""
        if self._factor is None or self._returns is None:
            return
        fwd = self._returns.shift(-self.forward_period)
        df = pd.DataFrame({"f": self._factor, "r": fwd}).dropna()
        long_rets = df.loc[df["f"] > 1.0, "r"] * 100
        short_rets = df.loc[df["f"] < -1.0, "r"] * 100
        neutral_rets = df.loc[df["f"].between(-1.0, 1.0), "r"] * 100
        if len(neutral_rets) > 0:
            ax.hist(neutral_rets, bins=30, alpha=0.35, color="grey", label="中性")
        if len(long_rets) > 0:
            ax.hist(long_rets, bins=30, alpha=0.6, color="#4CAF50", label="多头")
        if len(short_rets) > 0:
            ax.hist(short_rets, bins=30, alpha=0.6, color="#F44336", label="空头")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title("按信号分组的收益分布", fontsize=10)
        ax.set_xlabel("未来收益率 (%)")
        ax.legend(fontsize=7)

    def _plot_scatter(self, ax: plt.Axes) -> None:
        """绘制因子值与未来收益率的散点图（含 OLS 拟合线）。"""
        if self._factor is None or self._returns is None:
            return
        fwd = self._returns.shift(-self.forward_period)
        df = pd.DataFrame({"f": self._factor, "r": fwd * 100}).dropna()
        # 数据量过大时随机采样，避免绘图过慢
        if len(df) > 1000:
            df = df.sample(1000, random_state=0)
        ax.scatter(df["f"], df["r"], alpha=0.15, s=6, color="#2196F3")
        # 绘制 OLS 拟合直线
        z = np.polyfit(df["f"], df["r"], 1)
        x_line = np.linspace(df["f"].min(), df["f"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "r-", linewidth=1.5, label="OLS 拟合")
        ax.axhline(0, color="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.3)
        ax.set_title(f"因子值 vs 未来收益率（{self.forward_period}日）", fontsize=10)
        ax.set_xlabel("因子值")
        ax.set_ylabel("未来收益率 (%)")
        ax.legend(fontsize=7)

    def _plot_score_bars(self, ax: plt.Axes) -> None:
        """绘制各维度评分和综合评分的水平条形图。"""
        fs = self.score()
        labels = ["IC", "信号", "回归", "稳健性"]
        values = [fs.ic_score, fs.signal_score, fs.regression_score, fs.robustness_score]
        colors = [
            "#4CAF50" if v >= 0.60 else "#FF9800" if v >= 0.40 else "#F44336"
            for v in values
        ]
        bars = ax.barh(labels, values, color=colors, alpha=0.8,
                       edgecolor="black", linewidth=0.5)
        ax.set_xlim(0, 1.05)
        ax.axvline(0.60, color="green", linewidth=1.0, linestyle="--", alpha=0.6)
        for bar, v in zip(bars, values):
            ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=9)
        ax.set_title(
            f"综合评分：{fs.composite_score:.2f}  （等级 {fs.grade}）",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("评分 [0–1]")
