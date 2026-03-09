"""
回测模块
========
对择时因子信号进行事件驱动回测，生成策略表现评估与可视化图表。

策略逻辑
--------
- 信号 > long_threshold  → 全仓持有资产（仓位=1.0）
- 信号 < -short_threshold → 空仓（仓位=0.0，持有货币）
- 其他区间               → 维持上一期仓位（默认初始为空仓）

基准组合
--------
基准 = benchmark_asset_ratio × 资产日收益 + (1 - benchmark_asset_ratio) × 日无风险利率

三张图表
--------
1. 买卖点图：价格曲线 + 买入/卖出标记 + 信号值子图
2. 策略vs基准对比图：累计净值 + 超额收益 + 回撤曲线
3. 绩效归因图：月度热力图 + 年度收益 + 滚动Sharpe
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 全局字体设置（确保中文正常显示）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
                                    "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 数据容器
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """回测结果数据容器，存储所有收益序列与绩效指标。"""

    signal_name: str

    # 收益率序列
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    buyhold_returns: pd.Series

    # 仓位序列
    positions: pd.Series

    # 绩效指标字典
    metrics_strategy: Dict
    metrics_benchmark: Dict
    metrics_buyhold: Dict

    # 超额收益序列（策略 - 基准）
    excess_returns: pd.Series

    def summary(self) -> str:
        """生成回测结果对比摘要字符串。"""
        m  = self.metrics_strategy
        bm = self.metrics_benchmark
        lines = [
            f"\n{'=' * 60}",
            f"  回测结果汇总：{self.signal_name}",
            f"{'─' * 60}",
            f"  {'指标':<22} {'择时策略':>12} {'基准':>12}",
            f"  {'─' * 50}",
            f"  {'总收益率':<22} {m['total_return']:>+12.2%} {bm['total_return']:>+12.2%}",
            f"  {'年化收益率':<22} {m['annual_return']:>+12.2%} {bm['annual_return']:>+12.2%}",
            f"  {'年化波动率':<22} {m['annual_vol']:>12.2%} {bm['annual_vol']:>12.2%}",
            f"  {'Sharpe比率':<22} {m['sharpe']:>12.4f} {bm['sharpe']:>12.4f}",
            f"  {'最大回撤':<22} {m['max_drawdown']:>12.2%} {bm['max_drawdown']:>12.2%}",
            f"  {'Calmar比率':<22} {m['calmar']:>12.4f} {bm['calmar']:>12.4f}",
            f"  {'日胜率':<22} {m['win_rate']:>12.2%} {bm['win_rate']:>12.2%}",
            f"  {'交易次数':<22} {m.get('n_trades', 0):>12d}",
            f"{'=' * 60}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 回测器主类
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    简单的时序择时策略回测器。

    将连续的因子信号转化为仓位，计算策略日收益率，
    并与基准（资产 + 货币）进行全面对比。

    参数
    ----
    config : dict
        来自 config.yaml 的 backtest 配置段。
    """

    _ANNUAL = 252  # 年化因子（交易日）

    def __init__(self, config: dict) -> None:
        self.cfg              = config
        self.long_threshold   = float(config.get("long_threshold",  0.0)) #self.long_threshold: float
        self.short_threshold  = float(config.get("short_threshold", 0.0)) #self.short_threshold: float
        self.risk_free_rate   = float(config.get("risk_free_rate",  0.03)) #self.risk_free_rate: float
        self.transaction_cost = float(config.get("transaction_cost", 0.001)) #self.transaction_cost: float
        self.benchmark_ratio  = float(config.get("benchmark_asset_ratio", 0.5)) #self.benchmark_ratio: float

    # ─────────────────────────────────────────────────────────────────────────
    # 仓位生成
    # ─────────────────────────────────────────────────────────────────────────

    def generate_positions(self, signal: pd.Series) -> pd.Series:
        """
        将连续信号值转化为离散仓位（0 或 1）。

        规则：
        - signal > long_threshold  → 仓位 = 1.0（全仓做多）
        - signal < short_threshold → 仓位 = 0.0（空仓）
        - 其他情况                → 前向填充，保持上一期仓位

        返回的仓位序列延迟一个交易日（t 日信号用于 t+1 日持仓），
        以避免当日信号当日交易的未来信息泄漏。
        """
        pos = pd.Series(np.nan, index=signal.index, name="position")
        pos[signal > self.long_threshold]  = 1.0
        pos[signal <= self.short_threshold] = 0.0
        # 前向填充中性区间，初始默认空仓
        pos = pos.ffill().fillna(0.0)
        # 延迟一期（t 日信号 → t+1 日仓位）
        pos = pos.shift(1).fillna(0.0)
        return pos

    # ─────────────────────────────────────────────────────────────────────────
    # 收益率计算
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_strategy_returns(self,
                                    positions: pd.Series,
                                    price_returns: pd.Series) -> pd.Series:
        """
        计算含单边手续费的策略日收益率。

        手续费 = 换仓幅度 × transaction_cost（每次换仓时扣除）。
        """
        pos = positions.reindex(price_returns.index).fillna(0.0)
        gross    = pos * price_returns
        turnover = pos.diff().abs().fillna(0.0)
        cost     = turnover * self.transaction_cost
        return (gross - cost).rename("strategy_return")

    def calculate_benchmark_returns(self, price_returns: pd.Series) -> pd.Series:
        """
        基准收益率 = 资产占比 × 资产日收益 + 货币占比 × 日无风险利率。
        """
        daily_rf = self.risk_free_rate / self._ANNUAL
        bm = self.benchmark_ratio * price_returns + (1 - self.benchmark_ratio) * daily_rf
        return bm.rename("benchmark_return")

    # ─────────────────────────────────────────────────────────────────────────
    # 绩效指标计算
    # ─────────────────────────────────────────────────────────────────────────

    def performance_metrics(self,
                             returns: pd.Series,
                             name: str = "策略",
                             n_trades: int = 0) -> Dict:
        """
        计算常用的绩效指标字典。

        包含：总收益率、年化收益率、年化波动率、Sharpe比率、最大回撤、
        Calmar比率、日胜率、偏度、峰度、单日最佳/最差收益。
        """
        daily_rf = self.risk_free_rate / self._ANNUAL
        cum       = (1 + returns).cumprod()
        n         = len(returns)

        total_return  = cum.iloc[-1] - 1
        annual_return = (1 + total_return) ** (self._ANNUAL / n) - 1
        annual_vol    = returns.std() * (self._ANNUAL ** 0.5)

        excess = returns - daily_rf
        sharpe = (
            excess.mean() / excess.std() * (self._ANNUAL ** 0.5)
            if excess.std() > 1e-10 else 0.0
        )

        roll_max     = cum.cummax()
        drawdown     = (cum - roll_max) / roll_max
        max_drawdown = drawdown.min()

        calmar = (
            -annual_return / max_drawdown
            if max_drawdown < -1e-10 else 0.0
        )

        return {
            "name":          name,
            "total_return":  total_return,
            "annual_return": annual_return,
            "annual_vol":    annual_vol,
            "sharpe":        sharpe,
            "max_drawdown":  max_drawdown,
            "calmar":        calmar,
            "win_rate":      float((returns > 0).mean()),
            "skewness":      float(returns.skew()),
            "kurtosis":      float(returns.kurtosis()),
            "best_day":      float(returns.max()),
            "worst_day":     float(returns.min()),
            "n_days":        n,
            "n_trades":      n_trades,
        }

    def _count_trades(self, positions: pd.Series) -> int:
        """统计换仓次数（仓位从 0→1 或 1→0 的次数）。"""
        return int((positions.diff().fillna(0) != 0).sum())

    def _compute_extended_metrics(self,
                                   positions: pd.Series,
                                   strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict:
        """
        计算四项扩展绩效指标。

        返回字段
        --------
        trade_win_rate  : 交易胜率（完整开平仓交易中收益率>0的比例）
        holding_ratio   : 持有时间占比（持仓天数/总天数）
        odds_ratio      : 赔率（均盈利 / 均亏损绝对值）
        excess_win_rate : 超额胜率（策略日收益 > 基准日收益的比例）
        """
        # ── 持有时间占比 ──────────────────────────────────────────────────
        holding_ratio = float((positions > 0).mean())

        # ── 赔率 ──────────────────────────────────────────────────────────
        pos_rets = strategy_returns[strategy_returns > 0]
        neg_rets = strategy_returns[strategy_returns < 0]
        if len(pos_rets) > 0 and len(neg_rets) > 0 and abs(neg_rets.mean()) > 1e-10:
            odds_ratio = float(pos_rets.mean() / abs(neg_rets.mean()))
        else:
            odds_ratio = float("nan")

        # ── 超额胜率 ──────────────────────────────────────────────────────
        bench_aligned = benchmark_returns.reindex(strategy_returns.index).fillna(0.0)
        excess_win_rate = float((strategy_returns > bench_aligned).mean())

        # ── 交易胜率：识别完整开平仓交易区间 ─────────────────────────────
        pos_diff = positions.diff().fillna(0)
        entries  = pos_diff[pos_diff > 0].index.tolist()
        exits    = pos_diff[pos_diff < 0].index.tolist()

        trade_returns = []
        for entry in entries:
            # 找该 entry 之后的第一个 exit
            future_exits = [e for e in exits if e > entry]
            if future_exits:
                exit_date = future_exits[0]
                seg = strategy_returns.loc[entry:exit_date]
                trade_returns.append(float((1 + seg).prod() - 1))
            # 最后一笔未平仓的交易不纳入统计

        if trade_returns:
            trade_win_rate = float(np.mean([r > 0 for r in trade_returns]))
        else:
            trade_win_rate = float("nan")

        return {
            "trade_win_rate":  trade_win_rate,
            "holding_ratio":   holding_ratio,
            "odds_ratio":      odds_ratio,
            "excess_win_rate": excess_win_rate,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 图1：买卖点分析图
    # ─────────────────────────────────────────────────────────────────────────

    def plot_buy_sell_signals(self,
                               signal: pd.Series,
                               prices: pd.Series,
                               positions: pd.Series,
                               signal_name: str,
                               save_path: Path) -> None:
        """
        绘制价格曲线 + 买入/卖出标记 + 信号值子图。

        上图（3份高）：收盘价曲线、绿色▲买入点、红色▼卖出点、持仓背景底色。
        下图（1份高）：信号值曲线、做多/做空阈值虚线。
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.suptitle(f"{signal_name}  |  买卖点分析", fontsize=13, fontweight="bold")

        # ── 上图：价格与买卖点 ──────────────────────────────────────────────
        ax1.plot(prices.index, prices.values, color="#1565C0",
                 linewidth=1.2, label="收盘价", zorder=1)
        ax1.set_ylabel("价格", fontsize=10)
        ax1.grid(True, alpha=0.3)

        pos_change  = positions.diff().fillna(0)
        buy_dates   = pos_change[pos_change > 0].index
        sell_dates  = pos_change[pos_change < 0].index

        if len(buy_dates):
            bp = prices.reindex(buy_dates, method="nearest")
            ax1.scatter(buy_dates, bp, marker="^", color="#43A047",
                        s=90, zorder=5, label=f"买入（{len(buy_dates)}次）")
        if len(sell_dates):
            sp = prices.reindex(sell_dates, method="nearest")
            ax1.scatter(sell_dates, sp, marker="v", color="#E53935",
                        s=90, zorder=5, label=f"卖出（{len(sell_dates)}次）")

        # 持仓期间背景着色
        in_long = positions.reindex(prices.index, method="ffill").fillna(0) > 0
        p_min, p_max = prices.min() * 0.95, prices.max() * 1.05
        ax1.fill_between(prices.index, p_min, p_max,
                         where=in_long, alpha=0.07, color="#43A047", label="持仓期间")
        ax1.set_ylim(p_min, p_max)
        ax1.legend(fontsize=9, loc="upper left")

        # ── 下图：信号值 ────────────────────────────────────────────────────
        ax2.plot(signal.index, signal.values, color="#7B1FA2",
                 linewidth=1.0, label="信号值")
        ax2.axhline(self.long_threshold, color="#43A047", linewidth=1.2,
                    linestyle="--", alpha=0.9,
                    label=f"做多阈值 ({self.long_threshold:+.2f})")
        ax2.axhline(-self.short_threshold, color="#E53935", linewidth=1.2,
                    linestyle="--", alpha=0.9,
                    label=f"做空阈值 ({-self.short_threshold:+.2f})")
        ax2.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax2.fill_between(signal.index, 0, signal.values,
                         where=signal > self.long_threshold,
                         alpha=0.20, color="#43A047")
        ax2.fill_between(signal.index, 0, signal.values,
                         where=signal < -self.short_threshold,
                         alpha=0.20, color="#E53935")
        ax2.set_ylabel("信号值", fontsize=10)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # 图2：策略 vs 基准对比图
    # ─────────────────────────────────────────────────────────────────────────

    def plot_strategy_vs_benchmark(self,
                                    strategy_returns: pd.Series,
                                    benchmark_returns: pd.Series,
                                    buyhold_returns: pd.Series,
                                    strategy_metrics: dict,
                                    benchmark_metrics: dict,
                                    signal_name: str,
                                    save_path: Path) -> None:
        """
        绘制三子图对比图：累计净值 / 超额收益 / 最大回撤。
        """
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 10),
            gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
            sharex=True,
        )
        fig.suptitle(f"{signal_name}  |  择时策略 vs 基准对比", fontsize=13, fontweight="bold")

        cum_s  = (1 + strategy_returns).cumprod()
        cum_b  = (1 + benchmark_returns).cumprod()
        cum_bh = (1 + buyhold_returns).cumprod()

        # ── 上图：累计净值 ──────────────────────────────────────────────────
        ax = axes[0]
        sm, bm_ = strategy_metrics, benchmark_metrics
        ax.plot(cum_s.index,  cum_s.values,  color="#1565C0", linewidth=1.8,
                label=(f"择时策略  "
                       f"年化={sm['annual_return']:+.1%}  "
                       f"SR={sm['sharpe']:.2f}  "
                       f"MDD={sm['max_drawdown']:.1%}"))
        ax.plot(cum_b.index,  cum_b.values,  color="#EF6C00", linewidth=1.4,
                linestyle="--",
                label=(f"基准 ({self.benchmark_ratio:.0%}资产+{1-self.benchmark_ratio:.0%}货币)  "
                       f"年化={bm_['annual_return']:+.1%}  "
                       f"SR={bm_['sharpe']:.2f}"))
        ax.plot(cum_bh.index, cum_bh.values, color="#9E9E9E", linewidth=1.0,
                linestyle=":", label="纯持有资产（基准上限）")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_ylabel("累计净值", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        # 标注最终净值
        ax.annotate(f"终值 {cum_s.iloc[-1]:.3f}",
                    xy=(cum_s.index[-1], cum_s.iloc[-1]),
                    xytext=(-40, 8), textcoords="offset points",
                    fontsize=8, color="#1565C0",
                    arrowprops=dict(arrowstyle="-", color="#1565C0", lw=0.8))

        # ── 中图：相对基准超额收益 ──────────────────────────────────────────
        ax = axes[1]
        excess_cum = cum_s / cum_b - 1
        ax.fill_between(excess_cum.index, 0, excess_cum.values,
                        where=excess_cum >= 0, alpha=0.45, color="#43A047", label="超额为正")
        ax.fill_between(excess_cum.index, 0, excess_cum.values,
                        where=excess_cum < 0, alpha=0.45, color="#E53935", label="超额为负")
        ax.plot(excess_cum.index, excess_cum.values, color="#1565C0", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_ylabel("超额收益（vs 基准）", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # ── 下图：回撤曲线 ──────────────────────────────────────────────────
        ax = axes[2]
        def _drawdown(cum_series: pd.Series) -> pd.Series:
            rm = cum_series.cummax()
            return (cum_series - rm) / rm

        ax.fill_between(_drawdown(cum_s).index,  _drawdown(cum_s).values,  0,
                        alpha=0.6, color="#1565C0", label="策略回撤")
        ax.fill_between(_drawdown(cum_b).index,  _drawdown(cum_b).values,  0,
                        alpha=0.35, color="#EF6C00", label="基准回撤")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_ylabel("回撤幅度", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # 图3：绩效归因分析图
    # ─────────────────────────────────────────────────────────────────────────

    def plot_performance_attribution(self,
                                      strategy_returns: pd.Series,
                                      signal_name: str,
                                      save_path: Path) -> None:
        """
        绘制三子图绩效归因图：月度热力图 / 年度收益条形图 / 滚动Sharpe。
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"{signal_name}  |  绩效归因分析", fontsize=13, fontweight="bold")

        # ── 月度收益热力图 ────────────────────────────────────────────────
        ax = axes[0]
        monthly = strategy_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        m_df  = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret":  monthly.values,
        })
        pivot = m_df.pivot(index="year", columns="month", values="ret")
        month_labels = ["1月","2月","3月","4月","5月","6月",
                        "7月","8月","9月","10月","11月","12月"]
        pivot.columns = [month_labels[c - 1] for c in pivot.columns]

        valid_vals = pivot.values[~np.isnan(pivot.values)]
        vmax = max(abs(valid_vals).max(), 0.01) if len(valid_vals) else 0.05
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85,
                     format=mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title("月度收益热力图", fontsize=10)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(v) > vmax * 0.55 else "black")

        # ── 年度收益条形图 ────────────────────────────────────────────────
        ax = axes[1]
        annual = strategy_returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )
        colors_bar = ["#43A047" if v >= 0 else "#E53935" for v in annual.values]
        bars = ax.bar(annual.index.year, annual.values * 100,
                      color=colors_bar, alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("年份", fontsize=10)
        ax.set_ylabel("年度收益率（%）", fontsize=10)
        ax.set_title("年度收益分布", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, annual.values):
            offset = 0.4 if v >= 0 else -1.2
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{v:.1%}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)

        # ── 滚动Sharpe比率（63日）─────────────────────────────────────────
        ax = axes[2]
        daily_rf  = self.risk_free_rate / self._ANNUAL
        excess_r  = strategy_returns - daily_rf
        win       = 60  # 约3个月
        roll_sharpe = (
            excess_r.rolling(win).mean()
            / excess_r.rolling(win).std()
            * (self._ANNUAL ** 0.5)
        )
        ax.plot(roll_sharpe.index, roll_sharpe.values, color="#6A1B9A", linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axhline(1, color="#43A047", linewidth=1.0, linestyle="--", alpha=0.8,
                   label="Sharpe=1")
        ax.axhline(-1, color="#E53935", linewidth=1.0, linestyle="--", alpha=0.8,
                   label="Sharpe=-1")
        ax.fill_between(roll_sharpe.index, 0, roll_sharpe.values,
                        where=roll_sharpe >= 0, alpha=0.20, color="#43A047")
        ax.fill_between(roll_sharpe.index, 0, roll_sharpe.values,
                        where=roll_sharpe < 0, alpha=0.20, color="#E53935")
        ax.set_ylabel("滚动Sharpe（63日）", fontsize=10)
        ax.set_title("滚动Sharpe比率", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────────────────────

    def run(self,
            signal: pd.Series,
            prices: pd.Series,
            signal_name: str,
            save_dir: Path) -> BacktestResult:
        """
        执行完整回测流程并保存三张图表。

        参数
        ----
        signal      : 预处理后的因子信号值（与 prices 对齐）
        prices      : 资产收盘价序列
        signal_name : 信号名称（用于图表标题和文件名）
        save_dir    : 图表保存目录（不存在则自动创建）

        返回
        ----
        BacktestResult : 包含所有收益序列和绩效指标的结果对象
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 数据对齐
        price_ret  = prices.pct_change().dropna()
        common_idx = signal.index.intersection(price_ret.index)
        sig  = signal.reindex(common_idx).fillna(0.0)
        rets = price_ret.reindex(common_idx)
        prc  = prices.reindex(common_idx)

        # 生成仓位
        pos = self.generate_positions(sig)

        # 计算三条收益率序列
        strat_ret = self.calculate_strategy_returns(pos, rets)
        bench_ret = self.calculate_benchmark_returns(rets)
        bh_ret    = rets.rename("buyhold_return")

        # 绩效指标
        n_trades  = self._count_trades(pos)
        m_strat   = self.performance_metrics(strat_ret, "择时策略", n_trades)
        m_bench   = self.performance_metrics(bench_ret, "基准")
        m_bh      = self.performance_metrics(bh_ret,   "纯持有")
        # 扩展指标（交易胜率、持有时间占比、赔率、超额胜率）
        m_strat.update(self._compute_extended_metrics(pos, strat_ret, bench_ret))
        # 文件安全名（替换特殊字符）
        safe = signal_name.replace("/", "_").replace(" ", "_").replace("\\", "_")

        # 图1：买卖点图
        self.plot_buy_sell_signals(
            signal=sig, prices=prc, positions=pos,
            signal_name=signal_name,
            save_path=save_dir / f"{safe}_01_buy_sell.png",
        )
        
        # 图2：策略 vs 基准对比图
        self.plot_strategy_vs_benchmark(
            strategy_returns=strat_ret,
            benchmark_returns=bench_ret,
            buyhold_returns=bh_ret,
            strategy_metrics=m_strat,
            benchmark_metrics=m_bench,
            signal_name=signal_name,
            save_path=save_dir / f"{safe}_02_vs_benchmark.png",
        )

        # 图3：绩效归因图
        self.plot_performance_attribution(
            strategy_returns=strat_ret,
            signal_name=signal_name,
            save_path=save_dir / f"{safe}_03_attribution.png",
        )

        return BacktestResult(
            signal_name=signal_name,
            strategy_returns=strat_ret,
            benchmark_returns=bench_ret,
            buyhold_returns=bh_ret,
            positions=pos,
            metrics_strategy=m_strat,
            metrics_benchmark=m_bench,
            metrics_buyhold=m_bh,
            excess_returns=(strat_ret - bench_ret),
        )
