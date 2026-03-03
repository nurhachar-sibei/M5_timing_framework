"""
完整示例：A股择时因子评价框架
==============================

本示例使用合成A股指数行情数据（特征近似沪深300指数），
演示择时因子评价框架的完整使用流程。

评估四个典型择时因子：

    1. MA动量因子 (MA Momentum)
       - (MA_20 - MA_60) / MA_60
       - 短期均线高于长期均线时为正值 → 上升趋势信号

    2. RSI反转因子 (RSI Contrarian)
       - 将RSI-14转化为反转因子
       - RSI低（超卖）→ 看多；RSI高（超买）→ 看空

    3. 波动率因子 (Inverse Volatility)
       - 20日滚动收益率标准差的负值
       - 高波动率 → 看空信号（风险规避）

    4. 估值因子 (PE Valuation)
       - 模拟市盈率均值回归因子
       - 低PE（便宜）→ 看多；高PE（贵）→ 看空

每个因子均通过 TimingFactorEvaluator 进行评估，包含：
    - 三种信号生成方法（阈值法、均线法、极值法）
    - IC/ICIR相关性分析（多个预测周期）
    - OLS回归检验
    - 样本内外稳健性检验
    - 市场状态分析（牛市/熊市/震荡市）

额外演示：
    - 多因子正交化（对称Löwdin方法）
    - MA因子参数敏感性分析
    - 多因子横向对比

输出
----
所有图表保存至 ``plots/`` 子目录，
文字报告打印至控制台。

运行方式
--------
    cd M5_timing_framework
    python example/example_timing.py
"""

from __future__ import annotations

import os
import sys
import warnings

# ── 路径配置：从父目录导入 timing_framework 包 ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，在脚本环境下安全运行

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文字体设置
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
import numpy as np
import pandas as pd

from timing_framework import (
    FactorPreprocessor,
    TimingFactorEvaluator,
)
from timing_framework.robustness import RobustnessTester


# ══════════════════════════════════════════════════════════════════════════════
# 1. 合成行情数据生成 #该部分内容在实际过程中并不需要
# ══════════════════════════════════════════════════════════════════════════════

def generate_market_data(n_days: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    生成具有真实市场特征的合成A股指数数据。

    收益率序列构造具有如下特征：
    - 微小正向漂移（日均约 0.03%，年化约 7.5%）
    - GARCH型波动率聚集效应
    - 机制切换动量（牛市/熊市/震荡市循环）
    - 轻微正自相关（短期动量，使MA因子有效）

    数据中刻意嵌入了使四个因子均能观察到预测信号的结构。

    参数
    ----
    n_days : int
        生成的交易日数量。
    seed : int
        随机种子，保证可复现性。

    返回
    ----
    pd.DataFrame
        列：price（价格）、return（收益率）、log_return（对数收益率）、regime（市场状态）
    """
    np.random.seed(seed)

    dates = pd.date_range(start="2015-01-01", periods=n_days, freq="B")

    # ── 机制切换漂移 ──────────────────────────────────────────────────────────
    # 每约180个交易日在牛市/熊市/震荡市之间切换
    regime_cycle = np.zeros(n_days, dtype=int)
    cycle_len = 180
    regimes = [1, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0]  # 1=牛市, 2=熊市, 0=震荡
    for i in range(n_days):
        regime_cycle[i] = regimes[(i // cycle_len) % len(regimes)]

    drift_map = {0: 0.0002, 1: 0.0010, 2: -0.0008}
    drifts = np.array([drift_map[r] for r in regime_cycle])

    # ── GARCH(1,1)型波动率 ────────────────────────────────────────────────────
    sigma_base = 0.012
    vol = np.zeros(n_days)
    vol[0] = sigma_base
    noise_raw = np.random.randn(n_days)

    for i in range(1, n_days):
        # 熊市阶段波动率放大
        base_var = sigma_base ** 2 * (1.5 if regime_cycle[i] == 2 else 1.0)
        vol[i] = np.sqrt(
            0.20 * base_var
            + 0.75 * vol[i - 1] ** 2
            + 0.05 * (vol[i - 1] * noise_raw[i - 1]) ** 2
        )

    # ── 收益率：漂移 + 动量 + 噪声 ────────────────────────────────────────────
    # 微小AR(1)项构造短期动量（使MA因子有效）
    ar_coef = 0.08
    returns = np.zeros(n_days)
    for i in range(1, n_days):
        returns[i] = drifts[i] + ar_coef * returns[i - 1] + vol[i] * noise_raw[i]

    # ── 价格序列 ──────────────────────────────────────────────────────────────
    prices = 3000.0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "price": prices,
            "return": returns,
            "log_return": np.log(1 + returns),
            "regime": regime_cycle,  # 0=震荡, 1=牛市, 2=熊市
        },
        index=dates,
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. 因子计算函数
# ══════════════════════════════════════════════════════════════════════════════

def calc_ma_momentum(prices: pd.Series,
                     short_window: int = 20,
                     long_window: int = 60) -> pd.Series:
    """
    MA动量因子（趋势动量因子）
    --------------------------
    衡量短期均价相对于长期均价的偏离程度。

    公式：(MA短期 − MA长期) / MA长期

    正值 → 短期动量向上 → 看多信号。
    负值 → 短期动量向下 → 看空信号。

    适用预测周期：1–10日。

    参数
    ----
    prices : pd.Series
        收盘价序列。
    short_window : int
        短期均线窗口（默认20日）。
    long_window : int
        长期均线窗口（默认60日）。
    """
    ma_short = prices.rolling(short_window, min_periods=short_window // 2).mean()
    ma_long = prices.rolling(long_window, min_periods=long_window // 2).mean()
    factor = (ma_short - ma_long) / ma_long
    return factor.rename("ma_momentum")


def calc_rsi_contrarian(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI反转因子（RSI Contrarian）
    ------------------------------
    标准RSI衡量动量强度，本因子将其**取反**构建反转择时因子：

        因子 = (50 − RSI) / 50

    解读：
        RSI < 30（超卖）→ 因子 > 0.40 → 看多信号
        RSI > 70（超买）→ 因子 < -0.40 → 看空信号
        RSI ≈ 50（中性）→ 因子 ≈ 0

    本因子在均值回归型市场中效果最佳。

    参数
    ----
    prices : pd.Series
        收盘价序列。
    window : int
        RSI回望窗口（标准为14日）。
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # 反转变换：低RSI（超卖）→ 因子为正值
    factor = (50.0 - rsi) / 50.0
    return factor.rename("rsi_contrarian")


def calc_inv_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    波动率因子（Inverse Volatility）
    ---------------------------------
    高市场波动率通常与不确定性增加相关，常伴随或预示下行压力。

    公式：因子 = −滚动标准差(收益率, window)

    取负号的作用：
        低波动率 → 正值因子 → 看多
        高波动率 → 负值因子 → 看空

    本因子捕捉了广为人知的"低波动率异象"。

    参数
    ----
    returns : pd.Series
        日收益率序列。
    window : int
        滚动回望窗口（默认20日）。
    """
    rolling_vol = returns.rolling(window, min_periods=window // 2).std()
    factor = -rolling_vol
    return factor.rename("inv_volatility")


def calc_pe_valuation(prices: pd.Series, seed: int = 99) -> pd.Series:
    """
    估值因子（PE Valuation）
    -------------------------
    模拟基于市盈率的择时因子，PE向合理估值（15倍）均值回归。

    公式：因子 = −(PE − PE合理) / PE合理

    解读：
        PE < 15倍（便宜）→ 因子 > 0 → 市场低估 → 看多
        PE ≈ 15倍（合理）→ 因子 ≈ 0 → 无明显信号
        PE > 15倍（贵）  → 因子 < 0 → 市场高估 → 看空

    实盘中应使用Wind或交易所等来源的指数真实市盈率数据。

    参数
    ----
    prices : pd.Series
        价格序列（用于将PE与市场水位挂钩）。
    seed : int
        随机种子，用于PE随机模拟。
    """
    np.random.seed(seed)
    n = len(prices)
    pe_fair = 15.0
    speed = 0.01   # 均值回归速度
    sigma = 0.015  # 日波动率

    # 归一化价格水位影响PE漂移（贵的市场PE偏高）
    norm_price = prices / prices.iloc[60:61].values[0]  # 以第60日为基准归一化
    price_effect = (norm_price - 1.0) * 3.0

    pe = np.zeros(n)
    pe[0] = pe_fair
    noise = np.random.randn(n) * sigma

    for i in range(1, n):
        pe[i] = (
            pe[i - 1]
            + speed * (pe_fair - pe[i - 1])
            + price_effect.iloc[i] * 0.2
            + noise[i]
        )
        pe[i] = max(5.0, pe[i])  # PE不允许为负或极小值

    pe_series = pd.Series(pe, index=prices.index, name="pe")

    # 反转因子：便宜看多
    factor = -(pe_series - pe_fair) / pe_fair
    return factor.rename("pe_valuation")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 辅助函数：多因子对比可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_factor_comparison(
    evaluators: dict,
    scores: dict,
    save_path: Path,
) -> None:
    """生成2×2对比图，横向比较所有因子的关键指标。"""
    names = list(evaluators.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    periods = [1, 5, 10, 20]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("择时因子横向对比", fontsize=14, fontweight="bold")

    # ── 子图A：各预测周期IC均值 ──
    ax = axes[0, 0]
    for i, (name, ev) in enumerate(evaluators.items()):
        ic_means = [
            ev._ic_results[p].ic_mean if (ev._ic_results and p in ev._ic_results) else 0
            for p in periods
        ]
        ax.plot(periods, ic_means, marker="o", label=name,
                color=colors[i], linewidth=1.8, markersize=5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("各预测周期IC均值", fontsize=11)
    ax.set_xlabel("预测周期（天）")
    ax.set_ylabel("IC均值")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 子图B：1日预测ICIR ──
    ax = axes[0, 1]
    icirs = []
    for name in names:
        ev = evaluators[name]
        icir = (
            ev._ic_results[1].icir
            if (ev._ic_results and 1 in ev._ic_results)
            else 0.0
        )
        icirs.append(icir if not np.isnan(icir) else 0.0)
    bars = ax.bar(range(len(names)), icirs, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axhline(0.5, color="green", linewidth=1.2, linestyle="--", alpha=0.7,
               label="ICIR=0.5（良好）")
    ax.axhline(-0.5, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_title("ICIR（1日预测）", fontsize=11)
    ax.set_ylabel("ICIR")
    ax.legend(fontsize=8)

    # ── 子图C：各方法胜率对比 ──
    ax = axes[1, 0]
    methods = ["threshold", "moving_average", "percentile"]
    method_labels = ["阈值法", "均线法", "极值法"]
    x = np.arange(len(names))
    width = 0.26
    for j, (method, label) in enumerate(zip(methods, method_labels)):
        wrs = []
        for name in names:
            ev = evaluators[name]
            r = ev._signal_results.get(method) if ev._signal_results else None
            wrs.append(r.overall_win_rate * 100 if r else 50.0)
        ax.bar(x + j * width, wrs, width, label=label, alpha=0.8)
    ax.axhline(50, color="red", linewidth=1.2, linestyle="--", alpha=0.7, label="50%基准")
    ax.set_xticks(x + width)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_title("各信号方法胜率", fontsize=11)
    ax.set_ylabel("胜率（%）")
    ax.set_ylim(38, 72)
    ax.legend(fontsize=8)

    # ── 子图D：综合评分 ──
    ax = axes[1, 1]
    composite = [scores[n].composite_score for n in names]
    bars = ax.bar(range(len(names)), composite, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0.60, color="green", linewidth=1.2, linestyle="--", label="B级（0.60）")
    ax.axhline(0.75, color="blue", linewidth=1.2, linestyle="--", label="A级（0.75）")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_title("综合评分与评级", fontsize=11)
    ax.set_ylabel("评分")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    for i, (bar, sc, name) in enumerate(zip(bars, composite, names)):
        grade = scores[name].grade
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{grade}级",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ 对比图已保存：{save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 参数敏感性演示
# ══════════════════════════════════════════════════════════════════════════════

def demo_parameter_sensitivity(prices: pd.Series,
                                returns: pd.Series,
                                save_path: Path) -> None:
    """
    演示MA动量因子的参数敏感性分析。

    扫描短期窗口与长期窗口的组合，观察ICIR的变化情况。
    稳健因子在合理参数范围内应保持一致的ICIR水平。
    """
    print("\n  正在扫描MA动量因子参数敏感性...")

    tester = RobustnessTester(forward_period=1)

    # 定义因子构造函数
    def factor_func(short_window: int, long_window: int) -> pd.Series:
        f = calc_ma_momentum(prices, short_window, long_window)
        return FactorPreprocessor.preprocess(f, rolling_window=252)

    # 参数扫描网格
    param_grid = {
        "short_window": [10, 20, 30, 40],
        "long_window": [40, 60, 90, 120],
    }

    sensitivity_df = tester.parameter_sensitivity_test(
        factor_func=factor_func,
        param_grid=param_grid,
        returns=returns,
    )

    # 过滤无效组合（短窗口须小于长窗口）
    sensitivity_df = sensitivity_df[
        sensitivity_df["short_window"] < sensitivity_df["long_window"]
    ].copy()

    print("\n  MA动量因子参数敏感性（ICIR值）：")
    pivot = sensitivity_df.pivot(
        index="short_window", columns="long_window", values="icir"
    ).round(3)
    print(pivot.to_string())

    # 热力图可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MA动量因子：参数敏感性分析", fontsize=12)

    for ax, metric, cmap, title in [
        (axes[0], "icir", "RdYlGn", "ICIR"),
        (axes[1], "win_rate", "RdYlGn", "胜率"),
    ]:
        pivot = sensitivity_df.pivot(
            index="short_window", columns="long_window", values=metric
        )
        valid = pivot.values[~np.isnan(pivot.values)]
        vmin, vmax = (valid.min(), valid.max()) if len(valid) else (0, 1)
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"长={v}" for v in pivot.columns])
        ax.set_yticklabels([f"短={v}" for v in pivot.index])
        ax.set_xlabel("长期窗口")
        ax.set_ylabel("短期窗口")
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ 参数敏感性图已保存：{save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 多因子正交化演示
# ══════════════════════════════════════════════════════════════════════════════

def demo_orthogonalization(factors_raw: dict,
                            returns: pd.Series,
                            save_path: Path) -> None:
    """
    演示对称正交化方法，并对比正交化前后因子间的相关性。
    """
    print("\n  正在执行多因子正交化演示...")

    # 将所有因子预处理后合并为DataFrame
    factor_dict_proc = {}
    for name, f in factors_raw.items():
        proc = FactorPreprocessor.preprocess(f, rolling_window=252)
        factor_dict_proc[name] = proc

    factor_df = pd.DataFrame(factor_dict_proc).dropna()

    corr_before = factor_df.corr()

    try:
        factor_df_orth = FactorPreprocessor.symmetric_orthogonalize(factor_df)
        corr_after = factor_df_orth.corr()
        orth_ok = True
    except Exception as e:
        print(f"    正交化跳过：{e}")
        corr_after = corr_before
        orth_ok = False

    print("\n  正交化前因子相关性：")
    print(corr_before.round(3).to_string())
    if orth_ok:
        print("\n  对称正交化后因子相关性：")
        print(corr_after.round(3).to_string())

    # 热力图对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "因子相关性：对称正交化前后对比",
        fontsize=11
    )
    for ax, corr, title in [
        (axes[0], corr_before, "正交化前"),
        (axes[1], corr_after, "对称正交化后"),
    ]:
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.75)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        labels = [c.replace(" ", "\n") for c in corr.columns]
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if abs(corr.iloc[i, j]) < 0.6 else "white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ 正交化图已保存：{save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("  择时因子评价框架 – 完整示例")
    print("  合成A股市场数据仿真")
    print("=" * 68)

    # ── 输出目录配置 ────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 步骤1：生成合成行情数据 ─────────────────────────────────────────────
    print("\n[步骤1] 生成合成行情数据...")
    data = generate_market_data(n_days=2000, seed=42)
    prices = data["price"]
    returns = data["return"]

    print(f"  日期范围  ：{data.index[0].date()} → {data.index[-1].date()}")
    print(f"  交易日数量：{len(data)}")
    print(f"  价格区间  ：{prices.min():.0f} – {prices.max():.0f}")
    print(f"  年化收益率：{returns.mean() * 252 * 100:.1f}%（均值）")
    print(f"  年化波动率：{returns.std() * (252 ** 0.5) * 100:.1f}%")

    # ── 步骤2：计算原始因子 ─────────────────────────────────────────────────
    print("\n[步骤2] 计算择时因子...")
    factors_raw = {
        "MA动量":  calc_ma_momentum(prices, 20, 60),
        "RSI反转": calc_rsi_contrarian(prices, 14),
        "波动率":  calc_inv_volatility(returns, 20),
        "估值":    calc_pe_valuation(prices, seed=99),
    }
    for name, f in factors_raw.items():
        valid = f.dropna()
        print(f"  {name:<10}：{len(valid)} 条有效观测，"
              f"范围 [{valid.min():.3f}, {valid.max():.3f}]")

    # ── 步骤3：逐一评估因子 ─────────────────────────────────────────────────
    print("\n[步骤3] 综合评估（可能需要一点时间）...")
    evaluators: dict[str, TimingFactorEvaluator] = {}
    scores: dict = {}

    for factor_name, raw_factor in factors_raw.items():
        print(f"\n  ─── {factor_name} {'─' * (50 - len(factor_name))}")

        ev = TimingFactorEvaluator(
            factor_name=factor_name,
            forward_period=1,    # 预测1日后收益率
            ic_method="pearson",
        )

        ev.evaluate(
            factor=raw_factor,
            returns=returns,
            prices=prices,
            preprocess=True,               # 进行MAD去极值 + 滚动Z-score标准化
            rolling_window=252,            # 1年滚动窗口用于预处理
            run_robustness=True,           # 执行样本内外检验和市场状态分析
            run_rolling_regression=False,  # 跳过耗时的滚动回归
            ic_periods=[1, 5, 10, 20],
        )

        # 打印完整评估报告
        ev.report()

        # 生成单因子评估图
        fig = ev.plot(figsize=(16, 13))
        plot_path = (
            output_dir / f"{factor_name.lower().replace(' ', '_')}_evaluation.png"
        )
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ 单因子图已保存：{plot_path.name}")

        evaluators[factor_name] = ev
        scores[factor_name] = ev.score()

    # ── 步骤4：多因子横向对比 ──────────────────────────────────────────────
    print("\n[步骤4] 多因子横向对比...")
    print(
        f"\n  {'因子':<12} {'IC(1日)':>8} {'ICIR(1日)':>10} "
        f"{'胜率':>8} {'综合评分':>8} {'评级':>6}"
    )
    print("  " + "─" * 60)
    for name, sc in scores.items():
        ev = evaluators[name]
        ic1 = ev._ic_results.get(1)
        sig1 = ev._signal_results.get("threshold")
        ic_val = ic1.ic_mean if ic1 else 0.0
        icir_val = ic1.icir if ic1 and not np.isnan(ic1.icir) else 0.0
        wr_val = sig1.overall_win_rate if sig1 else 0.5
        print(
            f"  {name:<12} {ic_val:>+8.4f} {icir_val:>+10.4f} "
            f"{wr_val:>8.2%} {sc.composite_score:>8.2f} {sc.grade:>6}"
        )

    plot_factor_comparison(
        evaluators, scores,
        save_path=output_dir / "factor_comparison.png",
    )

    # ── 步骤5：市场状态分析 ────────────────────────────────────────────────
    print("\n[步骤5] 市场状态分解分析...")
    print(f"\n  {'因子':<12} {'牛市IC':>9} {'熊市IC':>9} {'震荡IC':>9}")
    print("  " + "─" * 45)
    for name, ev in evaluators.items():
        regime_res = ev._regime_results or {}
        bull_ic = regime_res.get("bull")
        bear_ic = regime_res.get("bear")
        side_ic = regime_res.get("sideways")
        print(
            f"  {name:<12} "
            f"{(bull_ic.ic_mean if bull_ic else float('nan')):>+9.4f} "
            f"{(bear_ic.ic_mean if bear_ic else float('nan')):>+9.4f} "
            f"{(side_ic.ic_mean if side_ic else float('nan')):>+9.4f}"
        )

    # ── 步骤6：参数敏感性分析（MA动量因子）──────────────────────────────────
    print("\n[步骤6] 参数敏感性分析（MA动量因子）...")
    demo_parameter_sensitivity(
        prices=prices,
        returns=returns,
        save_path=output_dir / "ma_parameter_sensitivity.png",
    )

    # ── 步骤7：多因子正交化 ────────────────────────────────────────────────
    print("\n[步骤7] 多因子正交化演示...")
    demo_orthogonalization(
        factors_raw=factors_raw,
        returns=returns,
        save_path=output_dir / "factor_orthogonalization.png",
    )

    # ── 汇总输出 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  全部完成！输出文件：")
    for f in sorted(output_dir.iterdir()):
        print(f"    {f.name}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
