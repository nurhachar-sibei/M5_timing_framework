"""
稳健性评估模块
==============

评估择时因子的预测能力是否真实可靠，还是仅仅是样本内过拟合的产物。

本模块提供三类稳健性检验：

1. 样本内外检验（样本内外检验）
   -----------------------------------------------
   将数据分为训练期和测试期，分别计算 IC 和胜率。
   好因子在样本外应保持与样本内相近的表现，说明其具有真实的泛化能力。

   稳健性判断标准：
       - 样本外 IC 与样本内 IC 符号相同（方向一致）
       - IC 衰减幅度 < 50%（即样本外 IC ≥ 样本内 IC 的 50%）

2. 参数敏感性检验（参数敏感性检验）
   -----------------------------------------------
   在一组合理的参数范围内扫描，考察因子表现对参数取值的依赖程度。
   稳健的因子在参数小幅变动时表现应基本稳定，而非在特定参数下才有效。

3. 市场状态检验（市场状态检验）
   -----------------------------------------------
   将历史数据分为牛市、熊市和震荡市，分别评估因子有效性。
   有助于理解因子的适用范围和失效条件。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .correlation_testing import CorrelationTester, ICTestResult
from .signal_testing import SignalTester, SignalTestResult


# ══════════════════════════════════════════════════════════════════════════════
# 结果容器
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class InSampleOutSampleResult:
    """
    样本内外对比检验结果。

    属性
    ----
    split_date : pd.Timestamp
        训练期与测试期的分割日期。
    insample_ic : float
        训练期（样本内）IC 均值。
    outsample_ic : float
        测试期（样本外）IC 均值。
    insample_icir : float
        训练期 ICIR。
    outsample_icir : float
        测试期 ICIR。
    insample_win_rate : float
        训练期阈值信号胜率。
    outsample_win_rate : float
        测试期阈值信号胜率。
    ic_degradation : float
        IC 衰减比率 = (IC_in - IC_out) / |IC_in|。
        负值表示样本外优于样本内。
    is_robust : bool
        True 表示因子符合稳健性标准（同向且衰减 < 50%）。
    """

    split_date: pd.Timestamp
    insample_ic: float
    outsample_ic: float
    insample_icir: float
    outsample_icir: float
    insample_win_rate: float
    outsample_win_rate: float
    ic_degradation: float
    is_robust: bool

    def summary(self) -> str:
        deg_sign = "+" if self.ic_degradation > 0 else ""
        lines = [
            "┌─── 样本内外检验 ──────────────────────────────────────────┐",
            f"│  分割日期       : {self.split_date.date()}",
            f"│  IC (样本内)    : {self.insample_ic:>+8.4f}   "
            f"ICIR: {self.insample_icir:>+7.4f}",
            f"│  IC (样本外)    : {self.outsample_ic:>+8.4f}   "
            f"ICIR: {self.outsample_icir:>+7.4f}",
            f"│  IC 衰减幅度    : {deg_sign}{self.ic_degradation:.2%}",
            f"│  胜率 (样本内)  : {self.insample_win_rate:>7.2%}",
            f"│  胜率 (样本外)  : {self.outsample_win_rate:>7.2%}",
            f"│  稳健性         : {'✓ 稳健' if self.is_robust else '✗ 不稳健':>5}",
            "└────────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"InSampleOutSampleResult(ic_in={self.insample_ic:+.4f}, "
            f"ic_out={self.outsample_ic:+.4f}, 稳健={self.is_robust})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 稳健性检验器
# ══════════════════════════════════════════════════════════════════════════════


class RobustnessTester:
    """
    择时因子稳健性评估套件。

    参数
    ----
    forward_period : int
        用于 IC / 信号评估的预测周期（天数）。
    """

    def __init__(self, forward_period: int = 1) -> None:
        self.forward_period = forward_period
        self._ic_tester = CorrelationTester(method="pearson")
        self._sig_tester = SignalTester(forward_period)

    # ────────────────────────────────────────────────────────────────── #
    #  1. 样本内外检验                                                     #
    # ────────────────────────────────────────────────────────────────── #

    def insample_outsample_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
        test_ratio: float = 0.30,
    ) -> InSampleOutSampleResult:
        """
        将数据按时间顺序分割，比较因子在两个子样本中的表现。

        参数
        ----
        factor : pd.Series
            预处理后的因子序列。
        returns : pd.Series
            与因子对齐的日收益率序列。
        test_ratio : float
            样本外测试期占全部数据的比例（默认 0.30，即最后 30%）。

        返回
        ----
        InSampleOutSampleResult
        """
        valid = factor.dropna()
        n = len(valid)
        split_idx = int(n * (1.0 - test_ratio))
        split_date = valid.index[split_idx]

        # 时间序列分割（前 70% 样本内，后 30% 样本外）
        is_factor = factor[:split_date]
        oos_factor = factor[split_date:]
        is_returns = returns[:split_date]
        oos_returns = returns[split_date:]

        # 分别计算 IC
        is_ic_res = self._ic_tester.run_test(is_factor, is_returns, self.forward_period)
        oos_ic_res = self._ic_tester.run_test(oos_factor, oos_returns, self.forward_period)

        # 分别计算阈值信号胜率
        is_sig = self._sig_tester.run_threshold_test(is_factor, is_returns)[0]
        oos_sig = self._sig_tester.run_threshold_test(oos_factor, oos_returns)[0]

        ic_in = is_ic_res.ic_mean
        ic_out = oos_ic_res.ic_mean
        ic_deg = (ic_out - ic_in) / abs(ic_in) if ic_in != 0 else 0.0

        # 稳健性判断：方向一致 且 衰减 < 50%
        is_robust = (
            np.sign(ic_out) == np.sign(ic_in)
            and abs(ic_deg) < 0.50
        )

        return InSampleOutSampleResult(
            split_date=split_date,
            insample_ic=ic_in,
            outsample_ic=ic_out,
            insample_icir=is_ic_res.icir if not np.isnan(is_ic_res.icir) else 0.0,
            outsample_icir=oos_ic_res.icir if not np.isnan(oos_ic_res.icir) else 0.0,
            insample_win_rate=is_sig.overall_win_rate,
            outsample_win_rate=oos_sig.overall_win_rate,
            ic_degradation=ic_deg,
            is_robust=is_robust,
        )

    # ────────────────────────────────────────────────────────────────── #
    #  2. 参数敏感性检验                                                   #
    # ────────────────────────────────────────────────────────────────── #

    def parameter_sensitivity_test(
        self,
        factor_func: Callable[..., pd.Series],
        param_grid: Dict[str, list],
        returns: pd.Series,
        base_params: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        在参数网格上遍历，记录各参数组合下的因子质量指标。

        参数
        ----
        factor_func : callable
            接受关键字参数并返回 pd.Series 因子序列的函数。
        param_grid : dict
            参数名 → 待测值列表的映射。
            例如：{'short_window': [10, 20, 30], 'long_window': [60, 120]}
        returns : pd.Series
            用于 IC / 信号评估的日收益率序列。
        base_params : dict 或 None
            传给 factor_func 的固定参数（与扫描参数合并后使用）。

        返回
        ----
        pd.DataFrame
            每行对应一种参数组合，包含参数名及 ic_mean, icir, win_rate, p_value 等指标。
        """
        records = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))
            full_params = {**(base_params or {}), **params}

            try:
                factor = factor_func(**full_params)
                ic_res = self._ic_tester.run_test(factor, returns, self.forward_period)
                sig_res = self._sig_tester.run_threshold_test(factor, returns)[0]

                records.append(
                    {
                        **params,
                        "ic_mean": ic_res.ic_mean,
                        "icir": ic_res.icir,
                        "win_rate": sig_res.overall_win_rate,
                        "p_value": ic_res.p_value,
                        "is_significant": ic_res.is_significant,
                    }
                )
            except Exception:
                # 参数组合无效时记录空结果
                records.append({**params, "ic_mean": np.nan, "icir": np.nan,
                                 "win_rate": np.nan, "p_value": 1.0,
                                 "is_significant": False})

        return pd.DataFrame(records)

    # ────────────────────────────────────────────────────────────────── #
    #  3. 市场状态检验                                                     #
    # ────────────────────────────────────────────────────────────────── #

    def market_regime_test(
        self,
        factor: pd.Series,
        returns: pd.Series,
        prices: pd.Series,
        regime_window: int = 120,
        bull_threshold: float = 0.15,
        bear_threshold: float = -0.15,
        min_regime_obs: int = 20,
    ) -> Dict[str, ICTestResult]:
        """
        分别在牛市、熊市和震荡市中评估因子有效性。

        市场状态判断方式：以 t 时刻结束的 `regime_window` 日滚动收益率：
            * 牛市（bull）    ：滚动收益 > bull_threshold
            * 熊市（bear）    ：滚动收益 < bear_threshold
            * 震荡市（sideways）：其余情形

        参数
        ----
        factor : pd.Series
            因子序列。
        returns : pd.Series
            日收益率序列。
        prices : pd.Series
            用于判断市场状态的价格序列。
        regime_window : int
            判断市场状态的滚动收益率回望窗口（天数）。
        bull_threshold : float
            牛市判定阈值（如 0.15 = 滚动涨幅超过 15%）。
        bear_threshold : float
            熊市判定阈值（如 -0.15 = 滚动跌幅超过 15%）。
        min_regime_obs : int
            某种市场状态下至少需要的有效观测数，不足则不计算 IC。

        返回
        ----
        dict
            {'bull': ICTestResult, 'bear': ICTestResult, 'sideways': ICTestResult}
            观测数不足的状态会被省略。
        """
        # 计算滚动收益率并对各时间点标注市场状态
        rolling_ret = prices.pct_change(regime_window)
        regime = pd.Series("sideways", index=prices.index)
        regime[rolling_ret > bull_threshold] = "bull"
        regime[rolling_ret < bear_threshold] = "bear"

        results = {}
        for regime_name in ("bull", "bear", "sideways"):
            mask = regime == regime_name
            r_factor = factor[mask]
            r_returns = returns[mask]

            if r_factor.dropna().shape[0] >= min_regime_obs:
                results[regime_name] = self._ic_tester.run_test(
                    r_factor, r_returns, forward_period=self.forward_period
                )

        return results
