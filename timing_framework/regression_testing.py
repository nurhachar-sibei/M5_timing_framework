"""
回归模型法模块
==============

实现《择时因子的择时框架》中的"回归模型法"，
通过 OLS 时间序列回归检验择时因子对未来收益率的线性预测能力。

模型
----
    R(t + n) = α + β · F(t) + ε(t)

其中：
    R(t+n) = t 时刻后第 n 期的实际收益率
    F(t)   = t 时刻的因子值（建议标准化）
    α      = 截距（反映市场长期漂移）
    β      = 因子系数——核心关注指标

判断标准
--------
* β > 0 且显著（p < 0.05）：因子正向预测未来收益率，是有效的趋势跟踪因子。
* β < 0 且显著           ：因子反向预测，是有效的反转/均值回归因子。
* β 不显著               ：因子与未来收益率无可靠线性关系。

滚动回归可追踪 β 随时间的稳定性，识别因子是否存在结构性变化或状态依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# 结果容器
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RegressionResult:
    """
    OLS 回归 R(t+n) = α + β · F(t) 的完整结果。

    属性
    ----
    alpha : float
        截距（日度数据乘以 252 可得年化值）。
    beta : float
        因子系数，为正表示因子预测正收益。
    alpha_tstat : float
        截距的 t 统计量。
    beta_tstat : float
        因子系数的 t 统计量。
    alpha_pvalue : float
        截距的双侧 p 值。
    beta_pvalue : float
        因子系数的双侧 p 值。
    r_squared : float
        拟合优度 R²。
    adj_r_squared : float
        调整后的 R²。
    f_statistic : float
        F 统计量（整体显著性检验）。
    f_pvalue : float
        F 检验的 p 值。
    n_observations : int
        参与回归的观测数量。
    is_significant : bool
        beta_pvalue < 0.05 时为 True。
    residuals : pd.Series
        回归残差序列（可用于诊断图）。
    """

    alpha: float
    beta: float
    alpha_tstat: float
    beta_tstat: float
    alpha_pvalue: float
    beta_pvalue: float
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    n_observations: int
    is_significant: bool
    residuals: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        lines = [
            "┌─── 回归结果 ───────────────────────────────────────────────┐",
            "│  模型：R(t+n) = α + β · F(t)                              │",
            f"│  观测数 : {self.n_observations}",
            f"│  α      : {self.alpha:>+10.6f}  (t={self.alpha_tstat:>+7.3f},  p={self.alpha_pvalue:.4f})",
            f"│  β      : {self.beta:>+10.6f}  (t={self.beta_tstat:>+7.3f},  p={self.beta_pvalue:.4f})",
            f"│  R²     : {self.r_squared:>8.4f}",
            f"│  Adj R² : {self.adj_r_squared:>8.4f}",
            f"│  F      : {self.f_statistic:>8.4f}  (p={self.f_pvalue:.4f})",
            f"│  β 显著性 (α=5%): {'✓ 显著' if self.is_significant else '✗ 不显著':>5}",
            "└────────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RegressionResult(β={self.beta:+.6f}, t={self.beta_tstat:+.3f}, "
            f"p={self.beta_pvalue:.4f}, R²={self.r_squared:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 回归检验器
# ══════════════════════════════════════════════════════════════════════════════


class RegressionTester:
    """
    基于 OLS 的回归模型法检验器。

    优先使用 statsmodels（输出更详细），若未安装则回退到 scipy.stats.linregress。
    """

    def __init__(self) -> None:
        self._use_statsmodels = self._check_statsmodels()

    @staticmethod
    def _check_statsmodels() -> bool:
        """检查 statsmodels 是否可用。"""
        try:
            import statsmodels.api  # noqa: F401
            return True
        except ImportError:
            return False

    # ────────────────────────────────────────────────────────────────── #
    #  核心回归                                                           #
    # ────────────────────────────────────────────────────────────────── #

    def run_regression(
        self,
        factor: pd.Series,
        returns: pd.Series,
        forward_period: int = 1,
    ) -> RegressionResult:
        """
        拟合 R(t+n) = α + β · F(t) 的 OLS 回归。

        参数
        ----
        factor : pd.Series
            t 时刻的择时因子值（建议标准化）。
        returns : pd.Series
            日收益率序列（内部自动按 forward_period 移位）。
        forward_period : int
            预测周期（天数）。

        返回
        ----
        RegressionResult
        """
        fwd = returns.shift(-forward_period)
        df = pd.DataFrame({"f": factor, "r": fwd}).dropna()

        if len(df) < 10:
            raise ValueError(
                f"数据量不足，无法完成回归：仅有 {len(df)} 条观测。"
            )

        X = df["f"].values
        y = df["r"].values*100
        # print("@@@@@@@@@@@@@@@@@@@@@")
        # print(X)
        # print(y)
        if self._use_statsmodels:
            return self._fit_statsmodels(X, y, df.index)
        else:
            return self._fit_scipy(X, y, df.index)

    @staticmethod
    def _fit_statsmodels(
        X: np.ndarray, y: np.ndarray, index: pd.Index
    ) -> RegressionResult:
        """使用 statsmodels 进行 OLS 拟合。"""
        import statsmodels.api as sm

        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()


        return RegressionResult(
            alpha=float(model.params[0]),
            beta=float(model.params[1]),
            alpha_tstat=float(model.tvalues[0]),
            beta_tstat=float(model.tvalues[1]),
            alpha_pvalue=float(model.pvalues[0]),
            beta_pvalue=float(model.pvalues[1]),
            r_squared=float(model.rsquared),
            adj_r_squared=float(model.rsquared_adj),
            f_statistic=float(model.fvalue) if model.fvalue is not None else np.nan,
            f_pvalue=float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
            n_observations=int(model.nobs),
            is_significant=(float(model.pvalues[1]) < 0.05),
            residuals=pd.Series(model.resid, index=index),
        )

    @staticmethod
    def _fit_scipy(
        X: np.ndarray, y: np.ndarray, index: pd.Index
    ) -> RegressionResult:
        """回退方案：使用 scipy.stats.linregress 进行 OLS 拟合。"""
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        n = len(X)
        r_squared = r_value ** 2
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else np.nan

        # 手动计算 t 统计量
        y_hat = intercept + slope * X
        residuals = y - y_hat
        mse = np.sum(residuals ** 2) / (n - 2)
        X_centered = X - X.mean()
        ss_xx = np.sum(X_centered ** 2)

        se_beta = np.sqrt(mse / ss_xx) if ss_xx > 0 else np.nan
        se_alpha = np.sqrt(mse * (1 / n + X.mean() ** 2 / ss_xx)) if ss_xx > 0 else np.nan

        beta_tstat = slope / se_beta if se_beta and se_beta > 0 else 0.0
        alpha_tstat = intercept / se_alpha if se_alpha and se_alpha > 0 else 0.0

        alpha_pvalue = 2 * stats.t.sf(abs(alpha_tstat), df=n - 2)
        f_stat = r_squared / (1 - r_squared) * (n - 2) if r_squared < 1 else np.nan
        f_pvalue = stats.f.sf(f_stat, 1, n - 2) if not np.isnan(f_stat) else np.nan

        return RegressionResult(
            alpha=float(intercept),
            beta=float(slope),
            alpha_tstat=float(alpha_tstat),
            beta_tstat=float(beta_tstat),
            alpha_pvalue=float(alpha_pvalue),
            beta_pvalue=float(p_value),
            r_squared=float(r_squared),
            adj_r_squared=float(adj_r_squared),
            f_statistic=float(f_stat) if not np.isnan(f_stat) else np.nan,
            f_pvalue=float(f_pvalue) if not np.isnan(f_pvalue) else np.nan,
            n_observations=n,
            is_significant=(float(p_value) < 0.05),
            residuals=pd.Series(residuals, index=index),
        )

    # ────────────────────────────────────────────────────────────────── #
    #  滚动回归                                                           #
    # ────────────────────────────────────────────────────────────────── #

    def rolling_regression(
        self,
        factor: pd.Series,
        returns: pd.Series,
        window: int = 252,
        forward_period: int = 1,
    ) -> pd.DataFrame:
        """
        滚动 OLS 回归，追踪因子系数随时间的稳定性。

        在 t 时刻，使用最近 `window` 个（因子值，未来收益率）对拟合模型。
        得到的 β 时间序列可揭示因子预测能力是持续稳定还是状态依赖。

        参数
        ----
        factor : pd.Series
            因子序列。
        returns : pd.Series
            日收益率序列。
        window : int
            滚动窗口大小（观测条数）。
        forward_period : int
            预测周期（天数）。

        返回
        ----
        pd.DataFrame
            列：alpha, beta, r_squared, beta_pvalue；索引为日期。
        """
        fwd = returns.shift(-forward_period)
        df = pd.DataFrame({"f": factor, "r": fwd}).dropna()

        records = []
        for i in range(window, len(df) + 1):
            sub = df.iloc[i - window : i]
            if len(sub) < window // 2:
                continue
            try:
                slope, intercept, r_val, p_val, _ = stats.linregress(
                    sub["f"], sub["r"]
                )
                records.append(
                    {
                        "date": df.index[i - 1],
                        "alpha": intercept,
                        "beta": slope,
                        "r_squared": r_val ** 2,
                        "beta_pvalue": p_val,
                    }
                )
            except Exception:
                continue

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("date")
