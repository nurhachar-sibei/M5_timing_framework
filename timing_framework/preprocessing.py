"""
因子预处理模块
==============

对择时因子进行预处理，消除未来函数（look-ahead bias），改善统计性质，
为后续检验提供标准化、无偏的因子序列。

核心操作
--------
1. 异常值处理  ：基于 MAD（中位数绝对偏差）的去极值处理
2. 标准化      ：Z-score 标准化（全样本或滚动窗口两种模式）
3. 因子正交化  ：对称正交化或 PCA 降维（用于多因子分析去相关）

设计原则
--------
所有滚动方法只使用截至 t 时刻的历史信息来处理 t 时刻的因子值，
严格杜绝未来函数（look-ahead bias）。
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


class FactorPreprocessor:
    """
    择时因子预处理管线。

    各方法均可作为静态方法独立使用，也可通过 :meth:`preprocess` 类方法
    一步完成"去极值 → 标准化"全流程。

    示例
    ----
    单步处理：
        >>> clean = FactorPreprocessor.mad_winsorize(raw_factor, n=3.0)

    滚动窗口全流程（推荐，避免未来函数）：
        >>> processed = FactorPreprocessor.preprocess(
        ...     raw_factor, winsorize=True, standardize=True, rolling_window=252
        ... )

    多因子正交化：
        >>> orth_df = FactorPreprocessor.symmetric_orthogonalize(factor_df)
    """

    # ────────────────────────────────────────────────────────────────── #
    #  异常值处理                                                         #
    # ────────────────────────────────────────────────────────────────── #

    @staticmethod
    def mad_winsorize(series: pd.Series, n: float = 3.0) -> pd.Series:
        """
        基于 MAD（中位数绝对偏差）的去极值处理。

        与均值 ± 3σ 相比，MAD 方法更稳健——中位数和 MAD 本身
        不受待处理极端值的影响。

        截断边界：median ± n × 1.4826 × MAD

        常数 1.4826 使 MAD 成为正态分布下标准差 σ 的一致估计量。

        参数
        ----
        series : pd.Series
            原始因子序列。
        n : float
            截断阈值，以缩放后的 MAD 为单位（默认 3.0）。

        返回
        ----
        pd.Series
            去极值后的因子序列。
        """
        median = series.median()
        mad = (series - median).abs().median()
        sigma_hat = mad * 1.4826  # σ 的一致估计量
        return series.clip(lower=median - n * sigma_hat,
                           upper=median + n * sigma_hat)

    @staticmethod
    def zscore_standardize(series: pd.Series) -> pd.Series:
        """
        全样本 Z-score 标准化：z = (x - μ) / σ

        .. 注意::
            使用全部可用数据计算均值和标准差。仅当确认数据管线中
            不存在未来函数时才应使用此方法。

        参数
        ----
        series : pd.Series
            因子序列（通常在去极值之后调用）。

        返回
        ----
        pd.Series
            均值约为 0、标准差约为 1 的标准化序列。
        """
        mu = series.mean()
        sigma = series.std(ddof=1)
        if sigma == 0:
            return series - mu
        return (series - mu) / sigma

    @staticmethod
    def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """
        滚动窗口 Z-score 标准化（推荐）。

        在 t 时刻，仅使用 [t-window, t-1] 区间的历史数据计算均值和标准差，
        严格避免未来函数。

        参数
        ----
        series : pd.Series
            因子序列。
        window : int
            滚动回望窗口大小（交易日数）。

        返回
        ----
        pd.Series
            滚动标准化后的因子值，前 `window` 个值为 NaN。
        """
        mu = series.rolling(window, min_periods=window // 2).mean()
        sigma = series.rolling(window, min_periods=window // 2).std(ddof=1)
        sigma = sigma.replace(0, np.nan)
        return (series - mu) / sigma

    @staticmethod
    def rolling_mad_winsorize(
        series: pd.Series, window: int, n: float = 3.0
    ) -> pd.Series:
        """
        滚动窗口 MAD 去极值处理。

        在 t 时刻，使用 [t-window, t-1] 区间计算中位数和 MAD，
        对 t 时刻的因子值进行截断。

        参数
        ----
        series : pd.Series
            原始因子序列。
        window : int
            滚动回望窗口大小。
        n : float
            截断阈值倍数（默认 3.0）。

        返回
        ----
        pd.Series
            滚动去极值后的因子序列。
        """
        result = series.astype(float).copy()
        values = series.values.astype(float)
        n_obs = len(values)

        for i in range(window, n_obs):
            w = values[i - window : i]
            w_clean = w[~np.isnan(w)]
            if len(w_clean) < 5:  # 数据不足则跳过
                continue
            med = np.median(w_clean)
            mad = np.median(np.abs(w_clean - med)) * 1.4826
            result.iloc[i] = np.clip(values[i], med - n * mad, med + n * mad) #即不能超过三倍标准差,超过的地方按上下限截断



        return result

    # ────────────────────────────────────────────────────────────────── #
    #  完整预处理管线                                                      #
    # ────────────────────────────────────────────────────────────────── #

    @classmethod
    def preprocess(
        cls,
        series: pd.Series,
        winsorize: bool = True,
        standardize: bool = True,
        rolling_window: Optional[int] = None,
        n_mad: float = 3.0,
    ) -> pd.Series:
        """
        可配置的完整预处理管线。

        处理顺序：去极值（winsorization） → 标准化（standardization）。

        参数
        ----
        series : pd.Series
            原始因子序列。
        winsorize : bool
            是否执行 MAD 去极值。
        standardize : bool
            是否执行 Z-score 标准化。
        rolling_window : int 或 None
            若指定，则使用滚动窗口方法（推荐，避免未来函数）；
            若为 None，则使用全样本统计量。
        n_mad : float
            MAD 去极值的截断倍数。

        返回
        ----
        pd.Series
            预处理后的因子序列。
        """
        result = series.astype(float).copy()

        if winsorize:
            if rolling_window:
                result = cls.rolling_mad_winsorize(result, rolling_window, n_mad)
            else:
                result = cls.mad_winsorize(result, n_mad)

        if standardize:
            if rolling_window:
                result = cls.rolling_zscore(result, rolling_window)
            else:
                result = cls.zscore_standardize(result)

        return result

    # ────────────────────────────────────────────────────────────────── #
    #  多因子正交化                                                        #
    # ────────────────────────────────────────────────────────────────── #

    @staticmethod
    def symmetric_orthogonalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        对称正交化（Löwdin 对称正交化），推荐用于多因子去相关。

        与逐步正交化（Gram-Schmidt）不同，对称正交化具有以下优点：
        - **顺序无关**：结果不依赖因子列的排列顺序
        - **最小失真**：每个正交化后的因子是与原因子最近的正交向量

        算法
        ----
        W = V × D^{-1/2} × V^T，其中 V、D 为相关系数矩阵的特征向量和特征值。
        然后 X_orth = X × W。

        参数
        ----
        df : pd.DataFrame
            各列为标准化后的择时因子的 DataFrame。

        返回
        ----
        pd.DataFrame
            正交化后的因子，保留原有列名和索引。
            计算前会自动删除含 NaN 的行。
        """
        df_clean = df.dropna()
        if df_clean.empty:
            raise ValueError("删除 NaN 行后无完整数据。")

        X = df_clean.values.astype(float)

        # 先对各列标准化，以便计算相关系数矩阵
        col_means = X.mean(axis=0)
        col_stds = X.std(axis=0, ddof=1)
        col_stds[col_stds == 0] = 1.0
        X_std = (X - col_means) / col_stds

        # 计算相关系数矩阵
        n = X_std.shape[0]
        corr = (X_std.T @ X_std) / (n - 1)

        # 特征值分解：corr = V D V^T
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # 数值稳定性

        # 计算对称平方根逆矩阵：W = V D^{-1/2} V^T
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        W = eigenvectors @ D_inv_sqrt @ eigenvectors.T

        X_orth = X_std @ W

        return pd.DataFrame(X_orth, index=df_clean.index, columns=df.columns)

    @staticmethod
    def pca_orthogonalize(
        df: pd.DataFrame, n_components: Optional[int] = None
    ) -> pd.DataFrame:
        """
        基于 PCA 的因子正交化 / 降维。

        将多因子投影到主成分空间，各主成分天然正交（不相关）。

        参数
        ----
        df : pd.DataFrame
            各列为择时因子的 DataFrame。
        n_components : int 或 None
            保留的主成分数量，默认等于输入因子数。

        返回
        ----
        pd.DataFrame
            PCA 变换后的因子，列名为 PC1, PC2, …
            各主成分的解释方差比例存储在 result.attrs['explained_variance_ratio'] 中。
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "PCA 正交化需要安装 scikit-learn，请执行：pip install scikit-learn"
            )

        df_clean = df.dropna()
        if df_clean.empty:
            raise ValueError("删除 NaN 行后无完整数据。")

        n_comp = n_components or df_clean.shape[1]
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(df_clean.values)

        cols = [f"PC{i + 1}" for i in range(n_comp)]
        result = pd.DataFrame(X_pca, index=df_clean.index, columns=cols)
        result.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
        return result
