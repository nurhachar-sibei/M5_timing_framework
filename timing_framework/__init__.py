"""
择时因子评价框架
================

基于《择时因子的择时框架》（蝴蝶量化，2026）构建的系统性择时因子评估工具包。

本框架覆盖完整的因子评估流程：
    1. 因子预处理      —— 异常值处理（MAD去极值）、标准化、正交化
    2. 信号检验法      —— 阈值法 / 均线法 / 极值法，计算胜率/盈亏比/显著性
    3. 相关性检验法    —— IC / ICIR 分析，支持多预测周期
    4. 回归模型法      —— OLS 时间序列回归，检验因子系数 β 的显著性
    5. 稳健性评估      —— 样本内外检验 / 参数敏感性分析 / 市场状态检验

快速上手
--------
    from timing_framework import TimingFactorEvaluator

    evaluator = TimingFactorEvaluator(factor_name='MA动量', forward_period=1)
    evaluator.evaluate(factor, returns, prices=prices)
    evaluator.report()   # 打印文本报告
    evaluator.plot()     # 生成8面板可视化图
"""

from .preprocessing import FactorPreprocessor
from .signal_testing import (
    SignalTester,
    SignalTestResult,
    generate_threshold_signals,
    generate_ma_signals,
    generate_percentile_signals,
    evaluate_signals,
)
from .correlation_testing import (
    CorrelationTester,
    ICTestResult,
    calculate_ic,
    calculate_rolling_ic,
)
from .regression_testing import RegressionTester, RegressionResult
from .robustness import RobustnessTester, InSampleOutSampleResult
from .evaluator import TimingFactorEvaluator, FactorScore

__version__ = "1.0.0"
__author__ = "Timing Framework"

__all__ = [
    # 核心评估器
    "TimingFactorEvaluator",
    "FactorScore",
    # 子模块
    "FactorPreprocessor",
    "SignalTester",
    "CorrelationTester",
    "RegressionTester",
    "RobustnessTester",
    # 结果数据类
    "SignalTestResult",
    "ICTestResult",
    "RegressionResult",
    "InSampleOutSampleResult",
    # 工具函数
    "generate_threshold_signals",
    "generate_ma_signals",
    "generate_percentile_signals",
    "evaluate_signals",
    "calculate_ic",
    "calculate_rolling_ic",
]
