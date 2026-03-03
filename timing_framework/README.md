# 择时因子评价框架

基于《择时因子的择时框架》（蝴蝶量化，2026年1月）所描述的完整方法论，
构建的系统性择时因子评估Python工具包。

---

## 概述

**择时因子**是一种定量信号，用于预测单一资产未来收益率的方向或大小。
与截面因子（在多只股票之间横向排名）不同，择时因子每期产生一个值，
指示应持有多头、空头还是保持空仓。

本框架通过三种互补方法加稳健性评估，系统衡量择时因子的质量：

```
原始因子
    │
    ├── [1] 预处理 ────────── MAD去极值  →  滚动Z-score标准化
    │
    ├── [2] 信号检验 ────────── 阈值法 / 均线法 / 极值法
    │         │                 → 胜率、预期收益、盈亏比、T检验
    │
    ├── [3] 相关性检验 ──────── 滚动IC与ICIR（多预测周期）
    │         │                 → IC均值、IC标准差、ICIR、IC正值占比
    │
    ├── [4] 回归模型法 ──────── OLS：R(t+n) = α + β·F(t)
    │         │                 → β显著性、R²、滚动β稳定性
    │
    └── [5] 稳健性评估 ──────── 样本内外检验 | 参数敏感性检验
                                市场状态分析（牛市/熊市/震荡市）
```

框架对每个因子输出 **综合评分（0–1）** 和 **字母评级（A–F）**，
支持跨策略的客观比较。

---

## 项目结构

```
M5_timing_framework/
├── timing_framework/              # 核心包
│   ├── __init__.py                # 公开API
│   ├── preprocessing.py           # MAD去极值、Z-score、正交化
│   ├── signal_testing.py          # 信号生成与评估指标
│   ├── correlation_testing.py     # IC / ICIR分析
│   ├── regression_testing.py      # OLS时序回归
│   ├── robustness.py              # 样本内外、参数敏感性、市场状态检验
│   └── evaluator.py               # TimingFactorEvaluator（主评估器）
├── example/
│   ├── example_timing.py          # 完整使用示例（4个因子）
│   ├── README.md                  # 示例专项说明
│   └── plots/                     # 图表输出目录
└── README.md                      # 本文件
```

---

## 安装

本包需要 Python 3.8+ 及以下依赖库：

```bash
pip install numpy pandas scipy matplotlib
# 可选但推荐：
pip install statsmodels   # 更丰富的回归输出
pip install scikit-learn  # PCA正交化
```

`timing_framework` 包本身无需额外安装，直接克隆或复制文件夹后即可导入使用。

---

## 快速上手

```python
import pandas as pd
from timing_framework import TimingFactorEvaluator

# 假设已准备好：
#   factor  : pd.Series  – 与行情对齐的因子值序列
#   returns : pd.Series  – 日收益率序列（prices.pct_change()）
#   prices  : pd.Series  – 收盘价序列

evaluator = TimingFactorEvaluator(
    factor_name="MA动量",
    forward_period=1,       # 预测1日后收益率
    ic_method="pearson",
)

evaluator.evaluate(
    factor=factor,
    returns=returns,
    prices=prices,          # 可选，市场状态分析所需
    preprocess=True,        # 执行MAD去极值 + 滚动Z-score
    rolling_window=252,     # 1年预处理滚动窗口
    run_robustness=True,
    ic_periods=[1, 5, 10, 20],
)

evaluator.report()          # 打印文字报告
fig = evaluator.plot()      # 生成8子图评估图
fig.savefig("output.png", dpi=100, bbox_inches="tight")
```

---

## 框架模块说明

### 1. 因子预处理（`FactorPreprocessor`）

| 方法 | 说明 |
|------|------|
| `mad_winsorize(series, n=3.0)` | 将异常值截断至 中位数 ± n × 1.4826 × MAD |
| `zscore_standardize(series)` | 全样本Z-score标准化：(x − μ) / σ |
| `rolling_zscore(series, window)` | 滚动Z-score——严格防止未来信息泄漏 |
| `rolling_mad_winsorize(series, window)` | 滚动MAD去极值 |
| `preprocess(series, ...)` | 完整流水线（去极值 → 标准化） |
| `symmetric_orthogonalize(df)` | Löwdin对称正交化（多因子场景） |
| `pca_orthogonalize(df)` | PCA降维正交化 |

**核心原则**：滚动方法仅使用t时刻及之前的数据，严格防止未来信息泄漏。

---

### 2. 信号检验（`SignalTester`）

三种信号生成方式：

| 方法 | 做多信号（+1） | 做空信号（−1） |
|------|--------------|--------------|
| **阈值法** | 因子 > 上界（默认+1σ） | 因子 < 下界（默认−1σ） |
| **均线法** | 因子 > 其自身N日均线 | 因子 < 其自身N日均线 |
| **极值法** | 因子 ≤ 下分位数（如20%） | 因子 ≥ 上分位数（如80%） |

各方法的评估指标：

| 指标 | 说明 |
|------|------|
| **胜率** | P(收益>0 \| 做多) 及 P(收益<0 \| 做空) |
| **预期收益** | E[收益 \| 信号]，做多应为正，做空应为负 |
| **盈亏比** | \|平均盈利 / 平均亏损\| |
| **T检验** | Welch检验：H₀：E[多头收益] = E[空头收益] |

---

### 3. 相关性检验（`CorrelationTester`）

计算**信息系数（IC）**——t时刻因子值与t+n时刻实现收益率之间的时序相关系数。

| 指标 | 参考标准 | 说明 |
|------|---------|------|
| **IC均值** | > 0.02 | 平均预测相关性 |
| **IC标准差** | 越小越稳定 | 滚动IC的波动性 |
| **ICIR** | > 0.5良好，> 1.0优秀 | IC均值 / IC标准差 |
| **IC > 0占比** | > 55% | IC为正的窗口占比 |

择时因子ICIR参考标准（通常高于截面因子，因时序数据量更多）：

```
ICIR < 0.3        ：偏弱——可能是噪声
0.3 ≤ ICIR < 0.5  ：边缘
0.5 ≤ ICIR < 1.0  ：良好——实践中可使用
ICIR ≥ 1.0        ：优秀
```

---

### 4. 回归模型法（`RegressionTester`）

OLS回归模型：**R(t+n) = α + β · F(t) + ε**

| 参数 | 解读 |
|------|------|
| `β > 0，p < 0.05` | 因子正向且显著预测收益率 |
| `β < 0，p < 0.05` | 因子为反转型预测（同样有效） |
| `p ≥ 0.05` | 无可靠的线性预测关系 |
| `R²` | 收益率方差中被解释的比例 |

滚动回归可揭示β是否随时间稳定，或存在状态依赖性。

---

### 5. 稳健性评估（`RobustnessTester`）

三项稳健性检验：

**样本内外检验**
- 最后30%数据作为样本外评估集
- 稳健标准：样本外IC与样本内IC方向一致，且衰减幅度 < 50%

**参数敏感性检验**
- 在构造参数网格上扫描
- 稳健因子在合理参数范围内应保持一致的ICIR水平

**市场状态分析**
- 按滚动N日收益率划分牛市/熊市/震荡市
- 分别在各市场状态下评估因子质量

---

### 6. 综合评分

| 评分维度 | 权重 | 使用指标 |
|---------|------|---------|
| IC评分 | 30% | \|ICIR\|，上限取1.0 |
| 信号评分 | 30% | 胜率 + 显著性加成 |
| 回归评分 | 20% | β显著性 + R²加成 |
| 稳健性评分 | 20% | 样本外同向 + IC衰减幅度 |

**评级标准**：A（≥ 0.75）| B（≥ 0.60）| C（≥ 0.45）| D（≥ 0.30）| F（< 0.30）

---

## 多因子正交化

同时分析多个因子时，可用对称正交化去除因子间相关性：

```python
import pandas as pd
from timing_framework import FactorPreprocessor

# 将预处理后的因子合并为DataFrame
factor_df = pd.DataFrame({
    "MA动量":  ma_factor,
    "RSI反转": rsi_factor,
    "波动率":  vol_factor,
})

# 对称（Löwdin）正交化——与因子排列顺序无关
factor_df_orth = FactorPreprocessor.symmetric_orthogonalize(factor_df)
```

对称方法与Gram-Schmidt（逐步）正交化不同，**与因子顺序无关**，
且对每个因子的信息损失最小。

---

## 关键设计原则

1. **无未来信息泄漏**：所有滚动预处理窗口仅使用t时刻及之前的数据。
   建议使用默认的 `preprocess(..., rolling_window=252)`。

2. **前向收益对齐**：`factor[t]` 始终与 `return[t+1, ..., t+n]` 配对，
   所有移位操作均在内部完成。

3. **关注点分离**：每个评估模块均可独立使用，也可通过 `TimingFactorEvaluator` 组合调用。

4. **statsmodels可选**：若未安装statsmodels，回归模块自动回退至 `scipy.stats.linregress`。

---

## 参考文献

本框架基于以下文章的方法论构建：
> *《择时因子的择时框架》*，蝴蝶量化，2026年1月。

框架涵盖文章中的四步验证流程：
**预处理 → 核心检验 → 稳健性评估 → 综合决策**
