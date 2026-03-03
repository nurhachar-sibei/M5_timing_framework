# 择时因子评价框架

推荐使用环境：研究环境

> 基于《择时因子的择时框架》所描述的完整方法论，
> 构建的系统性**择时因子**评估与回测工具包。

---

## 目录

1. [框架简介](#1-框架简介)
2. [项目结构](#2-项目结构)
3. [安装与依赖](#3-安装与依赖)
4. [快速开始](#4-快速开始)
5. [配置文件详解](#5-配置文件详解)
6. [数据格式要求](#6-数据格式要求)
7. [输出文件结构](#7-输出文件结构)
8. [核心模块说明](#8-核心模块说明)
9. [回测模块说明](#9-回测模块说明)
10. [独立示例（example）](#10-独立示例example)
11. [评分体系](#11-评分体系)
12. [参考资料](#12-参考资料)

---

## 1. 框架简介

**择时因子**（Timing Factor）是一种时序信号，用于预测单一资产未来收益率的方向或大小，与截面因子（对多只股票横向排名）有本质区别。

本框架通过四步系统化验证流程，对任意择时因子进行全面质量评估：

```
原始因子序列
    │
    ▼
[步骤1] 因子预处理
        MAD去极值 → 滚动Z-score标准化 → （可选）多因子正交化
    │
    ▼
[步骤2] 三维度核心检验
    ├── 信号检验法   — 阈值法 / 均线法 / 极值法
    │                  胜率、预期收益、盈亏比、T检验
    ├── 相关性检验法 — 多周期滚动 IC / ICIR 分析
    │                  IC均值、IC标准差、ICIR、IC正值占比
    └── 回归模型法   — OLS：R(t+n) = α + β·F(t)
                       β显著性、R²、滚动β稳定性
    │
    ▼
[步骤3] 稳健性评估
    ├── 样本内外检验 — 后30%数据作为样本外集
    ├── 参数敏感性   — 在参数网格上扫描ICIR变化
    └── 市场状态分析 — 牛市 / 熊市 / 震荡市分组评估
    │
    ▼
[步骤4] 综合评分与回测
        IC(30%) + 信号(30%) + 回归(20%) + 稳健性(20%)
        → 综合评分[0,1] + 字母评级(A/B/C/D/F)
        → 策略回测：买卖点图 / 策略vs基准 / 绩效归因
```

---

## 2. 项目结构

```
M5_timing_framework/
│
├── main.py                      # 主程序入口（生产使用）
├── config.yaml                  # 统一配置文件
│
├── timing_framework/            # 核心评估包
│   ├── __init__.py              # 公开API（对外暴露主要类）
│   ├── preprocessing.py         # 因子预处理（MAD、Z-score、正交化）
│   ├── signal_testing.py        # 信号检验法（阈值/均线/极值）
│   ├── correlation_testing.py   # 相关性检验法（IC/ICIR）
│   ├── regression_testing.py    # 回归模型法（OLS）
│   ├── robustness.py            # 稳健性评估（样本外/参数/状态）
│   └── evaluator.py             # 综合评价器（主类）
│
├── backtest/                    # 回测模块
│   ├── __init__.py
│   └── backtester.py            # 回测器（Backtester类，三张图表）
│
├── example/                     # 独立示例（合成数据演示）
│   ├── example_timing.py        # 四因子完整演示脚本
│   └── README.md                # 示例专项说明
│
├── data_ini/                    # 数据输入目录
│   ├── price_df.csv             # 多资产价格数据（用户提供）
│   └── signal.csv               # 信号因子数据（用户提供）
│
└── workspace/                   # 输出目录（自动创建）
    ├── {信号名}/                 # 每个信号独立文件夹
    │   ├── plots/eval/          # 因子评估图（8子图看板）
    │   ├── plots/backtest/      # 回测图（3张）
    │   └── timing_report.xlsx  # 该信号的完整Excel报告
    └── factor_comparison.png    # 多信号横向对比图
```

---

## 3. 安装与依赖

**Python 版本要求：** Python 3.8+

**必需依赖：**

```bash
pip install numpy pandas scipy matplotlib pyyaml openpyxl
```

**可选依赖（有则自动启用）：**

```bash
pip install statsmodels   # 更完整的回归输出（否则回退至scipy）
pip install scikit-learn  # PCA正交化功能
```

本框架无需额外安装，直接将项目目录加入Python路径即可使用。

---

## 4. 快速开始

### 4.1 使用主程序（main.py）

```bash
# 1. 将价格和信号数据放入 data_ini/
# 2. 编辑 config.yaml（见第5节）
# 3. 运行
python main.py
```

运行后自动完成：因子评估 → 评估图保存 → 策略回测 → 回测图保存 → Excel报告生成。

### 4.2 在代码中直接调用

```python
import pandas as pd
from timing_framework import TimingFactorEvaluator

# 准备数据
prices  : pd.Series  # 收盘价序列
returns : pd.Series  # 日收益率（prices.pct_change()）
factor  : pd.Series  # 与以上序列日期对齐的因子值

# 创建评估器并运行
ev = TimingFactorEvaluator(
    factor_name   = "我的因子",
    forward_period= 1,           # 预测1日后收益率
    ic_method     = "pearson",
)
ev.evaluate(
    factor        = factor,
    returns       = returns,
    prices        = prices,
    preprocess    = True,        # MAD去极值 + 滚动Z-score
    rolling_window= 252,         # 预处理滚动窗口（1年）
    run_robustness= True,
    ic_periods    = [1, 5, 10, 20],
)

ev.report()                      # 打印文字报告
fig = ev.plot()                  # 生成8子图评估图
fig.savefig("output.png", dpi=100, bbox_inches="tight")
print(ev.score().summary())      # 综合评分摘要
```

---

## 5. 配置文件详解

`config.yaml` 是主程序的唯一配置入口，所有参数均在此设置：

```yaml
# ── 资产配置 ────────────────────────────────────────────────────
asset:
  name: "沪深300"           # 资产显示名称（用于报告标题）
  code: "000300.SH"         # 资产代码（用于匹配价格文件中的列）
  price_field: "CLOSE"      # 价格字段名称

# ── 数据文件配置 ────────────────────────────────────────────────
data:
  price_file:  "price_df.csv"  # data_ini/ 中的价格文件名
  signal_file: "signal.csv"    # data_ini/ 中的信号文件名
  start_date:  "2020-01-01"    # 测试起始日期（空字符串 = 不限制）
  end_date:    ""              # 测试结束日期（空字符串 = 不限制）

# ── 信号配置 ────────────────────────────────────────────────────
signals:
  use_all: false               # true = 使用信号文件全部列
  selected:                    # use_all=false 时，指定要分析的信号列名
    - "rsi_factor"
    - "momentum_factor"

# ── 因子评估参数 ─────────────────────────────────────────────────
evaluation:
  forward_period: 1            # 预测周期（天）
  ic_method: "pearson"         # IC计算方法：pearson / spearman
  rolling_window: 252          # 预处理滚动窗口（天，约1年）
  ic_periods: [1, 5, 10, 20]  # 多周期IC分析的预测天数列表
  run_robustness: true         # 是否执行稳健性检验
  run_rolling_regression: false # 是否执行滚动回归（耗时较长）
  preprocess: true             # 是否对信号做预处理

# ── 回测参数 ─────────────────────────────────────────────────────
backtest:
  benchmark_asset_ratio: 0.5   # 基准中资产占比（其余为无风险资产）
  risk_free_rate: 0.03         # 年化无风险利率
  transaction_cost: 0.001      # 单边手续费率
  long_threshold:  0.0         # 做多阈值（信号 > 此值 → 全仓做多）
  short_threshold: 0.0         # 做空阈值（信号 < 此值负数 → 空仓）

# ── 输出配置 ──────────────────────────────────────────────────────
output:
  workspace_dir: "workspace"              # 工作目录
  eval_plots_subdir: "plots/eval"         # 评估图子目录
  backtest_plots_subdir: "plots/backtest" # 回测图子目录
  excel_filename: "timing_report.xlsx"    # Excel报告文件名
  dpi: 100                                # 图表分辨率
```

---

## 6. 数据格式要求

### 6.1 价格文件（price_df.csv）

- **索引**：日期列，可被 `pd.to_datetime` 解析
- **内容**：多个资产的截面数据，每行一个交易日

程序会按以下优先级自动匹配 `asset.code` 对应的价格列：

| 优先级 | 列名格式                          | 示例                   |
| :----: | --------------------------------- | ---------------------- |
|   1   | 列名 = 资产代码（该列即为收盘价） | `000300.SH`          |
|   2   | `{代码}_{字段}`                 | `000300.SH_CLOSE`    |
|   3   | 仅有一列名为字段名（单资产）      | `CLOSE`              |
|   4   | MultiIndex列                      | `(000300.SH, CLOSE)` |
|   5   | 大小写不敏感模糊匹配              | `close_000300`       |

**示例格式：**

```
date,000300.SH,000905.SH,000016.SH
2020-01-02,4154.13,5896.22,2918.30
2020-01-03,4119.81,5847.63,2894.15
...
```

### 6.2 信号文件（signal.csv）

- **索引**：日期列，需与价格文件日期可对齐（不要求完全相同）
- **列**：每列为一个信号，列名即信号名称，与 `config.yaml` 中的 `signals.selected` 对应
- **值**：连续数值型因子值（预处理前的原始值或已处理值均可）

**示例格式：**

```
date,rsi_factor,momentum_factor,vol_factor
2020-01-02,-0.132,0.045,-0.023
2020-01-03,0.211,-0.031,0.018
...
```

---

## 7. 输出文件结构

每次运行 `main.py` 后，`workspace/` 目录结构如下：

```
workspace/
├── {信号名A}/
│   ├── plots/
│   │   ├── eval/
│   │   │   └── {信号名A}_evaluation.png    ← 8子图因子评估看板
│   │   └── backtest/
│   │       ├── {信号名A}_01_buy_sell.png   ← 买卖点分析图
│   │       ├── {信号名A}_02_vs_benchmark.png ← 策略vs基准对比图
│   │       └── {信号名A}_03_attribution.png  ← 绩效归因图
│   └── timing_report.xlsx                 ← 该信号完整Excel报告
│
├── {信号名B}/
│   └── ...（结构同上）
│
└── factor_comparison.png                  ← 多信号横向对比图（≥2个信号时生成）
```

### Excel报告结构（timing_report.xlsx）

| Sheet                  | 内容                                                                             |
| ---------------------- | -------------------------------------------------------------------------------- |
| **综合概览**     | 信号全指标汇总：IC、胜率、β、稳健性、综合评分与评级                             |
| **IC相关性分析** | 各预测周期（1/5/10/20日）的IC均值、IC标准差、ICIR、IC正值占比                    |
| **信号检验结果** | 阈值法/均线法/极值法三种方法的多空样本数、胜率、均收益、盈亏比                   |
| **回归分析**     | OLS的α/β系数、T值、P值、R²、调整R²、F统计量                                  |
| **稳健性检验**   | 样本内外ICIR对比、IC衰减幅度、稳健性结论、牛熊震荡三市场IC分解                   |
| **回测绩效汇总** | 策略与基准的年化收益、波动率、Sharpe、最大回撤、Calmar、胜率、交易次数及超额指标 |

---

## 8. 核心模块说明

### 8.1 因子预处理（`FactorPreprocessor`）

```python
from timing_framework import FactorPreprocessor

# 完整预处理流水线（推荐）
factor_clean = FactorPreprocessor.preprocess(
    factor,
    winsorize     = True,   # MAD去极值：截断至 中位数 ± 3×MAD×1.4826
    standardize   = True,   # 标准化
    rolling_window= 252,    # 滚动窗口（严格防止未来信息泄漏）
)

# 多因子正交化（去除因子间相关性）
factor_df_orth = FactorPreprocessor.symmetric_orthogonalize(factor_df)
```

**设计原则**：所有滚动方法仅使用当期及之前的数据，确保无未来信息泄漏。

---

### 8.2 信号检验（`SignalTester`）

三种信号生成方式：

| 方法             | 做多条件           | 做空条件           | 适用场景              |
| ---------------- | ------------------ | ------------------ | --------------------- |
| **阈值法** | 因子 > +1σ        | 因子 < -1σ        | 因子经过Z-score处理后 |
| **均线法** | 因子 > 自身N日均线 | 因子 < 自身N日均线 | 捕捉趋势变化          |
| **极值法** | 因子 ≤ 20%分位数  | 因子 ≥ 80%分位数  | 反转型因子            |

各方法输出指标：

| 指标     | 说明                                |
| -------- | ----------------------------------- |
| 胜率     | P(收益>0\| 做多)，P(收益<0 \| 做空) |
| 预期收益 | E[收益\| 信号]                      |
| 盈亏比   | \|平均盈利 / 平均亏损\|             |
| T检验    | Welch检验 H₀：E[多头]= E[空头]     |

---

### 8.3 相关性检验（`CorrelationTester`）

计算 **IC**（信息系数）：因子 F(t) 与未来收益 R(t+n) 之间的时序相关系数。

```python
from timing_framework.correlation_testing import CorrelationTester

tester = CorrelationTester(method="pearson")
result = tester.run_test(factor, returns, forward_period=1, rolling_window=60)
# result.ic_mean, result.icir, result.ic_positive_ratio, ...

multi = tester.run_multi_period(factor, returns, periods=[1, 5, 10, 20])
```

**ICIR 参考标准：**

| ICIR      | 评价                         |
| --------- | ---------------------------- |
| < 0.3     | 偏弱，可能是噪声             |
| 0.3 ~ 0.5 | 边缘可用                     |
| 0.5 ~ 1.0 | **良好**，实践中可使用 |
| ≥ 1.0    | **优秀**               |

---

### 8.4 回归检验（`RegressionTester`）

OLS线性回归模型：**R(t+n) = α + β · F(t) + ε**

```python
from timing_framework.regression_testing import RegressionTester

tester = RegressionTester()
result = tester.run_regression(factor, returns, forward_period=1)
# result.beta, result.beta_tstat, result.r_squared, result.is_significant
```

- **β > 0 且 p < 0.05**：因子正向显著预测收益率
- **β < 0 且 p < 0.05**：反转型有效因子（同样有价值）
- `statsmodels` 未安装时自动回退至 `scipy.stats.linregress`

---

### 8.5 稳健性评估（`RobustnessTester`）

三项独立稳健性检验：

**① 样本内外检验**

- 最后30%数据作为样本外保留集
- 稳健标准：样本外IC方向与样本内一致，且衰减幅度 < 50%

**② 参数敏感性检验**

- 在用户定义的参数网格上扫描ICIR
- 稳健因子在合理参数范围内ICIR应保持一致

**③ 市场状态分析**

- 按滚动N日收益率划分牛市/熊市/震荡市
- 分市场状态分别计算IC，揭示因子的市场适用性

---

### 8.6 综合评价器（`TimingFactorEvaluator`）

整合以上所有模块的主类，提供统一调用接口：

```python
from timing_framework import TimingFactorEvaluator

ev = TimingFactorEvaluator(factor_name="MA动量", forward_period=1)
ev.evaluate(factor, returns, prices=prices)

ev.report()           # 打印完整文字报告
fig = ev.plot()       # 生成8子图评估图
sc = ev.score()       # 返回 FactorScore(ic/signal/regression/robustness/composite/grade)
```

**内部结果属性（可直接访问）：**

| 属性                   | 类型                            | 说明                   |
| ---------------------- | ------------------------------- | ---------------------- |
| `_factor`            | `pd.Series`                   | 预处理后的因子序列     |
| `_signal_results`    | `Dict[str, SignalTestResult]` | 三种方法的信号检验结果 |
| `_ic_results`        | `Dict[int, ICTestResult]`     | 各预测周期的IC结果     |
| `_reg_result`        | `RegressionResult`            | OLS回归结果            |
| `_robustness_result` | `InSampleOutSampleResult`     | 样本内外检验结果       |
| `_regime_results`    | `Dict[str, ICTestResult]`     | 各市场状态下的IC结果   |

---

## 9. 回测模块说明

### 策略逻辑

```
信号 > long_threshold   →  全仓持有资产（仓位 = 1.0）
信号 < -short_threshold →  空仓（仓位 = 0.0）
其他区间               →  维持上一期仓位（初始默认空仓）
```

信号延迟一个交易日使用，避免当日信号当日交易的未来信息泄漏。

### 基准组合

```
基准日收益 = benchmark_asset_ratio × 资产日收益
           + (1 - benchmark_asset_ratio) × 日无风险利率
```

默认基准：**50% 资产 + 50% 货币**（可在 `config.yaml` 中调整比例）。

### 绩效指标

| 指标       | 说明                        |
| ---------- | --------------------------- |
| 总收益率   | 累计净值 - 1                |
| 年化收益率 | (1 + 总收益)^(252/N) - 1    |
| 年化波动率 | 日收益率标准差 × √252     |
| Sharpe比率 | (年化超额收益) / 年化波动率 |
| 最大回撤   | 净值从最高点的最大跌幅      |
| Calmar比率 | 年化收益率 /\|最大回撤\|    |
| 日胜率     | 日收益率 > 0 的天数占比     |
| 交易次数   | 仓位发生变化的次数          |

### 三张图表

| 图表                     | 内容                                                   |
| ------------------------ | ------------------------------------------------------ |
| `_01_buy_sell.png`     | 收盘价曲线 + 买入（▲）/卖出（▼）标记 + 信号值子图    |
| `_02_vs_benchmark.png` | 累计净值对比 + 超额收益 + 回撤曲线（三子图）           |
| `_03_attribution.png`  | 月度收益热力图 + 年度收益条形图 + 滚动Sharpe（三子图） |

---

## 10. 独立示例（example）

`example/example_timing.py` 使用**合成A股数据**演示四个典型择时因子的完整评估流程，无需真实数据即可运行，适合快速了解框架能力。

```bash
cd M5_timing_framework
python example/example_timing.py
```

**四个演示因子：**

| 因子        | 公式                           | 逻辑                  |
| ----------- | ------------------------------ | --------------------- |
| MA动量因子  | `(MA₂₀ - MA₆₀) / MA₆₀` | 均线黄金/死叉趋势信号 |
| RSI反转因子 | `(50 - RSI₁₄) / 50`        | 超买超卖反转信号      |
| 波动率因子  | `-滚动标准差(收益率, 20)`    | 低波动率看多信号      |
| 估值因子    | `-(PE - PE合理) / PE合理`    | PE均值回归信号        |

示例还额外演示了参数敏感性分析与多因子正交化，详见 `example/README.md`。

---

## 11. 评分体系

### 综合评分计算

| 维度       | 权重 | 评分依据                                       |
| ---------- | ---- | ---------------------------------------------- |
| IC评分     | 30%  | `min(1.0,                                      |
| 信号评分   | 30%  | 胜率在[50%, 70%]线性映射 + 显著性加分（+0.20） |
| 回归评分   | 20%  | β显著→0.50基分，+ R² × 10加分（上限0.50）  |
| 稳健性评分 | 20%  | 样本外同向（0.50）+ IC衰减程度（0.50）         |

### 评级划分

| 评级        | 综合评分 | 建议                     |
| ----------- | -------- | ------------------------ |
| **A** | ≥ 0.75  | 优质因子，建议纳入策略   |
| **B** | ≥ 0.60  | 良好因子，可作为辅助信号 |
| **C** | ≥ 0.45  | 一般因子，需谨慎使用     |
| **D** | ≥ 0.30  | 较差，建议优化或放弃     |
| **F** | < 0.30   | 无效，不建议使用         |

---

## 12. 参考资料

本框架的方法论完整基于：

> **《择时因子的择时框架》**，蝴蝶量化，2026年1月。

文章描述的四步验证框架已完整实现于本项目：

| 文章内容     | 对应模块                                    |
| ------------ | ------------------------------------------- |
| 因子预处理   | `timing_framework/preprocessing.py`       |
| 信号检验法   | `timing_framework/signal_testing.py`      |
| 相关性检验法 | `timing_framework/correlation_testing.py` |
| 回归模型法   | `timing_framework/regression_testing.py`  |
| 稳健性评估   | `timing_framework/robustness.py`          |
| 综合评分     | `timing_framework/evaluator.py`           |
| 策略回测     | `backtest/backtester.py`                  |
| 主程序与报告 | `main.py` + `config.yaml`               |
