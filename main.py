"""
择时因子评价框架 — 主程序
=========================

使用方式
--------
1. 将 price_df.csv 和 signal.csv 放入 data_ini/ 目录
2. 编辑 config.yaml 中的资产代码、信号名称及各项参数
3. 运行：python main.py

输出
----
- workspace/plots/eval/      : 因子评估图（每个信号8子图 + 多因子对比图）
- workspace/plots/backtest/  : 回测图（每个信号3张图）
- workspace/timing_report.xlsx : 完整 Excel 报告（6个分析Sheet）

数据格式要求
-----------
price_df.csv
    支持以下列命名格式（会自动匹配）：
    1. 列名即为资产代码，如 "000300.SH"（CLOSE价格直接在该列）
    2. 列名为 "{资产代码}_{字段}"，如 "000300.SH_CLOSE"
    3. 多级索引列，如 ("000300.SH", "CLOSE")
    4. 单资产文件，列名直接为 "CLOSE" 或 "close"
    索引须为日期格式（可被 pd.to_datetime 解析）。

signal.csv
    每列为一个信号，列名即信号名称。
    索引须为日期格式，值为连续因子数值。
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 路径配置：确保能导入 timing_framework ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

# ── Matplotlib 后端与字体（必须在其他 import 之前设置）──────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
                                    "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd

# ── 第三方库（带友好报错）──────────────────────────────────────────────────
try:
    import yaml
except ImportError:
    raise ImportError("请安装 PyYAML：pip install pyyaml")

try:
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import (
        Alignment, Border, Font, PatternFill, Side,
        numbers as xl_numbers,
    )
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule
except ImportError:
    raise ImportError("请安装 openpyxl：pip install openpyxl")

# ── 内部模块 ──────────────────────────────────────────────────────────────
from timing_framework import FactorPreprocessor, TimingFactorEvaluator
from backtest.backtester import Backtester, BacktestResult


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """加载 YAML 配置文件，返回配置字典。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_data(data_dir: str | Path = "data_ini",
             price_file: str = "price_df.csv",
             signal_file: str = "signal.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 data_ini/ 目录加载价格和信号 CSV 文件。

    文件名通过 config.yaml 的 data.price_file / data.signal_file 指定。

    返回
    ----
    (price_df, signal_df)，索引均为 DatetimeIndex。
    """
    data_dir = Path(data_dir)
    price_path  = data_dir / price_file
    signal_path = data_dir / signal_file

    if not price_path.exists():
        raise FileNotFoundError(f"价格文件不存在：{price_path.resolve()}")
    if not signal_path.exists():
        raise FileNotFoundError(f"信号文件不存在：{signal_path.resolve()}")

    price_df  = pd.read_csv(price_path,  index_col=0, parse_dates=True)
    signal_df = pd.read_csv(signal_path, index_col=0, parse_dates=True)

    price_df.index  = pd.to_datetime(price_df.index)
    signal_df.index = pd.to_datetime(signal_df.index)

    price_df  = price_df.sort_index()
    signal_df = signal_df.sort_index()

    return price_df, signal_df


def extract_price_series(price_df: pd.DataFrame,
                          asset_code: str,
                          price_field: str = "CLOSE") -> pd.Series:
    """
    从多资产价格 DataFrame 中智能提取指定资产的价格序列。

    尝试顺序：
    1. 列名 = asset_code（列本身就是CLOSE价格）
    2. 列名 = "{asset_code}_{price_field}"
    3. 列名 = "{price_field}"（单资产文件）
    4. 多级索引 (asset_code, price_field)
    5. 大小写不敏感的模糊匹配
    """
    # 策略1：列名直接是 asset_code
    if asset_code in price_df.columns:
        return price_df[asset_code].dropna().rename("price")

    # 策略2：列名为 "{code}_{field}"
    col2 = f"{asset_code}_{price_field}"
    if col2 in price_df.columns:
        return price_df[col2].dropna().rename("price")

    # 策略3：单资产，列名直接是 price_field
    pf_lower = price_field.lower()
    exact_matches = [c for c in price_df.columns
                     if isinstance(c, str) and c.lower() == pf_lower]
    if len(exact_matches) == 1:
        return price_df[exact_matches[0]].dropna().rename("price")

    # 策略4：MultiIndex 列
    if isinstance(price_df.columns, pd.MultiIndex):
        if (asset_code, price_field) in price_df.columns:
            return price_df[(asset_code, price_field)].dropna().rename("price")
        if (asset_code, price_field.lower()) in price_df.columns:
            return price_df[(asset_code, price_field.lower())].dropna().rename("price")

    # 策略5：大小写不敏感模糊匹配（含资产代码和字段名）
    for col in price_df.columns:
        if isinstance(col, str):
            cl = col.lower()
            if asset_code.lower() in cl and pf_lower in cl:
                return price_df[col].dropna().rename("price")

    # 均未匹配，给出明确的错误信息
    raise ValueError(
        f"无法在 price_df 中找到资产 '{asset_code}' 的 '{price_field}' 价格列。\n"
        f"请检查 config.yaml 的 asset.code 和 asset.price_field 设置。\n"
        f"price_df 现有列（前20个）：{list(price_df.columns[:20])}"
    )


def select_signals(signal_df: pd.DataFrame, cfg_signals: dict) -> pd.DataFrame:
    """
    根据配置选择信号列。

    若 use_all=true，返回全部列；否则返回 selected 列表中的列。
    """
    if cfg_signals.get("use_all", False):
        return signal_df

    selected = cfg_signals.get("selected", [])
    if not selected:
        raise ValueError("config.yaml 中 signals.use_all=false 但 signals.selected 为空，"
                         "请至少指定一个信号名称。")

    missing = [s for s in selected if s not in signal_df.columns]
    if missing:
        raise ValueError(
            f"以下信号在 signal.csv 中不存在：{missing}\n"
            f"signal.csv 现有列：{list(signal_df.columns)}"
        )
    return signal_df[selected]


# ══════════════════════════════════════════════════════════════════════════════
# Excel 报告生成器
# ══════════════════════════════════════════════════════════════════════════════

class ExcelReporter:
    """
    将择时因子评估与回测结果写入格式化 Excel 报告。

    Sheet 布局
    ----------
    1. 综合概览     : 全信号汇总对比表
    2. IC相关性分析  : 多周期 IC / ICIR 详细数据
    3. 信号检验结果  : 三种方法的胜率与盈亏指标
    4. 回归分析     : OLS 系数、T值、P值、R²
    5. 稳健性检验   : 样本内外、市场状态分析
    6. 回测绩效汇总  : 策略 vs 基准绩效指标对比
    """

    # 颜色方案
    _C_HEADER   = "1565C0"  # 深蓝（表头背景）
    _C_SUBHEAD  = "E3F2FD"  # 浅蓝（次级表头）
    _C_POS      = "C8E6C9"  # 浅绿（正值高亮）
    _C_NEG      = "FFCDD2"  # 浅红（负值高亮）
    _C_ROW_ALT  = "F8F9FA"  # 交替行底色
    _C_TITLE    = "0D47A1"  # 标题字体颜色

    def __init__(self) -> None:
        self.wb = Workbook()
        self.wb.remove(self.wb.active)  # 删除默认空表

    # ── 样式辅助 ─────────────────────────────────────────────────────────────

    def _hfill(self, hex_color: str) -> PatternFill:
        return PatternFill("solid", fgColor=hex_color)

    def _border(self, style: str = "thin") -> Border:
        s = Side(style=style)
        return Border(left=s, right=s, top=s, bottom=s)

    def _header_font(self, bold: bool = True, color: str = "FFFFFF",
                     size: int = 10) -> Font:
        return Font(bold=bold, color=color, name="微软雅黑", size=size)

    def _data_font(self, bold: bool = False, size: int = 10) -> Font:
        return Font(bold=bold, name="微软雅黑", size=size)

    def _center(self) -> Alignment:
        return Alignment(horizontal="center", vertical="center", wrap_text=True)

    def _write_header_row(self, ws, row: int, cols: List[str],
                           col_start: int = 1) -> None:
        """写入带样式的表头行。"""
        for j, text in enumerate(cols, start=col_start):
            c = ws.cell(row=row, column=j, value=text)
            c.fill      = self._hfill(self._C_HEADER)
            c.font      = self._header_font()
            c.alignment = self._center()
            c.border    = self._border()

    def _write_data_row(self, ws, row: int, values: list,
                        col_start: int = 1, alt: bool = False) -> None:
        """写入数据行，可选交替底色。"""
        fill = self._hfill(self._C_ROW_ALT) if alt else None
        for j, v in enumerate(values, start=col_start):
            c = ws.cell(row=row, column=j, value=v)
            c.font      = self._data_font()
            c.alignment = self._center()
            c.border    = self._border("thin")
            if fill:
                c.fill = fill

    def _auto_width(self, ws, min_w: int = 8, max_w: int = 25) -> None:
        """自动调整列宽（基于内容长度）。"""
        for col in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col
            )
            ws.column_dimensions[get_column_letter(col[0].column)].width = (
                min(max_w, max(min_w, max_len + 2))
            )

    def _pct(self, v) -> str:
        """将浮点数格式化为百分比字符串，None/NaN 显示为 N/A。"""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.2%}"

    def _f4(self, v) -> str:
        """浮点数 4 位小数，None/NaN 显示 N/A。"""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.4f}"

    def _color_cell(self, cell, value, threshold: float = 0.0,
                    reverse: bool = False) -> None:
        """根据正负值为单元格着色（正→绿，负→红；reverse=True 时相反）。"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        is_good = value > threshold if not reverse else value < threshold
        cell.fill = self._hfill(self._C_POS if is_good else self._C_NEG)

    # ── Sheet 1：综合概览 ──────────────────────────────────────────────────

    def _sheet_overview(self,
                         results: List[Tuple],
                         asset_name: str,
                         cfg: dict) -> None:
        ws = self.wb.create_sheet("综合概览")
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A3"

        # 大标题
        ws.merge_cells("A1:N1")
        title_cell = ws["A1"]
        title_cell.value = f"择时因子评价报告  |  资产：{asset_name}"
        title_cell.font      = Font(bold=True, size=14, color=self._C_TITLE, name="微软雅黑")
        title_cell.alignment = self._center()
        title_cell.fill      = self._hfill("EBF5FB")
        ws.row_dimensions[1].height = 28

        headers = [
            "信号名称", "IC均值(1日)", "ICIR(1日)", "IC正值占比",
            "胜率(阈值法)", "胜率(均线法)", "胜率(极值法)",
            "β系数", "β显著性", "R²",
            "样本外稳健", "IC衰减幅度",
            "综合评分", "评级",
        ]
        self._write_header_row(ws, row=2, cols=headers)
        ws.row_dimensions[2].height = 22

        for idx, (sig_name, ev, bt_res) in enumerate(results):
            row = idx + 3
            alt = (idx % 2 == 1)

            # 取各指标
            ic1 = (ev._ic_results or {}).get(ev.forward_period)
            sig1 = (ev._signal_results or {}).get("threshold")
            sig2 = (ev._signal_results or {}).get("moving_average")
            sig3 = (ev._signal_results or {}).get("percentile")
            reg  = ev._reg_result
            rob  = ev._robustness_result
            sc   = ev.score()

            ic_mean   = ic1.ic_mean           if ic1  else None
            icir      = ic1.icir              if ic1  else None
            ic_pos_r  = ic1.ic_positive_ratio if ic1  else None
            wr1 = sig1.overall_win_rate if sig1 else None
            wr2 = sig2.overall_win_rate if sig2 else None
            wr3 = sig3.overall_win_rate if sig3 else None
            beta      = reg.beta       if reg  else None
            beta_sig  = "显著" if (reg and reg.is_significant) else "不显著"
            r2        = reg.r_squared  if reg  else None
            is_robust = "稳健" if (rob and rob.is_robust) else "不稳健"
            ic_deg    = rob.ic_degradation if rob else None

            values = [
                sig_name,
                self._f4(ic_mean),  self._f4(icir),   self._pct(ic_pos_r),
                self._pct(wr1),     self._pct(wr2),   self._pct(wr3),
                self._f4(beta),     beta_sig,          self._f4(r2),
                is_robust,          self._pct(ic_deg),
                f"{sc.composite_score:.4f}",
                sc.grade,
            ]
            self._write_data_row(ws, row=row, values=values, alt=alt)

            # 着色：IC/ICIR/胜率/评分
            cells = [ws.cell(row, j+1) for j in range(len(values))]
            for ci, val in [(1, ic_mean), (2, icir)]:
                self._color_cell(cells[ci], val if val is not None else 0)
            for ci, val in [(4, wr1), (5, wr2), (6, wr3)]:
                self._color_cell(cells[ci], val - 0.5 if val is not None else None)
            # 评级列着色
            grade_colors = {"A": "B2EBF2", "B": "C8E6C9", "C": "FFF9C4",
                            "D": "FFE0B2", "F": "FFCDD2"}
            cells[13].fill = self._hfill(grade_colors.get(sc.grade, "FFFFFF"))

        self._auto_width(ws)

    # ── Sheet 2：IC相关性分析 ───────────────────────────────────────────────

    def _sheet_ic(self, results: List[Tuple]) -> None:
        ws = self.wb.create_sheet("IC相关性分析")
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A2"

        headers = [
            "信号名称", "预测周期", "IC方法",
            "IC均值", "IC标准差", "ICIR",
            "IC>0占比", "T统计量", "P值", "是否显著",
        ]
        self._write_header_row(ws, row=1, cols=headers)

        row = 2
        for sig_name, ev, _ in results:
            for period, ic_res in sorted((ev._ic_results or {}).items()):
                alt = (row % 2 == 0)
                values = [
                    sig_name, f"{period}日", ic_res.method.upper(),
                    self._f4(ic_res.ic_mean),
                    self._f4(ic_res.ic_std),
                    self._f4(ic_res.icir),
                    self._pct(ic_res.ic_positive_ratio),
                    self._f4(ic_res.t_statistic),
                    self._f4(ic_res.p_value),
                    "✓ 显著" if ic_res.is_significant else "✗ 不显著",
                ]
                self._write_data_row(ws, row=row, values=values, alt=alt)
                # ICIR 着色
                c = ws.cell(row, 6)
                self._color_cell(c, ic_res.icir if not np.isnan(ic_res.icir) else 0)
                row += 1

        self._auto_width(ws)

    # ── Sheet 3：信号检验结果 ───────────────────────────────────────────────

    def _sheet_signal(self, results: List[Tuple]) -> None:
        ws = self.wb.create_sheet("信号检验结果")
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A2"

        headers = [
            "信号名称", "检验方法",
            "多头样本数", "多头胜率", "多头均收益", "多头盈亏比",
            "空头样本数", "空头胜率", "空头均收益", "空头盈亏比",
            "综合胜率", "多空收益价差", "T统计量", "P值", "是否显著",
        ]
        self._write_header_row(ws, row=1, cols=headers)

        method_label = {
            "threshold":     "阈值法",
            "moving_average":"均线法",
            "percentile":    "极值法",
        }

        row = 2
        for sig_name, ev, _ in results:
            for method, res in (ev._signal_results or {}).items():
                alt = (row % 2 == 0)
                values = [
                    sig_name, method_label.get(method, method),
                    res.n_long,
                    self._pct(res.long_win_rate),
                    self._f4(res.long_avg_return),
                    self._f4(res.long_pl_ratio),
                    res.n_short,
                    self._pct(res.short_win_rate),
                    self._f4(res.short_avg_return),
                    self._f4(getattr(res, "short_pl_ratio", None)),
                    self._pct(res.overall_win_rate),
                    self._f4(res.long_short_return_spread),
                    self._f4(res.t_statistic),
                    self._f4(res.p_value),
                    "✓ 显著" if res.is_significant else "✗ 不显著",
                ]
                self._write_data_row(ws, row=row, values=values, alt=alt)
                # 胜率着色
                self._color_cell(ws.cell(row, 4),
                                  res.long_win_rate - 0.5)
                self._color_cell(ws.cell(row, 11),
                                  res.overall_win_rate - 0.5)
                row += 1

        self._auto_width(ws)

    # ── Sheet 4：回归分析 ──────────────────────────────────────────────────

    def _sheet_regression(self, results: List[Tuple]) -> None:
        ws = self.wb.create_sheet("回归分析")
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A2"

        headers = [
            "信号名称",
            "Alpha", "Beta",
            "Alpha-T值", "Beta-T值",
            "Alpha-P值", "Beta-P值",
            "R²", "调整R²",
            "F统计量", "F-P值",
            "样本量", "β是否显著",
        ]
        self._write_header_row(ws, row=1, cols=headers)

        for idx, (sig_name, ev, _) in enumerate(results):
            reg  = ev._reg_result
            alt  = (idx % 2 == 1)
            if reg is None:
                values = [sig_name] + ["N/A"] * 12
            else:
                values = [
                    sig_name,
                    self._f4(reg.alpha),      self._f4(reg.beta),
                    self._f4(reg.alpha_tstat), self._f4(reg.beta_tstat),
                    self._f4(reg.alpha_pvalue),self._f4(reg.beta_pvalue),
                    self._f4(reg.r_squared),   self._f4(reg.adj_r_squared),
                    self._f4(reg.f_statistic),
                    self._f4(getattr(reg, "f_pvalue", None)),
                    reg.n_observations,
                    "✓ 显著" if reg.is_significant else "✗ 不显著",
                ]
            self._write_data_row(ws, row=idx + 2, values=values, alt=alt)
            if reg:
                self._color_cell(ws.cell(idx + 2, 3), reg.beta)   # beta 着色
                self._color_cell(ws.cell(idx + 2, 8),
                                  reg.r_squared - 0.01)             # R² 着色

        self._auto_width(ws)

    # ── Sheet 5：稳健性检验 ────────────────────────────────────────────────

    def _sheet_robustness(self, results: List[Tuple]) -> None:
        ws = self.wb.create_sheet("稳健性检验")
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A2"

        headers = [
            "信号名称",
            "样本分割日期",
            "样本内IC均值", "样本内ICIR",
            "样本外IC均值", "样本外ICIR",
            "IC衰减幅度",   "是否稳健",
            "牛市IC均值",   "牛市ICIR",
            "熊市IC均值",   "熊市ICIR",
            "震荡市IC均值", "震荡市ICIR",
        ]
        self._write_header_row(ws, row=1, cols=headers)

        for idx, (sig_name, ev, _) in enumerate(results):
            rob   = ev._robustness_result
            reg   = ev._regime_results or {}
            alt   = (idx % 2 == 1)

            def _regime_val(key: str, attr: str):
                r = reg.get(key)
                val = getattr(r, attr, None) if r else None
                return self._f4(val)

            if rob is None:
                values = [sig_name] + ["N/A"] * 13
            else:
                values = [
                    sig_name,
                    str(rob.split_date)[:10] if rob.split_date else "N/A",
                    self._f4(rob.insample_ic),   self._f4(rob.insample_icir),
                    self._f4(rob.outsample_ic),  self._f4(rob.outsample_icir),
                    self._pct(rob.ic_degradation),
                    "✓ 稳健" if rob.is_robust else "✗ 不稳健",
                    _regime_val("bull",     "ic_mean"),
                    _regime_val("bull",     "icir"),
                    _regime_val("bear",     "ic_mean"),
                    _regime_val("bear",     "icir"),
                    _regime_val("sideways", "ic_mean"),
                    _regime_val("sideways", "icir"),
                ]
            self._write_data_row(ws, row=idx + 2, values=values, alt=alt)
            if rob:
                # 稳健性列着色
                self._color_cell(ws.cell(idx + 2, 8),
                                  1.0 if rob.is_robust else -1.0)

        self._auto_width(ws)

    # ── Sheet 6：回测绩效汇总 ─────────────────────────────────────────────

    def _sheet_backtest(self, results: List[Tuple], benchmark_ratio: float) -> None:
        ws = self.wb.create_sheet("回测绩效汇总")
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A2"

        headers = [
            "信号名称",
            "总收益率(策略)", "年化收益(策略)", "年化波动(策略)",
            "Sharpe(策略)",  "最大回撤(策略)", "Calmar(策略)",
            "日胜率(策略)",  "交易次数",
            "总收益率(基准)", "年化收益(基准)", "Sharpe(基准)",
            "最大回撤(基准)",
            "超额年化收益",   "超额Sharpe",
        ]
        self._write_header_row(ws, row=1, cols=headers)

        for idx, (sig_name, _, bt_res) in enumerate(results):
            alt  = (idx % 2 == 1)
            ms   = bt_res.metrics_strategy
            mb   = bt_res.metrics_benchmark

            excess_annual = ms["annual_return"] - mb["annual_return"]
            excess_sharpe = ms["sharpe"] - mb["sharpe"]

            values = [
                sig_name,
                self._pct(ms["total_return"]),
                self._pct(ms["annual_return"]),
                self._pct(ms["annual_vol"]),
                self._f4(ms["sharpe"]),
                self._pct(ms["max_drawdown"]),
                self._f4(ms["calmar"]),
                self._pct(ms["win_rate"]),
                ms.get("n_trades", 0),
                self._pct(mb["total_return"]),
                self._pct(mb["annual_return"]),
                self._f4(mb["sharpe"]),
                self._pct(mb["max_drawdown"]),
                self._pct(excess_annual),
                self._f4(excess_sharpe),
            ]
            self._write_data_row(ws, row=idx + 2, values=values, alt=alt)

            r = idx + 2
            # 策略总收益着色
            self._color_cell(ws.cell(r, 2),  ms["total_return"])
            # Sharpe 着色
            self._color_cell(ws.cell(r, 5),  ms["sharpe"])
            # 最大回撤着色（越小越好）
            self._color_cell(ws.cell(r, 6),  ms["max_drawdown"], reverse=True)
            # 超额收益着色
            self._color_cell(ws.cell(r, 14), excess_annual)

        self._auto_width(ws)

    # ── 统一入口 ─────────────────────────────────────────────────────────────

    def write(self,
              results: List[Tuple],
              asset_name: str,
              cfg: dict,
              save_path: str | Path) -> None:
        """
        生成完整 Excel 报告并保存。

        参数
        ----
        results   : [(sig_name, evaluator, BacktestResult), ...]
        asset_name: 资产显示名称
        cfg       : 完整配置字典
        save_path : 输出 Excel 文件路径
        """
        self._sheet_overview(results, asset_name, cfg)
        self._sheet_ic(results)
        self._sheet_signal(results)
        self._sheet_regression(results)
        self._sheet_robustness(results)
        self._sheet_backtest(
            results,
            benchmark_ratio=cfg.get("backtest", {}).get("benchmark_asset_ratio", 0.5),
        )

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.wb.save(save_path)
        print(f"\n  ✓ Excel报告已保存：{save_path.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# 多因子横向对比图（复用 example 中的逻辑）
# ══════════════════════════════════════════════════════════════════════════════

def plot_multi_factor_comparison(results: List[Tuple], save_path: Path) -> None:
    """生成四子图多因子横向对比图并保存。"""
    if len(results) < 2:
        return  # 单因子无需对比图

    names  = [r[0] for r in results]
    evs    = {r[0]: r[1] for r in results}
    scores = {r[0]: r[1].score() for r in results}
    n      = len(names)
    colors = plt.cm.tab10(np.linspace(0, 0.8, n))
    periods = [1, 5, 10, 20]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("择时信号横向对比", fontsize=14, fontweight="bold")

    # 子图A：各预测周期IC均值
    ax = axes[0, 0]
    for i, (name, ev) in enumerate(evs.items()):
        ic_means = [
            (ev._ic_results[p].ic_mean if ev._ic_results and p in ev._ic_results else 0)
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

    # 子图B：1日ICIR
    ax = axes[0, 1]
    icirs = []
    for name in names:
        ev = evs[name]
        ic1 = (ev._ic_results or {}).get(1)
        icir = ic1.icir if ic1 and not np.isnan(ic1.icir) else 0.0
        icirs.append(icir)
    bars = ax.bar(range(n), icirs,
                  color=[colors[i] for i in range(n)], alpha=0.85, edgecolor="black")
    ax.axhline(0,   color="black", linewidth=0.5)
    ax.axhline(0.5, color="#43A047", linewidth=1.2, linestyle="--",
               alpha=0.8, label="ICIR=0.5（良好）")
    ax.axhline(-0.5, color="#E53935", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_title("ICIR（1日预测）", fontsize=11)
    ax.set_ylabel("ICIR")
    ax.legend(fontsize=8)

    # 子图C：三种方法胜率
    ax = axes[1, 0]
    methods = ["threshold", "moving_average", "percentile"]
    mlabels = ["阈值法", "均线法", "极值法"]
    x = np.arange(n)
    w = 0.25
    for j, (m, ml) in enumerate(zip(methods, mlabels)):
        wrs = []
        for name in names:
            ev = evs[name]
            res = (ev._signal_results or {}).get(m)
            wrs.append(res.overall_win_rate * 100 if res else 50.0)
        ax.bar(x + j * w, wrs, w, label=ml, alpha=0.85)
    ax.axhline(50, color="#E53935", linewidth=1.2, linestyle="--", label="50%基准")
    ax.set_xticks(x + w)
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_title("各方法胜率对比", fontsize=11)
    ax.set_ylabel("胜率（%）")
    ax.set_ylim(35, 75)
    ax.legend(fontsize=8)

    # 子图D：综合评分
    ax = axes[1, 1]
    composites = [scores[n].composite_score for n in names]
    bars = ax.bar(range(len(names)), composites,
                  color=[colors[i] for i in range(n)], alpha=0.85, edgecolor="black")
    ax.axhline(0.60, color="#43A047", linewidth=1.2, linestyle="--", label="B级（0.60）")
    ax.axhline(0.75, color="#1565C0", linewidth=1.2, linestyle="--", label="A级（0.75）")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_title("综合评分与评级", fontsize=11)
    ax.set_ylabel("评分 [0-1]")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8)
    for i, (bar, sc, name) in enumerate(zip(bars, composites, names)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{scores[name].grade}级",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 多因子对比图：{save_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ROOT = Path(__file__).parent

    # ── 加载配置 ─────────────────────────────────────────────────────────────
    cfg = load_config(ROOT / "config.yaml")

    asset_cfg  = cfg.get("asset",      {})
    sig_cfg    = cfg.get("signals",    {})
    eval_cfg   = cfg.get("evaluation", {})
    bt_cfg     = cfg.get("backtest",   {})
    out_cfg    = cfg.get("output",     {})

    asset_name  = asset_cfg.get("name",        "未知资产")
    asset_code  = asset_cfg.get("code",        "")
    price_field = asset_cfg.get("price_field", "CLOSE")

    workspace        = ROOT / out_cfg.get("workspace_dir", "workspace")
    eval_subdir      = out_cfg.get("eval_plots_subdir",    "plots/eval")
    bt_subdir        = out_cfg.get("backtest_plots_subdir", "plots/backtest")
    excel_filename   = out_cfg.get("excel_filename",       "timing_report.xlsx")
    dpi              = int(out_cfg.get("dpi", 100))

    workspace.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  择时因子评价框架  |  主程序启动")
    print(f"  资产：{asset_name}（{asset_code}）")
    print("=" * 68)

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    print("\n[Step 1] 加载数据...")
    data_cfg    = cfg.get("data", {})
    price_file  = data_cfg.get("price_file",  "price_df.csv")
    signal_file = data_cfg.get("signal_file", "signal.csv")
    price_df, signal_df = load_data(ROOT / "data_ini", price_file, signal_file)

    prices  = extract_price_series(price_df, asset_code, price_field)
    returns = prices.pct_change().dropna()

    # ── 日期范围过滤 ─────────────────────────────────────────────────────────
    start_date = data_cfg.get("start_date", "") or None
    end_date   = data_cfg.get("end_date",   "") or None
    if start_date or end_date:
        prices  = prices.loc[start_date:end_date]
        returns = returns.loc[start_date:end_date]
        sig_df_raw = signal_df.loc[start_date:end_date]
        date_info = f"{start_date or '起始'} → {end_date or '末尾'}"
        print(f"  日期过滤  ：{date_info}")
    else:
        sig_df_raw = signal_df

    sig_selected = select_signals(sig_df_raw, sig_cfg)
    signal_names = list(sig_selected.columns)

    print(f"  价格序列  ：{prices.index[0].date()} → {prices.index[-1].date()}"
          f"（共 {len(prices)} 条）")
    print(f"  选用信号  ：{signal_names}")

    # ── 逐信号评估与回测 ─────────────────────────────────────────────────────
    print("\n[Step 2] 逐信号评估与回测...")
    backtester = Backtester(bt_cfg)
    all_results: List[Tuple] = []   # [(sig_name, evaluator, BacktestResult)]

    for sig_name in signal_names:
        print(f"\n  ── {sig_name} {'─' * (52 - len(sig_name))}")
        raw_signal = sig_selected[sig_name].dropna()

        # 每个信号独立输出目录：workspace/{safe_name}/
        safe_name  = sig_name.replace("/", "_").replace(" ", "_").replace("\\", "_")
        sig_dir    = workspace / safe_name
        eval_dir   = sig_dir / eval_subdir
        bt_dir     = sig_dir / bt_subdir
        excel_path = sig_dir / excel_filename
        eval_dir.mkdir(parents=True, exist_ok=True)
        bt_dir.mkdir(parents=True, exist_ok=True)

        # ── 因子评估 ──────────────────────────────────────────────────────
        ev = TimingFactorEvaluator(
            factor_name   = sig_name,
            forward_period= int(eval_cfg.get("forward_period", 1)),
            ic_method     = eval_cfg.get("ic_method", "pearson"),
        )
        ev.evaluate(
            factor                = raw_signal,
            returns               = returns,
            prices                = prices,
            preprocess            = bool(eval_cfg.get("preprocess", True)),
            rolling_window        = int(eval_cfg.get("rolling_window", 252)),
            run_robustness        = bool(eval_cfg.get("run_robustness", True)),
            run_rolling_regression= bool(eval_cfg.get("run_rolling_regression", False)),
            ic_periods            = eval_cfg.get("ic_periods", [1, 5, 10, 20]),
        )

        # 打印评估报告（控制台）
        ev.report()

        # 保存评估图表
        fig = ev.plot(figsize=(16, 13))
        eval_plot_path = eval_dir / f"{safe_name}_evaluation.png"
        fig.savefig(eval_plot_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ 评估图：{eval_plot_path.relative_to(workspace)}")

        # ── 回测 ──────────────────────────────────────────────────────────
        factor_for_bt = ev._factor if ev._factor is not None else raw_signal
        bt_res = backtester.run(
            signal      = factor_for_bt,
            prices      = prices,
            signal_name = sig_name,
            save_dir    = bt_dir,
        )
        print(bt_res.summary())

        # ── 单信号 Excel 报告 ──────────────────────────────────────────────
        ExcelReporter().write(
            results    = [(sig_name, ev, bt_res)],
            asset_name = asset_name,
            cfg        = cfg,
            save_path  = excel_path,
        )

        all_results.append((sig_name, ev, bt_res))

    # ── 多因子对比图（存放在 workspace 根目录）────────────────────────────────
    print("\n[Step 3] 生成多因子对比图...")
    plot_multi_factor_comparison(
        results   = all_results,
        save_path = workspace / "factor_comparison.png",
    )

    # ── 汇总输出 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  全部完成！输出目录结构：")
    print(f"  {workspace.relative_to(ROOT)}/")
    for sig_name, ev, bt_res in all_results:
        safe = sig_name.replace("/", "_").replace(" ", "_").replace("\\", "_")
        print(f"  ├── {safe}/")
        print(f"  │   ├── {eval_subdir}/   （评估图）")
        print(f"  │   ├── {bt_subdir}/  （回测图）")
        print(f"  │   └── {excel_filename}")
    print(f"  └── factor_comparison.png   （多信号对比图）")
    print()
    print("  各信号综合评级：")
    for sig_name, ev, bt_res in all_results:
        sc = ev.score()
        ms = bt_res.metrics_strategy
        print(f"    {sig_name:<20}  评级={sc.grade}  "
              f"综合={sc.composite_score:.3f}  "
              f"年化={ms['annual_return']:+.2%}  "
              f"SR={ms['sharpe']:.3f}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
