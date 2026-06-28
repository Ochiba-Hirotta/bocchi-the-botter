"""Aggregate four physical metrics from trade series.

純粋関数 + frozen dataclass で構成し、I/O や副作用を持たない. テストは
``code/tests/test_physical_metrics.py`` に集約する.

記事 #7 の前提:
    L1: Expectancy は ``ReturnPct`` ベース (PnL ではない).
        分母 spread と単位を揃えるため.
    L2: 平均保有 bars は ``median`` (mean ではない).
        MAX_BARS=48 への張り付き検知に向く.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from statistics import median
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhysicalMetrics:
    """1 fold もしくは fold 横断集約の 4 指標 + トレード数.

    全 4 指標が NaN のまま ``n_trades=0`` を返すケース (空 trades) を
    エッジケースとして許容する.
    """

    expectancy_spread_ratio: float
    win_rate: float
    median_holding_bars: float
    profit_loss_ratio: float
    n_trades: int


def compute_physical_metrics(
    trades: pd.DataFrame, spread: float
) -> PhysicalMetrics:
    """1 fold 分の trades dataframe から 4 指標を計算する純粋関数.

    Args:
        trades: backtesting.py ``stats._trades`` 由来の DataFrame.
            必須列: ``ReturnPct``, ``EntryBar``, ``ExitBar``.
        spread: 片道相対値 spread (#5 α §7.1 確定値).

    Returns:
        ``PhysicalMetrics``. 空 trades は全 NaN + ``n_trades=0``.

    Raises:
        ValueError: ``spread <= 0``. 片道相対値 spread の前提を破った扱い.
    """
    if not (spread > 0):
        raise ValueError(f"spread must be positive, got {spread!r}")

    n_trades = len(trades)
    if n_trades == 0:
        return PhysicalMetrics(
            expectancy_spread_ratio=float("nan"),
            win_rate=float("nan"),
            median_holding_bars=float("nan"),
            profit_loss_ratio=float("nan"),
            n_trades=0,
        )

    ret = trades["ReturnPct"]
    pos = ret[ret > 0]
    neg = ret[ret < 0]

    expectancy_spread_ratio = float(ret.mean()) / spread
    win_rate = float((ret > 0).sum()) / n_trades

    holding = (trades["ExitBar"] - trades["EntryBar"]).astype(float)
    median_holding_bars = float(holding.median())

    if len(pos) == 0 or len(neg) == 0:
        profit_loss_ratio = float("nan")
    else:
        profit_loss_ratio = float(pos.mean()) / abs(float(neg.mean()))

    return PhysicalMetrics(
        expectancy_spread_ratio=expectancy_spread_ratio,
        win_rate=win_rate,
        median_holding_bars=median_holding_bars,
        profit_loss_ratio=profit_loss_ratio,
        n_trades=n_trades,
    )


def aggregate_per_grid(per_fold: list[PhysicalMetrics]) -> PhysicalMetrics:
    """fold 横断で 4 指標を中央値集約する. ``n_trades`` のみ合計.

    NaN を含む fold は ``np.nanmedian`` 同等の skipna で除外する.
    全 fold が NaN の指標は NaN のまま返す.

    Args:
        per_fold: 1 grid 分の per-fold ``PhysicalMetrics`` のリスト.

    Returns:
        fold 横断中央値で集約した ``PhysicalMetrics``.
        per_fold が空リストなら全 NaN + ``n_trades=0``.
    """
    if not per_fold:
        return PhysicalMetrics(
            expectancy_spread_ratio=float("nan"),
            win_rate=float("nan"),
            median_holding_bars=float("nan"),
            profit_loss_ratio=float("nan"),
            n_trades=0,
        )

    return PhysicalMetrics(
        expectancy_spread_ratio=_nanmedian(
            [m.expectancy_spread_ratio for m in per_fold]
        ),
        win_rate=_nanmedian([m.win_rate for m in per_fold]),
        median_holding_bars=_nanmedian([m.median_holding_bars for m in per_fold]),
        profit_loss_ratio=_nanmedian([m.profit_loss_ratio for m in per_fold]),
        n_trades=sum(m.n_trades for m in per_fold),
    )


def metrics_to_dict(m: PhysicalMetrics) -> dict[str, Any]:
    """``PhysicalMetrics`` を CSV 行用 dict に変換する.

    ``dataclasses.asdict`` の薄ラッパだが、列順を frozen dataclass の
    定義順に固定するために本書専用に置く.
    """
    return {f.name: getattr(m, f.name) for f in fields(m)}


def _nanmedian(values: list[float]) -> float:
    """NaN を除外した中央値. 全 NaN なら NaN.

    ``numpy.nanmedian`` の薄ラッパだが、空リスト判定も含めて取り扱う.
    """
    cleaned = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    if not cleaned:
        return float("nan")
    return float(median(cleaned))
