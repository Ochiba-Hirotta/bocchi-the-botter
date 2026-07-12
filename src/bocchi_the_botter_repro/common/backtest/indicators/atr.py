"""Wilder RMA ATR implementation.

数式:
    TR_i  = max(H_i - L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|),  TR_0 = H_0 - L_0
    ATR_{N-1} = mean(TR_0 .. TR_{N-1})                          （初期値は SMA）
    ATR_i = ((N - 1) * ATR_{i-1} + TR_i) / N                    （Wilder RMA）
    ATR_0 .. ATR_{N-2} は NaN
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_atr(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    close: np.ndarray | pd.Series,
    period: int = 14,
) -> np.ndarray:
    """Wilder RMA による ATR を計算する.

    Args:
        high: 高値系列.
        low: 安値系列.
        close: 終値系列.
        period: 平滑化期間（正の整数）.

    Returns:
        入力と同じ長さの ``float64 np.ndarray``.
        先頭 ``period - 1`` 個は NaN、以降は Wilder RMA の値.

    Raises:
        ValueError: ``period`` が 1 未満、または入力長が不揃いな場合.
    """
    if period < 1:
        raise ValueError(f"period は 1 以上の整数が必要です: period={period}")

    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    if not (len(h) == len(l) == len(c)):
        raise ValueError(
            f"入力系列の長さが揃っていません: len(high)={len(h)}, "
            f"len(low)={len(l)}, len(close)={len(c)}"
        )

    n = len(h)
    atr = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return atr

    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    prev_c = c[:-1]
    cur_h = h[1:]
    cur_l = l[1:]
    tr[1:] = np.maximum.reduce([
        cur_h - cur_l,
        np.abs(cur_h - prev_c),
        np.abs(cur_l - prev_c),
    ])

    atr[period - 1] = tr[:period].mean()
    alpha_prev = period - 1
    for i in range(period, n):
        atr[i] = (alpha_prev * atr[i - 1] + tr[i]) / period

    return atr
