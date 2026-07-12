"""Bollinger Bands Mean Reversion (BB-MR) strategy.

判定: バー確定時（Close）. 約定: 次バー Open（``trade_on_close=False``）.

エントリー（遷移判定）:
    - LONG:  Close が Lower Band を下抜け  → 次バー Open で成行買
    - SHORT: Close が Upper Band を上抜け  → 次バー Open で成行売

イグジット（優先順）:
    1. SL (``SL_ATR_MULT × ATR``) — backtesting.py が同バー内で自動判定
    2. TP (``TP_ATR_MULT × ATR``) — 同上
    3. SMA タッチ (Low ≤ SMA ≤ High) — 次バー Open で成行クローズ
    4. MAX_BARS 超過 — 次バー Open で成行クローズ

ポジションサイジング:
    ``units = min(units_risk, units_margin_cap)``. ただし
    ``units_risk > units_margin_cap`` の場合は機会見送り（``missed_entries`` 加算）.

SL/TP 基準値（記事時点の許容誤差）:
    SL/TP はシグナル確定バーの Close（``close_now``）を基準に絶対価格で固定する.
    実約定は次バー Open のため、ギャップ発生時は想定リスク（RISK_PCT=1%）と
    実効 R:R がズレる. 公開再現コードでは記事時点の条件に合わせてこの扱いを維持する.

抽象基底:
    `StrategyBase` を継承し ``MARGIN`` / ``MAX_BARS`` / ``ATR_N`` /
    ``SL_ATR_MULT`` / ``RISK_PCT`` は基底のデフォルト値をそのまま使う.
    BB-MR 固有の ``BB_N`` / ``BB_K`` / ``TP_ATR_MULT``
    のみ宣言する. 自前で close する経路（SMA タッチ / MAX_BARS）には
    ``_record_close_reason`` を呼んで集計する.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.atr import wilder_atr
from .base import StrategyBase
from .sizing import compute_units


def _bb_bands(close: np.ndarray, period: int, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sma, upper, lower) を返す. 先頭 ``period-1`` 本は NaN."""
    s = pd.Series(close)
    sma = s.rolling(period).mean().to_numpy()
    std = s.rolling(period).std(ddof=0).to_numpy()
    upper = sma + k * std
    lower = sma - k * std
    return sma, upper, lower


class BBMeanReversion(StrategyBase):
    """Bollinger Bands Mean Reversion strategy.

    ``MARGIN`` は ``Backtest(margin=...)`` と一致させる運用責任があることに注意.
    backtesting.py の Broker は public に margin を公開していないため, Strategy
    クラス変数として保持する方針を採る（§6.5 ``units_margin_cap`` 計算で参照）.
    本クラスは ``StrategyBase`` から ``MARGIN`` を継承する.
    """

    # BB-MR 固有のクラス変数のみ宣言. 共通項目 (MARGIN / MAX_BARS / ATR_N /
    # SL_ATR_MULT / RISK_PCT) は StrategyBase から継承する.
    BB_N: int = 20
    BB_K: float = 2.0
    TP_ATR_MULT: float = 2.0

    def init(self) -> None:
        super().init()  # _close_reasons を 4 区分で初期化

        close = np.asarray(self.data.Close, dtype=np.float64)
        high = np.asarray(self.data.High, dtype=np.float64)
        low = np.asarray(self.data.Low, dtype=np.float64)

        bb_n, bb_k = int(self.BB_N), float(self.BB_K)
        atr_n = int(self.ATR_N)

        self.sma = self.I(lambda: _bb_bands(close, bb_n, bb_k)[0], name="sma")
        self.upper = self.I(lambda: _bb_bands(close, bb_n, bb_k)[1], name="bb_upper")
        self.lower = self.I(lambda: _bb_bands(close, bb_n, bb_k)[2], name="bb_lower")
        self.atr = self.I(lambda: wilder_atr(high, low, close, atr_n), name="atr")

        self.missed_entries: int = 0
        self._spread: float = float(getattr(self._broker, "_spread", 0.0))

    def next(self) -> None:
        if self.position:
            self._try_exit()
            return

        atr_now = float(self.atr[-1])
        upper_now = float(self.upper[-1])
        lower_now = float(self.lower[-1])
        if np.isnan(atr_now) or np.isnan(upper_now) or np.isnan(lower_now):
            return
        if len(self.data) < 2:
            return
        upper_prev = float(self.upper[-2])
        lower_prev = float(self.lower[-2])
        if np.isnan(upper_prev) or np.isnan(lower_prev):
            return

        close_now = float(self.data.Close[-1])
        close_prev = float(self.data.Close[-2])

        long_signal = close_now < lower_now and close_prev >= lower_prev
        short_signal = close_now > upper_now and close_prev <= upper_prev
        if not (long_signal or short_signal):
            return

        units = self._compute_units(atr_now, close_now)
        if units <= 0:
            self.missed_entries += 1
            return

        stop_dist = self.SL_ATR_MULT * atr_now
        take_dist = self.TP_ATR_MULT * atr_now
        if long_signal:
            self.buy(
                size=units,
                sl=close_now - stop_dist,
                tp=close_now + take_dist,
            )
        else:
            self.sell(
                size=units,
                sl=close_now + stop_dist,
                tp=close_now - take_dist,
            )

    def _try_exit(self) -> None:
        """§6.4 3 / 4: SMA タッチと MAX_BARS 超過の判定.

        SMA タッチによるクローズは ``signal`` (平均回帰成功),
        MAX_BARS 超過は ``max_bars`` として ``_close_reasons`` に集計する.
        SL/TP は backtesting.py が自動判定するため Strategy 側からは記録不能
        （集計したい場合は事後 ``trade.exit_price`` ベースで別途実装）.
        """
        sma_now = float(self.sma[-1])
        high_now = float(self.data.High[-1])
        low_now = float(self.data.Low[-1])
        if not np.isnan(sma_now) and low_now <= sma_now <= high_now:
            self.position.close()
            self._record_close_reason("signal")
            return

        if self._check_max_bars_exit():
            self.position.close()
            self._record_close_reason("max_bars")

    def _compute_units(self, atr_now: float, price_now: float) -> int:
        return compute_units(
            equity=float(self.equity),
            atr=atr_now,
            price=price_now,
            risk_pct=self.RISK_PCT,
            sl_atr_mult=self.SL_ATR_MULT,
            margin=self.MARGIN,
            spread=self._spread,
        )
