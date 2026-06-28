"""Donchian Channel Breakout strategy.

判定: バー確定時（Close）. 約定: 次バー Open（``trade_on_close=False``）.

エントリー（遷移判定）:
    - LONG:  ``close_t > rolling_high_{t-1}(N)`` かつ
             ``close_{t-1} <= rolling_high_{t-2}(N)``  → 次バー Open で成行買
    - SHORT: ``close_t < rolling_low_{t-1}(N)`` かつ
             ``close_{t-1} >= rolling_low_{t-2}(N)``   → 次バー Open で成行売

イグジット（優先順）:
    1. SL (``SL_ATR_MULT × ATR``) — backtesting.py が同バー内で自動判定
    2. 中央線基準利確 (signal) — 次バー Open で成行クローズ
        - LONG:  ``close_t <= midline_t - (DC_EXIT - 1.0) × ATR_t``
        - SHORT: ``close_t >= midline_t + (DC_EXIT - 1.0) × ATR_t``
        - midline = ``(rolling_high(N) + rolling_low(N)) / 2`` (shift 後)
    3. MAX_BARS 超過 — 次バー Open で成行クローズ

DC_EXIT の解釈 (§4.1 (A), §4.2):
    - 0.5 → ロング: ``midline + 0.5 × ATR`` (中央線到達前、早めに切る)
    - 1.0 → ``midline`` ちょうど (中央線到達で利確、アンカー値)
    - 1.5 → ロング: ``midline - 0.5 × ATR`` (中央線を 0.5×ATR 越えて深く待つ)
    - 2.0 → ロング: ``midline - 1.0 × ATR`` (中央線を 1.0×ATR 越えて深く待つ)

TP (固定):
    指定しない (順張り戦略のため明示的 TP は不要).
    エグジット (A) が動的トレーリングプロフィット相当として利確を担う.

ポジションサイジング:
    BB-MR と同じ ``compute_units(...)`` を流用. ``sl_distance = SL_ATR_MULT × ATR``.

SL 基準値（記事時点の許容誤差）:
    SL はシグナル確定バーの Close (``close_now``) を基準に絶対価格で固定する.
    実約定は次バー Open のため、ギャップ発生時は想定リスク (RISK_PCT=1%) と
    実効 R が想定値とズレる. 公開再現コードでは記事時点の条件に合わせてこの扱いを維持する.

抽象基底:
    ``StrategyBase`` を継承し ``MARGIN`` / ``MAX_BARS`` / ``ATR_N`` /
    ``SL_ATR_MULT`` / ``RISK_PCT`` は基底のデフォルト値をそのまま使う
    (BB-MR と同値で揃える).
    Donchian 固有の ``DC_N`` / ``DC_EXIT`` のみ宣言する.

``signal_close_ratio`` の意味（戦略間意味差）:
    Donchian では「トレンド成功」を表す (BB-MR の「平均回帰成功」と対称).
    エグジット (A) の中央線基準利確で発火し ``_record_close_reason("signal")``
    に集計される.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.atr import wilder_atr
from .base import StrategyBase
from .sizing import compute_units


def _donchian_bands(
    high: np.ndarray, low: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(upper, lower, midline) を返す. ``shift(1)`` 適用済 (現在バー除く直前 N バー).

    各時刻 ``t`` における値は ``[t-N, t-1]`` の高値最大 / 安値最小 / その中央.
    ルックアヘッド回避のため現在バー (時刻 t) を含めない. 先頭 ``period`` 本は NaN.
    """
    h = pd.Series(high)
    l = pd.Series(low)
    upper = h.rolling(period).max().shift(1).to_numpy()
    lower = l.rolling(period).min().shift(1).to_numpy()
    midline = (upper + lower) / 2
    return upper, lower, midline


class DonchianBreakout(StrategyBase):
    """Donchian Channel Breakout strategy.

    BB-MR とは対称的な順張り戦略. 共通のリスク管理パラメータ
    (``MARGIN`` / ``RISK_PCT`` / ``SL_ATR_MULT`` / ``MAX_BARS`` / ``ATR_N``)
    は ``StrategyBase`` から継承する. ``signal_close_ratio`` の意味は
    「トレンド成功」(§10.1).
    """

    # Donchian 固有のクラス変数のみ宣言. 共通項目 (MARGIN / MAX_BARS / ATR_N /
    # SL_ATR_MULT / RISK_PCT) は StrategyBase から継承する.
    DC_N: int = 20
    DC_EXIT: float = 1.0

    def init(self) -> None:
        super().init()  # _close_reasons を 4 区分で初期化

        close = np.asarray(self.data.Close, dtype=np.float64)
        high = np.asarray(self.data.High, dtype=np.float64)
        low = np.asarray(self.data.Low, dtype=np.float64)

        dc_n = int(self.DC_N)
        atr_n = int(self.ATR_N)

        self.upper = self.I(
            lambda: _donchian_bands(high, low, dc_n)[0], name="dc_upper"
        )
        self.lower = self.I(
            lambda: _donchian_bands(high, low, dc_n)[1], name="dc_lower"
        )
        self.midline = self.I(
            lambda: _donchian_bands(high, low, dc_n)[2], name="dc_midline"
        )
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

        long_signal = close_now > upper_now and close_prev <= upper_prev
        short_signal = close_now < lower_now and close_prev >= lower_prev
        if not (long_signal or short_signal):
            return

        units = self._compute_units(atr_now, close_now)
        if units <= 0:
            self.missed_entries += 1
            return

        stop_dist = self.SL_ATR_MULT * atr_now
        if long_signal:
            self.buy(size=units, sl=close_now - stop_dist)
        else:
            self.sell(size=units, sl=close_now + stop_dist)

    def _try_exit(self) -> None:
        """§4.1 中央線基準利確 / MAX_BARS 超過の判定.

        中央線基準利確 (signal): トレンド成功. ``position.close()`` の前に
        ``_record_close_reason("signal")`` を呼んでカウント漏れを防ぐ.
        MAX_BARS 超過: ``_record_close_reason("max_bars")``.
        SL は backtesting.py が自動判定するため Strategy 側からは記録不能
        (集計したい場合は事後 ``trade.exit_price`` ベースで別途実装).
        """
        midline_now = float(self.midline[-1])
        atr_now = float(self.atr[-1])
        if not (np.isnan(midline_now) or np.isnan(atr_now)):
            close_now = float(self.data.Close[-1])
            offset = (float(self.DC_EXIT) - 1.0) * atr_now
            position = self.position
            if position.is_long:
                if close_now <= midline_now - offset:
                    self._record_close_reason("signal")
                    position.close()
                    return
            else:
                if close_now >= midline_now + offset:
                    self._record_close_reason("signal")
                    position.close()
                    return

        if self._check_max_bars_exit():
            self._record_close_reason("max_bars")
            self.position.close()

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
