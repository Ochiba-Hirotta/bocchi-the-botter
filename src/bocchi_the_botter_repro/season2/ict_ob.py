"""ICT Order Block verification infrastructure through stage 7.

This module owns the parts that verification 1 and verification 2 must share
unchanged, plus both detectors completed in stages 4 and 5:

* the already-frozen M5-to-M15 derivation;
* NY 17:00 trading-day candles derived directly from M5 bid/ask rows;
* k-bar-delayed swing confirmation and daily structure bias;
* one-position, one-pending-zone limit-order simulation;
* a complete half-open lifecycle audit for every detector-produced zone;
* the frozen Month 04 (official-source) Order Block detector and daily target;
* the frozen secondary sweep -> MSS -> FVG detector and M15 target.

Detectors supply immutable ``PendingZone`` events and target resolvers; the
execution loop never reinterprets the source videos or changes detector
thresholds.
"""
from __future__ import annotations

import datetime as dt
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..common.backtest.strategies.sizing import compute_units
from .minute_data import M15Aggregation, aggregate_m5_to_m15
from .orb_m15 import (
    ARTICLE_WINDOW_END_EXCLUSIVE,
    ARTICLE_WINDOW_START,
    validate_m15_frame,
)


NEW_YORK = ZoneInfo("America/New_York")
DAILY_MIN_M5 = 216
SWING_K = 2
INITIAL_CASH = 1_000_000.0
RISK_PCT = 0.01
MARGIN = 0.04
OSS_PACKAGE = "smartmoneyconcepts"
OSS_PACKAGE_VERSION = "0.0.27"
OFFICIAL_OB_DETECTOR = "ict_month04"
OFFICIAL_OB_LOOKBACK = 20
OFFICIAL_TARGET_LOOKBACK_DAYS = 60
SECONDARY_OB_DETECTOR = "ict_secondary_17m"
SECONDARY_MSS_MAX_BARS = 12
SECONDARY_TARGET_LOOKBACK_BARS = 400
M15_SECONDS = 900
VALID_BIAS_STATES = frozenset({"bullish", "bearish", "neutral", "unavailable"})

Side = Literal["long", "short"]
SwingDirection = Literal["high", "low"]
BiasState = Literal["bullish", "bearish", "neutral", "unavailable"]
ZoneEndReason = Literal[
    "consumed",
    "invalidated",
    "replaced",
    "data_end",
    "outside_window",
]


class IctObError(RuntimeError):
    """Base error for the frozen ICT Order Block verification path."""


class MarketDataValidationError(ValueError):
    """Raised when M5 input violates the inherited data contract."""


@dataclass(slots=True)
class DailyAggregation:
    """Accepted NY-17 daily candles and an audit row for every observed day."""

    candles: pd.DataFrame
    day_status: pd.DataFrame


@dataclass(slots=True)
class TimeframeBundle:
    """Shared derived timeframes used by both detector implementations."""

    m15: pd.DataFrame
    incomplete_m15: pd.DataFrame
    daily: pd.DataFrame
    daily_status: pd.DataFrame


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """One strict fractal swing together with its no-lookahead availability."""

    direction: SwingDirection
    source_ts_utc: int
    confirmed_at_utc: int
    level: float


@dataclass(frozen=True, slots=True)
class BiasEvent:
    """One change in the daily structure bias timeline."""

    effective_at_utc: int
    state: BiasState
    reason: str


@dataclass(frozen=True, slots=True)
class PendingZone:
    """A detector-produced zone that becomes fillable at one M15 start."""

    zone_id: str
    detector: str
    side: Side
    active_from_ts_utc: int
    lower: float
    upper: float
    entry_price: float
    stop_loss: float
    signal_ts_utc: int | None = None


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """One filled position, retained at end-of-data when no exit is observed."""

    zone_id: str
    detector: str
    side: Side
    entry_time_utc: int
    entry_price: float
    stop_loss: float
    take_profit: float
    initial_risk: float
    units: int
    equity_before: float


@dataclass(frozen=True, slots=True)
class ZoneTrade:
    """One closed trade produced by the shared execution model."""

    zone_id: str
    detector: str
    side: Side
    entry_time_utc: int
    entry_price: float
    stop_loss: float
    take_profit: float
    initial_risk: float
    units: int
    exit_time_utc: int
    exit_price: float
    exit_reason: str
    pnl: float
    equity_before: float
    equity_after: float
    realized_r: float


@dataclass(frozen=True, slots=True)
class ZoneLifecycle:
    """One detector zone's half-open active interval in the shared executor."""

    zone_id: str
    detector: str
    side: Side
    active_from_ts_utc: int
    end_exclusive_ts_utc: int
    lower: float
    upper: float
    end_reason: ZoneEndReason


@dataclass(slots=True)
class ZoneBacktestResult:
    """Closed trades, terminal state, and deterministic lifecycle counters."""

    trades: list[ZoneTrade]
    final_equity: float
    open_position: OpenPosition | None
    pending_zone: PendingZone | None
    counters: dict[str, int]
    zone_lifecycles: tuple[ZoneLifecycle, ...] = ()


@dataclass(frozen=True, slots=True)
class OfficialObDetectionResult:
    """Official-source zones and deterministic candidate lifecycle counts."""

    zones: tuple[PendingZone, ...]
    counters: dict[str, int]


@dataclass(frozen=True, slots=True)
class SecondaryObDetectionResult:
    """Secondary-source zones and deterministic setup lifecycle counts."""

    zones: tuple[PendingZone, ...]
    counters: dict[str, int]


@dataclass(frozen=True, slots=True)
class _OfficialObCandidate:
    """One not-yet-activated Month 04 candidate candle."""

    side: Side
    source_ts_utc: int
    bid_open: float
    bid_high: float
    bid_low: float
    ask_high: float
    body_range: float


@dataclass(frozen=True, slots=True)
class _SecondarySetup:
    """One swept-liquidity event awaiting its directional MSS."""

    side: Side
    sweep_index: int
    sweep_ts_utc: int
    swept_level: float
    mss_level: float


@dataclass(frozen=True, slots=True)
class OfficialDailyTargetResolver:
    """Resolve Month 04 external liquidity without future daily knowledge."""

    day_start_ts_utc: tuple[int, ...]
    day_available_ts_utc: tuple[int, ...]
    swings: tuple[SwingPoint, ...]
    lookback_days: int = OFFICIAL_TARGET_LOOKBACK_DAYS

    def __call__(self, zone: PendingZone, entry_ts_utc: int) -> float | None:
        known_day_count = bisect_right(self.day_available_ts_utc, entry_ts_utc)
        first_day = max(0, known_day_count - self.lookback_days)
        recent_sources = set(
            self.day_start_ts_utc[first_day:known_day_count]
        )
        direction: SwingDirection = "high" if zone.side == "long" else "low"
        eligible = [
            swing
            for swing in self.swings
            if swing.direction == direction
            and swing.confirmed_at_utc <= entry_ts_utc
            and swing.source_ts_utc in recent_sources
            and (
                swing.level > zone.entry_price
                if zone.side == "long"
                else swing.level < zone.entry_price
            )
        ]
        if not eligible:
            return None
        newest = max(
            eligible,
            key=lambda swing: (swing.source_ts_utc, swing.confirmed_at_utc),
        )
        return newest.level


@dataclass(frozen=True, slots=True)
class SecondaryM15TargetResolver:
    """Resolve the latest directional M15 swing without future bars."""

    bar_start_ts_utc: tuple[int, ...]
    swings: tuple[SwingPoint, ...]
    lookback_bars: int = SECONDARY_TARGET_LOOKBACK_BARS

    def __call__(self, zone: PendingZone, entry_ts_utc: int) -> float | None:
        entry_index = bisect_left(self.bar_start_ts_utc, entry_ts_utc)
        if (
            entry_index >= len(self.bar_start_ts_utc)
            or self.bar_start_ts_utc[entry_index] != entry_ts_utc
        ):
            raise ValueError("secondary target entry is outside the M15 input")
        first_bar = max(0, entry_index - self.lookback_bars)
        recent_sources = set(self.bar_start_ts_utc[first_bar:entry_index])
        direction: SwingDirection = "high" if zone.side == "long" else "low"
        eligible = [
            swing
            for swing in self.swings
            if swing.direction == direction
            and swing.confirmed_at_utc <= entry_ts_utc
            and swing.source_ts_utc in recent_sources
            and (
                swing.level > zone.entry_price
                if zone.side == "long"
                else swing.level < zone.entry_price
            )
        ]
        if not eligible:
            return None
        newest = max(
            eligible,
            key=lambda swing: (swing.source_ts_utc, swing.confirmed_at_utc),
        )
        return newest.level


TargetResolver = Callable[[PendingZone, int], float | None]


_M5_REQUIRED_COLUMNS = {
    "source",
    "instrument",
    "granularity",
    "price",
    "ts_utc",
    "complete",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
}

_DAILY_COLUMNS = (
    "source",
    "instrument",
    "granularity",
    "price",
    "trading_date_et",
    "start_ts_utc",
    "available_ts_utc",
    "component_count",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
)

_DAY_STATUS_COLUMNS = (
    "trading_date_et",
    "start_ts_utc",
    "available_ts_utc",
    "component_count",
    "accepted",
    "reason",
)


def _validate_m5_for_daily(frame: pd.DataFrame) -> None:
    """Validate the M5 subset needed by the NY daily aggregation."""

    if frame.empty:
        raise MarketDataValidationError("M5 input is empty")
    missing = _M5_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise MarketDataValidationError(
            f"M5 input is missing columns: {sorted(missing)}"
        )
    required = sorted(_M5_REQUIRED_COLUMNS)
    if frame[required].isna().any(axis=None):
        raise MarketDataValidationError("M5 input contains NULL contract values")
    if not frame["ts_utc"].is_monotonic_increasing:
        raise MarketDataValidationError("M5 input is not sorted by ts_utc")
    if frame["ts_utc"].duplicated().any():
        raise MarketDataValidationError("M5 input contains duplicate timestamps")
    if (frame["ts_utc"].astype("int64") % 300 != 0).any():
        raise MarketDataValidationError("M5 input contains off-boundary timestamps")
    if not (frame["source"] == "oanda_rest_v20").all():
        raise MarketDataValidationError("M5 input has an unexpected source")
    if not (frame["instrument"] == "USD_JPY").all():
        raise MarketDataValidationError("M5 input has an unexpected instrument")
    if not (frame["granularity"] == "M5").all():
        raise MarketDataValidationError("daily aggregation accepts only M5 input")
    if not (frame["price"] == "BA").all():
        raise MarketDataValidationError("daily aggregation accepts only BA input")
    if not (frame["complete"] == 1).all():
        raise MarketDataValidationError("daily aggregation accepts only complete M5")

    price_columns = [
        f"{side}_{field}"
        for side in ("bid", "ask")
        for field in ("open", "high", "low", "close")
    ]
    prices = frame[price_columns].to_numpy(dtype=float)
    if not np.isfinite(prices).all():
        raise MarketDataValidationError("M5 input contains non-finite prices")
    for side in ("bid", "ask"):
        opened = frame[f"{side}_open"]
        high = frame[f"{side}_high"]
        low = frame[f"{side}_low"]
        closed = frame[f"{side}_close"]
        valid = (
            (low <= opened)
            & (opened <= high)
            & (low <= closed)
            & (closed <= high)
        )
        if (~valid).any():
            raise MarketDataValidationError(f"M5 input contains invalid {side} OHLC")
    if (frame["ask_open"] < frame["bid_open"]).any() or (
        frame["ask_close"] < frame["bid_close"]
    ).any():
        raise MarketDataValidationError("M5 input contains negative quote width")
    volume = frame["volume"].to_numpy(dtype=float)
    if not np.isfinite(volume).all() or (volume < 0).any() or not np.equal(
        volume, np.floor(volume)
    ).all():
        raise MarketDataValidationError("M5 input contains invalid volume")


def _ny_boundary(trading_date: dt.date) -> tuple[int, int]:
    start = dt.datetime.combine(trading_date, dt.time(17), tzinfo=NEW_YORK)
    end = dt.datetime.combine(
        trading_date + dt.timedelta(days=1),
        dt.time(17),
        tzinfo=NEW_YORK,
    )
    return int(start.timestamp()), int(end.timestamp())


def _trading_date(timestamp: pd.Timestamp) -> dt.date:
    local_date = timestamp.date()
    if timestamp.time() < dt.time(17):
        return local_date - dt.timedelta(days=1)
    return local_date


def aggregate_m5_to_ny_daily(
    frame: pd.DataFrame,
    *,
    minimum_m5: int = DAILY_MIN_M5,
) -> DailyAggregation:
    """Aggregate M5 directly into NY-17 trading days, preserving bid and ask.

    A row is available only at the next NY 17:00 boundary.  Days below A12's
    minimum are kept in ``day_status`` but excluded from the daily candle
    series used by swings and bias.
    """

    if minimum_m5 <= 0:
        raise ValueError("minimum_m5 must be positive")
    _validate_m5_for_daily(frame)
    working = frame.copy()
    working["ts_utc_dt"] = pd.to_datetime(working["ts_utc"], unit="s", utc=True)
    working["ts_et"] = working["ts_utc_dt"].dt.tz_convert(NEW_YORK)
    working["trading_date_et"] = [
        _trading_date(pd.Timestamp(value)) for value in working["ts_et"]
    ]

    candle_records: list[dict[str, object]] = []
    status_records: list[dict[str, object]] = []
    for trading_date, group in working.groupby(
        "trading_date_et", sort=True, dropna=False
    ):
        ordered = group.sort_values("ts_utc", kind="stable")
        component_count = int(ordered["ts_utc"].nunique())
        start_ts, available_ts = _ny_boundary(trading_date)
        accepted = component_count >= minimum_m5
        status_records.append(
            {
                "trading_date_et": trading_date,
                "start_ts_utc": start_ts,
                "available_ts_utc": available_ts,
                "component_count": component_count,
                "accepted": accepted,
                "reason": "accepted" if accepted else "below_minimum_m5",
            }
        )
        if not accepted:
            continue
        candle_records.append(
            {
                "source": str(ordered.iloc[0]["source"]),
                "instrument": str(ordered.iloc[0]["instrument"]),
                "granularity": "D_NY17",
                "price": str(ordered.iloc[0]["price"]),
                "trading_date_et": trading_date,
                "start_ts_utc": start_ts,
                "available_ts_utc": available_ts,
                "component_count": component_count,
                "volume": int(ordered["volume"].sum()),
                "bid_open": float(ordered.iloc[0]["bid_open"]),
                "bid_high": float(ordered["bid_high"].max()),
                "bid_low": float(ordered["bid_low"].min()),
                "bid_close": float(ordered.iloc[-1]["bid_close"]),
                "ask_open": float(ordered.iloc[0]["ask_open"]),
                "ask_high": float(ordered["ask_high"].max()),
                "ask_low": float(ordered["ask_low"].min()),
                "ask_close": float(ordered.iloc[-1]["ask_close"]),
            }
        )

    candles = pd.DataFrame.from_records(candle_records, columns=_DAILY_COLUMNS)
    status = pd.DataFrame.from_records(
        status_records, columns=_DAY_STATUS_COLUMNS
    )
    return DailyAggregation(candles=candles, day_status=status)


def prepare_timeframes(
    m5: pd.DataFrame,
    *,
    start_inclusive: int,
    end_exclusive: int,
) -> TimeframeBundle:
    """Create the exact inherited M15 and the new NY-17 daily derivative."""

    m15_result: M15Aggregation = aggregate_m5_to_m15(
        m5,
        start_inclusive=start_inclusive,
        end_exclusive=end_exclusive,
    )
    if not m15_result.candles.empty:
        validate_m15_frame(m15_result.candles)
    daily_result = aggregate_m5_to_ny_daily(m5)
    return TimeframeBundle(
        m15=m15_result.candles,
        incomplete_m15=m15_result.incomplete_buckets,
        daily=daily_result.candles,
        daily_status=daily_result.day_status,
    )


def find_confirmed_swings(
    frame: pd.DataFrame,
    *,
    k: int = SWING_K,
    high_column: str = "bid_high",
    low_column: str = "bid_low",
    timestamp_column: str = "ts_utc",
    bar_close_column: str | None = None,
    bar_seconds: int | None = None,
) -> tuple[SwingPoint, ...]:
    """Return strict k-fractal swings with the confirming bar's close time."""

    if k <= 0:
        raise ValueError("k must be positive")
    required = {timestamp_column, high_column, low_column}
    if bar_close_column is not None:
        required.add(bar_close_column)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"swing frame is missing columns: {sorted(missing)}")
    if frame.empty:
        return ()
    timestamps = frame[timestamp_column].astype("int64")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("swing frame timestamps must be sorted and unique")
    highs = frame[high_column].to_numpy(dtype=float)
    lows = frame[low_column].to_numpy(dtype=float)
    if not np.isfinite(highs).all() or not np.isfinite(lows).all():
        raise ValueError("swing frame contains non-finite prices")
    if bar_close_column is None and (bar_seconds is None or bar_seconds <= 0):
        raise ValueError("bar_seconds is required without bar_close_column")

    points: list[SwingPoint] = []
    for index in range(k, len(frame) - k):
        before = slice(index - k, index)
        after = slice(index + 1, index + k + 1)
        if bar_close_column is not None:
            confirmed_at = int(frame.iloc[index + k][bar_close_column])
        else:
            assert bar_seconds is not None
            confirmed_at = int(timestamps.iloc[index + k]) + bar_seconds
        source_ts = int(timestamps.iloc[index])
        if (highs[index] > highs[before]).all() and (
            highs[index] > highs[after]
        ).all():
            points.append(
                SwingPoint("high", source_ts, confirmed_at, float(highs[index]))
            )
        if (lows[index] < lows[before]).all() and (
            lows[index] < lows[after]
        ).all():
            points.append(
                SwingPoint("low", source_ts, confirmed_at, float(lows[index]))
            )
    return tuple(
        sorted(
            points,
            key=lambda point: (
                point.confirmed_at_utc,
                point.source_ts_utc,
                point.direction,
            ),
        )
    )


def _structure_bias(
    highs: Sequence[float],
    lows: Sequence[float],
) -> tuple[BiasState, str]:
    if len(highs) < 2 or len(lows) < 2:
        return "unavailable", "insufficient_confirmed_swings"
    highs_up = highs[-1] > highs[-2]
    lows_up = lows[-1] > lows[-2]
    highs_down = highs[-1] < highs[-2]
    lows_down = lows[-1] < lows[-2]
    if highs_up and lows_up:
        return "bullish", "structure_up"
    if highs_down and lows_down:
        return "bearish", "structure_down"
    return "neutral", "mixed_structure"


def build_daily_bias_timeline(
    swings: Iterable[SwingPoint],
    day_status: pd.DataFrame,
) -> tuple[BiasEvent, ...]:
    """Build A6/A13 bias events from confirmed swings and accepted days."""

    required = {"available_ts_utc", "accepted"}
    missing = required.difference(day_status.columns)
    if missing:
        raise ValueError(f"day status is missing columns: {sorted(missing)}")
    status_by_time: dict[int, bool] = {}
    for row in day_status.itertuples(index=False):
        timestamp = int(getattr(row, "available_ts_utc"))
        accepted = bool(getattr(row, "accepted"))
        if timestamp in status_by_time:
            raise ValueError("day status contains duplicate availability times")
        status_by_time[timestamp] = accepted

    swings_by_time: dict[int, list[SwingPoint]] = defaultdict(list)
    for swing in swings:
        swings_by_time[swing.confirmed_at_utc].append(swing)
    event_times = sorted(set(status_by_time).union(swings_by_time))
    highs: list[float] = []
    lows: list[float] = []
    daily_missing = False
    events: list[BiasEvent] = []
    previous: tuple[BiasState, str] | None = None

    for timestamp in event_times:
        if timestamp in status_by_time:
            daily_missing = not status_by_time[timestamp]
        for swing in sorted(
            swings_by_time.get(timestamp, []),
            key=lambda point: (point.source_ts_utc, point.direction),
        ):
            if swing.direction == "high":
                highs.append(swing.level)
            else:
                lows.append(swing.level)
        current = (
            ("unavailable", "daily_missing")
            if daily_missing
            else _structure_bias(highs, lows)
        )
        if current != previous:
            state, reason = current
            events.append(BiasEvent(timestamp, state, reason))
            previous = current
    return tuple(events)


def _bias_event_at(
    timeline: Sequence[BiasEvent],
    timestamp: int,
) -> BiasEvent | None:
    if not timeline:
        return None
    effective = [event.effective_at_utc for event in timeline]
    index = bisect_right(effective, timestamp) - 1
    return None if index < 0 else timeline[index]


def bias_at(timeline: Sequence[BiasEvent], timestamp: int) -> BiasState:
    """Return the latest bias known by one UTC timestamp."""

    event = _bias_event_at(timeline, timestamp)
    return "unavailable" if event is None else event.state


def annotate_m15_bias(
    frame: pd.DataFrame,
    timeline: Sequence[BiasEvent],
) -> pd.DataFrame:
    """Annotate bias known at each M15 open and close without lookahead."""

    validate_m15_frame(frame)
    result = frame.copy()
    open_times = result["ts_utc"].astype("int64")
    close_times = result["ts_utc"].astype("int64") + 900
    effective = [event.effective_at_utc for event in timeline]

    def events_at(timestamps: Iterable[int]) -> list[BiasEvent | None]:
        events: list[BiasEvent | None] = []
        for timestamp in timestamps:
            index = bisect_right(effective, int(timestamp)) - 1
            events.append(None if index < 0 else timeline[index])
        return events

    open_events = events_at(open_times)
    close_events = events_at(close_times)
    result["daily_bias_at_open"] = [
        "unavailable" if event is None else event.state for event in open_events
    ]
    result["daily_bias_reason_at_open"] = [
        "no_daily_event" if event is None else event.reason for event in open_events
    ]
    result["daily_bias"] = [
        "unavailable" if event is None else event.state for event in close_events
    ]
    result["daily_bias_reason"] = [
        "no_daily_event" if event is None else event.reason for event in close_events
    ]
    return result


def _official_candidate(
    row: pd.Series,
    *,
    side: Side,
) -> _OfficialObCandidate:
    opened = float(row["bid_open"])
    closed = float(row["bid_close"])
    return _OfficialObCandidate(
        side=side,
        source_ts_utc=int(row["ts_utc"]),
        bid_open=opened,
        bid_high=float(row["bid_high"]),
        bid_low=float(row["bid_low"]),
        ask_high=float(row["ask_high"]),
        body_range=abs(opened - closed),
    )


def _validate_bias_column(
    frame: pd.DataFrame,
    column: str,
    *,
    required: bool,
) -> None:
    if column not in frame.columns:
        if required:
            raise ValueError(f"M15 input is missing bias column: {column}")
        return
    observed = set(frame[column].astype(str))
    unexpected = sorted(observed.difference(VALID_BIAS_STATES))
    if unexpected:
        raise ValueError(f"M15 input contains invalid bias values: {unexpected}")


def _candidate_on_bar(
    frame: pd.DataFrame,
    index: int,
    *,
    side: Side,
    lookback: int,
) -> _OfficialObCandidate | None:
    if index < lookback - 1:
        return None
    row = frame.iloc[index]
    window = frame.iloc[index - lookback + 1 : index + 1]
    opened = float(row["bid_open"])
    closed = float(row["bid_close"])
    if side == "long":
        qualifies = closed < opened and float(row["bid_low"]) == float(
            window["bid_low"].min()
        )
    else:
        qualifies = closed > opened and float(row["bid_high"]) == float(
            window["bid_high"].max()
        )
    return _official_candidate(row, side=side) if qualifies else None


def _candidate_is_more_extreme(
    current: _OfficialObCandidate,
    existing: _OfficialObCandidate,
) -> bool:
    if current.side == "long":
        return current.bid_low < existing.bid_low
    return current.bid_high > existing.bid_high


def _candidate_has_same_extreme(
    current: _OfficialObCandidate,
    existing: _OfficialObCandidate,
) -> bool:
    if current.side == "long":
        return current.bid_low == existing.bid_low
    return current.bid_high == existing.bid_high


def _candidate_break_and_invalidation(
    candidate: _OfficialObCandidate,
    row: pd.Series,
) -> tuple[bool, bool]:
    if candidate.side == "long":
        activated = float(row["bid_high"]) > candidate.bid_high
        invalidated = float(row["bid_low"]) < candidate.bid_low
    else:
        activated = float(row["bid_low"]) < candidate.bid_low
        invalidated = float(row["bid_high"]) > candidate.bid_high
    return activated, invalidated


def _zone_from_official_candidate(
    candidate: _OfficialObCandidate,
    *,
    activation_ts_utc: int,
    active_from_ts_utc: int,
) -> PendingZone:
    side = candidate.side
    return PendingZone(
        zone_id=(
            f"official:{side}:{candidate.source_ts_utc}:"
            f"{activation_ts_utc}"
        ),
        detector=OFFICIAL_OB_DETECTOR,
        side=side,
        active_from_ts_utc=active_from_ts_utc,
        lower=(candidate.bid_low if side == "long" else candidate.bid_open),
        upper=(candidate.bid_open if side == "long" else candidate.bid_high),
        entry_price=candidate.bid_open,
        stop_loss=(candidate.bid_low if side == "long" else candidate.ask_high),
        signal_ts_utc=activation_ts_utc,
    )


def detect_official_order_blocks(
    frame: pd.DataFrame,
    *,
    lookback: int = OFFICIAL_OB_LOOKBACK,
    bias_column: str = "daily_bias",
) -> OfficialObDetectionResult:
    """Detect the frozen Month 04 translation on complete M15 bid candles.

    A full W-bar window is required before a candidate can exist.  Candidate
    updates on the current candle are processed before breakout checks so that
    a fresh lower low / higher high cannot activate the candidate it replaces.
    If one candle crosses both the activation and invalidation levels, the
    unknown intrabar order is resolved conservatively as invalidation.  Bias is
    read at the activation candle's close, and the resulting limit becomes
    fillable from the next observed complete M15 candle.
    """

    validate_m15_frame(frame)
    if lookback <= 0:
        raise ValueError("official OB lookback must be positive")
    _validate_bias_column(frame, bias_column, required=True)

    counters: Counter[str] = Counter()
    candidates: dict[Side, _OfficialObCandidate | None] = {
        "long": None,
        "short": None,
    }
    zones: list[PendingZone] = []

    for index in range(len(frame)):
        row = frame.iloc[index]
        bias = str(row[bias_column])
        for side in ("long", "short"):
            current = _candidate_on_bar(
                frame,
                index,
                side=side,
                lookback=lookback,
            )
            existing = candidates[side]

            if existing is None:
                if current is not None:
                    candidates[side] = current
                    counters[f"{side}_candidate_selected"] += 1
                continue

            if current is not None and _candidate_is_more_extreme(
                current,
                existing,
            ):
                candidates[side] = current
                counters[f"{side}_candidate_replaced"] += 1
                continue

            if (
                current is not None
                and _candidate_has_same_extreme(current, existing)
                and current.body_range > existing.body_range
            ):
                candidates[side] = current
                counters[f"{side}_candidate_replaced"] += 1
                continue

            activated, invalidated = _candidate_break_and_invalidation(
                existing,
                row,
            )
            if invalidated:
                candidates[side] = None
                counters[f"{side}_candidate_invalidated"] += 1
                if activated:
                    counters[f"{side}_ambiguous_break_invalidated"] += 1
                if current is not None:
                    candidates[side] = current
                    counters[f"{side}_candidate_selected"] += 1
                continue
            if not activated:
                continue

            candidates[side] = None
            required_bias = "bullish" if side == "long" else "bearish"
            if bias != required_bias:
                if bias in {"neutral", "unavailable"}:
                    counters["bias_unavailable"] += 1
                else:
                    counters["bias_mismatch"] += 1
                continue
            if index + 1 >= len(frame):
                counters["activation_without_next_bar"] += 1
                continue
            zone = _zone_from_official_candidate(
                existing,
                activation_ts_utc=int(row["ts_utc"]),
                active_from_ts_utc=int(frame.iloc[index + 1]["ts_utc"]),
            )
            _validate_zone(zone)
            zones.append(zone)
            counters["zone_detected"] += 1
            counters[f"{side}_zone_detected"] += 1

    return OfficialObDetectionResult(
        zones=tuple(zones),
        counters=dict(sorted(counters.items())),
    )


def make_official_daily_target_resolver(
    daily: pd.DataFrame,
    swings: Iterable[SwingPoint],
    *,
    lookback_days: int = OFFICIAL_TARGET_LOOKBACK_DAYS,
) -> OfficialDailyTargetResolver:
    """Build the frozen 60-trading-day external-liquidity resolver."""

    if lookback_days <= 0:
        raise ValueError("official target lookback must be positive")
    required = {"start_ts_utc", "available_ts_utc"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily input is missing columns: {sorted(missing)}")
    starts = tuple(int(value) for value in daily["start_ts_utc"])
    available = tuple(int(value) for value in daily["available_ts_utc"])
    if tuple(sorted(starts)) != starts or len(set(starts)) != len(starts):
        raise ValueError("daily starts must be sorted and unique")
    if tuple(sorted(available)) != available or len(set(available)) != len(available):
        raise ValueError("daily availability times must be sorted and unique")
    if any(start >= known_at for start, known_at in zip(starts, available, strict=True)):
        raise ValueError("daily candles must become available after their start")
    swing_list = tuple(swings)
    unknown_sources = sorted(
        {swing.source_ts_utc for swing in swing_list}.difference(starts)
    )
    if unknown_sources:
        raise ValueError(
            f"daily swings reference unknown source timestamps: {unknown_sources[:3]}"
        )
    return OfficialDailyTargetResolver(
        day_start_ts_utc=starts,
        day_available_ts_utc=available,
        swings=swing_list,
        lookback_days=lookback_days,
    )


def _secondary_mss_confirmed(setup: _SecondarySetup, row: pd.Series) -> bool:
    close = float(row["bid_close"])
    if setup.side == "long":
        return close > setup.mss_level
    return close < setup.mss_level


def _secondary_leg_has_fvg(
    frame: pd.DataFrame,
    *,
    side: Side,
    sweep_index: int,
    mss_index: int,
) -> bool:
    """Return whether the known sweep-to-MSS leg contains a strict FVG.

    The central candle may be the sweep candle or the candle immediately
    before MSS.  MSS itself cannot be the central candle because its right
    neighbour is not known when MSS closes.
    """

    first_center = max(sweep_index, 1)
    for center in range(first_center, mss_index):
        before = frame.iloc[center - 1]
        after = frame.iloc[center + 1]
        if side == "long":
            if float(after["bid_low"]) > float(before["bid_high"]):
                return True
        elif float(after["bid_high"]) < float(before["bid_low"]):
            return True
    return False


def _secondary_ob_index(
    frame: pd.DataFrame,
    *,
    side: Side,
    sweep_index: int,
    mss_index: int,
) -> int | None:
    for index in range(mss_index - 1, sweep_index - 1, -1):
        row = frame.iloc[index]
        opened = float(row["bid_open"])
        closed = float(row["bid_close"])
        if (side == "long" and closed < opened) or (
            side == "short" and closed > opened
        ):
            return index
    return None


def _zone_from_secondary_setup(
    frame: pd.DataFrame,
    setup: _SecondarySetup,
    *,
    ob_index: int,
    mss_index: int,
) -> PendingZone:
    ob = frame.iloc[ob_index]
    mss_ts_utc = int(frame.iloc[mss_index]["ts_utc"])
    side = setup.side
    lower = float(ob["bid_low"])
    upper = float(ob["bid_high"])
    return PendingZone(
        zone_id=(
            f"secondary:{side}:{setup.sweep_ts_utc}:"
            f"{int(ob['ts_utc'])}:{mss_ts_utc}"
        ),
        detector=SECONDARY_OB_DETECTOR,
        side=side,
        active_from_ts_utc=int(frame.iloc[mss_index + 1]["ts_utc"]),
        lower=lower,
        upper=upper,
        entry_price=upper if side == "long" else lower,
        stop_loss=lower if side == "long" else float(ob["ask_high"]),
        signal_ts_utc=mss_ts_utc,
    )


def detect_secondary_order_blocks(
    frame: pd.DataFrame,
    *,
    max_mss_bars: int = SECONDARY_MSS_MAX_BARS,
    bias_column: str = "daily_bias",
) -> SecondaryObDetectionResult:
    """Detect the frozen secondary sweep -> MSS -> FVG translation.

    Swing references are strict k=2 M15 fractals known at the sweep candle's
    open.  A same-side sweep replaces an unresolved older sweep.  MSS may
    occur on bars 1 through K after the sweep.  The FVG and last opposite
    candle are then evaluated only from candles known when MSS closes, and a
    valid zone becomes fillable on the next observed complete M15 candle.
    """

    validate_m15_frame(frame)
    if max_mss_bars <= 0:
        raise ValueError("secondary MSS maximum must be positive")
    _validate_bias_column(frame, bias_column, required=True)

    swings = find_confirmed_swings(frame, bar_seconds=M15_SECONDS)
    swing_cursor = 0
    latest: dict[SwingDirection, SwingPoint | None] = {
        "high": None,
        "low": None,
    }
    setups: dict[Side, _SecondarySetup | None] = {
        "long": None,
        "short": None,
    }
    counters: Counter[str] = Counter()
    zones: list[PendingZone] = []

    for index in range(len(frame)):
        row = frame.iloc[index]
        timestamp = int(row["ts_utc"])
        while (
            swing_cursor < len(swings)
            and swings[swing_cursor].confirmed_at_utc <= timestamp
        ):
            swing = swings[swing_cursor]
            existing = latest[swing.direction]
            if existing is None or swing.source_ts_utc > existing.source_ts_utc:
                latest[swing.direction] = swing
            swing_cursor += 1

        for side in ("long", "short"):
            setup = setups[side]
            if setup is not None:
                bars_after_sweep = index - setup.sweep_index
                if bars_after_sweep > max_mss_bars:
                    setups[side] = None
                    counters["no_mss"] += 1
                    counters[f"{side}_no_mss"] += 1
                    setup = None

            low_reference = latest["low"]
            high_reference = latest["high"]
            swept = False
            swept_level = 0.0
            mss_level = 0.0
            if low_reference is not None and high_reference is not None:
                if side == "long":
                    swept = (
                        float(row["bid_low"]) < low_reference.level
                        and float(row["bid_close"]) > low_reference.level
                    )
                    swept_level = low_reference.level
                    mss_level = high_reference.level
                else:
                    swept = (
                        float(row["bid_high"]) > high_reference.level
                        and float(row["bid_close"]) < high_reference.level
                    )
                    swept_level = high_reference.level
                    mss_level = low_reference.level

            if swept:
                bias = str(row[bias_column])
                required_bias = "bullish" if side == "long" else "bearish"
                if bias != required_bias:
                    if bias in {"neutral", "unavailable"}:
                        counters["bias_unavailable"] += 1
                    else:
                        counters["bias_mismatch"] += 1
                else:
                    if setup is not None:
                        counters["sweep_replaced"] += 1
                        counters[f"{side}_sweep_replaced"] += 1
                    setups[side] = _SecondarySetup(
                        side=side,
                        sweep_index=index,
                        sweep_ts_utc=timestamp,
                        swept_level=swept_level,
                        mss_level=mss_level,
                    )
                    counters["sweep_detected"] += 1
                    counters[f"{side}_sweep_detected"] += 1
                    continue

            if setup is not None and _secondary_mss_confirmed(setup, row):
                setups[side] = None
                counters["mss_confirmed"] += 1
                counters[f"{side}_mss_confirmed"] += 1
                if not _secondary_leg_has_fvg(
                    frame,
                    side=side,
                    sweep_index=setup.sweep_index,
                    mss_index=index,
                ):
                    counters["no_fvg"] += 1
                    counters[f"{side}_no_fvg"] += 1
                else:
                    counters["fvg_confirmed"] += 1
                    ob_index = _secondary_ob_index(
                        frame,
                        side=side,
                        sweep_index=setup.sweep_index,
                        mss_index=index,
                    )
                    if ob_index is None:
                        counters["no_ob"] += 1
                        counters[f"{side}_no_ob"] += 1
                    else:
                        bias = str(row[bias_column])
                        required_bias = (
                            "bullish" if side == "long" else "bearish"
                        )
                        if bias != required_bias:
                            if bias in {"neutral", "unavailable"}:
                                counters["bias_unavailable"] += 1
                            else:
                                counters["bias_mismatch"] += 1
                        elif index + 1 >= len(frame):
                            counters["mss_without_next_bar"] += 1
                        else:
                            zone = _zone_from_secondary_setup(
                                frame,
                                setup,
                                ob_index=ob_index,
                                mss_index=index,
                            )
                            _validate_zone(zone)
                            zones.append(zone)
                            counters["zone_detected"] += 1
                            counters[f"{side}_zone_detected"] += 1

    return SecondaryObDetectionResult(
        zones=tuple(zones),
        counters=dict(sorted(counters.items())),
    )


def make_secondary_m15_target_resolver(
    frame: pd.DataFrame,
    swings: Iterable[SwingPoint] | None = None,
    *,
    lookback_bars: int = SECONDARY_TARGET_LOOKBACK_BARS,
) -> SecondaryM15TargetResolver:
    """Build the frozen 400-complete-M15-bar target resolver."""

    validate_m15_frame(frame)
    if lookback_bars <= 0:
        raise ValueError("secondary target lookback must be positive")
    starts = tuple(int(value) for value in frame["ts_utc"])
    swing_list = tuple(
        find_confirmed_swings(frame, bar_seconds=M15_SECONDS)
        if swings is None
        else swings
    )
    unknown_sources = sorted(
        {swing.source_ts_utc for swing in swing_list}.difference(starts)
    )
    if unknown_sources:
        raise ValueError(
            "M15 swings reference unknown source timestamps: "
            f"{unknown_sources[:3]}"
        )
    if any(
        swing.direction not in ("high", "low")
        or not math.isfinite(swing.level)
        or swing.confirmed_at_utc <= swing.source_ts_utc
        for swing in swing_list
    ):
        raise ValueError("M15 swings contain invalid contract values")
    return SecondaryM15TargetResolver(
        bar_start_ts_utc=starts,
        swings=swing_list,
        lookback_bars=lookback_bars,
    )


def _validate_zone(zone: PendingZone) -> None:
    values = (zone.lower, zone.upper, zone.entry_price, zone.stop_loss)
    if not zone.zone_id or not zone.detector:
        raise ValueError("zone_id and detector must be non-empty")
    if zone.side not in ("long", "short"):
        raise ValueError("zone side must be long or short")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("zone contains a non-finite price")
    if zone.lower > zone.upper:
        raise ValueError("zone lower must not exceed upper")
    if not zone.lower <= zone.entry_price <= zone.upper:
        raise ValueError("entry price must be inside the zone")
    if zone.side == "long" and zone.stop_loss > zone.lower:
        raise ValueError("long stop must be at or below the zone")
    if zone.side == "short" and zone.stop_loss < zone.upper:
        raise ValueError("short stop must be at or above the zone")
    if zone.active_from_ts_utc % 900:
        raise ValueError("zone activation must be an M15 boundary")
    if zone.signal_ts_utc is not None:
        if zone.signal_ts_utc % 900:
            raise ValueError("zone signal must be an M15 boundary")
        if zone.signal_ts_utc >= zone.active_from_ts_utc:
            raise ValueError("zone signal must precede activation")


def _entry_touched(zone: PendingZone, bar: pd.Series) -> bool:
    if zone.side == "long":
        return float(bar["ask_low"]) <= zone.entry_price
    return float(bar["bid_high"]) >= zone.entry_price


def _bias_allows_entry(zone: PendingZone, bar: pd.Series) -> tuple[bool, str | None]:
    """Apply C4 when the production bias annotation is present.

    The isolated stage-3 execution fixtures predate the detector and therefore
    have no bias column.  Such frames exercise quote mechanics only.  Detector
    pipelines necessarily carry ``daily_bias`` and are filtered here at fill.
    """

    if "daily_bias_at_open" not in bar.index:
        return True, None
    bias = str(bar["daily_bias_at_open"])
    required = "bullish" if zone.side == "long" else "bearish"
    if bias == required:
        return True, None
    if bias in {"neutral", "unavailable"}:
        return False, "bias_unavailable"
    if bias in {"bullish", "bearish"}:
        return False, "bias_mismatch"
    raise ValueError(f"M15 input contains invalid bias value: {bias}")


def _zone_invalidated(zone: PendingZone, bar: pd.Series) -> bool:
    close = float(bar["bid_close"])
    if zone.side == "long":
        return close < zone.lower
    return close > zone.upper


def _existing_position_exit(
    position: OpenPosition,
    bar: pd.Series,
) -> tuple[float, str] | None:
    if position.side == "long":
        opened = float(bar["bid_open"])
        if opened <= position.stop_loss:
            return opened, "gap_sl"
        if opened >= position.take_profit:
            return position.take_profit, "tp"
        hit_stop = float(bar["bid_low"]) <= position.stop_loss
        hit_target = float(bar["bid_high"]) >= position.take_profit
    else:
        opened = float(bar["ask_open"])
        if opened >= position.stop_loss:
            return opened, "gap_sl"
        if opened <= position.take_profit:
            return position.take_profit, "tp"
        hit_stop = float(bar["ask_high"]) >= position.stop_loss
        hit_target = float(bar["ask_low"]) <= position.take_profit
    if hit_stop:
        return position.stop_loss, "sl"
    if hit_target:
        return position.take_profit, "tp"
    return None


def _fill_bar_exit(
    position: OpenPosition,
    bar: pd.Series,
    *,
    entry_at_open: bool,
) -> tuple[float, str] | None:
    """Resolve the fill bar conservatively when its intrabar path is unknown."""

    if position.side == "long":
        if entry_at_open and float(bar["bid_open"]) <= position.stop_loss:
            return float(bar["bid_open"]), "gap_sl"
        hit_stop = float(bar["bid_low"]) <= position.stop_loss
        hit_target = float(bar["bid_high"]) >= position.take_profit
    else:
        if entry_at_open and float(bar["ask_open"]) >= position.stop_loss:
            return float(bar["ask_open"]), "gap_sl"
        hit_stop = float(bar["ask_high"]) >= position.stop_loss
        hit_target = float(bar["ask_low"]) <= position.take_profit
    if hit_stop:
        return position.stop_loss, "sl"
    if entry_at_open and hit_target:
        return position.take_profit, "tp"
    return None


def _close_position(
    position: OpenPosition,
    *,
    exit_time_utc: int,
    exit_price: float,
    exit_reason: str,
) -> ZoneTrade:
    per_unit = (
        exit_price - position.entry_price
        if position.side == "long"
        else position.entry_price - exit_price
    )
    pnl = position.units * per_unit
    equity_after = position.equity_before + pnl
    return ZoneTrade(
        zone_id=position.zone_id,
        detector=position.detector,
        side=position.side,
        entry_time_utc=position.entry_time_utc,
        entry_price=position.entry_price,
        stop_loss=position.stop_loss,
        take_profit=position.take_profit,
        initial_risk=position.initial_risk,
        units=position.units,
        exit_time_utc=exit_time_utc,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl=pnl,
        equity_before=position.equity_before,
        equity_after=equity_after,
        realized_r=per_unit / position.initial_risk,
    )


def simulate_zone_backtest(
    frame: pd.DataFrame,
    zones: Iterable[PendingZone],
    *,
    target_resolver: TargetResolver,
    initial_cash: float = INITIAL_CASH,
    window_start: dt.date = ARTICLE_WINDOW_START,
    window_end_exclusive: dt.date = ARTICLE_WINDOW_END_EXCLUSIVE,
) -> ZoneBacktestResult:
    """Run the frozen shared limit-order model over one detector's zones.

    New zones replace the pending zone before that M15 bar can fill.  Existing
    positions apply open-gap rules before range checks.  Limit orders reached
    after the open cannot take profit on their fill bar because OHLC does not
    reveal whether the target preceded the entry.  A position closed on a bar
    cannot be replaced by a new fill on the same bar.  When bias annotations
    are present, fills use only the state known at the bar open.  Signal and
    fill timestamps must both satisfy the article window.  No terminal
    liquidation or swap is applied.
    """

    validate_m15_frame(frame)
    _validate_bias_column(frame, "daily_bias", required=False)
    _validate_bias_column(frame, "daily_bias_at_open", required=False)
    if "daily_bias" in frame.columns and "daily_bias_at_open" not in frame.columns:
        raise ValueError(
            "M15 input with daily_bias must also include daily_bias_at_open"
        )
    if initial_cash <= 0 or not math.isfinite(initial_cash):
        raise ValueError("initial_cash must be finite and positive")
    if window_start >= window_end_exclusive:
        raise ValueError("window_start must precede window_end_exclusive")
    zone_list = list(zones)
    for zone in zone_list:
        _validate_zone(zone)
    known_timestamps = set(int(value) for value in frame["ts_utc"])
    unknown = sorted(
        {
            zone.active_from_ts_utc
            for zone in zone_list
            if zone.active_from_ts_utc not in known_timestamps
        }
    )
    if unknown:
        raise ValueError(f"zone activation is outside the M15 input: {unknown[:3]}")
    unknown_signals = sorted(
        {
            zone.signal_ts_utc
            for zone in zone_list
            if zone.signal_ts_utc is not None
            and zone.signal_ts_utc not in known_timestamps
        }
    )
    if unknown_signals:
        raise ValueError(f"zone signal is outside the M15 input: {unknown_signals[:3]}")

    zones_by_time: dict[int, list[PendingZone]] = defaultdict(list)
    for zone in zone_list:
        zones_by_time[zone.active_from_ts_utc].append(zone)
    counters: Counter[str] = Counter()
    trades: list[ZoneTrade] = []
    lifecycles: list[ZoneLifecycle] = []
    pending: PendingZone | None = None
    position: OpenPosition | None = None
    equity = float(initial_cash)

    def close_lifecycle(
        zone: PendingZone,
        *,
        end_exclusive_ts_utc: int,
        reason: ZoneEndReason,
    ) -> None:
        if end_exclusive_ts_utc < zone.active_from_ts_utc:
            raise RuntimeError("zone lifecycle ends before activation")
        lifecycles.append(
            ZoneLifecycle(
                zone_id=zone.zone_id,
                detector=zone.detector,
                side=zone.side,
                active_from_ts_utc=zone.active_from_ts_utc,
                end_exclusive_ts_utc=end_exclusive_ts_utc,
                lower=zone.lower,
                upper=zone.upper,
                end_reason=reason,
            )
        )

    for _, bar in frame.iterrows():
        timestamp = int(bar["ts_utc"])
        timestamp_et = dt.datetime.fromtimestamp(
            timestamp,
            tz=dt.UTC,
        ).astimezone(NEW_YORK)
        entry_allowed = window_start <= timestamp_et.date() < window_end_exclusive
        new_zones = zones_by_time.get(timestamp, [])
        if entry_allowed:
            for new_zone in new_zones:
                signal_timestamp = (
                    new_zone.active_from_ts_utc
                    if new_zone.signal_ts_utc is None
                    else new_zone.signal_ts_utc
                )
                signal_date_et = dt.datetime.fromtimestamp(
                    signal_timestamp,
                    tz=dt.UTC,
                ).astimezone(NEW_YORK).date()
                if not window_start <= signal_date_et < window_end_exclusive:
                    counters["zone_outside_window"] += 1
                    close_lifecycle(
                        new_zone,
                        end_exclusive_ts_utc=new_zone.active_from_ts_utc,
                        reason="outside_window",
                    )
                    continue
                if pending is not None:
                    counters["zone_replaced"] += 1
                    close_lifecycle(
                        pending,
                        end_exclusive_ts_utc=timestamp,
                        reason="replaced",
                    )
                pending = new_zone
                counters["zone_activated"] += 1
        else:
            counters["zone_outside_window"] += len(new_zones)
            for new_zone in new_zones:
                close_lifecycle(
                    new_zone,
                    end_exclusive_ts_utc=new_zone.active_from_ts_utc,
                    reason="outside_window",
                )

        closed_this_bar = False
        if position is not None:
            resolved = _existing_position_exit(position, bar)
            if resolved is not None:
                exit_price, exit_reason = resolved
                trade = _close_position(
                    position,
                    exit_time_utc=timestamp,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                )
                trades.append(trade)
                equity = trade.equity_after
                position = None
                closed_this_bar = True
                counters["trade_closed"] += 1

        if (
            position is None
            and not closed_this_bar
            and pending is not None
            and entry_allowed
        ):
            if _entry_touched(pending, bar):
                bias_allowed, bias_counter = _bias_allows_entry(pending, bar)
                if not bias_allowed:
                    assert bias_counter is not None
                    counters[bias_counter] += 1
                    if _zone_invalidated(pending, bar):
                        close_lifecycle(
                            pending,
                            end_exclusive_ts_utc=timestamp + M15_SECONDS,
                            reason="invalidated",
                        )
                        pending = None
                        counters["zone_invalidated"] += 1
                else:
                    zone = pending
                    close_lifecycle(
                        zone,
                        end_exclusive_ts_utc=timestamp + M15_SECONDS,
                        reason="consumed",
                    )
                    pending = None
                    counters["zone_consumed"] += 1
                    target = target_resolver(zone, timestamp)
                    if target is None:
                        counters["no_target"] += 1
                    else:
                        target = float(target)
                        if not math.isfinite(target):
                            raise ValueError(
                                "target resolver returned a non-finite price"
                            )
                        if zone.side == "long" and target <= zone.entry_price:
                            raise ValueError("long target must be above entry")
                        if zone.side == "short" and target >= zone.entry_price:
                            raise ValueError("short target must be below entry")
                        initial_risk = (
                            zone.entry_price - zone.stop_loss
                            if zone.side == "long"
                            else zone.stop_loss - zone.entry_price
                        )
                        if not math.isfinite(initial_risk) or initial_risk <= 0:
                            counters["invalid_risk"] += 1
                        else:
                            units = compute_units(
                                equity=equity,
                                atr=initial_risk,
                                price=zone.entry_price,
                                risk_pct=RISK_PCT,
                                sl_atr_mult=1.0,
                                margin=MARGIN,
                                spread=0.0,
                            )
                            if units <= 0:
                                counters["sizing_rejected"] += 1
                            else:
                                position = OpenPosition(
                                    zone_id=zone.zone_id,
                                    detector=zone.detector,
                                    side=zone.side,
                                    entry_time_utc=timestamp,
                                    entry_price=zone.entry_price,
                                    stop_loss=zone.stop_loss,
                                    take_profit=target,
                                    initial_risk=initial_risk,
                                    units=units,
                                    equity_before=equity,
                                )
                                counters["filled"] += 1
                                entry_at_open = (
                                    float(bar["ask_open"]) <= zone.entry_price
                                    if zone.side == "long"
                                    else float(bar["bid_open"])
                                    >= zone.entry_price
                                )
                                resolved = _fill_bar_exit(
                                    position,
                                    bar,
                                    entry_at_open=entry_at_open,
                                )
                                if resolved is not None:
                                    exit_price, exit_reason = resolved
                                    trade = _close_position(
                                        position,
                                        exit_time_utc=timestamp,
                                        exit_price=exit_price,
                                        exit_reason=exit_reason,
                                    )
                                    trades.append(trade)
                                    equity = trade.equity_after
                                    position = None
                                    counters["trade_closed"] += 1
            elif _zone_invalidated(pending, bar):
                close_lifecycle(
                    pending,
                    end_exclusive_ts_utc=timestamp + M15_SECONDS,
                    reason="invalidated",
                )
                pending = None
                counters["zone_invalidated"] += 1
        elif pending is not None and _zone_invalidated(pending, bar):
            close_lifecycle(
                pending,
                end_exclusive_ts_utc=timestamp + M15_SECONDS,
                reason="invalidated",
            )
            pending = None
            counters["zone_invalidated"] += 1

    if position is not None:
        counters["open_at_end"] += 1
    if pending is not None:
        close_lifecycle(
            pending,
            end_exclusive_ts_utc=int(frame.iloc[-1]["ts_utc"]) + M15_SECONDS,
            reason="data_end",
        )
    if len(lifecycles) != len(zone_list):
        raise RuntimeError("zone lifecycle audit does not cover every detector zone")
    return ZoneBacktestResult(
        trades=trades,
        final_equity=equity,
        open_position=position,
        pending_zone=pending,
        counters=dict(sorted(counters.items())),
        zone_lifecycles=tuple(lifecycles),
    )
