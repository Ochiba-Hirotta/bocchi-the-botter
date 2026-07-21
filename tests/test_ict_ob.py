"""Artificial-fixture tests for ICT Order Block stages 3 through 5."""
from __future__ import annotations

import datetime as dt
from dataclasses import replace
from importlib.metadata import version

import pandas as pd
import pytest

from bocchi_the_botter_repro.season2.ict_ob import (
    DAILY_MIN_M5,
    INITIAL_CASH,
    OSS_PACKAGE,
    OSS_PACKAGE_VERSION,
    BiasEvent,
    PendingZone,
    SwingPoint,
    aggregate_m5_to_ny_daily,
    annotate_m15_bias,
    bias_at,
    build_daily_bias_timeline,
    detect_official_order_blocks,
    detect_secondary_order_blocks,
    find_confirmed_swings,
    make_official_daily_target_resolver,
    make_secondary_m15_target_resolver,
    prepare_timeframes,
    simulate_zone_backtest,
)
from bocchi_the_botter_repro.season2.minute_data import parse_m5_boundary


NEW_YORK = "America/New_York"
FIXTURE_WINDOW = {
    "window_start": dt.date(1969, 12, 31),
    "window_end_exclusive": dt.date(1970, 1, 2),
}


def m5_rows_for_trading_day(
    trading_date: dt.date,
    *,
    limit: int | None = None,
    base: float = 100.0,
) -> list[dict[str, object]]:
    # Construct the local boundary explicitly; replacing a UTC timezone would
    # shift the intended NY wall-clock time.
    start_et = pd.Timestamp(f"{trading_date.isoformat()} 17:00", tz=NEW_YORK)
    end_et = pd.Timestamp(
        f"{(trading_date + dt.timedelta(days=1)).isoformat()} 17:00",
        tz=NEW_YORK,
    )
    timestamps = pd.date_range(
        start=start_et.tz_convert("UTC"),
        end=end_et.tz_convert("UTC"),
        freq="5min",
        inclusive="left",
    )
    if limit is not None:
        timestamps = timestamps[:limit]
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        price = base + index * 0.001
        rows.append(
            {
                "source": "oanda_rest_v20",
                "instrument": "USD_JPY",
                "granularity": "M5",
                "price": "BA",
                "ts_utc": int(timestamp.timestamp()),
                "fetched_at_utc": 1,
                "complete": 1,
                "volume": index + 1,
                "bid_open": price,
                "bid_high": price + 0.20,
                "bid_low": price - 0.20,
                "bid_close": price + 0.05,
                "ask_open": price + 0.02,
                "ask_high": price + 0.22,
                "ask_low": price - 0.18,
                "ask_close": price + 0.07,
            }
        )
    return rows


def m5_frame(*day_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows = [row for group in day_rows for row in group]
    return pd.DataFrame(rows).sort_values("ts_utc").reset_index(drop=True)


def m15_row(
    timestamp: int,
    *,
    bid_open: float = 100.2,
    bid_high: float = 100.8,
    bid_low: float = 99.5,
    bid_close: float = 100.2,
    quote_width: float = 0.02,
) -> dict[str, object]:
    return {
        "source": "oanda_rest_v20",
        "instrument": "USD_JPY",
        "granularity": "M15",
        "price": "BA",
        "ts_utc": timestamp,
        "complete": 1,
        "component_count": 3,
        "bid_open": bid_open,
        "bid_high": bid_high,
        "bid_low": bid_low,
        "bid_close": bid_close,
        "ask_open": bid_open + quote_width,
        "ask_high": bid_high + quote_width,
        "ask_low": bid_low + quote_width,
        "ask_close": bid_close + quote_width,
    }


def m15_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("ts_utc").reset_index(drop=True)


def biased_m15_frame(
    rows: list[dict[str, object]],
    *,
    bias: str,
) -> pd.DataFrame:
    frame = m15_frame(*rows)
    frame["daily_bias"] = bias
    frame["daily_bias_at_open"] = bias
    return frame


def official_warmup_rows(count: int = 19) -> list[dict[str, object]]:
    return [
        m15_row(
            900 * index,
            bid_open=100.0,
            bid_high=101.0,
            bid_low=99.0,
            bid_close=100.0,
        )
        for index in range(count)
    ]


def secondary_long_reference_rows() -> list[dict[str, object]]:
    """Create one confirmed high followed by one confirmed low."""

    return [
        m15_row(0, bid_open=100.0, bid_high=100.5, bid_low=99.5),
        m15_row(900, bid_open=100.0, bid_high=101.0, bid_low=99.3),
        m15_row(1_800, bid_open=100.0, bid_high=103.0, bid_low=99.4),
        m15_row(2_700, bid_open=100.0, bid_high=101.5, bid_low=99.2),
        m15_row(3_600, bid_open=100.0, bid_high=101.0, bid_low=99.0),
        m15_row(4_500, bid_open=100.0, bid_high=101.2, bid_low=99.4),
        m15_row(5_400, bid_open=100.0, bid_high=101.0, bid_low=99.5),
    ]


def secondary_long_detection_rows() -> list[dict[str, object]]:
    rows = secondary_long_reference_rows()
    rows.extend(
        [
            m15_row(
                6_300,
                bid_open=100.0,
                bid_high=100.4,
                bid_low=98.5,
                bid_close=99.5,
            ),
            m15_row(
                7_200,
                bid_open=100.0,
                bid_high=100.5,
                bid_low=99.3,
                bid_close=99.8,
            ),
            m15_row(
                8_100,
                bid_open=101.0,
                bid_high=103.5,
                bid_low=100.6,
                bid_close=103.2,
            ),
            m15_row(
                9_000,
                bid_open=100.48,
                bid_high=103.2,
                bid_low=100.47,
                bid_close=103.0,
            ),
        ]
    )
    return rows


def long_zone(timestamp: int, *, zone_id: str = "z1") -> PendingZone:
    return PendingZone(
        zone_id=zone_id,
        detector="fixture",
        side="long",
        active_from_ts_utc=timestamp,
        lower=99.0,
        upper=100.0,
        entry_price=100.0,
        stop_loss=99.0,
    )


def short_zone(timestamp: int, *, zone_id: str = "z1") -> PendingZone:
    return PendingZone(
        zone_id=zone_id,
        detector="fixture",
        side="short",
        active_from_ts_utc=timestamp,
        lower=100.0,
        upper=101.0,
        entry_price=100.0,
        stop_loss=101.0,
    )


def test_ny_daily_aggregation_tracks_both_dst_boundaries() -> None:
    normal = m5_rows_for_trading_day(dt.date(2025, 3, 7), base=100.0)
    spring_forward = m5_rows_for_trading_day(dt.date(2025, 3, 8), base=101.0)
    frame = m5_frame(normal, spring_forward)

    result = aggregate_m5_to_ny_daily(frame)

    assert list(result.candles["component_count"]) == [288, 276]
    assert list(result.candles["trading_date_et"]) == [
        dt.date(2025, 3, 7),
        dt.date(2025, 3, 8),
    ]
    assert list(result.candles["available_ts_utc"]) == [
        int(pd.Timestamp("2025-03-08 22:00:00Z").timestamp()),
        int(pd.Timestamp("2025-03-09 21:00:00Z").timestamp()),
    ]


def test_ny_daily_aggregation_applies_a12_and_keeps_bid_ask_separate() -> None:
    accepted = m5_rows_for_trading_day(
        dt.date(2025, 1, 6), limit=DAILY_MIN_M5, base=100.0
    )
    rejected = m5_rows_for_trading_day(
        dt.date(2025, 1, 7), limit=DAILY_MIN_M5 - 1, base=200.0
    )
    result = aggregate_m5_to_ny_daily(m5_frame(accepted, rejected))

    assert len(result.candles) == 1
    day = result.candles.iloc[0]
    assert int(day["component_count"]) == DAILY_MIN_M5
    assert float(day["bid_open"]) == pytest.approx(100.0)
    assert float(day["ask_open"]) == pytest.approx(100.02)
    assert float(day["bid_close"]) == pytest.approx(100.265)
    assert float(day["ask_close"]) == pytest.approx(100.285)
    assert int(day["volume"]) == sum(range(1, DAILY_MIN_M5 + 1))
    rejected_status = result.day_status.loc[
        result.day_status["trading_date_et"] == dt.date(2025, 1, 7)
    ].iloc[0]
    assert not bool(rejected_status["accepted"])
    assert rejected_status["reason"] == "below_minimum_m5"


def test_prepare_timeframes_reuses_exact_m15_aggregation() -> None:
    rows = m5_rows_for_trading_day(dt.date(2025, 1, 6), limit=DAILY_MIN_M5)
    frame = m5_frame(rows)
    start = int(frame.iloc[0]["ts_utc"])
    end = int(frame.iloc[-1]["ts_utc"]) + 300

    prepared = prepare_timeframes(
        frame,
        start_inclusive=start,
        end_exclusive=end,
    )

    assert len(prepared.m15) == DAILY_MIN_M5 // 3
    assert prepared.incomplete_m15.empty
    assert len(prepared.daily) == 1


def test_stage3_pins_the_current_oss_release() -> None:
    assert OSS_PACKAGE_VERSION == "0.0.27"
    assert version(OSS_PACKAGE) == OSS_PACKAGE_VERSION


def test_swing_is_not_available_until_k_later_bar_closes() -> None:
    start = parse_m5_boundary("2025-01-06T00:00:00Z")
    frame = pd.DataFrame(
        {
            "ts_utc": [start + 900 * index for index in range(5)],
            "bid_high": [100.0, 101.0, 105.0, 101.5, 100.5],
            "bid_low": [99.0, 99.2, 99.4, 99.3, 99.1],
        }
    )

    swings = find_confirmed_swings(frame, k=2, bar_seconds=900)

    assert swings == (
        SwingPoint(
            direction="high",
            source_ts_utc=start + 1800,
            confirmed_at_utc=start + 4500,
            level=105.0,
        ),
    )
    assert not [swing for swing in swings if swing.confirmed_at_utc <= start + 4499]


def test_swing_low_uses_the_same_strict_k2_and_confirmation_delay() -> None:
    start = parse_m5_boundary("2025-01-06T00:00:00Z")
    frame = pd.DataFrame(
        {
            "ts_utc": [start + 900 * index for index in range(5)],
            "bid_high": [101.0, 101.2, 101.4, 101.6, 101.8],
            "bid_low": [100.0, 99.0, 95.0, 99.2, 99.8],
        }
    )

    swings = find_confirmed_swings(frame, k=2, bar_seconds=900)

    assert swings == (
        SwingPoint(
            direction="low",
            source_ts_utc=start + 1800,
            confirmed_at_utc=start + 4500,
            level=95.0,
        ),
    )


def test_daily_bias_transitions_bullish_neutral_bearish() -> None:
    swings = (
        SwingPoint("high", 1, 10, 100.0),
        SwingPoint("low", 2, 20, 90.0),
        SwingPoint("high", 3, 30, 110.0),
        SwingPoint("low", 4, 40, 95.0),
        SwingPoint("high", 5, 50, 105.0),
        SwingPoint("low", 6, 60, 85.0),
    )
    status = pd.DataFrame(
        {
            "available_ts_utc": [10, 20, 30, 40, 50, 60],
            "accepted": [True] * 6,
        }
    )

    timeline = build_daily_bias_timeline(swings, status)

    assert bias_at(timeline, 39) == "unavailable"
    assert bias_at(timeline, 40) == "bullish"
    assert bias_at(timeline, 50) == "neutral"
    assert bias_at(timeline, 60) == "bearish"


def test_rejected_daily_bar_makes_bias_unavailable_until_next_accepted_bar() -> None:
    swings = (
        SwingPoint("high", 1, 10, 100.0),
        SwingPoint("low", 2, 20, 90.0),
        SwingPoint("high", 3, 30, 110.0),
        SwingPoint("low", 4, 40, 95.0),
    )
    status = pd.DataFrame(
        {
            "available_ts_utc": [10, 20, 30, 40, 45, 55],
            "accepted": [True, True, True, True, False, True],
        }
    )

    timeline = build_daily_bias_timeline(swings, status)

    assert bias_at(timeline, 44) == "bullish"
    assert bias_at(timeline, 45) == "unavailable"
    assert bias_at(timeline, 55) == "bullish"


def test_m15_bias_annotation_uses_each_bar_close_not_future_state() -> None:
    frame = m15_frame(
        m15_row(0),
        m15_row(900),
        m15_row(1800),
    )
    timeline = (
        BiasEvent(900, "bullish", "structure_up"),
        BiasEvent(2700, "bearish", "structure_down"),
    )

    annotated = annotate_m15_bias(frame, timeline)

    assert list(annotated["daily_bias_at_open"]) == [
        "unavailable",
        "bullish",
        "bullish",
    ]
    assert list(annotated["daily_bias"]) == ["bullish", "bullish", "bearish"]


def test_long_fill_checks_same_bar_and_sl_wins_over_tp() -> None:
    frame = m15_frame(
        m15_row(0, bid_open=100.2, bid_high=102.5, bid_low=98.5, bid_close=100.0)
    )

    result = simulate_zone_backtest(
        frame,
        [long_zone(0)],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "sl"
    assert trade.exit_price == pytest.approx(99.0)
    assert trade.pnl == pytest.approx(-10_000.0)
    assert trade.equity_after == pytest.approx(990_000.0)


def test_limit_fill_bar_does_not_assume_target_happened_after_entry() -> None:
    reached_after_open = m15_row(
        parse_m5_boundary("2025-01-06T15:00:00Z"),
        bid_open=100.5,
        bid_high=102.5,
        bid_low=99.5,
        bid_close=100.5,
    )
    filled_at_open = m15_row(
        parse_m5_boundary("2025-01-06T15:15:00Z"),
        bid_open=99.98,
        bid_high=102.5,
        bid_low=99.5,
        bid_close=100.5,
    )

    delayed_result = simulate_zone_backtest(
        m15_frame(reached_after_open),
        [long_zone(int(reached_after_open["ts_utc"]))],
        target_resolver=lambda _zone, _timestamp: 102.0,
    )
    open_result = simulate_zone_backtest(
        m15_frame(filled_at_open),
        [long_zone(int(filled_at_open["ts_utc"]))],
        target_resolver=lambda _zone, _timestamp: 102.0,
    )

    assert delayed_result.trades == []
    assert delayed_result.open_position is not None
    assert open_result.trades[0].exit_reason == "tp"


def test_gap_through_entry_and_stop_uses_adverse_open_on_fill_bar() -> None:
    frame = m15_frame(
        m15_row(
            0,
            bid_open=98.5,
            bid_high=100.5,
            bid_low=98.0,
            bid_close=100.0,
        )
    )

    result = simulate_zone_backtest(
        frame,
        [long_zone(0)],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert result.trades[0].entry_price == pytest.approx(100.0)
    assert result.trades[0].exit_reason == "gap_sl"
    assert result.trades[0].exit_price == pytest.approx(98.5)


def test_short_fill_uses_bid_touch_and_ask_exit() -> None:
    frame = m15_frame(
        m15_row(
            0,
            bid_open=100.5,
            bid_high=100.2,
            bid_low=97.5,
            bid_close=99.0,
        )
    )
    # Keep the OHLC valid after choosing a lower high for the fixture.
    frame.loc[0, "bid_open"] = 100.0
    frame.loc[0, "ask_open"] = 100.02

    result = simulate_zone_backtest(
        frame,
        [short_zone(0)],
        target_resolver=lambda _zone, _timestamp: 98.0,
        **FIXTURE_WINDOW,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "tp"
    assert trade.exit_price == pytest.approx(98.0)
    assert trade.pnl == pytest.approx(20_000.0)


@pytest.mark.parametrize(
    ("second_bar", "expected_reason", "expected_price"),
    [
        (
            m15_row(
                900,
                bid_open=98.5,
                bid_high=99.2,
                bid_low=98.0,
                bid_close=98.8,
            ),
            "gap_sl",
            98.5,
        ),
        (
            m15_row(
                900,
                bid_open=102.5,
                bid_high=103.0,
                bid_low=102.2,
                bid_close=102.8,
            ),
            "tp",
            102.0,
        ),
    ],
)
def test_existing_position_applies_adverse_and_favorable_gap_rules(
    second_bar: dict[str, object],
    expected_reason: str,
    expected_price: float,
) -> None:
    first = m15_row(
        0,
        bid_open=100.2,
        bid_high=101.0,
        bid_low=99.5,
        bid_close=100.5,
    )

    result = simulate_zone_backtest(
        m15_frame(first, second_bar),
        [long_zone(0)],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == expected_reason
    assert result.trades[0].exit_price == pytest.approx(expected_price)


def test_position_is_not_forced_closed_at_end_of_data() -> None:
    frame = m15_frame(
        m15_row(
            0,
            bid_open=100.2,
            bid_high=101.0,
            bid_low=99.5,
            bid_close=100.5,
        )
    )

    result = simulate_zone_backtest(
        frame,
        [long_zone(0)],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert result.trades == []
    assert result.open_position is not None
    assert result.final_equity == INITIAL_CASH
    assert result.counters["open_at_end"] == 1


def test_article_window_excludes_warmup_zones_from_trades() -> None:
    warmup_ts = parse_m5_boundary("2024-01-05T15:00:00Z")
    article_ts = parse_m5_boundary("2024-01-06T15:00:00Z")
    frame = m15_frame(
        m15_row(
            warmup_ts,
            bid_open=99.98,
            bid_high=102.5,
            bid_low=99.5,
        ),
        m15_row(
            article_ts,
            bid_open=99.98,
            bid_high=102.5,
            bid_low=99.5,
        ),
    )

    result = simulate_zone_backtest(
        frame,
        [
            long_zone(warmup_ts, zone_id="warmup"),
            long_zone(article_ts, zone_id="article"),
        ],
        target_resolver=lambda _zone, _timestamp: 102.0,
    )

    assert [trade.zone_id for trade in result.trades] == ["article"]
    assert result.counters["zone_outside_window"] == 1


def test_article_window_checks_signal_bar_across_the_midnight_boundary() -> None:
    signal_ts = parse_m5_boundary("2024-01-06T04:45:00Z")
    active_ts = parse_m5_boundary("2024-01-06T05:00:00Z")
    zone = replace(
        long_zone(active_ts, zone_id="boundary"),
        signal_ts_utc=signal_ts,
    )
    frame = m15_frame(
        m15_row(signal_ts),
        m15_row(
            active_ts,
            bid_open=99.98,
            bid_high=102.5,
            bid_low=99.5,
        ),
    )

    result = simulate_zone_backtest(
        frame,
        [zone],
        target_resolver=lambda _zone, _timestamp: 102.0,
    )

    assert result.trades == []
    assert result.counters["zone_outside_window"] == 1


def test_new_zone_replaces_old_pending_zone_before_fill() -> None:
    old_zone = replace(long_zone(0, zone_id="old"), entry_price=99.0)
    new_zone = long_zone(900, zone_id="new")
    frame = m15_frame(
        m15_row(0, bid_low=99.5),
        m15_row(900, bid_high=102.2, bid_low=99.5),
    )

    result = simulate_zone_backtest(
        frame,
        [old_zone, new_zone],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert result.counters["zone_replaced"] == 1
    assert result.open_position is not None
    assert result.open_position.zone_id == "new"
    assert [lifecycle.end_reason for lifecycle in result.zone_lifecycles] == [
        "replaced",
        "consumed",
    ]
    assert result.zone_lifecycles[0].end_exclusive_ts_utc == 900
    assert result.zone_lifecycles[1].end_exclusive_ts_utc == 1_800


def test_exit_bar_does_not_invent_a_second_fill_path() -> None:
    first = m15_row(
        0,
        bid_open=100.2,
        bid_high=101.0,
        bid_low=99.5,
        bid_close=100.5,
    )
    exit_and_touch = m15_row(
        900,
        bid_open=100.5,
        bid_high=102.5,
        bid_low=99.5,
        bid_close=101.0,
    )

    result = simulate_zone_backtest(
        m15_frame(first, exit_and_touch),
        [long_zone(0, zone_id="first"), long_zone(900, zone_id="second")],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert [trade.zone_id for trade in result.trades] == ["first"]
    assert result.pending_zone is not None
    assert result.pending_zone.zone_id == "second"


def test_no_target_consumes_first_touched_zone_without_trade() -> None:
    frame = m15_frame(m15_row(0, bid_low=99.5))

    result = simulate_zone_backtest(
        frame,
        [long_zone(0)],
        target_resolver=lambda _zone, _timestamp: None,
        **FIXTURE_WINDOW,
    )

    assert result.trades == []
    assert result.open_position is None
    assert result.counters["no_target"] == 1
    assert result.counters["zone_consumed"] == 1
    assert result.zone_lifecycles[0].end_reason == "consumed"
    assert result.zone_lifecycles[0].end_exclusive_ts_utc == 900


def test_pending_zone_can_invalidate_without_ask_fill() -> None:
    # The deliberately wide quote keeps ask_low above the long limit even
    # though bid closes below the zone. This isolates the close invalidation.
    frame = m15_frame(
        m15_row(
            0,
            bid_open=99.5,
            bid_high=99.8,
            bid_low=98.0,
            bid_close=98.5,
            quote_width=2.1,
        )
    )

    result = simulate_zone_backtest(
        frame,
        [long_zone(0)],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert result.counters["zone_invalidated"] == 1
    assert result.open_position is None
    assert result.zone_lifecycles[0].end_reason == "invalidated"
    assert result.zone_lifecycles[0].end_exclusive_ts_utc == 900


def test_pending_zone_lifecycle_is_censored_at_observed_data_end() -> None:
    frame = m15_frame(m15_row(0, bid_low=100.1))

    result = simulate_zone_backtest(
        frame,
        [long_zone(0)],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert result.pending_zone is not None
    assert result.zone_lifecycles[0].end_reason == "data_end"
    assert result.zone_lifecycles[0].end_exclusive_ts_utc == 900


def test_invalid_target_direction_is_a_contract_error() -> None:
    frame = m15_frame(m15_row(0, bid_low=99.5))

    with pytest.raises(ValueError, match="target must be above"):
        simulate_zone_backtest(
            frame,
            [long_zone(0)],
            target_resolver=lambda _zone, _timestamp: 99.5,
            **FIXTURE_WINDOW,
        )


def test_official_detector_requires_a_full_twenty_bar_candidate_window() -> None:
    rows = [
        m15_row(
            0,
            bid_open=100.0,
            bid_high=101.0,
            bid_low=95.0,
            bid_close=99.0,
        ),
        m15_row(
            900,
            bid_open=100.0,
            bid_high=102.0,
            bid_low=96.0,
            bid_close=101.0,
        ),
    ]

    result = detect_official_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert result.zones == ()
    assert result.counters == {}


def test_official_long_candidate_activates_on_a_later_break() -> None:
    rows = official_warmup_rows()
    rows.extend(
        [
            m15_row(
                17_100,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=95.0,
                bid_close=99.0,
            ),
            m15_row(
                18_000,
                bid_open=100.0,
                bid_high=102.0,
                bid_low=96.0,
                bid_close=101.0,
            ),
            m15_row(18_900),
        ]
    )

    result = detect_official_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert len(result.zones) == 1
    zone = result.zones[0]
    assert zone.side == "long"
    assert zone.active_from_ts_utc == 18_900
    assert zone.lower == pytest.approx(95.0)
    assert zone.upper == pytest.approx(100.0)
    assert zone.entry_price == pytest.approx(100.0)
    assert zone.stop_loss == pytest.approx(95.0)
    assert result.counters["long_zone_detected"] == 1


def test_official_activation_and_invalidation_thresholds_are_strict() -> None:
    rows = official_warmup_rows()
    rows.extend(
        [
            m15_row(
                17_100,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=95.0,
                bid_close=99.0,
            ),
            m15_row(
                18_000,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=95.0,
                bid_close=100.0,
            ),
            m15_row(
                18_900,
                bid_open=100.0,
                bid_high=101.1,
                bid_low=95.0,
                bid_close=100.5,
            ),
            m15_row(19_800),
        ]
    )

    result = detect_official_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert len(result.zones) == 1
    assert result.zones[0].active_from_ts_utc == 19_800
    assert ":18900" in result.zones[0].zone_id
    assert "long_candidate_invalidated" not in result.counters


def test_official_candidate_replacement_uses_lower_low_and_largest_body() -> None:
    rows = official_warmup_rows()
    rows.extend(
        [
            m15_row(
                17_100,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=95.0,
                bid_close=99.0,
            ),
            m15_row(
                18_000,
                bid_open=100.0,
                bid_high=100.5,
                bid_low=94.0,
                bid_close=98.5,
            ),
            m15_row(
                18_900,
                bid_open=100.0,
                bid_high=100.5,
                bid_low=94.0,
                bid_close=98.0,
            ),
            m15_row(
                19_800,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=95.0,
                bid_close=100.5,
            ),
            m15_row(20_700),
        ]
    )

    result = detect_official_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert len(result.zones) == 1
    zone = result.zones[0]
    assert ":18900:" in zone.zone_id
    assert zone.lower == pytest.approx(94.0)
    assert result.counters["long_candidate_replaced"] == 2


def test_official_same_bar_break_and_new_extreme_is_not_activated() -> None:
    rows = official_warmup_rows()
    rows.extend(
        [
            m15_row(
                17_100,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=95.0,
                bid_close=99.0,
            ),
            m15_row(
                18_000,
                bid_open=100.0,
                bid_high=102.0,
                bid_low=94.0,
                bid_close=100.5,
            ),
            m15_row(18_900),
        ]
    )

    result = detect_official_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert result.zones == ()
    assert result.counters["long_candidate_invalidated"] == 1
    assert result.counters["long_ambiguous_break_invalidated"] == 1


def test_official_activation_requires_the_directional_daily_bias() -> None:
    rows = official_warmup_rows()
    rows.extend(
        [
            m15_row(
                17_100,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=95.0,
                bid_close=99.0,
            ),
            m15_row(
                18_000,
                bid_open=100.0,
                bid_high=102.0,
                bid_low=96.0,
                bid_close=101.0,
            ),
            m15_row(18_900),
        ]
    )

    result = detect_official_order_blocks(
        biased_m15_frame(rows, bias="neutral")
    )

    assert result.zones == ()
    assert result.counters["bias_unavailable"] == 1


def test_pending_official_zone_waits_for_matching_bias_before_fill() -> None:
    frame = m15_frame(
        m15_row(0, bid_low=99.5, bid_close=100.0),
        m15_row(900, bid_low=99.5, bid_close=100.0),
    )
    frame["daily_bias"] = ["bullish", "bullish"]
    frame["daily_bias_at_open"] = ["neutral", "bullish"]

    result = simulate_zone_backtest(
        frame,
        [long_zone(0)],
        target_resolver=lambda _zone, _timestamp: 102.0,
        **FIXTURE_WINDOW,
    )

    assert result.counters["bias_unavailable"] == 1
    assert result.counters["filled"] == 1
    assert result.open_position is not None
    assert result.open_position.entry_time_utc == 900


def test_backtest_rejects_close_only_bias_to_prevent_fill_lookahead() -> None:
    frame = m15_frame(m15_row(0, bid_low=99.5, bid_close=100.0))
    frame["daily_bias"] = "bullish"

    with pytest.raises(ValueError, match="must also include daily_bias_at_open"):
        simulate_zone_backtest(
            frame,
            [long_zone(0)],
            target_resolver=lambda _zone, _timestamp: 102.0,
            **FIXTURE_WINDOW,
        )


def test_official_short_detector_is_the_quote_aware_mirror() -> None:
    rows = official_warmup_rows()
    rows.extend(
        [
            m15_row(
                17_100,
                bid_open=100.0,
                bid_high=105.0,
                bid_low=99.0,
                bid_close=104.0,
            ),
            m15_row(
                18_000,
                bid_open=100.0,
                bid_high=104.0,
                bid_low=98.0,
                bid_close=99.0,
            ),
            m15_row(18_900),
        ]
    )

    result = detect_official_order_blocks(
        biased_m15_frame(rows, bias="bearish")
    )

    assert len(result.zones) == 1
    zone = result.zones[0]
    assert zone.side == "short"
    assert zone.lower == pytest.approx(100.0)
    assert zone.upper == pytest.approx(105.0)
    assert zone.entry_price == pytest.approx(100.0)
    assert zone.stop_loss == pytest.approx(105.02)


def test_official_daily_target_is_recent_confirmed_and_directional() -> None:
    day = 86_400
    starts = [day * index for index in range(65)]
    daily = pd.DataFrame(
        {
            "start_ts_utc": starts,
            "available_ts_utc": [value + day for value in starts],
        }
    )
    entry_ts = 65 * day
    swings = (
        SwingPoint("high", 4 * day, 7 * day, 110.0),
        SwingPoint("high", 10 * day, 13 * day, 108.0),
        SwingPoint("high", 50 * day, 53 * day, 103.0),
        SwingPoint("high", 55 * day, 58 * day, 99.0),
        SwingPoint("high", 60 * day, 66 * day, 105.0),
        SwingPoint("low", 51 * day, 54 * day, 97.0),
    )
    resolver = make_official_daily_target_resolver(daily, swings)
    expired_only = make_official_daily_target_resolver(daily, [swings[0]])
    future_only = make_official_daily_target_resolver(daily, [swings[4]])

    assert resolver(long_zone(0), entry_ts) == pytest.approx(103.0)
    assert resolver(short_zone(0), entry_ts) == pytest.approx(97.0)
    assert expired_only(long_zone(0), entry_ts) is None
    assert future_only(long_zone(0), entry_ts) is None


def test_official_detector_and_daily_target_run_through_shared_execution() -> None:
    rows = official_warmup_rows()
    rows.extend(
        [
            m15_row(
                17_100,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=95.0,
                bid_close=99.0,
            ),
            m15_row(
                18_000,
                bid_open=100.0,
                bid_high=102.0,
                bid_low=96.0,
                bid_close=101.0,
            ),
            m15_row(
                18_900,
                bid_open=99.97,
                bid_high=103.0,
                bid_low=99.0,
                bid_close=102.0,
            ),
        ]
    )
    frame = biased_m15_frame(rows, bias="bullish")
    detection = detect_official_order_blocks(frame)
    day = 86_400
    daily = pd.DataFrame(
        {
            "start_ts_utc": [-3 * day, -2 * day, -day],
            "available_ts_utc": [-2 * day, -day, 0],
        }
    )
    resolver = make_official_daily_target_resolver(
        daily,
        [SwingPoint("high", -2 * day, 0, 102.5)],
    )

    backtest = simulate_zone_backtest(
        frame,
        detection.zones,
        target_resolver=resolver,
        **FIXTURE_WINDOW,
    )

    assert len(backtest.trades) == 1
    trade = backtest.trades[0]
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.stop_loss == pytest.approx(95.0)
    assert trade.take_profit == pytest.approx(102.5)
    assert trade.exit_reason == "tp"


def test_secondary_long_sweep_mss_fvg_builds_full_wick_zone() -> None:
    frame = biased_m15_frame(
        secondary_long_detection_rows(),
        bias="bullish",
    )

    result = detect_secondary_order_blocks(frame)

    assert len(result.zones) == 1
    zone = result.zones[0]
    assert zone.side == "long"
    assert zone.signal_ts_utc == 8_100
    assert zone.active_from_ts_utc == 9_000
    assert zone.lower == pytest.approx(99.3)
    assert zone.upper == pytest.approx(100.5)
    assert zone.entry_price == pytest.approx(100.5)
    assert zone.stop_loss == pytest.approx(99.3)
    assert result.counters["long_sweep_detected"] == 1
    assert result.counters["long_mss_confirmed"] == 1
    assert result.counters["long_zone_detected"] == 1


def test_secondary_new_same_side_sweep_replaces_unresolved_setup() -> None:
    rows = secondary_long_reference_rows()
    rows.extend(
        [
            m15_row(
                6_300,
                bid_open=100.0,
                bid_high=100.4,
                bid_low=98.5,
                bid_close=99.5,
            ),
            m15_row(
                7_200,
                bid_open=100.0,
                bid_high=100.5,
                bid_low=98.0,
                bid_close=99.4,
            ),
            m15_row(
                8_100,
                bid_open=101.0,
                bid_high=103.5,
                bid_low=100.6,
                bid_close=103.2,
            ),
            m15_row(
                9_000,
                bid_open=103.0,
                bid_high=103.4,
                bid_low=102.0,
                bid_close=103.0,
            ),
        ]
    )

    result = detect_secondary_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert len(result.zones) == 1
    assert ":7200:7200:8100" in result.zones[0].zone_id
    assert result.zones[0].lower == pytest.approx(98.0)
    assert result.counters["long_sweep_replaced"] == 1


def test_secondary_resweep_wins_when_same_bar_also_confirms_old_mss() -> None:
    rows = secondary_long_reference_rows()
    rows.extend(
        [
            m15_row(
                6_300,
                bid_open=100.0,
                bid_high=100.4,
                bid_low=98.5,
                bid_close=99.5,
            ),
            m15_row(
                7_200,
                bid_open=101.2,
                bid_high=102.0,
                bid_low=101.1,
                bid_close=101.5,
            ),
            m15_row(
                8_100,
                bid_open=104.0,
                bid_high=104.2,
                bid_low=98.5,
                bid_close=103.2,
            ),
            m15_row(
                9_000,
                bid_open=103.0,
                bid_high=103.8,
                bid_low=102.1,
                bid_close=103.5,
            ),
            m15_row(
                9_900,
                bid_open=103.5,
                bid_high=103.7,
                bid_low=103.0,
                bid_close=103.4,
            ),
        ]
    )

    result = detect_secondary_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert len(result.zones) == 1
    assert ":8100:8100:9000" in result.zones[0].zone_id
    assert result.zones[0].signal_ts_utc == 9_000
    assert result.counters["long_sweep_replaced"] == 1
    assert result.counters["long_mss_confirmed"] == 1


@pytest.mark.parametrize(
    ("mss_offset", "expected_zones", "expected_no_mss"),
    [(12, 1, 0), (13, 0, 1)],
)
def test_secondary_mss_twelve_bar_boundary_is_inclusive(
    mss_offset: int,
    expected_zones: int,
    expected_no_mss: int,
) -> None:
    rows = secondary_long_reference_rows()
    rows.append(
        m15_row(
            6_300,
            bid_open=100.0,
            bid_high=100.4,
            bid_low=98.5,
            bid_close=99.5,
        )
    )
    rows.append(
        m15_row(
            7_200,
            bid_open=100.0,
            bid_high=100.5,
            bid_low=99.3,
            bid_close=99.8,
        )
    )
    for index in range(9, 7 + mss_offset):
        rows.append(
            m15_row(
                index * 900,
                bid_open=101.0,
                bid_high=102.5,
                bid_low=100.6,
                bid_close=102.0,
            )
        )
    mss_index = 7 + mss_offset
    rows.append(
        m15_row(
            mss_index * 900,
            bid_open=101.0,
            bid_high=103.5,
            bid_low=100.0,
            bid_close=103.2,
        )
    )
    if mss_offset == 12:
        rows.append(
            m15_row(
                (mss_index + 1) * 900,
                bid_open=103.0,
                bid_high=103.4,
                bid_low=102.0,
                bid_close=103.0,
            )
        )

    result = detect_secondary_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert len(result.zones) == expected_zones
    assert result.counters.get("long_no_mss", 0) == expected_no_mss


def test_secondary_fvg_requires_a_strict_gap() -> None:
    rows = secondary_long_detection_rows()
    rows[9] = m15_row(
        8_100,
        bid_open=101.0,
        bid_high=103.5,
        bid_low=100.4,
        bid_close=103.2,
    )

    result = detect_secondary_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert result.zones == ()
    assert result.counters["long_no_fvg"] == 1


def test_secondary_fvg_central_candle_can_be_the_sweep() -> None:
    rows = secondary_long_reference_rows()
    rows.extend(
        [
            m15_row(
                6_300,
                bid_open=100.0,
                bid_high=100.4,
                bid_low=98.5,
                bid_close=99.5,
            ),
            m15_row(
                7_200,
                bid_open=101.2,
                bid_high=103.5,
                bid_low=101.1,
                bid_close=103.2,
            ),
            m15_row(
                8_100,
                bid_open=103.0,
                bid_high=103.4,
                bid_low=102.0,
                bid_close=103.0,
            ),
        ]
    )

    result = detect_secondary_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert len(result.zones) == 1
    assert result.zones[0].lower == pytest.approx(98.5)
    assert result.zones[0].upper == pytest.approx(100.4)


def test_secondary_does_not_use_a_future_bar_for_mss_centered_fvg() -> None:
    rows = secondary_long_detection_rows()
    rows[9] = m15_row(
        8_100,
        bid_open=101.0,
        bid_high=103.5,
        bid_low=100.4,
        bid_close=103.2,
    )
    rows[10] = m15_row(
        9_000,
        bid_open=101.0,
        bid_high=103.2,
        bid_low=100.6,
        bid_close=103.0,
    )

    result = detect_secondary_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert result.zones == ()
    assert result.counters["long_no_fvg"] == 1


def test_secondary_rejects_mss_leg_without_an_opposite_candle() -> None:
    rows = secondary_long_reference_rows()
    rows.extend(
        [
            m15_row(
                6_300,
                bid_open=99.2,
                bid_high=100.4,
                bid_low=98.5,
                bid_close=99.5,
            ),
            m15_row(
                7_200,
                bid_open=101.2,
                bid_high=103.5,
                bid_low=101.1,
                bid_close=103.2,
            ),
            m15_row(
                8_100,
                bid_open=103.0,
                bid_high=103.4,
                bid_low=102.0,
                bid_close=103.0,
            ),
        ]
    )

    result = detect_secondary_order_blocks(
        biased_m15_frame(rows, bias="bullish")
    )

    assert result.zones == ()
    assert result.counters["long_no_ob"] == 1


@pytest.mark.parametrize("mismatch_index", [7, 9])
def test_secondary_setup_requires_bias_at_sweep_and_mss(
    mismatch_index: int,
) -> None:
    frame = biased_m15_frame(
        secondary_long_detection_rows(),
        bias="bullish",
    )
    frame.loc[mismatch_index, "daily_bias"] = "neutral"

    result = detect_secondary_order_blocks(frame)

    assert result.zones == ()
    assert result.counters["bias_unavailable"] == 1


def test_secondary_short_detector_is_the_quote_aware_mirror() -> None:
    rows = [
        m15_row(0, bid_open=101.0, bid_high=102.0, bid_low=99.5),
        m15_row(900, bid_open=101.0, bid_high=102.3, bid_low=99.0),
        m15_row(1_800, bid_open=101.0, bid_high=102.0, bid_low=98.0),
        m15_row(2_700, bid_open=101.0, bid_high=102.5, bid_low=99.0),
        m15_row(3_600, bid_open=101.0, bid_high=103.0, bid_low=99.2),
        m15_row(4_500, bid_open=101.0, bid_high=102.4, bid_low=99.0),
        m15_row(5_400, bid_open=101.0, bid_high=102.3, bid_low=99.2),
        m15_row(
            6_300,
            bid_open=102.0,
            bid_high=104.0,
            bid_low=101.5,
            bid_close=102.5,
        ),
        m15_row(
            7_200,
            bid_open=99.0,
            bid_high=99.1,
            bid_low=97.5,
            bid_close=97.8,
        ),
        m15_row(
            8_100,
            bid_open=98.0,
            bid_high=99.0,
            bid_low=97.0,
            bid_close=98.0,
        ),
    ]

    result = detect_secondary_order_blocks(
        biased_m15_frame(rows, bias="bearish")
    )

    assert len(result.zones) == 1
    zone = result.zones[0]
    assert zone.side == "short"
    assert zone.lower == pytest.approx(101.5)
    assert zone.upper == pytest.approx(104.0)
    assert zone.entry_price == pytest.approx(101.5)
    assert zone.stop_loss == pytest.approx(104.02)


def test_secondary_m15_target_is_recent_confirmed_and_directional() -> None:
    rows = [m15_row(index * 900) for index in range(405)]
    frame = m15_frame(*rows)
    entry_ts = 404 * 900
    swings = (
        SwingPoint("high", 3 * 900, 6 * 900, 110.0),
        SwingPoint("high", 10 * 900, 13 * 900, 103.0),
        SwingPoint("high", 300 * 900, 303 * 900, 104.0),
        SwingPoint("high", 350 * 900, 353 * 900, 99.0),
        SwingPoint("high", 399 * 900, 405 * 900, 105.0),
        SwingPoint("low", 301 * 900, 304 * 900, 97.0),
    )
    resolver = make_secondary_m15_target_resolver(frame, swings)
    expired_only = make_secondary_m15_target_resolver(frame, [swings[0]])
    future_only = make_secondary_m15_target_resolver(frame, [swings[4]])

    assert resolver(long_zone(0), entry_ts) == pytest.approx(104.0)
    assert resolver(short_zone(0), entry_ts) == pytest.approx(97.0)
    assert expired_only(long_zone(0), entry_ts) is None
    assert future_only(long_zone(0), entry_ts) is None


def test_secondary_detector_and_m15_target_use_shared_execution() -> None:
    frame = biased_m15_frame(
        secondary_long_detection_rows(),
        bias="bullish",
    )
    detection = detect_secondary_order_blocks(frame)
    resolver = make_secondary_m15_target_resolver(frame)

    backtest = simulate_zone_backtest(
        frame,
        detection.zones,
        target_resolver=resolver,
        **FIXTURE_WINDOW,
    )

    assert len(backtest.trades) == 1
    trade = backtest.trades[0]
    assert trade.detector == "ict_secondary_17m"
    assert trade.entry_price == pytest.approx(100.5)
    assert trade.stop_loss == pytest.approx(99.3)
    assert trade.take_profit == pytest.approx(103.0)
    assert trade.exit_reason == "tp"
