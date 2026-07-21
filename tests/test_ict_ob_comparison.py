"""Artificial-fixture tests for the stage-7 ICT OB comparison."""
from __future__ import annotations

import pandas as pd

from bocchi_the_botter_repro.season2.ict_ob_comparison import (
    OSS_DETECTOR,
    ZoneWindow,
    build_oss_liquidity_records,
    build_oss_zone_windows,
    compare_zone_windows,
    zones_overlap,
)


def window(
    zone_id: str,
    *,
    source: str = "left",
    start: int = 0,
    end: int = 900,
    lower: float = 100.0,
    upper: float = 101.0,
) -> ZoneWindow:
    return ZoneWindow(
        zone_id=zone_id,
        source=source,
        side="long",
        active_from_ts_utc=start,
        end_exclusive_ts_utc=end,
        lower=lower,
        upper=upper,
        end_reason="consumed",
    )


def test_overlap_uses_closed_prices_and_half_open_time() -> None:
    price_boundary_touch = window(
        "right-price",
        source="right",
        lower=101.0,
        upper=102.0,
    )
    time_boundary_touch = window(
        "right-time",
        source="right",
        start=900,
        end=1_800,
    )

    assert zones_overlap(window("left"), price_boundary_touch) is True
    assert zones_overlap(window("left"), time_boundary_touch) is False


def test_zero_duration_zone_never_overlaps() -> None:
    zero_duration = window("zero", start=300, end=300)

    assert zones_overlap(zero_duration, window("other")) is False


def test_overlap_rates_count_each_zone_once_but_keep_all_pairs() -> None:
    left = [
        window("left-1", end=2_000),
        window("left-2", start=3_000, end=4_000),
    ]
    right = [
        window("right-1", source="right", end=1_000),
        window("right-2", source="right", start=500, end=1_500),
        window("right-3", source="right", start=5_000, end=6_000),
    ]

    summary = compare_zone_windows(
        left,
        right,
        left_source="left",
        right_source="right",
    )

    assert summary.left_overlapped == 1
    assert summary.left_overlap_pct == 50.0
    assert summary.right_overlapped == 2
    assert summary.right_overlap_pct == 2 / 3 * 100.0
    assert summary.overlapping_pair_count == 2


def test_oss_adapter_recovers_confirmation_and_mitigation_bar() -> None:
    base = int(pd.Timestamp("2024-01-08T00:00:00Z").timestamp())
    timestamps = [base + index * 900 for index in range(6)]
    m15 = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "volume": [1, 2, 3, 4, 5, 6],
            "bid_open": [9.5, 11.0, 9.0, 9.5, 12.0, 10.0],
            "bid_high": [10.0, 12.0, 11.0, 10.0, 13.0, 12.0],
            "bid_low": [9.0, 10.0, 8.0, 9.0, 11.0, 7.0],
            "bid_close": [9.5, 11.0, 10.0, 9.5, 12.5, 10.0],
        }
    )
    ohlcv = pd.DataFrame(
        {
            "open": m15["bid_open"],
            "high": m15["bid_high"],
            "low": m15["bid_low"],
            "close": m15["bid_close"],
            "volume": m15["volume"],
        }
    )
    swings = pd.DataFrame(
        {
            "HighLow": [float("nan"), 1.0, float("nan"), float("nan"), float("nan"), float("nan")],
            "Level": [float("nan"), 12.0, float("nan"), float("nan"), float("nan"), float("nan")],
        }
    )
    ob_output = pd.DataFrame(
        {
            "OB": [float("nan"), float("nan"), 1.0, float("nan"), float("nan"), float("nan")],
            "Top": [float("nan"), float("nan"), 11.0, float("nan"), float("nan"), float("nan")],
            "Bottom": [float("nan"), float("nan"), 8.0, float("nan"), float("nan"), float("nan")],
            "MitigatedIndex": [float("nan"), float("nan"), 4.0, float("nan"), float("nan"), float("nan")],
        }
    )

    result = build_oss_zone_windows(m15, ohlcv, swings, ob_output)

    assert len(result) == 1
    assert result[0].source == OSS_DETECTOR
    assert result[0].active_from_ts_utc == timestamps[4] + 900
    assert result[0].end_exclusive_ts_utc == timestamps[5] + 900
    assert result[0].end_reason == "mitigated"
    assert (result[0].lower, result[0].upper) == (8.0, 11.0)


def test_liquidity_adapter_keeps_library_indices_without_trading() -> None:
    base = int(pd.Timestamp("2024-01-08T00:00:00Z").timestamp())
    m15 = pd.DataFrame({"ts_utc": [base + index * 900 for index in range(4)]})
    output = pd.DataFrame(
        {
            "Liquidity": [1.0, float("nan"), float("nan"), float("nan")],
            "Level": [145.0, float("nan"), float("nan"), float("nan")],
            "End": [2.0, float("nan"), float("nan"), float("nan")],
            "Swept": [3.0, float("nan"), float("nan"), float("nan")],
        }
    )

    records = build_oss_liquidity_records(m15, output)

    assert len(records) == 1
    assert records[0].source_ts_utc == base
    assert records[0].group_end_ts_utc == base + 1_800
    assert records[0].swept_ts_utc == base + 2_700
