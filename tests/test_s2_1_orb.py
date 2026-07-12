"""Regression tests for the Season 2 chapter 1 ORB reproduction."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from bocchi_the_botter_repro.season2.orb import (
    ARTICLE_WINDOW_END_EXCLUSIVE,
    ARTICLE_WINDOW_START,
    EXPECTED_MAIN_TRADES,
    EXPECTED_REF_TRADES,
    N_SEGMENTS,
    SEGMENT_DAYS,
    WINDOW_DAYS,
    DataCoverageError,
    ReferenceDataError,
    article_window,
    fixed_segment_edges,
    fixed_segment_summary,
    run_reference,
    simulate,
    verify_reference_csvs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "results" / "reference" / "ch01_orb_1h_translation"


def test_article_window_is_fixed_at_exactly_720_calendar_days() -> None:
    assert ARTICLE_WINDOW_END_EXCLUSIVE - ARTICLE_WINDOW_START == dt.timedelta(
        days=WINDOW_DAYS
    )

    index = pd.date_range(
        "2024-07-20", "2026-07-11", freq="D", tz="America/New_York"
    )
    frame = pd.DataFrame({"et_date": index.date}, index=index)
    result = article_window(frame)

    assert min(result["et_date"]) == ARTICLE_WINDOW_START
    assert max(result["et_date"]) == (
        ARTICLE_WINDOW_END_EXCLUSIVE - dt.timedelta(days=1)
    )
    assert result["et_date"].nunique() == WINDOW_DAYS


def test_article_window_rejects_live_history_that_no_longer_reaches_start() -> None:
    index = pd.date_range(
        "2024-07-22", "2026-07-10", freq="D", tz="America/New_York"
    )
    frame = pd.DataFrame({"et_date": index.date}, index=index)

    with pytest.raises(DataCoverageError, match="does not cover"):
        article_window(frame)


def test_fixed_segment_edges_are_five_consecutive_144_day_ranges() -> None:
    edges = fixed_segment_edges()

    assert len(edges) == N_SEGMENTS + 1
    assert SEGMENT_DAYS == 144
    assert edges[0] == pd.Timestamp("2024-07-21")
    assert edges[-1] == pd.Timestamp("2026-07-11")
    assert all(
        edges[index + 1] - edges[index] == pd.Timedelta(days=SEGMENT_DAYS)
        for index in range(N_SEGMENTS)
    )


def test_boundary_days_belong_only_to_the_following_segment() -> None:
    start = pd.Timestamp(ARTICLE_WINDOW_START)
    offsets = [0, 143, 144, 287, 288, 431, 432, 575, 576, 719]
    trades = pd.DataFrame(
        {
            "date": [start + pd.Timedelta(days=offset) for offset in offsets],
            "pnl": list(range(1, 11)),
        }
    )

    segments = fixed_segment_summary(trades)

    assert segments["trade_count"].tolist() == [2, 2, 2, 2, 2]
    assert segments["pnl"].tolist() == pytest.approx([3, 7, 11, 15, 19])
    assert int(segments["trade_count"].sum()) == len(trades)


def test_trade_on_article_window_end_is_rejected() -> None:
    trades = pd.DataFrame(
        {"date": [pd.Timestamp(ARTICLE_WINDOW_END_EXCLUSIVE)], "pnl": [1.0]}
    )

    with pytest.raises(ValueError, match="outside"):
        fixed_segment_summary(trades)


def test_simulate_enters_next_bar_and_closes_at_1600_et() -> None:
    index = pd.DatetimeIndex(
        [
            "2025-01-02 09:00",
            "2025-01-02 10:00",
            "2025-01-02 11:00",
            "2025-01-02 16:00",
        ]
    ).tz_localize("America/New_York")
    frame = pd.DataFrame(
        {
            "Open": [100.2, 100.5, 101.1, 101.6],
            "High": [101.0, 101.3, 101.6, 101.8],
            "Low": [100.0, 100.4, 100.8, 101.4],
            "Close": [100.5, 101.2, 101.4, 101.7],
            "ATR": [0.5, 0.5, 0.5, 0.5],
            "et_date": index.date,
            "et_hour": index.hour,
        },
        index=index,
    )

    trades, missed = simulate(frame, cutoff=True, spread=0.0)

    assert missed == 0
    assert len(trades) == 1
    trade = trades[0]
    assert trade.side == "long"
    assert trade.entry_time == index[2]
    assert trade.exit_time == index[3]
    assert trade.exit_reason == "close_16"
    assert trade.entry_ref == pytest.approx(101.1)
    assert trade.exit_price == pytest.approx(101.6)
    assert trade.pnl > 0


def test_frozen_reference_csvs_reconstruct_article_results() -> None:
    result = verify_reference_csvs(REFERENCE_DIR)

    assert result.main_summary.n == EXPECTED_MAIN_TRADES
    assert result.ref_summary.n == EXPECTED_REF_TRADES
    assert result.main_summary.return_pct == pytest.approx(3.036561, abs=1e-6)
    assert result.ref_summary.return_pct == pytest.approx(6.461724, abs=1e-6)
    assert result.main_summary.positive_segments == 3
    assert result.main_summary.exit_counts == {"close_16": 76, "sl": 5, "tp": 1}
    assert result.segments["trade_count"].tolist() == [19, 22, 18, 13, 10]
    assert result.segments["pnl"].tolist() == pytest.approx(
        [29_606.50, 18_419.42, 12_328.85, -9_219.02, -20_770.14],
        abs=0.01,
    )


def test_reference_mode_writes_only_recomputed_outputs(tmp_path: Path) -> None:
    result = run_reference(REFERENCE_DIR, tmp_path)

    assert result.main_summary.n == EXPECTED_MAIN_TRADES
    assert (tmp_path / "reference_summary.csv").is_file()
    assert (
        tmp_path / "segments_S2-1_ORB_USDJPY_main_net.csv"
    ).is_file()
    assert not (tmp_path / "trades_S2-1_ORB_USDJPY_main_net.csv").exists()


def test_reference_validation_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReferenceDataError, match="not found"):
        verify_reference_csvs(tmp_path)
