"""Fixture and regression tests for the Season 2 chapter 3 M15 ORB."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from bocchi_the_botter_repro.season2.minute_data import M5Audit
from bocchi_the_botter_repro.season2.orb_m15 import (
    ARTICLE_WINDOW_END_EXCLUSIVE,
    ARTICLE_WINDOW_START,
    EXPECTED_COMPLETE_M15,
    EXPECTED_INCOMPLETE_M15,
    EXPECTED_INPUT_SHA256,
    EXPECTED_M5_ROWS,
    INITIAL_CASH,
    N_SEGMENTS,
    SEGMENT_DAYS,
    WINDOW_DAYS,
    BacktestResult,
    BacktestSummary,
    FrozenInputError,
    ReferenceVerificationError,
    Trade,
    add_bid_atr,
    expected_session_timestamps,
    fixed_segment_edges,
    fixed_segment_summary,
    simulate_sessions,
    summarize_trades,
    trades_frame,
    validate_frozen_input,
    validate_m15_frame,
    verify_private_against_reference,
    verify_row_free_reference,
    write_private_audit,
    write_reference_outputs,
)


SESSION_DATE = dt.date(2025, 3, 3)


def m15_row(
    timestamp: int,
    *,
    bid_open: float = 100.0,
    bid_high: float = 100.5,
    bid_low: float = 99.5,
    bid_close: float = 100.0,
    quote_width: float = 0.02,
) -> dict[str, object]:
    utc = pd.to_datetime(timestamp, unit="s", utc=True)
    et = utc.tz_convert("America/New_York")
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
        "ts_utc_dt": utc,
        "ts_et": et,
        "session_date_et": et.date(),
        "bid_atr": 1.0,
    }


def session_frame(
    *,
    signal_time: dt.time | None = dt.time(9, 45),
    side: str = "long",
    range_width: float = 1.5,
    session_date: dt.date = SESSION_DATE,
) -> pd.DataFrame:
    rows = [m15_row(timestamp) for timestamp in expected_session_timestamps(session_date)]
    center = 100.0
    range_low = center - range_width / 2
    range_high = center + range_width / 2
    rows[0].update(
        {
            "bid_open": center,
            "bid_high": range_high,
            "bid_low": range_low,
            "bid_close": center,
            "ask_open": center + 0.02,
            "ask_high": range_high + 0.02,
            "ask_low": range_low + 0.02,
            "ask_close": center + 0.02,
        }
    )
    if signal_time is not None:
        signal_index = next(
            index
            for index, row in enumerate(rows)
            if pd.Timestamp(row["ts_et"]).time() == signal_time
        )
        if side == "long":
            rows[signal_index].update(
                {
                    "bid_open": 100.4,
                    "bid_high": range_high + 0.35,
                    "bid_low": 100.2,
                    "bid_close": range_high + 0.25,
                    "ask_open": 100.42,
                    "ask_high": range_high + 0.37,
                    "ask_low": 100.22,
                    "ask_close": range_high + 0.27,
                }
            )
            if signal_index + 1 < len(rows):
                rows[signal_index + 1].update(
                    {
                        "bid_open": range_high + 0.20,
                        "bid_high": range_high + 0.40,
                        "bid_low": range_high + 0.10,
                        "bid_close": range_high + 0.25,
                        "ask_open": range_high + 0.22,
                        "ask_high": range_high + 0.42,
                        "ask_low": range_high + 0.12,
                        "ask_close": range_high + 0.27,
                    }
                )
        elif side == "short":
            rows[signal_index].update(
                {
                    "bid_open": 99.6,
                    "bid_high": 99.8,
                    "bid_low": range_low - 0.35,
                    "bid_close": range_low - 0.25,
                    "ask_open": 99.62,
                    "ask_high": 99.82,
                    "ask_low": range_low - 0.33,
                    "ask_close": range_low - 0.23,
                }
            )
            if signal_index + 1 < len(rows):
                rows[signal_index + 1].update(
                    {
                        "bid_open": range_low - 0.20,
                        "bid_high": range_low - 0.10,
                        "bid_low": range_low - 0.40,
                        "bid_close": range_low - 0.25,
                        "ask_open": range_low - 0.18,
                        "ask_high": range_low - 0.08,
                        "ask_low": range_low - 0.38,
                        "ask_close": range_low - 0.23,
                    }
                )
        else:
            raise ValueError(side)
    return pd.DataFrame(rows).sort_values("ts_utc").reset_index(drop=True)


def run_one_session(frame: pd.DataFrame) -> tuple[list[Trade], dict[str, int]]:
    trades, outcomes, _, _ = simulate_sessions(
        frame,
        window_start=SESSION_DATE,
        window_end_exclusive=SESSION_DATE + dt.timedelta(days=1),
    )
    return trades, outcomes


def frozen_audit() -> M5Audit:
    return M5Audit(
        row_count=EXPECTED_M5_ROWS,
        first_ts_utc=1,
        last_ts_utc=2,
        duplicate_count=0,
        off_boundary_count=0,
        null_required_count=0,
        invalid_volume_count=0,
        invalid_ohlc_count=0,
        negative_spread_count=0,
        sorted_ascending=True,
        gap_count=192,
        missing_m5_slots=77_276,
        weekend_gap_count=132,
        long_non_weekend_gap_count=4,
        short_gap_count=56,
        extraction_sha256=EXPECTED_INPUT_SHA256,
    )


def test_fixed_window_is_five_exact_184_day_segments() -> None:
    edges = fixed_segment_edges()

    assert ARTICLE_WINDOW_END_EXCLUSIVE - ARTICLE_WINDOW_START == dt.timedelta(
        days=WINDOW_DAYS
    )
    assert len(edges) == N_SEGMENTS + 1
    assert edges[0] == ARTICLE_WINDOW_START
    assert edges[-1] == ARTICLE_WINDOW_END_EXCLUSIVE
    assert all(
        right - left == dt.timedelta(days=SEGMENT_DAYS)
        for left, right in zip(edges, edges[1:])
    )


def test_session_contract_contains_27_starts_through_1600_et() -> None:
    timestamps = expected_session_timestamps(SESSION_DATE)
    local = pd.to_datetime(list(timestamps), unit="s", utc=True).tz_convert(
        "America/New_York"
    )

    assert len(timestamps) == 27
    assert local[0].time() == dt.time(9, 30)
    assert local[-1].time() == dt.time(16, 0)


def test_session_timestamp_selection_tracks_new_york_dst() -> None:
    before = expected_session_timestamps(dt.date(2025, 3, 7))[0]
    after = expected_session_timestamps(dt.date(2025, 3, 10))[0]

    assert pd.to_datetime(before, unit="s", utc=True).time() == dt.time(14, 30)
    assert pd.to_datetime(after, unit="s", utc=True).time() == dt.time(13, 30)


def test_missing_one_bar_invalidates_the_whole_session() -> None:
    frame = session_frame().drop(index=8).reset_index(drop=True)

    trades, outcomes = run_one_session(frame)

    assert trades == []
    assert outcomes == {"invalid_session": 1}


def test_zero_row_session_is_detected_without_external_calendar() -> None:
    frame = session_frame(session_date=SESSION_DATE + dt.timedelta(days=1))

    trades, outcomes = run_one_session(frame)

    assert trades == []
    assert outcomes == {"invalid_session": 1}


def test_atr_uses_bid_true_range_without_future_rows() -> None:
    frame = session_frame(signal_time=None)
    prior = []
    first_ts = int(frame.iloc[0]["ts_utc"])
    for index in range(13, 0, -1):
        prior.append(m15_row(first_ts - index * 900))
    combined = pd.concat([pd.DataFrame(prior), frame], ignore_index=True)
    baseline = add_bid_atr(combined)
    changed = combined.copy()
    changed.loc[14, ["bid_high", "ask_high"]] = [150.0, 150.02]
    changed_atr = add_bid_atr(changed)

    assert baseline.loc[13, "bid_atr"] == pytest.approx(14.5 / 14)
    assert changed_atr.loc[13, "bid_atr"] == pytest.approx(baseline.loc[13, "bid_atr"])
    assert changed_atr.loc[14, "bid_atr"] > baseline.loc[14, "bid_atr"]


def test_true_range_takes_the_maximum_of_all_three_candidates() -> None:
    frame = session_frame(signal_time=None)
    frame.loc[0, ["bid_close", "ask_close"]] = [100.0, 100.02]
    frame.loc[1, ["bid_open", "bid_high", "bid_low", "bid_close"]] = [
        102.5,
        103.0,
        102.0,
        102.5,
    ]
    frame.loc[1, ["ask_open", "ask_high", "ask_low", "ask_close"]] = [
        102.52,
        103.02,
        102.02,
        102.52,
    ]

    prepared = add_bid_atr(frame)

    assert prepared.loc[1, "bid_tr"] == pytest.approx(3.0)


def test_atr_unavailable_is_recorded_as_a_skip_reason() -> None:
    frame = session_frame()
    frame.loc[0, "bid_atr"] = float("nan")

    trades, outcomes = run_one_session(frame)

    assert trades == []
    assert outcomes == {"atr_unavailable": 1}


@pytest.mark.parametrize("range_width", [1.25, 3.0])
def test_atr_filter_includes_both_boundaries(range_width: float) -> None:
    trades, outcomes = run_one_session(session_frame(range_width=range_width))

    assert len(trades) == 1
    assert outcomes["atr_passed"] == 1
    assert outcomes["traded"] == 1


def test_long_uses_bid_signal_ask_entry_and_bid_exit() -> None:
    frame = session_frame(side="long")
    frame.loc[frame.index[-1], ["bid_open", "ask_open"]] = [101.25, 101.27]

    trades, outcomes = run_one_session(frame)

    assert outcomes["traded"] == 1
    trade = trades[0]
    range_high = float(frame.iloc[0]["bid_high"])
    assert trade.side == "long"
    assert trade.entry_time_utc == int(frame.iloc[2]["ts_utc"])
    assert trade.entry_price == pytest.approx(range_high + 0.22)
    assert trade.stop_loss == pytest.approx(frame.iloc[0]["bid_low"])
    assert trade.exit_price == pytest.approx(101.25)
    assert trade.exit_reason == "close_16"
    assert trade.pnl == pytest.approx(trade.units * (101.25 - trade.entry_price))


def test_short_uses_bid_entry_ask_stop_and_ask_exit() -> None:
    frame = session_frame(side="short")
    frame.loc[frame.index[-1], ["bid_open", "ask_open"]] = [98.70, 98.72]

    trades, _ = run_one_session(frame)
    trade = trades[0]

    assert trade.side == "short"
    assert trade.entry_price == pytest.approx(frame.iloc[2]["bid_open"])
    assert trade.stop_loss == pytest.approx(frame.iloc[0]["ask_high"])
    assert trade.exit_price == pytest.approx(98.72)
    assert trade.pnl == pytest.approx(trade.units * (trade.entry_price - 98.72))


def test_entry_bar_is_checked_and_sl_wins_when_both_levels_are_touched() -> None:
    frame = session_frame(side="long")
    range_low = float(frame.iloc[0]["bid_low"])
    entry = float(frame.iloc[2]["ask_open"])
    target = entry + 1.5 * (entry - range_low)
    frame.loc[2, ["bid_low", "ask_low"]] = [range_low - 0.1, range_low - 0.08]
    frame.loc[2, ["bid_high", "ask_high"]] = [target + 0.1, target + 0.12]

    trades, _ = run_one_session(frame)

    assert trades[0].exit_time_utc == int(frame.iloc[2]["ts_utc"])
    assert trades[0].exit_reason == "sl"
    assert trades[0].exit_price == pytest.approx(range_low)


def test_adverse_open_gap_is_filled_at_the_later_bar_open() -> None:
    frame = session_frame(side="long")
    stop = float(frame.iloc[0]["bid_low"])
    frame.loc[3, ["bid_open", "ask_open"]] = [stop - 0.2, stop - 0.18]
    frame.loc[3, ["bid_low", "ask_low"]] = [stop - 0.3, stop - 0.28]
    frame.loc[3, ["bid_high", "ask_high"]] = [stop, stop + 0.02]
    frame.loc[3, ["bid_close", "ask_close"]] = [stop - 0.1, stop - 0.08]

    trades, _ = run_one_session(frame)

    assert trades[0].exit_reason == "gap_sl"
    assert trades[0].exit_price == pytest.approx(stop - 0.2)


def test_favorable_open_gap_is_conservatively_filled_at_target() -> None:
    frame = session_frame(side="long")
    stop = float(frame.iloc[0]["bid_low"])
    entry = float(frame.iloc[2]["ask_open"])
    target = entry + 1.5 * (entry - stop)
    frame.loc[3, ["bid_open", "ask_open"]] = [target + 0.2, target + 0.22]
    frame.loc[3, ["bid_low", "ask_low"]] = [target + 0.1, target + 0.12]
    frame.loc[3, ["bid_high", "ask_high"]] = [target + 0.3, target + 0.32]
    frame.loc[3, ["bid_close", "ask_close"]] = [target + 0.2, target + 0.22]

    trades, _ = run_one_session(frame)

    assert trades[0].exit_reason == "tp"
    assert trades[0].exit_price == pytest.approx(target)


def test_1600_open_exit_precedes_gap_and_range_checks() -> None:
    frame = session_frame(side="long")
    stop = float(frame.iloc[0]["bid_low"])
    frame.loc[frame.index[-1], ["bid_open", "ask_open"]] = [stop - 1.0, stop - 0.98]
    frame.loc[frame.index[-1], ["bid_low", "ask_low"]] = [stop - 2.0, stop - 1.98]
    frame.loc[frame.index[-1], ["bid_high", "ask_high"]] = [105.0, 105.02]
    frame.loc[frame.index[-1], ["bid_close", "ask_close"]] = [100.0, 100.02]

    trades, _ = run_one_session(frame)

    assert trades[0].exit_reason == "close_16"
    assert trades[0].exit_price == pytest.approx(stop - 1.0)


def test_1145_entry_is_allowed_but_1200_entry_is_rejected() -> None:
    allowed, allowed_outcomes = run_one_session(
        session_frame(signal_time=dt.time(11, 30))
    )
    rejected, rejected_outcomes = run_one_session(
        session_frame(signal_time=dt.time(11, 45))
    )

    assert pd.to_datetime(allowed[0].entry_time_utc, unit="s", utc=True).tz_convert(
        "America/New_York"
    ).time() == dt.time(11, 45)
    assert allowed_outcomes["traded"] == 1
    assert rejected == []
    assert rejected_outcomes["cutoff"] == 1


def test_first_breakout_is_not_replaced_after_cutoff() -> None:
    frame = session_frame(signal_time=dt.time(11, 45), side="long")
    later = frame["ts_et"].map(lambda value: pd.Timestamp(value).time()) == dt.time(12, 15)
    frame.loc[later, ["bid_close", "ask_close"]] = [98.0, 98.02]
    frame.loc[later, ["bid_low", "ask_low"]] = [97.9, 97.92]

    trades, outcomes = run_one_session(frame)

    assert trades == []
    assert outcomes["cutoff"] == 1


def test_margin_cap_rejects_risk_units_instead_of_clipping_them() -> None:
    frame = session_frame(range_width=0.015)
    frame["bid_atr"] = 0.01
    range_high = float(frame.iloc[0]["bid_high"])
    frame.loc[1, ["bid_open", "bid_high", "bid_low", "bid_close"]] = [
        range_high,
        range_high + 0.002,
        range_high - 0.001,
        range_high + 0.001,
    ]
    frame.loc[1, ["ask_open", "ask_high", "ask_low", "ask_close"]] = [
        range_high + 0.02,
        range_high + 0.022,
        range_high + 0.019,
        range_high + 0.021,
    ]
    frame.loc[2, ["bid_open", "bid_high", "bid_low", "bid_close"]] = [
        range_high - 0.004,
        range_high - 0.002,
        range_high - 0.006,
        range_high - 0.004,
    ]
    frame.loc[2, ["ask_open", "ask_high", "ask_low", "ask_close"]] = [
        range_high + 0.016,
        range_high + 0.018,
        range_high + 0.014,
        range_high + 0.016,
    ]

    trades, outcomes = run_one_session(frame)

    assert trades == []
    assert outcomes["sizing_rejected"] == 1


def test_frozen_input_mismatch_stops_before_article_results() -> None:
    audit = replace(frozen_audit(), extraction_sha256="0" * 64)

    with pytest.raises(FrozenInputError, match="extraction_sha256"):
        validate_frozen_input(
            audit,
            complete_m15_count=EXPECTED_COMPLETE_M15,
            incomplete_m15_count=EXPECTED_INCOMPLETE_M15,
        )


def test_m15_validation_rejects_duplicate_and_incomplete_rows() -> None:
    frame = session_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True).sort_values(
        "ts_utc"
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_m15_frame(duplicated)

    incomplete = frame.copy()
    incomplete.loc[0, "component_count"] = 2
    with pytest.raises(ValueError, match="incomplete"):
        validate_m15_frame(incomplete)


def test_m15_validation_rejects_reversed_order_and_missing_columns() -> None:
    frame = session_frame()
    with pytest.raises(ValueError, match="not sorted"):
        validate_m15_frame(frame.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="missing columns"):
        validate_m15_frame(frame.drop(columns=["ask_close"]))


def two_trade_result() -> BacktestResult:
    first = Trade(
        session_date_et=ARTICLE_WINDOW_START,
        side="long",
        signal_time_utc=1,
        entry_time_utc=2,
        entry_price=100.02,
        stop_loss=99.0,
        take_profit=101.55,
        initial_risk=1.02,
        units=1_000,
        exit_time_utc=3,
        exit_price=101.04,
        exit_reason="tp",
        pnl=1_020.0,
        equity_before=INITIAL_CASH,
        equity_after=INITIAL_CASH + 1_020.0,
        realized_r=1.0,
        entry_quote_width=0.02,
    )
    second = Trade(
        session_date_et=ARTICLE_WINDOW_START + dt.timedelta(days=SEGMENT_DAYS),
        side="short",
        signal_time_utc=4,
        entry_time_utc=5,
        entry_price=100.0,
        stop_loss=101.0,
        take_profit=98.5,
        initial_risk=1.0,
        units=1_000,
        exit_time_utc=6,
        exit_price=100.5,
        exit_reason="sl",
        pnl=-500.0,
        equity_before=INITIAL_CASH + 1_020.0,
        equity_after=INITIAL_CASH + 520.0,
        realized_r=-0.5,
        entry_quote_width=0.02,
    )
    private = trades_frame([first, second])
    segments = fixed_segment_summary(private)
    summary = summarize_trades(private, segments)
    return BacktestResult(
        input_audit=frozen_audit(),
        complete_m15_count=EXPECTED_COMPLETE_M15,
        incomplete_m15_count=EXPECTED_INCOMPLETE_M15,
        trades=[first, second],
        summary=summary,
        segments=segments,
        exit_reasons={"sl": 1, "tp": 1},
        terminal_outcomes={"invalid_session": 918, "atr_passed": 2, "traded": 2},
        session_quality={
            "candidate_sessions": 920,
            "valid_sessions": 2,
            "invalid_sessions": 918,
        },
        atr_filter={
            "count": 2,
            "below_lower": 0,
            "passed": 2,
            "above_upper": 0,
        },
        entry_quote_width={"count": 2, "mean": 0.02},
        m15_spread_open={"count": 2, "mean": 0.02},
        m15_spread_close={"count": 2, "mean": 0.02},
    )


def test_public_outputs_exclude_prices_timestamps_and_trade_rows(tmp_path: Path) -> None:
    result = two_trade_result()
    private_dir = tmp_path / "private"
    public_dir = tmp_path / "public"
    write_private_audit(result, private_dir)
    write_reference_outputs(result, public_dir)

    assert (private_dir / "trades_private.csv").is_file()
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in public_dir.iterdir()
        if path.suffix in {".json", ".csv"}
    )
    assert "entry_price" not in public_text
    assert "exit_price" not in public_text
    assert "entry_time_utc" not in public_text
    assert not (public_dir / "trades.csv").exists()


def test_independent_verifier_recomputes_private_pnl_equity_and_segments(
    tmp_path: Path,
) -> None:
    result = two_trade_result()
    private_dir = tmp_path / "private"
    public_dir = tmp_path / "public"
    write_private_audit(result, private_dir)
    write_reference_outputs(result, public_dir)

    recomputed = verify_private_against_reference(
        private_dir / "trades_private.csv", public_dir
    )

    assert recomputed["trade_count"] == 2
    assert recomputed["final_equity"] == pytest.approx(INITIAL_CASH + 520.0)


def test_row_free_verifier_checks_hashes_counts_and_public_boundary(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    write_reference_outputs(two_trade_result(), public_dir)

    payload = verify_row_free_reference(public_dir)

    assert payload["summary"]["trade_count"] == 2


def test_row_free_verifier_rejects_tampered_aggregate(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    write_reference_outputs(two_trade_result(), public_dir)
    path = public_dir / "segments.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("1020.0", "1021.0"), encoding="utf-8")

    with pytest.raises(ReferenceVerificationError, match="hash mismatch"):
        verify_row_free_reference(public_dir)


def test_independent_verifier_rejects_tampered_pnl(tmp_path: Path) -> None:
    result = two_trade_result()
    private_dir = tmp_path / "private"
    public_dir = tmp_path / "public"
    write_private_audit(result, private_dir)
    write_reference_outputs(result, public_dir)
    path = private_dir / "trades_private.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "pnl"] += 1.0
    frame.to_csv(path, index=False)

    with pytest.raises(ReferenceVerificationError, match="PnL"):
        verify_private_against_reference(path, public_dir)


def test_reference_summary_declares_quote_model_and_zero_fixed_spread(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    write_reference_outputs(two_trade_result(), public_dir)
    payload = json.loads(
        (public_dir / "reference_summary.json").read_text(encoding="utf-8")
    )

    assert payload["execution"]["signal"] == "bid OHLC"
    assert payload["execution"]["long"] == "entry ask_open; exit bid"
    assert payload["execution"]["fixed_spread_addition"] == 0
