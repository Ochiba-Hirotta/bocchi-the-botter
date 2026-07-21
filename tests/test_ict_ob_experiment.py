"""Stage-6 audit and reproducibility tests for the ICT OB experiment."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from bocchi_the_botter_repro.season2.ict_ob import (
    INITIAL_CASH,
    OpenPosition,
    PendingZone,
    ZoneBacktestResult,
    ZoneTrade,
)
from bocchi_the_botter_repro.season2.ict_ob_experiment import (
    assert_reproducible,
    deterministic_sha256,
    fixed_segment_summary,
    summarize_detector,
    trades_frame,
    zones_frame,
)
from bocchi_the_botter_repro.season2.orb_m15 import FrozenInputError


def zone(*, suffix: str = "a") -> PendingZone:
    return PendingZone(
        zone_id=f"official:long:{suffix}",
        detector="ict_month04",
        side="long",
        active_from_ts_utc=1_704_520_800,
        lower=145.0,
        upper=145.2,
        entry_price=145.2,
        stop_loss=145.0,
        signal_ts_utc=1_704_519_900,
    )


def trade(
    *,
    entry_time_utc: int = 1_704_520_800,
    pnl: float = 2_000.0,
) -> ZoneTrade:
    equity_after = INITIAL_CASH + pnl
    return ZoneTrade(
        zone_id="official:long:a",
        detector="ict_month04",
        side="long",
        entry_time_utc=entry_time_utc,
        entry_price=145.2,
        stop_loss=145.0,
        take_profit=145.4,
        initial_risk=0.2,
        units=10_000,
        exit_time_utc=entry_time_utc + 900,
        exit_price=145.4 if pnl > 0 else 145.0,
        exit_reason="tp" if pnl > 0 else "sl",
        pnl=pnl,
        equity_before=INITIAL_CASH,
        equity_after=equity_after,
        realized_r=1.0 if pnl > 0 else -1.0,
    )


def test_deterministic_hash_is_stable_and_price_sensitive() -> None:
    original = deterministic_sha256("zones", [zone()])
    repeated = deterministic_sha256("zones", [zone()])
    changed = deterministic_sha256(
        "zones",
        [replace(zone(), entry_price=145.1)],
    )

    assert original == repeated
    assert original != changed


def test_private_frames_keep_stable_empty_and_nonempty_schemas() -> None:
    assert tuple(zones_frame([]).columns) == tuple(zones_frame([zone()]).columns)
    assert tuple(trades_frame([]).columns) == tuple(trades_frame([trade()]).columns)


def test_fixed_segments_assign_entry_by_et_date() -> None:
    first = trade(entry_time_utc=int(pd.Timestamp("2024-01-06T05:00:00Z").timestamp()))
    second = replace(
        trade(entry_time_utc=int(pd.Timestamp("2024-07-08T04:00:00Z").timestamp())),
        zone_id="official:long:b",
        equity_before=first.equity_after,
        equity_after=first.equity_after + 2_000.0,
    )

    segments = fixed_segment_summary([first, second])

    assert segments["trade_count"].tolist() == [1, 1, 0, 0, 0]
    assert segments["pnl_jpy"].tolist() == [2_000.0, 2_000.0, 0.0, 0.0, 0.0]


def test_summary_uses_sample_gate_before_profit_criterion() -> None:
    one_trade = trade()
    backtest = ZoneBacktestResult(
        trades=[one_trade],
        final_equity=one_trade.equity_after,
        open_position=None,
        pending_zone=None,
        counters={"trade_closed": 1},
    )
    segments = fixed_segment_summary(backtest.trades)

    summary = summarize_detector("ict_month04", [zone()], backtest, segments)

    assert summary.return_pct > 0
    assert summary.criterion == "insufficient_sample"
    assert summary.sample_sufficient is False


def test_summary_reports_open_position_without_fabricating_a_closed_trade() -> None:
    open_position = OpenPosition(
        zone_id="official:long:a",
        detector="ict_month04",
        side="long",
        entry_time_utc=1_704_520_800,
        entry_price=145.2,
        stop_loss=145.0,
        take_profit=145.4,
        initial_risk=0.2,
        units=10_000,
        equity_before=INITIAL_CASH,
    )
    backtest = ZoneBacktestResult(
        trades=[],
        final_equity=INITIAL_CASH,
        open_position=open_position,
        pending_zone=None,
        counters={"open_at_end": 1},
    )

    summary = summarize_detector(
        "ict_month04",
        [zone()],
        backtest,
        fixed_segment_summary([]),
    )

    assert summary.trade_count == 0
    assert summary.open_position_count == 1
    assert summary.win_rate_pct is None
    assert summary.average_realized_r is None
    assert summary.criterion == "insufficient_sample"


def test_assert_reproducible_rejects_a_result_hash_change() -> None:
    class Audit:
        extraction_sha256 = "input"

    class Result:
        input_audit = Audit()
        result_sha256 = "same"

    first = Result()
    second = Result()
    assert_reproducible(first, second)  # type: ignore[arg-type]
    second.result_sha256 = "different"

    with pytest.raises(FrozenInputError, match="different zones"):
        assert_reproducible(first, second)  # type: ignore[arg-type]
