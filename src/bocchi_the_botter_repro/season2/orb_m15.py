"""Season 2 chapter 3: USDJPY M15 ORB retranslation.

The article run is deliberately tied to the frozen Season 2 chapter 2 M5
projection.  The upstream SQLite database is opened read-only, projected over
one fixed half-open UTC interval, checked by extraction hash, and aggregated to
complete M15 candles before the strategy is evaluated.

Bid OHLC is the sole signal series.  Bid/ask OHLC is then used directionally
for fills: long entries use ask and long exits use bid; short entries use bid
and short exits use ask.  No fixed spread is added on top of those quotes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..common.backtest.strategies.sizing import compute_units
from .minute_data import (
    SOURCE,
    M5Audit,
    aggregate_m5_to_m15,
    audit_m5_frame,
    load_m5_candles,
    parse_m5_boundary,
)


INPUT_START_UTC = "2024-01-01T22:00:00Z"
INPUT_END_EXCLUSIVE_UTC = "2026-07-14T10:05:00Z"
EXPECTED_INPUT_SHA256 = (
    "f6d0e1cd1bd50ec11f7f3f0bd34e31b61a39970a687f3c5ac83682ae2ea1d512"
)
EXPECTED_M5_ROWS = 188_981
EXPECTED_COMPLETE_M15 = 62_957
EXPECTED_INCOMPLETE_M15 = 61

ARTICLE_WINDOW_START = dt.date(2024, 1, 6)
ARTICLE_WINDOW_END_EXCLUSIVE = dt.date(2026, 7, 14)
WINDOW_DAYS = 920
N_SEGMENTS = 5
SEGMENT_DAYS = 184

ATR_N = 14
ATR_LO = 1.25
ATR_HI = 3.0
RR = 1.5
RANGE_HOUR_ET = 9
RANGE_MINUTE_ET = 30
CUTOFF_HOUR_ET = 12
CLOSE_HOUR_ET = 16
RISK_PCT = 0.01
MARGIN = 0.04
INITIAL_CASH = 1_000_000.0

REFERENCE_SCHEMA_VERSION = 1
NEW_YORK = ZoneInfo("America/New_York")


class OrbM15Error(RuntimeError):
    """Base error for the frozen S2-3 reproduction."""


class FrozenInputError(OrbM15Error):
    """Raised when the fixed S2-2 projection no longer matches."""


class M15ValidationError(ValueError):
    """Raised when an M15 input violates the strategy contract."""


class ReferenceVerificationError(OrbM15Error):
    """Raised when independently recomputed results disagree."""


@dataclass(frozen=True, slots=True)
class Trade:
    """One private, price-bearing simulated trade."""

    session_date_et: dt.date
    side: str
    signal_time_utc: int
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
    entry_quote_width: float


TRADE_COLUMNS = tuple(field.name for field in fields(Trade))


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    """Article metrics calculated from closed trades only."""

    trade_count: int
    final_equity: float
    return_pct: float
    win_rate_pct: float
    max_drawdown_pct: float
    profit_factor: float | None
    average_realized_r: float
    positive_segments: int
    long_count: int
    short_count: int
    criterion_passed: bool


@dataclass(slots=True)
class BacktestResult:
    """All private and row-free outputs from one frozen run."""

    input_audit: M5Audit
    complete_m15_count: int
    incomplete_m15_count: int
    trades: list[Trade]
    summary: BacktestSummary
    segments: pd.DataFrame
    exit_reasons: dict[str, int]
    terminal_outcomes: dict[str, int]
    session_quality: dict[str, int]
    atr_filter: dict[str, float | int | None]
    entry_quote_width: dict[str, float | int | None]
    m15_spread_open: dict[str, float | int | None]
    m15_spread_close: dict[str, float | int | None]


def input_bounds() -> tuple[int, int]:
    """Return the frozen S2-2 UTC projection boundaries."""

    return (
        parse_m5_boundary(INPUT_START_UTC),
        parse_m5_boundary(INPUT_END_EXCLUSIVE_UTC),
    )


def fixed_segment_edges(
    window_start: dt.date = ARTICLE_WINDOW_START,
) -> list[dt.date]:
    """Return the six boundaries of five consecutive 184-day segments."""

    if WINDOW_DAYS != N_SEGMENTS * SEGMENT_DAYS:
        raise RuntimeError("window and fixed segment constants disagree")
    return [
        window_start + dt.timedelta(days=SEGMENT_DAYS * index)
        for index in range(N_SEGMENTS + 1)
    ]


def expected_session_timestamps(session_date: dt.date) -> tuple[int, ...]:
    """Return 09:30 through 16:00 ET M15 starts, both ends included."""

    opened = dt.datetime.combine(
        session_date,
        dt.time(RANGE_HOUR_ET, RANGE_MINUTE_ET),
        tzinfo=NEW_YORK,
    )
    return tuple(
        int((opened + dt.timedelta(minutes=15 * index)).timestamp())
        for index in range(27)
    )


def _required_m15_columns() -> set[str]:
    return {
        "source",
        "instrument",
        "granularity",
        "price",
        "ts_utc",
        "complete",
        "component_count",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    }


def validate_m15_frame(frame: pd.DataFrame) -> None:
    """Validate the complete, bid/ask M15 input used by the strategy."""

    if frame.empty:
        raise M15ValidationError("M15 input is empty")
    missing = _required_m15_columns().difference(frame.columns)
    if missing:
        raise M15ValidationError(f"M15 input is missing columns: {sorted(missing)}")
    required = sorted(_required_m15_columns())
    if frame[required].isna().any(axis=None):
        raise M15ValidationError("M15 input contains NULL contract values")
    if not frame["ts_utc"].is_monotonic_increasing:
        raise M15ValidationError("M15 input is not sorted by ts_utc")
    if frame["ts_utc"].duplicated().any():
        raise M15ValidationError("M15 input contains duplicate timestamps")
    if (frame["ts_utc"].astype("int64") % 900 != 0).any():
        raise M15ValidationError("M15 input contains off-boundary timestamps")
    if not (frame["source"] == SOURCE).all():
        raise M15ValidationError("M15 input has an unexpected source")
    if not (frame["instrument"] == "USD_JPY").all():
        raise M15ValidationError("M15 input has an unexpected instrument")
    if not (frame["granularity"] == "M15").all():
        raise M15ValidationError("M15 input has an unexpected granularity")
    if not (frame["price"] == "BA").all():
        raise M15ValidationError("M15 input has an unexpected price component")
    if not (frame["complete"] == 1).all() or not (
        frame["component_count"] == 3
    ).all():
        raise M15ValidationError("M15 input contains an incomplete bucket")

    price_columns = [
        f"{side}_{field}"
        for side in ("bid", "ask")
        for field in ("open", "high", "low", "close")
    ]
    values = frame[price_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise M15ValidationError("M15 input contains non-finite prices")
    for side in ("bid", "ask"):
        opened = frame[f"{side}_open"]
        high = frame[f"{side}_high"]
        low = frame[f"{side}_low"]
        closed = frame[f"{side}_close"]
        if (~((low <= opened) & (opened <= high) & (low <= closed) & (closed <= high))).any():
            raise M15ValidationError(f"M15 input contains invalid {side} OHLC")
    if (frame["ask_open"] < frame["bid_open"]).any() or (
        frame["ask_close"] < frame["bid_close"]
    ).any():
        raise M15ValidationError("M15 input contains negative open/close quote width")


def add_bid_atr(frame: pd.DataFrame) -> pd.DataFrame:
    """Add bid True Range and its current-bar-inclusive 14-row SMA."""

    validate_m15_frame(frame)
    result = frame.copy()
    if "ts_utc_dt" not in result.columns:
        result["ts_utc_dt"] = pd.to_datetime(result["ts_utc"], unit="s", utc=True)
    result["ts_et"] = result["ts_utc_dt"].dt.tz_convert("America/New_York")
    result["session_date_et"] = result["ts_et"].dt.date
    result["spread_open"] = result["ask_open"] - result["bid_open"]
    result["spread_close"] = result["ask_close"] - result["bid_close"]

    previous_close = result["bid_close"].shift(1)
    candidates = pd.concat(
        [
            result["bid_high"] - result["bid_low"],
            (result["bid_high"] - previous_close).abs(),
            (result["bid_low"] - previous_close).abs(),
        ],
        axis=1,
    )
    result["bid_tr"] = candidates.max(axis=1, skipna=True)
    result["bid_atr"] = result["bid_tr"].rolling(
        ATR_N, min_periods=ATR_N
    ).mean()
    return result


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "maximum": None,
        }
    return {
        "count": len(series),
        "minimum": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p75": float(series.quantile(0.75)),
        "maximum": float(series.max()),
    }


def _exit_trade(
    day: pd.DataFrame,
    *,
    entry_position: int,
    side: str,
    stop_loss: float,
    take_profit: float,
) -> tuple[int, float, str]:
    """Apply the fixed quote-side exit ordering from entry through 16:00 ET."""

    for position in range(entry_position, len(day)):
        bar = day.iloc[position]
        timestamp = int(bar["ts_utc"])
        local = pd.Timestamp(bar["ts_et"])
        exit_open = float(bar["bid_open"] if side == "long" else bar["ask_open"])
        if local.hour == CLOSE_HOUR_ET and local.minute == 0:
            return timestamp, exit_open, "close_16"

        if side == "long":
            if exit_open <= stop_loss:
                return timestamp, exit_open, "gap_sl"
            if exit_open >= take_profit:
                return timestamp, take_profit, "tp"
            hit_stop = float(bar["bid_low"]) <= stop_loss
            hit_target = float(bar["bid_high"]) >= take_profit
        else:
            if exit_open >= stop_loss:
                return timestamp, exit_open, "gap_sl"
            if exit_open <= take_profit:
                return timestamp, take_profit, "tp"
            hit_stop = float(bar["ask_high"]) >= stop_loss
            hit_target = float(bar["ask_low"]) <= take_profit

        if hit_stop:
            return timestamp, stop_loss, "sl"
        if hit_target:
            return timestamp, take_profit, "tp"
    raise RuntimeError("valid session ended without a 16:00 exit")


def simulate_sessions(
    frame: pd.DataFrame,
    *,
    window_start: dt.date = ARTICLE_WINDOW_START,
    window_end_exclusive: dt.date = ARTICLE_WINDOW_END_EXCLUSIVE,
    initial_cash: float = INITIAL_CASH,
) -> tuple[
    list[Trade],
    dict[str, int],
    dict[str, int],
    list[float],
]:
    """Evaluate each ET date exactly once and return trades plus audit counters."""

    required = {
        "ts_utc",
        "ts_et",
        "session_date_et",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "bid_atr",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise M15ValidationError(
            f"simulation frame is missing columns: {sorted(missing)}"
        )
    if window_start >= window_end_exclusive:
        raise ValueError("window_start must precede window_end_exclusive")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    indexed = frame.set_index("ts_utc", drop=False)
    if indexed.index.duplicated().any():
        raise M15ValidationError("simulation frame contains duplicate timestamps")
    available = set(int(value) for value in indexed.index)
    outcomes: Counter[str] = Counter()
    quality: Counter[str] = Counter()
    trades: list[Trade] = []
    atr_ratios: list[float] = []
    equity = float(initial_cash)

    cursor = window_start
    while cursor < window_end_exclusive:
        quality["candidate_sessions"] += 1
        expected = expected_session_timestamps(cursor)
        present_count = sum(timestamp in available for timestamp in expected)
        if present_count != len(expected):
            quality["invalid_sessions"] += 1
            quality["missing_required_bars"] += len(expected) - present_count
            quality[
                "zero_bar_sessions" if present_count == 0 else "partial_sessions"
            ] += 1
            outcomes["invalid_session"] += 1
            cursor += dt.timedelta(days=1)
            continue

        quality["valid_sessions"] += 1
        day = indexed.loc[list(expected)].copy().reset_index(drop=True)
        range_bar = day.iloc[0]
        atr = float(range_bar["bid_atr"])
        if not np.isfinite(atr) or atr <= 0:
            outcomes["atr_unavailable"] += 1
            cursor += dt.timedelta(days=1)
            continue

        range_high = float(range_bar["bid_high"])
        range_low = float(range_bar["bid_low"])
        width = range_high - range_low
        ratio = width / atr
        atr_ratios.append(ratio)
        if ratio < ATR_LO:
            outcomes["atr_below"] += 1
            cursor += dt.timedelta(days=1)
            continue
        if ratio > ATR_HI:
            outcomes["atr_above"] += 1
            cursor += dt.timedelta(days=1)
            continue
        outcomes["atr_passed"] += 1

        signal_position: int | None = None
        side: str | None = None
        for position in range(1, len(day) - 1):
            close = float(day.iloc[position]["bid_close"])
            if close > range_high:
                signal_position, side = position, "long"
                break
            if close < range_low:
                signal_position, side = position, "short"
                break
        if signal_position is None or side is None:
            outcomes["no_breakout"] += 1
            cursor += dt.timedelta(days=1)
            continue

        entry_position = signal_position + 1
        entry_bar = day.iloc[entry_position]
        entry_local = pd.Timestamp(entry_bar["ts_et"])
        if entry_local.hour >= CUTOFF_HOUR_ET:
            outcomes["cutoff"] += 1
            cursor += dt.timedelta(days=1)
            continue

        if side == "long":
            entry_price = float(entry_bar["ask_open"])
            stop_loss = range_low
            initial_risk = entry_price - stop_loss
            take_profit = entry_price + RR * initial_risk
        else:
            entry_price = float(entry_bar["bid_open"])
            stop_loss = float(range_bar["ask_high"])
            initial_risk = stop_loss - entry_price
            take_profit = entry_price - RR * initial_risk
        if not np.isfinite(initial_risk) or initial_risk <= 0:
            outcomes["invalid_risk"] += 1
            cursor += dt.timedelta(days=1)
            continue

        units = compute_units(
            equity=equity,
            atr=initial_risk,
            price=entry_price,
            risk_pct=RISK_PCT,
            sl_atr_mult=1.0,
            margin=MARGIN,
            spread=0.0,
        )
        if units <= 0:
            outcomes["sizing_rejected"] += 1
            cursor += dt.timedelta(days=1)
            continue

        exit_time, exit_price, exit_reason = _exit_trade(
            day,
            entry_position=entry_position,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        per_unit = (
            exit_price - entry_price
            if side == "long"
            else entry_price - exit_price
        )
        pnl = units * per_unit
        equity_before = equity
        equity += pnl
        trades.append(
            Trade(
                session_date_et=cursor,
                side=side,
                signal_time_utc=int(day.iloc[signal_position]["ts_utc"]),
                entry_time_utc=int(entry_bar["ts_utc"]),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                initial_risk=initial_risk,
                units=units,
                exit_time_utc=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                equity_before=equity_before,
                equity_after=equity,
                realized_r=per_unit / initial_risk,
                entry_quote_width=float(entry_bar["ask_open"])
                - float(entry_bar["bid_open"]),
            )
        )
        outcomes["traded"] += 1
        cursor += dt.timedelta(days=1)

    if sum(
        outcomes[key]
        for key in (
            "invalid_session",
            "atr_unavailable",
            "atr_below",
            "atr_above",
            "no_breakout",
            "cutoff",
            "invalid_risk",
            "sizing_rejected",
            "traded",
        )
    ) != quality["candidate_sessions"]:
        raise RuntimeError("terminal session outcomes do not cover the article window")
    return trades, dict(sorted(outcomes.items())), dict(sorted(quality.items())), atr_ratios


def trades_frame(trades: Iterable[Trade]) -> pd.DataFrame:
    """Convert private trades to their stable price-bearing schema."""

    return pd.DataFrame(
        [asdict(trade) for trade in trades],
        columns=TRADE_COLUMNS,
    )


def fixed_segment_summary(
    trades: pd.DataFrame,
    *,
    window_start: dt.date = ARTICLE_WINDOW_START,
) -> pd.DataFrame:
    """Assign each trade once and report row-free aggregate segment results."""

    missing = {"session_date_et", "pnl"}.difference(trades.columns)
    if missing:
        raise ValueError(f"trades are missing segment fields: {sorted(missing)}")
    edges = fixed_segment_edges(window_start)
    dates = pd.to_datetime(trades["session_date_et"], errors="raise")
    if not trades.empty:
        outside = (dates.dt.date < edges[0]) | (dates.dt.date >= edges[-1])
        if outside.any():
            raise ValueError("one or more trades fall outside the fixed window")

    rows: list[dict[str, Any]] = []
    segment_start_equity = INITIAL_CASH
    for index in range(N_SEGMENTS):
        lower, upper = edges[index], edges[index + 1]
        mask = (dates.dt.date >= lower) & (dates.dt.date < upper)
        pnl = float(trades.loc[mask, "pnl"].sum())
        rows.append(
            {
                "segment": index + 1,
                "start": lower.isoformat(),
                "end_exclusive": upper.isoformat(),
                "trade_count": int(mask.sum()),
                "pnl_jpy": pnl,
                "return_pct": pnl / segment_start_equity * 100.0,
            }
        )
        segment_start_equity += pnl
    result = pd.DataFrame(rows)
    if int(result["trade_count"].sum()) != len(trades):
        raise RuntimeError("fixed segment assignment contains a gap or overlap")
    return result


def summarize_trades(
    trades: pd.DataFrame,
    segments: pd.DataFrame,
) -> BacktestSummary:
    """Calculate the frozen article metrics from private trades."""

    if trades.empty:
        raise ValueError("the main strategy produced no trades")
    required = {
        "side",
        "pnl",
        "equity_after",
        "realized_r",
        "exit_reason",
    }
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"trades are missing summary fields: {sorted(missing)}")
    equity = pd.concat(
        [pd.Series([INITIAL_CASH], dtype=float), trades["equity_after"]],
        ignore_index=True,
    )
    drawdown = (equity - equity.cummax()) / equity.cummax()
    gains = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    losses = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    profit_factor = None if losses == 0 else gains / losses
    final_equity = float(trades["equity_after"].iloc[-1])
    return_pct = (final_equity / INITIAL_CASH - 1.0) * 100.0
    positive_segments = int((segments["pnl_jpy"] > 0).sum())
    return BacktestSummary(
        trade_count=len(trades),
        final_equity=final_equity,
        return_pct=return_pct,
        win_rate_pct=float((trades["pnl"] > 0).mean() * 100.0),
        max_drawdown_pct=float(drawdown.min() * 100.0),
        profit_factor=profit_factor,
        average_realized_r=float(trades["realized_r"].mean()),
        positive_segments=positive_segments,
        long_count=int((trades["side"] == "long").sum()),
        short_count=int((trades["side"] == "short").sum()),
        criterion_passed=bool(return_pct > 0 and positive_segments >= 3),
    )


def validate_frozen_input(
    audit: M5Audit,
    *,
    complete_m15_count: int,
    incomplete_m15_count: int,
) -> None:
    """Stop unless the fixed S2-2 projection and M15 derivation still match."""

    mismatches: list[str] = []
    if audit.extraction_sha256 != EXPECTED_INPUT_SHA256:
        mismatches.append(
            f"extraction_sha256={audit.extraction_sha256}"
        )
    if audit.row_count != EXPECTED_M5_ROWS:
        mismatches.append(f"m5_rows={audit.row_count}")
    if complete_m15_count != EXPECTED_COMPLETE_M15:
        mismatches.append(f"complete_m15={complete_m15_count}")
    if incomplete_m15_count != EXPECTED_INCOMPLETE_M15:
        mismatches.append(f"incomplete_m15={incomplete_m15_count}")
    if any(
        value != 0
        for value in (
            audit.duplicate_count,
            audit.off_boundary_count,
            audit.null_required_count,
            audit.invalid_volume_count,
            audit.invalid_ohlc_count,
            audit.negative_spread_count,
        )
    ) or not audit.sorted_ascending:
        mismatches.append("M5 quality counters violate the frozen contract")
    if mismatches:
        raise FrozenInputError("frozen S2-2 input mismatch: " + "; ".join(mismatches))


def run_backtest_from_m15(
    frame: pd.DataFrame,
    *,
    input_audit: M5Audit,
    incomplete_m15_count: int,
) -> BacktestResult:
    """Run S2-3 from already aggregated M15 data."""

    validate_frozen_input(
        input_audit,
        complete_m15_count=len(frame),
        incomplete_m15_count=incomplete_m15_count,
    )
    prepared = add_bid_atr(frame)
    trades, outcomes, quality, ratios = simulate_sessions(prepared)
    private = trades_frame(trades)
    segments = fixed_segment_summary(private)
    summary = summarize_trades(private, segments)
    exits = {
        str(key): int(value)
        for key, value in private["exit_reason"].value_counts().sort_index().items()
    }
    atr_distribution = _distribution(ratios)
    atr_filter: dict[str, float | int | None] = {
        **atr_distribution,
        "below_lower": int(outcomes.get("atr_below", 0)),
        "passed": int(outcomes.get("atr_passed", 0)),
        "above_upper": int(outcomes.get("atr_above", 0)),
        "lower_inclusive": ATR_LO,
        "upper_inclusive": ATR_HI,
    }
    return BacktestResult(
        input_audit=input_audit,
        complete_m15_count=len(frame),
        incomplete_m15_count=incomplete_m15_count,
        trades=trades,
        summary=summary,
        segments=segments,
        exit_reasons=exits,
        terminal_outcomes=outcomes,
        session_quality=quality,
        atr_filter=atr_filter,
        entry_quote_width=_distribution(private["entry_quote_width"].tolist()),
        m15_spread_open=_distribution(prepared["spread_open"].tolist()),
        m15_spread_close=_distribution(prepared["spread_close"].tolist()),
    )


def run_backtest_from_db(db_path: Path) -> BacktestResult:
    """Read the upstream SQLite safely and execute the frozen main variant."""

    resolved = db_path.expanduser().resolve()
    before = resolved.stat()
    start, end = input_bounds()
    m5 = load_m5_candles(
        resolved,
        source=SOURCE,
        instrument="USD_JPY",
        start_inclusive=start,
        end_exclusive=end,
    )
    audit = audit_m5_frame(
        m5,
        source=SOURCE,
        instrument="USD_JPY",
        start_inclusive=start,
        end_exclusive=end,
    )
    aggregated = aggregate_m5_to_m15(
        m5,
        start_inclusive=start,
        end_exclusive=end,
    )
    result = run_backtest_from_m15(
        aggregated.candles,
        input_audit=audit,
        incomplete_m15_count=len(aggregated.incomplete_buckets),
    )
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise FrozenInputError(
            "upstream SQLite changed during the read-only run; retry a stable snapshot"
        )
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(dict(payload)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_private_audit(result: BacktestResult, output_dir: Path) -> None:
    """Write price-bearing local evidence under a Git-ignored directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_frame(result.trades).to_csv(output_dir / "trades_private.csv", index=False)
    _write_json(
        output_dir / "run_audit.json",
        {
            "input_audit": asdict(result.input_audit),
            "complete_m15_count": result.complete_m15_count,
            "incomplete_m15_count": result.incomplete_m15_count,
            "summary": asdict(result.summary),
            "terminal_outcomes": result.terminal_outcomes,
            "session_quality": result.session_quality,
            "atr_filter": result.atr_filter,
            "entry_quote_width": result.entry_quote_width,
            "m15_spread_open": result.m15_spread_open,
            "m15_spread_close": result.m15_spread_close,
        },
    )


def write_reference_outputs(
    result: BacktestResult,
    output_dir: Path,
    *,
    code_paths: Iterable[Path] = (),
) -> None:
    """Write only row-free, price-free public reference artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "reference_summary.json"
    segments_path = output_dir / "segments.csv"
    exits_path = output_dir / "exit_reasons.csv"
    atr_path = output_dir / "atr_filter_summary.json"
    session_path = output_dir / "session_quality_summary.json"
    quote_path = output_dir / "quote_width_summary.json"

    _write_json(
        summary_path,
        {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "article_window_et": {
                "start_inclusive": ARTICLE_WINDOW_START,
                "end_exclusive": ARTICLE_WINDOW_END_EXCLUSIVE,
                "calendar_days": WINDOW_DAYS,
            },
            "input": {
                "source": SOURCE,
                "instrument": "USD_JPY",
                "granularity": "M5_to_M15",
                "price": "BA",
                "start_inclusive_utc": INPUT_START_UTC,
                "end_exclusive_utc": INPUT_END_EXCLUSIVE_UTC,
                "extraction_sha256": result.input_audit.extraction_sha256,
                "m5_rows": result.input_audit.row_count,
                "complete_m15": result.complete_m15_count,
                "incomplete_m15": result.incomplete_m15_count,
            },
            "execution": {
                "signal": "bid OHLC",
                "long": "entry ask_open; exit bid",
                "short": "entry bid_open; exit ask",
                "fixed_spread_addition": 0,
                "commission": 0,
                "ordinary_slippage": 0,
                "same_bar_priority": "SL",
                "forced_exit": "16:00 ET execution-side open",
            },
            "summary": asdict(result.summary),
            "criterion": "return_pct > 0 and positive_segments >= 3",
        },
    )
    result.segments.to_csv(segments_path, index=False)
    pd.DataFrame(
        [
            {"exit_reason": reason, "trade_count": count}
            for reason, count in sorted(result.exit_reasons.items())
        ]
    ).to_csv(exits_path, index=False)
    _write_json(atr_path, result.atr_filter)
    _write_json(
        session_path,
        {
            **result.session_quality,
            "terminal_outcomes": result.terminal_outcomes,
        },
    )
    _write_json(
        quote_path,
        {
            "entry_open_quote_width": result.entry_quote_width,
            "all_m15_open_quote_width": result.m15_spread_open,
            "all_m15_close_quote_width": result.m15_spread_close,
            "warning": (
                "Open/close quote widths are not intrabar maximum/minimum spread "
                "and do not reconstruct spread at SL/TP time."
            ),
        },
    )

    artifact_paths = [summary_path, segments_path, exits_path, atr_path, session_path, quote_path]
    _write_json(
        output_dir / "hashes.json",
        {
            "input_extraction_sha256": result.input_audit.extraction_sha256,
            "code_sha256": {
                path.name: _file_sha256(path)
                for path in sorted(code_paths, key=lambda item: item.as_posix())
            },
            "artifact_sha256": {
                path.name: _file_sha256(path) for path in artifact_paths
            },
        },
    )


def verify_private_against_reference(
    private_trades_path: Path,
    reference_dir: Path,
    *,
    absolute_tolerance: float = 1e-8,
) -> dict[str, float | int | bool | None]:
    """Independently recompute price, equity, segment, and summary invariants."""

    trades = pd.read_csv(private_trades_path)
    missing = set(TRADE_COLUMNS).difference(trades.columns)
    if missing:
        raise ReferenceVerificationError(
            f"private trade log is missing columns: {sorted(missing)}"
        )
    if trades.empty:
        raise ReferenceVerificationError("private trade log is empty")
    sides = set(trades["side"])
    if not sides.issubset({"long", "short"}):
        raise ReferenceVerificationError(f"unexpected side values: {sorted(sides)}")

    expected_per_unit = np.where(
        trades["side"] == "long",
        trades["exit_price"] - trades["entry_price"],
        trades["entry_price"] - trades["exit_price"],
    )
    expected_pnl = expected_per_unit * trades["units"]
    if not np.allclose(expected_pnl, trades["pnl"], rtol=0, atol=absolute_tolerance):
        raise ReferenceVerificationError("trade PnL does not match direction-side fills")
    expected_r = expected_per_unit / trades["initial_risk"]
    if not np.allclose(
        expected_r, trades["realized_r"], rtol=0, atol=absolute_tolerance
    ):
        raise ReferenceVerificationError("realized R does not match initial risk")

    expected_before = np.concatenate(
        ([INITIAL_CASH], trades["equity_after"].to_numpy(dtype=float)[:-1])
    )
    if not np.allclose(
        expected_before,
        trades["equity_before"],
        rtol=0,
        atol=absolute_tolerance,
    ):
        raise ReferenceVerificationError("equity chain is discontinuous")
    if not np.allclose(
        trades["equity_before"] + trades["pnl"],
        trades["equity_after"],
        rtol=0,
        atol=absolute_tolerance,
    ):
        raise ReferenceVerificationError("equity_after does not equal equity_before + PnL")

    summary_payload = json.loads(
        (reference_dir / "reference_summary.json").read_text(encoding="utf-8")
    )["summary"]
    public_segments = pd.read_csv(reference_dir / "segments.csv")
    dates = pd.to_datetime(trades["session_date_et"], errors="raise").dt.date
    recomputed_segment_pnl: list[float] = []
    recomputed_segment_count: list[int] = []
    for row in public_segments.itertuples(index=False):
        lower = dt.date.fromisoformat(str(row.start))
        upper = dt.date.fromisoformat(str(row.end_exclusive))
        mask = (dates >= lower) & (dates < upper)
        recomputed_segment_count.append(int(mask.sum()))
        recomputed_segment_pnl.append(float(trades.loc[mask, "pnl"].sum()))
    if recomputed_segment_count != public_segments["trade_count"].tolist():
        raise ReferenceVerificationError("public segment trade counts disagree")
    if not np.allclose(
        recomputed_segment_pnl,
        public_segments["pnl_jpy"],
        rtol=0,
        atol=absolute_tolerance,
    ):
        raise ReferenceVerificationError("public segment PnL disagrees")

    equity = np.concatenate(([INITIAL_CASH], trades["equity_after"].to_numpy(float)))
    peaks = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peaks) / peaks) * 100.0)
    final_equity = float(trades["equity_after"].iloc[-1])
    return_pct = (final_equity / INITIAL_CASH - 1.0) * 100.0
    gains = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    losses = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    profit_factor = None if losses == 0 else gains / losses
    positive_segments = sum(value > 0 for value in recomputed_segment_pnl)
    recomputed: dict[str, float | int | bool | None] = {
        "trade_count": len(trades),
        "final_equity": final_equity,
        "return_pct": return_pct,
        "win_rate_pct": float((trades["pnl"] > 0).mean() * 100.0),
        "max_drawdown_pct": max_dd,
        "profit_factor": profit_factor,
        "average_realized_r": float(trades["realized_r"].mean()),
        "positive_segments": positive_segments,
        "long_count": int((trades["side"] == "long").sum()),
        "short_count": int((trades["side"] == "short").sum()),
        "criterion_passed": bool(return_pct > 0 and positive_segments >= 3),
    }
    for key, actual in recomputed.items():
        expected = summary_payload[key]
        if actual is None or expected is None:
            if actual is not expected:
                raise ReferenceVerificationError(f"summary field {key} disagrees")
        elif isinstance(actual, bool) or isinstance(actual, int):
            if actual != expected:
                raise ReferenceVerificationError(f"summary field {key} disagrees")
        elif not np.isclose(actual, expected, rtol=0, atol=absolute_tolerance):
            raise ReferenceVerificationError(f"summary field {key} disagrees")
    return recomputed


def verify_row_free_reference(reference_dir: Path) -> dict[str, Any]:
    """Verify public aggregate artifacts without the private OANDA trade log."""

    required = {
        "reference_summary.json",
        "segments.csv",
        "exit_reasons.csv",
        "atr_filter_summary.json",
        "session_quality_summary.json",
        "quote_width_summary.json",
        "hashes.json",
    }
    missing = [name for name in sorted(required) if not (reference_dir / name).is_file()]
    if missing:
        raise ReferenceVerificationError(
            f"row-free reference files are missing: {missing}"
        )

    hashes = json.loads((reference_dir / "hashes.json").read_text(encoding="utf-8"))
    for name, expected in hashes["artifact_sha256"].items():
        actual = _file_sha256(reference_dir / name)
        if actual != expected:
            raise ReferenceVerificationError(f"artifact hash mismatch: {name}")
    for name, expected in hashes.get("figure_sha256", {}).items():
        actual = _file_sha256(reference_dir / "figures" / name)
        if actual != expected:
            raise ReferenceVerificationError(f"figure hash mismatch: {name}")

    payload = json.loads(
        (reference_dir / "reference_summary.json").read_text(encoding="utf-8")
    )
    summary = payload["summary"]
    segments = pd.read_csv(reference_dir / "segments.csv")
    exits = pd.read_csv(reference_dir / "exit_reasons.csv")
    sessions = json.loads(
        (reference_dir / "session_quality_summary.json").read_text(encoding="utf-8")
    )
    atr = json.loads(
        (reference_dir / "atr_filter_summary.json").read_text(encoding="utf-8")
    )

    if len(segments) != N_SEGMENTS:
        raise ReferenceVerificationError("row-free reference does not contain five segments")
    expected_edges = fixed_segment_edges()
    actual_starts = [dt.date.fromisoformat(str(value)) for value in segments["start"]]
    actual_ends = [
        dt.date.fromisoformat(str(value)) for value in segments["end_exclusive"]
    ]
    if actual_starts != expected_edges[:-1] or actual_ends != expected_edges[1:]:
        raise ReferenceVerificationError("row-free segment boundaries disagree")
    if int(segments["trade_count"].sum()) != int(summary["trade_count"]):
        raise ReferenceVerificationError("segment counts do not sum to trade_count")
    if int((segments["pnl_jpy"] > 0).sum()) != int(summary["positive_segments"]):
        raise ReferenceVerificationError("positive segment count disagrees")
    if int(exits["trade_count"].sum()) != int(summary["trade_count"]):
        raise ReferenceVerificationError("exit reason counts do not sum to trade_count")
    if int(summary["long_count"]) + int(summary["short_count"]) != int(
        summary["trade_count"]
    ):
        raise ReferenceVerificationError("direction counts do not sum to trade_count")

    candidate = int(sessions["candidate_sessions"])
    valid = int(sessions["valid_sessions"])
    invalid = int(sessions["invalid_sessions"])
    if candidate != WINDOW_DAYS or valid + invalid != candidate:
        raise ReferenceVerificationError("session quality counts do not cover the window")
    outcomes = sessions["terminal_outcomes"]
    terminal_keys = (
        "invalid_session",
        "atr_unavailable",
        "atr_below",
        "atr_above",
        "no_breakout",
        "cutoff",
        "invalid_risk",
        "sizing_rejected",
        "traded",
    )
    if sum(int(outcomes.get(key, 0)) for key in terminal_keys) != candidate:
        raise ReferenceVerificationError("terminal outcomes do not cover the window")
    if int(outcomes.get("traded", 0)) != int(summary["trade_count"]):
        raise ReferenceVerificationError("traded outcome disagrees with trade_count")
    if int(atr["below_lower"]) + int(atr["passed"]) + int(atr["above_upper"]) != int(
        atr["count"]
    ):
        raise ReferenceVerificationError("ATR categories do not cover evaluated sessions")

    forbidden = (
        "entry_price",
        "exit_price",
        "entry_time_utc",
        "exit_time_utc",
        "signal_time_utc",
    )
    for path in reference_dir.iterdir():
        if path.suffix not in {".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in forbidden):
            raise ReferenceVerificationError(
                f"private row field leaked into public artifact: {path.name}"
            )
    return payload
