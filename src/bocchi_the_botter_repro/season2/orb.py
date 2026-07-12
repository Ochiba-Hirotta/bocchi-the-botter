"""Season 2 chapter 1: USDJPY 1-hour opening-range breakout.

The article uses a fixed 720-calendar-day window, represented as the half-open
interval ``[2024-07-21, 2026-07-11)`` in New York dates.  Live execution fetches
``JPY=X`` hourly bars from Yahoo Finance through :mod:`yfinance`; reference-mode
execution verifies the frozen, derived CSV files shipped with this repository.

Raw Yahoo Finance OHLC data is intentionally not written by this module.  Yahoo
may revise historical bars or limit intraday history, so a later live run can
differ from the article snapshot or may no longer cover the fixed window.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf

from ..common.backtest.strategies.sizing import compute_units


# Article window and translated ORB rules.
ARTICLE_WINDOW_START = dt.date(2024, 7, 21)
ARTICLE_WINDOW_END_EXCLUSIVE = dt.date(2026, 7, 11)
WINDOW_DAYS = 720
N_SEGMENTS = 5
SEGMENT_DAYS = WINDOW_DAYS // N_SEGMENTS

RANGE_HOUR_ET = 9
ATR_N = 14
ATR_LO = 1.25
ATR_HI = 3.0
RR = 1.5
CUTOFF_HOUR_ET = 12
CLOSE_HOUR_ET = 16
RISK_PCT = 0.01
MARGIN = 0.04
INITIAL_CASH = 1_000_000.0
BREAKEVEN_WR = 1.0 / (1.0 + RR) * 100.0

# OANDA Securities observation retrieved on 2026-04-14.  The published
# bid-ask width was 0.3 sen; the simulation applies the one-way relative value
# to entry and exit separately.
USDJPY_SPREAD = 1.0e-5
SPREAD_SOURCE_NOTE = (
    "OANDA Securities observation retrieved 2026-04-14 "
    "(measurement period 2026-04-06 through 2026-04-10, Tokyo server, "
    "discretionary MT5 plan, lower bound 0.3 sen at the 96.56% distribution "
    "band). Full bid-ask width converted to one-way 0.0015 JPY / 150 "
    "approximately equals 1.0e-5."
)

MAIN_TRADES_FILENAME = "trades_S2-1_ORB_USDJPY_main_net.csv"
REF_TRADES_FILENAME = "trades_S2-1_ORB_USDJPY_ref_net.csv"
SEGMENTS_FILENAME = "segments_S2-1_ORB_USDJPY_main_net.csv"
REFERENCE_SUMMARY_FILENAME = "reference_summary.csv"

EXPECTED_MAIN_TRADES = 82
EXPECTED_REF_TRADES = 169
EXPECTED_SEGMENT_COUNTS = (19, 22, 18, 13, 10)
EXPECTED_SEGMENT_PNL = (
    29_606.504621251755,
    18_419.41636022806,
    12_328.852481862063,
    -9_219.019290017692,
    -20_770.14394617842,
)
EXPECTED_MAIN_FINAL_EQUITY = 1_030_365.610227146
EXPECTED_REF_FINAL_EQUITY = 1_064_617.2358149143

_REQUIRED_OHLC = ("Open", "High", "Low", "Close")


class DataCoverageError(ValueError):
    """Raised when live intraday data no longer covers the article window."""


class ReferenceDataError(ValueError):
    """Raised when a frozen reference CSV is missing or internally inconsistent."""


@dataclass(frozen=True)
class Trade:
    """One simulated ORB trade."""

    date: dt.date
    side: str
    entry_time: pd.Timestamp
    entry_ref: float
    entry_fill: float
    sl: float
    tp: float
    risk_width: float
    units: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl: float
    equity_before: float
    equity_after: float
    r_net: float


_TRADE_COLUMNS = tuple(field.name for field in fields(Trade))


@dataclass
class Summary:
    """Metrics and fixed-segment results for one simulation variant."""

    label: str
    n: int
    win_rate: float
    return_pct: float
    max_drawdown_pct: float
    avg_r: float
    positive_segments: int
    segments: pd.DataFrame
    exit_counts: dict[str, int]
    n_long: int
    n_short: int
    final_equity: float


@dataclass(frozen=True)
class AtrFilterStats:
    """Opening-range width divided by ATR statistics in the article window."""

    evaluated_days: int
    below_lower: int
    above_upper: int
    passed: int
    minimum: float
    p25: float
    median: float
    p75: float
    maximum: float


@dataclass
class BacktestResult:
    """Live article-window simulation and its three reported variants."""

    window: pd.DataFrame
    main_net: list[Trade]
    main_gross: list[Trade]
    ref_net: list[Trade]
    main_summary: Summary
    gross_summary: Summary
    ref_summary: Summary
    atr_filter: AtrFilterStats
    missed_entries: int


@dataclass
class ReferenceVerification:
    """Validated summaries reconstructed from the frozen derived CSVs."""

    reference_dir: Path
    main_summary: Summary
    ref_summary: Summary
    segments: pd.DataFrame


DownloadFunction = Callable[..., pd.DataFrame]


def load_et_bars(downloader: DownloadFunction | None = None) -> pd.DataFrame:
    """Fetch live ``JPY=X`` 1-hour bars and add New York date/hour and ATR.

    ``period="max"`` is an observed way to obtain roughly 730 calendar days of
    1-hour data; it is not a guaranteed retention contract.  ``downloader`` is
    injectable so callers can test normalization without network access.
    """

    download = downloader if downloader is not None else yf.download
    frame = download(
        "JPY=X",
        period="max",
        interval="1h",
        progress=False,
        auto_adjust=False,
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataCoverageError("Yahoo Finance returned no JPY=X 1-hour bars")
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    missing = set(_REQUIRED_OHLC).difference(frame.columns)
    if missing:
        raise ValueError(f"Yahoo Finance data is missing OHLC columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Yahoo Finance data must use a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("Yahoo Finance intraday index must be timezone-aware")

    frame = frame.dropna().sort_index()
    et = frame.tz_convert("America/New_York")
    previous_close = et["Close"].shift(1)
    true_range = pd.concat(
        [
            et["High"] - et["Low"],
            (et["High"] - previous_close).abs(),
            (et["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    et["ATR"] = true_range.rolling(ATR_N).mean()
    et["et_date"] = et.index.date
    et["et_hour"] = et.index.hour
    return et


def window_bounds() -> tuple[dt.date, dt.date]:
    """Return the article's fixed half-open New York-date interval."""

    return ARTICLE_WINDOW_START, ARTICLE_WINDOW_END_EXCLUSIVE


def slice_window(
    et: pd.DataFrame,
    *,
    start: dt.date = ARTICLE_WINDOW_START,
    end_exclusive: dt.date = ARTICLE_WINDOW_END_EXCLUSIVE,
) -> pd.DataFrame:
    """Slice an ET-normalized frame to a half-open date interval."""

    if start >= end_exclusive:
        raise ValueError("start must be earlier than end_exclusive")
    if "et_date" not in et.columns:
        raise ValueError("ET frame is missing the et_date column")
    dates = et["et_date"]
    return et[(dates >= start) & (dates < end_exclusive)].copy()


def article_window(et: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed 720-day article window and reject incomplete coverage."""

    if ARTICLE_WINDOW_END_EXCLUSIVE - ARTICLE_WINDOW_START != dt.timedelta(
        days=WINDOW_DAYS
    ):
        raise RuntimeError("article window constants no longer span 720 days")
    if et.empty or "et_date" not in et.columns:
        raise DataCoverageError("cannot build the article window from empty data")

    available_start = min(et["et_date"])
    available_end = max(et["et_date"])
    required_last_date = ARTICLE_WINDOW_END_EXCLUSIVE - dt.timedelta(days=1)
    if available_start > ARTICLE_WINDOW_START or available_end < required_last_date:
        raise DataCoverageError(
            "live JPY=X history does not cover the article window "
            f"[{ARTICLE_WINDOW_START}, {ARTICLE_WINDOW_END_EXCLUSIVE}); "
            f"available ET dates are [{available_start}, {available_end}]"
        )

    result = slice_window(et)
    if result.empty:
        raise DataCoverageError("the fixed article window contains no bars")
    return result


def simulate(
    et_window: pd.DataFrame,
    *,
    cutoff: bool,
    spread: float,
) -> tuple[list[Trade], int]:
    """Simulate the translated ORB rule once per New York trading day.

    With ``cutoff=True``, entry must occur before 12:00 ET (the article's main
    rule).  With ``cutoff=False``, entry may occur until the 16:00 ET close and
    is reported only as a reference variant.  When SL and TP occur within the
    same 1-hour bar, SL wins as the conservative ordering.
    """

    required = {"Open", "High", "Low", "Close", "ATR", "et_date", "et_hour"}
    missing = required.difference(et_window.columns)
    if missing:
        raise ValueError(f"simulation frame is missing columns: {sorted(missing)}")
    if spread < 0:
        raise ValueError("spread must be non-negative")

    equity = INITIAL_CASH
    trades: list[Trade] = []
    missed = 0
    entry_limit_hour = CUTOFF_HOUR_ET if cutoff else CLOSE_HOUR_ET

    for date_value, day in et_window.groupby("et_date", sort=True):
        day = day.sort_index()
        range_bars = day[day["et_hour"] == RANGE_HOUR_ET]
        if range_bars.empty:
            continue
        range_bar = range_bars.iloc[0]
        atr = range_bar["ATR"]
        if pd.isna(atr):
            continue

        range_high = float(range_bar["High"])
        range_low = float(range_bar["Low"])
        width = range_high - range_low
        if not (ATR_LO * float(atr) <= width <= ATR_HI * float(atr)):
            continue

        confirmation = day[day["et_hour"] > RANGE_HOUR_ET]
        side: str | None = None
        entry_bar: pd.Series[Any] | None = None
        for index in range(len(confirmation)):
            bar = confirmation.iloc[index]
            close = float(bar["Close"])
            if close > range_high:
                side = "long"
            elif close < range_low:
                side = "short"
            else:
                continue

            if index + 1 >= len(confirmation):
                side = None
                break
            entry_bar = confirmation.iloc[index + 1]
            if int(entry_bar["et_hour"]) >= entry_limit_hour:
                side = None
            break

        if side is None or entry_bar is None:
            continue

        entry_time = pd.Timestamp(entry_bar.name)
        entry_ref = float(entry_bar["Open"])
        if side == "long":
            stop_loss = range_low
            risk_width = entry_ref - stop_loss
            take_profit = entry_ref + RR * risk_width
        else:
            stop_loss = range_high
            risk_width = stop_loss - entry_ref
            take_profit = entry_ref - RR * risk_width
        if risk_width <= 0:
            continue

        units = compute_units(
            equity=equity,
            atr=risk_width,
            price=entry_ref,
            risk_pct=RISK_PCT,
            sl_atr_mult=1.0,
            margin=MARGIN,
            spread=spread,
        )
        if units <= 0:
            missed += 1
            continue

        entry_fill = (
            entry_ref * (1 + spread)
            if side == "long"
            else entry_ref * (1 - spread)
        )

        after_entry = day[day.index >= entry_time]
        exit_price: float | None = None
        exit_reason: str | None = None
        exit_time: pd.Timestamp | None = None
        for timestamp, bar in after_entry.iterrows():
            timestamp = pd.Timestamp(timestamp)
            if int(bar["et_hour"]) >= CLOSE_HOUR_ET:
                exit_price = float(bar["Open"])
                exit_reason = "close_16"
                exit_time = timestamp
                break
            high = float(bar["High"])
            low = float(bar["Low"])
            if side == "long":
                if low <= stop_loss:
                    exit_price, exit_reason, exit_time = stop_loss, "sl", timestamp
                    break
                if high >= take_profit:
                    exit_price, exit_reason, exit_time = take_profit, "tp", timestamp
                    break
            else:
                if high >= stop_loss:
                    exit_price, exit_reason, exit_time = stop_loss, "sl", timestamp
                    break
                if low <= take_profit:
                    exit_price, exit_reason, exit_time = take_profit, "tp", timestamp
                    break

        if exit_price is None:
            last = after_entry.iloc[-1]
            exit_price = float(last["Close"])
            exit_reason = "eod"
            exit_time = pd.Timestamp(last.name)
        if exit_reason is None or exit_time is None:
            raise RuntimeError("simulation failed to assign an exit")

        exit_fill = (
            exit_price * (1 - spread)
            if side == "long"
            else exit_price * (1 + spread)
        )
        pnl = (
            units * (exit_fill - entry_fill)
            if side == "long"
            else units * (entry_fill - exit_fill)
        )
        equity_before = equity
        equity += pnl
        r_net = (
            (exit_fill - entry_fill) / risk_width
            if side == "long"
            else (entry_fill - exit_fill) / risk_width
        )
        date = (
            date_value
            if isinstance(date_value, dt.date)
            else pd.Timestamp(date_value).date()
        )
        trades.append(
            Trade(
                date=date,
                side=side,
                entry_time=entry_time,
                entry_ref=entry_ref,
                entry_fill=entry_fill,
                sl=stop_loss,
                tp=take_profit,
                risk_width=risk_width,
                units=units,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                equity_before=equity_before,
                equity_after=equity,
                r_net=r_net,
            )
        )

    return trades, missed


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    """Convert trades to the stable public CSV schema."""

    return pd.DataFrame(
        ([trade.__dict__ for trade in trades]),
        columns=_TRADE_COLUMNS,
    )


def fixed_segment_edges(
    window_start: dt.date | str | pd.Timestamp = ARTICLE_WINDOW_START,
) -> list[pd.Timestamp]:
    """Return six boundaries for five consecutive 144-calendar-day segments."""

    if WINDOW_DAYS % N_SEGMENTS != 0:
        raise RuntimeError("WINDOW_DAYS must be divisible by N_SEGMENTS")
    start = pd.Timestamp(window_start)
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    start = start.normalize()
    return [
        start + pd.Timedelta(days=SEGMENT_DAYS * index)
        for index in range(N_SEGMENTS + 1)
    ]


def fixed_segment_summary(
    trades: pd.DataFrame,
    *,
    window_start: dt.date | str | pd.Timestamp = ARTICLE_WINDOW_START,
) -> pd.DataFrame:
    """Assign every trade to exactly one fixed half-open 144-day segment."""

    missing = {"date", "pnl"}.difference(trades.columns)
    if missing:
        raise ValueError(f"trades are missing segment columns: {sorted(missing)}")
    dates = pd.to_datetime(trades["date"], errors="raise")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    edges = fixed_segment_edges(window_start)
    outside = (dates < edges[0]) | (dates >= edges[-1])
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} trade(s) fall outside "
            f"[{edges[0].date()}, {edges[-1].date()})"
        )

    rows: list[dict[str, Any]] = []
    for index in range(N_SEGMENTS):
        lower, upper = edges[index], edges[index + 1]
        mask = (dates >= lower) & (dates < upper)
        rows.append(
            {
                "segment": index + 1,
                "start": lower,
                "end_exclusive": upper,
                "trade_count": int(mask.sum()),
                "pnl": float(trades.loc[mask, "pnl"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    if int(result["trade_count"].sum()) != len(trades):
        raise RuntimeError("fixed-segment assignment contains a gap or overlap")
    return result


def summarize_frame(frame: pd.DataFrame, label: str) -> Summary:
    """Calculate article metrics from a trade DataFrame."""

    required = {"date", "side", "pnl", "equity_after", "r_net", "exit_reason"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"trades are missing summary columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"cannot summarize an empty trade frame: {label}")

    n = len(frame)
    final_equity = float(frame["equity_after"].iloc[-1])
    equity = pd.concat(
        [pd.Series([INITIAL_CASH]), frame["equity_after"]], ignore_index=True
    )
    drawdown = (equity - equity.cummax()) / equity.cummax()
    segments = fixed_segment_summary(frame)
    return Summary(
        label=label,
        n=n,
        win_rate=float((frame["pnl"] > 0).mean() * 100),
        return_pct=(final_equity / INITIAL_CASH - 1) * 100,
        max_drawdown_pct=float(drawdown.min() * 100),
        avg_r=float(frame["r_net"].mean()),
        positive_segments=int((segments["pnl"] > 0).sum()),
        segments=segments,
        exit_counts={
            str(key): int(value)
            for key, value in frame["exit_reason"].value_counts().items()
        },
        n_long=int((frame["side"] == "long").sum()),
        n_short=int((frame["side"] == "short").sum()),
        final_equity=final_equity,
    )


def summarize(trades: list[Trade], label: str) -> Summary:
    """Calculate article metrics from simulated trades."""

    return summarize_frame(trades_frame(trades), label)


def atr_filter_stats(et_window: pd.DataFrame) -> AtrFilterStats:
    """Calculate the article's range-width/ATR distribution statistics."""

    ratios: list[float] = []
    for _, day in et_window.groupby("et_date", sort=True):
        range_bars = day[day["et_hour"] == RANGE_HOUR_ET]
        if range_bars.empty:
            continue
        range_bar = range_bars.iloc[0]
        atr = range_bar["ATR"]
        if pd.isna(atr):
            continue
        width = float(range_bar["High"]) - float(range_bar["Low"])
        ratios.append(width / float(atr))
    series = pd.Series(ratios, dtype=float)
    if series.empty:
        raise ValueError("no ATR-evaluable opening-range days were found")
    passed = (series >= ATR_LO) & (series <= ATR_HI)
    return AtrFilterStats(
        evaluated_days=len(series),
        below_lower=int((series < ATR_LO).sum()),
        above_upper=int((series > ATR_HI).sum()),
        passed=int(passed.sum()),
        minimum=float(series.min()),
        p25=float(series.quantile(0.25)),
        median=float(series.median()),
        p75=float(series.quantile(0.75)),
        maximum=float(series.max()),
    )


def run_backtest(et: pd.DataFrame | None = None) -> BacktestResult:
    """Run all article variants over the fixed window without writing files."""

    full = load_et_bars() if et is None else et
    window = article_window(full)
    main_net, missed = simulate(window, cutoff=True, spread=USDJPY_SPREAD)
    main_gross, _ = simulate(window, cutoff=True, spread=0.0)
    ref_net, _ = simulate(window, cutoff=False, spread=USDJPY_SPREAD)
    return BacktestResult(
        window=window,
        main_net=main_net,
        main_gross=main_gross,
        ref_net=ref_net,
        main_summary=summarize(main_net, "main net"),
        gross_summary=summarize(main_gross, "main gross"),
        ref_summary=summarize(ref_net, "reference net"),
        atr_filter=atr_filter_stats(window),
        missed_entries=missed,
    )


def write_trades_csv(trades: list[Trade], path: Path) -> None:
    """Write one simulation variant using the stable trade schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    trades_frame(trades).to_csv(path, index=False)


def write_segments_csv(segments: pd.DataFrame, path: Path) -> None:
    """Write segment boundaries as ISO dates and preserve the half-open end."""

    output = segments.copy()
    output["start"] = pd.to_datetime(output["start"]).dt.strftime("%Y-%m-%d")
    output["end_exclusive"] = pd.to_datetime(output["end_exclusive"]).dt.strftime(
        "%Y-%m-%d"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def write_result_csvs(result: BacktestResult, output_dir: Path | str) -> None:
    """Write the three public live-run CSV outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_trades_csv(result.main_net, output / MAIN_TRADES_FILENAME)
    write_trades_csv(result.ref_net, output / REF_TRADES_FILENAME)
    write_segments_csv(result.main_summary.segments, output / SEGMENTS_FILENAME)


def _validate_trade_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    expected_count: int,
) -> None:
    required = set(_TRADE_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ReferenceDataError(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) != expected_count:
        raise ReferenceDataError(
            f"{path} contains {len(frame)} trades; expected {expected_count}"
        )

    dates = pd.to_datetime(frame["date"], errors="raise")
    start = pd.Timestamp(ARTICLE_WINDOW_START)
    end = pd.Timestamp(ARTICLE_WINDOW_END_EXCLUSIVE)
    if ((dates < start) | (dates >= end)).any():
        raise ReferenceDataError(f"{path} contains a trade outside the article window")
    if not dates.is_monotonic_increasing:
        raise ReferenceDataError(f"{path} is not sorted by trade date")
    if dates.duplicated().any():
        raise ReferenceDataError(f"{path} contains more than one trade on an ET date")

    if not set(frame["side"]).issubset({"long", "short"}):
        raise ReferenceDataError(f"{path} contains an unknown side")
    if not set(frame["exit_reason"]).issubset({"sl", "tp", "close_16", "eod"}):
        raise ReferenceDataError(f"{path} contains an unknown exit_reason")

    entry_times = pd.to_datetime(frame["entry_time"], utc=True, errors="raise")
    exit_times = pd.to_datetime(frame["exit_time"], utc=True, errors="raise")
    if (exit_times < entry_times).any():
        raise ReferenceDataError(f"{path} contains an exit before entry")

    numeric_columns = [
        "entry_ref", "entry_fill", "sl", "tp", "risk_width", "units",
        "exit_price", "pnl", "equity_before", "equity_after", "r_net",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ReferenceDataError(f"{path} contains a non-finite numeric value")
    if (numeric["risk_width"] <= 0).any() or (numeric["units"] <= 0).any():
        raise ReferenceDataError(f"{path} contains a non-positive risk or size")
    if not np.allclose(
        numeric["equity_after"],
        numeric["equity_before"] + numeric["pnl"],
        rtol=1e-11,
        atol=1e-6,
    ):
        raise ReferenceDataError(f"{path} has an inconsistent equity recurrence")
    if not np.isclose(
        float(numeric["equity_before"].iloc[0]), INITIAL_CASH, atol=1e-6
    ):
        raise ReferenceDataError(f"{path} does not start at INITIAL_CASH")
    if len(frame) > 1 and not np.allclose(
        numeric["equity_before"].iloc[1:].to_numpy(),
        numeric["equity_after"].iloc[:-1].to_numpy(),
        rtol=1e-11,
        atol=1e-6,
    ):
        raise ReferenceDataError(f"{path} has a gap in the equity chain")


def _read_reference_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ReferenceDataError(f"reference file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pandas reports parser/encoding errors separately.
        raise ReferenceDataError(f"failed to read reference file: {path}") from exc


def verify_reference_csvs(reference_dir: Path | str) -> ReferenceVerification:
    """Validate the frozen article CSVs and reconstruct all reported metrics."""

    directory = Path(reference_dir)
    main_path = directory / MAIN_TRADES_FILENAME
    ref_path = directory / REF_TRADES_FILENAME
    segment_path = directory / SEGMENTS_FILENAME
    main = _read_reference_csv(main_path)
    reference = _read_reference_csv(ref_path)
    saved_segments = _read_reference_csv(segment_path)
    _validate_trade_frame(main, path=main_path, expected_count=EXPECTED_MAIN_TRADES)
    _validate_trade_frame(reference, path=ref_path, expected_count=EXPECTED_REF_TRADES)

    main_summary = summarize_frame(main, "main net reference")
    ref_summary = summarize_frame(reference, "reference net reference")
    recomputed = main_summary.segments

    required_segment_columns = {
        "segment", "start", "end_exclusive", "trade_count", "pnl"
    }
    missing = required_segment_columns.difference(saved_segments.columns)
    if missing:
        raise ReferenceDataError(
            f"{segment_path} is missing columns: {sorted(missing)}"
        )
    if len(saved_segments) != N_SEGMENTS:
        raise ReferenceDataError(
            f"{segment_path} contains {len(saved_segments)} segments; expected 5"
        )

    saved = saved_segments.sort_values("segment").reset_index(drop=True).copy()
    actual = recomputed.sort_values("segment").reset_index(drop=True).copy()
    for column in ("start", "end_exclusive"):
        saved[column] = pd.to_datetime(saved[column], errors="raise")
        actual[column] = pd.to_datetime(actual[column], errors="raise")
    for column in ("segment", "start", "end_exclusive", "trade_count"):
        if not saved[column].equals(actual[column]):
            raise ReferenceDataError(
                f"{segment_path} column {column!r} differs from main trades"
            )
    if not np.allclose(saved["pnl"], actual["pnl"], rtol=1e-11, atol=0.01):
        raise ReferenceDataError(
            f"{segment_path} PnL differs from the recomputed main-trade segments"
        )

    if tuple(actual["trade_count"].astype(int)) != EXPECTED_SEGMENT_COUNTS:
        raise ReferenceDataError("article segment trade counts changed")
    if not np.allclose(
        actual["pnl"], EXPECTED_SEGMENT_PNL, rtol=1e-11, atol=0.01
    ):
        raise ReferenceDataError("article segment PnL changed")
    if not np.isclose(
        main_summary.final_equity, EXPECTED_MAIN_FINAL_EQUITY, atol=0.01
    ):
        raise ReferenceDataError("article main final equity changed")
    if not np.isclose(
        ref_summary.final_equity, EXPECTED_REF_FINAL_EQUITY, atol=0.01
    ):
        raise ReferenceDataError("article reference final equity changed")
    if main_summary.exit_counts != {"close_16": 76, "sl": 5, "tp": 1}:
        raise ReferenceDataError("article main exit-reason counts changed")
    if main_summary.positive_segments != 3:
        raise ReferenceDataError("article positive-segment count changed")

    return ReferenceVerification(
        reference_dir=directory,
        main_summary=main_summary,
        ref_summary=ref_summary,
        segments=recomputed,
    )


def _summaries_frame(summaries: list[Summary]) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "label": summary.label,
                "trades": summary.n,
                "win_rate_pct": summary.win_rate,
                "return_pct": summary.return_pct,
                "max_drawdown_pct": summary.max_drawdown_pct,
                "avg_r": summary.avg_r,
                "positive_segments": summary.positive_segments,
                "final_equity": summary.final_equity,
                "long_trades": summary.n_long,
                "short_trades": summary.n_short,
                "exit_counts": json.dumps(summary.exit_counts, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def print_summary(summary: Summary) -> None:
    """Print one compact, human-readable reproduction summary."""

    print(f"\n[{summary.label}]")
    print(f"trades: {summary.n}")
    print(f"win rate: {summary.win_rate:.1f}% (break-even {BREAKEVEN_WR:.1f}%)")
    print(f"return: {summary.return_pct:+.2f}%")
    print(f"max drawdown: {summary.max_drawdown_pct:.2f}%")
    print(f"average realized R: {summary.avg_r:+.3f}")
    print(f"exit reasons: {summary.exit_counts}")
    print(f"positive fixed segments: {summary.positive_segments}/{N_SEGMENTS}")


def run_reference(
    reference_dir: Path | str,
    output_dir: Path | str | None = None,
) -> ReferenceVerification:
    """Verify frozen article CSVs without network access and print the result.

    When ``output_dir`` is supplied, a two-row summary and the independently
    recomputed fixed-segment CSV are written there.  Trade files are not copied.
    """

    verification = verify_reference_csvs(reference_dir)
    print("Reference CSV verification: OK")
    print_summary(verification.main_summary)
    print_summary(verification.ref_summary)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _summaries_frame(
            [verification.main_summary, verification.ref_summary]
        ).to_csv(output / REFERENCE_SUMMARY_FILENAME, index=False)
        write_segments_csv(verification.segments, output / SEGMENTS_FILENAME)
    return verification


def run_live(output_dir: Path | str) -> BacktestResult:
    """Fetch live data, run the fixed article window, write CSVs, and print."""

    result = run_backtest()
    write_result_csvs(result, output_dir)
    print(
        "Live Yahoo Finance run for fixed ET window: "
        f"[{ARTICLE_WINDOW_START}, {ARTICLE_WINDOW_END_EXCLUSIVE})"
    )
    print_summary(result.main_summary)
    print_summary(result.gross_summary)
    print_summary(result.ref_summary)
    return result


__all__ = [
    "ARTICLE_WINDOW_END_EXCLUSIVE",
    "ARTICLE_WINDOW_START",
    "ATR_HI",
    "ATR_LO",
    "ATR_N",
    "AtrFilterStats",
    "BacktestResult",
    "CLOSE_HOUR_ET",
    "CUTOFF_HOUR_ET",
    "DataCoverageError",
    "EXPECTED_MAIN_TRADES",
    "EXPECTED_REF_TRADES",
    "INITIAL_CASH",
    "MAIN_TRADES_FILENAME",
    "N_SEGMENTS",
    "RANGE_HOUR_ET",
    "REFERENCE_SUMMARY_FILENAME",
    "REF_TRADES_FILENAME",
    "RR",
    "ReferenceDataError",
    "ReferenceVerification",
    "SEGMENTS_FILENAME",
    "SEGMENT_DAYS",
    "SPREAD_SOURCE_NOTE",
    "Summary",
    "Trade",
    "USDJPY_SPREAD",
    "WINDOW_DAYS",
    "article_window",
    "atr_filter_stats",
    "fixed_segment_edges",
    "fixed_segment_summary",
    "load_et_bars",
    "print_summary",
    "run_backtest",
    "run_live",
    "run_reference",
    "simulate",
    "slice_window",
    "summarize",
    "summarize_frame",
    "trades_frame",
    "verify_reference_csvs",
    "window_bounds",
    "write_result_csvs",
    "write_segments_csv",
    "write_trades_csv",
]
