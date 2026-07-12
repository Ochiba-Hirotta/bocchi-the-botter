"""Shared execution helpers for chapter-level reproduction scripts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .backtest.adapter import to_backtest_frame
from .backtest.analysis.centroid import (
    PAIR_METRICS,
    PAIRS,
    compute_centroid,
    compute_grid_summary,
)
from .backtest.analysis.physical_metrics import (
    PhysicalMetrics,
    aggregate_per_grid,
    compute_physical_metrics,
    metrics_to_dict,
)
from .backtest.analysis.strategy_compare import (
    common_survival,
    compute_centroid as compute_strategy_centroid,
    compute_grid_summary as compute_strategy_grid_summary,
    fold_alignment,
    spearman_corr,
)
from .backtest.runner import ExecutionMeta, FXBacktestRunner
from .backtest.strategies import BBMeanReversion, DonchianBreakout
from .backtest.walk_forward import WalkForwardRunner, make_strategy_subclass
from .data import YfinanceProvider


DEFAULT_END_DATE_STR = "2026-04-29T00:00:00Z"
DEFAULT_END_DATE = datetime(2026, 4, 29, tzinfo=timezone.utc)
DEFAULT_INTERVAL = "1h"
DEFAULT_SHORT_DAYS = 180
DEFAULT_LONG_DAYS = 720


@dataclass(frozen=True)
class SpreadSpec:
    """One-way spread value and source note."""

    value: float
    note: str


@dataclass(frozen=True)
class GridSpec:
    """One fixed grid used by chapter #7."""

    pair: str
    strategy: Literal["BB_MR", "Donchian"]
    params: dict[str, Any]
    grid_id: str


ARTICLE_1_SPREAD = SpreadSpec(
    1.33e-5,
    "Article #1 original setting: 0.2 sen / 150 ~= 1.33e-5.",
)

USDJPY_SPREAD = SpreadSpec(
    1.0e-5,
    "USDJPY 0.3 sen bid-ask width, one-way converted, base price ~= 150.",
)

GBPJPY_SPREAD_LOW = SpreadSpec(
    2.135e-5,
    "GBPJPY 0.9 sen lower bound, one-way, base price=210.7530.",
)
GBPJPY_SPREAD_MID = SpreadSpec(
    4.033e-5,
    "GBPJPY 1.7 sen median, one-way, base price=210.7530.",
)
GBPJPY_SPREAD_HIGH = SpreadSpec(
    5.931e-5,
    "GBPJPY 2.5 sen upper bound, one-way, base price=210.7530.",
)

PAIR_SPREADS: dict[str, SpreadSpec] = {
    "USDJPY": USDJPY_SPREAD,
    "GBPJPY": GBPJPY_SPREAD_MID,
    "EURJPY": SpreadSpec(
        2.661e-5,
        "EURJPY 0.9 sen OANDA median, one-way, base price=169.176.",
    ),
    "AUDJPY": SpreadSpec(
        4.826e-5,
        "AUDJPY 0.95 sen OANDA median, one-way, base price=98.416.",
    ),
}

BB_MR_PARAM_GRID: dict[str, list[float]] = {
    "BB_N": [10, 14, 20, 28, 40],
    "BB_K": [1.5, 2.0, 2.5, 3.0],
}
DONCHIAN_PARAM_GRID: dict[str, list[float]] = {
    "DC_N": [10, 20, 40, 80, 160],
    "DC_EXIT": [0.5, 1.0, 1.5, 2.0],
}
TRAIN_DAYS = 180
TEST_DAYS = 90
STEP_DAYS = 90

PHYSICAL_GRID_SPECS: tuple[GridSpec, ...] = (
    GridSpec("USDJPY", "BB_MR", {"BB_N": 14, "BB_K": 1.5}, "BB_N14_K1p5"),
    GridSpec("USDJPY", "Donchian", {"DC_N": 10, "DC_EXIT": 1.5}, "DC_N10_EXIT1p5"),
    GridSpec("GBPJPY", "BB_MR", {"BB_N": 28, "BB_K": 2.5}, "BB_N28_K2p5"),
    GridSpec("GBPJPY", "Donchian", {"DC_N": 10, "DC_EXIT": 1.0}, "DC_N10_EXIT1p0"),
    GridSpec("EURJPY", "BB_MR", {"BB_N": 14, "BB_K": 2.0}, "BB_N14_K2p0"),
    GridSpec("EURJPY", "Donchian", {"DC_N": 10, "DC_EXIT": 2.0}, "DC_N10_EXIT2p0"),
    GridSpec("AUDJPY", "BB_MR", {"BB_N": 28, "BB_K": 2.5}, "BB_N28_K2p5"),
    GridSpec("AUDJPY", "Donchian", {"DC_N": 40, "DC_EXIT": 0.5}, "DC_N40_EXIT0p5"),
)


def parse_iso_utc(value: str) -> datetime:
    """Parse an ISO-8601 datetime and require timezone information."""
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid ISO-8601 datetime: {value!r}"
        ) from exc
    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"datetime must include timezone: {value!r}"
        )
    return dt.astimezone(timezone.utc)


def parse_pairs(value: str, allowed: tuple[str, ...] = PAIRS) -> list[str]:
    """Parse a comma-separated pair list."""
    pairs = [p.strip().upper() for p in value.split(",") if p.strip()]
    invalid = [p for p in pairs if p not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported pair(s): {invalid}. allowed={allowed}"
        )
    if not pairs:
        raise argparse.ArgumentTypeError("at least one pair is required")
    return pairs


def chapter_output_dir(
    repo_root: Path, chapter_slug: str, output_dir: Path | None
) -> Path:
    """Return the output directory for a chapter and create it."""
    path = output_dir if output_dir is not None else repo_root / "outputs" / chapter_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_spread(pair: str) -> SpreadSpec:
    try:
        return PAIR_SPREADS[pair]
    except KeyError as exc:
        raise ValueError(f"unsupported pair for spread: {pair}") from exc


def fetch_backtest_frame(
    *,
    pair: str,
    start: datetime,
    end: datetime,
    cache_root: Path | None,
) -> tuple[pd.DataFrame, datetime, int]:
    """Fetch yfinance bars and convert them to backtesting.py format."""
    provider = YfinanceProvider(cache_root=cache_root)
    fetched_at = datetime.now(tz=timezone.utc)
    raw = provider.fetch_bars(pair=pair, interval=DEFAULT_INTERVAL, start=start, end=end)
    df = to_backtest_frame(raw, expected_freq=None)
    expected = expected_24x5_bars(start, end)
    missing = max(0, expected - len(df))
    return df, fetched_at, missing


def expected_24x5_bars(start: datetime, end: datetime) -> int:
    """Approximate expected 1h FX bars under a 24/5 week."""
    total_hours = (end - start).total_seconds() / pd.Timedelta("1h").total_seconds()
    return int(total_hours * 5 / 7)


def run_bb_mr_cases(
    *,
    pair: str,
    start: datetime,
    end: datetime,
    cases: list[tuple[str, SpreadSpec]],
    cache_root: Path | None,
    output_dir: Path,
    output_prefix: str,
) -> pd.DataFrame:
    """Run BB-MR on one fetched dataset for one or more spread cases."""
    df, fetched_at, missing = fetch_backtest_frame(
        pair=pair, start=start, end=end, cache_root=cache_root
    )
    rows: list[dict[str, Any]] = []
    artifacts: list[tuple[str, pd.Series]] = []
    for label, spread in cases:
        stats, meta = run_bb_mr_on_frame(
            df=df,
            spread=spread.value,
            spread_note=spread.note,
            fetched_at=fetched_at,
            missing_bars=missing,
        )
        rows.append(
            summarize_stats(
                label=label,
                pair=pair,
                requested_start=start,
                requested_end=end,
                stats=stats,
                meta=meta,
                spread=spread,
            )
        )
        artifacts.append((label, stats))

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"{output_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_stats_artifacts(output_dir, output_prefix, artifacts)
    print(f"[output] summary: {summary_path}")
    return summary


def run_bb_mr_on_frame(
    *,
    df: pd.DataFrame,
    spread: float,
    spread_note: str,
    fetched_at: datetime,
    missing_bars: int,
) -> tuple[pd.Series, ExecutionMeta]:
    """Run BB-MR for a prepared backtesting.py frame."""
    runner = FXBacktestRunner()
    return runner.run(
        df,
        BBMeanReversion,
        spread=spread,
        spread_source_note=spread_note,
        data_fetched_at_utc=fetched_at,
        missing_bars=missing_bars,
    )


def run_bb_mr_segments(
    *,
    pair: str,
    start: datetime,
    end: datetime,
    spread: SpreadSpec,
    cache_root: Path | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Run BB-MR on first half, second half, and full span."""
    df, fetched_at, missing_full = fetch_backtest_frame(
        pair=pair, start=start, end=end, cache_root=cache_root
    )
    mid = len(df) // 2
    segments: list[tuple[str, pd.DataFrame, int]] = [
        ("first_half", df.iloc[:mid], max(0, missing_full // 2)),
        ("second_half", df.iloc[mid:], max(0, missing_full - missing_full // 2)),
        ("full", df, missing_full),
    ]

    rows: list[dict[str, Any]] = []
    artifacts: list[tuple[str, pd.Series]] = []
    for label, segment_df, missing in segments:
        stats, meta = run_bb_mr_on_frame(
            df=segment_df,
            spread=spread.value,
            spread_note=spread.note,
            fetched_at=fetched_at,
            missing_bars=missing,
        )
        row = summarize_stats(
            label=label,
            pair=pair,
            requested_start=start,
            requested_end=end,
            stats=stats,
            meta=meta,
            spread=spread,
        )
        row["split_mid_index"] = mid
        rows.append(row)
        artifacts.append((f"{pair}_{label}", stats))

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"bb_mr_{pair.lower()}_segments_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_stats_artifacts(output_dir, f"bb_mr_{pair.lower()}_segments", artifacts)
    print(f"[output] summary: {summary_path}")
    return summary


def summarize_stats(
    *,
    label: str,
    pair: str,
    requested_start: datetime,
    requested_end: datetime,
    stats: pd.Series,
    meta: ExecutionMeta,
    spread: SpreadSpec,
) -> dict[str, Any]:
    """Convert backtesting.py stats into a stable CSV row."""
    mdd = _safe_float(stats.get("Max. Drawdown [%]"))
    cagr = _safe_float(stats.get("CAGR [%]", stats.get("Return (Ann.) [%]")))
    mar = cagr / abs(mdd) if mdd and not np.isnan(mdd) else float("nan")
    total_trades = int(_safe_float(stats.get("# Trades")))
    finalized = count_trades_closing_on_last_bar(stats)
    strategy = stats.get("_strategy")
    return {
        "label": label,
        "pair": pair,
        "requested_start_utc": requested_start.isoformat(),
        "requested_end_utc": requested_end.isoformat(),
        "period_start_utc": meta.period_start_utc.isoformat(),
        "period_end_utc": meta.period_end_utc.isoformat(),
        "effective_bars": meta.effective_bars,
        "missing_bars": meta.missing_bars,
        "spread": spread.value,
        "spread_note": spread.note,
        "equity_final": _safe_float(stats.get("Equity Final [$]")),
        "return_pct": _safe_float(stats.get("Return [%]")),
        "return_ann_pct": _safe_float(stats.get("Return (Ann.) [%]")),
        "cagr_pct": cagr,
        "sharpe": _safe_float(stats.get("Sharpe Ratio")),
        "max_drawdown_pct": mdd,
        "mar": mar,
        "profit_factor": _safe_float(stats.get("Profit Factor")),
        "win_rate_pct": _safe_float(stats.get("Win Rate [%]")),
        "n_trades": total_trades,
        "avg_trade_duration": str(stats.get("Avg. Trade Duration")),
        "finalized_on_last_bar": finalized,
        "finalized_on_last_bar_pct": finalized / total_trades if total_trades else 0.0,
        "missed_entries": getattr(strategy, "missed_entries", 0),
    }


def count_trades_closing_on_last_bar(stats: pd.Series) -> int:
    """Approximate finalize_trades count via ExitBar == last equity bar."""
    trades = stats.get("_trades")
    equity_curve = stats.get("_equity_curve")
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return 0
    if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
        return 0
    return int((trades["ExitBar"] == len(equity_curve) - 1).sum())


def write_stats_artifacts(
    output_dir: Path, output_prefix: str, artifacts: list[tuple[str, pd.Series]]
) -> None:
    """Write trades and equity curves for a set of stats objects."""
    for label, stats in artifacts:
        safe = safe_name(label)
        trades = stats.get("_trades")
        equity = stats.get("_equity_curve")
        if isinstance(trades, pd.DataFrame):
            trades.to_csv(output_dir / f"{output_prefix}_{safe}_trades.csv", index=False)
        if isinstance(equity, pd.DataFrame):
            equity.to_csv(output_dir / f"{output_prefix}_{safe}_equity.csv")


def run_wfa_pairs(
    *,
    strategy: Literal["bb_mr", "donchian"],
    pairs: list[str],
    mode: Literal["smoke", "full"],
    days: int,
    end_date: datetime,
    cache_root: Path | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Run WFA for one strategy over multiple pairs."""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        out_path, df_out, failures = run_wfa_for_pair(
            strategy=strategy,
            pair=pair,
            mode=mode,
            days=days,
            end_date=end_date,
            cache_root=cache_root,
            output_dir=output_dir,
        )
        rows.append(
            {
                "strategy": strategy,
                "pair": pair,
                "mode": mode,
                "rows": len(df_out),
                "folds": int(df_out["fold"].nunique()) if "fold" in df_out else 0,
                "failures": failures,
                "output": str(out_path),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"wfa_{strategy}_run_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[output] run summary: {summary_path}")
    return summary


def run_wfa_for_pair(
    *,
    strategy: Literal["bb_mr", "donchian"],
    pair: str,
    mode: Literal["smoke", "full"],
    days: int,
    end_date: datetime,
    cache_root: Path | None,
    output_dir: Path,
) -> tuple[Path, pd.DataFrame, int]:
    """Run one WFA job and write the long-format CSV."""
    start = end_date - timedelta(days=days)
    df, _fetched_at, _missing = fetch_backtest_frame(
        pair=pair, start=start, end=end_date, cache_root=cache_root
    )
    spread = get_spread(pair)
    base_strategy: type
    param_grid: dict[str, list[float]]
    if strategy == "bb_mr":
        base_strategy = BBMeanReversion
        param_grid = BB_MR_PARAM_GRID
    elif strategy == "donchian":
        base_strategy = DonchianBreakout
        param_grid = DONCHIAN_PARAM_GRID
    else:
        raise ValueError(f"unsupported strategy: {strategy}")

    wfa = WalkForwardRunner(
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
    )
    max_folds = 1 if mode == "smoke" else None
    result = wfa.run(
        df,
        base_strategy=base_strategy,
        param_grid=param_grid,
        spread=spread.value,
        max_folds=max_folds,
    )
    df_out = result.to_dataframe()

    date_label = end_date.date().isoformat()
    mode_part = "_smoke" if mode == "smoke" else ""
    out_path = output_dir / f"wfa_{strategy}_{pair}{mode_part}_{date_label}.csv"
    df_out.to_csv(out_path, index=False)
    print(
        f"[output] {strategy} {pair}: {out_path} "
        f"({len(result.folds_metadata)} folds, {len(result.failures)} failures)"
    )
    return out_path, df_out, len(result.failures)


def write_bb_mr_centroid_summary(
    *, wfa_dir: Path, output_path: Path, end_date: datetime
) -> pd.DataFrame:
    """Build the chapter #5 centroid summary from BB-MR WFA CSVs."""
    rows: list[dict[str, Any]] = []
    date_label = end_date.date().isoformat()
    for pair in PAIRS:
        csv = first_existing(
            [
                wfa_dir / f"wfa_bb_mr_{pair}_{date_label}.csv",
                wfa_dir / f"wfa_bb_mr_{pair}.csv",
            ]
        )
        df = pd.read_csv(csv)
        grid_summary = compute_grid_summary(df)
        centroid = compute_centroid(grid_summary)
        rows.append(
            {
                "pair": pair,
                "n_centroid": centroid.n_centroid,
                "k_centroid": centroid.k_centroid,
                "n_positive": centroid.n_positive,
                "n_grid_total": centroid.n_grid_total,
                "n_degenerate": len(centroid.degenerate_grids),
                **PAIR_METRICS[pair],
            }
        )
    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[output] centroid summary: {output_path}")
    return out_df


def write_strategy_compare_outputs(
    *, wfa_dir: Path, output_dir: Path, end_date: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build chapter #6 strategy comparison CSVs from BB-MR and Donchian WFA CSVs."""
    date_label = end_date.date().isoformat()
    bb_per_pair: dict[str, pd.DataFrame] = {}
    dc_per_pair: dict[str, pd.DataFrame] = {}
    bb_grid: dict[str, pd.DataFrame] = {}
    dc_grid: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        bb_csv = first_existing(
            [
                wfa_dir / f"wfa_bb_mr_{pair}_{date_label}.csv",
                wfa_dir / f"wfa_bb_mr_{pair}.csv",
            ]
        )
        dc_csv = first_existing(
            [
                wfa_dir / f"wfa_donchian_{pair}_{date_label}.csv",
                wfa_dir / f"wfa_donchian_{pair}.csv",
            ]
        )
        bb_per_pair[pair] = pd.read_csv(bb_csv)
        dc_per_pair[pair] = pd.read_csv(dc_csv)
        bb_grid[pair] = compute_strategy_grid_summary(bb_per_pair[pair], "bb_mr")
        dc_grid[pair] = compute_strategy_grid_summary(dc_per_pair[pair], "donchian")

    fold_df = fold_alignment(bb_per_pair, dc_per_pair)
    fold_path = output_dir / "wfa_strategy_compare.csv"
    fold_df.to_csv(fold_path, index=False)

    bb_common = common_survival(bb_grid, "bb_mr")
    dc_common = common_survival(dc_grid, "donchian")
    bb_ns: list[float] = []
    dc_ns: list[float] = []
    rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        bb_n, bb_e, bb_pos, bb_total = compute_strategy_centroid(
            bb_grid[pair], "bb_mr"
        )
        dc_n, dc_e, dc_pos, dc_total = compute_strategy_centroid(
            dc_grid[pair], "donchian"
        )
        bb_ns.append(bb_n)
        dc_ns.append(dc_n)
        rows.append(
            {
                "pair": pair,
                "bb_mr_n_centroid": bb_n,
                "bb_mr_k_centroid": bb_e,
                "bb_mr_n_positive": bb_pos,
                "bb_mr_n_total": bb_total,
                "donchian_dc_n_centroid": dc_n,
                "donchian_dc_exit_centroid": dc_e,
                "donchian_n_positive": dc_pos,
                "donchian_n_total": dc_total,
            }
        )
    summary = pd.DataFrame(rows)
    summary["bb_common_alive_4"] = int((bb_common["alive_in"] == 4).sum())
    summary["donchian_common_alive_4"] = int((dc_common["alive_in"] == 4).sum())
    summary["sign_flip_folds"] = int(fold_df["sign_flip"].sum())
    summary["spearman_bb_n_vs_dc_n"] = spearman_corr(bb_ns, dc_ns)

    summary_path = output_dir / "wfa_strategy_compare_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[output] fold comparison: {fold_path}")
    print(f"[output] comparison summary: {summary_path}")
    return fold_df, summary


def run_physical_metrics_from_trades(
    *, trades_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute chapter #7 metrics from existing trade CSV files."""
    per_fold_rows: list[dict[str, Any]] = []
    per_grid_rows: list[dict[str, Any]] = []
    for spec in PHYSICAL_GRID_SPECS:
        spread = get_spread(spec.pair)
        fold_metrics: list[PhysicalMetrics] = []
        for trades_path in sorted(
            trades_dir.glob(f"trades_7_{spec.strategy}_{spec.pair}_{spec.grid_id}_fold*.csv")
        ):
            fold = _parse_fold_number(trades_path)
            trades = pd.read_csv(trades_path)
            metrics = compute_physical_metrics(trades, spread.value)
            fold_metrics.append(metrics)
            per_fold_rows.append(
                {
                    "pair": spec.pair,
                    "strategy": spec.strategy,
                    "grid_id": spec.grid_id,
                    "fold": fold,
                    **metrics_to_dict(metrics),
                }
            )
        if not fold_metrics:
            raise FileNotFoundError(
                f"no trade CSVs for {spec.pair} {spec.strategy} {spec.grid_id} "
                f"under {trades_dir}"
            )
        aggregate = aggregate_per_grid(fold_metrics)
        per_grid_rows.append(
            {
                "pair": spec.pair,
                "strategy": spec.strategy,
                "grid_id": spec.grid_id,
                **metrics_to_dict(aggregate),
            }
        )

    return write_physical_metric_outputs(
        per_fold_rows,
        per_grid_rows,
        output_dir,
        reference_per_fold_path=trades_dir / "wfa_results_7_per_fold.csv",
        reference_per_grid_path=trades_dir / "wfa_results_7_per_grid.csv",
    )


def recompute_physical_metrics_from_market(
    *,
    cache_root: Path | None,
    end_date: datetime,
    days: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute chapter #7 trades and metrics from market data."""
    df_cache: dict[str, pd.DataFrame] = {}
    per_fold_rows: list[dict[str, Any]] = []
    per_grid_rows: list[dict[str, Any]] = []
    start = end_date - timedelta(days=days)
    for spec in PHYSICAL_GRID_SPECS:
        spread = get_spread(spec.pair)
        if spec.pair not in df_cache:
            df_cache[spec.pair], _fetched_at, _missing = fetch_backtest_frame(
                pair=spec.pair, start=start, end=end_date, cache_root=cache_root
            )
        df = df_cache[spec.pair]
        fold_metrics = run_physical_grid(
            spec=spec,
            df=df,
            spread=spread.value,
            output_dir=output_dir,
            dump_trades=True,
        )
        per_fold_rows.extend(fold_metrics[1])
        aggregate = aggregate_per_grid(fold_metrics[0])
        per_grid_rows.append(
            {
                "pair": spec.pair,
                "strategy": spec.strategy,
                "grid_id": spec.grid_id,
                **metrics_to_dict(aggregate),
            }
        )
    return write_physical_metric_outputs(per_fold_rows, per_grid_rows, output_dir)


def run_physical_grid(
    *,
    spec: GridSpec,
    df: pd.DataFrame,
    spread: float,
    output_dir: Path,
    dump_trades: bool,
) -> tuple[list[PhysicalMetrics], list[dict[str, Any]]]:
    """Run one chapter #7 grid over all test folds."""
    wfa = WalkForwardRunner(
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
    )
    base_cls = BBMeanReversion if spec.strategy == "BB_MR" else DonchianBreakout
    strategy_cls = make_strategy_subclass(base_cls, spec.params)
    runner = FXBacktestRunner()
    fold_metrics: list[PhysicalMetrics] = []
    rows: list[dict[str, Any]] = []
    for fold in wfa.make_folds(df):
        test_df = _slice_window(df, fold.test_start, fold.test_end)
        if len(test_df) < wfa.min_test_bars:
            continue
        stats, _meta = runner.run(test_df, strategy_cls, spread=spread)
        trades = stats["_trades"].copy()
        if dump_trades:
            trades.to_csv(
                output_dir
                / f"trades_7_{spec.strategy}_{spec.pair}_{spec.grid_id}_fold{fold.fold_index}.csv",
                index=False,
            )
        metrics = compute_physical_metrics(trades, spread)
        fold_metrics.append(metrics)
        rows.append(
            {
                "pair": spec.pair,
                "strategy": spec.strategy,
                "grid_id": spec.grid_id,
                "fold": fold.fold_index,
                **metrics_to_dict(metrics),
                "oos_sharpe": _safe_float(stats.get("Sharpe Ratio")),
                "oos_n_trades_raw": _safe_float(stats.get("# Trades")),
            }
        )
    return fold_metrics, rows


def write_physical_metric_outputs(
    per_fold_rows: list[dict[str, Any]],
    per_grid_rows: list[dict[str, Any]],
    output_dir: Path,
    reference_per_fold_path: Path | None = None,
    reference_per_grid_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write chapter #7 per-fold and per-grid CSVs."""
    per_fold = pd.DataFrame(per_fold_rows).sort_values(["pair", "strategy", "fold"])
    per_grid = pd.DataFrame(per_grid_rows).sort_values(["pair", "strategy"])
    if reference_per_fold_path is not None and reference_per_fold_path.exists():
        per_fold = align_with_reference_columns(
            per_fold,
            reference_per_fold_path,
            keys=["pair", "strategy", "grid_id", "fold"],
        )
    if reference_per_grid_path is not None and reference_per_grid_path.exists():
        per_grid = align_with_reference_columns(
            per_grid,
            reference_per_grid_path,
            keys=["pair", "strategy", "grid_id"],
        )
    per_fold_path = output_dir / "wfa_results_7_per_fold.csv"
    per_grid_path = output_dir / "wfa_results_7_per_grid.csv"
    per_fold.to_csv(per_fold_path, index=False)
    per_grid.to_csv(per_grid_path, index=False)
    print(f"[output] per-fold metrics: {per_fold_path}")
    print(f"[output] per-grid metrics: {per_grid_path}")
    return per_fold, per_grid


def align_with_reference_columns(
    frame: pd.DataFrame, reference_path: Path, keys: list[str]
) -> pd.DataFrame:
    """Align row order, column order, and optional auxiliary columns with reference."""
    reference = pd.read_csv(reference_path)
    if not all(key in reference.columns for key in keys):
        return frame

    merged = frame.copy()
    extra_columns = [col for col in reference.columns if col not in frame.columns]
    if extra_columns:
        merged = merged.merge(
            reference[keys + extra_columns],
            on=keys,
            how="left",
            validate="one_to_one",
        )

    reference_order = reference[keys].reset_index(names="_reference_order")
    merged = merged.merge(reference_order, on=keys, how="left", validate="one_to_one")
    if merged["_reference_order"].notna().all():
        merged = merged.sort_values("_reference_order")
    merged = merged.drop(columns=["_reference_order"])
    ordered_columns = [col for col in reference.columns if col in merged.columns]
    ordered_columns.extend(col for col in merged.columns if col not in ordered_columns)
    return merged[ordered_columns]


def first_existing(candidates: list[Path]) -> Path:
    """Return the first existing path or raise with all candidates."""
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "none of the candidate files exists: "
        + ", ".join(str(path) for path in candidates)
    )


def safe_name(value: str) -> str:
    """Return a conservative filename fragment."""
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)]


def _parse_fold_number(path: Path) -> int:
    stem = path.stem
    marker = "_fold"
    if marker not in stem:
        return -1
    try:
        return int(stem.rsplit(marker, 1)[1])
    except ValueError:
        return -1
