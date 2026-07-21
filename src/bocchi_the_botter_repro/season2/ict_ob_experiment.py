"""Stage-6 execution and reproducibility layer for the ICT OB experiment.

The detector rules live in :mod:`ict_ob` and remain frozen.  This module only
connects those rules to the fixed Season 2 M5 projection, calculates the
predeclared aggregate metrics, and writes price-bearing evidence below a
caller-selected private output directory.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import pandas as pd

from .ict_ob import (
    INITIAL_CASH,
    NEW_YORK,
    OFFICIAL_OB_DETECTOR,
    SECONDARY_OB_DETECTOR,
    BiasEvent,
    OfficialObDetectionResult,
    PendingZone,
    SecondaryObDetectionResult,
    TimeframeBundle,
    ZoneBacktestResult,
    ZoneTrade,
    annotate_m15_bias,
    build_daily_bias_timeline,
    detect_official_order_blocks,
    detect_secondary_order_blocks,
    find_confirmed_swings,
    make_official_daily_target_resolver,
    make_secondary_m15_target_resolver,
    prepare_timeframes,
    simulate_zone_backtest,
)
from .minute_data import M5Audit, SOURCE, audit_m5_frame, load_m5_candles
from .orb_m15 import (
    ARTICLE_WINDOW_END_EXCLUSIVE,
    ARTICLE_WINDOW_START,
    FrozenInputError,
    fixed_segment_edges,
    input_bounds,
    validate_frozen_input,
)


PRIVATE_SCHEMA_VERSION = 1
MINIMUM_SAMPLE_SIZE = 30

Criterion = Literal["passed", "failed", "insufficient_sample"]


@dataclass(frozen=True, slots=True)
class DetectorSummary:
    """Frozen stage-6 aggregate metrics for one detector translation."""

    detector: str
    zone_count: int
    trade_count: int
    open_position_count: int
    final_equity: float
    return_pct: float
    win_rate_pct: float | None
    max_drawdown_pct: float
    profit_factor: float | None
    average_realized_r: float | None
    positive_segments: int
    long_count: int
    short_count: int
    sample_sufficient: bool
    criterion: Criterion


@dataclass(slots=True)
class DetectorRunResult:
    """Private rows and row-free audit values for one detector."""

    detector: str
    zones: tuple[PendingZone, ...]
    backtest: ZoneBacktestResult
    summary: DetectorSummary
    segments: pd.DataFrame
    detection_counters: dict[str, int]
    execution_counters: dict[str, int]
    exit_reasons: dict[str, int]
    zone_sha256: str
    trade_sha256: str
    terminal_sha256: str
    result_sha256: str


@dataclass(slots=True)
class IctObExperimentResult:
    """One complete official/secondary run over a shared derived input."""

    input_audit: M5Audit
    complete_m15_count: int
    incomplete_m15_count: int
    accepted_daily_count: int
    rejected_daily_count: int
    daily_swing_count: int
    m15_swing_count: int
    bias_at_open_counts: dict[str, int]
    bias_at_close_counts: dict[str, int]
    official: DetectorRunResult
    secondary: DetectorRunResult
    result_sha256: str


ZONE_COLUMNS = tuple(field.name for field in fields(PendingZone))
TRADE_COLUMNS = tuple(field.name for field in fields(ZoneTrade))
SEGMENT_COLUMNS = (
    "segment",
    "start",
    "end_exclusive",
    "trade_count",
    "pnl_jpy",
    "return_pct",
)


def zones_frame(zones: Iterable[PendingZone]) -> pd.DataFrame:
    """Convert zones to the stable, private price-bearing schema."""

    return pd.DataFrame(
        [asdict(zone) for zone in zones],
        columns=ZONE_COLUMNS,
    )


def trades_frame(trades: Iterable[ZoneTrade]) -> pd.DataFrame:
    """Convert closed trades to the stable, private price-bearing schema."""

    return pd.DataFrame(
        [asdict(trade) for trade in trades],
        columns=TRADE_COLUMNS,
    )


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot hash a non-finite float")
        return format(value, ".17g")
    if hasattr(value, "item"):
        return _canonical_value(value.item())
    raise TypeError(f"unsupported canonical hash value: {type(value).__name__}")


def deterministic_sha256(kind: str, records: Iterable[Any]) -> str:
    """Hash an ordered record stream with stable float and JSON encoding."""

    if not kind:
        raise ValueError("hash kind must be non-empty")
    digest = hashlib.sha256()
    header = {"schema_version": PRIVATE_SCHEMA_VERSION, "kind": kind}
    for value in (header, *records):
        line = json.dumps(
            _canonical_value(value),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def fixed_segment_summary(
    trades: Iterable[ZoneTrade],
    *,
    window_start: dt.date = ARTICLE_WINDOW_START,
) -> pd.DataFrame:
    """Assign realized PnL to the same five 184-day ET segments as S2-3."""

    trade_list = list(trades)
    edges = fixed_segment_edges(window_start)
    entry_dates = [
        dt.datetime.fromtimestamp(trade.entry_time_utc, tz=dt.UTC)
        .astimezone(NEW_YORK)
        .date()
        for trade in trade_list
    ]
    if any(date < edges[0] or date >= edges[-1] for date in entry_dates):
        raise ValueError("one or more ICT OB trades fall outside the fixed window")

    rows: list[dict[str, Any]] = []
    segment_start_equity = INITIAL_CASH
    assigned = 0
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        selected = [
            trade
            for trade, entry_date in zip(trade_list, entry_dates, strict=True)
            if lower <= entry_date < upper
        ]
        pnl = float(sum(trade.pnl for trade in selected))
        rows.append(
            {
                "segment": index + 1,
                "start": lower.isoformat(),
                "end_exclusive": upper.isoformat(),
                "trade_count": len(selected),
                "pnl_jpy": pnl,
                "return_pct": pnl / segment_start_equity * 100.0,
            }
        )
        assigned += len(selected)
        segment_start_equity += pnl
    if assigned != len(trade_list):
        raise RuntimeError("ICT OB segment assignment contains a gap or overlap")
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def summarize_detector(
    detector: str,
    zones: Iterable[PendingZone],
    backtest: ZoneBacktestResult,
    segments: pd.DataFrame,
) -> DetectorSummary:
    """Calculate the stage-6 metrics without inventing empty-sample values."""

    zone_list = list(zones)
    trades = backtest.trades
    equity = [INITIAL_CASH, *(trade.equity_after for trade in trades)]
    peak = equity[0]
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, (value - peak) / peak * 100.0)
    gains = float(sum(trade.pnl for trade in trades if trade.pnl > 0))
    losses = float(-sum(trade.pnl for trade in trades if trade.pnl < 0))
    trade_count = len(trades)
    sample_sufficient = trade_count >= MINIMUM_SAMPLE_SIZE
    return_pct = (backtest.final_equity / INITIAL_CASH - 1.0) * 100.0
    positive_segments = int((segments["pnl_jpy"] > 0).sum())
    if not sample_sufficient:
        criterion: Criterion = "insufficient_sample"
    elif return_pct > 0 and positive_segments >= 3:
        criterion = "passed"
    else:
        criterion = "failed"
    return DetectorSummary(
        detector=detector,
        zone_count=len(zone_list),
        trade_count=trade_count,
        open_position_count=int(backtest.open_position is not None),
        final_equity=backtest.final_equity,
        return_pct=return_pct,
        win_rate_pct=(
            None
            if not trades
            else sum(trade.pnl > 0 for trade in trades) / trade_count * 100.0
        ),
        max_drawdown_pct=max_drawdown,
        profit_factor=None if losses == 0 else gains / losses,
        average_realized_r=(
            None
            if not trades
            else sum(trade.realized_r for trade in trades) / trade_count
        ),
        positive_segments=positive_segments,
        long_count=sum(trade.side == "long" for trade in trades),
        short_count=sum(trade.side == "short" for trade in trades),
        sample_sufficient=sample_sufficient,
        criterion=criterion,
    )


def _detector_result(
    detection: OfficialObDetectionResult | SecondaryObDetectionResult,
    backtest: ZoneBacktestResult,
    *,
    detector: str,
) -> DetectorRunResult:
    zones = detection.zones
    unexpected = sorted({zone.detector for zone in zones}.difference({detector}))
    if unexpected:
        raise ValueError(f"detector output contains unexpected identities: {unexpected}")
    segments = fixed_segment_summary(backtest.trades)
    summary = summarize_detector(detector, zones, backtest, segments)
    exit_reasons = dict(
        sorted(Counter(trade.exit_reason for trade in backtest.trades).items())
    )
    zone_hash = deterministic_sha256(f"{detector}:zones", zones)
    trade_hash = deterministic_sha256(f"{detector}:trades", backtest.trades)
    terminal_payload = {
        "final_equity": backtest.final_equity,
        "open_position": backtest.open_position,
        "pending_zone": backtest.pending_zone,
    }
    terminal_hash = deterministic_sha256(
        f"{detector}:terminal",
        [terminal_payload],
    )
    result_hash = deterministic_sha256(
        f"{detector}:result",
        [
            asdict(summary),
            detection.counters,
            backtest.counters,
            exit_reasons,
            segments.to_dict(orient="records"),
            zone_hash,
            trade_hash,
            terminal_hash,
        ],
    )
    return DetectorRunResult(
        detector=detector,
        zones=zones,
        backtest=backtest,
        summary=summary,
        segments=segments,
        detection_counters=detection.counters,
        execution_counters=backtest.counters,
        exit_reasons=exit_reasons,
        zone_sha256=zone_hash,
        trade_sha256=trade_hash,
        terminal_sha256=terminal_hash,
        result_sha256=result_hash,
    )


def run_ict_ob_from_timeframes(
    bundle: TimeframeBundle,
    *,
    input_audit: M5Audit,
    enforce_frozen_input: bool = True,
) -> IctObExperimentResult:
    """Run both frozen translations over one shared timeframe bundle."""

    if enforce_frozen_input:
        validate_frozen_input(
            input_audit,
            complete_m15_count=len(bundle.m15),
            incomplete_m15_count=len(bundle.incomplete_m15),
        )
    daily_swings = find_confirmed_swings(
        bundle.daily,
        timestamp_column="start_ts_utc",
        bar_close_column="available_ts_utc",
    )
    bias_timeline: tuple[BiasEvent, ...] = build_daily_bias_timeline(
        daily_swings,
        bundle.daily_status,
    )
    m15 = annotate_m15_bias(bundle.m15, bias_timeline)
    m15_swings = find_confirmed_swings(m15, bar_seconds=900)

    official_detection = detect_official_order_blocks(m15)
    official_backtest = simulate_zone_backtest(
        m15,
        official_detection.zones,
        target_resolver=make_official_daily_target_resolver(
            bundle.daily,
            daily_swings,
        ),
    )
    official = _detector_result(
        official_detection,
        official_backtest,
        detector=OFFICIAL_OB_DETECTOR,
    )

    secondary_detection = detect_secondary_order_blocks(m15)
    secondary_backtest = simulate_zone_backtest(
        m15,
        secondary_detection.zones,
        target_resolver=make_secondary_m15_target_resolver(m15, m15_swings),
    )
    secondary = _detector_result(
        secondary_detection,
        secondary_backtest,
        detector=SECONDARY_OB_DETECTOR,
    )

    accepted_daily_count = int(bundle.daily_status["accepted"].sum())
    rejected_daily_count = len(bundle.daily_status) - accepted_daily_count
    bias_at_open_counts = {
        str(key): int(value)
        for key, value in m15["daily_bias_at_open"]
        .value_counts()
        .sort_index()
        .items()
    }
    bias_at_close_counts = {
        str(key): int(value)
        for key, value in m15["daily_bias"].value_counts().sort_index().items()
    }
    experiment_payload = {
        "input_extraction_sha256": input_audit.extraction_sha256,
        "m5_rows": input_audit.row_count,
        "complete_m15_count": len(m15),
        "incomplete_m15_count": len(bundle.incomplete_m15),
        "accepted_daily_count": accepted_daily_count,
        "rejected_daily_count": rejected_daily_count,
        "daily_swing_count": len(daily_swings),
        "m15_swing_count": len(m15_swings),
        "bias_at_open_counts": bias_at_open_counts,
        "bias_at_close_counts": bias_at_close_counts,
        "official_result_sha256": official.result_sha256,
        "secondary_result_sha256": secondary.result_sha256,
    }
    return IctObExperimentResult(
        input_audit=input_audit,
        complete_m15_count=len(m15),
        incomplete_m15_count=len(bundle.incomplete_m15),
        accepted_daily_count=accepted_daily_count,
        rejected_daily_count=rejected_daily_count,
        daily_swing_count=len(daily_swings),
        m15_swing_count=len(m15_swings),
        bias_at_open_counts=bias_at_open_counts,
        bias_at_close_counts=bias_at_close_counts,
        official=official,
        secondary=secondary,
        result_sha256=deterministic_sha256(
            "ict_ob_stage6_experiment",
            [experiment_payload],
        ),
    )


def run_ict_ob_from_db(db_path: Path) -> IctObExperimentResult:
    """Read the fixed upstream SQLite in read-only mode and run both models."""

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
    bundle = prepare_timeframes(
        m5,
        start_inclusive=start,
        end_exclusive=end,
    )
    result = run_ict_ob_from_timeframes(bundle, input_audit=audit)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise FrozenInputError(
            "upstream SQLite changed during the read-only ICT OB run; "
            "retry a stable snapshot"
        )
    return result


def assert_reproducible(
    first: IctObExperimentResult,
    second: IctObExperimentResult,
) -> None:
    """Raise unless two complete runs reproduce the same semantic result."""

    if first.input_audit.extraction_sha256 != second.input_audit.extraction_sha256:
        raise FrozenInputError("stage-6 repeated runs used different M5 extractions")
    if first.result_sha256 != second.result_sha256:
        raise FrozenInputError(
            "stage-6 repeated runs produced different zones, trades, or counters"
        )


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detector_audit(result: DetectorRunResult) -> dict[str, Any]:
    return {
        "summary": asdict(result.summary),
        "segments": result.segments.to_dict(orient="records"),
        "detection_counters": result.detection_counters,
        "execution_counters": result.execution_counters,
        "exit_reasons": result.exit_reasons,
        "hashes": {
            "zones_sha256": result.zone_sha256,
            "trades_sha256": result.trade_sha256,
            "terminal_sha256": result.terminal_sha256,
            "result_sha256": result.result_sha256,
        },
        "terminal": {
            "open_position": result.backtest.open_position,
            "pending_zone": result.backtest.pending_zone,
        },
    }


def write_private_outputs(
    result: IctObExperimentResult,
    output_dir: Path,
    *,
    reproducibility_runs: int = 1,
) -> None:
    """Write private row evidence and a deterministic stage-6 audit."""

    if reproducibility_runs <= 0:
        raise ValueError("reproducibility_runs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    for label, detector in (
        ("official", result.official),
        ("secondary", result.secondary),
    ):
        zone_path = output_dir / f"{label}_zones_private.csv"
        trade_path = output_dir / f"{label}_trades_private.csv"
        segment_path = output_dir / f"{label}_segments.csv"
        zones_frame(detector.zones).to_csv(zone_path, index=False)
        trades_frame(detector.backtest.trades).to_csv(trade_path, index=False)
        detector.segments.to_csv(segment_path, index=False)
        artifact_paths.extend((zone_path, trade_path, segment_path))

    payload = {
        "schema_version": PRIVATE_SCHEMA_VERSION,
        "reproducibility_runs": reproducibility_runs,
        "article_window_et": {
            "start_inclusive": ARTICLE_WINDOW_START,
            "end_exclusive": ARTICLE_WINDOW_END_EXCLUSIVE,
        },
        "input_audit": asdict(result.input_audit),
        "derived": {
            "complete_m15_count": result.complete_m15_count,
            "incomplete_m15_count": result.incomplete_m15_count,
            "accepted_daily_count": result.accepted_daily_count,
            "rejected_daily_count": result.rejected_daily_count,
            "daily_swing_count": result.daily_swing_count,
            "m15_swing_count": result.m15_swing_count,
            "bias_at_open_counts": result.bias_at_open_counts,
            "bias_at_close_counts": result.bias_at_close_counts,
        },
        "official": _detector_audit(result.official),
        "secondary": _detector_audit(result.secondary),
        "result_sha256": result.result_sha256,
        "private_artifact_sha256": {
            path.name: _file_sha256(path) for path in artifact_paths
        },
    }
    audit_path = output_dir / "run_audit.json"
    audit_path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
