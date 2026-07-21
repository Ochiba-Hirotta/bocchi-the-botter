"""Stage-7 comparison for the three frozen ICT Order Block definitions.

The official-source and secondary translations retain the stage-6 shared
execution model.  ``smartmoneyconcepts==0.0.27`` is run with its defaults over
the same complete M15 bid OHLCV rows, but its output is used for zone comparison
only.  Price-bearing lifecycle rows remain private outputs.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import importlib.metadata
import inspect
import io
import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from .ict_ob import (
    M15_SECONDS,
    NEW_YORK,
    OSS_PACKAGE,
    OSS_PACKAGE_VERSION,
    TimeframeBundle,
    ZoneLifecycle,
    prepare_timeframes,
)
from .ict_ob_experiment import (
    IctObExperimentResult,
    assert_reproducible,
    deterministic_sha256,
    run_ict_ob_from_timeframes,
)
from .minute_data import M5Audit, SOURCE, audit_m5_frame, load_m5_candles
from .orb_m15 import (
    ARTICLE_WINDOW_END_EXCLUSIVE,
    ARTICLE_WINDOW_START,
    FrozenInputError,
    input_bounds,
)


OSS_DETECTOR = "smartmoneyconcepts_ob_0_0_27"
COMPARISON_SCHEMA_VERSION = 1
WindowEndReason = Literal[
    "consumed",
    "invalidated",
    "replaced",
    "data_end",
    "outside_window",
    "mitigated",
]


@dataclass(frozen=True, slots=True)
class ZoneWindow:
    """One price interval and half-open active time interval."""

    zone_id: str
    source: str
    side: str
    active_from_ts_utc: int
    end_exclusive_ts_utc: int
    lower: float
    upper: float
    end_reason: WindowEndReason


@dataclass(frozen=True, slots=True)
class OssLiquidityRecord:
    """One default-parameter OSS liquidity row retained for private audit."""

    liquidity_id: str
    side: str
    source_ts_utc: int
    level: float
    group_end_ts_utc: int
    swept_ts_utc: int | None


@dataclass(frozen=True, slots=True)
class OverlapSummary:
    """Symmetric any-overlap rates and the many-to-many pair count."""

    left_source: str
    right_source: str
    left_total: int
    left_overlapped: int
    left_overlap_pct: float | None
    right_total: int
    right_overlapped: int
    right_overlap_pct: float | None
    overlapping_pair_count: int


@dataclass(frozen=True, slots=True)
class OssSummary:
    """Row-free counts from the pinned third-party implementation."""

    package: str
    version: str
    raw_ob_count: int
    comparison_ob_count: int
    bullish_ob_count: int
    bearish_ob_count: int
    mitigated_ob_count: int
    active_at_data_end_count: int
    raw_liquidity_count: int
    comparison_liquidity_count: int
    bullish_liquidity_count: int
    bearish_liquidity_count: int


@dataclass(frozen=True, slots=True)
class OssAnalysis:
    """Private OSS rows and stable audit hashes."""

    zones: tuple[ZoneWindow, ...]
    liquidity: tuple[OssLiquidityRecord, ...]
    summary: OssSummary
    zone_sha256: str
    liquidity_sha256: str
    result_sha256: str


@dataclass(slots=True)
class IctObComparisonResult:
    """One complete stage-7 comparison over a single shared M15 derivative."""

    experiment: IctObExperimentResult
    official_windows: tuple[ZoneWindow, ...]
    secondary_windows: tuple[ZoneWindow, ...]
    oss: OssAnalysis
    official_secondary: OverlapSummary
    official_oss: OverlapSummary
    secondary_oss: OverlapSummary
    performance_delta_official_minus_secondary: dict[str, float | int | None]
    official_lifecycle_sha256: str
    secondary_lifecycle_sha256: str
    result_sha256: str


@dataclass(frozen=True, slots=True)
class _OssCreationEvent:
    ob_index: int
    side: str
    confirmation_index: int
    output_bottom: float
    output_top: float


ZONE_WINDOW_COLUMNS = tuple(field.name for field in fields(ZoneWindow))
LIQUIDITY_COLUMNS = tuple(field.name for field in fields(OssLiquidityRecord))


def lifecycle_windows(lifecycles: Iterable[ZoneLifecycle]) -> tuple[ZoneWindow, ...]:
    """Convert executor lifecycles without changing their price/time semantics."""

    return tuple(
        ZoneWindow(
            zone_id=item.zone_id,
            source=item.detector,
            side=item.side,
            active_from_ts_utc=item.active_from_ts_utc,
            end_exclusive_ts_utc=item.end_exclusive_ts_utc,
            lower=item.lower,
            upper=item.upper,
            end_reason=item.end_reason,
        )
        for item in lifecycles
    )


def _validate_window(window: ZoneWindow) -> None:
    if not window.zone_id or not window.source:
        raise ValueError("zone windows require non-empty identities")
    if window.side not in {"long", "short"}:
        raise ValueError("zone window side must be long or short")
    if window.end_exclusive_ts_utc < window.active_from_ts_utc:
        raise ValueError("zone window ends before it starts")
    if not all(math.isfinite(value) for value in (window.lower, window.upper)):
        raise ValueError("zone window contains a non-finite price")
    if window.lower > window.upper:
        raise ValueError("zone window lower price exceeds upper price")


def zones_overlap(left: ZoneWindow, right: ZoneWindow) -> bool:
    """Return the frozen price-inclusive/time-half-open overlap predicate."""

    _validate_window(left)
    _validate_window(right)
    price_intersects = max(left.lower, right.lower) <= min(left.upper, right.upper)
    time_intersects = max(
        left.active_from_ts_utc,
        right.active_from_ts_utc,
    ) < min(left.end_exclusive_ts_utc, right.end_exclusive_ts_utc)
    return price_intersects and time_intersects


def compare_zone_windows(
    left: Iterable[ZoneWindow],
    right: Iterable[ZoneWindow],
    *,
    left_source: str,
    right_source: str,
) -> OverlapSummary:
    """Count each zone once for rates while retaining every overlapping pair."""

    left_items = tuple(left)
    right_items = tuple(right)
    for item in (*left_items, *right_items):
        _validate_window(item)
    left_hits: set[int] = set()
    right_hits: set[int] = set()
    pair_count = 0
    for left_index, left_item in enumerate(left_items):
        for right_index, right_item in enumerate(right_items):
            if zones_overlap(left_item, right_item):
                left_hits.add(left_index)
                right_hits.add(right_index)
                pair_count += 1
    return OverlapSummary(
        left_source=left_source,
        right_source=right_source,
        left_total=len(left_items),
        left_overlapped=len(left_hits),
        left_overlap_pct=(
            None if not left_items else len(left_hits) / len(left_items) * 100.0
        ),
        right_total=len(right_items),
        right_overlapped=len(right_hits),
        right_overlap_pct=(
            None if not right_items else len(right_hits) / len(right_items) * 100.0
        ),
        overlapping_pair_count=pair_count,
    )


def _oss_ohlcv(m15: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ts_utc",
        "volume",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
    }
    missing = required.difference(m15.columns)
    if missing:
        raise ValueError(f"OSS M15 input is missing columns: {sorted(missing)}")
    timestamps = m15["ts_utc"].astype("int64")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("OSS M15 timestamps must be sorted and unique")
    result = pd.DataFrame(
        {
            "open": m15["bid_open"].to_numpy(dtype=float),
            "high": m15["bid_high"].to_numpy(dtype=float),
            "low": m15["bid_low"].to_numpy(dtype=float),
            "close": m15["bid_close"].to_numpy(dtype=float),
            "volume": m15["volume"].to_numpy(dtype=float),
        }
    )
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("OSS M15 input contains non-finite OHLCV values")
    return result


def _oss_creation_events(
    ohlcv: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
) -> tuple[_OssCreationEvent, ...]:
    """Recover 0.0.27's confirmation index, which its OB output omits.

    The returned OB prices still come from the package.  This adapter mirrors
    only the two break-trigger/search blocks needed to map each surviving OB
    source row to the candle where the package first knew about it.
    """

    highs = ohlcv["high"].to_numpy(dtype=float)
    lows = ohlcv["low"].to_numpy(dtype=float)
    closes = ohlcv["close"].to_numpy(dtype=float)
    swing_values = swing_highs_lows["HighLow"].to_numpy(dtype=float)
    swing_high_indices = np.flatnonzero(swing_values == 1)
    swing_low_indices = np.flatnonzero(swing_values == -1)
    crossed = np.full(len(ohlcv), False, dtype=bool)
    events: list[_OssCreationEvent] = []

    for close_index in range(len(ohlcv)):
        position = int(np.searchsorted(swing_high_indices, close_index))
        last_top_index = (
            int(swing_high_indices[position - 1]) if position > 0 else None
        )
        if (
            last_top_index is not None
            and closes[close_index] > highs[last_top_index]
            and not crossed[last_top_index]
        ):
            crossed[last_top_index] = True
            ob_index = close_index - 1
            output_bottom = highs[ob_index]
            output_top = lows[ob_index]
            if close_index - last_top_index > 1:
                start = last_top_index + 1
                segment = lows[start:close_index]
                if segment.size:
                    candidates = np.flatnonzero(segment == segment.min())
                    ob_index = start + int(candidates[-1])
                    output_bottom = lows[ob_index]
                    output_top = highs[ob_index]
            events.append(
                _OssCreationEvent(
                    ob_index=ob_index,
                    side="long",
                    confirmation_index=close_index,
                    output_bottom=output_bottom,
                    output_top=output_top,
                )
            )

    for close_index in range(len(ohlcv)):
        position = int(np.searchsorted(swing_low_indices, close_index))
        last_bottom_index = (
            int(swing_low_indices[position - 1]) if position > 0 else None
        )
        if (
            last_bottom_index is not None
            and closes[close_index] < lows[last_bottom_index]
            and not crossed[last_bottom_index]
        ):
            crossed[last_bottom_index] = True
            ob_index = close_index - 1
            if close_index - last_bottom_index > 1:
                start = last_bottom_index + 1
                segment = highs[start:close_index]
                if segment.size:
                    candidates = np.flatnonzero(segment == segment.max())
                    ob_index = start + int(candidates[-1])
            events.append(
                _OssCreationEvent(
                    ob_index=ob_index,
                    side="short",
                    confirmation_index=close_index,
                    output_bottom=lows[ob_index],
                    output_top=highs[ob_index],
                )
            )
    return tuple(events)


def _inside_article_window(timestamp: int) -> bool:
    date_et = dt.datetime.fromtimestamp(timestamp, tz=dt.UTC).astimezone(NEW_YORK).date()
    return ARTICLE_WINDOW_START <= date_et < ARTICLE_WINDOW_END_EXCLUSIVE


def build_oss_zone_windows(
    m15: pd.DataFrame,
    ohlcv: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
    ob_output: pd.DataFrame,
) -> tuple[ZoneWindow, ...]:
    """Map package OB rows to break-derived retrospective comparison windows."""

    required = {"OB", "Top", "Bottom", "MitigatedIndex"}
    missing = required.difference(ob_output.columns)
    if missing:
        raise ValueError(f"OSS OB output is missing columns: {sorted(missing)}")
    if len(m15) != len(ohlcv) or len(m15) != len(ob_output):
        raise ValueError("OSS input and output lengths differ")
    timestamps = m15["ts_utc"].to_numpy(dtype=np.int64)
    events_by_key: dict[tuple[int, str], list[_OssCreationEvent]] = {}
    for event in _oss_creation_events(ohlcv, swing_highs_lows):
        events_by_key.setdefault((event.ob_index, event.side), []).append(event)

    windows: list[ZoneWindow] = []
    observed = ob_output[ob_output["OB"].notna()]
    for ob_index, row in observed.iterrows():
        index = int(ob_index)
        side = "long" if int(row["OB"]) == 1 else "short"
        if int(row["OB"]) not in {-1, 1}:
            raise ValueError("OSS OB output contains an unexpected direction")
        output_bottom = float(row["Bottom"])
        output_top = float(row["Top"])
        lower = min(output_bottom, output_top)
        upper = max(output_bottom, output_top)
        matching = [
            event
            for event in events_by_key.get((index, side), [])
            if float(np.float32(event.output_bottom)) == output_bottom
            and float(np.float32(event.output_top)) == output_top
        ]
        if not matching:
            raise RuntimeError(
                f"cannot recover OSS confirmation index for output row {index}"
            )
        event = matching[-1]
        active_from = int(timestamps[event.confirmation_index]) + M15_SECONDS
        if not _inside_article_window(active_from):
            continue

        breach_index: int | None = None
        if side == "long":
            candidates = np.flatnonzero(
                ohlcv["low"].to_numpy(dtype=float)[event.confirmation_index + 1 :]
                < output_bottom
            )
        else:
            candidates = np.flatnonzero(
                ohlcv["high"].to_numpy(dtype=float)[event.confirmation_index + 1 :]
                > output_top
            )
        if candidates.size:
            breach_index = event.confirmation_index + 1 + int(candidates[0])
        reported_mitigation_index = int(row["MitigatedIndex"])
        expected_mitigation_index = (
            0
            if breach_index is None
            else breach_index - 1 if side == "long" else breach_index
        )
        if reported_mitigation_index != expected_mitigation_index:
            raise RuntimeError(
                "recovered OSS mitigation index differs from package output at "
                f"row {index}: {expected_mitigation_index} != "
                f"{reported_mitigation_index}"
            )
        if breach_index is None:
            end_exclusive = int(timestamps[-1]) + M15_SECONDS
            end_reason: WindowEndReason = "data_end"
        else:
            end_exclusive = int(timestamps[breach_index]) + M15_SECONDS
            end_reason = "mitigated"
        windows.append(
            ZoneWindow(
                zone_id=f"{OSS_DETECTOR}:{side}:{index}:{event.confirmation_index}",
                source=OSS_DETECTOR,
                side=side,
                active_from_ts_utc=active_from,
                end_exclusive_ts_utc=end_exclusive,
                lower=lower,
                upper=upper,
                end_reason=end_reason,
            )
        )
    return tuple(windows)


def build_oss_liquidity_records(
    m15: pd.DataFrame,
    liquidity_output: pd.DataFrame,
) -> tuple[OssLiquidityRecord, ...]:
    """Retain library-native liquidity labels as an audit, not trade signals."""

    required = {"Liquidity", "Level", "End", "Swept"}
    missing = required.difference(liquidity_output.columns)
    if missing:
        raise ValueError(f"OSS liquidity output is missing columns: {sorted(missing)}")
    if len(m15) != len(liquidity_output):
        raise ValueError("OSS liquidity input and output lengths differ")
    timestamps = m15["ts_utc"].to_numpy(dtype=np.int64)
    records: list[OssLiquidityRecord] = []
    for source_index, row in liquidity_output[
        liquidity_output["Liquidity"].notna()
    ].iterrows():
        index = int(source_index)
        source_ts = int(timestamps[index])
        if not _inside_article_window(source_ts):
            continue
        direction = int(row["Liquidity"])
        if direction not in {-1, 1}:
            raise ValueError("OSS liquidity output contains an unexpected direction")
        group_end_index = int(row["End"])
        swept_index = int(row["Swept"])
        if not 0 <= group_end_index < len(timestamps):
            raise ValueError("OSS liquidity End index is outside the M15 input")
        if swept_index and not 0 <= swept_index < len(timestamps):
            raise ValueError("OSS liquidity Swept index is outside the M15 input")
        side = "long" if direction == 1 else "short"
        records.append(
            OssLiquidityRecord(
                liquidity_id=f"smartmoneyconcepts_liquidity:{side}:{index}",
                side=side,
                source_ts_utc=source_ts,
                level=float(row["Level"]),
                group_end_ts_utc=int(timestamps[group_end_index]),
                swept_ts_utc=(
                    None if swept_index == 0 else int(timestamps[swept_index])
                ),
            )
        )
    return tuple(records)


def _assert_oss_defaults(smc: Any) -> None:
    expected = {
        "swing_highs_lows": ("swing_length", 50),
        "ob": ("close_mitigation", False),
        "liquidity": ("range_percent", 0.01),
    }
    for function_name, (parameter_name, expected_default) in expected.items():
        signature = inspect.signature(getattr(smc, function_name))
        actual = signature.parameters[parameter_name].default
        if actual != expected_default:
            raise RuntimeError(
                f"unexpected {function_name} default for {parameter_name}: {actual!r}"
            )


def run_oss_default_analysis(m15: pd.DataFrame) -> OssAnalysis:
    """Call pinned OB and Liquidity functions without overriding defaults."""

    version = importlib.metadata.version(OSS_PACKAGE)
    if version != OSS_PACKAGE_VERSION:
        raise RuntimeError(
            f"expected {OSS_PACKAGE}=={OSS_PACKAGE_VERSION}, found {version}"
        )
    with contextlib.redirect_stdout(io.StringIO()):
        package = importlib.import_module(OSS_PACKAGE)
    smc = package.smc
    _assert_oss_defaults(smc)
    ohlcv = _oss_ohlcv(m15)
    swings = smc.swing_highs_lows(ohlcv)
    ob_output = smc.ob(ohlcv, swings)
    liquidity_output = smc.liquidity(ohlcv, swings)
    zones = build_oss_zone_windows(m15, ohlcv, swings, ob_output)
    liquidity = build_oss_liquidity_records(m15, liquidity_output)
    raw_ob_count = int(ob_output["OB"].notna().sum())
    raw_liquidity_count = int(liquidity_output["Liquidity"].notna().sum())
    summary = OssSummary(
        package=OSS_PACKAGE,
        version=version,
        raw_ob_count=raw_ob_count,
        comparison_ob_count=len(zones),
        bullish_ob_count=sum(zone.side == "long" for zone in zones),
        bearish_ob_count=sum(zone.side == "short" for zone in zones),
        mitigated_ob_count=sum(zone.end_reason == "mitigated" for zone in zones),
        active_at_data_end_count=sum(zone.end_reason == "data_end" for zone in zones),
        raw_liquidity_count=raw_liquidity_count,
        comparison_liquidity_count=len(liquidity),
        bullish_liquidity_count=sum(item.side == "long" for item in liquidity),
        bearish_liquidity_count=sum(item.side == "short" for item in liquidity),
    )
    zone_hash = deterministic_sha256("ict_ob_stage7:oss_zones", zones)
    liquidity_hash = deterministic_sha256(
        "ict_ob_stage7:oss_liquidity",
        liquidity,
    )
    return OssAnalysis(
        zones=zones,
        liquidity=liquidity,
        summary=summary,
        zone_sha256=zone_hash,
        liquidity_sha256=liquidity_hash,
        result_sha256=deterministic_sha256(
            "ict_ob_stage7:oss_result",
            [asdict(summary), zone_hash, liquidity_hash],
        ),
    )


def run_ict_ob_comparison_from_timeframes(
    bundle: TimeframeBundle,
    *,
    input_audit: M5Audit,
) -> IctObComparisonResult:
    """Run stage 6 and the OSS comparison over one shared timeframe bundle."""

    experiment = run_ict_ob_from_timeframes(bundle, input_audit=input_audit)
    official_windows = lifecycle_windows(
        experiment.official.backtest.zone_lifecycles
    )
    secondary_windows = lifecycle_windows(
        experiment.secondary.backtest.zone_lifecycles
    )
    if len(official_windows) != experiment.official.summary.zone_count:
        raise RuntimeError("official lifecycle count differs from detector zone count")
    if len(secondary_windows) != experiment.secondary.summary.zone_count:
        raise RuntimeError("secondary lifecycle count differs from detector zone count")
    oss = run_oss_default_analysis(bundle.m15)
    official_secondary = compare_zone_windows(
        official_windows,
        secondary_windows,
        left_source=experiment.official.detector,
        right_source=experiment.secondary.detector,
    )
    official_oss = compare_zone_windows(
        official_windows,
        oss.zones,
        left_source=experiment.official.detector,
        right_source=OSS_DETECTOR,
    )
    secondary_oss = compare_zone_windows(
        secondary_windows,
        oss.zones,
        left_source=experiment.secondary.detector,
        right_source=OSS_DETECTOR,
    )
    official_summary = experiment.official.summary
    secondary_summary = experiment.secondary.summary
    performance_delta: dict[str, float | int | None] = {
        "zone_count": official_summary.zone_count - secondary_summary.zone_count,
        "trade_count": official_summary.trade_count - secondary_summary.trade_count,
        "return_pct_points": (
            official_summary.return_pct - secondary_summary.return_pct
        ),
        "win_rate_pct_points": (
            None
            if official_summary.win_rate_pct is None
            or secondary_summary.win_rate_pct is None
            else official_summary.win_rate_pct - secondary_summary.win_rate_pct
        ),
        "max_drawdown_pct_points": (
            official_summary.max_drawdown_pct
            - secondary_summary.max_drawdown_pct
        ),
        "positive_segments": (
            official_summary.positive_segments - secondary_summary.positive_segments
        ),
    }
    official_lifecycle_hash = deterministic_sha256(
        "ict_ob_stage7:official_lifecycles",
        official_windows,
    )
    secondary_lifecycle_hash = deterministic_sha256(
        "ict_ob_stage7:secondary_lifecycles",
        secondary_windows,
    )
    result_hash = deterministic_sha256(
        "ict_ob_stage7:comparison",
        [
            experiment.result_sha256,
            official_lifecycle_hash,
            secondary_lifecycle_hash,
            oss.result_sha256,
            asdict(official_secondary),
            asdict(official_oss),
            asdict(secondary_oss),
            performance_delta,
        ],
    )
    return IctObComparisonResult(
        experiment=experiment,
        official_windows=official_windows,
        secondary_windows=secondary_windows,
        oss=oss,
        official_secondary=official_secondary,
        official_oss=official_oss,
        secondary_oss=secondary_oss,
        performance_delta_official_minus_secondary=performance_delta,
        official_lifecycle_sha256=official_lifecycle_hash,
        secondary_lifecycle_sha256=secondary_lifecycle_hash,
        result_sha256=result_hash,
    )


def run_ict_ob_comparison_from_db(db_path: Path) -> IctObComparisonResult:
    """Read the fixed SQLite once and apply all three definitions to its M15."""

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
    bundle = prepare_timeframes(m5, start_inclusive=start, end_exclusive=end)
    result = run_ict_ob_comparison_from_timeframes(bundle, input_audit=audit)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise FrozenInputError(
            "upstream SQLite changed during the read-only stage-7 comparison"
        )
    return result


def assert_comparison_reproducible(
    first: IctObComparisonResult,
    second: IctObComparisonResult,
) -> None:
    """Raise unless both stage-6 results and all stage-7 comparisons match."""

    assert_reproducible(first.experiment, second.experiment)
    if first.result_sha256 != second.result_sha256:
        raise FrozenInputError(
            "stage-7 repeated runs produced different lifecycle or OSS comparisons"
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_comparison_outputs(
    result: IctObComparisonResult,
    output_dir: Path,
    *,
    reproducibility_runs: int = 1,
) -> None:
    """Write private zone windows and a row-free stage-7 summary."""

    if reproducibility_runs <= 0:
        raise ValueError("reproducibility_runs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "official_zone_lifecycles_private.csv": pd.DataFrame(
            [asdict(item) for item in result.official_windows],
            columns=ZONE_WINDOW_COLUMNS,
        ),
        "secondary_zone_lifecycles_private.csv": pd.DataFrame(
            [asdict(item) for item in result.secondary_windows],
            columns=ZONE_WINDOW_COLUMNS,
        ),
        "oss_ob_zone_lifecycles_private.csv": pd.DataFrame(
            [asdict(item) for item in result.oss.zones],
            columns=ZONE_WINDOW_COLUMNS,
        ),
        "oss_liquidity_private.csv": pd.DataFrame(
            [asdict(item) for item in result.oss.liquidity],
            columns=LIQUIDITY_COLUMNS,
        ),
    }
    artifact_paths: list[Path] = []
    for filename, frame in frames.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        artifact_paths.append(path)
    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "reproducibility_runs": reproducibility_runs,
        "article_window_et": {
            "start_inclusive": ARTICLE_WINDOW_START,
            "end_exclusive": ARTICLE_WINDOW_END_EXCLUSIVE,
        },
        "comparison_contract": {
            "price_interval": "closed",
            "active_time_interval": "half_open",
            "event_bar_end": "bar_start_plus_900_seconds",
            "direction_required_for_overlap": False,
            "fill_required_for_population": False,
            "oss_parameters": "all_defaults",
            "oss_execution_or_pnl": False,
        },
        "input_extraction_sha256": result.experiment.input_audit.extraction_sha256,
        "stage6_result_sha256": result.experiment.result_sha256,
        "performance": {
            "official": asdict(result.experiment.official.summary),
            "secondary": asdict(result.experiment.secondary.summary),
            "official_minus_secondary": (
                result.performance_delta_official_minus_secondary
            ),
        },
        "oss": {
            "summary": asdict(result.oss.summary),
            "hashes": {
                "zones_sha256": result.oss.zone_sha256,
                "liquidity_sha256": result.oss.liquidity_sha256,
                "result_sha256": result.oss.result_sha256,
            },
        },
        "overlap": {
            "official_secondary": asdict(result.official_secondary),
            "official_oss": asdict(result.official_oss),
            "secondary_oss": asdict(result.secondary_oss),
        },
        "figure_candidates": [
            "three-definition zone counts and any-overlap rates",
            "official versus secondary return and maximum drawdown",
            "official versus secondary fixed-five-segment realized PnL",
            "detector and execution funnel counts",
        ],
        "hashes": {
            "official_lifecycles_sha256": result.official_lifecycle_sha256,
            "secondary_lifecycles_sha256": result.secondary_lifecycle_sha256,
            "result_sha256": result.result_sha256,
            "private_artifact_sha256": {
                path.name: _file_sha256(path) for path in artifact_paths
            },
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
