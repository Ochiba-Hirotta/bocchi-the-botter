"""Row-free Stage-8 evidence manifest for the ICT Order Block experiment."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .ict_ob import INITIAL_CASH, NEW_YORK
from .ict_ob_comparison import OSS_DETECTOR, IctObComparisonResult
from .ict_ob_experiment import (
    MINIMUM_SAMPLE_SIZE,
    DetectorRunResult,
    deterministic_sha256,
)
from .minute_data import SOURCE
from .orb_m15 import (
    ARTICLE_WINDOW_END_EXCLUSIVE,
    ARTICLE_WINDOW_START,
    INPUT_END_EXCLUSIVE_UTC,
    INPUT_START_UTC,
    fixed_segment_edges,
)


REFERENCE_SCHEMA_VERSION = 1
REQUIRED_REPRODUCIBILITY_RUNS = 2
MANIFEST_FILENAME = "manifest.json"
MANIFEST_HASH_FILENAME = "manifest.sha256"


class IctObEvidenceError(ValueError):
    """Raised when a public ICT OB evidence pack is incomplete or inconsistent."""


def stage8_code_paths(repo_root: Path, chapter_dir: Path) -> dict[str, Path]:
    """Return the exact implementation files pinned by the Stage-8 manifest."""

    season2 = repo_root / "src" / "bocchi_the_botter_repro" / "season2"
    return {
        "pyproject.toml": repo_root / "pyproject.toml",
        "chapters/season2/ch04_ict_order_blocks/run.py": chapter_dir / "run.py",
        "chapters/season2/ch04_ict_order_blocks/compare.py": chapter_dir
        / "compare.py",
        "chapters/season2/ch04_ict_order_blocks/build_manifest.py": chapter_dir
        / "build_manifest.py",
        "chapters/season2/ch04_ict_order_blocks/verify.py": chapter_dir
        / "verify.py",
        "src/bocchi_the_botter_repro/season2/minute_data.py": season2
        / "minute_data.py",
        "src/bocchi_the_botter_repro/season2/orb_m15.py": season2 / "orb_m15.py",
        "src/bocchi_the_botter_repro/season2/ict_ob.py": season2 / "ict_ob.py",
        "src/bocchi_the_botter_repro/season2/ict_ob_experiment.py": season2
        / "ict_ob_experiment.py",
        "src/bocchi_the_botter_repro/season2/ict_ob_comparison.py": season2
        / "ict_ob_comparison.py",
        "src/bocchi_the_botter_repro/season2/ict_ob_evidence.py": season2
        / "ict_ob_evidence.py",
    }


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_text(timestamp: int) -> str:
    value = dt.datetime.fromtimestamp(timestamp, tz=dt.UTC)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise IctObEvidenceError(f"{label} is not a lowercase SHA-256")


def _assert_safe_command(command: str) -> None:
    if not command.strip():
        raise IctObEvidenceError("reproduction command must not be empty")
    if "OANDA_API_TOKEN" in command or "Bearer " in command:
        raise IctObEvidenceError("reproduction command must not mention credentials")
    if re.search(r"(?:^|[\s\"'=])/(?!/)", command):
        raise IctObEvidenceError("reproduction command must not contain an absolute path")


def _lifecycle_summary(result: DetectorRunResult) -> dict[str, Any]:
    lifecycles = result.backtest.zone_lifecycles
    return {
        "total": len(lifecycles),
        "by_side": dict(sorted(Counter(item.side for item in lifecycles).items())),
        "by_end_reason": dict(
            sorted(Counter(item.end_reason for item in lifecycles).items())
        ),
        "zero_duration_count": sum(
            item.active_from_ts_utc == item.end_exclusive_ts_utc
            for item in lifecycles
        ),
    }


def _detector_manifest(
    result: DetectorRunResult,
    *,
    lifecycle_sha256: str,
    translation: str,
    target: str,
) -> dict[str, Any]:
    return {
        "translation": translation,
        "target": target,
        "summary": asdict(result.summary),
        "segments": result.segments.to_dict(orient="records"),
        "detection_counters": result.detection_counters,
        "execution_counters": result.execution_counters,
        "exit_reasons": result.exit_reasons,
        "lifecycle": _lifecycle_summary(result),
        "hashes": {
            "zones_sha256": result.zone_sha256,
            "trades_sha256": result.trade_sha256,
            "terminal_sha256": result.terminal_sha256,
            "detector_result_sha256": result.result_sha256,
            "lifecycles_sha256": lifecycle_sha256,
        },
    }


def build_evidence_manifest(
    result: IctObComparisonResult,
    *,
    db_basename: str,
    code_commit: str,
    generated_at_utc: str,
    command: str,
    verify_command: str,
    code_paths: Mapping[str, Path],
    reproducibility_runs: int = REQUIRED_REPRODUCIBILITY_RUNS,
) -> dict[str, Any]:
    """Build the article-time manifest without market, zone, or trade rows."""

    if reproducibility_runs != REQUIRED_REPRODUCIBILITY_RUNS:
        raise IctObEvidenceError("Stage 8 requires two complete reproducibility runs")
    if not db_basename or Path(db_basename).name != db_basename:
        raise IctObEvidenceError("db_basename must not contain a path")
    if not code_commit or "/" in code_commit or "\\" in code_commit:
        raise IctObEvidenceError("code_commit must be a commit id, optionally -dirty")
    try:
        generated = dt.datetime.strptime(generated_at_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise IctObEvidenceError("generated_at_utc must be UTC second precision") from exc
    if generated.tzinfo is not None:
        raise IctObEvidenceError("generated_at_utc must use the literal Z suffix")
    _assert_safe_command(command)
    _assert_safe_command(verify_command)
    if not code_paths:
        raise IctObEvidenceError("at least one code path must be pinned")
    missing_code = [label for label, path in code_paths.items() if not path.is_file()]
    if missing_code:
        raise IctObEvidenceError(f"code paths are missing: {sorted(missing_code)}")

    experiment = result.experiment
    audit = experiment.input_audit
    official = _detector_manifest(
        experiment.official,
        lifecycle_sha256=result.official_lifecycle_sha256,
        translation="ICT Mentorship 2022 Month 04 frozen mechanical translation",
        target="latest confirmed eligible daily external-liquidity swing",
    )
    secondary = _detector_manifest(
        experiment.secondary,
        lifecycle_sha256=result.secondary_lifecycle_sha256,
        translation="17-minute sweep -> MSS -> displacement/FVG frozen translation",
        target="latest confirmed eligible M15 opposing swing",
    )
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "smartmoneyconcepts")
    }
    return _json_ready(
        {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "generated_at_utc": generated_at_utc,
            "code_commit": code_commit,
            "environment": {
                "python": platform.python_version(),
                "dependencies": dependencies,
            },
            "reproducibility_runs": reproducibility_runs,
            "article_window_et": {
                "start_inclusive": ARTICLE_WINDOW_START,
                "end_exclusive": ARTICLE_WINDOW_END_EXCLUSIVE,
                "segment_count": 5,
            },
            "input": {
                "source": SOURCE,
                "source_environment": "practice",
                "instrument": "USD_JPY",
                "granularity": "M5_to_complete_M15",
                "price": "BA",
                "db": {
                    "role": "upstream_read_only_source",
                    "basename": db_basename,
                    "binary_hash_policy": (
                        "not pinned here because the append-only upstream file may grow; "
                        "the fixed projection is identified by extraction_sha256"
                    ),
                },
                "range_utc": {
                    "start_inclusive": INPUT_START_UTC,
                    "end_exclusive": INPUT_END_EXCLUSIVE_UTC,
                },
                "m5": {
                    "row_count": audit.row_count,
                    "first_ts_utc": _utc_text(audit.first_ts_utc),
                    "last_ts_utc": _utc_text(audit.last_ts_utc),
                    "duplicate_count": audit.duplicate_count,
                    "off_boundary_count": audit.off_boundary_count,
                    "null_required_count": audit.null_required_count,
                    "invalid_volume_count": audit.invalid_volume_count,
                    "invalid_ohlc_count": audit.invalid_ohlc_count,
                    "negative_spread_count": audit.negative_spread_count,
                    "sorted_ascending": audit.sorted_ascending,
                    "gap_count": audit.gap_count,
                    "missing_m5_slots": audit.missing_m5_slots,
                    "gap_classification": {
                        "weekend_closure_candidate": audit.weekend_gap_count,
                        "long_non_weekend_closure_candidate": (
                            audit.long_non_weekend_gap_count
                        ),
                        "short_unclassified": audit.short_gap_count,
                    },
                    "extraction_sha256": audit.extraction_sha256,
                },
                "derived": {
                    "complete_m15_count": experiment.complete_m15_count,
                    "incomplete_m15_count": experiment.incomplete_m15_count,
                    "accepted_daily_count": experiment.accepted_daily_count,
                    "rejected_daily_count": experiment.rejected_daily_count,
                    "daily_swing_count": experiment.daily_swing_count,
                    "m15_swing_count": experiment.m15_swing_count,
                    "bias_at_open_counts": experiment.bias_at_open_counts,
                    "bias_at_close_counts": experiment.bias_at_close_counts,
                },
            },
            "shared_execution": {
                "signal_prices": "bid OHLC",
                "long": "entry ask; exits bid",
                "short": "entry bid; exits ask",
                "fixed_spread_addition": 0,
                "commission": 0,
                "ordinary_slippage": 0,
                "swap": 0,
                "same_bar_priority": "SL",
                "forced_exit": None,
                "position_risk_fraction": 0.01,
                "minimum_sample_size": MINIMUM_SAMPLE_SIZE,
                "criterion": (
                    "sample >= 30 closed trades, return_pct > 0, "
                    "and positive_segments >= 3 of 5"
                ),
            },
            "detectors": {
                "official": official,
                "secondary": secondary,
            },
            "comparison": {
                "contract": {
                    "price_interval": "closed",
                    "active_time_interval": "half_open",
                    "event_bar_end": "bar_start_plus_900_seconds",
                    "direction_required_for_overlap": False,
                    "fill_required_for_population": False,
                    "zero_duration_overlap": False,
                },
                "overlap": {
                    "official_secondary": asdict(result.official_secondary),
                    "official_oss": asdict(result.official_oss),
                    "secondary_oss": asdict(result.secondary_oss),
                },
                "performance_delta_official_minus_secondary": (
                    result.performance_delta_official_minus_secondary
                ),
            },
            "oss": {
                "parameters": {
                    "swing_highs_lows.swing_length": 50,
                    "ob.close_mitigation": False,
                    "liquidity.range_percent": 0.01,
                    "all_arguments": "package defaults",
                },
                "summary": asdict(result.oss.summary),
                "hashes": {
                    "zones_sha256": result.oss.zone_sha256,
                    "liquidity_sha256": result.oss.liquidity_sha256,
                    "oss_result_sha256": result.oss.result_sha256,
                },
                "execution_or_pnl": False,
                "population_note": (
                    "OB rows are the package's final surviving output, not all "
                    "historical detection events. Default centered swings are "
                    "retrospective labels."
                ),
            },
            "hashes": {
                "stage6_result_sha256": experiment.result_sha256,
                "stage7_result_sha256": result.result_sha256,
                "code_sha256": {
                    label: _file_sha256(path)
                    for label, path in sorted(code_paths.items())
                },
            },
            "public_boundary": {
                "included": "aggregate counts, metrics, contracts, versions, hashes",
                "excluded": (
                    "market rows, zone rows, liquidity rows, trade rows, individual "
                    "prices, individual timestamps, SQLite, credentials, absolute paths"
                ),
                "full_recalculation_requires": (
                    "a private read-only SQLite whose fixed M5 projection matches "
                    "input.m5.extraction_sha256"
                ),
                "claims": (
                    "Both frozen translations failed the predeclared criterion. "
                    "No detector superiority or live-trading claim is made."
                ),
            },
            "commands": {
                "rebuild": command,
                "verify_row_free": verify_command,
            },
        }
    )


def write_evidence_pack(reference_dir: Path, manifest: Mapping[str, Any]) -> str:
    """Write the UTF-8 manifest and a detached SHA-256 checksum."""

    reference_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reference_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(_json_ready(dict(manifest)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = _file_sha256(manifest_path)
    (reference_dir / MANIFEST_HASH_FILENAME).write_text(
        f"{digest}  {MANIFEST_FILENAME}\n",
        encoding="ascii",
    )
    return digest


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
        raise IctObEvidenceError(f"{label} disagrees")


def _verify_detector(label: str, payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    segments = payload["segments"]
    lifecycle = payload["lifecycle"]
    trade_count = int(summary["trade_count"])
    zone_count = int(summary["zone_count"])
    if int(summary["long_count"]) + int(summary["short_count"]) != trade_count:
        raise IctObEvidenceError(f"{label} trade direction counts disagree")
    if sum(int(value) for value in payload["exit_reasons"].values()) != trade_count:
        raise IctObEvidenceError(f"{label} exit reason counts disagree")
    if len(segments) != 5:
        raise IctObEvidenceError(f"{label} must contain five fixed segments")
    edges = fixed_segment_edges()
    if [row["start"] for row in segments] != [value.isoformat() for value in edges[:-1]]:
        raise IctObEvidenceError(f"{label} segment starts disagree")
    if [row["end_exclusive"] for row in segments] != [
        value.isoformat() for value in edges[1:]
    ]:
        raise IctObEvidenceError(f"{label} segment ends disagree")
    if sum(int(row["trade_count"]) for row in segments) != trade_count:
        raise IctObEvidenceError(f"{label} segment trade counts disagree")
    if sum(float(row["pnl_jpy"]) > 0 for row in segments) != int(
        summary["positive_segments"]
    ):
        raise IctObEvidenceError(f"{label} positive segment count disagrees")
    equity = INITIAL_CASH
    for row in segments:
        _assert_close(
            float(row["return_pct"]),
            float(row["pnl_jpy"]) / equity * 100.0,
            label=f"{label} segment return",
        )
        equity += float(row["pnl_jpy"])
    _assert_close(equity, float(summary["final_equity"]), label=f"{label} equity")
    _assert_close(
        (equity / INITIAL_CASH - 1.0) * 100.0,
        float(summary["return_pct"]),
        label=f"{label} return",
    )
    sample_sufficient = trade_count >= MINIMUM_SAMPLE_SIZE
    if bool(summary["sample_sufficient"]) != sample_sufficient:
        raise IctObEvidenceError(f"{label} sample sufficiency disagrees")
    expected_criterion = (
        "insufficient_sample"
        if not sample_sufficient
        else "passed"
        if float(summary["return_pct"]) > 0
        and int(summary["positive_segments"]) >= 3
        else "failed"
    )
    if summary["criterion"] != expected_criterion:
        raise IctObEvidenceError(f"{label} criterion disagrees")
    if int(lifecycle["total"]) != zone_count:
        raise IctObEvidenceError(f"{label} lifecycle total disagrees")
    if sum(int(value) for value in lifecycle["by_side"].values()) != zone_count:
        raise IctObEvidenceError(f"{label} lifecycle side counts disagree")
    if sum(int(value) for value in lifecycle["by_end_reason"].values()) != zone_count:
        raise IctObEvidenceError(f"{label} lifecycle end counts disagree")
    if int(payload["detection_counters"]["zone_detected"]) != zone_count:
        raise IctObEvidenceError(f"{label} detector zone count disagrees")
    execution = payload["execution_counters"]
    if int(execution["zone_activated"]) != zone_count:
        raise IctObEvidenceError(f"{label} activated zone count disagrees")
    if int(execution["filled"]) != trade_count or int(
        execution["trade_closed"]
    ) != trade_count:
        raise IctObEvidenceError(f"{label} execution trade counts disagree")
    for hash_label, value in payload["hashes"].items():
        _assert_sha256(value, label=f"{label}.{hash_label}")
    hashes = payload["hashes"]
    expected_result_sha256 = deterministic_sha256(
        f"{summary['detector']}:result",
        [
            summary,
            payload["detection_counters"],
            payload["execution_counters"],
            payload["exit_reasons"],
            segments,
            hashes["zones_sha256"],
            hashes["trades_sha256"],
            hashes["terminal_sha256"],
        ],
    )
    if hashes["detector_result_sha256"] != expected_result_sha256:
        raise IctObEvidenceError(f"{label} detector result hash disagrees")
    return expected_result_sha256


def _verify_overlap(payload: Mapping[str, Any]) -> None:
    left_total = int(payload["left_total"])
    right_total = int(payload["right_total"])
    left_count = int(payload["left_overlapped"])
    right_count = int(payload["right_overlapped"])
    pairs = int(payload["overlapping_pair_count"])
    if not 0 <= left_count <= left_total or not 0 <= right_count <= right_total:
        raise IctObEvidenceError("overlap count exceeds its population")
    expected_left = None if left_total == 0 else left_count / left_total * 100.0
    expected_right = None if right_total == 0 else right_count / right_total * 100.0
    for side, actual, expected in (
        ("left", payload["left_overlap_pct"], expected_left),
        ("right", payload["right_overlap_pct"], expected_right),
    ):
        if expected is None:
            if actual is not None:
                raise IctObEvidenceError(f"{side} empty overlap rate must be null")
        else:
            _assert_close(float(actual), expected, label=f"{side} overlap rate")
    if pairs < max(left_count, right_count) or pairs > left_total * right_total:
        raise IctObEvidenceError("overlap pair count is inconsistent")


def verify_evidence_pack(
    reference_dir: Path,
    *,
    code_paths: Mapping[str, Path] | None = None,
    verify_environment: bool = True,
) -> dict[str, Any]:
    """Verify the detached hash, row-free boundary, and aggregate invariants."""

    manifest_path = reference_dir / MANIFEST_FILENAME
    checksum_path = reference_dir / MANIFEST_HASH_FILENAME
    missing = [path.name for path in (manifest_path, checksum_path) if not path.is_file()]
    if missing:
        raise IctObEvidenceError(f"evidence files are missing: {missing}")
    checksum_text = checksum_path.read_text(encoding="ascii")
    match = re.fullmatch(r"([0-9a-f]{64})  manifest\.json\n", checksum_text)
    if match is None or _file_sha256(manifest_path) != match.group(1):
        raise IctObEvidenceError("manifest hash mismatch")
    raw_text = manifest_path.read_text(encoding="utf-8")
    if re.search(r"(?:^|[\s\"'=])/(?!/)", raw_text):
        raise IctObEvidenceError("absolute path leaked into the manifest")
    row_artifacts = [
        path.name
        for path in reference_dir.iterdir()
        if path.suffix.lower() in {".csv", ".parquet", ".sqlite", ".db"}
    ]
    if row_artifacts:
        raise IctObEvidenceError(f"row-bearing public artifacts found: {row_artifacts}")
    forbidden = (
        "/Users/",
        "entry_price",
        "exit_price",
        "entry_time_utc",
        "exit_time_utc",
        "active_from_ts_utc",
        "zone_id",
        "liquidity_id",
        "pending_zone",
        "private_artifact_sha256",
    )
    leaked = [name for name in forbidden if name in raw_text]
    if leaked:
        raise IctObEvidenceError(f"private row field leaked into manifest: {leaked}")
    payload = json.loads(raw_text)
    if int(payload.get("schema_version", -1)) != REFERENCE_SCHEMA_VERSION:
        raise IctObEvidenceError("unsupported ICT OB manifest schema")
    try:
        dt.datetime.strptime(payload["generated_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, TypeError, ValueError) as exc:
        raise IctObEvidenceError("manifest generation timestamp is invalid") from exc
    if int(payload["reproducibility_runs"]) != REQUIRED_REPRODUCIBILITY_RUNS:
        raise IctObEvidenceError("manifest was not produced by two complete runs")
    for command in payload["commands"].values():
        _assert_safe_command(str(command))
    if verify_environment:
        if payload["environment"]["python"] != platform.python_version():
            raise IctObEvidenceError("Python version differs from the manifest")
        for name, expected in payload["environment"]["dependencies"].items():
            if importlib.metadata.version(name) != expected:
                raise IctObEvidenceError(f"dependency version differs: {name}")
    m5 = payload["input"]["m5"]
    db_basename = payload["input"]["db"]["basename"]
    if not isinstance(db_basename, str) or Path(db_basename).name != db_basename:
        raise IctObEvidenceError("database basename contains a path")
    _assert_sha256(m5["extraction_sha256"], label="input extraction")
    gap_classes = m5["gap_classification"]
    if sum(int(value) for value in gap_classes.values()) != int(m5["gap_count"]):
        raise IctObEvidenceError("M5 gap classes do not cover all gaps")
    derived = payload["input"]["derived"]
    complete_m15 = int(derived["complete_m15_count"])
    if sum(int(value) for value in derived["bias_at_open_counts"].values()) != complete_m15:
        raise IctObEvidenceError("bias-at-open counts do not cover complete M15")
    if sum(int(value) for value in derived["bias_at_close_counts"].values()) != complete_m15:
        raise IctObEvidenceError("bias-at-close counts do not cover complete M15")
    detector_result_hashes = {
        label: _verify_detector(label, payload["detectors"][label])
        for label in ("official", "secondary")
    }
    for label in ("stage6_result_sha256", "stage7_result_sha256"):
        _assert_sha256(payload["hashes"][label], label=label)
    expected_stage6_sha256 = deterministic_sha256(
        "ict_ob_stage6_experiment",
        [
            {
                "input_extraction_sha256": m5["extraction_sha256"],
                "m5_rows": m5["row_count"],
                "complete_m15_count": derived["complete_m15_count"],
                "incomplete_m15_count": derived["incomplete_m15_count"],
                "accepted_daily_count": derived["accepted_daily_count"],
                "rejected_daily_count": derived["rejected_daily_count"],
                "daily_swing_count": derived["daily_swing_count"],
                "m15_swing_count": derived["m15_swing_count"],
                "bias_at_open_counts": derived["bias_at_open_counts"],
                "bias_at_close_counts": derived["bias_at_close_counts"],
                "official_result_sha256": detector_result_hashes["official"],
                "secondary_result_sha256": detector_result_hashes["secondary"],
            }
        ],
    )
    if payload["hashes"]["stage6_result_sha256"] != expected_stage6_sha256:
        raise IctObEvidenceError("Stage 6 result hash disagrees")

    overlaps = payload["comparison"]["overlap"]
    for overlap in overlaps.values():
        _verify_overlap(overlap)
    official_zones = int(payload["detectors"]["official"]["summary"]["zone_count"])
    secondary_zones = int(payload["detectors"]["secondary"]["summary"]["zone_count"])
    oss_zones = int(payload["oss"]["summary"]["comparison_ob_count"])
    expected_populations = {
        "official_secondary": (official_zones, secondary_zones),
        "official_oss": (official_zones, oss_zones),
        "secondary_oss": (secondary_zones, oss_zones),
    }
    for name, (left, right) in expected_populations.items():
        if (
            int(overlaps[name]["left_total"]),
            int(overlaps[name]["right_total"]),
        ) != (left, right):
            raise IctObEvidenceError(f"overlap population disagrees: {name}")
    official_name = payload["detectors"]["official"]["summary"]["detector"]
    secondary_name = payload["detectors"]["secondary"]["summary"]["detector"]
    oss_name = OSS_DETECTOR
    expected_sources = {
        "official_secondary": (official_name, secondary_name),
        "official_oss": (official_name, oss_name),
        "secondary_oss": (secondary_name, oss_name),
    }
    for name, (left_source, right_source) in expected_sources.items():
        if (
            overlaps[name]["left_source"],
            overlaps[name]["right_source"],
        ) != (left_source, right_source):
            raise IctObEvidenceError(f"overlap sources disagree: {name}")

    delta = payload["comparison"]["performance_delta_official_minus_secondary"]
    official_summary = payload["detectors"]["official"]["summary"]
    secondary_summary = payload["detectors"]["secondary"]["summary"]
    exact_deltas = {
        "zone_count": int(official_summary["zone_count"])
        - int(secondary_summary["zone_count"]),
        "trade_count": int(official_summary["trade_count"])
        - int(secondary_summary["trade_count"]),
        "positive_segments": int(official_summary["positive_segments"])
        - int(secondary_summary["positive_segments"]),
    }
    for label, expected in exact_deltas.items():
        if int(delta[label]) != expected:
            raise IctObEvidenceError(f"performance delta disagrees: {label}")
    float_deltas = {
        "return_pct_points": float(official_summary["return_pct"])
        - float(secondary_summary["return_pct"]),
        "max_drawdown_pct_points": float(official_summary["max_drawdown_pct"])
        - float(secondary_summary["max_drawdown_pct"]),
        "win_rate_pct_points": float(official_summary["win_rate_pct"])
        - float(secondary_summary["win_rate_pct"]),
    }
    for label, expected in float_deltas.items():
        _assert_close(float(delta[label]), expected, label=f"performance {label}")

    oss_payload = payload["oss"]
    oss = oss_payload["summary"]
    if int(oss["raw_ob_count"]) < oss_zones:
        raise IctObEvidenceError("OSS comparison OB count exceeds raw output")
    if int(oss["raw_liquidity_count"]) < int(oss["comparison_liquidity_count"]):
        raise IctObEvidenceError("OSS comparison liquidity count exceeds raw output")
    if int(oss["bullish_ob_count"]) + int(oss["bearish_ob_count"]) != oss_zones:
        raise IctObEvidenceError("OSS OB direction counts disagree")
    if int(oss["mitigated_ob_count"]) + int(oss["active_at_data_end_count"]) != oss_zones:
        raise IctObEvidenceError("OSS OB lifecycle counts disagree")
    if int(oss["bullish_liquidity_count"]) + int(
        oss["bearish_liquidity_count"]
    ) != int(oss["comparison_liquidity_count"]):
        raise IctObEvidenceError("OSS liquidity direction counts disagree")
    oss_hashes = oss_payload["hashes"]
    for label, value in oss_hashes.items():
        _assert_sha256(value, label=f"oss.{label}")
    expected_oss_sha256 = deterministic_sha256(
        "ict_ob_stage7:oss_result",
        [oss, oss_hashes["zones_sha256"], oss_hashes["liquidity_sha256"]],
    )
    if oss_hashes["oss_result_sha256"] != expected_oss_sha256:
        raise IctObEvidenceError("OSS result hash disagrees")
    expected_stage7_sha256 = deterministic_sha256(
        "ict_ob_stage7:comparison",
        [
            expected_stage6_sha256,
            payload["detectors"]["official"]["hashes"]["lifecycles_sha256"],
            payload["detectors"]["secondary"]["hashes"]["lifecycles_sha256"],
            expected_oss_sha256,
            overlaps["official_secondary"],
            overlaps["official_oss"],
            overlaps["secondary_oss"],
            delta,
        ],
    )
    if payload["hashes"]["stage7_result_sha256"] != expected_stage7_sha256:
        raise IctObEvidenceError("Stage 7 result hash disagrees")
    for label, value in payload["hashes"]["code_sha256"].items():
        _assert_sha256(value, label=f"code.{label}")
    if code_paths is not None:
        recorded = payload["hashes"]["code_sha256"]
        if set(recorded) != set(code_paths):
            raise IctObEvidenceError("pinned code path set differs from the manifest")
        for label, path in code_paths.items():
            if not path.is_file() or _file_sha256(path) != recorded[label]:
                raise IctObEvidenceError(f"code hash mismatch: {label}")
    return payload


PRIVATE_TRADE_COLUMNS = (
    "zone_id",
    "detector",
    "side",
    "entry_time_utc",
    "entry_price",
    "stop_loss",
    "take_profit",
    "initial_risk",
    "units",
    "exit_time_utc",
    "exit_price",
    "exit_reason",
    "pnl",
    "equity_before",
    "equity_after",
    "realized_r",
)


def _assert_recomputed_value(
    actual: object,
    expected: object,
    *,
    label: str,
    absolute_tolerance: float,
) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise IctObEvidenceError(f"private recomputation disagrees: {label}")
        return
    if isinstance(actual, (bool, int, str)):
        if actual != expected:
            raise IctObEvidenceError(f"private recomputation disagrees: {label}")
        return
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    ):
        raise IctObEvidenceError(f"private recomputation disagrees: {label}")


def verify_private_detector_trades(
    private_trades_path: Path,
    detector_payload: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Independently recompute one detector's frozen metrics from trade rows."""

    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must not be negative")
    if not private_trades_path.is_file():
        raise IctObEvidenceError(
            f"private trade log is missing: {private_trades_path.name}"
        )
    trades = pd.read_csv(private_trades_path)
    missing = set(PRIVATE_TRADE_COLUMNS).difference(trades.columns)
    if missing:
        raise IctObEvidenceError(
            f"private trade log is missing columns: {sorted(missing)}"
        )
    if trades.empty:
        raise IctObEvidenceError("private trade log is empty")

    summary = detector_payload["summary"]
    detector = str(summary["detector"])
    if set(trades["detector"]) != {detector}:
        raise IctObEvidenceError("private trade detector identity disagrees")
    sides = set(trades["side"])
    if not sides.issubset({"long", "short"}):
        raise IctObEvidenceError(f"unexpected private trade sides: {sorted(sides)}")
    if trades["zone_id"].duplicated().any():
        raise IctObEvidenceError("private trade log reuses a zone_id")

    numeric_columns = (
        "entry_time_utc",
        "entry_price",
        "stop_loss",
        "take_profit",
        "initial_risk",
        "units",
        "exit_time_utc",
        "exit_price",
        "pnl",
        "equity_before",
        "equity_after",
        "realized_r",
    )
    try:
        numeric = trades.loc[:, numeric_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise IctObEvidenceError("private trade numeric field is invalid") from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise IctObEvidenceError("private trade numeric field is non-finite")
    if (numeric["initial_risk"] <= 0).any() or (numeric["units"] <= 0).any():
        raise IctObEvidenceError("private trade risk and units must be positive")
    for column in ("entry_time_utc", "exit_time_utc", "units"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise IctObEvidenceError(f"private trade {column} is not integral")
    if (numeric["exit_time_utc"] < numeric["entry_time_utc"]).any():
        raise IctObEvidenceError("private trade exits before entry")

    expected_per_unit = np.where(
        trades["side"] == "long",
        numeric["exit_price"] - numeric["entry_price"],
        numeric["entry_price"] - numeric["exit_price"],
    )
    expected_pnl = expected_per_unit * numeric["units"].to_numpy(dtype=float)
    if not np.allclose(
        expected_pnl,
        numeric["pnl"],
        rtol=0,
        atol=absolute_tolerance,
    ):
        raise IctObEvidenceError("private trade PnL disagrees with its fills")
    expected_realized_r = expected_per_unit / numeric["initial_risk"].to_numpy(
        dtype=float
    )
    if not np.allclose(
        expected_realized_r,
        numeric["realized_r"],
        rtol=0,
        atol=absolute_tolerance,
    ):
        raise IctObEvidenceError("private realized R disagrees with initial risk")

    equity_after = numeric["equity_after"].to_numpy(dtype=float)
    expected_before = np.concatenate(([INITIAL_CASH], equity_after[:-1]))
    if not np.allclose(
        expected_before,
        numeric["equity_before"],
        rtol=0,
        atol=absolute_tolerance,
    ):
        raise IctObEvidenceError("private trade equity chain is discontinuous")
    if not np.allclose(
        numeric["equity_before"] + numeric["pnl"],
        numeric["equity_after"],
        rtol=0,
        atol=absolute_tolerance,
    ):
        raise IctObEvidenceError("private equity_after disagrees with PnL")

    expected_segments = detector_payload["segments"]
    edges = fixed_segment_edges()
    if len(expected_segments) != len(edges) - 1:
        raise IctObEvidenceError("public detector does not contain five segments")
    entry_dates = (
        pd.to_datetime(
            numeric["entry_time_utc"],
            unit="s",
            utc=True,
            errors="raise",
        )
        .dt.tz_convert(NEW_YORK)
        .dt.date
    )
    segment_pnl: list[float] = []
    segment_start_equity = INITIAL_CASH
    for index, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        expected_segment = expected_segments[index]
        if (
            int(expected_segment["segment"]) != index + 1
            or expected_segment["start"] != lower.isoformat()
            or expected_segment["end_exclusive"] != upper.isoformat()
        ):
            raise IctObEvidenceError("public fixed segment boundary disagrees")
        mask = (entry_dates >= lower) & (entry_dates < upper)
        trade_count = int(mask.sum())
        pnl = float(numeric.loc[mask, "pnl"].sum())
        return_pct = pnl / segment_start_equity * 100.0
        for key, actual in (
            ("trade_count", trade_count),
            ("pnl_jpy", pnl),
            ("return_pct", return_pct),
        ):
            _assert_recomputed_value(
                actual,
                expected_segment[key],
                label=f"{detector}.segment{index + 1}.{key}",
                absolute_tolerance=absolute_tolerance,
            )
        segment_pnl.append(pnl)
        segment_start_equity += pnl
    if int(sum(int(item["trade_count"]) for item in expected_segments)) != len(
        trades
    ):
        raise IctObEvidenceError("private trades fall outside the fixed segments")

    equity = np.concatenate(([INITIAL_CASH], equity_after))
    peaks = np.maximum.accumulate(equity)
    max_drawdown_pct = float(np.min((equity - peaks) / peaks) * 100.0)
    final_equity = float(equity_after[-1])
    return_pct = (final_equity / INITIAL_CASH - 1.0) * 100.0
    pnl_values = numeric["pnl"].to_numpy(dtype=float)
    gains = float(pnl_values[pnl_values > 0].sum())
    losses = float(-pnl_values[pnl_values < 0].sum())
    profit_factor = None if losses == 0 else gains / losses
    positive_segments = sum(value > 0 for value in segment_pnl)
    sample_sufficient = len(trades) >= MINIMUM_SAMPLE_SIZE
    if not sample_sufficient:
        criterion = "insufficient_sample"
    elif return_pct > 0 and positive_segments >= 3:
        criterion = "passed"
    else:
        criterion = "failed"
    recomputed: dict[str, Any] = {
        "trade_count": len(trades),
        "final_equity": final_equity,
        "return_pct": return_pct,
        "win_rate_pct": float((pnl_values > 0).mean() * 100.0),
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": profit_factor,
        "average_realized_r": float(expected_realized_r.mean()),
        "positive_segments": positive_segments,
        "long_count": int((trades["side"] == "long").sum()),
        "short_count": int((trades["side"] == "short").sum()),
        "sample_sufficient": sample_sufficient,
        "criterion": criterion,
    }
    for key, actual in recomputed.items():
        _assert_recomputed_value(
            actual,
            summary[key],
            label=f"{detector}.summary.{key}",
            absolute_tolerance=absolute_tolerance,
        )

    exit_reasons = dict(
        sorted(Counter(str(value) for value in trades["exit_reason"]).items())
    )
    expected_exit_reasons = {
        str(key): int(value) for key, value in detector_payload["exit_reasons"].items()
    }
    if exit_reasons != expected_exit_reasons:
        raise IctObEvidenceError(f"private exit reasons disagree: {detector}")
    return {**recomputed, "exit_reasons": exit_reasons}


def verify_private_trades_against_evidence(
    private_output_dir: Path,
    reference_dir: Path,
    *,
    absolute_tolerance: float = 1e-8,
    evidence_payload: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify both private trade logs against the frozen public manifest."""

    payload = (
        verify_evidence_pack(reference_dir, verify_environment=False)
        if evidence_payload is None
        else evidence_payload
    )
    summaries: dict[str, dict[str, Any]] = {}
    for label in ("official", "secondary"):
        summaries[label] = verify_private_detector_trades(
            private_output_dir / f"{label}_trades_private.csv",
            payload["detectors"][label],
            absolute_tolerance=absolute_tolerance,
        )
    return summaries
