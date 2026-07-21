"""Regression tests for the row-free Stage-8 ICT OB evidence pack."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from bocchi_the_botter_repro.season2.ict_ob_evidence import (
    IctObEvidenceError,
    stage8_code_paths,
    verify_evidence_pack,
    verify_private_detector_trades,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = REPO_ROOT / "chapters" / "season2" / "ch04_ict_order_blocks"
REFERENCE_DIR = REPO_ROOT / "results" / "reference" / "ch04_ict_order_blocks"
STAGE7_SHA256 = "4785466c033356924d9ff04402fe14ebaf2fb2cdca7eecaede10b165ffff6931"


def _rewrite_checksum(reference_dir: Path) -> None:
    manifest = reference_dir / "manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (reference_dir / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n",
        encoding="ascii",
    )


def _private_trade_fixture() -> tuple[pd.DataFrame, dict[str, object]]:
    first_entry = int(pd.Timestamp("2024-01-07T12:00:00Z").timestamp())
    second_entry = int(pd.Timestamp("2024-07-09T12:00:00Z").timestamp())
    trades = pd.DataFrame(
        [
            {
                "zone_id": "fixture:long:1",
                "detector": "fixture_detector",
                "side": "long",
                "entry_time_utc": first_entry,
                "entry_price": 145.0,
                "stop_loss": 144.8,
                "take_profit": 145.2,
                "initial_risk": 0.2,
                "units": 10_000,
                "exit_time_utc": first_entry + 900,
                "exit_price": 145.2,
                "exit_reason": "tp",
                "pnl": 2_000.0,
                "equity_before": 1_000_000.0,
                "equity_after": 1_002_000.0,
                "realized_r": 1.0,
            },
            {
                "zone_id": "fixture:short:2",
                "detector": "fixture_detector",
                "side": "short",
                "entry_time_utc": second_entry,
                "entry_price": 145.0,
                "stop_loss": 145.1,
                "take_profit": 144.8,
                "initial_risk": 0.1,
                "units": 10_000,
                "exit_time_utc": second_entry + 900,
                "exit_price": 145.1,
                "exit_reason": "sl",
                "pnl": -1_000.0,
                "equity_before": 1_002_000.0,
                "equity_after": 1_001_000.0,
                "realized_r": -1.0,
            },
        ]
    )
    drawdown_pct = -1_000.0 / 1_002_000.0 * 100.0
    payload: dict[str, object] = {
        "summary": {
            "detector": "fixture_detector",
            "zone_count": 2,
            "trade_count": 2,
            "open_position_count": 0,
            "final_equity": 1_001_000.0,
            "return_pct": 0.1,
            "win_rate_pct": 50.0,
            "max_drawdown_pct": drawdown_pct,
            "profit_factor": 2.0,
            "average_realized_r": 0.0,
            "positive_segments": 1,
            "long_count": 1,
            "short_count": 1,
            "sample_sufficient": False,
            "criterion": "insufficient_sample",
        },
        "segments": [
            {
                "segment": 1,
                "start": "2024-01-06",
                "end_exclusive": "2024-07-08",
                "trade_count": 1,
                "pnl_jpy": 2_000.0,
                "return_pct": 0.2,
            },
            {
                "segment": 2,
                "start": "2024-07-08",
                "end_exclusive": "2025-01-08",
                "trade_count": 1,
                "pnl_jpy": -1_000.0,
                "return_pct": drawdown_pct,
            },
            {
                "segment": 3,
                "start": "2025-01-08",
                "end_exclusive": "2025-07-11",
                "trade_count": 0,
                "pnl_jpy": 0.0,
                "return_pct": 0.0,
            },
            {
                "segment": 4,
                "start": "2025-07-11",
                "end_exclusive": "2026-01-11",
                "trade_count": 0,
                "pnl_jpy": 0.0,
                "return_pct": 0.0,
            },
            {
                "segment": 5,
                "start": "2026-01-11",
                "end_exclusive": "2026-07-14",
                "trade_count": 0,
                "pnl_jpy": 0.0,
                "return_pct": 0.0,
            },
        ],
        "exit_reasons": {"sl": 1, "tp": 1},
    }
    return trades, payload


def test_checked_in_evidence_pack_verifies_code_and_aggregates() -> None:
    payload = verify_evidence_pack(
        REFERENCE_DIR,
        code_paths=stage8_code_paths(REPO_ROOT, CHAPTER_DIR),
    )

    assert payload["reproducibility_runs"] == 2
    assert payload["hashes"]["stage7_result_sha256"] == STAGE7_SHA256
    assert payload["detectors"]["official"]["summary"]["criterion"] == "failed"
    assert payload["detectors"]["secondary"]["summary"]["criterion"] == "failed"


def test_stage8_code_labels_are_repo_relative_files() -> None:
    code_paths = stage8_code_paths(REPO_ROOT, CHAPTER_DIR)

    assert all(path == REPO_ROOT / label for label, path in code_paths.items())
    assert all(path.is_file() for path in code_paths.values())


def test_public_manifest_excludes_row_data_paths_and_credentials() -> None:
    text = (REFERENCE_DIR / "manifest.json").read_text(encoding="utf-8")

    for forbidden in (
        "/Users/",
        "OANDA_API_TOKEN",
        "entry_price",
        "exit_price",
        "active_from_ts_utc",
        "zone_id",
        "liquidity_id",
        "private_artifact_sha256",
    ):
        assert forbidden not in text


def test_verifier_rejects_manifest_changed_without_checksum(tmp_path: Path) -> None:
    copied = tmp_path / "reference"
    shutil.copytree(REFERENCE_DIR, copied)
    path = copied / "manifest.json"
    path.write_text(
        path.read_text(encoding="utf-8").replace("1121", "1122", 1),
        encoding="utf-8",
    )

    with pytest.raises(IctObEvidenceError, match="hash mismatch"):
        verify_evidence_pack(copied, verify_environment=False)


def test_verifier_rejects_internally_inconsistent_rehashed_manifest(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "reference"
    shutil.copytree(REFERENCE_DIR, copied)
    path = copied / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["detectors"]["official"]["summary"]["trade_count"] += 1
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rewrite_checksum(copied)

    with pytest.raises(IctObEvidenceError, match="trade direction counts"):
        verify_evidence_pack(copied, verify_environment=False)


def test_verifier_rejects_aggregate_changed_without_semantic_rehash(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "reference"
    shutil.copytree(REFERENCE_DIR, copied)
    path = copied / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["detectors"]["official"]["summary"]["profit_factor"] += 0.01
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rewrite_checksum(copied)

    with pytest.raises(IctObEvidenceError, match="detector result hash disagrees"):
        verify_evidence_pack(copied, verify_environment=False)


def test_private_verifier_independently_recomputes_frozen_metrics(
    tmp_path: Path,
) -> None:
    trades, detector_payload = _private_trade_fixture()
    path = tmp_path / "trades.csv"
    trades.to_csv(path, index=False)

    recomputed = verify_private_detector_trades(path, detector_payload)

    assert recomputed["trade_count"] == 2
    assert recomputed["final_equity"] == pytest.approx(1_001_000.0)
    assert recomputed["profit_factor"] == pytest.approx(2.0)
    assert recomputed["positive_segments"] == 1
    assert recomputed["criterion"] == "insufficient_sample"
    assert recomputed["exit_reasons"] == {"sl": 1, "tp": 1}


def test_private_verifier_rejects_tampered_trade_pnl(tmp_path: Path) -> None:
    trades, detector_payload = _private_trade_fixture()
    trades.loc[0, "pnl"] += 1.0
    path = tmp_path / "trades.csv"
    trades.to_csv(path, index=False)

    with pytest.raises(IctObEvidenceError, match="trade PnL disagrees"):
        verify_private_detector_trades(path, detector_payload)
