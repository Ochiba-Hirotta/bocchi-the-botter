"""Fixture tests for the Season 2 chapter 2 M5 SQLite ingestion."""
from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pytest
import requests

from bocchi_the_botter_repro.season2.minute_data import (
    CHUNK_DAYS,
    GRANULARITY,
    MAX_ATTEMPTS,
    PRICE,
    ApiRequestError,
    Candle,
    CandleValidationError,
    DataProjectionError,
    DatabaseSchemaError,
    IngestionSummary,
    OandaCandleClient,
    aggregate_m5_to_m15,
    audit_m5_frame,
    build_evidence_manifest,
    extraction_sha256,
    fetch_to_sqlite,
    find_m5_gaps,
    format_utc_timestamp,
    iter_chunks,
    load_m5_candles,
    normalize_response,
    open_database,
    open_read_only_database,
    parse_m5_boundary,
    select_m15_et_time,
    upsert_candles,
)


TOKEN = "fixture-secret-token"
START = parse_m5_boundary("2026-07-06T13:00:00Z")
END = parse_m5_boundary("2026-07-06T13:15:00Z")


class FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = deque(outcomes)
        self.calls: list[dict[str, object]] = []
        self.trust_env = True
        self.closed = False

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str],
        headers: Mapping[str, str],
        timeout: tuple[float, float],
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "params": dict(params),
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, FakeResponse)
        return outcome

    def close(self) -> None:
        self.closed = True


def candle_payload(
    timestamp: str,
    *,
    complete: bool = True,
    bid: Mapping[str, str] | None = None,
    ask: Mapping[str, str] | None = None,
    volume: object = 10,
) -> dict[str, object]:
    return {
        "time": timestamp,
        "complete": complete,
        "volume": volume,
        "bid": (
            bid
            if bid is not None
            else {"o": "100.00", "h": "100.20", "l": "99.90", "c": "100.10"}
        ),
        "ask": (
            ask
            if ask is not None
            else {"o": "100.02", "h": "100.22", "l": "99.92", "c": "100.12"}
        ),
    }


def api_payload(*candles: object) -> dict[str, object]:
    return {
        "instrument": "USD/JPY",
        "granularity": GRANULARITY,
        "candles": list(candles),
    }


def validated_candle(*, ts_utc: int = START, fetched_at: int = 1) -> Candle:
    batch = normalize_response(
        api_payload(candle_payload(format_utc_timestamp(ts_utc))),
        instrument="USD_JPY",
        chunk_start=ts_utc,
        chunk_end=ts_utc + 300,
        fetched_at_utc=fetched_at,
    )
    return batch.candles[0]


def test_chunk_ranges_cover_interval_without_overlap_or_gap() -> None:
    end = START + 30 * 24 * 60 * 60
    chunks = list(iter_chunks(START, end))

    assert chunks[0][0] == START
    assert chunks[-1][1] == end
    assert len(chunks) == 3
    assert all(right[0] == left[1] for left, right in zip(chunks, chunks[1:]))
    assert all(
        chunk_end - chunk_start <= CHUNK_DAYS * 24 * 60 * 60
        for chunk_start, chunk_end in chunks
    )


def test_client_sends_fixed_m5_ba_parameters_and_clips_to_half_open_chunk() -> None:
    payload = api_payload(
        candle_payload("2026-07-06T13:00:00.000000000Z"),
        candle_payload("2026-07-06T13:15:00.000000000Z"),
    )
    session = FakeSession([FakeResponse(payload)])
    client = OandaCandleClient(TOKEN, session=session, sleep=lambda _: None)

    batch = client.fetch_chunk(
        instrument="USD_JPY",
        chunk_start=START,
        chunk_end=END,
        fetched_at_utc=1,
    )

    assert session.trust_env is False
    assert len(batch.candles) == 1
    assert batch.outside_chunk_skipped == 1
    call = session.calls[0]
    assert call["url"] == "https://api-fxpractice.oanda.com/v3/instruments/USD_JPY/candles"
    assert call["params"] == {
        "from": "2026-07-06T13:00:00.000000000Z",
        "to": "2026-07-06T13:15:00.000000000Z",
        "granularity": "M5",
        "price": PRICE,
        "smooth": "false",
        "includeFirst": "true",
    }
    assert "count" not in call["params"]
    assert call["timeout"] == (5.0, 30.0)


def test_incomplete_candle_is_counted_but_not_saved() -> None:
    batch = normalize_response(
        api_payload(
            candle_payload("2026-07-06T13:00:00.000000000Z", complete=False),
            candle_payload("2026-07-06T13:05:00.000000000Z"),
        ),
        instrument="USD_JPY",
        chunk_start=START,
        chunk_end=END,
        fetched_at_utc=1,
    )

    assert batch.incomplete_skipped == 1
    assert batch.complete_received == 1
    assert [candle.ts_utc for candle in batch.candles] == [START + 300]


@pytest.mark.parametrize(
    "raw, message",
    [
        (
            candle_payload("2026-07-06T13:01:00.000000000Z"),
            "off the M5 boundary",
        ),
        (
            candle_payload("2026-07-06T13:00:00.000000000Z", bid={}),
            "missing bid.o",
        ),
        (
            candle_payload(
                "2026-07-06T13:00:00.000000000Z",
                bid={"o": "100", "h": "99", "l": "98", "c": "98.5"},
            ),
            "inconsistent",
        ),
        (
            candle_payload("2026-07-06T13:00:00.000000000Z", volume=-1),
            "non-negative integer",
        ),
    ],
)
def test_malformed_completed_candles_fail_instead_of_being_skipped(
    raw: dict[str, object], message: str
) -> None:
    with pytest.raises(CandleValidationError, match=message):
        normalize_response(
            api_payload(raw),
            instrument="USD_JPY",
            chunk_start=START,
            chunk_end=END,
            fetched_at_utc=1,
        )


def test_nine_digit_rfc3339_timestamp_is_parsed_to_integer_epoch() -> None:
    parsed = parse_m5_boundary("2026-07-06T13:00:00.000000000Z")

    assert parsed == START
    assert format_utc_timestamp(parsed) == "2026-07-06T13:00:00.000000000Z"


def test_retry_is_bounded_and_honors_retry_after() -> None:
    session = FakeSession(
        [
            requests.Timeout("temporary"),
            FakeResponse({}, status_code=429, headers={"Retry-After": "3"}),
            FakeResponse(api_payload()),
        ]
    )
    delays: list[float] = []
    client = OandaCandleClient(TOKEN, session=session, sleep=delays.append)

    batch = client.fetch_chunk(
        instrument="USD_JPY",
        chunk_start=START,
        chunk_end=END,
        fetched_at_utc=1,
    )

    assert batch.candles == ()
    assert len(session.calls) == MAX_ATTEMPTS
    assert delays == [1.0, 3.0]


def test_non_retryable_error_redacts_token_and_stops_after_one_request() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {},
                status_code=401,
                headers={"RequestID": "fixture-request"},
                text=f"bad token {TOKEN}",
            )
        ]
    )
    client = OandaCandleClient(TOKEN, session=session, sleep=lambda _: None)

    with pytest.raises(ApiRequestError) as captured:
        client.fetch_chunk(
            instrument="USD_JPY",
            chunk_start=START,
            chunk_end=END,
            fetched_at_utc=1,
        )

    assert len(session.calls) == 1
    assert TOKEN not in str(captured.value)
    assert "<redacted>" in str(captured.value)


def test_schema_has_only_the_sixteen_contract_columns(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "rates.sqlite")
    try:
        columns = connection.execute("PRAGMA table_info(oanda_candles)").fetchall()
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        connection.close()

    assert user_version == 1
    assert check == "ok"
    assert len(columns) == 16
    assert [column[1] for column in columns] == [
        "source",
        "instrument",
        "granularity",
        "price",
        "ts_utc",
        "fetched_at_utc",
        "complete",
        "volume",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]
    assert all("token" not in column[1] and "account" not in column[1] for column in columns)


def test_upsert_is_idempotent_and_updates_completed_values(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "rates.sqlite")
    try:
        first = validated_candle(fetched_at=1)
        second = replace(
            first,
            fetched_at_utc=2,
            volume=12,
            bid_close=100.11,
            ask_close=100.13,
        )
        upsert_candles(connection, [first])
        upsert_candles(connection, [second])
        row = connection.execute(
            "SELECT COUNT(*), fetched_at_utc, volume, bid_close, ask_close "
            "FROM oanda_candles"
        ).fetchone()
    finally:
        connection.close()

    assert row == pytest.approx((1, 2, 12, 100.11, 100.13))


def test_constraint_failure_rolls_back_entire_chunk(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "rates.sqlite")
    good = validated_candle()
    bad = replace(good, ts_utc=START + 1)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            upsert_candles(connection, [good, bad])
        count = connection.execute("SELECT COUNT(*) FROM oanda_candles").fetchone()[0]
    finally:
        connection.close()

    assert count == 0


def test_existing_incompatible_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE oanda_candles (ts_utc INTEGER PRIMARY KEY)")
    connection.execute("PRAGMA user_version = 1")
    connection.close()

    with pytest.raises(DatabaseSchemaError, match="incompatible"):
        open_database(path)


def test_fixture_ingestion_builds_a_new_db_and_is_idempotent(tmp_path: Path) -> None:
    payload = api_payload(candle_payload("2026-07-06T13:00:00.000000000Z"))
    session = FakeSession([FakeResponse(payload), FakeResponse(payload)])
    client = OandaCandleClient(TOKEN, session=session, sleep=lambda _: None)
    db_path = tmp_path / "rates.sqlite"

    first = fetch_to_sqlite(
        token=TOKEN,
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=END,
        db_path=db_path,
        client=client,
        fetched_at_utc=1,
    )
    second = fetch_to_sqlite(
        token=TOKEN,
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=END,
        db_path=db_path,
        client=client,
        fetched_at_utc=2,
    )

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT COUNT(*), fetched_at_utc FROM oanda_candles"
        ).fetchone()
    finally:
        connection.close()

    assert first.rows_upserted == 1
    assert second.rows_upserted == 1
    assert first.request_count == second.request_count == 1
    assert row == (1, 2)


def build_projection_db(path: Path) -> None:
    connection = open_database(path)
    try:
        rows = [
            validated_candle(ts_utc=START),
            validated_candle(ts_utc=START + 300),
            validated_candle(ts_utc=START + 900),
            validated_candle(ts_utc=START + 1200),
        ]
        upsert_candles(connection, rows)
    finally:
        connection.close()


def test_read_only_projection_is_sorted_half_open_and_utc_aware(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rates.sqlite"
    build_projection_db(db_path)

    frame = load_m5_candles(
        db_path,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=START + 1200,
    )

    assert frame["ts_utc"].tolist() == [START, START + 300, START + 900]
    assert str(frame["ts_utc_dt"].dt.tz) == "UTC"
    assert frame["spread_open"].tolist() == pytest.approx([0.02, 0.02, 0.02])
    assert frame["spread_close"].tolist() == pytest.approx([0.02, 0.02, 0.02])
    assert not any(column.startswith("mid_") for column in frame.columns)


def test_read_only_connection_rejects_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "rates.sqlite"
    build_projection_db(db_path)
    connection = open_read_only_database(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE forbidden (value INTEGER)")
    finally:
        connection.close()


def test_projection_rejects_an_empty_half_open_range(tmp_path: Path) -> None:
    db_path = tmp_path / "rates.sqlite"
    build_projection_db(db_path)

    with pytest.raises(DataProjectionError, match="empty"):
        load_m5_candles(
            db_path,
            source="oanda_rest_v20",
            instrument="USD_JPY",
            start_inclusive=START + 3600,
            end_exclusive=START + 3900,
        )


def test_gap_audit_does_not_fill_the_missing_middle_candle(tmp_path: Path) -> None:
    db_path = tmp_path / "rates.sqlite"
    build_projection_db(db_path)
    end = START + 1200
    frame = load_m5_candles(
        db_path,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=end,
    )

    gaps = find_m5_gaps(frame)
    audit = audit_m5_frame(
        frame,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=end,
    )

    assert len(frame) == 3
    assert len(gaps) == 1
    assert gaps.iloc[0]["missing_m5_slots"] == 1
    assert gaps.iloc[0]["classification"] == "short_unclassified"
    assert audit.gap_count == 1
    assert audit.missing_m5_slots == 1
    assert audit.short_gap_count == 1
    assert audit.duplicate_count == 0
    assert audit.off_boundary_count == 0
    assert len(audit.extraction_sha256) == 64


def test_gap_classification_keeps_short_sunday_gap_separate_from_weekend_close() -> None:
    timestamps = [
        int(value.timestamp())
        for value in pd.to_datetime(
            [
                "2026-07-10T21:55:00Z",
                "2026-07-12T21:50:00Z",
                "2026-07-12T22:00:00Z",
            ],
            utc=True,
        )
    ]
    frame = pd.DataFrame({"ts_utc": timestamps})

    gaps = find_m5_gaps(frame)

    assert gaps["classification"].tolist() == [
        "weekend_closure_candidate",
        "short_unclassified",
    ]


def test_extraction_hash_is_stable_and_includes_conditions(tmp_path: Path) -> None:
    db_path = tmp_path / "rates.sqlite"
    build_projection_db(db_path)
    end = START + 1200
    frame = load_m5_candles(
        db_path,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=end,
    )

    first = extraction_sha256(
        frame,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=end,
    )
    second = extraction_sha256(
        frame.copy(),
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=end,
    )
    changed_condition = extraction_sha256(
        frame,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START - 300,
        end_exclusive=end,
    )

    assert first == second
    assert first != changed_condition


def test_projection_rejects_negative_derived_spread(tmp_path: Path) -> None:
    db_path = tmp_path / "rates.sqlite"
    build_projection_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE oanda_candles SET ask_open = bid_open - 0.01 "
            "WHERE ts_utc = ?",
            (START,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DataProjectionError, match="negative open/close spread"):
        load_m5_candles(
            db_path,
            source="oanda_rest_v20",
            instrument="USD_JPY",
            start_inclusive=START,
            end_exclusive=START + 1200,
        )


def test_projection_rejects_fractional_volume(tmp_path: Path) -> None:
    db_path = tmp_path / "rates.sqlite"
    build_projection_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE oanda_candles SET volume = 1.5 WHERE ts_utc = ?",
            (START,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DataProjectionError, match="invalid volume"):
        load_m5_candles(
            db_path,
            source="oanda_rest_v20",
            instrument="USD_JPY",
            start_inclusive=START,
            end_exclusive=START + 1200,
        )


def load_fixture_rows(
    db_path: Path,
    rows: list[Candle],
    *,
    start: int,
    end: int,
) -> pd.DataFrame:
    connection = open_database(db_path)
    try:
        upsert_candles(connection, rows)
    finally:
        connection.close()
    return load_m5_candles(
        db_path,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=start,
        end_exclusive=end,
    )


def test_three_exact_m5_candles_aggregate_bid_and_ask_separately(
    tmp_path: Path,
) -> None:
    rows = [
        replace(
            validated_candle(ts_utc=START),
            volume=10,
            bid_open=100.00,
            bid_high=100.20,
            bid_low=99.90,
            bid_close=100.10,
            ask_open=100.02,
            ask_high=100.22,
            ask_low=99.92,
            ask_close=100.12,
        ),
        replace(
            validated_candle(ts_utc=START + 300),
            volume=20,
            bid_open=100.10,
            bid_high=100.50,
            bid_low=100.00,
            bid_close=100.40,
            ask_open=100.12,
            ask_high=100.52,
            ask_low=100.02,
            ask_close=100.42,
        ),
        replace(
            validated_candle(ts_utc=START + 600),
            volume=30,
            bid_open=100.40,
            bid_high=100.45,
            bid_low=99.80,
            bid_close=100.00,
            ask_open=100.42,
            ask_high=100.47,
            ask_low=99.82,
            ask_close=100.02,
        ),
    ]
    frame = load_fixture_rows(
        tmp_path / "rates.sqlite",
        rows,
        start=START,
        end=START + 900,
    )

    result = aggregate_m5_to_m15(
        frame,
        start_inclusive=START,
        end_exclusive=START + 900,
    )

    assert len(result.candles) == 1
    assert result.incomplete_buckets.empty
    candle = result.candles.iloc[0]
    assert candle["ts_utc"] == START
    assert candle["granularity"] == "M15"
    assert candle["price"] == "BA"
    assert candle["component_count"] == 3
    assert candle["complete"] == 1
    assert candle["volume"] == 60
    assert candle["bid_open"] == pytest.approx(100.00)
    assert candle["bid_high"] == pytest.approx(100.50)
    assert candle["bid_low"] == pytest.approx(99.80)
    assert candle["bid_close"] == pytest.approx(100.00)
    assert candle["ask_open"] == pytest.approx(100.02)
    assert candle["ask_high"] == pytest.approx(100.52)
    assert candle["ask_low"] == pytest.approx(99.82)
    assert candle["ask_close"] == pytest.approx(100.02)
    assert candle["spread_open"] == pytest.approx(0.02)
    assert candle["spread_close"] == pytest.approx(0.02)
    assert not any(column.startswith("mid_") for column in result.candles.columns)
    assert "present_ts_utc" not in result.candles.columns


def test_missing_middle_m5_rejects_bucket_without_filling(tmp_path: Path) -> None:
    rows = [
        validated_candle(ts_utc=START),
        validated_candle(ts_utc=START + 600),
    ]
    frame = load_fixture_rows(
        tmp_path / "rates.sqlite",
        rows,
        start=START,
        end=START + 900,
    )

    result = aggregate_m5_to_m15(
        frame,
        start_inclusive=START,
        end_exclusive=START + 900,
    )

    assert result.candles.empty
    assert len(result.incomplete_buckets) == 1
    rejected = result.incomplete_buckets.iloc[0]
    assert rejected["component_count"] == 2
    assert rejected["missing_ts_utc"] == (START + 300,)
    assert rejected["reason"] == "missing_m5"


def test_partial_first_bucket_is_labeled_as_extraction_boundary(
    tmp_path: Path,
) -> None:
    start = START + 600
    end = START + 900
    frame = load_fixture_rows(
        tmp_path / "rates.sqlite",
        [validated_candle(ts_utc=start)],
        start=start,
        end=end,
    )

    result = aggregate_m5_to_m15(
        frame,
        start_inclusive=start,
        end_exclusive=end,
    )

    assert result.candles.empty
    rejected = result.incomplete_buckets.iloc[0]
    assert rejected["missing_ts_utc"] == (START, START + 300)
    assert rejected["reason"] == "extraction_boundary"


def test_new_york_0930_label_tracks_dst_without_changing_utc(
    tmp_path: Path,
) -> None:
    winter_bucket = parse_m5_boundary("2026-03-06T14:30:00Z")
    summer_bucket = parse_m5_boundary("2026-03-09T13:30:00Z")
    rows = [
        validated_candle(ts_utc=base + offset)
        for base in (winter_bucket, summer_bucket)
        for offset in (0, 300, 600)
    ]
    frame = load_fixture_rows(
        tmp_path / "rates.sqlite",
        rows,
        start=winter_bucket,
        end=summer_bucket + 900,
    )

    result = aggregate_m5_to_m15(
        frame,
        start_inclusive=winter_bucket,
        end_exclusive=summer_bucket + 900,
    )
    selected = select_m15_et_time(result.candles, hour=9, minute=30)

    assert result.candles["ts_utc"].tolist() == [winter_bucket, summer_bucket]
    assert len(selected) == 2
    assert selected["ts_et"].dt.strftime("%H:%M").tolist() == ["09:30", "09:30"]
    assert selected["ts_et"].map(lambda value: value.utcoffset().total_seconds()).tolist() == [
        -18_000.0,
        -14_400.0,
    ]
    assert selected["ts_utc_dt"].dt.strftime("%H:%M").tolist() == ["14:30", "13:30"]


def test_evidence_manifest_has_required_audit_values_without_rows_or_paths(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rates.sqlite"
    rows = [validated_candle(ts_utc=START + offset) for offset in (0, 300, 600)]
    frame = load_fixture_rows(
        db_path,
        rows,
        start=START,
        end=START + 900,
    )
    m5_audit = audit_m5_frame(
        frame,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=START + 900,
    )
    m15 = aggregate_m5_to_m15(
        frame,
        start_inclusive=START,
        end_exclusive=START + 900,
    )

    manifest = build_evidence_manifest(
        db_path=db_path,
        environment="practice",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=START + 900,
        m5_audit=m5_audit,
        m15=m15,
        code_commit="0123456789abcdef-dirty",
        generated_at_utc="2026-07-14T12:00:00Z",
        command=(
            "python chapters/season2/ch02_minute_data_db/build_manifest.py "
            '--db "$SOURCE_DB" --instrument USD_JPY '
            "--start 2026-07-06T13:00:00Z --end 2026-07-06T13:15:00Z "
            "--manifest results/reference/ch02_minute_data_db/manifest.json"
        ),
    )

    rendered = str(manifest)
    assert manifest["schema_version"] == 1
    assert manifest["db"]["basename"] == "rates.sqlite"
    assert len(manifest["db"]["sha256"]) == 64
    assert manifest["db"]["quick_check"] == "ok"
    assert manifest["m5"]["row_count"] == 3
    assert manifest["m15"]["complete_bucket_count"] == 1
    assert manifest["m15"]["incomplete_bucket_count"] == 0
    assert manifest["fetch"]["request_count"] is None
    assert str(tmp_path) not in rendered
    assert TOKEN not in rendered
    assert "bid_open" not in rendered

    same_run = build_evidence_manifest(
        db_path=db_path,
        environment="practice",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=START + 900,
        m5_audit=m5_audit,
        m15=m15,
        code_commit="0123456789abcdef-dirty",
        generated_at_utc="2026-07-14T12:00:00Z",
        command=(
            "python chapters/season2/ch02_minute_data_db/run.py "
            "--instrument USD_JPY --start 2026-07-06T13:00:00Z "
            "--end 2026-07-06T13:15:00Z "
            "--db outputs/ch02_minute_data_db/rates.sqlite "
            "--manifest outputs/ch02_minute_data_db/manifest.json"
        ),
        ingestion=IngestionSummary(
            instrument="USD_JPY",
            start_inclusive=START,
            end_exclusive=START + 900,
            chunk_count=1,
            request_count=1,
            complete_received=3,
            incomplete_skipped=0,
            outside_chunk_skipped=0,
            rows_upserted=3,
        ),
    )
    assert same_run["db"]["role"] == "article_generated"
    assert same_run["fetch"]["counters_scope"] == "same_run"
    assert same_run["fetch"]["request_count"] == 1
    assert same_run["fetch"]["complete_received"] == 3


@pytest.mark.parametrize(
    "command",
    [
        "python build_manifest.py --db /Users/example/rates.sqlite",
        "OANDA_API_TOKEN=secret python build_manifest.py",
    ],
)
def test_evidence_manifest_rejects_unsafe_reproduction_command(
    tmp_path: Path,
    command: str,
) -> None:
    db_path = tmp_path / "rates.sqlite"
    frame = load_fixture_rows(
        db_path,
        [validated_candle(ts_utc=START)],
        start=START,
        end=START + 300,
    )
    audit = audit_m5_frame(
        frame,
        source="oanda_rest_v20",
        instrument="USD_JPY",
        start_inclusive=START,
        end_exclusive=START + 300,
    )
    m15 = aggregate_m5_to_m15(
        frame,
        start_inclusive=START,
        end_exclusive=START + 300,
    )

    with pytest.raises(ValueError, match="command"):
        build_evidence_manifest(
            db_path=db_path,
            environment="practice",
            instrument="USD_JPY",
            start_inclusive=START,
            end_exclusive=START + 300,
            m5_audit=audit,
            m15=m15,
            code_commit="0123456789abcdef-dirty",
            command=command,
        )
