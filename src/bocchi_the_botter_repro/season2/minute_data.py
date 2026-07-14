"""Season 2 chapter 2: fetch complete OANDA M5 bid/ask candles to SQLite.

This module is intentionally independent from ``paper-trader``.  It implements
only the article's small ingestion path: one instrument, OANDA's practice REST
endpoint, M5 bid/ask candles, and an idempotent SQLite upsert.

Raw responses, account information, and credentials are never persisted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import platform
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import requests


OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
OANDA_CANDLES_PATH = "/v3/instruments/{instrument}/candles"
TOKEN_ENV_VAR = "OANDA_API_TOKEN"
SOURCE = "oanda_rest_v20"
GRANULARITY = "M5"
PRICE = "BA"
SCHEMA_VERSION = 1
CHUNK_DAYS = 14
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_FIVE_MINUTES = 300
_RFC3339_UTC = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)
_INSTRUMENT = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")


class MinuteDataError(RuntimeError):
    """Base error for the chapter's minute-data ingestion path."""


class ApiRequestError(MinuteDataError):
    """Raised when an OANDA candle request cannot be completed safely."""


class CandleValidationError(MinuteDataError):
    """Raised when a completed candle violates the M5 BA contract."""


class DatabaseSchemaError(MinuteDataError):
    """Raised when an existing SQLite file does not match schema version 1."""


class DataProjectionError(ValueError):
    """Raised when a read-only M5 projection violates the data contract."""


class ResponseLike(Protocol):
    """Small response surface used by the client and artificial fixtures."""

    status_code: int
    headers: Mapping[str, str]
    text: str

    def json(self) -> Any: ...


class SessionLike(Protocol):
    """Small requests.Session surface used by the client."""

    trust_env: bool

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str],
        headers: Mapping[str, str],
        timeout: tuple[float, float],
    ) -> ResponseLike: ...

    def close(self) -> None: ...


SleepFunction = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class Candle:
    """One validated, completed OANDA M5 bid/ask candle."""

    source: str
    instrument: str
    granularity: str
    price: str
    ts_utc: int
    fetched_at_utc: int
    complete: int
    volume: int
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float

    def as_sql_values(self) -> tuple[object, ...]:
        """Return values in the version-1 SQLite column order."""

        return (
            self.source,
            self.instrument,
            self.granularity,
            self.price,
            self.ts_utc,
            self.fetched_at_utc,
            self.complete,
            self.volume,
            self.bid_open,
            self.bid_high,
            self.bid_low,
            self.bid_close,
            self.ask_open,
            self.ask_high,
            self.ask_low,
            self.ask_close,
        )


@dataclass(frozen=True, slots=True)
class NormalizedBatch:
    """Validated rows and audit counts from one API response."""

    candles: tuple[Candle, ...]
    complete_received: int
    incomplete_skipped: int
    outside_chunk_skipped: int


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """Small, credential-free summary returned by one ingestion run."""

    instrument: str
    start_inclusive: int
    end_exclusive: int
    chunk_count: int
    request_count: int
    complete_received: int
    incomplete_skipped: int
    outside_chunk_skipped: int
    rows_upserted: int


@dataclass(frozen=True, slots=True)
class M5Audit:
    """Deterministic quality summary for one M5 DataFrame projection."""

    row_count: int
    first_ts_utc: int
    last_ts_utc: int
    duplicate_count: int
    off_boundary_count: int
    null_required_count: int
    invalid_volume_count: int
    invalid_ohlc_count: int
    negative_spread_count: int
    sorted_ascending: bool
    gap_count: int
    missing_m5_slots: int
    weekend_gap_count: int
    long_non_weekend_gap_count: int
    short_gap_count: int
    extraction_sha256: str


@dataclass(slots=True)
class M15Aggregation:
    """Complete derived M15 candles and buckets rejected as incomplete."""

    candles: pd.DataFrame
    incomplete_buckets: pd.DataFrame


AuditManifest = dict[str, object]


COMMON_DB_COLUMNS = (
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
)

_PRICE_COLUMNS = (
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
)

_M15_COLUMNS = (
    "source",
    "instrument",
    "granularity",
    "price",
    "ts_utc",
    "fetched_at_utc",
    "complete",
    "component_count",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "ts_utc_dt",
    "ts_et",
    "session_date_et",
    "spread_open",
    "spread_close",
)

_INCOMPLETE_BUCKET_COLUMNS = (
    "source",
    "instrument",
    "price",
    "bucket_ts_utc",
    "bucket_utc",
    "bucket_et",
    "component_count",
    "present_ts_utc",
    "expected_ts_utc",
    "missing_ts_utc",
    "reason",
)


_DDL = """
CREATE TABLE IF NOT EXISTS oanda_candles (
    source TEXT NOT NULL CHECK (source = 'oanda_rest_v20'),
    instrument TEXT NOT NULL,
    granularity TEXT NOT NULL CHECK (granularity = 'M5'),
    price TEXT NOT NULL CHECK (price = 'BA'),
    ts_utc INTEGER NOT NULL CHECK (ts_utc % 300 = 0),
    fetched_at_utc INTEGER NOT NULL,
    complete INTEGER NOT NULL CHECK (complete = 1),
    volume INTEGER NOT NULL CHECK (volume >= 0),
    bid_open REAL NOT NULL,
    bid_high REAL NOT NULL,
    bid_low REAL NOT NULL,
    bid_close REAL NOT NULL,
    ask_open REAL NOT NULL,
    ask_high REAL NOT NULL,
    ask_low REAL NOT NULL,
    ask_close REAL NOT NULL,
    PRIMARY KEY (source, instrument, granularity, price, ts_utc)
) WITHOUT ROWID
"""

_UPSERT = """
INSERT INTO oanda_candles (
    source, instrument, granularity, price, ts_utc, fetched_at_utc,
    complete, volume,
    bid_open, bid_high, bid_low, bid_close,
    ask_open, ask_high, ask_low, ask_close
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (source, instrument, granularity, price, ts_utc) DO UPDATE SET
    fetched_at_utc = excluded.fetched_at_utc,
    complete = excluded.complete,
    volume = excluded.volume,
    bid_open = excluded.bid_open,
    bid_high = excluded.bid_high,
    bid_low = excluded.bid_low,
    bid_close = excluded.bid_close,
    ask_open = excluded.ask_open,
    ask_high = excluded.ask_high,
    ask_low = excluded.ask_low,
    ask_close = excluded.ask_close
"""

_EXPECTED_COLUMNS = (
    ("source", "TEXT", 1, 1),
    ("instrument", "TEXT", 1, 2),
    ("granularity", "TEXT", 1, 3),
    ("price", "TEXT", 1, 4),
    ("ts_utc", "INTEGER", 1, 5),
    ("fetched_at_utc", "INTEGER", 1, 0),
    ("complete", "INTEGER", 1, 0),
    ("volume", "INTEGER", 1, 0),
    ("bid_open", "REAL", 1, 0),
    ("bid_high", "REAL", 1, 0),
    ("bid_low", "REAL", 1, 0),
    ("bid_close", "REAL", 1, 0),
    ("ask_open", "REAL", 1, 0),
    ("ask_high", "REAL", 1, 0),
    ("ask_low", "REAL", 1, 0),
    ("ask_close", "REAL", 1, 0),
)


def validate_instrument(instrument: str) -> str:
    """Return an accepted OANDA instrument name or raise."""

    if _INSTRUMENT.fullmatch(instrument) is None:
        raise ValueError("instrument must match AAA_BBB, for example USD_JPY")
    return instrument


def parse_utc_timestamp(value: str) -> int:
    """Parse an RFC3339 UTC value and preserve the M5 integer-second contract."""

    match = _RFC3339_UTC.fullmatch(value)
    if match is None:
        raise ValueError(f"timestamp must be RFC3339 UTC ending in Z: {value!r}")
    fraction = match.group("fraction") or ""
    if any(character != "0" for character in fraction):
        raise ValueError("M5 candle timestamps must not contain sub-second time")
    try:
        parsed = dt.datetime.strptime(
            match.group("base"), "%Y-%m-%dT%H:%M:%S"
        ).replace(tzinfo=dt.UTC)
    except ValueError as exc:
        raise ValueError(f"invalid RFC3339 UTC timestamp: {value!r}") from exc
    return int(parsed.timestamp())


def parse_m5_boundary(value: str) -> int:
    """Parse a CLI boundary and require exact five-minute alignment."""

    timestamp = parse_utc_timestamp(value)
    if timestamp % _FIVE_MINUTES != 0:
        raise ValueError(f"timestamp is not on a five-minute boundary: {value!r}")
    return timestamp


def format_utc_timestamp(timestamp: int) -> str:
    """Format integer epoch seconds as OANDA's RFC3339 UTC representation."""

    value = dt.datetime.fromtimestamp(timestamp, tz=dt.UTC)
    return value.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def iter_chunks(
    start_inclusive: int,
    end_exclusive: int,
    *,
    chunk_days: int = CHUNK_DAYS,
) -> Iterator[tuple[int, int]]:
    """Yield consecutive half-open chunks covering the requested interval."""

    if start_inclusive >= end_exclusive:
        raise ValueError("start must be earlier than end")
    if start_inclusive % _FIVE_MINUTES or end_exclusive % _FIVE_MINUTES:
        raise ValueError("start and end must be five-minute boundaries")
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")

    step = int(dt.timedelta(days=chunk_days).total_seconds())
    cursor = start_inclusive
    while cursor < end_exclusive:
        chunk_end = min(cursor + step, end_exclusive)
        yield cursor, chunk_end
        cursor = chunk_end


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise CandleValidationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CandleValidationError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise CandleValidationError(f"{label} must be a finite number")
    return number


def _ohlc(component: object, *, label: str) -> tuple[float, float, float, float]:
    if not isinstance(component, Mapping):
        raise CandleValidationError(f"completed candle is missing {label} OHLC")
    try:
        opened = _finite_float(component["o"], label=f"{label}.o")
        high = _finite_float(component["h"], label=f"{label}.h")
        low = _finite_float(component["l"], label=f"{label}.l")
        closed = _finite_float(component["c"], label=f"{label}.c")
    except KeyError as exc:
        raise CandleValidationError(
            f"completed candle is missing {label}.{exc.args[0]}"
        ) from exc
    if not low <= opened <= high or not low <= closed <= high:
        raise CandleValidationError(f"{label} OHLC values are inconsistent")
    return opened, high, low, closed


def _volume(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandleValidationError("volume must be a non-negative integer")
    return value


def normalize_response(
    payload: object,
    *,
    instrument: str,
    chunk_start: int,
    chunk_end: int,
    fetched_at_utc: int,
) -> NormalizedBatch:
    """Validate one API payload and clip it to the local half-open chunk."""

    validate_instrument(instrument)
    if not isinstance(payload, Mapping):
        raise CandleValidationError("OANDA response must be a JSON object")
    if payload.get("granularity") != GRANULARITY:
        raise CandleValidationError("OANDA response granularity is not M5")
    raw_candles = payload.get("candles")
    if not isinstance(raw_candles, list):
        raise CandleValidationError("OANDA response candles must be a list")

    candles: list[Candle] = []
    complete_received = 0
    incomplete_skipped = 0
    outside_chunk_skipped = 0

    for raw in raw_candles:
        if not isinstance(raw, Mapping):
            raise CandleValidationError("each OANDA candle must be a JSON object")
        if raw.get("complete") is not True:
            incomplete_skipped += 1
            continue

        complete_received += 1
        raw_time = raw.get("time")
        if not isinstance(raw_time, str):
            raise CandleValidationError("completed candle is missing time")
        try:
            ts_utc = parse_utc_timestamp(raw_time)
        except ValueError as exc:
            raise CandleValidationError(str(exc)) from exc
        if ts_utc % _FIVE_MINUTES:
            raise CandleValidationError("completed candle is off the M5 boundary")

        bid_open, bid_high, bid_low, bid_close = _ohlc(
            raw.get("bid"), label="bid"
        )
        ask_open, ask_high, ask_low, ask_close = _ohlc(
            raw.get("ask"), label="ask"
        )
        volume = _volume(raw.get("volume"))
        if ask_open < bid_open or ask_close < bid_close:
            raise CandleValidationError("completed candle has a negative spread")

        if not chunk_start <= ts_utc < chunk_end:
            outside_chunk_skipped += 1
            continue

        candles.append(
            Candle(
                source=SOURCE,
                instrument=instrument,
                granularity=GRANULARITY,
                price=PRICE,
                ts_utc=ts_utc,
                fetched_at_utc=fetched_at_utc,
                complete=1,
                volume=volume,
                bid_open=bid_open,
                bid_high=bid_high,
                bid_low=bid_low,
                bid_close=bid_close,
                ask_open=ask_open,
                ask_high=ask_high,
                ask_low=ask_low,
                ask_close=ask_close,
            )
        )

    return NormalizedBatch(
        candles=tuple(candles),
        complete_received=complete_received,
        incomplete_skipped=incomplete_skipped,
        outside_chunk_skipped=outside_chunk_skipped,
    )


class OandaCandleClient:
    """Small practice-only OANDA candle client with bounded retries."""

    def __init__(
        self,
        token: str,
        *,
        session: SessionLike | None = None,
        sleep: SleepFunction = time.sleep,
    ) -> None:
        if not token:
            raise ValueError(f"{TOKEN_ENV_VAR} is empty")
        self._token = token
        self._owns_session = session is None
        self._session: SessionLike = session or requests.Session()
        self._session.trust_env = False
        self._sleep = sleep
        self.request_count = 0

    def close(self) -> None:
        """Close a Session created by this client."""

        if self._owns_session:
            self._session.close()

    def __enter__(self) -> OandaCandleClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_chunk(
        self,
        *,
        instrument: str,
        chunk_start: int,
        chunk_end: int,
        fetched_at_utc: int,
    ) -> NormalizedBatch:
        """Fetch and normalize one half-open chunk."""

        validate_instrument(instrument)
        url = OANDA_BASE_URL + OANDA_CANDLES_PATH.format(instrument=instrument)
        params = {
            "from": format_utc_timestamp(chunk_start),
            "to": format_utc_timestamp(chunk_end),
            "granularity": GRANULARITY,
            "price": PRICE,
            "smooth": "false",
            "includeFirst": "true",
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept-Datetime-Format": "RFC3339",
        }
        response = self._get(url=url, params=params, headers=headers)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiRequestError("OANDA returned invalid JSON") from exc
        return normalize_response(
            payload,
            instrument=instrument,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            fetched_at_utc=fetched_at_utc,
        )

    def _get(
        self,
        *,
        url: str,
        params: Mapping[str, str],
        headers: Mapping[str, str],
    ) -> ResponseLike:
        for attempt in range(MAX_ATTEMPTS):
            self.request_count += 1
            try:
                response = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt + 1 == MAX_ATTEMPTS:
                    message = self._sanitize(str(exc))
                    raise ApiRequestError(
                        f"OANDA request failed after {MAX_ATTEMPTS} attempts: {message}"
                    ) from exc
                self._sleep(float(2**attempt))
                continue

            if response.status_code == 200:
                return response
            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt + 1 < MAX_ATTEMPTS
            ):
                self._sleep(self._retry_delay(response, attempt))
                continue
            raise ApiRequestError(self._response_error(response))

        raise AssertionError("unreachable retry loop")

    def _retry_delay(self, response: ResponseLike, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(min(max(int(retry_after), 0), 30))
            except ValueError:
                pass
        return float(2**attempt)

    def _response_error(self, response: ResponseLike) -> str:
        request_id = response.headers.get("RequestID", "unknown")
        body = self._sanitize(response.text)[:500]
        return (
            f"OANDA request failed with HTTP {response.status_code}; "
            f"RequestID={request_id}; body={body!r}"
        )

    def _sanitize(self, value: str) -> str:
        return value.replace(self._token, "<redacted>")


def open_database(path: Path) -> sqlite3.Connection:
    """Create or open the article SQLite file and validate schema version 1."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        _initialize_or_validate_schema(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _initialize_or_validate_schema(connection: sqlite3.Connection) -> None:
    table_exists = (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("oanda_candles",),
        ).fetchone()
        is not None
    )
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])

    if not table_exists:
        if user_version not in (0, SCHEMA_VERSION):
            raise DatabaseSchemaError(
                f"unsupported SQLite user_version: {user_version}"
            )
        with connection:
            connection.execute(_DDL)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    elif user_version != SCHEMA_VERSION:
        raise DatabaseSchemaError(
            "existing oanda_candles table does not declare user_version 1"
        )

    actual = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(oanda_candles)")
    )
    if actual != _EXPECTED_COLUMNS:
        raise DatabaseSchemaError("existing oanda_candles schema is incompatible")


def upsert_candles(
    connection: sqlite3.Connection,
    candles: Sequence[Candle],
) -> int:
    """Upsert one fully normalized chunk in a single transaction."""

    if not candles:
        return 0
    values = [candle.as_sql_values() for candle in candles]
    with connection:
        connection.executemany(_UPSERT, values)
    return len(values)


def quick_check(connection: sqlite3.Connection) -> None:
    """Raise unless SQLite reports a healthy database."""

    result = connection.execute("PRAGMA quick_check").fetchone()
    if result is None or result[0] != "ok":
        detail = "no result" if result is None else str(result[0])
        raise MinuteDataError(f"SQLite quick_check failed: {detail}")


def fetch_to_sqlite(
    *,
    token: str,
    instrument: str,
    start_inclusive: int,
    end_exclusive: int,
    db_path: Path,
    client: OandaCandleClient | None = None,
    fetched_at_utc: int | None = None,
) -> IngestionSummary:
    """Fetch complete M5 BA candles and idempotently upsert them to SQLite."""

    validate_instrument(instrument)
    chunks = tuple(iter_chunks(start_inclusive, end_exclusive))
    run_timestamp = (
        int(dt.datetime.now(tz=dt.UTC).timestamp())
        if fetched_at_utc is None
        else fetched_at_utc
    )
    owns_client = client is None
    active_client = client or OandaCandleClient(token)
    connection = open_database(db_path)

    complete_received = 0
    incomplete_skipped = 0
    outside_chunk_skipped = 0
    rows_upserted = 0
    request_count_before = active_client.request_count
    try:
        for chunk_start, chunk_end in chunks:
            batch = active_client.fetch_chunk(
                instrument=instrument,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                fetched_at_utc=run_timestamp,
            )
            rows_upserted += upsert_candles(connection, batch.candles)
            complete_received += batch.complete_received
            incomplete_skipped += batch.incomplete_skipped
            outside_chunk_skipped += batch.outside_chunk_skipped
        quick_check(connection)
    finally:
        connection.close()
        if owns_client:
            active_client.close()

    return IngestionSummary(
        instrument=instrument,
        start_inclusive=start_inclusive,
        end_exclusive=end_exclusive,
        chunk_count=len(chunks),
        request_count=active_client.request_count - request_count_before,
        complete_received=complete_received,
        incomplete_skipped=incomplete_skipped,
        outside_chunk_skipped=outside_chunk_skipped,
        rows_upserted=rows_upserted,
    )


def open_read_only_database(path: Path) -> sqlite3.Connection:
    """Open an existing SQLite file in URI read-only and query-only mode."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def load_m5_candles(
    db_path: Path,
    *,
    source: str,
    instrument: str,
    start_inclusive: int,
    end_exclusive: int,
) -> pd.DataFrame:
    """Project complete M5 BA rows from SQLite without modifying the source DB."""

    if source != SOURCE:
        raise ValueError(f"source must be {SOURCE!r}")
    validate_instrument(instrument)
    if start_inclusive >= end_exclusive:
        raise ValueError("start must be earlier than end")
    if start_inclusive % _FIVE_MINUTES or end_exclusive % _FIVE_MINUTES:
        raise ValueError("start and end must be five-minute boundaries")

    columns_sql = ", ".join(COMMON_DB_COLUMNS)
    query = f"""
        SELECT {columns_sql}
        FROM oanda_candles
        WHERE source = ?
          AND instrument = ?
          AND granularity = ?
          AND price = ?
          AND complete = 1
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc ASC
    """
    connection = open_read_only_database(db_path)
    try:
        frame = pd.read_sql_query(
            query,
            connection,
            params=(
                source,
                instrument,
                GRANULARITY,
                PRICE,
                start_inclusive,
                end_exclusive,
            ),
        )
    finally:
        connection.close()

    if frame.empty:
        raise DataProjectionError(
            "M5 projection is empty for the requested source, instrument, and range"
        )

    frame["ts_utc_dt"] = pd.to_datetime(frame["ts_utc"], unit="s", utc=True)
    frame["spread_open"] = frame["ask_open"] - frame["bid_open"]
    frame["spread_close"] = frame["ask_close"] - frame["bid_close"]
    _validate_m5_projection(frame)
    return frame


def _validate_m5_projection(frame: pd.DataFrame) -> None:
    missing_columns = set(COMMON_DB_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise DataProjectionError(
            f"M5 projection is missing columns: {sorted(missing_columns)}"
        )

    required = list(COMMON_DB_COLUMNS)
    if frame[required].isna().any(axis=None):
        raise DataProjectionError("M5 projection contains NULL contract values")

    numeric = frame[list(_PRICE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise DataProjectionError("M5 projection contains non-finite prices")

    if frame.duplicated(
        subset=["source", "instrument", "granularity", "price", "ts_utc"]
    ).any():
        raise DataProjectionError("M5 projection contains duplicate candle keys")
    if not frame["ts_utc"].is_monotonic_increasing:
        raise DataProjectionError("M5 projection is not sorted by ts_utc")
    if (frame["ts_utc"] % _FIVE_MINUTES != 0).any():
        raise DataProjectionError("M5 projection contains off-boundary timestamps")
    if _invalid_volume_mask(frame).any():
        raise DataProjectionError("M5 projection contains invalid volume")

    invalid_ohlc = _invalid_ohlc_mask(frame)
    if invalid_ohlc.any():
        raise DataProjectionError("M5 projection contains inconsistent OHLC values")
    if (frame["spread_open"] < 0).any() or (frame["spread_close"] < 0).any():
        raise DataProjectionError("M5 projection contains negative open/close spread")


def _invalid_ohlc_mask(frame: pd.DataFrame) -> pd.Series:
    invalid = pd.Series(False, index=frame.index)
    for prefix in ("bid", "ask"):
        opened = frame[f"{prefix}_open"]
        high = frame[f"{prefix}_high"]
        low = frame[f"{prefix}_low"]
        closed = frame[f"{prefix}_close"]
        invalid |= ~(
            (low <= opened)
            & (opened <= high)
            & (low <= closed)
            & (closed <= high)
        )
    return invalid


def _invalid_volume_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isfinite(numeric) | (numeric < 0) | (numeric != np.floor(numeric))
    return pd.Series(invalid, index=frame.index)


def find_m5_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Return gaps larger than one M5 step without filling missing candles."""

    if "ts_utc" not in frame.columns:
        raise DataProjectionError("M5 frame is missing ts_utc")
    timestamps = frame["ts_utc"].astype("int64").sort_values().drop_duplicates()
    previous = timestamps.shift(1)
    delta = timestamps - previous
    positions = delta > _FIVE_MINUTES
    records: list[dict[str, object]] = []
    for next_ts, previous_ts, delta_seconds in zip(
        timestamps[positions], previous[positions], delta[positions]
    ):
        prior = int(previous_ts)
        following = int(next_ts)
        seconds = int(delta_seconds)
        records.append(
            {
                "previous_ts_utc": prior,
                "next_ts_utc": following,
                "previous_utc": pd.to_datetime(prior, unit="s", utc=True),
                "next_utc": pd.to_datetime(following, unit="s", utc=True),
                "delta_seconds": seconds,
                "missing_m5_slots": seconds // _FIVE_MINUTES - 1,
                "classification": _classify_gap(prior, following, seconds),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "previous_ts_utc",
            "next_ts_utc",
            "previous_utc",
            "next_utc",
            "delta_seconds",
            "missing_m5_slots",
            "classification",
        ],
    )


def _classify_gap(previous_ts: int, next_ts: int, delta_seconds: int) -> str:
    if delta_seconds < int(dt.timedelta(hours=12).total_seconds()):
        return "short_unclassified"
    previous_date = dt.datetime.fromtimestamp(previous_ts, tz=dt.UTC).date()
    next_date = dt.datetime.fromtimestamp(next_ts, tz=dt.UTC).date()
    cursor = previous_date
    includes_weekend = False
    while cursor <= next_date:
        if cursor.weekday() >= 5:
            includes_weekend = True
            break
        cursor += dt.timedelta(days=1)
    if includes_weekend:
        return "weekend_closure_candidate"
    return "long_non_weekend_closure_candidate"


def extraction_sha256(
    frame: pd.DataFrame,
    *,
    source: str,
    instrument: str,
    start_inclusive: int,
    end_exclusive: int,
) -> str:
    """Hash the ordered common projection and its half-open extraction terms."""

    missing_columns = set(COMMON_DB_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise DataProjectionError(
            f"cannot hash projection missing columns: {sorted(missing_columns)}"
        )
    if not frame["ts_utc"].is_monotonic_increasing:
        raise DataProjectionError("cannot hash an unsorted M5 projection")

    digest = hashlib.sha256()
    conditions = {
        "source": source,
        "instrument": instrument,
        "granularity": GRANULARITY,
        "price": PRICE,
        "start_inclusive": int(start_inclusive),
        "end_exclusive": int(end_exclusive),
    }
    digest.update(_canonical_json_line(conditions))

    for values in frame[list(COMMON_DB_COLUMNS)].itertuples(index=False, name=None):
        record = {
            "source": str(values[0]),
            "instrument": str(values[1]),
            "granularity": str(values[2]),
            "price": str(values[3]),
            "ts_utc": int(values[4]),
            "fetched_at_utc": int(values[5]),
            "complete": int(values[6]),
            "volume": int(values[7]),
            "bid_open": format(float(values[8]), ".17g"),
            "bid_high": format(float(values[9]), ".17g"),
            "bid_low": format(float(values[10]), ".17g"),
            "bid_close": format(float(values[11]), ".17g"),
            "ask_open": format(float(values[12]), ".17g"),
            "ask_high": format(float(values[13]), ".17g"),
            "ask_low": format(float(values[14]), ".17g"),
            "ask_close": format(float(values[15]), ".17g"),
        }
        digest.update(_canonical_json_line(record))
    return digest.hexdigest()


def _canonical_json_line(record: Mapping[str, object]) -> bytes:
    return (
        json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def audit_m5_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    instrument: str,
    start_inclusive: int,
    end_exclusive: int,
) -> M5Audit:
    """Build the deterministic Stage-4 quality summary for a projection."""

    if frame.empty:
        raise DataProjectionError("cannot audit an empty M5 projection")
    missing_columns = set(COMMON_DB_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise DataProjectionError(
            f"cannot audit projection missing columns: {sorted(missing_columns)}"
        )

    gaps = find_m5_gaps(frame)
    classifications = gaps["classification"].value_counts()
    null_required_count = int(
        frame[list(COMMON_DB_COLUMNS)].isna().any(axis=1).sum()
    )
    price_values = frame[list(_PRICE_COLUMNS)].to_numpy(dtype=float)
    non_finite_rows = ~np.isfinite(price_values).all(axis=1)
    invalid_ohlc_count = int((_invalid_ohlc_mask(frame) | non_finite_rows).sum())
    invalid_volume_count = int(_invalid_volume_mask(frame).sum())
    negative_spread_count = int(
        ((frame["ask_open"] - frame["bid_open"] < 0)
        | (frame["ask_close"] - frame["bid_close"] < 0)).sum()
    )

    return M5Audit(
        row_count=len(frame),
        first_ts_utc=int(frame["ts_utc"].min()),
        last_ts_utc=int(frame["ts_utc"].max()),
        duplicate_count=int(
            frame.duplicated(
                subset=[
                    "source",
                    "instrument",
                    "granularity",
                    "price",
                    "ts_utc",
                ]
            ).sum()
        ),
        off_boundary_count=int((frame["ts_utc"] % _FIVE_MINUTES != 0).sum()),
        null_required_count=null_required_count,
        invalid_volume_count=invalid_volume_count,
        invalid_ohlc_count=invalid_ohlc_count,
        negative_spread_count=negative_spread_count,
        sorted_ascending=bool(frame["ts_utc"].is_monotonic_increasing),
        gap_count=len(gaps),
        missing_m5_slots=(
            0 if gaps.empty else int(gaps["missing_m5_slots"].sum())
        ),
        weekend_gap_count=int(
            classifications.get("weekend_closure_candidate", 0)
        ),
        long_non_weekend_gap_count=int(
            classifications.get("long_non_weekend_closure_candidate", 0)
        ),
        short_gap_count=int(classifications.get("short_unclassified", 0)),
        extraction_sha256=extraction_sha256(
            frame,
            source=source,
            instrument=instrument,
            start_inclusive=start_inclusive,
            end_exclusive=end_exclusive,
        ),
    )


def aggregate_m5_to_m15(
    frame: pd.DataFrame,
    *,
    start_inclusive: int,
    end_exclusive: int,
) -> M15Aggregation:
    """Aggregate only exact three-candle M5 groups into bid/ask M15 candles."""

    if frame.empty:
        raise DataProjectionError("cannot aggregate an empty M5 projection")
    if start_inclusive >= end_exclusive:
        raise ValueError("start must be earlier than end")
    if start_inclusive % _FIVE_MINUTES or end_exclusive % _FIVE_MINUTES:
        raise ValueError("start and end must be five-minute boundaries")

    working = frame.copy()
    if "ts_utc_dt" not in working.columns:
        working["ts_utc_dt"] = pd.to_datetime(
            working["ts_utc"], unit="s", utc=True
        )
    if "spread_open" not in working.columns:
        working["spread_open"] = working["ask_open"] - working["bid_open"]
    if "spread_close" not in working.columns:
        working["spread_close"] = working["ask_close"] - working["bid_close"]
    _validate_m5_projection(working)

    if not (working["granularity"] == GRANULARITY).all():
        raise DataProjectionError("M15 aggregation accepts only M5 input")
    if not (working["price"] == PRICE).all():
        raise DataProjectionError("M15 aggregation accepts only BA input")
    if not (working["complete"] == 1).all():
        raise DataProjectionError("M15 aggregation accepts only complete M5 input")
    if (
        (working["ts_utc"] < start_inclusive)
        | (working["ts_utc"] >= end_exclusive)
    ).any():
        raise DataProjectionError("M5 input contains rows outside the extraction range")

    working["bucket_ts_utc"] = working["ts_utc"] - (
        working["ts_utc"] % 900
    )
    key_columns = ["source", "instrument", "price", "bucket_ts_utc"]
    working = working.sort_values([*key_columns, "ts_utc"], kind="stable")
    groups = working.groupby(key_columns, sort=True, dropna=False)
    stats = groups["ts_utc"].agg(
        component_count="size",
        unique_count="nunique",
        first_ts="min",
        last_ts="max",
    ).reset_index()
    complete_mask = (
        (stats["component_count"] == 3)
        & (stats["unique_count"] == 3)
        & (stats["first_ts"] == stats["bucket_ts_utc"])
        & (stats["last_ts"] == stats["bucket_ts_utc"] + 600)
    )
    complete_keys = stats.loc[complete_mask, key_columns]
    complete_rows = working.merge(complete_keys, on=key_columns, how="inner")

    if complete_rows.empty:
        candles = pd.DataFrame(columns=_M15_COLUMNS)
    else:
        candles = (
            complete_rows.groupby(key_columns, sort=True, as_index=False)
            .agg(
                fetched_at_utc=("fetched_at_utc", "max"),
                volume=("volume", "sum"),
                bid_open=("bid_open", "first"),
                bid_high=("bid_high", "max"),
                bid_low=("bid_low", "min"),
                bid_close=("bid_close", "last"),
                ask_open=("ask_open", "first"),
                ask_high=("ask_high", "max"),
                ask_low=("ask_low", "min"),
                ask_close=("ask_close", "last"),
            )
            .rename(columns={"bucket_ts_utc": "ts_utc"})
        )
        candles.insert(2, "granularity", "M15")
        candles.insert(6, "complete", 1)
        candles.insert(7, "component_count", 3)
        candles["volume"] = candles["volume"].astype("int64")
        candles["ts_utc"] = candles["ts_utc"].astype("int64")
        candles["fetched_at_utc"] = candles["fetched_at_utc"].astype("int64")
        candles["ts_utc_dt"] = pd.to_datetime(
            candles["ts_utc"], unit="s", utc=True
        )
        candles["ts_et"] = candles["ts_utc_dt"].dt.tz_convert(
            "America/New_York"
        )
        candles["session_date_et"] = candles["ts_et"].dt.date
        candles["spread_open"] = candles["ask_open"] - candles["bid_open"]
        candles["spread_close"] = candles["ask_close"] - candles["bid_close"]
        candles = candles[list(_M15_COLUMNS)]

    complete_key_set = {
        tuple(row)
        for row in complete_keys.itertuples(index=False, name=None)
    }
    incomplete_records: list[dict[str, object]] = []
    for key, group in groups:
        normalized_key = key if isinstance(key, tuple) else (key,)
        if normalized_key in complete_key_set:
            continue
        source, instrument, price, bucket_value = normalized_key
        bucket_ts = int(bucket_value)
        present = tuple(sorted(int(value) for value in group["ts_utc"].unique()))
        expected = (bucket_ts, bucket_ts + 300, bucket_ts + 600)
        missing = tuple(value for value in expected if value not in present)
        boundary_missing = any(
            value < start_inclusive or value >= end_exclusive for value in missing
        )
        bucket_utc = pd.to_datetime(bucket_ts, unit="s", utc=True)
        incomplete_records.append(
            {
                "source": str(source),
                "instrument": str(instrument),
                "price": str(price),
                "bucket_ts_utc": bucket_ts,
                "bucket_utc": bucket_utc,
                "bucket_et": bucket_utc.tz_convert("America/New_York"),
                "component_count": len(present),
                "present_ts_utc": present,
                "expected_ts_utc": expected,
                "missing_ts_utc": missing,
                "reason": (
                    "extraction_boundary" if boundary_missing else "missing_m5"
                ),
            }
        )
    incomplete = pd.DataFrame.from_records(
        incomplete_records,
        columns=_INCOMPLETE_BUCKET_COLUMNS,
    )
    return M15Aggregation(candles=candles, incomplete_buckets=incomplete)


def select_m15_et_time(
    candles: pd.DataFrame,
    *,
    hour: int,
    minute: int,
) -> pd.DataFrame:
    """Select M15 starts by New York wall-clock time without changing UTC."""

    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("hour and minute are outside the clock range")
    if "ts_et" not in candles.columns:
        raise DataProjectionError("M15 frame is missing the ts_et label")
    if candles.empty:
        return candles.copy()
    selected = candles[
        (candles["ts_et"].dt.hour == hour)
        & (candles["ts_et"].dt.minute == minute)
    ]
    return selected.copy()


def file_sha256(path: Path) -> str:
    """Hash one stable file and fail if it changes while being read."""

    resolved = path.expanduser().resolve()
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise MinuteDataError("SQLite file changed while its SHA-256 was computed")
    return digest.hexdigest()


def build_evidence_manifest(
    *,
    db_path: Path,
    environment: str,
    instrument: str,
    start_inclusive: int,
    end_exclusive: int,
    m5_audit: M5Audit,
    m15: M15Aggregation,
    code_commit: str,
    command: str,
    generated_at_utc: str | None = None,
    ingestion: IngestionSummary | None = None,
) -> AuditManifest:
    """Build the row-free Stage-6 evidence manifest for one fixed extraction."""

    if environment != "practice":
        raise ValueError("environment must be 'practice'")
    validate_instrument(instrument)
    if not code_commit or "/" in code_commit or "\\" in code_commit:
        raise ValueError("code_commit must be a commit id, optionally suffixed -dirty")
    if TOKEN_ENV_VAR in command or "Bearer " in command:
        raise ValueError("command must not mention credentials")
    if re.search(r"(?:^|[\s\"'=])/(?!/)", command):
        raise ValueError("command must not contain an absolute path")

    resolved = db_path.expanduser().resolve()
    connection = open_read_only_database(resolved)
    try:
        quick_check(connection)
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()

    if ingestion is not None:
        if (
            ingestion.instrument != instrument
            or ingestion.start_inclusive != start_inclusive
            or ingestion.end_exclusive != end_exclusive
        ):
            raise ValueError("ingestion summary does not match manifest extraction")
        request_count: int | None = ingestion.request_count
        complete_received: int | None = ingestion.complete_received
        incomplete_skipped: int | None = ingestion.incomplete_skipped
        counters_scope = "same_run"
    else:
        request_count = None
        complete_received = None
        incomplete_skipped = None
        counters_scope = "not_recorded_for_upstream_snapshot"

    reason_counts = m15.incomplete_buckets.get(
        "reason", pd.Series(dtype="object")
    ).value_counts()
    component_counts = m15.incomplete_buckets.get(
        "component_count", pd.Series(dtype="int64")
    ).value_counts()
    first_m15 = (
        None
        if m15.candles.empty
        else _manifest_timestamp(int(m15.candles["ts_utc"].min()))
    )
    last_m15 = (
        None
        if m15.candles.empty
        else _manifest_timestamp(int(m15.candles["ts_utc"].max()))
    )
    generated = generated_at_utc or _manifest_timestamp(
        int(dt.datetime.now(tz=dt.UTC).timestamp())
    )

    return {
        "schema_version": 1,
        "generated_at_utc": generated,
        "code_commit": code_commit,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "requests_version": requests.__version__,
        "source": SOURCE,
        "environment": environment,
        "instrument": instrument,
        "granularity": GRANULARITY,
        "price": PRICE,
        "range": {
            "start_inclusive": _manifest_timestamp(start_inclusive),
            "end_exclusive": _manifest_timestamp(end_exclusive),
        },
        "db": {
            "role": (
                "article_generated" if ingestion is not None else "upstream_read_only_source"
            ),
            "basename": resolved.name,
            "sha256": file_sha256(resolved),
            "quick_check": "ok",
            "user_version": user_version,
        },
        "fetch": {
            "policy": "article_reproduction_code",
            "chunk_days": CHUNK_DAYS,
            "request_count": request_count,
            "complete_received": complete_received,
            "incomplete_skipped": incomplete_skipped,
            "counters_scope": counters_scope,
        },
        "m5": {
            "row_count": m5_audit.row_count,
            "first_ts_utc": _manifest_timestamp(m5_audit.first_ts_utc),
            "last_ts_utc": _manifest_timestamp(m5_audit.last_ts_utc),
            "duplicate_count": m5_audit.duplicate_count,
            "off_boundary_count": m5_audit.off_boundary_count,
            "null_required_count": m5_audit.null_required_count,
            "invalid_volume_count": m5_audit.invalid_volume_count,
            "invalid_ohlc_count": m5_audit.invalid_ohlc_count,
            "negative_spread_count": m5_audit.negative_spread_count,
            "sorted_ascending": m5_audit.sorted_ascending,
            "gap_count": m5_audit.gap_count,
            "missing_m5_slots": m5_audit.missing_m5_slots,
            "gap_classification": {
                "weekend_closure_candidate": m5_audit.weekend_gap_count,
                "long_non_weekend_closure_candidate": (
                    m5_audit.long_non_weekend_gap_count
                ),
                "short_unclassified": m5_audit.short_gap_count,
            },
            "extraction_sha256": m5_audit.extraction_sha256,
        },
        "m15": {
            "complete_bucket_count": len(m15.candles),
            "incomplete_bucket_count": len(m15.incomplete_buckets),
            "incomplete_reason": {
                "missing_m5": int(reason_counts.get("missing_m5", 0)),
                "extraction_boundary": int(
                    reason_counts.get("extraction_boundary", 0)
                ),
            },
            "incomplete_component_count": {
                "one": int(component_counts.get(1, 0)),
                "two": int(component_counts.get(2, 0)),
            },
            "first_ts_utc": first_m15,
            "last_ts_utc": last_m15,
            "m5_rows_in_complete_buckets": len(m15.candles) * 3,
            "m5_rows_in_incomplete_buckets": int(
                m15.incomplete_buckets.get(
                    "component_count", pd.Series(dtype="int64")
                ).sum()
            ),
            "zero_row_bucket_note": (
                "Buckets with no M5 rows are represented only by the M5 gap audit."
            ),
        },
        "derivation": {
            "m15_bucket": "UTC floor(ts_utc / 900) * 900",
            "required_m5_offsets_seconds": [0, 300, 600],
            "ohlc": "open=first, high=max, low=min, close=last; bid/ask separate",
            "volume": "sum of three complete M5 candles",
            "missing_policy": "do not fill; reject incomplete M15 buckets",
            "timezone": "UTC authoritative; America/New_York labels derived",
            "information_loss": (
                "M5 component OHLC, intra-M5 high/low order, price path, and "
                "authoritative mid high/low are not recoverable from M15"
            ),
        },
        "command": command,
    }


def write_evidence_manifest(path: Path, manifest: AuditManifest) -> None:
    """Write a small UTF-8 JSON manifest without row-level candle data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def git_code_commit(repo_root: Path) -> str:
    """Return HEAD and append ``-dirty`` when tracked or untracked files differ."""

    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{commit}-dirty" if dirty else commit


def _manifest_timestamp(timestamp: int) -> str:
    value = dt.datetime.fromtimestamp(timestamp, tz=dt.UTC)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")
