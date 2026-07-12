"""yfinance-backed DataProvider implementation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from .cache import REQUIRED_COLUMNS, ParquetCache
from .errors import (
    DataNotFoundError,
    PeriodLimitExceededError,
)
from .provider import DataProvider
from .symbols import (
    SUPPORTED_PAIRS,
    YF_INTERVAL_LIMITS,
    Interval,
    bar_duration,
    parse_interval,
    to_yfinance_interval,
    to_yfinance_symbol,
    validate_pair,
)


class YfinanceProvider(DataProvider):
    """yfinance をバックエンドとする DataProvider.

    キャッシュ有効時は `ParquetCache` と協調して差分更新を行う.
    """

    _NAME = "yfinance"

    def __init__(
        self,
        cache_root: Path | str | None = None,
        downloader: Any = None,
    ) -> None:
        """
        Args:
            cache_root: キャッシュのルートディレクトリ. None ならキャッシュ無効.
            downloader: `yf.download` と同シグネチャのテスト用インジェクト. 省略で本物.
        """
        self._cache: ParquetCache | None = (
            ParquetCache(Path(cache_root), self._NAME) if cache_root is not None else None
        )
        self._download = downloader if downloader is not None else yf.download

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------
    def name(self) -> str:
        return self._NAME

    def supported_pairs(self) -> list[str]:
        return list(SUPPORTED_PAIRS)

    def fetch_bars(
        self,
        pair: str,
        interval: str | Interval,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        validate_pair(pair)
        iv = parse_interval(interval)

        now = datetime.now(tz=timezone.utc)
        end_eff = end if end is not None else now

        _require_utc_aware("start", start)
        if end is not None:
            _require_utc_aware("end", end)
        if start >= end_eff:
            raise ValueError(f"start must be < end. start={start}, end={end_eff}")

        _check_period_limit(iv, start, now)

        # --- キャッシュあり: 差分取得 ---
        if self._cache is not None:
            cached = self._cache.load(pair, iv)
            gaps = ParquetCache.missing_ranges(cached, start, end_eff, bar_duration(iv))
            for gap_start, gap_end in gaps:
                new_df = self._download_bars(pair, iv, gap_start, gap_end)
                if not new_df.empty:
                    cached = self._cache.merge(pair, iv, new_df)
            result = cached
        else:
            result = self._download_bars(pair, iv, start, end_eff)

        # 要求範囲でスライス（end 排他的）
        sliced = result[(result.index >= start) & (result.index < end_eff)]
        if sliced.empty:
            raise DataNotFoundError(
                _empty_message(pair, iv, start, end_eff)
            )
        return sliced

    # ------------------------------------------------------------------
    # Internal: download + normalize
    # ------------------------------------------------------------------
    def _download_bars(
        self,
        pair: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """1 回の yfinance 取得. 必要なら 4h リサンプルへフォールバックする（§6.4）."""
        if interval == Interval.H4 and _requires_h4_fallback(start):
            raw = self._raw_download(pair, Interval.H1, start, end)
            if raw.empty:
                return raw
            return _resample_to_4h(raw)

        return self._raw_download(pair, interval, start, end)

    def _raw_download(
        self,
        pair: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        symbol = to_yfinance_symbol(pair)
        yf_interval = to_yfinance_interval(interval)
        df = self._download(
            symbol,
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=False,
            progress=False,
        )
        if df is None or df.empty:
            return _empty_normalized()
        return _normalize_yfinance_frame(df)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_utc_aware(label: str, ts: datetime) -> None:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware (UTC). got: {ts!r}")


def _check_period_limit(
    interval: Interval,
    start: datetime,
    now: datetime,
) -> None:
    """履歴上限（§6.3）の事前バリデーション.

    4h は §6.4 フォールバックで実質 730 日超も 1h 経由で扱えるため、
    730 日超でも例外にしない（内部で 1h にフォールバック）.
    """
    limit = YF_INTERVAL_LIMITS[interval].max_lookback_days
    if limit is None:
        return
    if interval == Interval.H4:
        # フォールバックで無制限扱い（1h の 730 日上限は別途チェック）
        return
    oldest_allowed = now - timedelta(days=limit)
    if start < oldest_allowed:
        raise PeriodLimitExceededError(
            f"interval={interval.value} supports only last {limit} days, "
            f"but start={start.isoformat()} exceeds that (now={now.isoformat()})."
        )


def _requires_h4_fallback(start: datetime) -> bool:
    """4h 要求が 1h の履歴上限（730 日）を跨ぐかどうか."""
    limit = YF_INTERVAL_LIMITS[Interval.H1].max_lookback_days
    if limit is None:
        return False
    return start < datetime.now(tz=timezone.utc) - timedelta(days=limit - 1)


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """1h → 4h リサンプル（§6.4, label='left', closed='left' 必須）."""
    agg = df_1h.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna(subset=["open", "high", "low", "close"], how="all")


def _normalize_yfinance_frame(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance の生 DF を共通スキーマ（§4）に変換.

    - MultiIndex columns を flatten
    - Adj Close を削除
    - カラム名を lower-case
    - Volume=0 を NaN に変換
    - index を UTC tz-aware / name='ts' に正規化
    """
    out = df.copy()

    # MultiIndex flatten（§10.1）
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    # Adj Close 削除（§10.3）
    if "Adj Close" in out.columns:
        out = out.drop(columns=["Adj Close"])

    # カラム lower-case
    rename_map = {c: c.lower() for c in out.columns if isinstance(c, str)}
    out = out.rename(columns=rename_map)

    # Volume=0 → NaN（§10.3）
    if "volume" in out.columns:
        out["volume"] = out["volume"].astype("float64").where(out["volume"] != 0, other=float("nan"))

    # 必須カラム補完
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")

    # index 正規化（§10.2）
    idx = out.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx.name = "ts"
    out.index = idx

    out = out[~out.index.duplicated(keep="last")].sort_index()
    # 列順を揃える（必須を先頭へ）
    ordered = list(REQUIRED_COLUMNS) + [c for c in out.columns if c not in REQUIRED_COLUMNS]
    return out[ordered]


def _empty_normalized() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in REQUIRED_COLUMNS}, index=idx)


def _empty_message(pair: str, interval: Interval, start: datetime, end: datetime) -> str:
    """§7.2 ヒント付き例外メッセージ."""
    hint = ""
    if _is_weekend_or_holiday_only(start, end):
        hint = " (Maybe weekend/holiday?)"
    return (
        f"No data for pair={pair}, interval={interval.value}, "
        f"start={start.isoformat()}, end={end.isoformat()}.{hint}"
    )


def _is_weekend_or_holiday_only(start: datetime, end: datetime) -> bool:
    """要求範囲に平日 UTC が 1 つも含まれないかをざっくり判定.

    厳密な FX 市場カレンダ判定はしない. UTC 曜日のみ見る.
    """
    if start >= end:
        return False
    cur = start
    step = timedelta(hours=1)
    max_iter = 24 * 8  # 最大 8 日分だけ覗く（無限ループ対策）
    for _ in range(max_iter):
        if cur >= end:
            break
        if cur.weekday() < 5:  # Mon-Fri
            return False
        cur += step
    return True
