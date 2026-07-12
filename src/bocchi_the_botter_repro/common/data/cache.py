"""Parquet-based local cache for downloaded market data."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .symbols import Interval, validate_pair


# 共通スキーマの必須カラム（§4.2）.
REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class ParquetCache:
    """`(provider_name, pair, interval)` 単位で 1 Parquet ファイルを管理する.

    ファイルパス: `{root}/{provider_name}/{pair}_{interval}.parquet` (§8.3)
    ファイル内スキーマ: 共通スキーマ（§4）そのまま.
    index は UTC tz-aware / name="ts" / 昇順ソート済み / 重複なし（§8.5）.
    """

    def __init__(self, root: Path, provider_name: str) -> None:
        self._root = Path(root)
        self._provider_name = provider_name

    # ------------------------------------------------------------------
    # Path
    # ------------------------------------------------------------------
    def path_for(self, pair: str, interval: Interval) -> Path:
        validate_pair(pair)
        return self._root / self._provider_name / f"{pair}_{interval.value}.parquet"

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------
    def load(self, pair: str, interval: Interval) -> pd.DataFrame:
        """既存キャッシュを読む. 無ければ空の共通スキーマ DF を返す."""
        path = self.path_for(pair, interval)
        if not path.exists():
            return _empty_frame()
        df = pd.read_parquet(path)
        return _normalize(df)

    def save(self, pair: str, interval: Interval, df: pd.DataFrame) -> None:
        """DF を共通スキーマとして保存（§8.5 整合性を担保）."""
        path = self.path_for(pair, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        _normalize(df).to_parquet(path)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def merge(
        self,
        pair: str,
        interval: Interval,
        new_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """既存キャッシュと new_df を統合し、保存した上でマージ後の DF を返す.

        マージ規則（§8.5）:
        - index 昇順ソート
        - index 重複は「後勝ち」で new_df 側を採用（最新再取得の優先）
        """
        existing = self.load(pair, interval)
        combined = pd.concat([existing, new_df])
        # 後勝ちのため keep='last'. concat 順で new_df が後ろにあることが前提.
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        self.save(pair, interval, combined)
        return combined

    # ------------------------------------------------------------------
    # Gap analysis (§9.1)
    # ------------------------------------------------------------------
    @staticmethod
    def missing_ranges(
        cached: pd.DataFrame,
        start: datetime,
        end: datetime,
        bar_duration: timedelta,
    ) -> list[tuple[datetime, datetime]]:
        """要求範囲 [start, end) のうち、キャッシュで満たせない部分を返す.

        本実装では「キャッシュ範囲外の前方/後方」のみ扱い、内部の穴埋め（hole
        filling）はしない.

        Args:
            cached: 共通スキーマの DataFrame（空可）.
            start, end: 要求範囲（end 排他的）.
            bar_duration: interval 1 本あたりの長さ. キャッシュ末尾バー
                `cached_end` は `[cached_end, cached_end + bar_duration)` を
                カバーすると解釈する.

        Returns:
            `[(gap_start, gap_end), ...]` のリスト（end 排他的）.
            欠損無しなら空リスト.
        """
        if start >= end:
            return []
        if cached.empty:
            return [(start, end)]

        cached_start = cached.index.min().to_pydatetime()
        cached_end_exclusive = cached.index.max().to_pydatetime() + bar_duration

        gaps: list[tuple[datetime, datetime]] = []
        if start < cached_start:
            gaps.append((start, min(cached_start, end)))
        if end > cached_end_exclusive:
            gap_from = max(cached_end_exclusive, start)
            if gap_from < end:
                gaps.append((gap_from, end))
        return gaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in REQUIRED_COLUMNS},
        index=idx,
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """index を UTC tz-aware / name="ts" / 昇順ソート / 重複除去に整える."""
    if df.empty:
        return _empty_frame()

    out = df.copy()
    idx = out.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx.name = "ts"
    out.index = idx
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    # 必須カラムが全て揃っているかだけ保証する（欠けていたら NaN で補う）.
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    return out[list(REQUIRED_COLUMNS) + [c for c in out.columns if c not in REQUIRED_COLUMNS]]
