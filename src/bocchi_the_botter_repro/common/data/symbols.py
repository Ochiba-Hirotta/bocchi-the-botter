"""Currency-pair and interval normalization helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from .errors import InvalidIntervalError, InvalidPairError


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

class Interval(str, Enum):
    """内部 interval 表記（§6.1）."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1wk"


@dataclass(frozen=True)
class IntervalLimit:
    """interval ごとの履歴上限情報（yfinance 基準、§6.3）.

    max_lookback_days=None は「事実上無制限」を意味する.
    """

    max_lookback_days: int | None


# yfinance 基準の履歴上限テーブル（§6.3）
YF_INTERVAL_LIMITS: dict[Interval, IntervalLimit] = {
    Interval.M1: IntervalLimit(max_lookback_days=7),
    Interval.M5: IntervalLimit(max_lookback_days=60),
    Interval.M15: IntervalLimit(max_lookback_days=60),
    Interval.M30: IntervalLimit(max_lookback_days=60),
    Interval.H1: IntervalLimit(max_lookback_days=730),
    Interval.H4: IntervalLimit(max_lookback_days=730),  # フォールバック境界
    Interval.D1: IntervalLimit(max_lookback_days=None),
    Interval.W1: IntervalLimit(max_lookback_days=None),
}


def parse_interval(value: str | Interval) -> Interval:
    """文字列 → Interval. 不正なら InvalidIntervalError."""
    if isinstance(value, Interval):
        return value
    try:
        return Interval(value)
    except ValueError as exc:
        supported = [i.value for i in Interval]
        raise InvalidIntervalError(
            f"Unsupported interval: {value!r}. Supported: {supported}"
        ) from exc


# yfinance は 1h 以下を `{N}m` / `{N}h` 形式、日足以上を `1d` / `1wk` で受ける.
# 4h は yfinance ネイティブ対応（PoC #1 で確認済み）だが、730 日超の要求時は
# 1h からリサンプル（§6.4）するため、yf 呼び出し用文字列と内部表記を分離.
_YF_INTERVAL_MAP: dict[Interval, str] = {
    Interval.M1: "1m",
    Interval.M5: "5m",
    Interval.M15: "15m",
    Interval.M30: "30m",
    Interval.H1: "1h",
    Interval.H4: "4h",
    Interval.D1: "1d",
    Interval.W1: "1wk",
}


def to_yfinance_interval(interval: Interval) -> str:
    """内部 Interval → yfinance interval 文字列."""
    return _YF_INTERVAL_MAP[interval]


# interval ごとの 1 バー長（キャッシュのギャップ計算に利用）.
_BAR_DURATION: dict[Interval, timedelta] = {
    Interval.M1: timedelta(minutes=1),
    Interval.M5: timedelta(minutes=5),
    Interval.M15: timedelta(minutes=15),
    Interval.M30: timedelta(minutes=30),
    Interval.H1: timedelta(hours=1),
    Interval.H4: timedelta(hours=4),
    Interval.D1: timedelta(days=1),
    Interval.W1: timedelta(weeks=1),
}


def bar_duration(interval: Interval) -> timedelta:
    """interval が表す 1 バーの長さ."""
    return _BAR_DURATION[interval]


# ---------------------------------------------------------------------------
# Currency pair
# ---------------------------------------------------------------------------

_PAIR_RE = re.compile(r"^[A-Z]{6}$")


def validate_pair(pair: str) -> str:
    """ペア表記を検証し、正規化キー（6文字大文字）を返す.

    Raises:
        InvalidPairError: 6文字大文字 ASCII 以外
    """
    if not isinstance(pair, str) or not _PAIR_RE.match(pair):
        raise InvalidPairError(
            f"Pair must be 6 uppercase ASCII letters (e.g. 'USDJPY'), got: {pair!r}"
        )
    return pair


def to_yfinance_symbol(pair: str) -> str:
    """内部キー → yfinance シンボル（§5.2）. 例: 'USDJPY' → 'USDJPY=X'."""
    return f"{validate_pair(pair)}=X"


def from_yfinance_symbol(sym: str) -> str:
    """yfinance シンボル → 内部キー. 例: 'USDJPY=X' → 'USDJPY'."""
    if not sym.endswith("=X"):
        raise InvalidPairError(f"Not a yfinance FX symbol: {sym!r}")
    return validate_pair(sym[:-2])


# Article reproduction targets.
SUPPORTED_PAIRS: tuple[str, ...] = (
    "USDJPY",
    "GBPJPY",
    "EURJPY",
    "AUDJPY",
)
