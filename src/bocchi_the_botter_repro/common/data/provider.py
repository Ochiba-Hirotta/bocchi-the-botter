"""DataProvider abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from .symbols import Interval


class DataProvider(ABC):
    """FX データ取得の抽象インターフェース."""

    @abstractmethod
    def fetch_bars(
        self,
        pair: str,
        interval: str | Interval,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """共通スキーマ（§4）の DataFrame を返す.

        Args:
            pair: 内部正規化キー（例: "USDJPY"）.
            interval: 内部 interval 表記（§6.1）. 文字列 or `Interval`.
            start: 取得開始時刻（UTC tz-aware 必須）.
            end: 取得終了時刻（UTC tz-aware, 排他的）. None なら最新まで.

        Returns:
            UTC tz-aware DatetimeIndex（name='ts'）を持つ DataFrame.
            カラムは `open, high, low, close, volume`.

        Raises:
            DataNotFoundError: データソースが空結果を返した.
            InvalidIntervalError: interval がサポート外.
            InvalidPairError: ペア表記がサポート外.
            PeriodLimitExceededError: interval の履歴上限を超える要求.
        """

    @abstractmethod
    def supported_pairs(self) -> list[str]:
        """このプロバイダが取得可能な通貨ペア（正規化キー）の一覧."""

    @abstractmethod
    def name(self) -> str:
        """プロバイダ識別子（例: "yfinance", "oanda"）."""
