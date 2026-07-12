"""DataProvider layer exceptions."""
from __future__ import annotations


class DataProviderError(Exception):
    """Provider 層の基底例外."""


class DataNotFoundError(DataProviderError):
    """データソースが空結果を返した（存在しないペア、未来日、土日のみの範囲など）."""


class InvalidIntervalError(DataProviderError):
    """interval がサポート外."""


class InvalidPairError(DataProviderError):
    """ペア表記がサポート外."""


class PeriodLimitExceededError(DataProviderError):
    """interval の履歴上限を超える要求."""
