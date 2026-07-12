"""DataProvider abstraction used by the reproduction scripts.

公開 API:
    from bocchi_the_botter_repro.common.data import (
        DataProvider, YfinanceProvider,
        Interval,
        DataProviderError, DataNotFoundError,
        InvalidIntervalError, InvalidPairError, PeriodLimitExceededError,
    )
"""
from __future__ import annotations

from .errors import (
    DataNotFoundError,
    DataProviderError,
    InvalidIntervalError,
    InvalidPairError,
    PeriodLimitExceededError,
)
from .provider import DataProvider
from .symbols import SUPPORTED_PAIRS, Interval
from .yfinance_provider import YfinanceProvider

__all__ = [
    "DataProvider",
    "YfinanceProvider",
    "Interval",
    "SUPPORTED_PAIRS",
    "DataProviderError",
    "DataNotFoundError",
    "InvalidIntervalError",
    "InvalidPairError",
    "PeriodLimitExceededError",
]
