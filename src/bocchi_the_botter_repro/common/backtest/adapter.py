"""Convert DataProvider frames into backtesting.py input frames."""
from __future__ import annotations

import pandas as pd


class DataQualityError(Exception):
    """Adapter がバー品質要件を満たさないと判断した場合."""


_REQUIRED_COLUMNS = ("open", "high", "low", "close")
_COLUMN_RENAME = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def to_backtest_frame(
    df: pd.DataFrame,
    volume_fill: float = 1.0,
    expected_freq: str | None = None,
    max_missing_ratio: float = 0.01,
) -> pd.DataFrame:
    """DataProvider 共通スキーマの DF を backtesting.py 入力形式へ変換.

    処理内容（§4.1 / §4.5）:
        1. 必須カラム (`open, high, low, close`) の存在確認.
        2. カラム名を Capitalize（`Open/High/Low/Close/Volume`）.
        3. Volume 欠如なら NaN 追加、NaN を ``volume_fill`` で埋める（§4.2）.
        4. ``expected_freq`` が指定された場合のみバー連続性チェック（§4.5）.
        5. tz-aware index をそのまま維持.

    Args:
        df: DataProvider の共通スキーマ DataFrame（小文字カラム, UTC tz-aware index）.
        volume_fill: Volume の NaN を埋める値. デフォルトは ``1.0``（§4.2 参照）.
        expected_freq: 期待するバー間隔（例: ``"1h"``）. ``None`` なら連続性チェックを省略.
        max_missing_ratio: 欠損バー比率の許容上限. デフォルト 1%.

    Returns:
        backtesting.py 入力形式の DataFrame（Capitalized カラム, tz-aware index 維持）.

    Raises:
        KeyError: 必須カラムが欠如している場合.
        DataQualityError: 空 DF、または欠損率が ``max_missing_ratio`` を超える場合.
    """
    if len(df) == 0:
        raise DataQualityError("入力 DataFrame が空です")

    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"必須カラムが欠如しています: {missing_cols}")

    if expected_freq is not None:
        _check_bar_continuity(df.index, expected_freq, max_missing_ratio)

    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = float("nan")
    out = out.rename(columns=_COLUMN_RENAME)

    out["Volume"] = out["Volume"].fillna(volume_fill)

    return out[["Open", "High", "Low", "Close", "Volume"]]


def _check_bar_continuity(
    index: pd.Index,
    expected_freq: str,
    max_missing_ratio: float,
) -> None:
    """バー連続性チェック（§4.5）.

    `start`/`end` の span から期待バー数を算出し、実バー数との差分を欠損として扱う.
    欠損率が閾値を超える場合は ``DataQualityError``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise DataQualityError(
            f"index が DatetimeIndex ではありません: {type(index).__name__}"
        )

    freq_td = pd.Timedelta(expected_freq)
    span = index.max() - index.min()
    expected_total = int(span / freq_td) + 1

    missing = max(0, expected_total - len(index))
    if expected_total > 0 and missing / expected_total > max_missing_ratio:
        raise DataQualityError(
            f"バー欠損率が閾値を超えました: "
            f"missing={missing}, expected={expected_total}, "
            f"ratio={missing / expected_total:.4f} > {max_missing_ratio}"
        )
