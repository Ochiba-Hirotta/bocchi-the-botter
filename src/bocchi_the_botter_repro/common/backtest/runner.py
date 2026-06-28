"""Thin FX-specific wrapper around backtesting.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
from backtesting import Backtest


_TRACKED_PACKAGES = ("backtesting", "yfinance", "pandas", "numpy", "pyarrow")


@dataclass(frozen=True)
class ExecutionMeta:
    """バックテスト実行のメタ情報（§11.7 再現性ログ）.

    記事末尾・smoke script の出力・テスト fixture で参照する再現用の情報を集約する.
    """

    executed_at_utc: datetime
    period_start_utc: datetime
    period_end_utc: datetime
    effective_bars: int
    package_versions: dict[str, str]
    spread: float
    spread_source_note: str = ""
    data_fetched_at_utc: datetime | None = None
    missing_bars: int = 0
    extra: dict[str, object] = field(default_factory=dict)


class FXBacktestRunner:
    """§5.1 の FX デフォルトパラメータを埋め込んだ Backtest ラッパ.

    `finalize_trades=True` を**明示的に**指定している（backtesting.py の既定は ``False``）.
    記事時点の再現条件に合わせている.
    """

    def __init__(
        self,
        cash: float = 1_000_000,
        commission: float = 0.0,
        margin: float = 0.04,
        trade_on_close: bool = False,
        exclusive_orders: bool = True,
        hedging: bool = False,
        finalize_trades: bool = True,
    ) -> None:
        self.cash = cash
        self.commission = commission
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.exclusive_orders = exclusive_orders
        self.hedging = hedging
        self.finalize_trades = finalize_trades

    def run(
        self,
        df: pd.DataFrame,
        strategy_cls: type,
        spread: float,
        *,
        spread_source_note: str = "",
        data_fetched_at_utc: datetime | None = None,
        missing_bars: int = 0,
    ) -> tuple[pd.Series, ExecutionMeta]:
        """バックテストを実行し、stats と ExecutionMeta を返す.

        Args:
            df: ``to_backtest_frame()`` で変換済みの DataFrame
                （Capitalized カラム, UTC tz-aware index）.
            strategy_cls: ``backtesting.Strategy`` のサブクラス.
            spread: §5.2 で算出した相対値スプレッド.
            spread_source_note: スプレッド値の採番根拠（例: 出典 URL と閲覧日時）.
            data_fetched_at_utc: データ取得時刻（呼出元から渡す. DataProvider に依存させない）.
            missing_bars: adapter から受け取る欠損バー数.

        Returns:
            (stats, ExecutionMeta). ``stats`` は backtesting.py の ``Backtest.run()`` 戻り値.

        Raises:
            ValueError: ``strategy_cls.MARGIN`` が定義されていて ``self.margin`` と
                一致しない場合. Strategy 側の ``units_margin_cap`` 計算と Broker 側
                の余力判定が食い違うと境界ケースで無警告キャンセルが起きるため.
        """
        strategy_margin = getattr(strategy_cls, "MARGIN", None)
        if strategy_margin is not None and float(strategy_margin) != float(self.margin):
            raise ValueError(
                f"Strategy.MARGIN={strategy_margin} と Runner.margin={self.margin} "
                f"が不一致です. units_margin_cap の前提が崩れるため実行を中止しました."
            )

        bt = Backtest(
            df,
            strategy_cls,
            cash=self.cash,
            commission=self.commission,
            spread=spread,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            exclusive_orders=self.exclusive_orders,
            hedging=self.hedging,
            finalize_trades=self.finalize_trades,
        )
        stats = bt.run()

        meta = ExecutionMeta(
            executed_at_utc=datetime.now(tz=timezone.utc),
            period_start_utc=df.index.min().to_pydatetime(),
            period_end_utc=df.index.max().to_pydatetime(),
            effective_bars=len(df),
            package_versions=_collect_package_versions(),
            spread=spread,
            spread_source_note=spread_source_note,
            data_fetched_at_utc=data_fetched_at_utc,
            missing_bars=missing_bars,
        )
        return stats, meta


def _collect_package_versions() -> dict[str, str]:
    """§11.7 の再現ログ用に、主要依存のバージョンを収集する."""
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions
