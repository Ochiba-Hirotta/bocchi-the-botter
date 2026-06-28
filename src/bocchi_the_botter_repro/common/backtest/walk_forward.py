"""WalkForwardRunner: time-series Walk-Forward Analysis.

責務:
    - 時系列 DataFrame を rolling fold (train + test) に分割
    - 各 fold × 各 parameter grid 点でバックテスト実行
    - fold メタデータ（レジーム等）の記録
    - 結果を集計可能な形（``WFAResult`` / ``to_dataframe()``）で返却

設計上の制約:
    - ``base_strategy`` の ``MARGIN`` は ``param_grid`` で上書きしないこと.
      ``FXBacktestRunner.run()`` が strategy/runner 間の margin 一致を検証するため,
      MARGIN を grid に入れると ValueError. BB-MR では BB_N / BB_K のみを動かす想定.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Strategy

from .indicators.atr import wilder_atr
from .runner import FXBacktestRunner


# backtesting.py stats から抽出する主要メトリクス.
# Main and supporting metrics extracted from backtesting.py stats.
METRIC_KEYS: tuple[str, ...] = (
    "Sharpe Ratio",
    "Return [%]",
    "Return (Ann.) [%]",
    "Max. Drawdown [%]",
    "Win Rate [%]",
    "Profit Factor",
    "# Trades",
)

# Strategy._close_reasons から計算する close 理由比率.
# Close-ratio metrics added for the strategy comparison chapters.
# sl / finalize は backtesting.py 自動判定 / runner finalize_trades=True のため
# Strategy 側の _record_close_reason から呼べず、ここでは集計しない (TODO-2).
CLOSE_RATIO_KEYS: tuple[str, ...] = (
    "max_bars_close_ratio",
    "signal_close_ratio",
)


@dataclass(frozen=True)
class FoldSpec:
    """1 fold の時間窓仕様（train_end / test_end は排他境界）."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class FoldMetadata:
    """1 fold の時間窓・レジーム情報.

    test_regime は test 区間の close リターンで判定:
        |return| < ``regime_threshold`` → "flat"
        return > 0 → "up"
        return < 0 → "down"
    """

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_bars: int
    test_bars: int
    test_close_return: float
    test_regime: str
    test_atr_median_normalized: float
    test_close_range_pct: float


@dataclass(frozen=True)
class WFAFoldResult:
    """1 fold × 1 grid 点の評価結果."""

    fold_index: int
    params: dict[str, Any]
    is_metrics: dict[str, float]
    oos_metrics: dict[str, float]


@dataclass(frozen=True)
class WFAResult:
    """全 fold × 全 grid の集計結果."""

    folds_metadata: list[FoldMetadata]
    results: list[WFAFoldResult]
    failures: list[dict[str, Any]]

    def to_dataframe(self) -> pd.DataFrame:
        """fold × params × IS/OOS metrics を long-format DataFrame に変換."""
        meta_by_fold = {m.fold_index: m for m in self.folds_metadata}
        rows: list[dict[str, Any]] = []
        for r in self.results:
            row: dict[str, Any] = {"fold": r.fold_index}
            row.update({f"param_{k}": v for k, v in r.params.items()})
            row.update({f"is_{k}": v for k, v in r.is_metrics.items()})
            row.update({f"oos_{k}": v for k, v in r.oos_metrics.items()})
            meta = meta_by_fold.get(r.fold_index)
            if meta is not None:
                row["test_regime"] = meta.test_regime
                row["test_close_return"] = meta.test_close_return
                row["test_atr_median_normalized"] = meta.test_atr_median_normalized
                row["test_close_range_pct"] = meta.test_close_range_pct
                row["test_bars"] = meta.test_bars
                row["train_bars"] = meta.train_bars
            rows.append(row)
        return pd.DataFrame(rows)


class WalkForwardRunner:
    """Rolling Walk-Forward Analysis.

    各 fold は ``[train_start, train_start + train_days)`` を train、
    ``[train_start + train_days, train_start + train_days + test_days)`` を test とする.
    次の fold は ``train_start += step_days`` でスライド. test_end が data 末尾を
    超える fold は捨てる.

    Args:
        train_days: train (in-sample) 窓の日数
        test_days: test (out-of-sample) 窓の日数
        step_days: 次 fold までのスライド幅（日）
        regime_threshold: |test close return| < threshold で "flat" 判定（既定 2%）
        min_train_bars: train 窓の最小バー数. 未満の fold はスキップ
        min_test_bars: test 窓の最小バー数. 未満の fold はスキップ
    """

    def __init__(
        self,
        *,
        train_days: int,
        test_days: int,
        step_days: int,
        regime_threshold: float = 0.02,
        min_train_bars: int = 100,
        min_test_bars: int = 50,
    ) -> None:
        if train_days <= 0 or test_days <= 0 or step_days <= 0:
            raise ValueError(
                f"train/test/step_days must be positive: "
                f"train={train_days}, test={test_days}, step={step_days}"
            )
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.regime_threshold = regime_threshold
        self.min_train_bars = min_train_bars
        self.min_test_bars = min_test_bars

    # ------------------------------------------------------------------
    # Fold split
    # ------------------------------------------------------------------
    def make_folds(self, df: pd.DataFrame) -> list[FoldSpec]:
        """time-based rolling 分割 → ``FoldSpec`` のリスト.

        境界規約:
            train: ``[train_start, train_end)`` （右排他）
            test:  ``[test_start, test_end)``  （右排他）
            train_end == test_start なので train と test は重複も間隙もない.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"df.index must be DatetimeIndex, got: {type(df.index).__name__}"
            )
        if len(df) == 0:
            return []

        data_start = df.index[0]
        data_end = df.index[-1]

        train_td = pd.Timedelta(days=self.train_days)
        test_td = pd.Timedelta(days=self.test_days)
        step_td = pd.Timedelta(days=self.step_days)

        folds: list[FoldSpec] = []
        fold_idx = 0
        train_start = data_start
        # data_end は包含側. test_end は排他なので test_end <= data_end + ε を許容.
        epsilon = pd.Timedelta(seconds=1)
        while True:
            train_end = train_start + train_td
            test_start = train_end
            test_end = test_start + test_td
            if test_end > data_end + epsilon:
                break
            folds.append(
                FoldSpec(
                    fold_index=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_idx += 1
            train_start = train_start + step_td
        return folds

    # ------------------------------------------------------------------
    # Run grid × fold
    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        base_strategy: type[Strategy],
        param_grid: dict[str, list[Any]],
        *,
        spread: float,
        backtest_runner: FXBacktestRunner | None = None,
        max_folds: int | None = None,
    ) -> WFAResult:
        """全 fold × 全 grid 点でバックテスト実行.

        Args:
            df: backtesting.py 入力形式（Capitalized columns, tz-aware DatetimeIndex）
            base_strategy: ``backtesting.Strategy`` のサブクラス
            param_grid: ``{param_name: [values, ...]}``. デカルト積で展開
            spread: 全 fold 共通の spread 相対値
            backtest_runner: 既存 ``FXBacktestRunner``. None なら既定値で生成
            max_folds: 先頭から ``max_folds`` 個の fold だけ実行. ``None`` なら全 fold.
                事前 smoke で実行時間を見積もる用途.

        Returns:
            ``WFAResult``. ``results`` は (fold × params) のリスト. 失敗した
            (fold × params) は ``failures`` に記録され ``results`` には NaN メトリクスで
            含まれる.
        """
        if backtest_runner is None:
            backtest_runner = FXBacktestRunner()

        folds = self.make_folds(df)
        if max_folds is not None:
            folds = folds[:max_folds]
        if not folds:
            return WFAResult(folds_metadata=[], results=[], failures=[])

        # 全期間 ATR median を fold メタの正規化基準に使う
        atr_full = wilder_atr(
            df["High"].to_numpy(),
            df["Low"].to_numpy(),
            df["Close"].to_numpy(),
            period=14,
        )
        atr_full_median = float(np.nanmedian(atr_full))
        if not np.isfinite(atr_full_median):
            atr_full_median = 0.0

        param_combos = _enumerate_grid(param_grid)

        folds_metadata: list[FoldMetadata] = []
        results: list[WFAFoldResult] = []
        failures: list[dict[str, Any]] = []

        for fold in folds:
            train_df = _slice_window(df, fold.train_start, fold.train_end)
            test_df = _slice_window(df, fold.test_start, fold.test_end)

            if len(train_df) < self.min_train_bars or len(test_df) < self.min_test_bars:
                continue

            meta = self._compute_fold_metadata(
                fold=fold,
                train_df=train_df,
                test_df=test_df,
                atr_full_median=atr_full_median,
            )
            folds_metadata.append(meta)

            for params in param_combos:
                strategy_cls = make_strategy_subclass(base_strategy, params)

                is_metrics, is_err = self._run_one(
                    train_df, strategy_cls, spread, backtest_runner
                )
                oos_metrics, oos_err = self._run_one(
                    test_df, strategy_cls, spread, backtest_runner
                )

                if is_err is not None or oos_err is not None:
                    failures.append(
                        {
                            "fold_index": fold.fold_index,
                            "params": dict(params),
                            "is_error": is_err,
                            "oos_error": oos_err,
                        }
                    )

                results.append(
                    WFAFoldResult(
                        fold_index=fold.fold_index,
                        params=dict(params),
                        is_metrics=is_metrics,
                        oos_metrics=oos_metrics,
                    )
                )

        return WFAResult(
            folds_metadata=folds_metadata,
            results=results,
            failures=failures,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_fold_metadata(
        self,
        *,
        fold: FoldSpec,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        atr_full_median: float,
    ) -> FoldMetadata:
        test_close = test_df["Close"].to_numpy()
        first_close = float(test_close[0])
        last_close = float(test_close[-1])
        if first_close > 0:
            ret = (last_close - first_close) / first_close
        else:
            ret = float("nan")

        if np.isnan(ret):
            regime = "unknown"
        elif abs(ret) < self.regime_threshold:
            regime = "flat"
        elif ret > 0:
            regime = "up"
        else:
            regime = "down"

        test_atr = wilder_atr(
            test_df["High"].to_numpy(),
            test_df["Low"].to_numpy(),
            test_df["Close"].to_numpy(),
            period=14,
        )
        atr_med = float(np.nanmedian(test_atr))
        if atr_full_median > 0 and np.isfinite(atr_med):
            atr_norm = atr_med / atr_full_median
        else:
            atr_norm = float("nan")

        max_close = float(np.nanmax(test_close))
        min_close = float(np.nanmin(test_close))
        mean_close = float(np.nanmean(test_close))
        if mean_close > 0:
            range_pct = (max_close - min_close) / mean_close
        else:
            range_pct = float("nan")

        return FoldMetadata(
            fold_index=fold.fold_index,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            train_bars=len(train_df),
            test_bars=len(test_df),
            test_close_return=ret,
            test_regime=regime,
            test_atr_median_normalized=atr_norm,
            test_close_range_pct=range_pct,
        )

    def _run_one(
        self,
        df: pd.DataFrame,
        strategy_cls: type[Strategy],
        spread: float,
        backtest_runner: FXBacktestRunner,
    ) -> tuple[dict[str, float], str | None]:
        """1 (fold-region × strategy) のバックテスト. 失敗時は NaN dict + エラー文字列.

        戻り値の metrics dict は ``METRIC_KEYS`` (backtesting.py stats 由来) +
        ``CLOSE_RATIO_KEYS`` (StrategyBase._close_reasons 由来) を含む.
        ``# Trades`` が 0 / Strategy が ``StrategyBase`` を継承していない場合は
        ``CLOSE_RATIO_KEYS`` 値は NaN.
        """
        nan_metrics = {k: float("nan") for k in METRIC_KEYS}
        nan_metrics.update({k: float("nan") for k in CLOSE_RATIO_KEYS})
        if len(df) == 0:
            return nan_metrics, "empty dataframe"
        try:
            stats, _meta = backtest_runner.run(df, strategy_cls, spread=spread)
        except Exception as exc:  # noqa: BLE001 — backtesting.py の任意例外を捕捉
            return nan_metrics, f"{type(exc).__name__}: {exc}"

        metrics = {k: _safe_float(stats.get(k)) for k in METRIC_KEYS}
        metrics.update(_compute_close_ratios(stats, metrics["# Trades"]))
        return metrics, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_window(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """``[start, end)`` の右排他スライス."""
    mask = (df.index >= start) & (df.index < end)
    return df[mask]


def _enumerate_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """``{name: [v1, v2]}`` → ``[{name: v1}, {name: v2}]`` のデカルト積展開."""
    if not param_grid:
        return [{}]
    names = list(param_grid.keys())
    values = [param_grid[k] for k in names]
    return [dict(zip(names, combo)) for combo in product(*values)]


def make_strategy_subclass(
    base: type[Strategy], params: dict[str, Any]
) -> type[Strategy]:
    """``base`` の class attributes を ``params`` で上書きしたサブクラスを動的生成.

    backtesting.py の Strategy は class 属性で動作するため、grid の各点で異なる
    属性値を持つサブクラスを作る必要がある.

    Note:
        ``MARGIN`` をここで上書きすると ``FXBacktestRunner.run()`` の
        strategy/runner margin 一致検証が失敗する. 本ランナの設計上、
        ``param_grid`` には ``MARGIN`` を含めない前提.
    """
    suffix = "_".join(f"{k}{_format_for_classname(v)}" for k, v in params.items())
    name = f"{base.__name__}_{suffix}" if suffix else base.__name__
    return type(name, (base,), dict(params))


def _format_for_classname(v: Any) -> str:
    s = str(v)
    return s.replace(".", "p").replace("-", "m")


def _safe_float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(f) or np.isinf(f):
        return f if np.isnan(f) else float("nan")
    return f


def _compute_close_ratios(stats: pd.Series, n_trades: float) -> dict[str, float]:
    """``stats["_strategy"]._close_reasons`` から ``CLOSE_RATIO_KEYS`` を計算する.

    分母は ``# Trades`` (全クローズ数). ``_close_reasons`` の sum とは限らず、
    sl / finalize 集計を Strategy 側で取れない分が ``1 - (max_bars + signal)``
    の差分として残る.

    ``# Trades`` が 0 / Strategy が ``StrategyBase`` 非継承の場合は NaN を返す.
    """
    nan_ratios = {k: float("nan") for k in CLOSE_RATIO_KEYS}
    if not (n_trades > 0):
        return nan_ratios
    strategy_inst = stats.get("_strategy")
    close_reasons = getattr(strategy_inst, "_close_reasons", None)
    if not isinstance(close_reasons, dict):
        return nan_ratios
    return {
        "max_bars_close_ratio": close_reasons.get("max_bars", 0) / n_trades,
        "signal_close_ratio": close_reasons.get("signal", 0) / n_trades,
    }
