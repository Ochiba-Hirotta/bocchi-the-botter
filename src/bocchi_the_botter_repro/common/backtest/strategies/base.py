"""StrategyBase abstract base class.

責務:
    - クラス変数のデフォルト値統一 (``MARGIN`` / ``MAX_BARS`` / ``ATR_N`` /
      ``SL_ATR_MULT`` / ``RISK_PCT`` / ``INITIAL_CASH`` / ``COMMISSION_PCT``).
      子クラスは差分のみ override する.
    - ``_check_max_bars_exit`` / ``_bars_since_entry`` の共通実装で
      MAX_BARS 集計を戦略間で揃える.
    - ``_record_close_reason`` で close 理由を ``_close_reasons`` に記録.
      ``{"max_bars", "sl", "signal", "finalize"}`` の 4 区分.

抽象化しない項目:
    - ``init()`` / ``next()`` の ``@abstractmethod`` 化はしない.
      backtesting.py の ``Strategy`` 継承都合でフィクスチャが複雑化するため.
    - エントリ条件 / エグジット条件の判定は子クラスに委譲する.
      戦略ごとに本質的に異なる (BB-MR は中央線到達, Donchian は
      DC_EXIT × ATR 位置, ボラ収縮ブレイクアウトはボラ拡大).

``signal_close_ratio`` の戦略間意味差:
    BB-MR では「平均回帰成功」, Donchian では「トレンド成功」と意味が異なる.
    列名は共通 ``signal`` として記録するが, 解釈は戦略ごとに違う点を
    #6 記事 Method の説明と対応する.

``finalize`` の取り扱い:
    ``finalize_trades=True`` で発生する期間末強制クローズは ``_close_reasons``
    では集計するが, 公開出力列には出さない (取るが出さない).
"""
from __future__ import annotations

from backtesting import Strategy

# 公開する close reason の 4 区分.
CLOSE_REASONS: tuple[str, ...] = ("max_bars", "sl", "signal", "finalize")


class StrategyBase(Strategy):
    """全戦略の抽象基底. 共通クラス変数 + 共通ヘルパを提供する.

    子クラスは ``init()`` で ``super().init()`` を呼び ``_close_reasons`` を
    初期化すること. ``next()`` 内で MAX_BARS 経過時は ``_check_max_bars_exit``
    を判定し, close 後に ``_record_close_reason("max_bars")`` を呼ぶ規律.

    クラス変数の責務:
        - ``MARGIN``: ``Backtest(margin=...)`` と一致させる運用責任. backtesting.py
          の Broker は public に margin を公開していないため, Strategy 側に保持し
          ``compute_units`` の ``units_margin_cap`` 計算で参照する.
          ``runner.py`` 側で ``strategy_cls.MARGIN`` と ``self.margin`` の照合あり.
        - ``INITIAL_CASH`` / ``COMMISSION_PCT``: ``Backtest(cash=, commission=)``
          に渡す標準値の文書化. Strategy 内では参照しない (将来 runner 側で
          参照する場合に備えたデフォルト値統一).
        - ``RISK_PCT`` / ``SL_ATR_MULT`` / ``MAX_BARS`` / ``ATR_N``: 戦略横断
          で揃えるサイズ計算 / リスク管理パラメータ.
    """

    # ===== Backtest インスタンスに渡す標準値 (Strategy 内で参照しない) =====
    INITIAL_CASH: float = 100_000.0
    COMMISSION_PCT: float = 0.0
    MARGIN: float = 0.04  # 25x レバレッジ

    # ===== Strategy 内で参照する戦略横断パラメータ =====
    RISK_PCT: float = 0.01  # 1 トレードあたりリスク
    SL_ATR_MULT: float = 1.5  # SL 距離 (ATR 倍率)
    MAX_BARS: int = 48  # ポジション最大保持バー数
    ATR_N: int = 14  # Wilder ATR の N

    def init(self) -> None:
        """``_close_reasons`` を初期化する. 子クラスは ``super().init()`` を呼ぶこと.

        backtesting.py の ``Strategy.init`` は no-op だが, ここでも明示的には
        super を呼ばない (将来 backtesting.py 側に init が追加された場合は
        本基底側で対応する設計).
        """
        self._close_reasons: dict[str, int] = {key: 0 for key in CLOSE_REASONS}

    # ===== 共通ヘルパ =====

    def _bars_since_entry(self) -> int | None:
        """直近トレードのエントリからの経過バー数を返す. 未エントリなら ``None``.

        既存 ``BBMeanReversion._try_exit`` の MAX_BARS 判定ロジックを共通化したもの.
        backtesting.py の ``self.trades`` 末尾を最新ポジションのエントリとみなす.
        """
        if not self.trades:
            return None
        entry_bar = self.trades[-1].entry_bar
        current_bar = len(self.data) - 1
        return current_bar - entry_bar

    def _check_max_bars_exit(self) -> bool:
        """MAX_BARS 超過判定. ``True`` なら ``self.position.close()`` する責務は呼び出し側.

        子クラスは ``next()`` 内で
        ``if self._check_max_bars_exit(): self.position.close(); self._record_close_reason("max_bars")``
        の規律で使う. close 自体は子に任せ, 判定だけを基底で揃える理由は, 既存
        BB-MR の判定式 ``current_bar - entry_bar >= MAX_BARS`` を Donchian
        含む全戦略で完全一致させるため.
        """
        bars = self._bars_since_entry()
        if bars is None:
            return False
        return bars >= self.MAX_BARS

    def _record_close_reason(self, reason: str) -> None:
        """close 理由を ``_close_reasons`` に記録する.

        ``reason`` は ``CLOSE_REASONS`` のいずれか. それ以外は ``ValueError``.
        ``finalize`` は集計するが公開出力列には出さない.
        """
        if reason not in self._close_reasons:
            raise ValueError(
                f"Unknown close reason: {reason!r}. "
                f"Expected one of {list(CLOSE_REASONS)}"
            )
        self._close_reasons[reason] += 1
