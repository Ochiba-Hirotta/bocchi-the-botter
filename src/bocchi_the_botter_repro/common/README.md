# Common Code

複数章で共有する再現用コードです。

## data

`yfinance` から FX の OHLCV を取得し、共通スキーマへ正規化します。

公開版のサポート対象は、記事で使う次の 4 ペアです。

```text
USDJPY, GBPJPY, EURJPY, AUDJPY
```

1h 足は yfinance 側の履歴上限により、実行日からおおむね 730 日以内の `start` を指定してください。記事掲載値との照合は `results/reference/` の派生結果 CSV を基準にします。

## backtest

`backtesting.py` 向けのデータ変換、FX デフォルト設定の runner、戦略、walk-forward analysis、集計関数を置きます。

章別の実行入口は `chapters/season1/chXX_*/run.py` です。共通の実行補助は `reproduction.py` に集約しています。
