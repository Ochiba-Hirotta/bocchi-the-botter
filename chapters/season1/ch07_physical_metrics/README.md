# ch07_physical_metrics

## 再現対象

8 grid x 5 fold の取引系列から、4 つの物理量を計算する検証を再現します。

## 実行

```bash
python chapters/season1/ch07_physical_metrics/run.py
```

既定では `results/reference/ch07_physical_metrics/` に置かれた `trades_7_*.csv` から 4 指標を再集計します。市場データから取引系列も再生成する場合は `--recompute-trades` を使います。

## 参照出力

記事掲載値と照合するための CSV は `results/reference/ch07_physical_metrics/` に置いています。

- `wfa_results_7_per_fold.csv`
- `wfa_results_7_per_grid.csv`
- `trades_7_*.csv` 40 ファイル

`trades_7_*.csv` は、#7 の 4 物理量を再集計するための取引系列です。市場データの Parquet キャッシュ本体は含めていません。

`wfa_results_7_per_fold.csv` の `oos_sharpe` と `oos_n_trades_raw` は元 WFA 由来の補助列です。既定実行では、同じ参照フォルダにこの CSV がある場合だけ補助列も出力へ引き継ぎます。
