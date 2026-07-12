# Reference Results

記事掲載値と照合するための派生結果 CSV をここに置きます。

市場データの Parquet キャッシュ本体は、ライセンスと再配布リスクがあるため公開対象にしません。

各章の参照出力は章フォルダと同じ名前のサブディレクトリに配置します。

```text
results/reference/ch04_wfa_two_pairs/
results/reference/ch05_wfa_four_pairs/
results/reference/ch01_orb_1h_translation/
```

## 配置済み

- `ch04_wfa_two_pairs/`: USDJPY / GBPJPY の BB-MR WFA と 2 ペア比較表。
- `ch05_wfa_four_pairs/`: 4 ペアの BB-MR WFA と重心 summary。
- `ch06_donchian_compare/`: 4 ペアの BB-MR / Donchian WFA と戦略比較表。
- `ch07_physical_metrics/`: 8 grid x 5 fold の取引系列、fold 別集計、grid 別集計。
- `ch01_orb_1h_translation/`: Season 2 #1 の主版・参考版取引明細と、720 暦日窓を分けた固定 144 日×5 区間の集計。

`ch07_physical_metrics/trades_7_*.csv` は、市場データ本体ではなく、#7 の 4 物理量を再集計するための取引系列です。`wfa_results_7_per_fold.csv` の `oos_sharpe` と `oos_n_trades_raw` は元 WFA 由来の補助列で、取引系列だけから再計算する物理量ではありません。

`ch01_orb_1h_translation/` の CSV は記事時点の派生結果です。`figures/` には記事で使用した最終 PNG 4 点を置いています。元 OHLC は同梱していません。Yahoo Finance の修正・欠損・仕様変更により、後日のライブ再計算が参照 CSV や最終 PNG と一致しない可能性があります。
