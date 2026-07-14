# Chapters

各章の再現入口を season ごとに置きます。

章フォルダは、Note 本文の Appendix から直接リンクできる粒度にします。

各章の数値再現入口は `run.py` です。出力は既定で `outputs/chXX_*/` に保存します。

## Season 1

| 章 | フォルダ | 役割 |
|---|---|---|
| #0 | `season1/ch00_prologue/` | 環境説明 |
| #1 | `season1/ch01_usdjpy_bb_mr/` | USDJPY BB-MR 初回検証 |
| #1.1 | `season1/ch01_1_spread_correction/` | spread 訂正比較 |
| #2 | `season1/ch02_gbpjpy_spread_sensitivity/` | GBPJPY spread 感度分析 |
| #3 | `season1/ch03_two_year_segments/` | 720 日・前半/後半/全期間 |
| #4 | `season1/ch04_wfa_two_pairs/` | USDJPY / GBPJPY WFA |
| #5 | `season1/ch05_wfa_four_pairs/` | 4 ペア WFA + 重心 |
| #6 | `season1/ch06_donchian_compare/` | BB-MR vs Donchian |
| #7 | `season1/ch07_physical_metrics/` | 8 grid x 5 fold の 4 物理量 |

## Season 2

| 章 | フォルダ | 役割 |
|---|---|---|
| S2-1 | `season2/ch01_orb_1h_translation/` | EURUSD・15 分足の ORB 最終形を USDJPY・1 時間足へ翻訳 |
| S2-2 | `season2/ch02_minute_data_db/` | OANDA M5をSQLiteへ保存し、完全な三本だけをM15へ集約 |
