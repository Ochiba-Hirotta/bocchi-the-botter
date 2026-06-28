# ch05_wfa_four_pairs

## 再現対象

USDJPY / GBPJPY / EURJPY / AUDJPY の walk-forward analysis と重心計算を再現します。

## 実行

```bash
python chapters/season1/ch05_wfa_four_pairs/run.py
```

既定では 4 ペアの BB-MR WFA を full mode で実行し、重心 summary も出力します。動作確認だけなら `--mode smoke` を使います。

## 参照出力

記事掲載値と照合するための CSV は `results/reference/ch05_wfa_four_pairs/` に置いています。

- `wfa_bb_mr_USDJPY_2026-04-29.csv`
- `wfa_bb_mr_GBPJPY_2026-04-29.csv`
- `wfa_bb_mr_EURJPY_2026-04-29.csv`
- `wfa_bb_mr_AUDJPY_2026-04-29.csv`
- `wfa_bb_mr_4pairs_summary.csv`
