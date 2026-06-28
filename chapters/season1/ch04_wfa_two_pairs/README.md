# ch04_wfa_two_pairs

## 再現対象

USDJPY / GBPJPY の Bollinger Band mean reversion walk-forward analysis を再現します。

## 実行

```bash
python chapters/season1/ch04_wfa_two_pairs/run.py
```

既定では USDJPY / GBPJPY の BB-MR WFA を full mode で実行し、`outputs/ch04_wfa_two_pairs/` に long-format CSV を出力します。動作確認だけなら `--mode smoke` を使います。

## 参照出力

記事掲載値と照合するための CSV は `results/reference/ch04_wfa_two_pairs/` に置いています。

- `wfa_bb_mr_USDJPY_2026-04-29.csv`
- `wfa_bb_mr_GBPJPY_2026-04-29.csv`
- `wfa_bb_mr_combined.csv`
