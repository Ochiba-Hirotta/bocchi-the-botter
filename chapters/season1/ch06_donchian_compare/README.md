# ch06_donchian_compare

## 再現対象

Bollinger Band mean reversion と Donchian breakout の比較を再現します。

## 実行

```bash
python chapters/season1/ch06_donchian_compare/run.py
```

既定では 4 ペアの BB-MR / Donchian WFA を full mode で実行し、戦略比較 CSV も出力します。動作確認だけなら `--mode smoke` を使います。

## 参照出力

記事掲載値と照合するための CSV は `results/reference/ch06_donchian_compare/` に置いています。

- `wfa_bb_mr_USDJPY_2026-04-29.csv`
- `wfa_bb_mr_GBPJPY_2026-04-29.csv`
- `wfa_bb_mr_EURJPY_2026-04-29.csv`
- `wfa_bb_mr_AUDJPY_2026-04-29.csv`
- `wfa_donchian_USDJPY_2026-04-29.csv`
- `wfa_donchian_GBPJPY_2026-04-29.csv`
- `wfa_donchian_EURJPY_2026-04-29.csv`
- `wfa_donchian_AUDJPY_2026-04-29.csv`
- `wfa_strategy_compare.csv`
