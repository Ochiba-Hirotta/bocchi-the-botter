# ch01_usdjpy_bb_mr

## 再現対象

USDJPY の Bollinger Band mean reversion 初回検証を再現します。

## 実行

```bash
python chapters/season1/ch01_usdjpy_bb_mr/run.py
```

既定では記事 #1 当時の訂正前 spread `1.33e-5` で USDJPY 180 日分の BB-MR を実行し、`outputs/ch01_usdjpy_bb_mr/` に summary / trades / equity CSV を出力します。

## 次の移行作業

- 記事掲載値の参照 CSV を `results/reference/ch01_usdjpy_bb_mr/` に置く。
