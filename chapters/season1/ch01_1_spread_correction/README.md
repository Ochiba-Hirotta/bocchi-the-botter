# ch01_1_spread_correction

## 再現対象

spread 訂正前後の比較を再現します。

## 実行

```bash
python chapters/season1/ch01_1_spread_correction/run.py
```

同じ USDJPY 180 日データに対して、訂正前 `1.33e-5` と訂正後 `1.0e-5` の spread を比較し、`outputs/ch01_1_spread_correction/` に CSV を出力します。

## 次の移行作業

- 記事掲載値の参照 CSV を `results/reference/ch01_1_spread_correction/` に置く。
