# ch02_gbpjpy_spread_sensitivity

## 再現対象

GBPJPY の spread 感度分析を再現します。

## 実行

```bash
python chapters/season1/ch02_gbpjpy_spread_sensitivity/run.py
```

GBPJPY 180 日データに対して、0.9 銭 / 1.7 銭 / 2.5 銭の spread 感度分析を行い、`outputs/ch02_gbpjpy_spread_sensitivity/` に CSV を出力します。

## 次の移行作業

- 記事掲載値の参照 CSV を `results/reference/ch02_gbpjpy_spread_sensitivity/` に置く。
