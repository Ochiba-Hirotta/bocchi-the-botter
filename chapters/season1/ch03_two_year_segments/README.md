# ch03_two_year_segments

## 再現対象

720 日相当の期間を前半、後半、全期間に分けた検証を再現します。

## 実行

```bash
python chapters/season1/ch03_two_year_segments/run.py
```

既定では USDJPY / GBPJPY の 720 日データを前半、後半、全期間に分け、`outputs/ch03_two_year_segments/` に CSV を出力します。

## 次の移行作業

- 記事掲載値の参照 CSV を `results/reference/ch03_two_year_segments/` に置く。
