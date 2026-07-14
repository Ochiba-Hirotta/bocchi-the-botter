# ch02_minute_data_db

Season 2 #2で使用する、OANDAの完了済み5分足を記事用SQLiteへ保存する最小コードです。

`paper-trader`のコードやDBは使用しません。practice REST APIから一通貨ペアの`M5 / BA`だけを取得し、同じ期間を再実行しても複合主キーで行が増殖しない形でupsertします。

## 実行

Personal Access TokenはCLI引数へ書かず、環境変数から渡します。

```bash
export OANDA_API_TOKEN="..."

python chapters/season2/ch02_minute_data_db/run.py \
  --instrument USD_JPY \
  --start 2026-07-06T13:00:00Z \
  --end 2026-07-06T15:00:00Z \
  --db outputs/ch02_minute_data_db/rates.sqlite \
  --manifest outputs/ch02_minute_data_db/manifest.json
```

期間はstart inclusive / end exclusiveの半開区間です。startとendはUTCの5分境界を指定します。取得先は`https://api-fxpractice.oanda.com`に固定しています。

## 保存するもの

- source、instrument、granularity、price、UTC開始時刻
- 取得時刻、complete、volume
- bid OHLC、ask OHLC

未確定足は保存しません。token、account ID、APIのraw JSON、mid、spreadはDBへ保存しません。spreadは後続のDataFrame投影時にbid/askから派生します。

OANDAの生レート、SQLite本体、行単位CSVは本リポジトリへ同梱しません。

## SQLiteからDataFrameへ戻す

既存SQLiteはread-onlyで開き、start inclusive / end exclusiveの範囲を`ts_utc`昇順で投影します。

```python
from pathlib import Path

from bocchi_the_botter_repro.season2.minute_data import (
    SOURCE,
    audit_m5_frame,
    load_m5_candles,
    parse_m5_boundary,
)

start = parse_m5_boundary("2026-07-06T13:00:00Z")
end = parse_m5_boundary("2026-07-06T15:00:00Z")
frame = load_m5_candles(
    Path("outputs/ch02_minute_data_db/rates.sqlite"),
    source=SOURCE,
    instrument="USD_JPY",
    start_inclusive=start,
    end_exclusive=end,
)
audit = audit_m5_frame(
    frame,
    source=SOURCE,
    instrument="USD_JPY",
    start_inclusive=start,
    end_exclusive=end,
)
```

DataFrameにはtimezone-awareな`ts_utc_dt`と、bid/askから計算した`spread_open`、`spread_close`を追加します。mid OHLCは作りません。gapは検出しますが、補間やforward fillは行いません。

## M5をM15へまとめる

```python
from bocchi_the_botter_repro.season2.minute_data import (
    aggregate_m5_to_m15,
    select_m15_et_time,
)

result = aggregate_m5_to_m15(
    frame,
    start_inclusive=start,
    end_exclusive=end,
)
m15 = result.candles
rejected = result.incomplete_buckets
opening_ranges = select_m15_et_time(m15, hour=9, minute=30)
```

M15はUTCの15分開始境界ごとに、期待するM5三本が揃った場合だけ作ります。bid/askは別々に集約し、volumeは三本の合計です。欠けたbucketは`incomplete_buckets`へ残し、補間しません。

New York時刻は`America/New_York`から派生するため、9:30 ETのUTC時刻を固定offsetで決めません。M15からM5内部の価格経路は復元できないため、M15をSQLiteへ書き戻さず元のM5を残します。

## 既存DBのevidence manifestを作る

記事時点の監査値は、SQLite本体や行単位レートではなく、小さなJSON manifestへ固定します。入力DBは読み取り専用で扱い、manifestにはbasename、DB SHA-256、抽出SHA-256、行数、gap、M15完全性、依存version、再実行commandだけを残します。

```bash
export SOURCE_DB=/path/to/read-only/rates.sqlite

python chapters/season2/ch02_minute_data_db/build_manifest.py \
  --db "$SOURCE_DB" \
  --instrument USD_JPY \
  --start 2024-01-01T22:00:00Z \
  --end 2026-07-14T10:05:00Z \
  --manifest results/reference/ch02_minute_data_db/manifest.json
```

上流DBに取得runのrequest数などが保存されていない場合、そのcounterは推測せず`null`にします。公開再現コードから新規DBとmanifestを同時生成した場合は、同じrunのcounterを記録します。

## テスト

テストはすべて架空価格のfixtureを使い、通常実行ではネットワークへ接続しません。

```bash
pytest -q tests/test_s2_2_minute_data.py
```
