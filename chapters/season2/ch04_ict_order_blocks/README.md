# ch04_ict_order_blocks

Season 2 #4 の工程6・7実行入口です。S2-2で固定したOANDA `USD_JPY / M5 / BA`の読み取り専用SQLiteから、同じcomplete M15・NY17日足・日足バイアスを派生し、本家Month 04翻訳と二次sweep→MSS→FVG翻訳を同じ約定モデルで走らせます。工程7では同じcomplete M15へ`smartmoneyconcepts==0.0.27`のOB・Liquidity関数をデフォルトのまま適用し、ゾーンだけを比較します。

固定M5投影hashが次と一致しない入力では停止します。

```text
f6d0e1cd1bd50ec11f7f3f0bd34e31b61a39970a687f3c5ac83682ae2ea1d512
```

## 工程6本走

`--repeat 2`はSQLiteから全工程を二度実行し、ゾーン・トレード・terminal state・counterを含む結果hashが一致しなければ失敗します。

```bash
python chapters/season2/ch04_ict_order_blocks/run.py \
  --db "$SOURCE_DB" \
  --repeat 2
```

既定出力先は`outputs/ch04_ict_order_blocks/`です。価格・timestampを含むzone/trade CSVと私的監査JSONを置くため、Git管理しません。

約定は凍結仕様どおり、bid/ask別OHLC、手数料0、通常スリッページ0、スワップ0、同一足SL優先、強制手仕舞いなしです。これは実運用成績ではなく、事前固定した翻訳の検証結果です。

## 工程7比較

```bash
python chapters/season2/ch04_ict_order_blocks/compare.py \
  --db "$SOURCE_DB" \
  --repeat 2
```

比較は価格閉区間と半開の有効時間窓がともに交差する場合を重複と数えます。fillなしを含む本家・二次の全ゾーンlifecycle、OSS OB/Liquidity明細は既定で`outputs/ch04_ict_order_blocks/stage7/`へ保存し、Git管理しません。`--repeat 2`は工程6結果、両lifecycle、OSS出力、三組の重なりsummaryを含む工程7hashの一致を要求します。

OSSのOB関数は期間中の全履歴イベントではなく、実行終了時に出力配列へ残ったOB行を返します。そのためOSS重なり率は、同版の最終出力との答え合わせであり、三検出器の網羅的な優劣比較ではありません。OSSには約定・損益を接続しません。

## 工程8 evidence pack

工程7全体を固定SQLiteから二度再実行し、記事で使いうる集計値、条件、依存version、code hash、工程6・7hash、再実行commandをrow-free manifestへ固定します。

```bash
python chapters/season2/ch04_ict_order_blocks/build_manifest.py \
  --db "$SOURCE_DB" \
  --reference-dir results/reference/ch04_ict_order_blocks
```

SQLiteを持たない環境でも、manifestのdetached hash、pinned code、依存version、集計内整合、公開境界を検証できます。

```bash
python chapters/season2/ch04_ict_order_blocks/verify.py \
  --reference-dir results/reference/ch04_ict_order_blocks
```

同じSQLiteから工程6で出力したprivate trade CSVがある環境では、S2-3の
`--trades`モードと同様に、方向別fillからPnL・実現R・残高chain・5区間・
主要指標・exit内訳を独立再計算してmanifestへ照合できます。

```bash
python chapters/season2/ch04_ict_order_blocks/verify.py \
  --reference-dir results/reference/ch04_ict_order_blocks \
  --trades outputs/ch04_ict_order_blocks
```

数値監査の正本は`manifest.json`と`manifest.sha256`です。市場行、zone/trade/liquidity行、個別価格・個別timestamp、SQLite、credential、ユーザー固有の絶対pathは含みません。全結果の再計算には、manifest記載のM5抽出hashと一致する非公開SQLiteが必要です。

## 記事図

row-freeの`manifest.json`から、買い側ルールの模式図、固定184日区間の境界残高、決済理由内訳の3枚を生成します。境界残高図は区間内の残高曲線を補間せず、manifestに固定された5区間の実現損益だけを累積します。

```bash
python chapters/season2/ch04_ict_order_blocks/figures.py
```

既定出力先は`results/reference/ch04_ict_order_blocks/figures/`です。`figure_hashes.json`へ入力manifestと各PNGのSHA-256を記録します。
