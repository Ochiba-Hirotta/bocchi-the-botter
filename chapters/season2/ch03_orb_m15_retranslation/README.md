# ch03_orb_m15_retranslation

Season 2 #3 の主版です。S2-2で固定したOANDA `USD_JPY / M5 / BA`の読み取り専用SQLiteからcomplete M15を派生し、9:30–9:45 ETの一本をレンジにしたORBを実行します。

記事参照結果は、固定UTC区間のM5投影hashが次と一致する入力でしか生成しません。

```text
f6d0e1cd1bd50ec11f7f3f0bd34e31b61a39970a687f3c5ac83682ae2ea1d512
```

## 実行

価格付きtrade logはGit管理外へ置きます。`--reference-dir`を指定した場合だけ、行単位価格を含まない公開参照物も生成します。

```bash
python chapters/season2/ch03_orb_m15_retranslation/run.py \
  --db "$SOURCE_DB" \
  --private-output-dir ../../code/data/s2_3_orb_m15 \
  --reference-dir results/reference/ch03_orb_m15_retranslation
```

上流SQLiteはURI `mode=ro`と`PRAGMA query_only=ON`で読みます。固定区間のhash、M5/M15行数、品質counterが一致しない場合は停止します。通常のスプレッドを追加控除せず、longはask entry/bid exit、shortはbid entry/ask exitを使います。

## 独立再計算

主版実行後、価格側のPnL、equity chain、realized R、5区間、主要指標をtrade logから再計算します。

```bash
python chapters/season2/ch03_orb_m15_retranslation/verify.py \
  --trades ../../code/data/s2_3_orb_m15/trades_private.csv \
  --reference-dir results/reference/ch03_orb_m15_retranslation
```

非公開trade logを持たない環境でも、集計ファイルのhash、5区間、件数、判定、公開境界は検証できます。

```bash
python chapters/season2/ch03_orb_m15_retranslation/verify.py \
  --reference-dir results/reference/ch03_orb_m15_retranslation
```

## 図の再生成

5区間、決済理由、ATR分類、27本セッション完全性の4図は、行単位レートを含まない公開集計だけから再生成します。

```bash
python chapters/season2/ch03_orb_m15_retranslation/figures.py
```

## 公開境界

公開参照物にはOANDAのM5/M15行、timestamp付き取引、entry/exit価格を含めません。公開コードだけでは記事時点の全取引を再生成できず、同じ非公開SQLiteが必要です。人工fixtureとrow-free集計の整合性はテストで確認できます。
