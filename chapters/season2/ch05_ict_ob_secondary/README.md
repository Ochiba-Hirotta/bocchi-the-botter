# ch05_ict_ob_secondary

Season 2 #5（二次17分翻訳と三定義の比較）の参照先です。

**この章に実行コードはありません。** #5で報告する数値は、#4と同じ工程6・7の一度の実行から出ています。二つの翻訳は、どちらかの結果を見てから条件を決めることがないよう、走らせる前に一括で凍結し、同じ入力・同じ約定モデル・同じ期間で同時に走らせました。したがって二次翻訳だけを別に走らせる入口を作らず、`ch04_ict_order_blocks` をそのまま正本とします。

## 実体の所在

| 対象 | 場所 |
|---|---|
| 検出・約定・比較の実装 | `src/bocchi_the_botter_repro/season2/ict_ob.py`、`ict_ob_comparison.py` |
| 工程6本走の入口 | `chapters/season2/ch04_ict_order_blocks/run.py` |
| 工程7比較の入口 | `chapters/season2/ch04_ict_order_blocks/compare.py` |
| evidence pack・独立検証 | `chapters/season2/ch04_ict_order_blocks/verify.py` |
| 記事図の生成 | `chapters/season2/ch04_ict_order_blocks/figures.py` |
| row-free manifest | `results/reference/ch04_ict_order_blocks/manifest.json` |
| 図とhash | `results/reference/ch04_ict_order_blocks/figures/`、`figure_hashes.json` |

## 再現

```bash
python chapters/season2/ch04_ict_order_blocks/compare.py \
  --db "$SOURCE_DB" \
  --repeat 2
```

`--repeat 2` はSQLiteの読み取りから比較集計まで二度実行し、結果hashが一致しなければ失敗します。

記事図（#5の3点）は次で再生成します。#4の3点も同時に生成され、固定済みのバイト列と一致することが要求されます。

```bash
python chapters/season2/ch04_ict_order_blocks/figures.py
```

## #5が参照する主な値

| 項目 | 値 |
|---|---:|
| 二次翻訳の検出ゾーン / 決済 | 321 / 194 |
| 二次翻訳のリターン | -23.132819% |
| 主判定 | 不成立 |
| 40分翻訳 → 17分翻訳の重なり | 46 / 1,121（4.103479%） |
| 17分翻訳 → 40分翻訳の重なり | 45 / 321（14.018692%） |
| 40分翻訳 → 流通実装の重なり | 15 / 1,121（1.338091%） |
| 17分翻訳 → 流通実装の重なり | 0 / 321（0%） |

流通実装は `smartmoneyconcepts==0.0.27` をデフォルトパラメータで実行し、**約定と損益は計算していません**。出力の27件は期間中の全検出数ではなく、関数を通したあとに残った行です。デフォルトのスイング判定は前後50本の中央窓を使うため、その時点で利用可能な非先読みの検出時刻ではありません。この比較は流通している実装の最終出力との突き合わせであって、正解との照合ではありません。

二つの翻訳の優劣、原手法や出典者の評価、実運用可能性、他通貨ペア・他期間への一般化は、この検証からは言えません。
