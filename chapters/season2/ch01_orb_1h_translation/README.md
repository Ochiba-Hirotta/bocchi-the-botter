# ch01_orb_1h_translation

## 再現対象

Season 2 #1 「ORB を 1 時間足に移したら、2 本の線が引っ越せなかった」の検証を再現します。

EURUSD・15 分足で使われていた ORB の最終形を、USDJPY・1 時間足へ翻訳した検証です。主版は 12:00 ET 以降の新規エントリーを行わず、参考版はそのカットを外します。

元動画は Trading Steady「[Is the ORB Strategy Actually Profitable? 6 Month Results](https://youtu.be/CVjiYdvtLpE)」です。動画の6か月をそのまま再現するコードではありません。動画途中で追加された最終ATR閾値を全期間へ固定し、動画では確認できなかったATRの期間・平滑化と保有終了時刻を、14本SMA・16:00 ETとして実装しています。

## 実行

既定実行は、`results/reference/ch01_orb_1h_translation/` の記事時点 CSV を読み、主要値と固定 144 日区間を検証します。市場データは取得しません。

```bash
python chapters/season2/ch01_orb_1h_translation/run.py
```

検証結果の要約と再集計した区間 CSV を保存する場合は、`--output-dir` を指定します。

```bash
python chapters/season2/ch01_orb_1h_translation/run.py --output-dir /path/to/output
```

Yahoo Finance の現在の 1 時間足を取得し、記事と同じ半開区間 `[2024-07-21, 2026-07-11)` で再計算する場合は `--live` を使います。

```bash
python chapters/season2/ch01_orb_1h_translation/run.py --live
```

ライブ再計算の CSV は、既定で `outputs/ch01_orb_1h_translation/` に出力されます。保存先は `--output-dir` で変更できます。

固定窓、144日境界、参照CSVをまとめて確認する回帰テストは次で実行できます。

```bash
pytest -q tests/test_s2_1_orb.py
```

## 図の再生成

記事の 4 図を再生成する場合は `figures.py` を実行します。決済理由と固定 144 日区間は参照 CSV、ATR 分布と 1 トレード図はライブ取得した記事窓のデータを使います。

```bash
python chapters/season2/ch01_orb_1h_translation/figures.py
```

参照 CSV の場所は `--results-dir`、PNG の保存先は `--output-dir` で変更できます。既定の保存先は `outputs/ch01_orb_1h_translation/figures/` です。

## 記事時点の条件

- データ: Yahoo Finance `JPY=X` / 1h
- 検証窓: ET 日付の `[2024-07-21, 2026-07-11)`（720 暦日）
- レンジ: 9:00〜10:00 ET
- エントリー: レンジ外で終値確定後、次の足の始値
- ATR: True Range の 14 本単純平均。レンジ幅が `1.25×ATR`〜`3.0×ATR` の日だけを対象
- ストップ: レンジの反対端
- 利確: 初期リスク幅の 1.5 倍
- 強制決済: 16:00 ET の 1 時間足 Open
- 足内でストップと利確の両方に届いた場合はストップを優先
- 固定 bid-ask 全幅 0.3 銭相当のみ反映
- 初期残高 1,000,000 円、1 トレードの初期リスク 1%、証拠金率 4%

## 参照出力

記事掲載値と照合するための派生 CSV は `results/reference/ch01_orb_1h_translation/` に置いています。

- `trades_S2-1_ORB_USDJPY_main_net.csv`: 主版 82 トレード
- `trades_S2-1_ORB_USDJPY_ref_net.csv`: 12:00 カットを外した参考版 169 トレード
- `segments_S2-1_ORB_USDJPY_main_net.csv`: 主版を窓の開始日から 144 日ずつに分けた 5 区間
- `figures/`: 記事で使用した最終 PNG 4 点

元 OHLC は同梱していません。参照 CSV は記事時点の派生結果を固定するためのものです。

Yahoo Finance の過去データは、配信元の修正、欠損、取得上限、仕様変更で後から変わることがあります。そのため `--live` の結果は、同じコードと固定窓を使っても参照 CSV と一致しない可能性があります。
