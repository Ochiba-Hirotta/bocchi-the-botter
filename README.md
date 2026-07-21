# bocchi-the-botter

Note 連載「ぼっち・ざ・ぼったー」の検証を再現するためのコードです。

## 目的

- 各章の再現コードを `chapters/season*/` 配下に章別で置く。
- 複数章で使うバックテスト基盤、データ取得、指標計算、戦略実装を `src/bocchi_the_botter_repro/common/` に集約する。
- yfinance の検証窓は絶対期間で固定し、`period="720d"` のような実行日依存の窓を避ける。取得に `period="max"` を使う章も、取得後に記事の固定窓へ切り出す。
- yfinance の履歴上限や再取得差分に備え、記事時点の派生結果 CSV を `results/reference/` に固定する。

## 構成

```text
bocchi-the-botter/
├── README.md
├── requirements.txt
├── pyproject.toml
├── chapters/
│   ├── README.md
│   ├── season1/
│   │   ├── ch00_prologue/
│   │   │   ├── README.md
│   │   │   └── run.py
│   │   ├── ch01_usdjpy_bb_mr/
│   │   │   ├── README.md
│   │   │   └── run.py
│   │   ├── ch01_1_spread_correction/
│   │   │   ├── README.md
│   │   │   └── run.py
│   │   ├── ch02_gbpjpy_spread_sensitivity/
│   │   │   ├── README.md
│   │   │   └── run.py
│   │   ├── ch03_two_year_segments/
│   │   │   ├── README.md
│   │   │   └── run.py
│   │   ├── ch04_wfa_two_pairs/
│   │   │   ├── README.md
│   │   │   └── run.py
│   │   ├── ch05_wfa_four_pairs/
│   │   │   ├── README.md
│   │   │   └── run.py
│   │   ├── ch06_donchian_compare/
│   │   │   ├── README.md
│   │   │   └── run.py
│   │   └── ch07_physical_metrics/
│   │       ├── README.md
│   │       └── run.py
│   └── season2/
│       ├── ch01_orb_1h_translation/
│       │   ├── README.md
│       │   ├── run.py
│       │   └── figures.py
│       ├── ch02_minute_data_db/
│       │   ├── README.md
│       │   ├── run.py
│       │   ├── build_manifest.py
│       │   └── figures.py
│       ├── ch03_orb_m15_retranslation/
│       │   ├── README.md
│       │   ├── run.py
│       │   ├── verify.py
│       │   └── figures.py
│       └── ch04_ict_order_blocks/
│           ├── README.md
│           ├── run.py
│           ├── compare.py
│           ├── build_manifest.py
│           └── verify.py
├── src/
│   └── bocchi_the_botter_repro/
│       ├── common/
│       │   ├── data/
│       │   ├── backtest/
│       │   └── reproduction.py
│       └── season2/
│           ├── orb.py
│           ├── minute_data.py
│           └── orb_m15.py
├── tests/
│   ├── test_s2_1_orb.py
│   ├── test_s2_2_minute_data.py
│   └── test_s2_3_orb_m15.py
├── results/
│   └── reference/
├── outputs/
└── data_cache/
```

## セットアップ

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 章別実行

各章は `chapters/season*/chXX_*/run.py` を入口にします。

```bash
python chapters/season1/ch04_wfa_two_pairs/run.py --mode smoke
python chapters/season1/ch05_wfa_four_pairs/run.py --mode smoke
python chapters/season1/ch06_donchian_compare/run.py --mode smoke
python chapters/season2/ch01_orb_1h_translation/run.py
python chapters/season2/ch02_minute_data_db/run.py --help
python chapters/season2/ch02_minute_data_db/build_manifest.py --help
python chapters/season2/ch02_minute_data_db/figures.py --help
python chapters/season2/ch03_orb_m15_retranslation/run.py --help
python chapters/season2/ch03_orb_m15_retranslation/verify.py --help
python chapters/season2/ch03_orb_m15_retranslation/figures.py --help
python chapters/season2/ch04_ict_order_blocks/run.py --help
python chapters/season2/ch04_ict_order_blocks/compare.py --help
python chapters/season2/ch04_ict_order_blocks/build_manifest.py --help
python chapters/season2/ch04_ict_order_blocks/verify.py --help
```

Season 1 の記事時点の既定終端は `2026-04-29T00:00:00Z` です。

Season 2 #1 は、既定で記事時点の参照 CSV を検証します。Yahoo Finance から現在のデータを取得し、記事の固定窓で再計算する場合は `--live` を付けます。

Season 2 #2 の最小コードは、OANDA practice APIの完了済み`M5 / BA`を独立した記事用SQLiteへ保存します。token、account情報、生レートはリポジトリへ保存しません。

固定manifestは、元SQLiteのhash、固定範囲の抽出hash、gap、M15完全性など、行データを含まない監査値だけを保持します。記事図は人工M5三本から再生成できます。

Season 2 #3 は、S2-2で固定したM5 SQLiteから集約したcomplete M15だけで、ORBのレンジを9:30–9:45 ETへ差し戻して検証します。参照結果の再計算には同一の非公開SQLiteが必要です。非公開DBがない環境でも、`verify.py`のrow-freeモードで集計の整合性を、`figures.py`で記事図の再生成を確認できます。

Season 2 #4 の工程6は、同じ固定M5 SQLiteからcomplete M15とNY17日足バイアスを派生し、本家Month 04と二次sweep→MSS→FVGのOB翻訳を同じbid/ask約定モデルで実行します。工程7は全検出ゾーンの有効時間を監査し、同じcomplete M15へpin済みOSSのデフォルトOB・Liquidityを適用して、価格帯と有効時間がともに交差する割合を対称に集計します。`run.py --repeat 2`で工程6を、`compare.py --repeat 2`でlifecycle・OSS・比較を含む工程7を再現できます。工程8の`build_manifest.py`は工程7全体を二度再実行してrow-free manifestを固定し、`verify.py`は非公開SQLiteなしでdetached hash・code hash・依存version・集計内整合・公開境界を検証します。価格・timestamp付き明細は`outputs/`だけへ置きます。

## テスト

```bash
pytest -q
```

## 再現性について

yfinance の 1h 足は取得できる期間に上限があります。公開後に再取得する場合は、`--end-date` を現在から約 730 日以内に収まる範囲へ変更してください。

また、yfinance のデータは取得時期、配信元の修正、欠損、仕様変更により結果が変わる場合があります。そのため、記事時点の主要な派生結果 CSV を `results/reference/` に配置しています。

## 参照出力

記事時点の主要な派生結果 CSV は `results/reference/` に配置しています。

- `ch04_wfa_two_pairs/`: USDJPY / GBPJPY の BB-MR WFA と 2 ペア比較表。
- `ch05_wfa_four_pairs/`: 4 ペアの BB-MR WFA と重心 summary。
- `ch06_donchian_compare/`: 4 ペアの BB-MR / Donchian WFA と戦略比較表。
- `ch07_physical_metrics/`: 8 grid x 5 fold の取引系列、fold 別集計、grid 別集計。
- `ch01_orb_1h_translation/`: Season 2 #1 ORB 検証の主版・参考版取引明細、固定 144 日区間集計、記事使用図。
- `ch02_minute_data_db/`: Season 2 #2 の行データなし監査manifestと、人工M5から生成したM5→M15模式図。
- `ch03_orb_m15_retranslation/`: Season 2 #3 のrow-free参照集計（主要指標、固定5区間、決済理由、ATR分類、session品質、quote幅）、hash、記事使用図。
- `ch04_ict_order_blocks/`: Season 2 #4 の入力・両翻訳・比較・OSS集計、依存version、code/stage hash、再実行commandを固定したrow-free manifest。

市場データの Parquet キャッシュ本体は含めていません。

## 注意事項

本リポジトリの検証は、取得時点の公開データと記載した条件に基づくものです。データ取得元の仕様変更、欠損、修正、配信遅延などにより、結果が変わる場合があります。本リポジトリは投資助言ではなく、売買判断はご自身の責任でお願いします。
