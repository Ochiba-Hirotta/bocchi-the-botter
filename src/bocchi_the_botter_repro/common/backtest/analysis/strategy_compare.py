"""Cross-strategy analysis for BB-MR vs Donchian.

§10.2 の 7 表のうち 6 表をカバー (Expectancy / spread は TODO-2 で別タスク化):

    [1] 共通生存点表 (BB-MR / Donchian それぞれ)
    [2] fold 単位照合表 (4 ペア × 5 fold = 20 行 + W 判定)
    [3] 重心比較表 (4 ペア × 2 戦略)
    [4] Spearman ρ (BB-MR 重心 N vs Donchian 重心 DC_N)
    [5] VOL_drift × Donchian Sharpe 中央値 相関 (ペア別)
    [6] MAX_BARS 強制クローズ比率表 (D3 / §11 Discussion 素材)

入力 (8 CSV):
    results/reference/ch06_donchian_compare/wfa_bb_mr_<PAIR>_2026-04-29.csv × 4
    results/reference/ch06_donchian_compare/wfa_donchian_<PAIR>_2026-04-29.csv × 4

出力 1 (CSV, --output 指定時):
    outputs/ch06_donchian_compare/wfa_strategy_compare.csv
        fold 単位照合 long format (4 ペア × 5 fold = 20 行).

出力 2 (stdout, markdown):
    [1]〜[6] の表 + X/Y/Z 主判定候補.

実行:
    python -m bocchi_the_botter_repro.common.backtest.analysis.strategy_compare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .centroid import PAIR_METRICS, spearman_corr

PAIRS: tuple[str, ...] = ("USDJPY", "GBPJPY", "EURJPY", "AUDJPY")
CACHE_DATE: str = "2026-04-29"
W_THRESHOLD: int = 11  # 符号逆転 fold >= 11/20 で W 成立


def grid_keys(strategy: str) -> tuple[str, str]:
    if strategy == "bb_mr":
        return "param_BB_N", "param_BB_K"
    if strategy == "donchian":
        return "param_DC_N", "param_DC_EXIT"
    raise ValueError(f"unknown strategy: {strategy}")


def load_csv(research_dir: Path, strategy: str, pair: str) -> pd.DataFrame:
    csv = research_dir / f"wfa_{strategy}_{pair}_{CACHE_DATE}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing: {csv}")
    return pd.read_csv(csv)


def compute_grid_summary(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    n_col, exit_col = grid_keys(strategy)
    return (
        df.groupby([n_col, exit_col])
        .agg(
            oos_sharpe_median=("oos_Sharpe Ratio", "median"),
            oos_trades_median=("oos_# Trades", "median"),
        )
        .reset_index()
    )


def compute_centroid(
    grid_summary_df: pd.DataFrame, strategy: str
) -> tuple[float, float, int, int]:
    """質量中心方式. weight = OOS Sharpe 中央値 (Sharpe>0 のみ)."""
    n_col, exit_col = grid_keys(strategy)
    n_total = len(grid_summary_df)
    pos = grid_summary_df[grid_summary_df["oos_sharpe_median"] > 0]
    n_pos = len(pos)
    if n_pos == 0:
        return float("nan"), float("nan"), n_pos, n_total
    weights = pos["oos_sharpe_median"].to_numpy()
    n_arr = pos[n_col].to_numpy(dtype=float)
    e_arr = pos[exit_col].to_numpy(dtype=float)
    total_w = weights.sum()
    if total_w <= 0:
        return float("nan"), float("nan"), n_pos, n_total
    return (
        float((weights * n_arr).sum() / total_w),
        float((weights * e_arr).sum() / total_w),
        n_pos,
        n_total,
    )


def common_survival(
    per_pair_summary: dict[str, pd.DataFrame], strategy: str
) -> pd.DataFrame:
    """戦略の grid summary 4 ペア分から共通生存点リストを作成."""
    n_col, exit_col = grid_keys(strategy)
    grid_alive: dict[tuple[float, float], dict[str, float]] = {}
    for pair, gs in per_pair_summary.items():
        for _, row in gs.iterrows():
            key = (float(row[n_col]), float(row[exit_col]))
            grid_alive.setdefault(key, {})[pair] = float(row["oos_sharpe_median"])
    rows: list[dict[str, object]] = []
    for key, vals in sorted(grid_alive.items()):
        alive_in = sum(1 for v in vals.values() if v > 0 and not np.isnan(v))
        rows.append(
            {
                n_col: key[0],
                exit_col: key[1],
                "alive_in": alive_in,
                **{p: vals.get(p, float("nan")) for p in PAIRS},
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["alive_in", n_col, exit_col], ascending=[False, True, True]
    ).reset_index(drop=True)


def fold_alignment(
    per_pair_bb: dict[str, pd.DataFrame], per_pair_dc: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """fold 単位で両戦略の OOS Sharpe 中央値を照合 (W 判定の主役).

    集約: 各 (pair, fold) で 20 grid の OOS Sharpe 中央値を fold の代表値として採用.
    パラメータ依存性を抑え, 戦略間の比較を fold × pair の 20 セルに揃える.
    """
    rows: list[dict[str, object]] = []
    for pair in PAIRS:
        bb = per_pair_bb[pair]
        dc = per_pair_dc[pair]
        folds = sorted(set(bb["fold"].unique()) & set(dc["fold"].unique()))
        for fold in folds:
            bb_f = bb[bb["fold"] == fold]
            dc_f = dc[dc["fold"] == fold]
            bb_med = float(bb_f["oos_Sharpe Ratio"].median())
            dc_med = float(dc_f["oos_Sharpe Ratio"].median())
            regimes = bb_f["test_regime"].unique()
            regime = str(regimes[0]) if len(regimes) >= 1 else "?"
            close_return = float(bb_f["test_close_return"].iloc[0])
            sign_flip = (bb_med > 0 and dc_med < 0) or (bb_med < 0 and dc_med > 0)
            if sign_flip:
                label = "W"
            elif bb_med > 0 and dc_med > 0:
                label = "++"
            elif bb_med < 0 and dc_med < 0:
                label = "--"
            else:
                label = "0"
            bb_max = float(bb_f["oos_max_bars_close_ratio"].median())
            dc_max = float(dc_f["oos_max_bars_close_ratio"].median())
            rows.append(
                {
                    "pair": pair,
                    "fold": int(fold),
                    "regime": regime,
                    "test_close_return": close_return,
                    "bb_mr_oos_sharpe_median": bb_med,
                    "donchian_oos_sharpe_median": dc_med,
                    "label": label,
                    "sign_flip": sign_flip,
                    "bb_mr_max_bars_ratio_median": bb_max,
                    "donchian_max_bars_ratio_median": dc_max,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BB-MR vs Donchian 戦略横断統合分析 (#6 §13 工程 9-A)"
    )
    parser.add_argument(
        "--research-dir",
        type=Path,
        default=Path(__file__).resolve().parents[5]
        / "results"
        / "reference"
        / "ch06_donchian_compare",
        help="WFA CSV の置き場",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="fold 単位照合 CSV の出力先 (省略時は保存しない)",
    )
    args = parser.parse_args()

    research_dir: Path = args.research_dir
    print("# BB-MR vs Donchian 戦略横断統合分析 (#6 §13 工程 9-A)")
    print()
    print(f"- research dir: `{research_dir}`")
    print(f"- cache date: `{CACHE_DATE}`")
    print(f"- pairs: {', '.join(PAIRS)}")
    print()

    bb_per_pair: dict[str, pd.DataFrame] = {}
    dc_per_pair: dict[str, pd.DataFrame] = {}
    bb_gs: dict[str, pd.DataFrame] = {}
    dc_gs: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        bb_per_pair[pair] = load_csv(research_dir, "bb_mr", pair)
        dc_per_pair[pair] = load_csv(research_dir, "donchian", pair)
        bb_gs[pair] = compute_grid_summary(bb_per_pair[pair], "bb_mr")
        dc_gs[pair] = compute_grid_summary(dc_per_pair[pair], "donchian")

    # [1] 共通生存点表
    print("## [1] 共通生存点表")
    print()
    print("### BB-MR")
    bb_common = common_survival(bb_gs, "bb_mr")
    print(bb_common.to_string(index=False))
    bb_common_count = int((bb_common["alive_in"] == 4).sum())
    print()
    print(f"4 ペア共通生存点 (alive_in == 4): **{bb_common_count}**")
    print()
    print("### Donchian")
    dc_common = common_survival(dc_gs, "donchian")
    print(dc_common.to_string(index=False))
    dc_common_count = int((dc_common["alive_in"] == 4).sum())
    print()
    print(f"4 ペア共通生存点 (alive_in == 4): **{dc_common_count}**")
    print()

    # [2] fold 単位照合表
    print("## [2] fold 単位照合表 (W 判定の主役)")
    print()
    fold_df = fold_alignment(bb_per_pair, dc_per_pair)
    print(fold_df.to_string(index=False))
    sign_flip_count = int(fold_df["sign_flip"].sum())
    total_folds = len(fold_df)
    print()
    print(f"符号逆転 fold 数: **{sign_flip_count} / {total_folds}**")
    if sign_flip_count >= W_THRESHOLD:
        print(f"→ **W 判定 (補完的) 成立** (≥ {W_THRESHOLD}/{total_folds})")
    else:
        print(f"→ W 判定不成立 (< {W_THRESHOLD}/{total_folds})")
    # ラベル分布
    label_counts = fold_df["label"].value_counts().to_dict()
    print(f"ラベル分布: {label_counts}  (W=符号逆転 / ++=両正 / --=両負 / 0=境界)")
    print()

    # [3] 重心比較表
    print("## [3] 重心比較表 (4 ペア × 2 戦略)")
    print()
    centroid_rows: list[dict[str, object]] = []
    for pair in PAIRS:
        bb_n, bb_e, bb_pos, bb_total = compute_centroid(bb_gs[pair], "bb_mr")
        dc_n, dc_e, dc_pos, dc_total = compute_centroid(dc_gs[pair], "donchian")
        centroid_rows.append(
            {
                "pair": pair,
                "bb_mr_N_cent": bb_n,
                "bb_mr_K_cent": bb_e,
                "bb_mr_n_pos": f"{bb_pos}/{bb_total}",
                "donchian_DC_N_cent": dc_n,
                "donchian_DC_EXIT_cent": dc_e,
                "donchian_n_pos": f"{dc_pos}/{dc_total}",
            }
        )
    centroid_df = pd.DataFrame(centroid_rows)
    print(centroid_df.to_string(index=False))
    print()

    # [4] Spearman ρ (BB-MR N_cent vs Donchian DC_N_cent)
    print("## [4] Spearman ρ (BB-MR 重心 N vs Donchian 重心 DC_N)")
    print()
    bb_ns = centroid_df["bb_mr_N_cent"].tolist()
    dc_ns = centroid_df["donchian_DC_N_cent"].tolist()
    rho = spearman_corr(bb_ns, dc_ns)
    n_valid = sum(1 for a, b in zip(bb_ns, dc_ns) if not (np.isnan(a) or np.isnan(b)))
    print(f"  ρ = {rho:.3f}  (n_valid = {n_valid} / 4)")
    if n_valid < 4:
        print(f"  ⚠ 重心未定義のペアあり → 探索的兆候止まり")
    print()

    # [5] VOL_drift × Donchian Sharpe 中央値
    print("## [5] VOL_drift × Donchian Sharpe 中央値 相関")
    print()
    drift_rows: list[dict[str, object]] = []
    for pair in PAIRS:
        vol_drift = PAIR_METRICS[pair]["VOL_drift"]
        dc_overall = float(dc_per_pair[pair]["oos_Sharpe Ratio"].median())
        drift_rows.append(
            {
                "pair": pair,
                "VOL_drift": vol_drift,
                "donchian_oos_sharpe_overall_median": dc_overall,
            }
        )
    drift_df = pd.DataFrame(drift_rows)
    print(drift_df.to_string(index=False))
    rho_drift = spearman_corr(
        drift_df["VOL_drift"].tolist(),
        drift_df["donchian_oos_sharpe_overall_median"].tolist(),
    )
    print()
    print(f"  Spearman ρ (VOL_drift, Donchian Sharpe 中央値) = {rho_drift:.3f}  (n = 4)")
    print()

    # [6] MAX_BARS 強制クローズ比率表
    print("## [6] MAX_BARS 強制クローズ比率表 (oos_max_bars_close_ratio 中央値)")
    print()
    max_rows: list[dict[str, object]] = []
    for pair in PAIRS:
        bb_max = float(bb_per_pair[pair]["oos_max_bars_close_ratio"].median())
        dc_max = float(dc_per_pair[pair]["oos_max_bars_close_ratio"].median())
        max_rows.append(
            {
                "pair": pair,
                "bb_mr_max_bars_ratio_median": bb_max,
                "donchian_max_bars_ratio_median": dc_max,
            }
        )
    max_df = pd.DataFrame(max_rows)
    print(max_df.to_string(index=False))
    print()

    # X/Y/Z 主判定候補 (W は [2] で判定済)
    print("## [付] X/Y/Z 主判定候補")
    print()
    print(
        f"- 4 ペア共通生存点: BB-MR={bb_common_count} / Donchian={dc_common_count}"
    )
    if bb_common_count == 0 and dc_common_count == 0:
        verdict = "**X 候補** (両戦略でゼロ — ペア依存)"
    elif dc_common_count >= 1 and bb_common_count == 0:
        verdict = "**Y 候補** (Donchian のみ生存 — BB-MR 固有の問題)"
    elif dc_common_count < bb_common_count:
        verdict = "**Z 候補** (Donchian < BB-MR)"
    else:
        verdict = "判定保留 (Donchian ≥ BB-MR かつ両方 ≥ 1)"
    print(f"- 主判定: {verdict}")
    if sign_flip_count >= W_THRESHOLD:
        print(f"- 副判定: **W 成立** (符号逆転 {sign_flip_count}/{total_folds})")
    print()

    # CSV 出力
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(args.output, index=False)
        print(f"[output] saved fold alignment CSV to `{args.output}`")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
