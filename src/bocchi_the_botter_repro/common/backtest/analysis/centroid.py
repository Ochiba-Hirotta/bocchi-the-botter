"""Four-pair centroid and pair-metric correlation analysis.

入力:
    - 各ペアの WFA long-format CSV (`wfa_bb_mr_<PAIR>.csv`)
    - ペア特性指標（VOL_atr / VOL_ret_std / VOL_drift / VOL_range）
        → 記事 #5 の検証時に固定した参照値

出力 (stdout):
    - ペアごとの重心 (N_centroid, K_centroid) + Sharpe>0 点数
    - 共通生存点リスト
    - degenerate grid 点（trades 中央値 = 0 または NaN）
    - 4 指標 × N/K 重心の Spearman ρ

実行:
    python -m bocchi_the_botter_repro.common.backtest.analysis.centroid

CSV 出力:
    --output 引数で `wfa_bb_mr_4pairs_summary.csv` を保存可能.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Pair metrics fixed for the article #5 reproduction.
PAIR_METRICS: dict[str, dict[str, float]] = {
    "USDJPY": {
        "VOL_atr": 0.001584,        # ATR/Close 中央値
        "VOL_ret_std": 0.001263,    # 1h log-ret σ
        "VOL_drift": 0.0371,        # |期間 Close 変動|
        "VOL_range": 0.1456,        # (max-min)/mean
    },
    "GBPJPY": {
        "VOL_atr": 0.001536,
        "VOL_ret_std": 0.001193,
        "VOL_drift": 0.1193,
        "VOL_range": 0.1737,
    },
    "EURJPY": {
        "VOL_atr": 0.001418,
        "VOL_ret_std": 0.001117,
        "VOL_drift": 0.1301,
        "VOL_range": 0.1933,
    },
    "AUDJPY": {
        "VOL_atr": 0.001930,
        "VOL_ret_std": 0.001517,
        "VOL_drift": 0.1222,
        "VOL_range": 0.2770,
    },
}

PAIRS: tuple[str, ...] = ("USDJPY", "GBPJPY", "EURJPY", "AUDJPY")
METRIC_NAMES: tuple[str, ...] = ("VOL_atr", "VOL_ret_std", "VOL_drift", "VOL_range")


@dataclass
class PairCentroid:
    """1 ペアの重心 + degenerate 情報."""

    pair: str
    n_centroid: float = float("nan")
    k_centroid: float = float("nan")
    n_positive: int = 0           # Sharpe > 0 の grid 数
    n_grid_total: int = 0
    degenerate_grids: list[tuple[int, float]] = field(default_factory=list)
    grid_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_wfa_csv(csv_path: Path) -> pd.DataFrame:
    """WFA long-format CSV を読み込み, grid × OOS Sharpe 中央値 / trades 中央値を返す."""
    df = pd.read_csv(csv_path)
    return df


def compute_grid_summary(df: pd.DataFrame) -> pd.DataFrame:
    """grid (BB_N, BB_K) ごとに OOS Sharpe / trades の中央値を計算."""
    return (
        df.groupby(["param_BB_N", "param_BB_K"])
        .agg(
            oos_sharpe_median=("oos_Sharpe Ratio", "median"),
            oos_trades_median=("oos_# Trades", "median"),
        )
        .reset_index()
    )


def compute_centroid(grid_summary: pd.DataFrame) -> PairCentroid:
    """質量中心方式で重心を算出.

    重み w_ij = OOS Sharpe 中央値（Sharpe > 0 の点のみ採用）.
    """
    pc = PairCentroid(pair="(temp)")
    pc.n_grid_total = len(grid_summary)

    # degenerate: trades 中央値 == 0 or NaN
    deg_mask = (grid_summary["oos_trades_median"].fillna(0) == 0)
    for _, row in grid_summary[deg_mask].iterrows():
        pc.degenerate_grids.append(
            (int(row["param_BB_N"]), float(row["param_BB_K"]))
        )

    # Sharpe > 0 のみ採用
    pos = grid_summary[grid_summary["oos_sharpe_median"] > 0].copy()
    pc.n_positive = len(pos)

    if len(pos) > 0:
        weights = pos["oos_sharpe_median"].to_numpy()
        n_arr = pos["param_BB_N"].to_numpy(dtype=float)
        k_arr = pos["param_BB_K"].to_numpy(dtype=float)
        total_w = weights.sum()
        if total_w > 0:
            pc.n_centroid = float((weights * n_arr).sum() / total_w)
            pc.k_centroid = float((weights * k_arr).sum() / total_w)

    pc.grid_summary = grid_summary
    return pc


def spearman_corr(x: list[float], y: list[float]) -> float:
    """Spearman 順位相関係数（NaN 含む場合は除外）."""
    arr = np.array([(a, b) for a, b in zip(x, y) if not (np.isnan(a) or np.isnan(b))])
    if len(arr) < 2:
        return float("nan")
    rx = pd.Series(arr[:, 0]).rank().to_numpy()
    ry = pd.Series(arr[:, 1]).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="4 ペア重心計算 + 4 指標相関分析 (#5 α 工程 5)"
    )
    parser.add_argument(
        "--research-dir",
        type=Path,
        default=Path(__file__).resolve().parents[5]
        / "results"
        / "reference"
        / "ch05_wfa_four_pairs",
        help="WFA CSV が置かれているディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="4 ペア統合 summary CSV の出力先（省略時は保存しない）",
    )
    args = parser.parse_args()

    research_dir: Path = args.research_dir
    print(f"# 4 ペア重心 + 4 指標相関分析 (#5 α)")
    print(f"research dir: {research_dir}")
    print()

    centroids: list[PairCentroid] = []
    for pair in PAIRS:
        csv = research_dir / f"wfa_bb_mr_{pair}.csv"
        if not csv.exists():
            print(f"[skip] {pair}: {csv} 不在")
            continue
        df = load_wfa_csv(csv)
        gs = compute_grid_summary(df)
        pc = compute_centroid(gs)
        pc.pair = pair
        centroids.append(pc)

    # ペアごとの重心レポート
    print("## ペアごとの重心 + degenerate")
    print(
        f"{'pair':>8} | {'N_cent':>8} | {'K_cent':>8} | "
        f"{'n_pos':>5} / {'total':>5} | degenerate"
    )
    print("-" * 80)
    for pc in centroids:
        deg_str = ", ".join(f"({n},{k})" for n, k in pc.degenerate_grids) or "-"
        if np.isnan(pc.n_centroid):
            n_str = "  N/A   "
            k_str = "  N/A   "
        else:
            n_str = f"{pc.n_centroid:>8.3f}"
            k_str = f"{pc.k_centroid:>8.3f}"
        print(
            f"{pc.pair:>8} | {n_str} | {k_str} | "
            f"{pc.n_positive:>5} / {pc.n_grid_total:>5} | {deg_str}"
        )
    print()

    # 共通生存点（4 ペアで OOS Sharpe 中央値 > 0 だった grid）
    print("## 共通生存点（4 ペア OOS Sharpe 中央値 > 0 の重複度）")
    grid_keys: dict[tuple[int, float], dict[str, float]] = {}
    for pc in centroids:
        for _, row in pc.grid_summary.iterrows():
            key = (int(row["param_BB_N"]), float(row["param_BB_K"]))
            grid_keys.setdefault(key, {})[pc.pair] = float(row["oos_sharpe_median"])

    rows: list[dict[str, object]] = []
    for key, vals in sorted(grid_keys.items()):
        n_alive = sum(1 for v in vals.values() if v > 0 and not np.isnan(v))
        rows.append(
            {
                "BB_N": key[0],
                "BB_K": key[1],
                "alive_in": n_alive,
                **{p: vals.get(p, float("nan")) for p in PAIRS},
            }
        )
    grid_table = pd.DataFrame(rows).sort_values(
        by=["alive_in", "BB_N", "BB_K"], ascending=[False, True, True]
    )
    print(grid_table.to_string(index=False))
    print()

    # BB(14, 2.0) ペア横断成績
    print("## BB(14, 2.0) ペア横断成績")
    target = grid_table[(grid_table["BB_N"] == 14) & (grid_table["BB_K"] == 2.0)]
    print(target.to_string(index=False))
    print()

    # degenerate map
    print("## degenerate grid マップ（trades 中央値 = 0）")
    for pc in centroids:
        deg_str = ", ".join(f"({n},{k})" for n, k in pc.degenerate_grids) or "なし"
        print(f"  {pc.pair}: {deg_str}")
    print()

    # 4 指標 × 重心 N/K の Spearman ρ
    print("## 4 指標 × 重心 (N, K) Spearman ρ（H1〜H3 検証）")
    valid = [pc for pc in centroids if not np.isnan(pc.n_centroid)]
    n_valid = len(valid)
    print(f"重心が定義されたペア: {n_valid} / {len(centroids)}")
    if n_valid >= 2:
        print(
            f"{'metric':>12} | {'ρ(metric, N_cent)':>20} | "
            f"{'ρ(metric, K_cent)':>20} | n_pairs"
        )
        print("-" * 80)
        for metric in METRIC_NAMES:
            mvals = [PAIR_METRICS[pc.pair][metric] for pc in valid]
            n_vals = [pc.n_centroid for pc in valid]
            k_vals = [pc.k_centroid for pc in valid]
            rho_n = spearman_corr(mvals, n_vals)
            rho_k = spearman_corr(mvals, k_vals)
            print(
                f"{metric:>12} | {rho_n:>20.3f} | {rho_k:>20.3f} | {n_valid}"
            )
        print()
        print(
            "判定（|ρ| ≥ 0.7 → 仮説支持 / |ρ| ≤ 0.3 → 不支持 / "
            "それ以外は弱い相関）"
        )
        # 標本数 n_valid における信頼区間の広さに留意
        if n_valid < 4:
            print(
                f"⚠ 標本数 {n_valid} → 判定は探索的"
                f"（重心未定義のペアあり, 結論は『強い兆候』止まり）"
            )
    else:
        print("重心定義ペアが 2 未満のため相関計算不可")
    print()

    # CSV 出力
    if args.output:
        summary_rows: list[dict[str, object]] = []
        for pc in centroids:
            row: dict[str, object] = {
                "pair": pc.pair,
                "n_centroid": pc.n_centroid,
                "k_centroid": pc.k_centroid,
                "n_positive": pc.n_positive,
                "n_grid_total": pc.n_grid_total,
                "n_degenerate": len(pc.degenerate_grids),
            }
            row.update(PAIR_METRICS.get(pc.pair, {}))
            summary_rows.append(row)
        out_df = pd.DataFrame(summary_rows)
        out_df.to_csv(args.output, index=False)
        print(f"[output] saved summary CSV to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
