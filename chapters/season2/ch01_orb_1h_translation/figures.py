"""S2-1 ORB 記事の図4点を再生成する。

ATR分布と1トレード図は、Yahoo Financeから再取得した記事時点の窓を使う。
決済理由と固定144日区間は、記事掲載値を固定した参照CSVから描く。

実行例::

    python chapters/season2/ch01_orb_1h_translation/figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    """公開用リポジトリのルートを探す。"""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not find repository root")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bocchi_the_botter_repro.season2.orb import (  # noqa: E402
    ARTICLE_WINDOW_END_EXCLUSIVE,
    ARTICLE_WINDOW_START,
    ATR_HI,
    ATR_LO,
    ATR_N,
    RANGE_HOUR_ET,
    RR,
    article_window,
    load_et_bars,
)


CHAPTER = "ch01_orb_1h_translation"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "reference" / CHAPTER
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / CHAPTER / "figures"
TRADES_CSV = "trades_S2-1_ORB_USDJPY_main_net.csv"
SEGMENTS_CSV = "segments_S2-1_ORB_USDJPY_main_net.csv"

BLUE = "#2f5d9f"
LIGHT_BLUE = "#bfd7ea"
RED = "#b2473e"
GREEN = "#287a52"
GRAY = "#6b7280"
DARK = "#1f2937"

_FONT_CANDIDATES = (
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Yu Gothic",
    "YuGothic",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
)


def configure_japanese_font() -> str:
    """利用可能な日本語フォントを選び、無ければsans-serifへフォールバックする。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in _FONT_CANDIDATES if name in available), None)
    if selected is not None:
        plt.rcParams["font.family"] = selected
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [*_FONT_CANDIDATES, "DejaVu Sans"]
        selected = "sans-serif fallback"
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    """図に必要な列が揃っていることを確認する。"""
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} に必要な列がありません: {sorted(missing)}")


def save_figure(fig: plt.Figure, path: Path) -> None:
    """スマートフォンでも文字を読める解像度でPNGを保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")


def day_ratios(et_window: pd.DataFrame) -> pd.Series:
    """記事窓の評価可能日ごとに、レンジ幅÷ATRを返す。"""
    require_columns(
        et_window,
        ("et_date", "et_hour", "High", "Low", "ATR"),
        "記事窓データ",
    )
    ratios: list[float] = []
    for _, day in et_window.groupby("et_date", sort=True):
        range_bars = day[day["et_hour"] == RANGE_HOUR_ET]
        if range_bars.empty:
            continue
        range_bar = range_bars.iloc[0]
        atr = float(range_bar["ATR"])
        if pd.isna(atr) or atr <= 0:
            continue
        width = float(range_bar["High"]) - float(range_bar["Low"])
        ratios.append(width / atr)
    if not ratios:
        raise ValueError("記事窓にATRを評価できる日がありません")
    return pd.Series(ratios, dtype=float)


def draw_atr_ratio_hist(et_window: pd.DataFrame, output_dir: Path) -> None:
    """レンジ幅÷ATRの分布とフィルタ境界を描く。"""
    ratios = day_ratios(et_window)
    median = float(ratios.median())
    below = float((ratios < ATR_LO).mean() * 100.0)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.hist(
        ratios.clip(upper=4.0),
        bins=40,
        color=LIGHT_BLUE,
        edgecolor=BLUE,
        linewidth=0.8,
    )
    ax.axvline(
        ATR_LO,
        color=RED,
        linewidth=2.8,
        label=f"下限 {ATR_LO:g}（下回る日 {below:.0f}%）",
    )
    ax.axvline(
        ATR_HI,
        color=RED,
        linestyle=":",
        linewidth=2.2,
        label=f"上限 {ATR_HI:g}",
    )
    ax.axvline(
        median,
        color=BLUE,
        linestyle="--",
        linewidth=2.4,
        label=f"中央値 {median:.2f}",
    )
    ax.set_title(
        f"レンジ幅 ÷ ATR({ATR_N}) の分布\n"
        "USDJPY・1時間足 / 記事時点の720暦日",
        fontsize=18,
        fontweight="bold",
        color=DARK,
        pad=14,
    )
    ax.set_xlabel(f"9:00〜10:00のレンジ幅 ÷ ATR({ATR_N})", fontsize=14)
    ax.set_ylabel("日数", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir / "atr_ratio_hist.png")
    print(
        f"[ATR] n={len(ratios)} median={median:.3f} "
        f"below_{ATR_LO:g}={below:.1f}%"
    )


def draw_exit_reasons(trades: pd.DataFrame, output_dir: Path) -> None:
    """参照トレードCSVから決済理由の内訳を描く。"""
    require_columns(trades, ("exit_reason",), "トレードCSV")
    reasons = ("close_16", "sl", "tp")
    labels = ("16:00 時間切れ", "損切り", "利確")
    colors = (GRAY, RED, GREEN)
    counts = [int((trades["exit_reason"] == reason).sum()) for reason in reasons]
    unknown = int((~trades["exit_reason"].isin(reasons)).sum())
    if unknown:
        raise ValueError(f"未対応の決済理由が {unknown} 件あります")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(labels))
    bars = ax.barh(y, counts, color=colors, height=0.62)
    ax.set_yticks(y, labels=labels, fontsize=14)
    ax.invert_yaxis()
    max_count = max(counts, default=1)
    ax.set_xlim(0, max_count * 1.18)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max_count * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{count}本",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=DARK,
        )
    ax.set_title(
        f"{len(trades)}本は、どう終わったか\n"
        f"利確まで届いたのは{counts[2]}本",
        fontsize=18,
        fontweight="bold",
        color=DARK,
        pad=14,
    )
    ax.set_xlabel("トレード数", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "exit_reasons.png")
    print(f"[exit] {dict(zip(reasons, counts))}")


def draw_segment_results(segments: pd.DataFrame, output_dir: Path) -> None:
    """参照CSVの5×144日損益を、一行ずつ読める表型の図にする。"""
    require_columns(
        segments,
        ("segment", "start", "end_exclusive", "trade_count", "pnl"),
        "区間CSV",
    )
    data = segments.sort_values("segment").copy()
    if len(data) != 5 or data["segment"].nunique() != 5:
        raise ValueError(f"区間CSVは相異なる5区間である必要があります: {len(data)}行")
    data["start"] = pd.to_datetime(data["start"])
    data["end_exclusive"] = pd.to_datetime(data["end_exclusive"])
    if (data["end_exclusive"] <= data["start"]).any():
        raise ValueError("区間CSVの終了境界は開始日より後である必要があります")

    rows: list[tuple[str, str, int, float]] = []
    for row in data.itertuples(index=False):
        # CSVは半開区間。記事と図では読者が見る包含末日へ直して表示する。
        inclusive_end = row.end_exclusive - pd.Timedelta(days=1)
        rows.append(
            (
                f"区間{int(row.segment)}",
                f"{row.start:%Y-%m-%d} 〜 {inclusive_end:%Y-%m-%d}",
                int(row.trade_count),
                float(row.pnl),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 6.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.94,
        "固定144日区間の損益",
        fontsize=20,
        fontweight="bold",
        va="center",
        color=DARK,
    )
    ax.text(
        0.96,
        0.94,
        "区間1〜3はプラス / 区間4〜5はマイナス",
        fontsize=12,
        ha="right",
        va="center",
        color="#4b5563",
    )

    header_y = 0.82
    ax.add_patch(plt.Rectangle((0.03, header_y - 0.046), 0.94, 0.092, color="#334155"))
    columns = (
        (0.07, "区間", "left"),
        (0.21, "期間（ET）", "left"),
        (0.72, "取引数", "center"),
        (0.93, "損益", "right"),
    )
    for x, label, alignment in columns:
        ax.text(
            x,
            header_y,
            label,
            color="white",
            fontsize=12,
            fontweight="bold",
            ha=alignment,
            va="center",
        )

    first_y = 0.69
    row_height = 0.125
    for index, (segment, period, count, pnl) in enumerate(rows):
        y = first_y - index * row_height
        positive = pnl > 0
        accent = GREEN if positive else RED
        background = "#edf8f1" if positive else "#fceeed"
        verdict = "プラス" if positive else "マイナス"
        ax.add_patch(
            plt.Rectangle(
                (0.03, y - 0.052),
                0.94,
                0.104,
                color=background,
                ec="#d1d5db",
                lw=0.7,
            )
        )
        ax.add_patch(plt.Rectangle((0.03, y - 0.052), 0.012, 0.104, color=accent))
        ax.text(0.07, y, segment, fontsize=14, fontweight="bold", va="center", color=DARK)
        ax.text(0.21, y, period, fontsize=12, va="center", color="#374151")
        ax.text(0.72, y, f"{count}本", fontsize=12, ha="center", va="center", color="#374151")
        ax.text(
            0.93,
            y + 0.014,
            f"{pnl:+,.0f}円",
            fontsize=14,
            fontweight="bold",
            ha="right",
            va="center",
            color=accent,
        )
        ax.text(0.93, y - 0.027, verdict, fontsize=10, ha="right", va="center", color=accent)

    ax.text(
        0.04,
        0.045,
        "期間は開始日・末日の両方を含む表示 / 固定スプレッド控除後",
        fontsize=10.5,
        color=GRAY,
        va="center",
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "segment_results.png")


def select_live_trade_day(
    et_window: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """中央値付近の時間切れlongから、ライブ足が残る1本を選ぶ。"""
    require_columns(
        trades,
        (
            "date",
            "side",
            "entry_time",
            "entry_ref",
            "sl",
            "tp",
            "risk_width",
            "exit_time",
            "exit_price",
            "exit_reason",
        ),
        "トレードCSV",
    )
    require_columns(
        et_window,
        ("et_date", "et_hour", "High", "Low", "Close"),
        "記事窓データ",
    )
    candidates = trades[
        (trades["exit_reason"] == "close_16") & (trades["side"] == "long")
    ].copy()
    if candidates.empty:
        raise ValueError("時間切れで終わったlongトレードがありません")
    median_risk = float(trades["risk_width"].median())
    candidates["distance_from_median"] = (candidates["risk_width"] - median_risk).abs()

    for _, candidate in candidates.sort_values("distance_from_median").iterrows():
        entry_time = pd.Timestamp(candidate["entry_time"])
        exit_time = pd.Timestamp(candidate["exit_time"])
        if entry_time.hour != 11 or exit_time.hour != 16:
            continue
        trade_date = pd.Timestamp(candidate["date"]).date()
        day = et_window[
            (et_window["et_date"] == trade_date)
            & et_window["et_hour"].between(RANGE_HOUR_ET, 16)
        ].sort_index()
        if not day.empty and (day["et_hour"] == RANGE_HOUR_ET).any():
            return candidate, day
    raise ValueError("参照トレードに対応する記事窓のライブ足が見つかりません")


def draw_one_trade(
    et_window: pd.DataFrame,
    trades: pd.DataFrame,
    output_dir: Path,
) -> None:
    """ライブ1h足に、11:00エントリーと届かなかった2本の線を重ねる。"""
    trade, day = select_live_trade_day(et_window, trades)
    trade_date = pd.Timestamp(trade["date"]).date()
    range_bar = day[day["et_hour"] == RANGE_HOUR_ET].iloc[0]
    range_high = float(range_bar["High"])
    range_low = float(range_bar["Low"])
    entry = float(trade["entry_ref"])
    stop = float(trade["sl"])
    target = float(trade["tp"])
    exit_price = float(trade["exit_price"])

    x = np.arange(len(day))
    hours = day["et_hour"].astype(int).tolist()
    high = day["High"].to_numpy(dtype=float)
    low = day["Low"].to_numpy(dtype=float)
    close = day["Close"].to_numpy(dtype=float)
    exit_positions = np.flatnonzero(day["et_hour"].to_numpy(dtype=int) == 16)
    exit_x = int(exit_positions[0]) if len(exit_positions) else len(day) - 1

    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.vlines(x, low, high, color=BLUE, linewidth=1.5, alpha=0.5)
    ax.plot(x, close, color=BLUE, linewidth=2.3, marker="o", markersize=5, label="終値")
    ax.axhspan(
        range_low,
        range_high,
        color=LIGHT_BLUE,
        alpha=0.42,
        label="9:00〜10:00のレンジ",
    )
    ax.axhline(
        target,
        color=GREEN,
        linestyle="--",
        linewidth=2.5,
        label=f"利確（+{RR:g}R）",
    )
    ax.axhline(
        entry,
        color="black",
        linewidth=2.0,
        label="エントリー（11:00 ET）",
    )
    ax.axhline(
        stop,
        color=RED,
        linestyle="--",
        linewidth=2.5,
        label="損切り（レンジ下端）",
    )
    ax.scatter(
        [exit_x],
        [exit_price],
        color=BLUE,
        edgecolor="white",
        linewidth=1.2,
        s=100,
        zorder=5,
        label="16:00 時間切れ",
    )

    target_pips = abs(target - entry) * 100.0
    stop_pips = abs(entry - stop) * 100.0
    ax.text(
        0.02,
        0.92,
        f"利確まで約{target_pips:.0f}pips / 損切りまで約{stop_pips:.0f}pips",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color=DARK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 4},
    )
    ax.set_xticks(x, labels=[f"{hour}:00" for hour in hours])
    ax.set_xlabel("時刻（ET）", fontsize=14)
    ax.set_ylabel("USDJPY", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(
        f"線まで届かなかった1本\n{trade_date}・買い / 11:00 → 16:00",
        fontsize=18,
        fontweight="bold",
        color=DARK,
        pad=14,
    )
    ax.grid(alpha=0.2)
    ax.legend(
        fontsize=11,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, output_dir / "one_trade_schema.png")
    print(
        f"[trade] date={trade_date} entry=11:00 exit=16:00 "
        f"entry_price={entry:.3f} sl={stop:.3f} tp={target:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2-1 ORB記事の図4点を再生成します。")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"参照CSVのディレクトリ（既定: {DEFAULT_RESULTS_DIR}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"図の出力ディレクトリ（既定: {DEFAULT_OUTPUT_DIR}）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    trades_path = results_dir / TRADES_CSV
    segments_path = results_dir / SEGMENTS_CSV
    if not trades_path.is_file():
        raise FileNotFoundError(f"参照トレードCSVがありません: {trades_path}")
    if not segments_path.is_file():
        raise FileNotFoundError(f"参照区間CSVがありません: {segments_path}")

    selected_font = configure_japanese_font()
    print(f"[font] {selected_font}")
    print(
        "[live window] "
        f"[{ARTICLE_WINDOW_START}, {ARTICLE_WINDOW_END_EXCLUSIVE})"
    )

    trades = pd.read_csv(trades_path)
    segments = pd.read_csv(segments_path)
    live_bars = article_window(load_et_bars())
    if live_bars.empty:
        raise ValueError("記事時点のライブ窓を取得できませんでした")

    draw_atr_ratio_hist(live_bars, output_dir)
    draw_exit_reasons(trades, output_dir)
    draw_segment_results(segments, output_dir)
    draw_one_trade(live_bars, trades, output_dir)
    print(f"[OK] 4図を生成しました: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
