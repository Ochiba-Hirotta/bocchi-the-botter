"""S2-2の記事用模式図を人工M5から再生成する。

実レートや上流SQLiteは使わない。三本の人工M5を公開実装でM15へ
集約し、M15へ残るものと、M15だけでは戻らないものを一枚に描く。

実行例::

    python chapters/season2/ch02_minute_data_db/figures.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "bocchi-the-botter-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd


def find_repo_root(start: Path) -> Path:
    """公開用リポジトリのルートを探す。"""

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src"
        ).is_dir():
            return candidate
    raise RuntimeError("Could not find repository root")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bocchi_the_botter_repro.season2.minute_data import (  # noqa: E402
    SOURCE,
    aggregate_m5_to_m15,
)


CHAPTER = "ch02_minute_data_db"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "reference" / CHAPTER / "figures"
FIGURE_NAME = "m5_to_m15_information_loss.png"

GREEN = "#287a52"
RED = "#b2473e"
BLUE = "#2f5d9f"
DARK = "#1f2937"
GRAY = "#64748b"
LIGHT_BLUE = "#e8f1fb"
LIGHT_RED = "#fdf0ef"

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
    """利用可能な日本語フォントを選ぶ。"""

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


def artificial_m5_frame() -> pd.DataFrame:
    """記事図専用の人工M5三本を返す。"""

    start = int(pd.Timestamp("2026-03-09T13:30:00Z").timestamp())
    fetched_at = start + 3_600
    bid_rows = (
        # offset, volume, open, high, low, close
        (0, 80, 150.000, 150.100, 149.960, 150.060),
        (300, 95, 150.060, 150.180, 150.020, 150.140),
        (600, 75, 150.140, 150.160, 150.040, 150.080),
    )
    records: list[dict[str, object]] = []
    for offset, volume, opened, high, low, closed in bid_rows:
        records.append(
            {
                "source": SOURCE,
                "instrument": "USD_JPY",
                "granularity": "M5",
                "price": "BA",
                "ts_utc": start + offset,
                "fetched_at_utc": fetched_at,
                "complete": 1,
                "volume": volume,
                "bid_open": opened,
                "bid_high": high,
                "bid_low": low,
                "bid_close": closed,
                "ask_open": opened + 0.003,
                "ask_high": high + 0.003,
                "ask_low": low + 0.003,
                "ask_close": closed + 0.003,
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_artificial_m15(frame: pd.DataFrame) -> pd.Series:
    """公開実装で人工M5を集約し、期待値を検査して一行を返す。"""

    start = int(frame["ts_utc"].min())
    result = aggregate_m5_to_m15(
        frame,
        start_inclusive=start,
        end_exclusive=start + 900,
    )
    if len(result.candles) != 1 or not result.incomplete_buckets.empty:
        raise ValueError("人工M5三本が完全なM15一本になりませんでした")
    candle = result.candles.iloc[0]
    expected = {
        "bid_open": 150.000,
        "bid_high": 150.180,
        "bid_low": 149.960,
        "bid_close": 150.080,
        "ask_open": 150.003,
        "ask_high": 150.183,
        "ask_low": 149.963,
        "ask_close": 150.083,
    }
    for column, value in expected.items():
        if not math.isclose(float(candle[column]), value, abs_tol=1e-12):
            raise ValueError(f"人工M15の{column}が期待値と一致しません")
    if int(candle["volume"]) != 250:
        raise ValueError("人工M15のvolume合計が250ではありません")
    return candle


def draw_candle(
    ax: plt.Axes,
    *,
    x: float,
    opened: float,
    high: float,
    low: float,
    closed: float,
    width: float,
) -> None:
    """一本の簡易ローソク足を描く。"""

    color = GREEN if closed >= opened else RED
    ax.vlines(x, low, high, color=color, linewidth=2.4, zorder=2)
    body_low = min(opened, closed)
    body_height = max(abs(closed - opened), 0.002)
    ax.add_patch(
        Rectangle(
            (x - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.4,
            zorder=3,
        )
    )


def style_price_axis(ax: plt.Axes) -> None:
    """左右のローソク足で共通の価格軸を使う。"""

    ax.set_ylim(149.94, 150.20)
    ax.set_yticks((149.96, 150.04, 150.12, 150.20))
    ax.tick_params(axis="y", labelsize=10, colors=GRAY)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def draw_information_loss(
    frame: pd.DataFrame,
    m15: pd.Series,
    output_dir: Path,
) -> Path:
    """M5三本からM15一本への集約模式図を保存する。"""

    fig = plt.figure(figsize=(12, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(2.2, 1.05),
        width_ratios=(1.7, 0.3, 1.0),
        hspace=0.35,
        wspace=0.12,
    )
    left = fig.add_subplot(grid[0, 0])
    arrow = fig.add_subplot(grid[0, 1])
    right = fig.add_subplot(grid[0, 2], sharey=left)
    notes = fig.add_subplot(grid[1, :])

    fig.suptitle(
        "M5 三本 → M15 一本",
        fontsize=20,
        fontweight="bold",
        color=DARK,
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "三本が揃った区間だけを集約 / 人工値（実レート不使用）",
        ha="center",
        fontsize=12.5,
        color=GRAY,
    )

    x_values = (0.8, 2.0, 3.2)
    labels = ("09:30\nvolume 80", "09:35\nvolume 95", "09:40\nvolume 75")
    for x, row in zip(x_values, frame.itertuples(index=False)):
        draw_candle(
            left,
            x=x,
            opened=float(row.bid_open),
            high=float(row.bid_high),
            low=float(row.bid_low),
            closed=float(row.bid_close),
            width=0.42,
        )
    left.set_xlim(0.2, 3.8)
    left.set_xticks(x_values, labels=labels)
    left.tick_params(axis="x", labelsize=11, colors=DARK, pad=8)
    left.set_title("M5 × 3（SQLiteに残す）", fontsize=16, fontweight="bold", color=BLUE)
    left.set_ylabel("人工bid価格", fontsize=11, color=GRAY)
    style_price_axis(left)

    arrow.axis("off")
    arrow.annotate(
        "",
        xy=(0.95, 0.52),
        xytext=(0.05, 0.52),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 3.0, "color": BLUE},
    )
    arrow.text(
        0.5,
        0.62,
        "集約",
        transform=arrow.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=BLUE,
    )

    draw_candle(
        right,
        x=1.0,
        opened=float(m15["bid_open"]),
        high=float(m15["bid_high"]),
        low=float(m15["bid_low"]),
        closed=float(m15["bid_close"]),
        width=0.55,
    )
    right.set_xlim(0.25, 1.75)
    right.set_xticks((1.0,), labels=("09:30 M15\nvolume 250",))
    right.tick_params(axis="x", labelsize=11, colors=DARK, pad=8)
    right.tick_params(axis="y", labelleft=False)
    right.set_title("M15 × 1（派生DataFrame）", fontsize=16, fontweight="bold", color=BLUE)
    style_price_axis(right)

    notes.set_xlim(0, 1)
    notes.set_ylim(0, 1)
    notes.axis("off")
    boxes = (
        (
            0.025,
            LIGHT_BLUE,
            BLUE,
            "M15に残る",
            "開始時刻 / bid・ask別 OHLC / volume合計",
        ),
        (
            0.515,
            LIGHT_RED,
            RED,
            "M15だけでは戻らない",
            "M5三本それぞれのOHLC / 高値・安値に触れた順序\n足内の価格経路",
        ),
    )
    for x, background, accent, heading, body in boxes:
        notes.add_patch(
            FancyBboxPatch(
                (x, 0.20),
                0.46,
                0.60,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor=background,
                edgecolor=accent,
                linewidth=1.4,
            )
        )
        notes.text(
            x + 0.025,
            0.65,
            heading,
            fontsize=15,
            fontweight="bold",
            color=accent,
            va="center",
        )
        notes.text(
            x + 0.025,
            0.40,
            body,
            fontsize=12.5,
            color=DARK,
            va="center",
            wrap=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / FIGURE_NAME
    fig.savefig(output, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2-2の記事用模式図を再生成します。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"図の出力ディレクトリ（既定: {DEFAULT_OUTPUT_DIR}）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_font = configure_japanese_font()
    frame = artificial_m5_frame()
    m15 = aggregate_artificial_m15(frame)
    output = draw_information_loss(frame, m15, args.output_dir.resolve())
    print(f"[font] {selected_font}")
    print(
        "[artificial M15] "
        f"open={m15['bid_open']:.3f} high={m15['bid_high']:.3f} "
        f"low={m15['bid_low']:.3f} close={m15['bid_close']:.3f} "
        f"volume={int(m15['volume'])}"
    )
    print(f"[figure] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
