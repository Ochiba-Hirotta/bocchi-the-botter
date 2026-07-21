from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


BLUE = "#4472C4"
GREEN = "#4F9D69"
RED = "#C95D63"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9DEE7"
INITIAL_EQUITY_JPY = 1_000_000.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(reference_dir: Path) -> dict[str, Any]:
    manifest_path = reference_dir / "manifest.json"
    digest_path = reference_dir / "manifest.sha256"
    manifest_digest = file_sha256(manifest_path)
    expected_digest = digest_path.read_text(encoding="utf-8").strip().split()[0]
    if manifest_digest != expected_digest:
        raise ValueError("manifest.json does not match manifest.sha256")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    official = payload["detectors"]["official"]
    summary = official["summary"]
    exit_reasons = official["exit_reasons"]
    segments = official["segments"]
    if int(summary["trade_count"]) != sum(int(value) for value in exit_reasons.values()):
        raise ValueError("exit-reason counts do not sum to the official trade count")
    if int(summary["trade_count"]) != sum(
        int(segment["trade_count"]) for segment in segments
    ):
        raise ValueError("segment trade counts do not sum to the official trade count")
    final_from_segments = INITIAL_EQUITY_JPY + sum(
        float(segment["pnl_jpy"]) for segment in segments
    )
    if abs(final_from_segments - float(summary["final_equity"])) > 1e-6:
        raise ValueError("segment PnL does not reconcile to final equity")
    return payload


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "bocchi-the-botter-repro"},
    )
    plt.close(figure)


def draw_candle(
    axis: plt.Axes,
    x: float,
    opened: float,
    high: float,
    low: float,
    closed: float,
    *,
    alpha: float = 1.0,
    width: float = 0.55,
) -> None:
    color = GREEN if closed >= opened else RED
    axis.vlines(x, low, high, color=color, linewidth=1.15, alpha=alpha, zorder=3)
    body_low = min(opened, closed)
    body_height = max(abs(closed - opened), 0.08)
    axis.add_patch(
        Rectangle(
            (x - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            zorder=4,
        )
    )


def plot_bullish_ob_schema(output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9.0, 5.0))

    pre_candles = [
        (5.8, 6.2, 5.5, 6.0),
        (6.0, 6.3, 5.6, 5.7),
        (5.7, 6.0, 5.3, 5.5),
        (5.5, 5.9, 5.2, 5.8),
        (5.8, 6.1, 5.4, 5.6),
        (5.6, 5.9, 5.1, 5.3),
        (5.3, 5.7, 5.0, 5.5),
        (5.5, 5.8, 5.0, 5.2),
        (5.2, 5.6, 4.9, 5.4),
        (5.4, 5.7, 5.0, 5.1),
        (5.1, 5.5, 4.8, 5.3),
        (5.3, 5.6, 4.9, 5.0),
        (5.0, 5.4, 4.7, 5.2),
        (5.2, 5.5, 4.8, 5.0),
        (5.0, 5.3, 4.6, 4.8),
        (4.8, 5.2, 4.5, 5.0),
        (5.0, 5.3, 4.6, 4.8),
        (4.8, 5.1, 4.3, 4.5),
        (4.5, 4.9, 4.1, 4.7),
        (4.8, 5.1, 2.6, 3.5),
    ]
    post_candles = [
        (3.5, 5.0, 3.2, 4.6),
        (4.6, 6.4, 4.4, 6.0),
        (6.0, 7.0, 5.8, 6.6),
        (6.6, 6.8, 5.5, 5.9),
        (5.9, 6.1, 4.75, 5.1),
        (5.1, 6.4, 4.9, 6.2),
        (6.2, 7.7, 6.0, 7.4),
        (7.4, 8.7, 7.2, 8.5),
    ]

    axis.axvspan(-0.45, 19.45, color=LIGHT_GRAY, alpha=0.28, zorder=0)
    for index, candle in enumerate(pre_candles):
        draw_candle(
            axis,
            float(index),
            *candle,
            alpha=1.0 if index == 19 else 0.42,
        )
    for offset, candle in enumerate(post_candles, start=20):
        draw_candle(axis, float(offset), *candle)

    candidate_x = 19.0
    candidate_high = 5.1
    entry = 4.8
    stop = 2.6
    target = 8.4
    activation_x = 21.0
    fill_x = 24.0

    axis.add_patch(
        Rectangle(
            (candidate_x - 0.35, stop),
            fill_x - candidate_x + 0.7,
            entry - stop,
            facecolor=BLUE,
            edgecolor=BLUE,
            alpha=0.12,
            linewidth=1.2,
            zorder=1,
        )
    )
    axis.hlines(
        candidate_high,
        candidate_x,
        activation_x + 0.4,
        colors=GRAY,
        linestyles="--",
        linewidth=1.2,
        zorder=2,
    )
    axis.hlines(
        entry,
        candidate_x,
        fill_x + 0.45,
        colors=BLUE,
        linewidth=1.5,
        zorder=2,
    )
    axis.hlines(
        stop,
        candidate_x - 0.4,
        fill_x + 0.45,
        colors=RED,
        linestyles="--",
        linewidth=1.4,
        zorder=2,
    )
    axis.hlines(
        target,
        activation_x + 0.7,
        28.2,
        colors=GREEN,
        linestyles="--",
        linewidth=1.4,
        zorder=2,
    )

    axis.annotate(
        "W = 20 lookback\n(translation choice)",
        xy=(9.5, 9.15),
        ha="center",
        va="center",
        fontsize=10,
    )
    axis.annotate(
        "",
        xy=(-0.2, 8.8),
        xytext=(19.2, 8.8),
        arrowprops={"arrowstyle": "<->", "color": GRAY, "linewidth": 1.2},
    )
    axis.annotate(
        "candidate\ndown-close + lowest bid low",
        xy=(candidate_x, 3.3),
        xytext=(14.0, 1.65),
        ha="left",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 1.0},
    )
    axis.annotate(
        "activation\nlater bid high breaks candidate high",
        xy=(activation_x, 6.35),
        xytext=(20.7, 7.55),
        ha="center",
        va="bottom",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 1.0},
    )
    axis.annotate(
        "limit = OB bid open\nfill when ask low reaches it",
        xy=(fill_x, entry),
        xytext=(25.0, 3.85),
        ha="left",
        va="center",
        fontsize=9,
        color="#333333",
        arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 1.1},
    )
    axis.text(28.35, target, "TP: eligible confirmed\ndaily swing high", va="center", fontsize=9)
    axis.text(28.35, stop, "SL: OB bid low", va="center", fontsize=9)
    axis.text(
        0.0,
        1.15,
        "Equal extreme candidates are tie-broken by the largest candle body.",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#555555",
    )

    axis.set_xlim(-0.8, 32.2)
    axis.set_ylim(0.8, 9.65)
    axis.set_title("Frozen bullish order-block rule (schematic)", pad=12)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    output_path = output_dir / "bullish_ob_rule_schema.png"
    save_figure(figure, output_path)
    return output_path


def plot_segment_boundary_balance(
    manifest: dict[str, Any], output_dir: Path
) -> Path:
    official = manifest["detectors"]["official"]
    segments = official["segments"]
    balances = [INITIAL_EQUITY_JPY]
    for segment in segments:
        balances.append(balances[-1] + float(segment["pnl_jpy"]))
    values = [balance / 10_000.0 for balance in balances]

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    for index, segment in enumerate(segments):
        pnl = float(segment["pnl_jpy"])
        color = GREEN if pnl > 0 else RED
        axis.axvspan(index, index + 1, color=color, alpha=0.08, zorder=0)
        axis.text(
            index + 0.5,
            103.0,
            f"S{index + 1}  {pnl / 10_000:+.1f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#444444",
        )

    axis.plot(
        range(len(values)),
        values,
        color=BLUE,
        linewidth=2.1,
        marker="o",
        markersize=5.5,
        zorder=3,
    )
    for index, value in enumerate(values):
        offset = 3.0 if index in (0, 3) else -4.5
        va = "bottom" if offset > 0 else "top"
        axis.text(
            index,
            value + offset,
            f"{value:.1f}",
            ha="center",
            va=va,
            fontsize=9,
            color="#333333",
        )

    axis.set_title("Balance at fixed 184-day segment boundaries")
    axis.set_xlabel("Boundary")
    axis.set_ylabel("Closed-trade balance (10k JPY)")
    axis.set_xticks(range(6), ["Start", "S1", "S2", "S3", "S4", "S5"])
    axis.set_ylim(0, 110)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path = output_dir / "segment_boundary_balance.png"
    save_figure(figure, output_path)
    return output_path


def plot_exit_reasons(manifest: dict[str, Any], output_dir: Path) -> Path:
    official = manifest["detectors"]["official"]
    counts = official["exit_reasons"]
    labels = ["TP", "Gap SL", "SL"]
    values = [int(counts["tp"]), int(counts["gap_sl"]), int(counts["sl"])]
    colors = [GREEN, GRAY, RED]

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.barh(labels, values, color=colors, height=0.58)
    axis.set_title(f"Exit reasons ({sum(values)} closed trades)")
    axis.set_xlabel("Trades")
    axis.grid(axis="x", alpha=0.25)
    axis.bar_label(bars, padding=4)
    axis.set_xlim(0, max(values) * 1.12)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path = output_dir / "exit_reasons.png"
    save_figure(figure, output_path)
    return output_path


def write_figure_hashes(
    reference_dir: Path, output_paths: list[Path]
) -> Path:
    payload = {
        "schema_version": 1,
        "source_manifest_sha256": file_sha256(reference_dir / "manifest.json"),
        "generator_sha256": file_sha256(Path(__file__)),
        "figure_sha256": {
            path.name: file_sha256(path) for path in sorted(output_paths)
        },
    }
    output_path = reference_dir / "figure_hashes.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the three S2-4 article figures from frozen row-free evidence."
    )
    chapter_dir = Path(__file__).resolve().parent
    repo_root = chapter_dir.parents[2]
    default_reference = repo_root / "results" / "reference" / chapter_dir.name
    parser.add_argument("--reference-dir", type=Path, default=default_reference)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_reference / "figures",
    )
    parser.add_argument(
        "--article-output-dir",
        type=Path,
        default=None,
        help="Optional second directory receiving byte-identical article PNGs.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.reference_dir)
    output_paths = [
        plot_bullish_ob_schema(args.output_dir),
        plot_segment_boundary_balance(manifest, args.output_dir),
        plot_exit_reasons(manifest, args.output_dir),
    ]
    hashes_path = write_figure_hashes(args.reference_dir, output_paths)

    if args.article_output_dir is not None:
        args.article_output_dir.mkdir(parents=True, exist_ok=True)
        for path in output_paths:
            shutil.copy2(path, args.article_output_dir / path.name)

    print(f"wrote three S2-4 figures to {args.output_dir}")
    print(f"wrote figure hashes to {hashes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
