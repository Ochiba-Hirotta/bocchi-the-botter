from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


BLUE = "#4472C4"
GREEN = "#4F9D69"
RED = "#C95D63"
GRAY = "#7A7A7A"


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_segments(reference_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(reference_dir / "segments.csv")
    values = frame["pnl_jpy"] / 10_000.0
    colors = [GREEN if value > 0 else RED for value in values]
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    bars = axis.bar(frame["segment"].astype(str), values, color=colors, width=0.65)
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_title("Realized PnL by fixed 184-day segment")
    axis.set_xlabel("Segment")
    axis.set_ylabel("PnL (10k JPY)")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        offset = 0.25 if value >= 0 else -0.25
        va = "bottom" if value >= 0 else "top"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.1f}",
            ha="center",
            va=va,
            fontsize=9,
        )
    save_figure(figure, output_dir / "segment_results.png")


def plot_exit_reasons(reference_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(reference_dir / "exit_reasons.csv").sort_values(
        "trade_count", ascending=True
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.barh(frame["exit_reason"], frame["trade_count"], color=BLUE)
    axis.set_title("Exit reasons")
    axis.set_xlabel("Trades")
    axis.grid(axis="x", alpha=0.25)
    axis.bar_label(bars, padding=3)
    save_figure(figure, output_dir / "exit_reasons.png")


def plot_atr_filter(reference_dir: Path, output_dir: Path) -> None:
    payload = json.loads(
        (reference_dir / "atr_filter_summary.json").read_text(encoding="utf-8")
    )
    labels = ["Below 1.25", "Passed", "Above 3.0"]
    values = [payload["below_lower"], payload["passed"], payload["above_upper"]]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(labels, values, color=[GRAY, GREEN, RED], width=0.62)
    axis.set_title("Opening-range width / bid ATR(14)")
    axis.set_ylabel("Valid sessions")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, padding=3)
    save_figure(figure, output_dir / "atr_filter_counts.png")


def plot_session_quality(reference_dir: Path, output_dir: Path) -> None:
    payload = json.loads(
        (reference_dir / "session_quality_summary.json").read_text(encoding="utf-8")
    )
    labels = ["Valid (27/27)", "Invalid"]
    values = [payload["valid_sessions"], payload["invalid_sessions"]]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(labels, values, color=[GREEN, GRAY], width=0.58)
    axis.set_title("09:30–16:00 ET session completeness")
    axis.set_ylabel("Calendar dates")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, padding=3)
    axis.text(
        0.99,
        0.02,
        "A valid session contains all 27 M15 starts.",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    save_figure(figure, output_dir / "session_quality.png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate S2-3 figures from row-free reference aggregates."
    )
    chapter_dir = Path(__file__).resolve().parent
    repo_root = chapter_dir.parents[2]
    default_reference = repo_root / "results" / "reference" / chapter_dir.name
    parser.add_argument(
        "--reference-dir", type=Path, default=default_reference
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_reference / "figures"
    )
    args = parser.parse_args()

    plot_segments(args.reference_dir, args.output_dir)
    plot_exit_reasons(args.reference_dir, args.output_dir)
    plot_atr_filter(args.reference_dir, args.output_dir)
    plot_session_quality(args.reference_dir, args.output_dir)
    hashes_path = args.reference_dir / "hashes.json"
    if hashes_path.is_file():
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        hashes["figure_sha256"] = {
            path.name: file_sha256(path)
            for path in sorted(args.output_dir.glob("*.png"))
        }
        hashes_path.write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"wrote four S2-3 figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
