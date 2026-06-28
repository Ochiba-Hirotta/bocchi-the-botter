from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not find repository root")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bocchi_the_botter_repro.common.reproduction import (  # noqa: E402
    DEFAULT_END_DATE_STR,
    DEFAULT_LONG_DAYS,
    chapter_output_dir,
    parse_iso_utc,
    recompute_physical_metrics_from_market,
    run_physical_metrics_from_trades,
)

CHAPTER = "ch07_physical_metrics"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter #7: physical metrics from 8 grids x 5 folds."
    )
    parser.add_argument(
        "--trades-dir",
        type=Path,
        default=REPO_ROOT / "results" / "reference" / CHAPTER,
        help="Directory containing trades_7_*.csv when not recomputing.",
    )
    parser.add_argument(
        "--recompute-trades",
        action="store_true",
        help="Fetch market data and recompute the 40 trade CSVs before aggregating.",
    )
    parser.add_argument(
        "--end-date", type=parse_iso_utc, default=parse_iso_utc(DEFAULT_END_DATE_STR)
    )
    parser.add_argument("--days", type=int, default=DEFAULT_LONG_DAYS)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = chapter_output_dir(REPO_ROOT, CHAPTER, args.output_dir)
    if args.recompute_trades:
        recompute_physical_metrics_from_market(
            cache_root=args.cache_root,
            end_date=args.end_date,
            days=args.days,
            output_dir=output_dir,
        )
    else:
        run_physical_metrics_from_trades(
            trades_dir=args.trades_dir,
            output_dir=output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
