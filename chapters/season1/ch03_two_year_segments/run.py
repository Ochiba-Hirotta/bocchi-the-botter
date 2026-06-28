from __future__ import annotations

import argparse
import sys
from datetime import timedelta
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
    get_spread,
    parse_iso_utc,
    parse_pairs,
    run_bb_mr_segments,
)

CHAPTER = "ch03_two_year_segments"
DEFAULT_PAIRS = "USDJPY,GBPJPY"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter #3: BB-MR first-half / second-half / full-span runs."
    )
    parser.add_argument("--pairs", type=parse_pairs, default=parse_pairs(DEFAULT_PAIRS))
    parser.add_argument(
        "--end-date", type=parse_iso_utc, default=parse_iso_utc(DEFAULT_END_DATE_STR)
    )
    parser.add_argument("--days", type=int, default=DEFAULT_LONG_DAYS)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    end = args.end_date
    start = end - timedelta(days=args.days)
    output_dir = chapter_output_dir(REPO_ROOT, CHAPTER, args.output_dir)
    for pair in args.pairs:
        run_bb_mr_segments(
            pair=pair,
            start=start,
            end=end,
            spread=get_spread(pair),
            cache_root=args.cache_root,
            output_dir=output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
