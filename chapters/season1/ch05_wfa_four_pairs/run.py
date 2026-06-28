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
    PAIRS,
    chapter_output_dir,
    parse_iso_utc,
    parse_pairs,
    run_wfa_pairs,
    write_bb_mr_centroid_summary,
)

CHAPTER = "ch05_wfa_four_pairs"
DEFAULT_PAIRS = "USDJPY,GBPJPY,EURJPY,AUDJPY"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter #5: four-pair BB-MR WFA and centroid summary."
    )
    parser.add_argument("--pairs", type=parse_pairs, default=parse_pairs(DEFAULT_PAIRS))
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument(
        "--end-date", type=parse_iso_utc, default=parse_iso_utc(DEFAULT_END_DATE_STR)
    )
    parser.add_argument("--days", type=int, default=DEFAULT_LONG_DAYS)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-centroid",
        action="store_true",
        help="Only write WFA CSVs; skip centroid summary.",
    )
    args = parser.parse_args()

    output_dir = chapter_output_dir(REPO_ROOT, CHAPTER, args.output_dir)
    run_wfa_pairs(
        strategy="bb_mr",
        pairs=args.pairs,
        mode=args.mode,
        days=args.days,
        end_date=args.end_date,
        cache_root=args.cache_root,
        output_dir=output_dir,
    )
    has_all_pairs = set(args.pairs) == set(PAIRS) and len(args.pairs) == len(PAIRS)
    if not args.skip_centroid and args.mode == "full" and has_all_pairs:
        write_bb_mr_centroid_summary(
            wfa_dir=output_dir,
            output_path=output_dir / "wfa_bb_mr_4pairs_summary.csv",
            end_date=args.end_date,
        )
    elif not args.skip_centroid and args.mode == "full":
        print("[skip] centroid summary requires all four default pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
