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
    write_strategy_compare_outputs,
)

CHAPTER = "ch06_donchian_compare"
DEFAULT_PAIRS = "USDJPY,GBPJPY,EURJPY,AUDJPY"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapter #6: BB-MR vs Donchian WFA comparison."
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
        "--strategy",
        choices=["both", "bb_mr", "donchian"],
        default="both",
        help="Run both strategies by default.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Only write WFA CSVs; skip the cross-strategy comparison CSVs.",
    )
    args = parser.parse_args()

    output_dir = chapter_output_dir(REPO_ROOT, CHAPTER, args.output_dir)
    if args.strategy in {"both", "bb_mr"}:
        run_wfa_pairs(
            strategy="bb_mr",
            pairs=args.pairs,
            mode=args.mode,
            days=args.days,
            end_date=args.end_date,
            cache_root=args.cache_root,
            output_dir=output_dir,
        )
    if args.strategy in {"both", "donchian"}:
        run_wfa_pairs(
            strategy="donchian",
            pairs=args.pairs,
            mode=args.mode,
            days=args.days,
            end_date=args.end_date,
            cache_root=args.cache_root,
            output_dir=output_dir,
        )
    has_all_pairs = set(args.pairs) == set(PAIRS) and len(args.pairs) == len(PAIRS)
    if (
        args.strategy == "both"
        and args.mode == "full"
        and not args.skip_compare
        and has_all_pairs
    ):
        write_strategy_compare_outputs(
            wfa_dir=output_dir,
            output_dir=output_dir,
            end_date=args.end_date,
        )
    elif args.strategy == "both" and args.mode == "full" and not args.skip_compare:
        print("[skip] strategy comparison requires all four default pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
