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

from bocchi_the_botter_repro.season2.orb import run_live, run_reference  # noqa: E402


CHAPTER = "ch01_orb_1h_translation"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Season 2 #1: USDJPY 1h ORB translation reproduction."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch current Yahoo Finance data and recompute the fixed article window.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REPO_ROOT / "results" / "reference" / CHAPTER,
        help="Directory containing the three article-time reference CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Live mode defaults to outputs/ch01_orb_1h_translation/.",
    )
    args = parser.parse_args()

    if args.live:
        output_dir = args.output_dir or REPO_ROOT / "outputs" / CHAPTER
        run_live(output_dir=output_dir)
    else:
        run_reference(
            reference_dir=args.reference_dir,
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
