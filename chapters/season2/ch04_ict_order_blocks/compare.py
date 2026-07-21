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

from bocchi_the_botter_repro.season2.ict_ob_comparison import (  # noqa: E402
    assert_comparison_reproducible,
    run_ict_ob_comparison_from_db,
    write_comparison_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Season 2 #4 stage 7: compare frozen ICT OB zones, performance, "
            "and smartmoneyconcepts defaults."
        )
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Read-only upstream SQLite containing the fixed USD_JPY M5 rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "outputs" / "ch04_ict_order_blocks" / "stage7"
        ),
        help="Git-ignored directory for private lifecycle rows and comparison audit.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use 2 to enforce complete stage-7 semantic hash reproducibility.",
    )
    args = parser.parse_args()

    first = run_ict_ob_comparison_from_db(args.db)
    result = first
    if args.repeat == 2:
        result = run_ict_ob_comparison_from_db(args.db)
        assert_comparison_reproducible(first, result)
    write_comparison_outputs(
        result,
        args.output_dir,
        reproducibility_runs=args.repeat,
    )

    print(
        "S2-4 stage 7 G4 PASS: "
        f"runs={args.repeat} result_sha256={result.result_sha256}"
    )
    for summary in (
        result.official_secondary,
        result.official_oss,
        result.secondary_oss,
    ):
        left_pct = (
            "n/a"
            if summary.left_overlap_pct is None
            else f"{summary.left_overlap_pct:.6f}%"
        )
        right_pct = (
            "n/a"
            if summary.right_overlap_pct is None
            else f"{summary.right_overlap_pct:.6f}%"
        )
        print(
            f"{summary.left_source} -> {summary.right_source}: "
            f"{summary.left_overlapped}/{summary.left_total} ({left_pct}); "
            f"reverse={summary.right_overlapped}/{summary.right_total} "
            f"({right_pct}); pairs={summary.overlapping_pair_count}"
        )
    oss = result.oss.summary
    print(
        f"OSS {oss.package}=={oss.version}: "
        f"raw_ob={oss.raw_ob_count} comparison_ob={oss.comparison_ob_count} "
        f"raw_liquidity={oss.raw_liquidity_count} "
        f"comparison_liquidity={oss.comparison_liquidity_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
