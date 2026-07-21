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

from bocchi_the_botter_repro.season2.ict_ob_experiment import (  # noqa: E402
    assert_reproducible,
    run_ict_ob_from_db,
    write_private_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Season 2 #4 stage 6: run both frozen ICT Order Block translations."
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
        default=REPO_ROOT / "outputs" / "ch04_ict_order_blocks",
        help="Git-ignored directory for private zones, trades, and the run audit.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use 2 to rerun from SQLite and enforce the stage-6 G3 hash gate.",
    )
    args = parser.parse_args()

    first = run_ict_ob_from_db(args.db)
    result = first
    if args.repeat == 2:
        result = run_ict_ob_from_db(args.db)
        assert_reproducible(first, result)
    write_private_outputs(
        result,
        args.output_dir,
        reproducibility_runs=args.repeat,
    )

    print(
        "S2-4 stage 6 G3 PASS: "
        f"runs={args.repeat} result_sha256={result.result_sha256}"
    )
    for label, detector in (
        ("official", result.official),
        ("secondary", result.secondary),
    ):
        summary = detector.summary
        print(
            f"{label}: zones={summary.zone_count} trades={summary.trade_count} "
            f"return={summary.return_pct:.6f}% "
            f"positive_segments={summary.positive_segments}/5 "
            f"criterion={summary.criterion} "
            f"zones_sha256={detector.zone_sha256} "
            f"trades_sha256={detector.trade_sha256}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
