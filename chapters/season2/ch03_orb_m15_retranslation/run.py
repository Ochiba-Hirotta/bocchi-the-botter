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

from bocchi_the_botter_repro.season2.orb_m15 import (  # noqa: E402
    run_backtest_from_db,
    write_private_audit,
    write_reference_outputs,
)


CHAPTER = "ch03_orb_m15_retranslation"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Season 2 #3: run the frozen USDJPY M15 ORB retranslation."
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Read-only upstream SQLite path containing the frozen M5 projection.",
    )
    parser.add_argument(
        "--private-output-dir",
        required=True,
        type=Path,
        help="Git-ignored directory for the price-bearing trade log and audit.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Optional directory for row-free public reference artifacts.",
    )
    args = parser.parse_args()

    result = run_backtest_from_db(args.db)
    write_private_audit(result, args.private_output_dir)
    if args.reference_dir is not None:
        chapter_dir = Path(__file__).resolve().parent
        write_reference_outputs(
            result,
            args.reference_dir,
            code_paths=[
                SRC_ROOT / "bocchi_the_botter_repro" / "season2" / "orb_m15.py",
                SRC_ROOT / "bocchi_the_botter_repro" / "season2" / "minute_data.py",
                Path(__file__).resolve(),
                chapter_dir / "verify.py",
                chapter_dir / "figures.py",
            ],
        )

    summary = result.summary
    verdict = "PASS" if summary.criterion_passed else "FAIL"
    print(
        f"S2-3 {verdict}: trades={summary.trade_count} "
        f"return={summary.return_pct:.6f}% "
        f"positive_segments={summary.positive_segments}/{5} "
        f"max_dd={summary.max_drawdown_pct:.6f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
