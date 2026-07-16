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
    verify_private_against_reference,
    verify_row_free_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently recompute S2-3 metrics from the private trade log."
    )
    parser.add_argument(
        "--trades",
        type=Path,
        default=None,
        help="Optional private trade log for full price/equity verification.",
    )
    parser.add_argument("--reference-dir", required=True, type=Path)
    args = parser.parse_args()

    payload = verify_row_free_reference(args.reference_dir)
    if args.trades is None:
        summary = payload["summary"]
        label = "row-free"
    else:
        summary = verify_private_against_reference(args.trades, args.reference_dir)
        label = "full"
    print(
        f"S2-3 {label} independent verification PASS: "
        f"trades={summary['trade_count']} "
        f"return={float(summary['return_pct']):.6f}% "
        f"positive_segments={summary['positive_segments']}/5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
