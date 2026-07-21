from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not find repository root")


CHAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(CHAPTER_DIR)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bocchi_the_botter_repro.season2.ict_ob_evidence import (  # noqa: E402
    stage8_code_paths,
    verify_evidence_pack,
    verify_private_trades_against_evidence,
)


DEFAULT_REFERENCE_DIR = REPO_ROOT / "results" / "reference" / CHAPTER_DIR.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the Season 2 #4 row-free manifest hash, code versions, "
            "public boundary, and aggregate invariants."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
    )
    parser.add_argument(
        "--trades",
        type=Path,
        default=None,
        help=(
            "Optional private output directory containing official and secondary "
            "trade CSVs for independent fill, PnL, equity, segment, and metric "
            "recomputation."
        ),
    )
    args = parser.parse_args()

    payload = verify_evidence_pack(
        args.reference_dir,
        code_paths=stage8_code_paths(REPO_ROOT, CHAPTER_DIR),
    )
    if args.trades is None:
        official = payload["detectors"]["official"]["summary"]
        secondary = payload["detectors"]["secondary"]["summary"]
        print(
            "S2-4 row-free evidence PASS: "
            f"official={official['zone_count']} zones/"
            f"{official['trade_count']} trades; "
            f"secondary={secondary['zone_count']} zones/"
            f"{secondary['trade_count']} trades; "
            f"stage7_sha256={payload['hashes']['stage7_result_sha256']}"
        )
        return 0

    verified = verify_private_trades_against_evidence(
        args.trades,
        args.reference_dir,
        evidence_payload=payload,
    )
    official = verified["official"]
    secondary = verified["secondary"]
    print(
        "S2-4 full independent verification PASS: "
        f"official={official['trade_count']} trades/"
        f"{float(official['return_pct']):.6f}% return; "
        f"secondary={secondary['trade_count']} trades/"
        f"{float(secondary['return_pct']):.6f}% return; "
        f"stage7_sha256={payload['hashes']['stage7_result_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
