from __future__ import annotations

import argparse
import datetime as dt
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

from bocchi_the_botter_repro.season2.ict_ob_comparison import (  # noqa: E402
    assert_comparison_reproducible,
    run_ict_ob_comparison_from_db,
)
from bocchi_the_botter_repro.season2.ict_ob_evidence import (  # noqa: E402
    build_evidence_manifest,
    stage8_code_paths,
    verify_evidence_pack,
    write_evidence_pack,
)
from bocchi_the_botter_repro.season2.minute_data import git_code_commit  # noqa: E402


DEFAULT_REFERENCE_DIR = REPO_ROOT / "results" / "reference" / CHAPTER_DIR.name


def _now_utc() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Season 2 #4 stage 8: rebuild the full comparison twice and write "
            "a row-free article-time evidence manifest."
        )
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Read-only upstream SQLite containing the fixed USD_JPY M5 rows.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory for manifest.json and manifest.sha256.",
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Optional article-time timestamp in YYYY-MM-DDTHH:MM:SSZ form.",
    )
    args = parser.parse_args()

    generated_at_utc = args.generated_at_utc or _now_utc()
    manifest_path = args.reference_dir.expanduser().resolve() / "manifest.json"
    if args.db.expanduser().resolve() == manifest_path:
        parser.error("--db must not be the output manifest")

    first = run_ict_ob_comparison_from_db(args.db)
    result = run_ict_ob_comparison_from_db(args.db)
    assert_comparison_reproducible(first, result)

    canonical_reference = "results/reference/ch04_ict_order_blocks"
    rebuild_command = (
        "python chapters/season2/ch04_ict_order_blocks/build_manifest.py "
        '--db "$SOURCE_DB" '
        f"--reference-dir {canonical_reference} "
        f"--generated-at-utc {generated_at_utc}"
    )
    verify_command = (
        "python chapters/season2/ch04_ict_order_blocks/verify.py "
        f"--reference-dir {canonical_reference}"
    )
    code_paths = stage8_code_paths(REPO_ROOT, CHAPTER_DIR)
    manifest = build_evidence_manifest(
        result,
        db_basename=args.db.expanduser().resolve().name,
        code_commit=git_code_commit(REPO_ROOT),
        generated_at_utc=generated_at_utc,
        command=rebuild_command,
        verify_command=verify_command,
        code_paths=code_paths,
    )
    digest = write_evidence_pack(args.reference_dir, manifest)
    verified = verify_evidence_pack(
        args.reference_dir,
        code_paths=code_paths,
    )
    print(
        "S2-4 stage 8 evidence PASS: "
        f"runs={verified['reproducibility_runs']} "
        f"stage7_sha256={verified['hashes']['stage7_result_sha256']} "
        f"manifest_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
