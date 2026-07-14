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

from bocchi_the_botter_repro.season2.minute_data import (  # noqa: E402
    SOURCE,
    aggregate_m5_to_m15,
    audit_m5_frame,
    build_evidence_manifest,
    git_code_commit,
    load_m5_candles,
    parse_m5_boundary,
    write_evidence_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the row-free Season 2 #2 evidence manifest."
    )
    parser.add_argument("--db", required=True, type=Path, help="Read-only SQLite path")
    parser.add_argument("--instrument", required=True, help="OANDA name, e.g. USD_JPY")
    parser.add_argument("--start", required=True, help="Inclusive RFC3339 UTC boundary")
    parser.add_argument("--end", required=True, help="Exclusive RFC3339 UTC boundary")
    parser.add_argument("--manifest", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    if args.db.expanduser().resolve() == args.manifest.expanduser().resolve():
        parser.error("--db and --manifest must be different files")
    start = parse_m5_boundary(args.start)
    end = parse_m5_boundary(args.end)
    db_before = args.db.expanduser().resolve().stat()
    frame = load_m5_candles(
        args.db,
        source=SOURCE,
        instrument=args.instrument,
        start_inclusive=start,
        end_exclusive=end,
    )
    m5_audit = audit_m5_frame(
        frame,
        source=SOURCE,
        instrument=args.instrument,
        start_inclusive=start,
        end_exclusive=end,
    )
    m15 = aggregate_m5_to_m15(
        frame,
        start_inclusive=start,
        end_exclusive=end,
    )
    output = args.manifest
    try:
        output_label = output.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        output_label = output.name
    command = (
        "python chapters/season2/ch02_minute_data_db/build_manifest.py "
        f'--db "$SOURCE_DB" --instrument {args.instrument} '
        f"--start {args.start} --end {args.end} --manifest {output_label}"
    )
    manifest = build_evidence_manifest(
        db_path=args.db,
        environment="practice",
        instrument=args.instrument,
        start_inclusive=start,
        end_exclusive=end,
        m5_audit=m5_audit,
        m15=m15,
        code_commit=git_code_commit(REPO_ROOT),
        command=command,
    )
    db_after = args.db.expanduser().resolve().stat()
    if (db_before.st_size, db_before.st_mtime_ns) != (
        db_after.st_size,
        db_after.st_mtime_ns,
    ):
        parser.error("SQLite changed during the evidence run; retry a stable snapshot")
    write_evidence_manifest(output, manifest)
    print(
        f"wrote evidence manifest: {output_label} "
        f"m5={m5_audit.row_count} m15={len(m15.candles)} "
        f"incomplete={len(m15.incomplete_buckets)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
