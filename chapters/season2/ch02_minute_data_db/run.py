from __future__ import annotations

import argparse
import os
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
    TOKEN_ENV_VAR,
    aggregate_m5_to_m15,
    audit_m5_frame,
    build_evidence_manifest,
    fetch_to_sqlite,
    format_utc_timestamp,
    git_code_commit,
    load_m5_candles,
    parse_m5_boundary,
    write_evidence_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Season 2 #2: fetch complete OANDA M5 BA candles to SQLite."
    )
    parser.add_argument("--instrument", required=True, help="OANDA name, e.g. USD_JPY")
    parser.add_argument("--start", required=True, help="Inclusive RFC3339 UTC boundary")
    parser.add_argument("--end", required=True, help="Exclusive RFC3339 UTC boundary")
    parser.add_argument("--db", required=True, type=Path, help="Output SQLite path")
    parser.add_argument("--manifest", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    if args.db.expanduser().resolve() == args.manifest.expanduser().resolve():
        parser.error("--db and --manifest must be different files")

    token = os.environ.get(TOKEN_ENV_VAR, "")
    if not token:
        parser.error(f"{TOKEN_ENV_VAR} environment variable is required")
    try:
        start = parse_m5_boundary(args.start)
        end = parse_m5_boundary(args.end)
        summary = fetch_to_sqlite(
            token=token,
            instrument=args.instrument,
            start_inclusive=start,
            end_exclusive=end,
            db_path=args.db,
        )
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
        db_label = args.db.name
        manifest_label = args.manifest.name
        command = (
            "python chapters/season2/ch02_minute_data_db/run.py "
            f"--instrument {args.instrument} --start {args.start} --end {args.end} "
            f"--db outputs/ch02_minute_data_db/{db_label} "
            f"--manifest outputs/ch02_minute_data_db/{manifest_label}"
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
            ingestion=summary,
        )
        write_evidence_manifest(args.manifest, manifest)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        "stored M5 BA candles: "
        f"instrument={summary.instrument} "
        f"range=[{format_utc_timestamp(summary.start_inclusive)}, "
        f"{format_utc_timestamp(summary.end_exclusive)}) "
        f"chunks={summary.chunk_count} requests={summary.request_count} "
        f"upserted={summary.rows_upserted} "
        f"incomplete_skipped={summary.incomplete_skipped}"
    )
    print(f"wrote evidence manifest: {args.manifest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
