from __future__ import annotations

import platform
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

from bocchi_the_botter_repro.common.data import SUPPORTED_PAIRS  # noqa: E402
from bocchi_the_botter_repro.common.reproduction import DEFAULT_END_DATE_STR  # noqa: E402


def main() -> int:
    print("# ch00_prologue")
    print(f"Python: {platform.python_version()}")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Default article end-date: {DEFAULT_END_DATE_STR}")
    print(f"Supported pairs: {', '.join(SUPPORTED_PAIRS)}")
    print("No market data is fetched by this chapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
