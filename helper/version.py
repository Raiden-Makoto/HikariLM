"""A stamp of the helper's code, so a copy that is older than the files can be noticed."""

from __future__ import annotations

from pathlib import Path


def code_stamp() -> int:
    """The newest change time of the helper's program files (not its tests)."""
    folder = Path(__file__).resolve().parent
    times = [path.stat().st_mtime_ns for path in folder.glob("*.py") if not path.name.startswith("test_")]
    return max(times, default=0)
