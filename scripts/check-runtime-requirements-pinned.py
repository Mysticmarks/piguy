#!/usr/bin/env python3
"""Fail CI when runtime requirements are not pinned."""

from __future__ import annotations

from pathlib import Path
import re
import sys

REQUIREMENT_FILES = [
    Path("requirements-core.txt"),
    Path("requirements-speech.txt"),
]

PINNED_RE = re.compile(r"^[A-Za-z0-9_.-]+(\[[^\]]+\])?==[^\s;]+")
VCS_PINNED_RE = re.compile(r"^[A-Za-z0-9_.-]+\s*@\s*git\+.+@[0-9a-fA-F]{7,}")


def _is_valid_requirement(line: str) -> bool:
    if line.startswith(("#", "-r", "--requirement", "-c", "--constraint")):
        return True
    return bool(PINNED_RE.match(line) or VCS_PINNED_RE.match(line))


def main() -> int:
    violations: list[str] = []

    for requirement_file in REQUIREMENT_FILES:
        for idx, raw_line in enumerate(requirement_file.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if not _is_valid_requirement(line):
                violations.append(f"{requirement_file}:{idx}: {raw_line}")

    if violations:
        print("Unpinned runtime requirements detected:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        print(
            "Use exact pins (package==x.y.z) or VCS commit pins (name @ git+...@<commit>).",
            file=sys.stderr,
        )
        return 1

    print("Runtime requirements are pinned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
