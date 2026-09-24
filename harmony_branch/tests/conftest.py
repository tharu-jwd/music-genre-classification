"""Test-path setup matching direct harmony CLI execution."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for import_root in (
    PROJECT_ROOT,
    PROJECT_ROOT / "harmony_branch" / "src",
    PROJECT_ROOT / "harmony_branch" / "scripts",
):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)
