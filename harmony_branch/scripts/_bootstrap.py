"""Make repository and harmony package imports stable for direct CLI execution."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARMONY_SRC = PROJECT_ROOT / "harmony_branch" / "src"

for import_root in (str(PROJECT_ROOT), str(HARMONY_SRC)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
