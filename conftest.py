"""Pytest bootstrap.

The project is not pip-installed, so this puts the repo root (for `config`)
and `src/` (for `voc`) on the import path before any test module loads.
Keeping it explicit here avoids an editable-install step and keeps the
`python scripts/...` workflow and the `pytest` workflow identical.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
