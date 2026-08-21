"""Import-path bootstrap for scripts run directly with `python scripts/...`.

Import this first in every script:

    import _bootstrap  # noqa: F401  (adds repo root + src/ to sys.path)

The alternative would be `pip install -e .`, which works but adds an install
step and hides the mechanism. Four explicit lines are easier to reason about.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
