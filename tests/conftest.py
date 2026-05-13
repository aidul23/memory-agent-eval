"""Pytest config.

1. Makes ``src`` importable from inside tests.
2. Auto-loads ``.env`` from the project root so live smoke tests pick up
   ``ACONTEXT_API_KEY``, ``ZEP_API_URL``, ``HINDSIGHT_API_URL`` etc.
   without forcing the user to ``export`` them in their shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env without overriding values already exported in the shell.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    # python-dotenv is in requirements.txt; if it's missing the tests still
    # work, they just won't auto-pick up .env values.
    pass
