"""
Pytest bootstrap.

Puts the repository root on ``sys.path`` so ``analyst`` imports, and the
``text2sql`` subproject on it too so its modules keep working with their
``from src... import`` / ``from config... import`` style imports.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for path in (ROOT, ROOT / "text2sql"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
