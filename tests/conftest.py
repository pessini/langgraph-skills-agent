"""Pytest configuration for the skills-agent overlay.

Adds ``graphs/`` to ``sys.path`` so tests can import ``core`` and
``skills_agent`` without an editable install.
"""

import sys
from pathlib import Path

GRAPHS_DIR = Path(__file__).resolve().parent.parent / "graphs"
if str(GRAPHS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPHS_DIR))
