"""CLI entry point for ``rag-asr-retrieve``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main():
    script = Path(__file__).resolve().parents[2] / "scripts" / "retrieve.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
