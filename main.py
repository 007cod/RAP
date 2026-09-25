#!/usr/bin/env python3
"""Standalone entry point for the copied Tool LLM runtime."""

from __future__ import annotations

import os
from pathlib import Path
import sys


RAP_ROOT = Path(__file__).resolve().parent
os.chdir(RAP_ROOT)
if str(RAP_ROOT) not in sys.path:
    sys.path.insert(0, str(RAP_ROOT))

from src.run_tool_llm import main  # noqa: E402


if __name__ == "__main__":
    main()
