#!/usr/bin/env python3
"""
report/latex/scripts/generate_figures.py
=======================================
Legacy alias redirecting to the unified, reproducible figure generator:
report/latex/scripts/generate_paper_figures.py
"""

import os
import sys
import runpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "generate_paper_figures.py")

if __name__ == "__main__":
    sys.argv[0] = MAIN_SCRIPT
    runpy.run_path(MAIN_SCRIPT, run_name="__main__")
