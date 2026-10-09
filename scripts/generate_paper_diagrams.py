#!/usr/bin/env python3
"""
scripts/generate_paper_diagrams.py
==================================
Root-level entry point for reproducing and generating all publication figures
for the Nebium research report directly from Weights & Biases (WandB) run data.

Usage:
------
    python scripts/generate_paper_diagrams.py
    python scripts/generate_paper_diagrams.py --api-key <YOUR_WANDB_API_KEY>
"""

import os
import sys

# Add report scripts directory to path and execute main
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_SCRIPT = os.path.join(ROOT_DIR, "report", "latex", "scripts", "generate_paper_figures.py")

if not os.path.exists(TARGET_SCRIPT):
    print(f"[ERROR] Target script not found: {TARGET_SCRIPT}", file=sys.stderr)
    sys.exit(1)

# Execute target script with original arguments
import runpy
sys.argv[0] = TARGET_SCRIPT
runpy.run_path(TARGET_SCRIPT, run_name="__main__")
