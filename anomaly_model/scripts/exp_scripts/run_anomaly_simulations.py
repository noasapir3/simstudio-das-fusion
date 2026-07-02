#!/usr/bin/env python3
"""
run_anomaly_simulations.py — headless batch runner for anomaly scenarios.

Runs every anomaly ``*.sim.json`` under ``anomaly_model/simulations/`` through
the SimStudio engine and writes a tracking-audit export folder for each. This
is the *simulate-only* step of the full pipeline, so this wrapper forwards to
``scripts/run_anomaly_pipeline.py`` with ``--skip-extract`` automatically added
(run ``run_extraction.py`` afterwards to extract features):

    python anomaly_model/scripts/run_anomaly_simulations.py
    python anomaly_model/scripts/run_anomaly_simulations.py --regions 120,140
"""
import runpy
import sys
from pathlib import Path

# anomaly_model/scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET = REPO_ROOT / "scripts" / "run_anomaly_pipeline.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Pipeline script not found: {TARGET}")

args = sys.argv[1:]
if "--skip-extract" not in args:          # simulate only — no feature extraction
    args = args + ["--skip-extract"]

sys.argv = [str(TARGET)] + args
runpy.run_path(str(TARGET), run_name="__main__")
