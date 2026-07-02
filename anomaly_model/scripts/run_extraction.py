#!/usr/bin/env python3
"""
run_extraction.py — anomaly feature-extraction entry point.

The extraction implementation lives in the repo-root script
``scripts/extract_anomaly_per_sim.py`` (it walks every
``anomaly_model/simulations/region_*/`` export folder and writes a
``features.csv`` next to each scenario). This thin wrapper forwards all
command-line arguments to it, so either of these works:

    python anomaly_model/scripts/run_extraction.py --dry-run
    python anomaly_model/scripts/run_extraction.py --region 120 --overwrite
"""
import runpy
import sys
from pathlib import Path

# anomaly_model/scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = REPO_ROOT / "scripts" / "extract_anomaly_per_sim.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Extraction script not found: {TARGET}")

sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
