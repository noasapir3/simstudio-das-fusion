#!/usr/bin/env bash
# retrain_model.sh
# ─────────────────────────────────────────────────────────────────────────────
# Full model retrain after feature extractor fixes.
#
# Run from the repo root (the "final proj" folder):
#   bash retrain_model.sh
#
# What this does:
#   1. Re-extracts features from ALL normal simulation tracking_audit.xlsx files
#      using the updated feature_extractor.py (P99 heading rate, P99 lateral
#      speed, deduplication + dt-floor fixes).
#      → updates  anomaly_model/outputs/features_normal.csv
#      → takes ~20–40 minutes depending on your machine
#
#   2. Retrains model_live.pkl (the lightweight GUI model) from the new CSV.
#      → updates  anomaly_model/outputs/model_live.pkl
#      → takes ~5 seconds
#
#   3. Optionally retrains the full Isolation Forest model.pkl.
#      Uncomment the last line to include this step (~2–5 min).
#
# After running, restart the SimStudio GUI — it will auto-load the new model.
# ─────────────────────────────────────────────────────────────────────────────

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo "  SimStudio Anomaly Model — Full Retrain"
echo "  $(date)"
echo "========================================================"
echo ""

# Step 1: Re-extract normal features (forces complete re-extraction)
echo "[1/2] Re-extracting normal features (this takes 20–40 min) ..."
python3 anomaly_model/scripts/run_extraction.py --no-resume
echo ""

# Step 2: Rebuild model_live.pkl from fresh features_normal.csv
echo "[2/2] Retraining model_live.pkl ..."
python3 anomaly_model/scripts/scenario_scorer.py --train-live
echo ""

# Optional Step 3: Rebuild full Isolation Forest model.pkl
# Uncomment if you also use the --score or --eval commands:
# echo "[3/3] Retraining full IsolationForest model.pkl ..."
# python3 anomaly_model/scripts/scenario_scorer.py --train --contamination 0.01

echo "========================================================"
echo "  Done. Restart the GUI to use the updated model."
echo "========================================================"
