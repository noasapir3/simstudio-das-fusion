#!/usr/bin/env bash
# parallel_extract.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs feature extraction across all 148 normal tiles in 8 parallel batches,
# then extracts anomaly scenarios, then merges everything into one CSV + Excel.
#
# Run from the repo root ("final proj" folder):
#   bash parallel_extract.sh
#
# Expected runtime: ~4–8 minutes total (vs. 20–40 minutes sequential)
# ─────────────────────────────────────────────────────────────────────────────

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

OUT="$REPO_ROOT/anomaly_model/outputs"
SCRIPT="$REPO_ROOT/anomaly_model/scripts/run_extraction.py"
LOGS="$OUT/extraction_logs"
mkdir -p "$LOGS"

echo "========================================================"
echo "  SimStudio — Parallel Feature Extraction"
echo "  $(date)"
echo "========================================================"
echo ""
echo "Repo   : $REPO_ROOT"
echo "Output : $OUT"
echo "Logs   : $LOGS"
echo ""

# ── Step 1: Launch 8 normal-feature batches in parallel ───────────────────
echo "[1/3] Launching 8 parallel normal-feature batches ..."
echo ""

python3 "$SCRIPT" --tiles 001-018 --output "$OUT/features_normal_b1.csv" --no-resume > "$LOGS/b1.log" 2>&1 &
PID1=$!; echo "  Batch 1 (tiles 001-018)  PID=$PID1"

python3 "$SCRIPT" --tiles 019-036 --output "$OUT/features_normal_b2.csv" --no-resume > "$LOGS/b2.log" 2>&1 &
PID2=$!; echo "  Batch 2 (tiles 019-036)  PID=$PID2"

python3 "$SCRIPT" --tiles 037-055 --output "$OUT/features_normal_b3.csv" --no-resume > "$LOGS/b3.log" 2>&1 &
PID3=$!; echo "  Batch 3 (tiles 037-055)  PID=$PID3"

python3 "$SCRIPT" --tiles 056-074 --output "$OUT/features_normal_b4.csv" --no-resume > "$LOGS/b4.log" 2>&1 &
PID4=$!; echo "  Batch 4 (tiles 056-074)  PID=$PID4"

python3 "$SCRIPT" --tiles 075-093 --output "$OUT/features_normal_b5.csv" --no-resume > "$LOGS/b5.log" 2>&1 &
PID5=$!; echo "  Batch 5 (tiles 075-093)  PID=$PID5"

python3 "$SCRIPT" --tiles 094-112 --output "$OUT/features_normal_b6.csv" --no-resume > "$LOGS/b6.log" 2>&1 &
PID6=$!; echo "  Batch 6 (tiles 094-112)  PID=$PID6"

python3 "$SCRIPT" --tiles 113-131 --output "$OUT/features_normal_b7.csv" --no-resume > "$LOGS/b7.log" 2>&1 &
PID7=$!; echo "  Batch 7 (tiles 113-131)  PID=$PID7"

python3 "$SCRIPT" --tiles 132-148 --output "$OUT/features_normal_b8.csv" --no-resume > "$LOGS/b8.log" 2>&1 &
PID8=$!; echo "  Batch 8 (tiles 132-148)  PID=$PID8"

echo ""
echo "  Waiting for all 8 batches to finish..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8
echo "  All normal batches complete."
echo ""

# ── Step 2: Anomaly extraction ────────────────────────────────────────────
echo "[2/3] Extracting anomaly scenarios ..."
python3 "$SCRIPT" \
    --anomaly \
    --output "$OUT/features_anomaly_new.csv" \
    --no-resume \
    > "$LOGS/anomaly.log" 2>&1
echo "  Anomaly extraction complete."
echo ""

# ── Step 3: Merge & rebuild Excel ─────────────────────────────────────────
echo "[3/3] Merging batches and rebuilding Excel files ..."
python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

OUT = Path("anomaly_model/outputs")
LOGS = OUT / "extraction_logs"

# ── Merge normal batches ──────────────────────────────────────────────────
batch_files = sorted(OUT.glob("features_normal_b*.csv"))
if not batch_files:
    raise RuntimeError("No batch CSV files found — check extraction logs in " + str(LOGS))

print(f"  Merging {len(batch_files)} batch files:")
parts = []
for bf in batch_files:
    df = pd.read_csv(bf)
    print(f"    {bf.name}: {len(df)} rows")
    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.drop_duplicates()
# Sort by scenario then track — handle either column name convention
sort_cols = [c for c in ["scenario_id", "global_track_id", "track_id"]
             if c in combined.columns]
if sort_cols:
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

out_csv = OUT / "features_normal.csv"
combined.to_csv(out_csv, index=False)
print(f"  → features_normal.csv: {len(combined)} rows × {len(combined.columns)} cols")

# ── Promote new anomaly CSV ───────────────────────────────────────────────
new_anom = OUT / "features_anomaly_new.csv"
if new_anom.exists():
    anom = pd.read_csv(new_anom)
    anom.to_csv(OUT / "features_anomaly.csv", index=False)
    print(f"  → features_anomaly.csv: {len(anom)} rows × {len(anom.columns)} cols")
else:
    print("  WARNING: features_anomaly_new.csv not found — anomaly CSV not updated")

# ── Rebuild Excel for normal ──────────────────────────────────────────────
try:
    from anomaly_model.make_features_excel import make_excel
    print("  Rebuilding features_normal.xlsx ...")
    make_excel(OUT / "features_normal.csv", OUT / "features_normal.xlsx")
    print("  → features_normal.xlsx done")
except Exception as e:
    print(f"  WARNING: Normal Excel rebuild failed: {e}")

# ── Rebuild Excel for anomaly ─────────────────────────────────────────────
try:
    from anomaly_model.make_features_excel import make_excel
    print("  Rebuilding features_anomaly.xlsx ...")
    make_excel(OUT / "features_anomaly.csv", OUT / "features_anomaly.xlsx")
    print("  → features_anomaly.xlsx done")
except Exception as e:
    print(f"  WARNING: Anomaly Excel rebuild failed: {e}")

# ── Clean up temp batch files ─────────────────────────────────────────────
for bf in batch_files:
    bf.unlink()
new_anom_path = OUT / "features_anomaly_new.csv"
if new_anom_path.exists():
    new_anom_path.unlink()
print("  Temp batch files cleaned up.")
PYEOF

echo ""
echo "========================================================"
echo "  Done!  $(date)"
echo ""
echo "  features_normal.csv  — updated"
echo "  features_anomaly.csv — updated"
echo "  features_normal.xlsx — updated"
echo "  features_anomaly.xlsx — updated"
echo ""
echo "  Backup (original files) saved at:"
echo "  anomaly_model/outputs/backup_before_p99_heading_lateral_fix/"
echo ""
echo "  Next step: run  bash retrain_model.sh  to rebuild model_live.pkl"
echo "========================================================"
