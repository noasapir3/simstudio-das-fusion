#!/usr/bin/env python3
"""
patch_speed_limits.py
─────────────────────────────────────────────────────────────────────────────
Fixes the sc_speed_limit_mps column in features_normal.csv and
features_anomaly.csv by reading the correct speed limit from each
tile's region_info.json instead of using the sim.json minimum
(which always resolves to 30 km/h because every tile has at least
one 30 km/h segment).

Strategy: use the DOMINANT speed limit — the one covering the most
road segments in the tile's region_info.json.

Run from the repo root:
    python3 patch_speed_limits.py

Optional flags:
    --dry-run       Print what would change without writing anything
    --no-excel      Skip rebuilding the .xlsx files (fast mode)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent   # scripts/ -> repo root
MAPS_ROOT  = REPO_ROOT / "maps" / "tiles_north_tlv_south"
OUT_DIR    = REPO_ROOT / "anomaly_model" / "outputs"
NORMAL_CSV = OUT_DIR / "features_normal.csv"
ANOMALY_CSV= OUT_DIR / "features_anomaly.csv"


# ── Speed-limit lookup ────────────────────────────────────────────────────────

def _load_region_info_cache() -> dict[str, dict]:
    """Read every tile's region_info.json once; return dict keyed by zero-padded tile string."""
    cache: dict[str, dict] = {}
    for tile_dir in MAPS_ROOT.iterdir():
        ri = tile_dir / "region_info.json"
        if ri.exists():
            try:
                cache[tile_dir.name.lstrip("0") or "0"] = json.loads(ri.read_text())
                cache[tile_dir.name] = cache[tile_dir.name.lstrip("0") or "0"]  # both forms
            except Exception:
                pass
    return cache


def dominant_speed_limit_mps(region_info: dict) -> float:
    """
    Return the dominant speed limit in m/s for this region.

    'Dominant' = the km/h value covering the most road segments.
    Falls back to avg_speed_mps if speed_limit_counts is absent.
    """
    counts: dict = region_info.get("speed_limit_counts", {})
    if counts:
        # Most common speed limit (highest count wins)
        dominant_kmh = max(counts, key=lambda k: counts[k])
        return round(int(dominant_kmh) / 3.6, 6)
    avg = region_info.get("avg_speed_mps")
    if avg is not None:
        return float(avg)
    return float("nan")


def tile_from_scenario_id(scenario_id: str) -> str:
    """
    Extract the tile number string from a scenario_id.

    Normal  : '001/normal_fast_legal'   → '001'
    Anomaly : 'region_003/anomaly_...'  → '003'
    """
    part = scenario_id.split("/")[0]          # '001' or 'region_003'
    part = part.replace("region_", "")        # '001' or '003'
    return part.zfill(3)                       # always 3-digit zero-padded


# ── Patch one DataFrame ───────────────────────────────────────────────────────

def patch_df(df: pd.DataFrame, cache: dict, label: str, dry_run: bool) -> pd.DataFrame:
    if "sc_speed_limit_mps" not in df.columns:
        print(f"  [{label}] Column 'sc_speed_limit_mps' not found — skipping.")
        return df

    import numpy as np

    # Build the corrected speed-limit series row by row
    new_limit = df["sc_speed_limit_mps"].copy().astype(float)
    not_found = Counter()

    for idx, scenario_id in df["scenario_id"].items():
        tile = tile_from_scenario_id(str(scenario_id))
        ri   = cache.get(tile)
        if ri is None:
            not_found[tile] += 1
            continue
        new_limit.at[idx] = dominant_speed_limit_mps(ri)

    limit_changed = (new_limit - df["sc_speed_limit_mps"].astype(float)).abs() > 1e-6

    # Summary
    old_dist = df["sc_speed_limit_mps"].astype(float).apply(
        lambda v: round(v * 3.6)).value_counts().to_dict()
    new_dist  = new_limit.apply(
        lambda v: round(float(v) * 3.6)).value_counts().to_dict()

    print(f"  [{label}] sc_speed_limit_mps  before → {old_dist} km/h")
    print(f"  [{label}] sc_speed_limit_mps  after  → {new_dist} km/h")
    print(f"  [{label}] Limit rows updated: {limit_changed.sum()} / {len(df)}")
    if not_found:
        print(f"  [{label}] Tiles not found in region_info: {dict(not_found)}")

    # ── Recompute derived speed-excess features ───────────────────────────────
    # kin_speed_excess_max_mps  = max(0, peak_speed  − corrected_limit)  — exact
    # kin_speed_excess_mean_mps = max(0, mean_speed  − corrected_limit)  — approximate
    #   (exact value needs per-timestep speed trace; mean speed is the best proxy
    #    available in the summary CSV)
    # kin_speed_over_limit_frac — cannot recompute without raw trajectory data;
    #   left unchanged here. Run a full re-extraction to get the exact value.

    derived_stats: dict = {}
    for col_out, col_speed in [
        ("kin_speed_excess_max_mps",  "kin_speed_max_mps"),
        ("kin_speed_excess_mean_mps", "kin_speed_mean_mps"),
    ]:
        if col_out not in df.columns or col_speed not in df.columns:
            continue
        old_vals = df[col_out].astype(float)
        new_vals = (df[col_speed].astype(float) - new_limit).clip(lower=0.0)
        changed  = (new_vals - old_vals).abs() > 1e-6
        derived_stats[col_out] = int(changed.sum())
        if not dry_run:
            df[col_out] = new_vals

    for col, cnt in derived_stats.items():
        print(f"  [{label}] {col}: {cnt} rows recomputed")

    # kin_speed_over_limit_frac note
    if "kin_speed_over_limit_frac" in df.columns:
        print(f"  [{label}] kin_speed_over_limit_frac: ⚠ needs full re-extraction "
              f"(raw trajectory required) — left unchanged")

    if not dry_run:
        df["sc_speed_limit_mps"] = new_limit

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Patch sc_speed_limit_mps in feature CSVs")
    ap.add_argument("--dry-run",   action="store_true", help="Print changes without writing")
    ap.add_argument("--no-excel",  action="store_true", help="Skip Excel rebuild")
    args = ap.parse_args()

    print("=" * 60)
    print("  Patching sc_speed_limit_mps")
    print("=" * 60)

    print("\nLoading region_info cache ...")
    cache = _load_region_info_cache()
    print(f"  Loaded {len(cache) // 2} tiles")   # cache has both padded+unpadded keys

    # ── Normal CSV ────────────────────────────────────────────────────────────
    print(f"\nNormal features: {NORMAL_CSV}")
    if not NORMAL_CSV.exists():
        print("  ERROR: file not found — run extraction first")
        sys.exit(1)
    df_n = pd.read_csv(NORMAL_CSV, low_memory=False)
    df_n = patch_df(df_n, cache, "normal", args.dry_run)
    if not args.dry_run:
        df_n.to_csv(NORMAL_CSV, index=False)
        print(f"  Saved {NORMAL_CSV}")

    # ── Anomaly CSV ───────────────────────────────────────────────────────────
    print(f"\nAnomaly features: {ANOMALY_CSV}")
    if not ANOMALY_CSV.exists():
        print("  ERROR: file not found — run extraction first")
        sys.exit(1)
    df_a = pd.read_csv(ANOMALY_CSV, low_memory=False)
    df_a = patch_df(df_a, cache, "anomaly", args.dry_run)
    if not args.dry_run:
        df_a.to_csv(ANOMALY_CSV, index=False)
        print(f"  Saved {ANOMALY_CSV}")

    # ── Excel rebuild ─────────────────────────────────────────────────────────
    if not args.dry_run and not args.no_excel:
        print("\nRebuilding Excel files ...")
        try:
            from anomaly_model.make_features_excel import make_excel
            print("  Building features_anomaly.xlsx ...")
            make_excel(ANOMALY_CSV, OUT_DIR / "features_anomaly.xlsx")
            print("  → done")
        except Exception as e:
            print(f"  WARNING: Anomaly Excel failed: {e}")

        print()
        print("  features_normal.xlsx is large (~2 min) — run separately if needed:")
        print("    python3 -c \"from pathlib import Path; from anomaly_model.make_features_excel import make_excel; make_excel(Path('anomaly_model/outputs/features_normal.csv'), Path('anomaly_model/outputs/features_normal.xlsx'))\"")
    elif args.dry_run:
        print("\n[dry-run] No files written.")

    print("\n" + "=" * 60)
    print("  Done.")
    if not args.dry_run:
        print()
        print("  Next: retrain the model on the updated features:")
        print("    bash retrain_model.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
