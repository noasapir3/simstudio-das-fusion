#!/usr/bin/env python3
"""
extract_anomaly_per_sim.py
──────────────────────────────────────────────────────────────────────────────
For every anomaly simulation under  anomaly_model/simulations/region_*/
that has already been run in the GUI (i.e. has a *_tracking_audit export
folder), extract features and save a CSV into a subfolder next to the sim:

    anomaly_model/simulations/
        region_120/
            region_120_anomaly_collision.sim.json
            region_120_anomaly_collision_export_<ts>_tracking_audit/
            anomaly_collision/                  ← created by this script
                features.csv

Files named  *_clean.sim.json  and  *_sensors.sim.json  are always skipped.

Run from the repo root:
    python3 extract_anomaly_per_sim.py

Optional flags:
    --dry-run        Show what would be extracted without writing anything
    --region 120     Only process region_120 (can repeat: --region 120 140)
    --overwrite      Re-extract even if features.csv already exists
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent   # scripts/ -> repo root
SIMS_ROOT  = REPO_ROOT / "anomaly_model" / "simulations"
sys.path.insert(0, str(REPO_ROOT))   # makes `anomaly_model` importable


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_export_folder(region_dir: Path, true_stem: str) -> Path | None:
    """
    Look for the newest *_tracking_audit folder that belongs to this sim file.
    Matches both  <true_stem>_export_*_tracking_audit
    and           <true_stem>.sim_export_*_tracking_audit  (legacy naming).
    """
    candidates = sorted(
        list(region_dir.glob(f"{true_stem}_export_*_tracking_audit")) +
        list(region_dir.glob(f"{true_stem}.sim_export_*_tracking_audit")),
        reverse=True,   # newest first
    )
    for c in candidates:
        if c.is_dir() and (c / "tracking_audit.xlsx").exists():
            return c
    return None


def discover(region_filter: list[str]) -> list[dict]:
    """
    Walk SIMS_ROOT and return one record per processable sim file:
        {region_dir, sim_json, true_stem, test_name, export_folder, out_dir}
    """
    records = []
    if not SIMS_ROOT.exists():
        print(f"ERROR: simulations folder not found: {SIMS_ROOT}")
        sys.exit(1)

    for region_dir in sorted(SIMS_ROOT.iterdir()):
        if not region_dir.is_dir() or not region_dir.name.startswith("region_"):
            continue
        # e.g. "120" from "region_120"
        region_num = region_dir.name.replace("region_", "")
        if region_filter and region_num not in region_filter:
            continue

        for sim_json in sorted(region_dir.glob("*.sim.json")):
            true_stem = sim_json.name.replace(".sim.json", "")

            # Skip base map files
            if true_stem.endswith("_clean") or true_stem.endswith("_sensors"):
                continue

            # Derive a short test name by removing the "region_XXX_" prefix
            # e.g. "region_120_anomaly_collision" → "anomaly_collision"
            prefix = f"region_{region_num}_"
            test_name = true_stem.replace(prefix, "", 1) if true_stem.startswith(prefix) else true_stem

            export_folder = _find_export_folder(region_dir, true_stem)
            out_dir = region_dir / test_name

            records.append({
                "region_dir":    region_dir,
                "sim_json":      sim_json,
                "true_stem":     true_stem,
                "test_name":     test_name,
                "export_folder": export_folder,   # None if sim not yet run
                "out_dir":       out_dir,
                "scenario_id":   f"{region_dir.name}/{test_name}",
            })

    return records


def extract_one(rec: dict) -> "pd.DataFrame":
    """Run ScenarioLoader for a single sim and return the feature DataFrame."""
    from anomaly_model.feature_extractor import ScenarioLoader
    loader = ScenarioLoader(
        folder=rec["export_folder"],
        scenario_id=rec["scenario_id"],
        scenario_json=rec["sim_json"],
    )
    df = loader.extract()
    df.insert(3, "label", "anomaly")   # ground-truth label
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract features per anomaly simulation → per-sim CSV"
    )
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print what would happen without writing anything")
    ap.add_argument("--region",    action="append", default=[],
                    metavar="NNN",
                    help="Limit to this region number (repeat for multiple, e.g. --region 120 --region 140)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract even if features.csv already exists in the output folder")
    args = ap.parse_args()

    region_filter = [r.zfill(0) for r in args.region]  # keep as-is (user passes "120" not "region_120")

    print("=" * 60)
    print("  Per-simulation anomaly feature extraction")
    print("=" * 60)
    print(f"  Simulations root : {SIMS_ROOT}")
    if region_filter:
        print(f"  Region filter    : {', '.join(region_filter)}")
    print()

    records = discover(region_filter)
    if not records:
        print("No anomaly sim files found.")
        return

    # ── Summary table ────────────────────────────────────────────────────────
    ready    = [r for r in records if r["export_folder"] is not None]
    not_run  = [r for r in records if r["export_folder"] is None]
    done     = [r for r in ready  if (r["out_dir"] / "features.csv").exists() and not args.overwrite]
    to_run   = [r for r in ready  if r not in done]

    print(f"  Found {len(records)} anomaly sim(s) across regions:")
    for r in records:
        has_export = r["export_folder"] is not None
        has_csv    = (r["out_dir"] / "features.csv").exists()
        status = (
            "✓ already extracted" if has_csv and not args.overwrite else
            "→ will extract"      if has_export else
            "⚠ NOT RUN in GUI yet (no export folder)"
        )
        print(f"    {r['region_dir'].name}/{r['test_name']:<35}  {status}")

    print()
    if not_run:
        print(f"  ⚠  {len(not_run)} sim(s) have never been run in the GUI — skipping.")
        print(f"     Open them in SimStudio → Run → Export, then re-run this script.\n")

    if args.dry_run:
        print(f"  [dry-run] Would extract {len(to_run)} sim(s). No files written.")
        return

    if not to_run:
        print("  Nothing new to extract — all ready sims already have features.csv.")
        print("  Use --overwrite to force re-extraction.")
        return

    # ── Extract ───────────────────────────────────────────────────────────────
    import pandas as pd

    ok = fail = 0
    for rec in to_run:
        label = f"{rec['region_dir'].name}/{rec['test_name']}"
        print(f"  Extracting  {label:<45} … ", end="", flush=True)
        try:
            df = extract_one(rec)
            rec["out_dir"].mkdir(parents=True, exist_ok=True)
            out_csv = rec["out_dir"] / "features.csv"
            df.to_csv(out_csv, index=False)
            print(f"OK  ({len(df)} track(s))  →  {out_csv.relative_to(REPO_ROOT)}")
            ok += 1
        except Exception as exc:
            print(f"FAIL  →  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            fail += 1

    print()
    print("=" * 60)
    print(f"  Done: {ok} extracted, {fail} failed, {len(done)} already existed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
