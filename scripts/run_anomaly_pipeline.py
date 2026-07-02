#!/usr/bin/env python3
"""
run_anomaly_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
One-shot pipeline:
  Step 1 — Run every anomaly .sim.json through the SimStudio engine headlessly
            (same as  anomaly_model/scripts/run_anomaly_simulations.py)
  Step 2 — Extract features from each new export folder and write
            features.csv into the matching anomaly sub-folder
            (same as  extract_anomaly_per_sim.py  --overwrite)

Run from the repo root:

    python3 run_anomaly_pipeline.py                  # all regions
    python3 run_anomaly_pipeline.py --regions 120          # one region
    python3 run_anomaly_pipeline.py --regions 120,140      # multiple
    python3 run_anomaly_pipeline.py --skip-sim             # extract only
    python3 run_anomaly_pipeline.py --skip-extract         # simulate only
    python3 run_anomaly_pipeline.py --dry-run              # show plan, no writes
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent   # scripts/ -> repo root
ANOMALY_ROOT  = REPO_ROOT / "anomaly_model"
SIMS_ROOT     = ANOMALY_ROOT / "simulations"
SRC           = REPO_ROOT / "src"

for p in (str(REPO_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

SIM_DT               = 0.02     # 50 Hz — matches the GUI loop
DEFAULT_MAX_DURATION = 120.0    # seconds per scenario (safety cap)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_sim_files(region_filter: Optional[List[str]]) -> List[Path]:
    """Return every anomaly .sim.json (skipping _clean and _sensors base files)."""
    if not SIMS_ROOT.exists():
        raise FileNotFoundError(f"Simulations root not found: {SIMS_ROOT}")
    results = []
    for region_dir in sorted(SIMS_ROOT.iterdir()):
        if not region_dir.is_dir() or not region_dir.name.startswith("region_"):
            continue
        if region_filter and region_dir.name not in region_filter:
            continue
        for sim_json in sorted(region_dir.glob("*.sim.json")):
            true_stem = sim_json.name.replace(".sim.json", "")
            if true_stem.endswith("_clean") or true_stem.endswith("_sensors"):
                continue
            results.append(sim_json)
    return results


def _find_export_folder(region_dir: Path, true_stem: str) -> Optional[Path]:
    """Return the newest tracking_audit export folder for a sim file, or None."""
    candidates = sorted(
        list(region_dir.glob(f"{true_stem}_export_*_tracking_audit")) +
        list(region_dir.glob(f"{true_stem}.sim_export_*_tracking_audit")),
        reverse=True,
    )
    for c in candidates:
        if c.is_dir() and (c / "tracking_audit.xlsx").exists():
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Simulate
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one_scenario(sim_json: Path, max_duration_s: float, verbose: bool) -> dict:
    from simstudio.bus import EventBus
    from simstudio.project_io import load_world
    from simstudio.sim_core import Simulation
    import simstudio.audit as audit_mod

    t_start = time.time()

    world = load_world(sim_json)
    for seg in world.segments.values():
        try:
            seg.lanes = 1
        except Exception:
            pass

    bus = EventBus()
    sim = Simulation(bus, world)
    sim.rebuild_lanes()

    CAPTURE_TOPICS = (
        "world.vehicle_state", "world.vehicle_stuck", "world.collision",
        "world.route_complete", "sensor.gps", "sensor.camera", "sensor.das",
    )
    TRACKER_TOPICS = (
        "world.vehicle_state", "world.vehicle_stuck",
        "sensor.gps", "sensor.camera", "sensor.das",
    )

    all_events: list = []
    for topic in CAPTURE_TOPICS:
        bus.subscribe(topic, all_events.append)

    try:
        from simstudio.tracking import TrackManager
        tm = TrackManager(world, bus=bus)
        def _tm_cb(ev):
            try:
                tm.on_event(ev)
            except Exception:
                pass
        for topic in TRACKER_TOPICS:
            bus.subscribe(topic, _tm_cb)
    except Exception as exc:
        if verbose:
            print(f"    [warn] TrackManager unavailable: {exc}")

    stuck_ids: set = set()
    def _track_stuck(ev):
        vid = (ev.payload or {}).get("vehicle_id")
        if vid:
            stuck_ids.add(str(vid))
    bus.subscribe("world.vehicle_stuck", _track_stuck)

    safety_steps = int(max_duration_s / SIM_DT)
    steps_run    = 0
    stop_reason  = "safety_cap"

    for _ in range(safety_steps):
        sim.step(SIM_DT)
        steps_run += 1
        if not world.vehicles:
            stop_reason = "all_exited"
            break
        if world.vehicles and stuck_ids.issuperset(world.vehicles.keys()):
            stop_reason = "all_stuck"
            break

    if stop_reason == "safety_cap" and verbose:
        print(f"    [warn] safety cap ({max_duration_s:.0f}s) reached; "
              f"{len(world.vehicles)} vehicle(s) still on map")

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = sim_json.parent / f"{sim_json.stem}_export_{ts}_tracking_audit"

    try:
        audit_mod.export_all(
            all_events,
            out_dir,
            world=world,
            progress_cb=(lambda msg: print(f"    {msg}")) if verbose else None,
        )
    except (KeyboardInterrupt, SystemExit):
        import shutil
        try:
            if out_dir.exists():
                shutil.rmtree(out_dir)
        except Exception:
            pass
        raise

    return {
        "events":      len(all_events),
        "steps":       steps_run,
        "sim_time_s":  steps_run * SIM_DT,
        "out_dir":     out_dir,
        "elapsed_s":   time.time() - t_start,
        "stop_reason": stop_reason,
    }


def run_simulations(sim_files: List[Path], max_duration_s: float,
                    verbose: bool, stop_on_error: bool) -> bool:
    """
    Run each sim file headlessly.
    Returns True if all succeeded, False if any failed.
    """
    n  = len(sim_files)
    ok = fail = 0
    fail_list: List[str] = []

    print(f"\n{'─'*60}")
    print(f"  STEP 1 — Running {n} anomaly simulation(s)")
    print(f"{'─'*60}")

    for idx, sim_json in enumerate(sim_files, 1):
        label  = f"{sim_json.parent.name}/{sim_json.stem}"
        prefix = f"  [{idx}/{n}]"
        print(f"{prefix} {label} … ", end="", flush=True)

        try:
            result  = _run_one_scenario(sim_json, max_duration_s, verbose)
            elapsed = result["elapsed_s"]
            ev      = result["events"]
            sim_t   = result["sim_time_s"]
            reason  = result["stop_reason"]
            tag     = "" if reason == "all_exited" else f" [{reason}]"
            print(f"OK  ({ev} events | {sim_t:.1f}s sim in {elapsed:.1f}s real{tag})")
            ok += 1

        except KeyboardInterrupt:
            print("INTERRUPTED")
            return False

        except Exception as exc:
            print(f"FAIL  →  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            fail += 1
            fail_list.append(label)
            if stop_on_error:
                print("  Stopping batch (--stop-on-error).")
                break

    print(f"\n  Result: {ok}/{ok+fail} succeeded, {fail} failed")
    if fail_list:
        for s in fail_list:
            print(f"    ✗ {s}")

    return fail == 0


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Extract features
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(sim_files: List[Path]) -> bool:
    """
    For each sim file, find its newest export folder and run feature extraction.
    Returns True if all succeeded, False if any failed.
    """
    import pandas as pd
    from anomaly_model.feature_extractor import ScenarioLoader

    n  = len(sim_files)
    ok = fail = skip = 0

    print(f"\n{'─'*60}")
    print(f"  STEP 2 — Extracting features for {n} simulation(s)")
    print(f"{'─'*60}")

    for idx, sim_json in enumerate(sim_files, 1):
        region_dir = sim_json.parent
        true_stem  = sim_json.name.replace(".sim.json", "")
        region_num = region_dir.name.replace("region_", "")
        prefix_key = f"region_{region_num}_"
        test_name  = (true_stem.replace(prefix_key, "", 1)
                      if true_stem.startswith(prefix_key) else true_stem)
        scenario_id = f"{region_dir.name}/{test_name}"
        out_dir     = region_dir / test_name
        label       = f"{region_dir.name}/{test_name}"

        print(f"  [{idx}/{n}] {label:<45} … ", end="", flush=True)

        export_folder = _find_export_folder(region_dir, true_stem)
        if export_folder is None:
            print("SKIP  (no export folder found)")
            skip += 1
            continue

        try:
            loader = ScenarioLoader(
                folder=export_folder,
                scenario_id=scenario_id,
                scenario_json=sim_json,
            )
            df = loader.extract()
            df.insert(3, "label", "anomaly")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "features.csv"
            df.to_csv(out_csv, index=False)
            print(f"OK  ({len(df)} track(s))  →  {out_csv.relative_to(REPO_ROOT)}")
            ok += 1

        except KeyboardInterrupt:
            print("INTERRUPTED")
            return False

        except Exception as exc:
            print(f"FAIL  →  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            fail += 1

    print(f"\n  Result: {ok} extracted, {skip} skipped (no export), {fail} failed")
    return fail == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_anomaly_pipeline",
        description=(
            "Anomaly pipeline: simulate → extract features.\n\n"
            "Runs every anomaly .sim.json headlessly, then extracts features\n"
            "into anomaly_model/simulations/region_XXX/<test_name>/features.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--regions", default=None, metavar="LIST",
        help="Comma-separated region numbers to process (e.g. 120,140). Default: all.",
    )
    p.add_argument(
        "--skip-sim", action="store_true",
        help="Skip Step 1 (simulation); only run feature extraction on existing exports.",
    )
    p.add_argument(
        "--skip-extract", action="store_true",
        help="Skip Step 2 (feature extraction); only run simulations.",
    )
    p.add_argument(
        "--max-duration", type=float, default=DEFAULT_MAX_DURATION,
        metavar="SECONDS", dest="max_duration",
        help=f"Safety cap per scenario in seconds (default: {DEFAULT_MAX_DURATION:.0f}).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-event export progress during simulation.",
    )
    p.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort on first failure in either step.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without running anything.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Build region filter — user can pass "120" or "region_120"
    region_filter: Optional[List[str]] = None
    if args.regions:
        raw = [r.strip() for r in args.regions.split(",") if r.strip()]
        region_filter = [
            r if r.startswith("region_") else f"region_{r}"
            for r in raw
        ]

    # Discover sim files
    try:
        sim_files = _find_sim_files(region_filter)
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return

    if not sim_files:
        print("No anomaly scenario files found — nothing to do.")
        sys.exit(1)

    # Header
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         SimStudio — Anomaly Pipeline                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Simulations root : {SIMS_ROOT}")
    print(f"  Scenarios found  : {len(sim_files)}")
    if region_filter:
        print(f"  Region filter    : {', '.join(region_filter)}")
    steps = []
    if not args.skip_sim:
        steps.append(f"Step 1: simulate ({args.max_duration:.0f}s cap per scenario)")
    if not args.skip_extract:
        steps.append("Step 2: extract features → features.csv")
    for s in steps:
        print(f"  {s}")

    if args.dry_run:
        print("\n  [dry-run] Scenarios that would be processed:")
        for sf in sim_files:
            print(f"    {sf.parent.name}/{sf.stem}")
        print("\n  No files written (--dry-run).")
        return

    # Run pipeline
    t_pipeline = time.time()
    overall_ok = True

    if not args.skip_sim:
        ok = run_simulations(
            sim_files,
            max_duration_s=args.max_duration,
            verbose=args.verbose,
            stop_on_error=args.stop_on_error,
        )
        overall_ok = overall_ok and ok
        if not ok and args.stop_on_error:
            sys.exit(1)

    if not args.skip_extract:
        ok = extract_features(sim_files)
        overall_ok = overall_ok and ok

    # Summary
    elapsed = time.time() - t_pipeline
    print(f"\n{'═'*60}")
    status = "✓ All steps completed successfully" if overall_ok else "✗ Pipeline finished with errors"
    print(f"  {status}  (total: {elapsed:.1f}s)")
    print(f"{'═'*60}\n")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
