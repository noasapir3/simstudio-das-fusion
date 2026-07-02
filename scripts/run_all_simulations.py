"""
run_all_simulations.py
======================
Headless batch runner for SimStudio scenarios.

Usage
-----
  python scripts/run_all_simulations.py all
  python scripts/run_all_simulations.py 001
  python scripts/run_all_simulations.py 001-020
  python scripts/run_all_simulations.py 010,025,047      # comma list
  python scripts/run_all_simulations.py all --dry-run
  python scripts/run_all_simulations.py all --max-duration 600

For every discovered .sim.json the script:
  1. Loads a fresh World from the JSON file.
  2. Builds a new Simulation + EventBus (no shared state with other runs).
  3. Wires a TrackManager to the bus.
  4. Steps until every vehicle has left the map (exited or stuck).
     A safety cap (--max-duration, default 600 s) prevents infinite loops.
  5. Calls export_all() and writes the tracking-audit bundle into the
     scenario folder, named  <stem>_export_<YYYYMMDD_HHMMSS>_tracking_audit/.

All objects are discarded between runs — data can never bleed across scenarios.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup – make sure we can import simstudio regardless of cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAPS_ROOT: Path = _REPO_ROOT / "maps" / "tiles_north_tlv_south"

SIM_DT: float = 0.02          # simulation time-step (s) — matches the GUI's 50 Hz loop
DEFAULT_MAX_DURATION: float = 120.0  # safety cap (s) — stops a stuck simulation

# Scenario folder names to skip entirely during discovery.
EXCLUDED_SCENARIOS: set = {
    "normal_medium_cam_gps_only",
}

# Topics captured into the event list passed to export_all
CAPTURE_TOPICS = (
    "world.vehicle_state",
    "world.vehicle_stuck",
    "world.collision",
    "world.route_complete",
    "sensor.gps",
    "sensor.camera",
    "sensor.das",
)

# Topics the TrackManager must receive (subset of CAPTURE_TOPICS)
TRACKER_TOPICS = (
    "world.vehicle_state",
    "world.vehicle_stuck",
    "sensor.gps",
    "sensor.camera",
    "sensor.das",
)


# ---------------------------------------------------------------------------
# Area / scenario discovery
# ---------------------------------------------------------------------------

def _all_area_names() -> List[str]:
    """Return sorted list of 3-digit area folder names that exist on disk."""
    return sorted(
        d.name for d in MAPS_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 3
    )


def parse_area_spec(spec: str) -> List[str]:
    """
    Parse an area specification string and return a list of 3-digit area names.

    Accepted formats
    ----------------
    ``all``         → every area found in MAPS_ROOT
    ``001``         → single area
    ``001-020``     → inclusive range
    ``001,010,047`` → explicit comma-separated list
    """
    spec = spec.strip()
    if spec.lower() == "all":
        return _all_area_names()

    if "," in spec:
        return [s.strip().zfill(3) for s in spec.split(",") if s.strip()]

    if "-" in spec:
        parts = spec.split("-", 1)
        try:
            lo, hi = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid area range '{spec}'. Use format '001-020'."
            )
        return [f"{i:03d}" for i in range(lo, hi + 1)]

    # Single area
    return [spec.zfill(3)]


def find_scenario_files(areas: List[str]) -> List[Path]:
    """
    Discover all .sim.json scenario files for the given area names.

    Only files under  maps/tiles_north_tlv_south/{NNN}/simulations/**/*.sim.json
    are returned, matching the convention used by all 13 scenario types.
    """
    found: List[Path] = []
    for area in areas:
        sim_root = MAPS_ROOT / area / "simulations"
        if not sim_root.exists():
            continue
        for p in sorted(sim_root.glob("**/*.sim.json")):
            if p.parent.name in EXCLUDED_SCENARIOS:
                continue
            found.append(p)
    return found


# ---------------------------------------------------------------------------
# Single-scenario runner
# ---------------------------------------------------------------------------

def _normalize_lanes(world) -> None:
    """Enforce one lane per direction — mirrors what the GUI does on load."""
    for seg in world.segments.values():
        try:
            seg.lanes = 1
        except Exception:
            pass


def run_scenario(
    sim_json: Path,
    max_duration_s: float = DEFAULT_MAX_DURATION,
    verbose: bool = False,
) -> dict:
    """
    Run one .sim.json scenario headlessly and write export_all artefacts.

    The simulation runs until every vehicle has left the map (via a boundary
    exit node, a stuck event, or natural route completion).  A safety cap of
    ``max_duration_s`` prevents an infinite loop if a vehicle somehow never
    leaves.

    Returns a summary dict with keys: events, steps, sim_time_s, out_dir,
    elapsed_s, stop_reason.
    Raises on unrecoverable errors so the caller can log and continue.

    State isolation
    ---------------
    Every object created here (EventBus, World, Simulation, TrackManager) is
    local to this function call.  Nothing is shared with other invocations.
    """
    # -- lazy imports (only done once per process, cached by Python) ----------
    from simstudio.bus import EventBus
    from simstudio.project_io import load_world
    from simstudio.sim_core import Simulation
    import simstudio.audit as audit_mod

    t_start = time.time()

    # 1. Load world -----------------------------------------------------------
    world = load_world(sim_json)
    _normalize_lanes(world)

    # 2. Bus + simulation -----------------------------------------------------
    bus = EventBus()
    sim = Simulation(bus, world)
    sim.rebuild_lanes()

    # 3. Event capture --------------------------------------------------------
    all_events: list = []

    def _capture(ev, _lst=all_events):
        _lst.append(ev)

    for topic in CAPTURE_TOPICS:
        bus.subscribe(topic, _capture)

    # 4. TrackManager (best-effort; simulation works without it) --------------
    try:
        from simstudio.tracking import TrackManager

        tm = TrackManager(world, bus=bus)

        def _tm_cb(ev, _m=tm):
            try:
                _m.on_event(ev)
            except Exception:
                pass

        for topic in TRACKER_TOPICS:
            bus.subscribe(topic, _tm_cb)

    except Exception as exc:
        if verbose:
            print(f"    [warn] TrackManager unavailable: {exc}")

    # 5. Step simulation — run until all vehicles have left the map -----------
    #
    # Primary stop condition : world.vehicles becomes empty (every vehicle
    #   exited via a boundary node or completed its route).
    # Secondary stop condition: all remaining vehicles have emitted a
    #   world.vehicle_stuck event (they'll never move again).
    # Safety cap : max_duration_s prevents an infinite loop for edge cases.
    #
    safety_steps = int(max_duration_s / SIM_DT)
    stuck_ids: set = set()
    steps_run = 0
    stop_reason = "safety_cap"

    def _track_stuck(ev, _s=stuck_ids):
        vid = (ev.payload or {}).get("vehicle_id")
        if vid:
            _s.add(str(vid))

    bus.subscribe("world.vehicle_stuck", _track_stuck)

    for _ in range(safety_steps):
        sim.step(SIM_DT)
        steps_run += 1

        # All vehicles gone from the world
        if not world.vehicles:
            stop_reason = "all_exited"
            break

        # All remaining vehicles are permanently stuck
        if world.vehicles and stuck_ids.issuperset(world.vehicles.keys()):
            stop_reason = "all_stuck"
            break

    remaining = len(world.vehicles)
    if stop_reason == "safety_cap" and verbose:
        print(
            f"    [warn] safety cap reached ({max_duration_s:.0f} s); "
            f"{remaining} vehicle(s) still on map"
        )
    elif stop_reason == "all_stuck" and verbose:
        print(f"    [info] {remaining} vehicle(s) stuck — stopping early")

    # 6. Export ---------------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = sim_json.parent / f"{sim_json.stem}_export_{ts}_tracking_audit"

    try:
        audit_mod.export_all(
            all_events,
            out_dir,
            world=world,
            progress_cb=(lambda msg: print(f"    {msg}")) if verbose else None,
        )
    except (KeyboardInterrupt, SystemExit):
        # User interrupted mid-export.  Remove the partial output folder so
        # it is not mistaken for a valid completed export.
        _remove_partial_export(out_dir)
        raise  # propagate so the batch loop can print "INTERRUPTED" and exit

    elapsed = time.time() - t_start
    return {
        "events": len(all_events),
        "steps": steps_run,
        "sim_time_s": steps_run * SIM_DT,
        "out_dir": str(out_dir),
        "elapsed_s": elapsed,
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_partial_export(out_dir: Path) -> None:
    """Delete an incomplete export folder so it is not mistaken for valid data."""
    import shutil
    try:
        if out_dir.exists():
            shutil.rmtree(out_dir)
    except Exception:
        pass  # best-effort only


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_all_simulations",
        description=(
            "Headless batch runner for SimStudio scenarios.\n\n"
            "Area spec examples:\n"
            "  all          → all 149 areas\n"
            "  001          → single area\n"
            "  001-020      → inclusive range\n"
            "  001,010,047  → explicit list\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "areas",
        help="Area spec: 'all', single area ('001'), range ('001-020'), or comma list",
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        metavar="SECONDS",
        dest="max_duration",
        help=(
            f"Safety cap in seconds (default: {DEFAULT_MAX_DURATION:.0f}). "
            "Each simulation stops as soon as all vehicles exit — this cap "
            "only fires if a vehicle somehow never leaves."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered scenarios without running them",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-step progress from export_all",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the entire batch on the first scenario failure",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # -- Resolve areas --------------------------------------------------------
    try:
        areas = parse_area_spec(args.areas)
    except ValueError as exc:
        parser.error(str(exc))

    # Filter to areas that actually exist on disk
    existing_areas = [a for a in areas if (MAPS_ROOT / a).exists()]
    missing = set(areas) - set(existing_areas)
    if missing:
        print(f"[warn] {len(missing)} area(s) not found on disk: {sorted(missing)[:10]}{'…' if len(missing) > 10 else ''}")

    scenarios = find_scenario_files(existing_areas)

    if not scenarios:
        print("No .sim.json scenario files found for the requested areas.")
        sys.exit(1)

    print(
        f"Found {len(scenarios)} scenario(s) across {len(existing_areas)} area(s)"
        + (" [DRY RUN]" if args.dry_run else f" — running each until all vehicles exit (cap {args.max_duration:.0f} s)")
    )

    if args.dry_run:
        for s in scenarios:
            # Show relative path from MAPS_ROOT for readability
            try:
                rel = s.relative_to(MAPS_ROOT)
            except ValueError:
                rel = s
            print(f"  {rel}")
        return

    # -- Batch run ------------------------------------------------------------
    ok = fail = 0
    fail_list: List[str] = []

    for idx, sim_json in enumerate(scenarios, 1):
        # Friendly label: "095/normal_light_traffic"
        try:
            label = str(sim_json.relative_to(MAPS_ROOT / sim_json.parts[-4] / "simulations").parent)
            area = sim_json.parts[-4]
            label = f"{area}/{sim_json.parent.name}"
        except Exception:
            label = str(sim_json)

        prefix = f"[{idx:>{len(str(len(scenarios)))}}/{len(scenarios)}]"
        print(f"{prefix} {label} … ", end="", flush=True)

        try:
            result = run_scenario(sim_json, max_duration_s=args.max_duration, verbose=args.verbose)
            elapsed = result["elapsed_s"]
            ev = result["events"]
            sim_t = result["sim_time_s"]
            reason = result["stop_reason"]
            reason_tag = "" if reason == "all_exited" else f" [{reason}]"
            print(
                f"OK  "
                f"({ev} events | {sim_t:.1f} s sim-time in {elapsed:.1f} s real-time{reason_tag})"
            )
            ok += 1

        except KeyboardInterrupt:
            print("INTERRUPTED — partial output cleaned up")
            break

        except Exception as exc:
            print(f"FAIL  →  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            fail += 1
            fail_list.append(label)
            if args.stop_on_error:
                print("Stopping batch (--stop-on-error set).")
                break

    # -- Summary --------------------------------------------------------------
    total = ok + fail
    print()
    print("=" * 60)
    print(f"Batch complete:  {ok}/{total} succeeded,  {fail} failed")
    if fail_list:
        print("Failed scenarios:")
        for name in fail_list:
            print(f"  ✗ {name}")
    print("=" * 60)

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
