#!/usr/bin/env python3
"""
populate_vehicles.py
====================
Places vehicles into every normal-scenario simulation file under
maps/tiles_north_tlv_south/<area>/simulations/<scenario>/

Each scenario type gets a specific vehicle count range, weight profile,
and speed profile. The script is idempotent: running it again on a file
that already has vehicles will OVERWRITE them (to make re-runs safe).

Usage
-----
    # Dry run — show what would be written, write nothing
    python scripts/populate_vehicles.py --dry-run

    # Run for real (defaults to tiles_north_tlv_south)
    python scripts/populate_vehicles.py

    # Run on a specific area only
    python scripts/populate_vehicles.py --area 024

    # Force a fixed random seed (default: 42 for reproducibility)
    python scripts/populate_vehicles.py --seed 99

Scenario vehicle specs
----------------------
  normal_light_traffic           2–5   standard cars
  normal_medium_traffic          6–12  standard mix
  normal_heavy_traffic           13–18 standard mix  (busy-but-flowing, not congested)
  normal_heavy_vehicles          3–6   trucks / buses
  normal_stop_and_go             6–12  standard mix, low minimum speed (stop-capable)
  normal_fast_legal              5–10  standard mix, speed near limit
  normal_medium_sparse           6–12  standard mix  (sensors pre-thinned)
  normal_medium_das_only         6–12  standard mix  (DAS-only sensors)
  normal_medium_das_cam_only     6–12  standard mix  (DAS + cameras, no GPS)
  normal_medium_minimal          6–12  standard mix  (25% sensors)
  normal_heavy_traffic_sparse    13–18 standard mix  (50% sensors)
  normal_heavy_vehicles_das_only 3–6   trucks / buses (DAS-only sensors)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Scenario specifications
# ---------------------------------------------------------------------------

# Each entry:
#   n_min, n_max      : vehicle count range (inclusive)
#   weight            : 'standard' | 'heavy'
#   speed_factor      : multiplier applied to road speed limit for mean speed
#   stop_and_go       : if True, speed_min_mps is near 0 (vehicles can stop)
SCENARIO_SPECS: Dict[str, dict] = {
    "normal_light_traffic":           {"n_min": 2,  "n_max": 5,  "weight": "standard", "speed_factor": 1.00, "stop_and_go": False},
    "normal_medium_traffic":          {"n_min": 6,  "n_max": 12, "weight": "standard", "speed_factor": 1.00, "stop_and_go": False},
    "normal_heavy_traffic":           {"n_min": 13, "n_max": 18, "weight": "standard", "speed_factor": 1.00, "stop_and_go": False},
    "normal_heavy_vehicles":          {"n_min": 3,  "n_max": 6,  "weight": "heavy",    "speed_factor": 0.70, "stop_and_go": False},
    "normal_stop_and_go":             {"n_min": 6,  "n_max": 12, "weight": "standard", "speed_factor": 0.60, "stop_and_go": True},
    "normal_fast_legal":              {"n_min": 5,  "n_max": 10, "weight": "standard", "speed_factor": 0.92, "stop_and_go": False},
    # Sensor-axis (same traffic as medium)
    "normal_medium_sparse":           {"n_min": 6,  "n_max": 12, "weight": "standard", "speed_factor": 1.00, "stop_and_go": False},
    "normal_medium_das_only":         {"n_min": 6,  "n_max": 12, "weight": "standard", "speed_factor": 1.00, "stop_and_go": False},
    "normal_medium_das_cam_only":     {"n_min": 6,  "n_max": 12, "weight": "standard", "speed_factor": 1.00, "stop_and_go": False},
    "normal_medium_minimal":          {"n_min": 6,  "n_max": 12, "weight": "standard", "speed_factor": 1.00, "stop_and_go": False},
    # Cross-axis
    "normal_heavy_traffic_sparse":    {"n_min": 13, "n_max": 18, "weight": "standard", "speed_factor": 1.00, "stop_and_go": False},
    "normal_heavy_vehicles_das_only": {"n_min": 3,  "n_max": 6,  "weight": "heavy",    "speed_factor": 0.70, "stop_and_go": False},
}

DEFAULT_SPEED_LIMIT_MPS: float = 13.9   # ~50 km/h
SPEED_MIN_KMH: float = 20.0
SPEED_MAX_KMH: float = 100.0

# ---------------------------------------------------------------------------
# Geometry helpers (no SimStudio import required)
# ---------------------------------------------------------------------------

def polyline_length(pts: List[List[float]]) -> float:
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        total += math.hypot(dx, dy)
    return total


def point_at_s(pts: List[List[float]], s: float) -> Tuple[float, float]:
    """Return (x, y) at curvilinear position s along a polyline."""
    remaining = s
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        seg_len = math.hypot(dx, dy)
        if remaining <= seg_len or i == len(pts) - 1:
            t = remaining / max(1e-9, seg_len)
            return (pts[i-1][0] + t * dx, pts[i-1][1] + t * dy)
        remaining -= seg_len
    return (pts[-1][0], pts[-1][1])


def lane_heading(pts: List[List[float]], s: float) -> float:
    p1 = point_at_s(pts, s)
    p2 = point_at_s(pts, s + 1.0)
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def trunc_gauss(mean: float, std: float, lo: float, hi: float) -> float:
    for _ in range(20):
        v = random.gauss(mean, std)
        if lo <= v <= hi:
            return v
    return clamp(mean, lo, hi)

# ---------------------------------------------------------------------------
# Weight sampling
# ---------------------------------------------------------------------------

def sample_weight(profile: str) -> float:
    """
    'standard' : urban vehicle mix (cars dominant, ~8% SUV/van, 5% buses, 3% trucks)
    'heavy'    : trucks and buses only (for normal_heavy_vehicles scenarios)
    """
    if profile == "heavy":
        r = random.random()
        if r < 0.60:
            return trunc_gauss(10_000.0, 1_500.0, 8_000.0, 14_000.0)  # city bus
        else:
            return trunc_gauss(8_500.0,  2_000.0, 4_500.0, 15_000.0)  # HGV
    else:  # standard
        r = random.random()
        if r < 0.55:
            return trunc_gauss(1_050.0, 120.0, 800.0,   1_400.0)  # small/compact
        elif r < 0.80:
            return trunc_gauss(1_400.0, 140.0, 1_200.0, 1_800.0)  # sedan
        elif r < 0.92:
            return trunc_gauss(1_850.0, 220.0, 1_500.0, 2_800.0)  # SUV/van
        elif r < 0.97:
            return trunc_gauss(10_000.0, 1_500.0, 8_000.0, 14_000.0)  # bus
        else:
            return trunc_gauss(8_500.0,  2_000.0, 4_500.0, 15_000.0)  # truck

# ---------------------------------------------------------------------------
# Vehicle placement
# ---------------------------------------------------------------------------

def build_vehicle(vid: str, lane_id: str, lane_poly: List[List[float]],
                  speed_limit_mps: float, spec: dict) -> dict:
    """Construct a fully-specified vehicle dict for one vehicle."""
    L = polyline_length(lane_poly)

    # Position: spread across 90% of the lane so vehicles aren't bunched at start
    s = random.uniform(0.0, max(5.0, 0.9 * L))

    weight_kg = sample_weight(spec["weight"])

    # --- Speed profile ---
    vlim = speed_limit_mps if speed_limit_mps > 0 else DEFAULT_SPEED_LIMIT_MPS
    vlim_kmh = vlim * 3.6

    # mean speed is speed_factor × limit, with gaussian noise
    mean_kmh = clamp(spec["speed_factor"] * vlim_kmh, SPEED_MIN_KMH, SPEED_MAX_KMH)
    v_kmh    = trunc_gauss(mean_kmh, 8.0, SPEED_MIN_KMH, SPEED_MAX_KMH)
    v_mps    = v_kmh / 3.6

    if spec["stop_and_go"]:
        speed_min  = trunc_gauss(0.5, 0.3, 0.0, 2.0)        # can stop near 0
        speed_mean = max(speed_min + 1.0, v_mps)
        speed_max  = min(vlim * 1.05, speed_mean + random.uniform(3.0, 8.0))
        # Slower response time — stop-and-go needs longer braking/accelerating
        max_decel  = random.uniform(3.0, 5.0)                # heavier braking
        max_accel  = random.uniform(0.8, 1.8)
    else:
        speed_min  = max(6.0, v_mps - random.uniform(3.0, 8.0))
        speed_mean = max(speed_min + 1.5, v_mps)
        speed_max  = min(42.0, max(speed_mean + random.uniform(3.0, 8.0), vlim * random.uniform(0.95, 1.05)))
        max_decel  = random.uniform(1.8, 3.3)
        max_accel  = random.uniform(1.1, 2.4)

    # Heavy vehicles: lower acceleration / deceleration caps
    if spec["weight"] == "heavy":
        max_accel = random.uniform(0.5, 1.2)
        max_decel = random.uniform(1.2, 2.2)

    speed_std         = max(0.5, 0.18 * max(1.0, speed_max - speed_min))
    interval_min      = random.uniform(3.0, 6.5)
    interval_max      = random.uniform(max(interval_min + 2.0, 7.0), 13.0)
    interval_mean     = 0.5 * (interval_min + interval_max)

    return {
        "id":                           vid,
        "lane_id":                      lane_id,
        "s":                            round(s, 6),
        "v":                            round(v_mps, 6),
        "a_long_mps2":                  0.0,
        "weight_kg":                    round(weight_kg, 1),
        "lateral_offset_m":             0.0,
        "speed_mean_mps":               round(speed_mean, 6),
        "speed_std_mps":                round(speed_std, 6),
        "speed_min_mps":                round(speed_min, 6),
        "speed_max_mps":                round(speed_max, 6),
        "target_speed_mps":             round(v_mps, 6),
        "speed_change_interval_mean_s": round(interval_mean, 6),
        "speed_change_interval_min_s":  round(interval_min, 6),
        "speed_change_interval_max_s":  round(interval_max, 6),
        "next_speed_change_t":          0.0,
        "last_speed_change_t":          -1_000_000_000.0,
        "cruise_hold_probability":      0.3 if spec["stop_and_go"] else 0.0,
        "accel_response_s":             1.5,
        "max_accel_mps2":               round(max_accel, 6),
        "max_decel_mps2":               round(max_decel, 6),
        "heading_rad":                  round(lane_heading(lane_poly, s), 6),
        "ax_world_mps2":                0.0,
        "ay_world_mps2":                0.0,
    }


def populate_scenario(sim_path: Path, spec: dict, dry_run: bool) -> int:
    """Read a scenario file, place vehicles, write back.  Returns vehicle count."""
    with open(sim_path) as f:
        data = json.load(f)

    lanes    = data.get("lanes", {})
    segments = data.get("segments", {})
    if not lanes:
        return 0

    # Map lane_id → speed_limit_mps
    lane_speed: Dict[str, float] = {}
    for lid, lane in lanes.items():
        seg = segments.get(lane.get("segment_id", ""), {})
        lane_speed[lid] = float(seg.get("speed_limit_mps") or DEFAULT_SPEED_LIMIT_MPS)

    lane_ids = list(lanes.keys())
    n_veh = random.randint(spec["n_min"], spec["n_max"])

    vehicles: Dict[str, dict] = {}
    for i in range(1, n_veh + 1):
        vid      = f"veh{i:03d}"
        lane_id  = random.choice(lane_ids)
        lane_obj = lanes[lane_id]
        poly     = lane_obj.get("polyline", [[0, 0], [1, 0]])
        vlim     = lane_speed[lane_id]
        veh      = build_vehicle(vid, lane_id, poly, vlim, spec)
        vehicles[vid] = veh

    data["vehicles"] = vehicles

    if not dry_run:
        with open(sim_path, "w") as f:
            json.dump(data, f, indent=2)

    return n_veh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Populate vehicles into simulation scenario files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written without modifying any files.")
    parser.add_argument("--area", default=None,
                        help="Process a single area only (e.g. '024').")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--base", default=None,
                        help="Path to tiles_north_tlv_south directory.")
    args = parser.parse_args()

    random.seed(args.seed)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base = Path(args.base) if args.base else project_root / "maps" / "tiles_north_tlv_south"

    if not base.exists():
        print(f"ERROR: base directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[DRY RUN] No files will be modified.\n")

    areas = sorted([d for d in os.listdir(base)
                    if (base / d).is_dir() and d != ".DS_Store"])
    if args.area:
        if args.area not in areas:
            print(f"ERROR: area '{args.area}' not found in {base}", file=sys.stderr)
            sys.exit(1)
        areas = [args.area]

    total_files   = 0
    total_vehicles = 0
    skipped        = 0

    for area in areas:
        sims_dir = base / area / "simulations"
        if not sims_dir.exists():
            continue
        for scenario_name, spec in SCENARIO_SPECS.items():
            scenario_dir = sims_dir / scenario_name
            if not scenario_dir.exists():
                continue
            # Find the single scenario .sim.json in this folder
            sim_files = list(scenario_dir.glob("*.sim.json"))
            if not sim_files:
                skipped += 1
                continue
            sim_path = sim_files[0]
            n = populate_scenario(sim_path, spec, dry_run=args.dry_run)
            total_files   += 1
            total_vehicles += n
            if args.dry_run:
                print(f"  {area}/{scenario_name}: {n} vehicles  [{sim_path.name}]")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done.")
    print(f"  Scenario files {'processed' if not args.dry_run else 'would be processed'}: {total_files}")
    print(f"  Total vehicles {'placed' if not args.dry_run else 'to place'}: {total_vehicles}")
    if skipped:
        print(f"  Skipped (no .sim.json found): {skipped}")


if __name__ == "__main__":
    main()
