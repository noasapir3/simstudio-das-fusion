#!/usr/bin/env python3
"""
auto_clean_sensors.py
=====================
One-shot pipeline that turns raw sim map files into ready-to-simulate files
with clean boundary exit nodes and dense sensor coverage.

Two output files are produced for every source map:

  <region>_clean.sim.json
      The original road graph with `boundary_exit_nodes` added — the set of
      nodes where a vehicle has nowhere further to drive (dead-ends at tile
      edges).  No sensors are placed yet, so you can inspect and tweak the
      topology before running simulations.

  <region>_sensors.sim.json
      Starts from the clean file and auto-places three sensor types for
      near-full coverage:
        • DAS   — one per road segment, covering its full fibre length
        • GPS   — every GPS_INTERVAL_M metres along each segment's polyline
        • Camera — one per node per outgoing segment direction

Supported map collections
--------------------------
  tiles_north_tlv_south/
      Subdirectory layout: each numeric folder (001/, 002/, …) contains a
      single raw file named region_<id>.sim.json.

  tiles_florentin/
      Flat directory layout: tile_<id>_clean.sim.json files directly inside
      the folder (these are already geometrically clean; we only add
      boundary_exit_nodes and sensors).

Usage
-----
    python auto_clean_sensors.py                     # auto-detect paths
    python auto_clean_sensors.py --dry-run           # show what would be written
    python auto_clean_sensors.py --north path/to/tiles_north_tlv_south
    python auto_clean_sensors.py --florentin path/to/tiles_florentin
    python auto_clean_sensors.py --only north        # skip florentin
    python auto_clean_sensors.py --only florentin    # skip north

Sensor placement parameters (edit the constants section below to tune):

    GPS_INTERVAL_M  = 60   m   — distance between consecutive GPS sensors
    GPS_SIGMA_M     = 2.5  m   — position noise std-dev
    GPS_RADIUS_M    = 35   m   — detection radius around each sensor
    CAM_RANGE_M     = 200  m   — maximum detection range per camera
    CAM_FOV_DEG     = 70       — camera field-of-view in degrees
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sensor placement parameters — edit here to tune coverage density / quality
# ---------------------------------------------------------------------------

GPS_INTERVAL_M: float = 80.0   # one GPS sensor every this many metres
GPS_SIGMA_M: float    = 2.5    # position noise std-dev (m)
GPS_UPDATE_HZ: float  = 5.0    # sensor update rate (Hz)
GPS_RADIUS_M: float   = 25.0   # detection radius (m)

CAM_FOV_DEG: float    = 70.0   # field-of-view (degrees)
CAM_RANGE_M: float    = 80.0   # max detection range (m)
CAM_UPDATE_HZ: float  = 10.0   # update rate (Hz)

DAS_UPDATE_HZ: float       = 30.0  # update rate (Hz)
DAS_NOISE_STD: float       = 30.0  # measurement noise std-dev
DAS_FIBER_OFFSET_M: float  = 2.0   # fibre lateral offset (m)
DAS_D0_M: float            = 0.7   # minimum detectable gap (m)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

Point = Tuple[float, float]


def _polyline_length(points: List[Point]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _points_along_polyline(
    points: List[Point], interval_m: float
) -> Iterator[Point]:
    """Yield (x, y) positions spaced *interval_m* apart along a polyline.

    The first point is placed half an interval from the start so sensors are
    centred rather than bunched at segment boundaries.  Very short segments
    (shorter than half an interval) still get one sensor at their midpoint.
    """
    length = _polyline_length(points)
    if length == 0:
        yield (points[0][0], points[0][1])
        return

    first_d = min(interval_m / 2.0, length / 2.0)
    next_d = first_d
    cumlen = 0.0

    for i in range(1, len(points)):
        seg_dx = points[i][0] - points[i - 1][0]
        seg_dy = points[i][1] - points[i - 1][1]
        seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
        if seg_len == 0:
            continue
        while next_d <= cumlen + seg_len:
            frac = (next_d - cumlen) / seg_len
            yield (
                points[i - 1][0] + frac * seg_dx,
                points[i - 1][1] + frac * seg_dy,
            )
            next_d += interval_m
        cumlen += seg_len


def _first_heading(points: List[Point]) -> float:
    """Return the heading (radians, atan2 convention) of the first polyline step."""
    if len(points) < 2:
        return 0.0
    return math.atan2(
        points[1][1] - points[0][1],
        points[1][0] - points[0][0],
    )


# ---------------------------------------------------------------------------
# Dead-end node detection
# ---------------------------------------------------------------------------

def find_boundary_exit_nodes(segments: Dict) -> List[str]:
    """Return every map-boundary node where a vehicle should fire
    *route_complete* rather than stall.

    Three patterns are detected:

    1. **Pure dead-end** — appears as n1 (reachable) but never as n0
       (nowhere to go from here).  Classic one-way stub or terminus.

    2. **Entry-only** — appears as n0 (vehicles leave from here) but never
       as n1 (no segment leads to it).  These are one-way on-ramp tips at
       the edge of the simulated area.  Marking them lets the simulation
       treat them symmetrically with exit nodes.

    3. **Bidirectional turnaround** — the node has exactly one outgoing
       neighbour *and* that neighbour is also one of its incoming sources.
       The only road out simply reverses direction back into the network,
       so the node is effectively a dead-end arm.  Example: n→A where A→n
       also exists and n has no other exits.
    """
    out_nbrs: Dict[str, set] = {}  # node → set of reachable neighbours
    in_nbrs: Dict[str, set] = {}   # node → set of nodes that reach it

    for seg in segments.values():
        out_nbrs.setdefault(seg["n0"], set()).add(seg["n1"])
        in_nbrs.setdefault(seg["n1"], set()).add(seg["n0"])

    all_nodes = set(out_nbrs) | set(in_nbrs)
    exits: List[str] = []

    for node in all_nodes:
        o = out_nbrs.get(node, set())
        i = in_nbrs.get(node, set())

        if not o:
            # Case 1: pure dead-end — can arrive, nowhere to go
            exits.append(node)
        elif not i:
            # Case 2: entry-only — one-way map-edge tip
            exits.append(node)
        elif len(o) == 1 and o <= i:
            # Case 3: bidirectional turnaround — the single outgoing road
            # leads straight back toward where vehicles came from
            exits.append(node)

    return sorted(exits)


# ---------------------------------------------------------------------------
# Sensor generation
# ---------------------------------------------------------------------------

def build_sensors(
    segments: Dict, nodes: Dict
) -> Tuple[Dict, Dict, Dict]:
    """Return (gps, cameras, das) dicts with auto-placed sensors.

    Coverage strategy
    -----------------
    DAS     One sensor per road segment, spanning from channel 0 to the
            segment's arc length.  DAS gives continuous fibre-optic
            detection along the entire road surface.

    GPS     Sensors placed every GPS_INTERVAL_M metres along each segment's
            polyline.  Overlapping radius circles (GPS_RADIUS_M) ensure every
            point on the road is within range of at least one GPS sensor once
            the interval is ≤ 2 × GPS_RADIUS_M (default: 60 m ≤ 70 m).

    Camera  One camera per (source node, outgoing segment) pair, positioned at
            the node and facing along the first segment step.  This captures
            all traffic entering each segment at its start.  At intersections
            with N outgoing roads, N cameras share the same node position but
            have different headings.
    """
    gps: Dict = {}
    cameras: Dict = {}
    das: Dict = {}

    gps_n = cam_n = das_n = 1

    # Build per-node outgoing segment list once.
    outgoing: Dict[str, List] = {}
    for seg in segments.values():
        outgoing.setdefault(seg["n0"], []).append(seg)

    for seg_id, seg in segments.items():
        pts: List[Point] = seg.get("points", [])

        # ── DAS ──────────────────────────────────────────────────────────────
        seg_len = _polyline_length(pts)
        das_id = f"das{das_n}"
        das[das_id] = {
            "id": das_id,
            "segment_id": seg_id,
            "channel_start": 0,
            "channel_end": max(1, int(math.ceil(seg_len))),
            "update_hz": DAS_UPDATE_HZ,
            "noise_std": DAS_NOISE_STD,
            "fiber_offset_m": DAS_FIBER_OFFSET_M,
            "d0_m": DAS_D0_M,
        }
        das_n += 1

        # ── GPS ───────────────────────────────────────────────────────────────
        for gx, gy in _points_along_polyline(pts, GPS_INTERVAL_M):
            gps_id = f"gps{gps_n}"
            gps[gps_id] = {
                "id": gps_id,
                "x": round(gx, 6),
                "y": round(gy, 6),
                "sigma_m": GPS_SIGMA_M,
                "update_hz": GPS_UPDATE_HZ,
                "radius_m": GPS_RADIUS_M,
            }
            gps_n += 1

    # ── Cameras ───────────────────────────────────────────────────────────────
    for node_id, segs_out in outgoing.items():
        node = nodes.get(node_id, {})
        nx = node.get("x", 0.0)
        ny = node.get("y", 0.0)
        for seg in segs_out:
            pts = seg.get("points", [])
            cam_id = f"cam{cam_n}"
            cameras[cam_id] = {
                "id": cam_id,
                "x": round(nx, 6),
                "y": round(ny, 6),
                "heading_rad": round(_first_heading(pts), 6),
                "fov_deg": CAM_FOV_DEG,
                "range_m": CAM_RANGE_M,
                "update_hz": CAM_UPDATE_HZ,
            }
            cam_n += 1

    return gps, cameras, das


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    src: Path,
    clean_dst: Path,
    sensors_dst: Path,
    dry_run: bool,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict:
    """Load *src*, produce clean and sensors variants, write unless dry_run.

    If *overwrite* is False (the default) and an output file already exists,
    that output is skipped — this protects hand-crafted clean files from being
    silently replaced.  Pass overwrite=True or use --overwrite to replace them.
    """
    raw = json.loads(src.read_text(encoding="utf-8"))
    segments: Dict = raw.get("segments", {})
    nodes: Dict = raw.get("nodes", {})

    # ── Build clean data ─────────────────────────────────────────────────────
    exits = find_boundary_exit_nodes(segments)
    clean_data = dict(raw)
    clean_data["boundary_exit_nodes"] = exits
    # Strip any existing sensors so the clean file is sensor-free.
    clean_data["gps"] = {}
    clean_data["cameras"] = {}
    clean_data["das"] = {}

    # ── Build sensors data ───────────────────────────────────────────────────
    gps, cameras, das = build_sensors(segments, nodes)
    sensors_data = dict(clean_data)
    sensors_data["gps"] = gps
    sensors_data["cameras"] = cameras
    sensors_data["das"] = das

    # ── Write outputs ────────────────────────────────────────────────────────
    wrote_clean = wrote_sensors = False
    if not dry_run:
        if overwrite or not clean_dst.exists() or clean_dst == src:
            clean_dst.parent.mkdir(parents=True, exist_ok=True)
            clean_dst.write_text(
                json.dumps(clean_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            wrote_clean = True
        else:
            if verbose:
                print(f"    [SKIP clean] {clean_dst.name} already exists — use --overwrite to replace")

        if overwrite or not sensors_dst.exists():
            sensors_dst.parent.mkdir(parents=True, exist_ok=True)
            sensors_dst.write_text(
                json.dumps(sensors_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            wrote_sensors = True
        else:
            if verbose:
                print(f"    [SKIP sensors] {sensors_dst.name} already exists — use --overwrite to replace")

    result = {
        "source": str(src.name),
        "clean": str(clean_dst.name),
        "sensors": str(sensors_dst.name),
        "n_segments": len(segments),
        "boundary_exit_nodes": exits,
        "n_gps": len(gps),
        "n_cameras": len(cameras),
        "n_das": len(das),
        "wrote_clean": wrote_clean,
        "wrote_sensors": wrote_sensors,
    }

    if verbose:
        status = "[DRY RUN] " if dry_run else ""
        print(
            f"  {status}{src.name}"
            f"  →  exits={len(exits)}"
            f"  segs={len(segments)}"
            f"  gps={len(gps)}"
            f"  cams={len(cameras)}"
            f"  das={len(das)}"
        )

    return result


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_north_tlv(root: Path) -> Iterator[Tuple[Path, Path, Path]]:
    """Yield (src, clean_dst, sensors_dst) for each north-TLV region."""
    # Layout: root/<id>/region_<id>.sim.json
    # NOTE: .sim.json is a double extension; use str replacement rather than
    # Path.stem (which only strips ".json", leaving "region_002.sim").
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.glob("region_*.sim.json"):
            # Skip any files that are already derived outputs.
            if re.search(r"_(clean|sensors|optimized|full|combined|all_)", f.name):
                continue
            base = f.name[: -len(".sim.json")]  # e.g. "region_003"
            yield (
                f,
                subdir / f"{base}_clean.sim.json",
                subdir / f"{base}_sensors.sim.json",
            )


def _discover_florentin(root: Path) -> Iterator[Tuple[Path, Path, Path]]:
    """Yield (src, clean_dst, sensors_dst) for each florentin tile.

    The florentin files are already geometrically clean (short segments
    removed).  The clean_dst overwrites the source file in-place to add
    boundary_exit_nodes; sensors_dst is a new file alongside it.
    """
    for f in sorted(root.glob("tile_*_clean.sim.json")):
        base = f.name[: -len(".sim.json")]  # e.g. "tile_001_clean"
        # Produce tile_XXX_sensors.sim.json alongside the existing clean file.
        sensors_name = base.replace("_clean", "_sensors") + ".sim.json"
        yield f, f, root / sensors_name  # clean_dst == src (in-place update)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Default paths — relative to this script's location
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_NORTH = _SCRIPT_DIR / "maps" / "tiles_north_tlv_south"
_DEFAULT_FLORENTIN = _SCRIPT_DIR / "maps" / "florentin" / "tiles_florentin"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Auto-clean and auto-place sensors on sim map files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--north", type=Path, default=_DEFAULT_NORTH,
        help="Path to tiles_north_tlv_south/ directory.",
    )
    parser.add_argument(
        "--florentin", type=Path, default=_DEFAULT_FLORENTIN,
        help="Path to tiles_florentin/ directory.",
    )
    parser.add_argument(
        "--only", choices=["north", "florentin"],
        help="Process only one collection.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without creating any files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing _clean and _sensors files (default: skip them).",
    )
    args = parser.parse_args(argv)

    all_results: List[Dict] = []
    total_clean = total_sensors = 0

    collections = []
    if args.only != "florentin":
        collections.append(("North TLV", args.north, _discover_north_tlv))
    if args.only != "north":
        collections.append(("Florentin", args.florentin, _discover_florentin))

    for label, root, discover_fn in collections:
        if not root.exists():
            print(f"  [SKIP] {label}: path not found: {root}")
            continue
        print(f"\n{'='*60}")
        print(f" {label}  ({root})")
        print(f"{'='*60}")
        for src, clean_dst, sensors_dst in discover_fn(root):
            try:
                r = process_file(src, clean_dst, sensors_dst, args.dry_run,
                                 overwrite=args.overwrite)
                all_results.append(r)
                if not args.dry_run:
                    total_clean += r.get("wrote_clean", False)
                    total_sensors += r.get("wrote_sensors", False)
            except Exception as exc:
                print(f"  [ERROR] {src.name}: {exc}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Summary")
    print(f"{'='*60}")
    print(f"  Files processed : {len(all_results)}")
    if not args.dry_run:
        print(f"  _clean written  : {total_clean}")
        print(f"  _sensors written: {total_sensors}")
    total_exits   = sum(len(r["boundary_exit_nodes"]) for r in all_results)
    total_gps_all = sum(r["n_gps"] for r in all_results)
    total_cams    = sum(r["n_cameras"] for r in all_results)
    total_das     = sum(r["n_das"] for r in all_results)
    print(f"  Boundary exits  : {total_exits}  (across all maps)")
    print(f"  GPS sensors     : {total_gps_all}")
    print(f"  Camera sensors  : {total_cams}")
    print(f"  DAS sensors     : {total_das}")
    if args.dry_run:
        print("\n  (dry-run — no files were written)")


if __name__ == "__main__":
    main()
