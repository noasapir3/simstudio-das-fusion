"""World randomization utilities.

Goal:
    Given an already-designed scene (nodes / segments), populate it with a
    realistic mix of vehicles and sensors in plausible / strategic locations.

Design constraints:
    - No dependency on the GUI layer.
    - No mutations of Simulation.step() logic.
    - Reproducible via an explicit seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random

from .constants import (
    DEFAULT_LANE_WIDTH_M,
    DEFAULT_SPEED_LIMIT_MPS,
    SPEED_MIN_KMH,
    SPEED_MAX_KMH,
    SPEED_STD_RATIO,
)
from .geometry import point_at_s, polyline_length
from .models import World, Vehicle, GPSSensor, CameraSensor, DASSensor
from .utils import clamp, trunc_gauss, sample_vehicle_profile

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Public configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RandomizeSpec:
    """Parameters that control how a world is randomly populated.

    *Count* fields follow the convention:
        - If ``n_<type>`` is set it is used as an exact count.
        - Otherwise a random value in ``[<type>_min, <type>_max]`` is chosen.
        - If neither is set, a heuristic based on scene size is used.
    """

    seed: Optional[int] = None

    # --- Vehicle counts ---
    n_vehicles: Optional[int] = None
    vehicles_min: Optional[int] = None
    vehicles_max: Optional[int] = None

    # --- Sensor counts ---
    n_gps: Optional[int] = None
    gps_min: Optional[int] = None
    gps_max: Optional[int] = None

    n_cameras: Optional[int] = None
    cameras_min: Optional[int] = None
    cameras_max: Optional[int] = None

    n_das: Optional[int] = None
    das_min: Optional[int] = None
    das_max: Optional[int] = None

    # --- Vehicle property overrides ---
    # spec format: {"kind": "gauss"|"uniform", "min": a, "max": b, "mean": m, "std": s}
    vehicle_speed_kmh: Optional[dict] = None
    vehicle_weight_kg: Optional[dict] = None

    # --- Continuous spawning ---
    spawn_enabled: bool = False
    spawn_rate_vpm: float = 0.0        # vehicles per minute
    spawn_max_total: Optional[int] = None

    clear_existing: bool = True


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_choice(items: List[str]) -> Optional[str]:
    return random.choice(items) if items else None


def _lane_heading(lane_poly: List[Point], s: float) -> float:
    """Approximate tangent heading at curvilinear position *s* on *lane_poly*."""
    if len(lane_poly) < 2:
        return 0.0
    L = polyline_length(lane_poly)
    s1 = clamp(s, 0.0, L)
    s2 = clamp(s + 1.0, 0.0, L)
    p1 = point_at_s(lane_poly, s1)
    p2 = point_at_s(lane_poly, s2)
    vx, vy = (p2[0] - p1[0], p2[1] - p1[1])
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return 0.0
    return math.atan2(vy, vx)


def _next_id(prefix: str, existing: Dict[str, object], start: int = 1) -> str:
    """Return the first unused key of the form ``{prefix}{i:03d}``."""
    i = start
    while True:
        cand = f"{prefix}{i:03d}"
        if cand not in existing:
            return cand
        i += 1


def _scene_bounds(world: World) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) over all nodes, or (0,0,0,0) if empty."""
    xs = [float(n.x) for n in world.nodes.values()]
    ys = [float(n.y) for n in world.nodes.values()]
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _sample_from_spec(spec: Optional[dict], default_value: float) -> float:
    """Sample a scalar from a distribution specification dict.

    Spec format::

        {"kind": "gauss"|"uniform", "min": a, "max": b, "mean": m, "std": s}
    """
    if not spec:
        return float(default_value)
    kind = str(spec.get("kind") or "").lower().strip()
    a = spec.get("min")
    b = spec.get("max")
    try:
        a = float(a) if a is not None else None
        b = float(b) if b is not None else None
    except Exception:
        a, b = None, None

    def _clamp_if(x: float) -> float:
        if a is not None and b is not None:
            return clamp(float(x), float(a), float(b))
        if a is not None:
            return max(float(a), float(x))
        if b is not None:
            return min(float(b), float(x))
        return float(x)

    if kind == "uniform":
        if a is None or b is None:
            return float(default_value)
        return _clamp_if(random.uniform(float(a), float(b)))

    # Default: Gaussian.
    try:
        m = float(spec.get("mean"))
        s = float(spec.get("std"))
    except Exception:
        return float(default_value)
    if s <= 0:
        return _clamp_if(m)
    return _clamp_if(random.gauss(m, s))


def _choose_count(
    n_exact: Optional[int],
    n_min: Optional[int],
    n_max: Optional[int],
    fallback: int,
) -> int:
    """Resolve a count from exact / range / fallback specification."""
    if n_exact is not None:
        return max(0, int(n_exact))
    if n_min is not None or n_max is not None:
        a = 0 if n_min is None else int(n_min)
        b = a if n_max is None else int(n_max)
        if b < a:
            a, b = b, a
        return int(random.randint(a, b))
    return max(0, int(fallback))


def _node_degrees(lane_meta: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    """Count the number of lanes incident on each node."""
    deg: Dict[str, int] = {}
    for _, m in lane_meta.items():
        for role in ("from", "to"):
            nid = m.get(role)
            if nid:
                deg[nid] = deg.get(nid, 0) + 1
    return deg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def randomize_world(
    world: World,
    lane_meta: Dict[str, Dict[str, str]],
    spec: RandomizeSpec,
    *,
    on_log: Optional[callable] = None,
) -> Dict[str, int]:
    """Populate *world* with vehicles and sensors according to *spec*.

    Args:
        world:     The World object to mutate in-place.
        lane_meta: Mapping of lane_id → {"from": node_id, "to": node_id, "dir": ...}
                   as produced by ``Simulation.rebuild_lanes()``.
        spec:      :class:`RandomizeSpec` controlling counts and distributions.
        on_log:    Optional callback for progress messages (used by the GUI).

    Returns:
        A dict with inserted counts, e.g.
        ``{"vehicles": 12, "gps": 3, "cameras": 2, "das": 1}``.
    """
    if spec.seed is not None:
        random.seed(int(spec.seed))

    if spec.clear_existing:
        world.vehicles.clear()
        world.gps.clear()
        world.cameras.clear()
        world.das.clear()

    lane_ids = list(world.lanes.keys())
    if not lane_ids:
        raise ValueError(
            "Cannot randomize: world has no lanes.  Call rebuild_lanes() first."
        )

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    # Heuristics: choose counts proportional to scene topology size.
    lanes_n = len(lane_ids)
    seg_n = len(world.segments)
    n_veh = _choose_count(spec.n_vehicles, spec.vehicles_min, spec.vehicles_max,
                          int(clamp(lanes_n // 3, 6, 45)))
    n_gps = _choose_count(spec.n_gps, spec.gps_min, spec.gps_max,
                          int(clamp(max(1, seg_n // 6), 1, 10)))
    n_cam = _choose_count(spec.n_cameras, spec.cameras_min, spec.cameras_max,
                          int(clamp(max(1, seg_n // 8), 1, 10)))
    n_das = _choose_count(spec.n_das, spec.das_min, spec.das_max,
                          int(clamp(max(0, seg_n // 10), 0, 8)))

    deg = _node_degrees(lane_meta)
    nodes_sorted = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)
    complex_nodes = [nid for nid, d in nodes_sorted if d >= 4]
    if not complex_nodes:
        complex_nodes = [nid for nid, _ in nodes_sorted[: max(1, min(6, len(nodes_sorted)))]]
    end_nodes = [nid for nid, d in deg.items() if d <= 1]

    # Outgoing lanes grouped by origin node.
    out_by_node: Dict[str, List[str]] = {}
    for lid, m in lane_meta.items():
        a = m.get("from")
        if a:
            out_by_node.setdefault(a, []).append(lid)

    xmin, ymin, xmax, ymax = _scene_bounds(world)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    # ---- Vehicles ----
    veh_created = 0
    for _ in range(n_veh):
        use_complex = (random.random() < 0.60) and bool(complex_nodes)
        if use_complex:
            nid = _safe_choice(complex_nodes)
        else:
            nid = _safe_choice(end_nodes) or _safe_choice(complex_nodes)

        cand_lanes = out_by_node.get(nid, []) if nid else []
        if not cand_lanes:
            cand_lanes = lane_ids
        lid = random.choice(cand_lanes)

        lane = world.lanes[lid]
        L = polyline_length(lane.polyline)
        s = random.random() * clamp(0.35 * L, 5.0, max(5.0, L))

        seg = world.segments.get(lane.segment_id)
        vlim = (
            float(getattr(seg, "speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS) or DEFAULT_SPEED_LIMIT_MPS)
            if seg else DEFAULT_SPEED_LIMIT_MPS
        )

        # Base realistic profile from the shared vehicle sampler.
        w, v = sample_vehicle_profile(vlim)

        # Apply any user-supplied distribution overrides.
        v_kmh = _sample_from_spec(spec.vehicle_speed_kmh, v * 3.6)
        v_kmh = clamp(float(v_kmh), SPEED_MIN_KMH, SPEED_MAX_KMH)
        v = float(v_kmh) / 3.6
        w = clamp(float(_sample_from_spec(spec.vehicle_weight_kg, w)), 500.0, 40000.0)

        vid = _next_id("veh", world.vehicles)
        speed_min = max(6.0, min(v - random.uniform(3.0, 8.0), vlim * 0.85))
        speed_mean = max(speed_min + 1.5, v)
        speed_max = min(
            42.0,
            max(speed_mean + random.uniform(3.0, 10.0), vlim * random.uniform(0.95, 1.12)),
        )
        speed_std = max(0.5, SPEED_STD_RATIO * max(1.0, speed_max - speed_min))
        interval_min = random.uniform(3.0, 6.5)
        interval_max = random.uniform(max(interval_min + 2.0, 7.0), 13.0)
        interval_mean = 0.5 * (interval_min + interval_max)

        world.vehicles[vid] = Vehicle(
            id=vid,
            lane_id=lid,
            s=float(s),
            v=float(v),
            weight_kg=float(w),
            speed_mean_mps=float(speed_mean),
            speed_std_mps=float(speed_std),
            speed_min_mps=float(speed_min),
            speed_max_mps=float(speed_max),
            target_speed_mps=float(v),
            speed_change_interval_mean_s=float(interval_mean),
            speed_change_interval_min_s=float(interval_min),
            speed_change_interval_max_s=float(interval_max),
            cruise_hold_probability=0.0,
            accel_response_s=1.5,
            max_accel_mps2=float(random.uniform(1.1, 2.4)),
            max_decel_mps2=float(random.uniform(1.8, 3.3)),
        )
        veh_created += 1

    _log(f"Randomized vehicles: {veh_created}")

    # ---- GPS sensors ----
    gps_created = 0
    for _ in range(n_gps):
        nid = _safe_choice(complex_nodes) or _safe_choice(list(world.nodes.keys()))
        if nid and nid in world.nodes:
            nx, ny = world.nodes[nid].x, world.nodes[nid].y
        else:
            nx, ny = cx, cy

        x = float(nx + random.uniform(-20.0, 20.0))
        y = float(ny + random.uniform(-20.0, 20.0))

        sid = _next_id("gps", world.gps)
        world.gps[sid] = GPSSensor(
            id=sid,
            x=x,
            y=y,
            sigma_m=float(random.uniform(0.7, 5.5)),
            update_hz=float(random.uniform(1.0, 8.0)),
            radius_m=float(random.uniform(30.0, 120.0)),
        )
        gps_created += 1

    _log(f"Randomized GPS sensors: {gps_created}")

    # ---- Camera sensors ----
    cam_created = 0
    for _ in range(n_cam):
        nid = _safe_choice(complex_nodes) or _safe_choice(list(world.nodes.keys()))
        cand_lanes = out_by_node.get(nid, []) if nid else []
        lid = _safe_choice(cand_lanes) or _safe_choice(lane_ids)
        if not lid:
            break

        lane = world.lanes[lid]
        L = polyline_length(lane.polyline)
        s0 = clamp(random.uniform(0.0, 8.0), 0.0, L)
        x0, y0 = point_at_s(lane.polyline, s0)
        hd = _lane_heading(lane.polyline, s0)

        back = random.uniform(8.0, 25.0)
        side = random.uniform(-12.0, 12.0)
        x = float(x0 - math.cos(hd) * back - math.sin(hd) * side)
        y = float(y0 - math.sin(hd) * back + math.cos(hd) * side)

        sid = _next_id("cam", world.cameras)
        world.cameras[sid] = CameraSensor(
            id=sid,
            x=x,
            y=y,
            heading_rad=float(hd),
            fov_deg=float(random.uniform(55.0, 110.0)),
            range_m=float(random.uniform(120.0, 320.0)),
            update_hz=float(random.uniform(6.0, 18.0)),
        )
        cam_created += 1

    _log(f"Randomized cameras: {cam_created}")

    # ---- DAS sensors ----
    das_created = 0
    if n_das > 0 and world.segments:
        # Prefer longer segments (more fiber coverage).
        seg_lengths = []
        for seg_id, _seg in world.segments.items():
            cand = [lid for lid, ln in world.lanes.items() if ln.segment_id == seg_id]
            if not cand:
                continue
            L = polyline_length(world.lanes[cand[0]].polyline)
            seg_lengths.append((seg_id, L))
        seg_lengths.sort(key=lambda kv: kv[1], reverse=True)
        seg_pool = [sid for sid, _ in seg_lengths] or list(world.segments.keys())

        for _ in range(n_das):
            seg_id = random.choice(seg_pool[: max(1, min(len(seg_pool), 20))])
            sid = _next_id("das", world.das)
            c0 = int(random.uniform(0, 200))
            c1 = int(c0 + random.uniform(800, 1400))
            world.das[sid] = DASSensor(
                id=sid,
                segment_id=str(seg_id),
                channel_start=int(c0),
                channel_end=int(c1),
                update_hz=float(random.uniform(15.0, 40.0)),
                noise_std=float(random.uniform(0.01, 0.05)),
            )
            das_created += 1

    _log(f"Randomized DAS sensors: {das_created}")

    return {
        "vehicles": veh_created,
        "gps": gps_created,
        "cameras": cam_created,
        "das": das_created,
    }


def spawn_vehicles_over_time(
    world: World,
    lane_meta: Dict[str, Dict[str, str]],
    spec: RandomizeSpec,
    dt: float,
    state: dict,
) -> int:
    """Spawn vehicles gradually according to ``spec.spawn_rate_vpm``.

    *state* is a mutable dict owned by the caller (the GUI) and must persist
    across calls.  It accumulates fractional-vehicle debt between steps.

    Args:
        world:     World to mutate.
        lane_meta: Lane topology metadata from ``Simulation.rebuild_lanes()``.
        spec:      Must have ``spawn_enabled=True`` and ``spawn_rate_vpm > 0``.
        dt:        Elapsed simulation time since last call (seconds).
        state:     Persistent accumulator dict (pass the same dict each call).

    Returns:
        Number of vehicles spawned in this call (0 or more).
    """
    if not spec.spawn_enabled or float(spec.spawn_rate_vpm or 0.0) <= 0.0:
        return 0

    if spec.spawn_max_total is not None and len(world.vehicles) >= int(spec.spawn_max_total):
        return 0

    acc = float(state.get("acc", 0.0)) + float(dt)
    interval = 60.0 / max(1e-6, float(spec.spawn_rate_vpm))

    spawned = 0
    while acc >= interval:
        acc -= interval
        tmp = RandomizeSpec(
            seed=None,
            n_vehicles=1,
            n_gps=0,
            n_cameras=0,
            n_das=0,
            clear_existing=False,
            vehicle_speed_kmh=spec.vehicle_speed_kmh,
            vehicle_weight_kg=spec.vehicle_weight_kg,
        )
        randomize_world(world, lane_meta, tmp)
        spawned += 1
        if spec.spawn_max_total is not None and len(world.vehicles) >= int(spec.spawn_max_total):
            break

    state["acc"] = acc
    return int(spawned)
