"""Urban-environment simulation engine for SimStudio.

Responsibilities:
    - Maintain the simulation clock and advance vehicle kinematics each tick.
    - Generate synthetic sensor measurements (GPS, Camera, DAS) and publish
      them via the EventBus.
    - Provide shortest-path routing and destination-following for vehicles.
    - Maintain optional traffic-density targets per road segment.

Key design decisions:
    - Vehicles are modelled as *points* on a lane centreline (no length / bounding box).
    - Lateral drift follows an Ornstein–Uhlenbeck process bounded to ±½ lane width.
    - All random sampling uses the module-level ``random`` PRNG so results are
      reproducible when the caller seeds it before calling ``rebuild_lanes()``.
"""

from __future__ import annotations

import copy
import heapq
import math
import random
from typing import Dict, List, Optional, Tuple

from .bus import EventBus
from .constants import (
    ACCEL_CMD_DEADBAND_MPS,
    ACCEL_RESPONSE_TAU_S,
    CAMERA_PIXEL_WIDTH,
    CAMERA_SIGMA_PX,
    CRUISE_DEADBAND_MPS,
    DEFAULT_LANE_WIDTH_M,
    DEFAULT_MAX_ACCEL_MPS2,
    DEFAULT_MAX_DECEL_MPS2,
    DEFAULT_SPEED_LIMIT_MPS,
    IDM_A_MAX_MPS2,
    IDM_B_MPS2,
    IDM_DELTA,
    IDM_S0_M,
    IDM_T_S,
    IDM_VEHICLE_LENGTH_M,
    LATERAL_DRIFT_TAU_S,
    TRAFFIC_DENSITY_VPK,
)
from .geometry import (
    offset_polyline,
    point_at_s,
    polyline_length,
    polyline_nearest_s,
    tangent_at_s,
)
from .models import LaneGeom, Vehicle, World, make_lane_polylines
from .utils import sample_vehicle_profile, trunc_gauss, wrap_angle

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _tangent_heading(poly: List[Point], s: float) -> float:
    """Approximate tangent heading (radians) of *poly* at arc-length *s*.

    Returns 0.0 for degenerate polylines shorter than 2 points.
    """
    if len(poly) < 2:
        return 0.0
    L = polyline_length(poly)
    s1 = max(0.0, min(L, s))
    s2 = max(0.0, min(L, s + 1.0))
    p1 = point_at_s(poly, s1)
    p2 = point_at_s(poly, s2)
    vx, vy = (p2[0] - p1[0], p2[1] - p1[1])
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return 0.0
    return math.atan2(vy, vx)


def _pose_with_lateral(poly: List[Point], s: float, lateral_offset_m: float) -> Point:
    """Return the world-space position of a vehicle at arc-length *s* with lateral offset."""
    x_center, y_center = point_at_s(poly, s)
    heading = _tangent_heading(poly, s)
    nx, ny = (-math.sin(heading), math.cos(heading))
    return (x_center + lateral_offset_m * nx, y_center + lateral_offset_m * ny)


def _two_way_lane_polylines(
    n0: Point, n1: Point, lanes_per_dir: int, lane_width: float
) -> Tuple[List[List[Point]], List[List[Point]]]:
    """Return (polys_fwd, polys_bwd) for a two-way road with right-hand traffic.

    Forward lanes are offset to the right of the n0→n1 direction; backward
    lanes run in the opposite direction on the left.  Each direction gets
    *lanes_per_dir* lanes.
    """
    x0, y0 = n0
    x1, y1 = n1
    dx, dy = (x1 - x0, y1 - y0)
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return [[n0, n1]], [[n1, n0]]
    nx, ny = (-dy / L, dx / L)
    polys_fwd = []
    polys_bwd = []
    for i in range(lanes_per_dir):
        off_mag = (i + 0.5) * lane_width
        polys_fwd.append(
            [(x0 + nx * off_mag, y0 + ny * off_mag), (x1 + nx * off_mag, y1 + ny * off_mag)]
        )
        polys_bwd.append(
            [(x1 - nx * off_mag, y1 - ny * off_mag), (x0 - nx * off_mag, y0 - ny * off_mag)]
        )
    return polys_fwd, polys_bwd


# ---------------------------------------------------------------------------
# Routing sentinel
# ---------------------------------------------------------------------------

# Returned by _pick_next_lane() when the segment-visit cap has been reached
# for every candidate and the vehicle should exit cleanly.  Must be a string
# that is never a valid lane_id.
_LOOP_CAP_EXIT: str = "__loop_cap_exit__"

# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------

class Simulation:
    """Traffic simulator and synthetic sensor generator.

    Lifecycle:
        1. Construct with an ``EventBus`` and a ``World``.
        2. Call ``rebuild_lanes()`` after any road-network change.
        3. Call ``step(dt)`` in a loop to advance time.
        4. Subscribe to topics on the bus to receive sensor events.

    Vehicle routing:
        - At every junction a vehicle randomly chooses a non-U-turn lane.
        - Planned routes override random choices; use ``set_route_to_node()``
          or ``set_route_to_point()`` / ``set_route_queue()`` for point routing.
        - Vehicles that reach a dead-end emit ``world.vehicle_stuck`` and stop.
    """

    def __init__(self, bus: EventBus, world: World) -> None:
        self.bus = bus
        self.world = world
        self.t: float = 0.0

        # Tracks the last time each sensor/vehicle emitted an event (for rate control).
        self._last_emit: Dict[str, float] = {}

        # Lane-level topology metadata produced by rebuild_lanes().
        self._lane_meta: Dict[str, Dict[str, str]] = {}

        # Vehicles that have reached a dead-end and are no longer moving.
        self._stuck: Dict[str, bool] = {}

        # Route state: planned lane sequences and point destinations.
        self._route_next: Dict[str, List[str]] = {}
        self._route_dest_node: Dict[str, str] = {}
        self._route_dest_lane: Dict[str, str] = {}
        self._route_dest_s: Dict[str, float] = {}
        self._route_queue: Dict[str, List[Tuple[str, float]]] = {}

        # Per-vehicle segment visit counts for loop prevention.
        # Maps vehicle_id → {segment_id: visit_count}.
        # Incremented every time a vehicle enters a new lane.
        self._seg_visit_count: Dict[str, Dict[str, int]] = {}

        # Per-vehicle stochastic dynamics profile (seeded lazily).
        self._veh_dyn: Dict[str, Dict[str, float]] = {}

        # Traffic-density control layer.
        # Reads auto_spawn from the World (set via "auto_spawn": false in the
        # scenario JSON) so that validation scenarios are not contaminated by
        # automatically spawned vehicles.
        self._traffic_enabled: bool = bool(getattr(world, "auto_spawn", True))
        self._traffic_density_vpkm_per_lane: Dict[str, float] = dict(TRAFFIC_DENSITY_VPK)

        # Previous DAS (x_meas, y_meas) per (sensor_id, vehicle_id) used for 2D
        # finite-difference velocity estimation.  Keyed by (sid, vid) → (t_prev, x_prev, y_prev).
        # Reset whenever rebuild_lanes() is called so stale entries don't survive
        # a scenario reload.
        self._das_prev_s: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

        # Anomaly bookkeeping.  Keyed by frozenset({follower_id, leader_id}) so
        # the same collision is never re-published if the bumper-to-bumper
        # overlap persists across multiple ticks (vehicles freeze on impact
        # but the pair stays geometrically overlapping forever).
        self._collided_pairs: set = set()
        # Vehicles whose route has already emitted ``world.route_complete``.
        # Used to suppress duplicate events when the dest is held for many ticks.
        self._route_completed: Dict[str, bool] = {}

        # Per-vehicle ``random.Random`` instances for deterministic lateral
        # ``random_walk`` mode.  Seeded lazily on first use from
        # ``Vehicle.lateral_random_seed``; if that field is 0 the global
        # ``random`` module is used (non-deterministic across runs).
        self._lateral_rng: Dict[str, "random.Random"] = {}

    # ------------------------------------------------------------------
    # Public lane / dynamics management
    # ------------------------------------------------------------------

    def reset_vehicle_dynamics_profile(self, vehicle_id: Optional[str] = None) -> None:
        """Forget the cached dynamics profile for one or all vehicles.

        The profile will be re-seeded from the Vehicle dataclass on the next
        ``step()`` call.
        """
        if vehicle_id is None:
            self._veh_dyn.clear()
        else:
            self._veh_dyn.pop(str(vehicle_id), None)

    def apply_vehicle_macro_profile(self, vehicle_id: str) -> None:
        """Re-seed the dynamics profile for *vehicle_id* from its Vehicle fields."""
        veh = self.world.vehicles.get(str(vehicle_id))
        if veh is None:
            return
        self.reset_vehicle_dynamics_profile(str(vehicle_id))
        seg_id = str(getattr(self.world.lanes.get(veh.lane_id), "segment_id", "") or "")
        seg = self.world.segments.get(seg_id) if seg_id else None
        speed_limit = float(getattr(seg, "speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS) or DEFAULT_SPEED_LIMIT_MPS)
        self._ensure_vehicle_dynamics_profile(veh, speed_limit)

    def rebuild_lanes(self) -> None:
        """Rebuild the lane geometry from the current segment/node topology.

        Must be called after any structural change to ``world.segments`` or
        ``world.nodes``.  Also remaps existing vehicles to valid lanes and
        clears stuck flags.
        """
        lanes_new: Dict[str, LaneGeom] = {}
        meta: Dict[str, Dict[str, str]] = {}

        for seg_id, seg in self.world.segments.items():
            # Enforce project constraint: one lane per direction.
            # one_way is intentionally NOT overridden here — bidirectional
            # segments (one_way=False) must reach the two-way geometry branch.
            try:
                seg.lanes = 1
            except Exception:
                pass

            n0 = self.world.nodes.get(seg.n0)
            n1 = self.world.nodes.get(seg.n1)
            if not n0 or not n1:
                continue

            base_poly = list(getattr(seg, "points", []) or [n0.p(), n1.p()])

            if getattr(seg, "one_way", False):
                # One-way: lanes centred on the segment centreline.
                if getattr(seg, "points", None):
                    lanes_count = max(1, min(7, int(getattr(seg, "lanes", 1))))
                    mid = (lanes_count - 1) / 2.0
                    for i in range(lanes_count):
                        off = (i - mid) * float(getattr(seg, "lane_width", DEFAULT_LANE_WIDTH_M))
                        poly = offset_polyline(base_poly, off)
                        lid = f"{seg_id}_fwd_lane{i + 1}"
                        lanes_new[lid] = LaneGeom(id=lid, segment_id=seg_id, offset_index=i, polyline=poly)
                        meta[lid] = {"from": seg.n0, "to": seg.n1, "dir": "fwd"}
                else:
                    for i, poly in enumerate(
                        make_lane_polylines(n0.p(), n1.p(), seg.lanes, seg.lane_width)
                    ):
                        lid = f"{seg_id}_fwd_lane{i + 1}"
                        lanes_new[lid] = LaneGeom(id=lid, segment_id=seg_id, offset_index=i, polyline=poly)
                        meta[lid] = {"from": seg.n0, "to": seg.n1, "dir": "fwd"}
            else:
                # Two-way: separated lane centrelines (right-hand traffic).
                if getattr(seg, "points", None):
                    lanes_per_dir = max(1, min(7, int(getattr(seg, "lanes", 1))))
                    for i in range(lanes_per_dir):
                        off_mag = (i + 0.5) * float(getattr(seg, "lane_width", DEFAULT_LANE_WIDTH_M))
                        poly_f = offset_polyline(base_poly, +off_mag)
                        poly_b = list(reversed(offset_polyline(base_poly, -off_mag)))
                        lidf = f"{seg_id}_fwd_lane{i + 1}"
                        lidb = f"{seg_id}_bwd_lane{i + 1}"
                        lanes_new[lidf] = LaneGeom(id=lidf, segment_id=seg_id, offset_index=i, polyline=poly_f)
                        lanes_new[lidb] = LaneGeom(id=lidb, segment_id=seg_id, offset_index=i, polyline=poly_b)
                        meta[lidf] = {"from": seg.n0, "to": seg.n1, "dir": "fwd"}
                        meta[lidb] = {"from": seg.n1, "to": seg.n0, "dir": "bwd"}
                else:
                    polys_fwd, polys_bwd = _two_way_lane_polylines(
                        n0.p(), n1.p(), seg.lanes, seg.lane_width
                    )
                    for i, poly in enumerate(polys_fwd):
                        lid = f"{seg_id}_fwd_lane{i + 1}"
                        lanes_new[lid] = LaneGeom(id=lid, segment_id=seg_id, offset_index=i, polyline=poly)
                        meta[lid] = {"from": seg.n0, "to": seg.n1, "dir": "fwd"}
                    for i, poly in enumerate(polys_bwd):
                        lid = f"{seg_id}_bwd_lane{i + 1}"
                        lanes_new[lid] = LaneGeom(id=lid, segment_id=seg_id, offset_index=i, polyline=poly)
                        meta[lid] = {"from": seg.n1, "to": seg.n0, "dir": "bwd"}

        self.world.lanes = lanes_new
        self._lane_meta = meta

        # Best-effort remap vehicles to valid lanes (e.g. after toggling one_way).
        for _, v in self.world.vehicles.items():
            if v.lane_id in self.world.lanes:
                continue
            # Extract the segment prefix from IDs like "seg1_fwd_lane1" or
            # "seg1_bwd_lane1".  Try both directions so backward-lane vehicles
            # are not incorrectly remapped to an invalid candidate.
            remapped = False
            for marker in ("_fwd_lane", "_bwd_lane"):
                if marker in v.lane_id:
                    seg_part = v.lane_id.split(marker)[0]
                    for cand in (f"{seg_part}_fwd_lane1", f"{seg_part}_bwd_lane1"):
                        if cand in self.world.lanes:
                            v.lane_id = cand
                            v.s = 0.0
                            remapped = True
                            break
                    break
            if remapped:
                continue
            any_lane = next(iter(self.world.lanes), None)
            if any_lane:
                v.lane_id = any_lane
                v.s = 0.0

        self._stuck = {}
        self._seg_visit_count.clear()
        self._das_prev_s.clear()
        self._collided_pairs = set()
        self._route_completed = {}
        self._lateral_rng = {}

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    def _should_emit(self, key: str, hz: float) -> bool:
        """Return True if enough time has elapsed to emit at the given frequency."""
        if hz <= 0:
            return False
        period = 1.0 / hz
        last = self._last_emit.get(key, -1e9)
        if (self.t - last) >= period:
            self._last_emit[key] = self.t
            return True
        return False

    def _outgoing_lanes_from_node(self, node_id: str) -> List[str]:
        return [lid for lid, m in self._lane_meta.items() if m.get("from") == node_id]

    def _pick_next_lane(self, current_lane_id: str, vehicle_id: str = "") -> str:
        """Randomly pick a non-U-turn outgoing lane from the end of *current_lane_id*.

        When *vehicle_id* is supplied the selection uses a **tiered preference**
        to avoid routing loops while still allowing a vehicle to continue
        moving when no fresh roads are available:

        * **Tier 1 — unvisited** (count == 0): always preferred.
        * **Tier 2 — once-visited** (count == 1): accepted when no Tier-1
          candidate exists.  A vehicle passing the same segment twice is still
          realistic (turnaround, shared arterial, etc.).
        * **Cap** (all candidates count ≥ 2): the vehicle has exhausted the
          local topology and is in a genuine routing loop.  Returns
          ``_LOOP_CAP_EXIT`` so the caller can remove the vehicle cleanly
          without emitting a "stuck" event.

        Returns an empty string if the vehicle has reached a genuine dead-end
        (no outgoing lanes at all), or ``_LOOP_CAP_EXIT`` on loop-cap.
        """
        lane = self.world.lanes.get(current_lane_id)
        if lane is None:
            return current_lane_id

        meta = self._lane_meta.get(current_lane_id, {})
        to_node = meta.get("to")
        from_node = meta.get("from")
        if not to_node:
            return current_lane_id

        candidates = self._outgoing_lanes_from_node(to_node)
        if not candidates:
            return ""

        # Avoid immediate U-turns when alternatives exist.
        if from_node:
            non_uturn = [lid for lid in candidates if self._lane_meta.get(lid, {}).get("to") != from_node]
            if non_uturn:
                candidates = non_uturn

        # Prefer staying on the same lane index for a straight-ahead feel.
        desired_idx = lane.offset_index
        same_idx = [
            lid for lid in candidates
            if self.world.lanes.get(lid) and self.world.lanes[lid].offset_index == desired_idx
        ]
        if same_idx:
            candidates = same_idx

        # ── Tiered segment-visit preference ───────────────────────────────────
        # Applied only for random routing (vehicle_id supplied).
        # Planned routes (_route_next) bypass this entirely — they are consumed
        # in _pick_next_lane_for_vehicle before this method is called.
        if vehicle_id:
            _visits = self._seg_visit_count.get(vehicle_id, {})

            def _visit_count(lid: str) -> int:
                seg = getattr(self.world.lanes.get(lid), "segment_id", "")
                return _visits.get(seg, 0)

            # Tier 1: segments not yet visited at all.
            tier1 = [lid for lid in candidates if _visit_count(lid) == 0]
            if tier1:
                candidates = tier1
            else:
                # Tier 2: segments visited exactly once (turnaround / shared road).
                tier2 = [lid for lid in candidates if _visit_count(lid) == 1]
                if tier2:
                    candidates = tier2
                else:
                    # Every reachable segment is at the cap — genuine loop trap
                    # with no exit.  Signal a clean removal to the caller.
                    return _LOOP_CAP_EXIT

        candidates.sort()
        return random.choice(candidates)

    # ------------------------------------------------------------------
    # Public routing API
    # ------------------------------------------------------------------

    def set_vehicle_route(self, vehicle_id: str, route_lanes: List[str], dest_node: str) -> None:
        """Set a planned lane sequence for a vehicle.

        *route_lanes* should begin with the vehicle's current lane.
        Only the *subsequent* lanes are stored internally (the vehicle is
        already on the first one).
        """
        if not route_lanes:
            self._route_next.pop(vehicle_id, None)
            self._route_dest_node.pop(vehicle_id, None)
            return
        self._route_next[vehicle_id] = list(route_lanes[1:])
        self._route_dest_node[vehicle_id] = dest_node

    def clear_vehicle_route(self, vehicle_id: str) -> None:
        """Clear all route state for *vehicle_id* (does not affect the queue)."""
        self._route_next.pop(vehicle_id, None)
        self._route_dest_node.pop(vehicle_id, None)
        self._route_dest_lane.pop(vehicle_id, None)
        self._route_dest_s.pop(vehicle_id, None)

    def clear_route(self, vehicle_id: str) -> None:
        """Clear route state *and* the waypoint queue for *vehicle_id*."""
        self._route_queue.pop(vehicle_id, None)
        self.clear_vehicle_route(vehicle_id)

    def shortest_route_lanes(self, start_lane_id: str, dest_node: str) -> List[str]:
        """Compute the shortest lane sequence from *start_lane_id* to *dest_node*.

        Returns a lane-id list beginning with *start_lane_id*, or an empty list
        if no path exists.
        """
        meta0 = self._lane_meta.get(start_lane_id)
        if not meta0:
            return []
        start_node = meta0.get("to")
        if not start_node:
            return []
        if start_node == dest_node:
            return [start_lane_id]

        # Build a directed node-adjacency graph weighted by lane length.
        adj: Dict[str, List[Tuple[str, str, float]]] = {}
        for lid, m in self._lane_meta.items():
            a = m.get("from")
            b = m.get("to")
            if not a or not b:
                continue
            lane = self.world.lanes.get(lid)
            if not lane:
                continue
            w = polyline_length(lane.polyline)
            adj.setdefault(a, []).append((b, lid, w))

        INF = 1e18
        dist_map: Dict[str, float] = {start_node: 0.0}
        prev_node: Dict[str, str] = {}
        prev_lane: Dict[str, str] = {}
        pq: List[Tuple[float, str]] = [(0.0, start_node)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist_map.get(u, INF):
                continue
            if u == dest_node:
                break
            for v, lid, w in adj.get(u, []):
                nd = d + w
                if nd < dist_map.get(v, INF):
                    dist_map[v] = nd
                    prev_node[v] = u
                    prev_lane[v] = lid
                    heapq.heappush(pq, (nd, v))

        if dest_node not in dist_map:
            return []

        lanes: List[str] = []
        cur = dest_node
        while cur != start_node:
            lid = prev_lane.get(cur)
            pu = prev_node.get(cur)
            if not lid or not pu:
                return []
            lanes.append(lid)
            cur = pu
        lanes.reverse()
        return [start_lane_id] + lanes

    def set_route_to_node(self, vehicle_id: str, dest_node_id: str) -> bool:
        """Route *vehicle_id* to *dest_node_id* via the shortest path.

        Returns ``True`` if a route was found, ``False`` otherwise.
        """
        veh = self.world.vehicles.get(vehicle_id)
        if veh is None:
            return False
        cur_lane = self.world.lanes.get(veh.lane_id)
        if cur_lane is None:
            return False
        meta = self._lane_meta.get(veh.lane_id, {})
        start_node = meta.get("to")
        if not start_node:
            return False
        if dest_node_id == start_node:
            self._route_next[vehicle_id] = []
            self._route_dest_node[vehicle_id] = dest_node_id
            return True

        path_nodes = self._shortest_path_nodes(start_node, dest_node_id)
        if not path_nodes:
            return False

        lanes: List[str] = []
        prev_lane_idx = cur_lane.offset_index
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            cand = [
                lid for lid, m in self._lane_meta.items()
                if m.get("from") == a and m.get("to") == b
            ]
            if not cand:
                return False
            same = [
                lid for lid in cand
                if self.world.lanes.get(lid)
                and self.world.lanes[lid].offset_index == prev_lane_idx
            ]
            if same:
                cand = same
            cand.sort()
            chosen = cand[0]
            lanes.append(chosen)
            prev_lane_idx = self.world.lanes[chosen].offset_index

        self._route_next[vehicle_id] = lanes
        self._route_dest_node[vehicle_id] = dest_node_id
        self._route_dest_lane.pop(vehicle_id, None)
        self._route_dest_s.pop(vehicle_id, None)
        return True

    def set_route_to_point(self, vehicle_id: str, dest_lane_id: str, dest_s: float) -> bool:
        """Route *vehicle_id* to the point ``(dest_lane_id, dest_s)`` on the map.

        Returns ``True`` if a route was found, ``False`` otherwise.
        """
        veh = self.world.vehicles.get(vehicle_id)
        if veh is None:
            return False
        if veh.lane_id not in self.world.lanes:
            return False
        if dest_lane_id not in self.world.lanes:
            return False

        dest_lane = self.world.lanes[dest_lane_id]
        Ld = polyline_length(dest_lane.polyline)
        dest_s = max(0.0, min(Ld, float(dest_s)))

        if veh.lane_id == dest_lane_id:
            self._route_next[vehicle_id] = []
            self._route_dest_lane[vehicle_id] = dest_lane_id
            self._route_dest_s[vehicle_id] = dest_s
            self._route_dest_node.pop(vehicle_id, None)
            return True

        meta_cur = self._lane_meta.get(veh.lane_id, {})
        meta_dst = self._lane_meta.get(dest_lane_id, {})
        start_node = meta_cur.get("to")
        dest_from = meta_dst.get("from")
        if not start_node or not dest_from:
            return False

        if start_node == dest_from:
            lanes = [dest_lane_id]
        else:
            path_nodes = self._shortest_path_nodes(start_node, dest_from)
            if not path_nodes:
                return False
            lanes = []
            prev_lane_idx = self.world.lanes[veh.lane_id].offset_index
            for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                cand = [
                    lid for lid, m in self._lane_meta.items()
                    if m.get("from") == a and m.get("to") == b
                ]
                if not cand:
                    return False
                same = [
                    lid for lid in cand
                    if self.world.lanes.get(lid)
                    and self.world.lanes[lid].offset_index == prev_lane_idx
                ]
                if same:
                    cand = same
                cand.sort()
                chosen = cand[0]
                lanes.append(chosen)
                prev_lane_idx = self.world.lanes[chosen].offset_index
            lanes.append(dest_lane_id)

        self._route_next[vehicle_id] = lanes
        self._route_dest_lane[vehicle_id] = dest_lane_id
        self._route_dest_s[vehicle_id] = dest_s
        self._route_dest_node.pop(vehicle_id, None)
        return True

    def set_route_queue(self, vehicle_id: str, waypoints: List[Tuple[str, float]]) -> bool:
        """Set an ordered list of destination points ``(lane_id, s)``.

        The simulation routes the vehicle to each waypoint in order.
        Returns ``False`` if the first waypoint is unreachable.
        """
        if not waypoints:
            self._route_queue.pop(vehicle_id, None)
            self.clear_vehicle_route(vehicle_id)
            return True

        self._route_queue[vehicle_id] = list(waypoints)
        first_lane, first_s = self._route_queue[vehicle_id][0]
        ok = self.set_route_to_point(vehicle_id, first_lane, first_s)
        if not ok:
            self._route_queue.pop(vehicle_id, None)
            return False
        return True

    def _shortest_path_nodes(self, start_node: str, dest_node: str) -> List[str]:
        """Dijkstra on the directed lane graph; returns node-id list including endpoints."""
        if start_node == dest_node:
            return [start_node]

        adj: Dict[str, List[Tuple[str, float]]] = {}
        for lid, m in self._lane_meta.items():
            a = m.get("from")
            b = m.get("to")
            if not a or not b:
                continue
            lane = self.world.lanes.get(lid)
            if not lane:
                continue
            adj.setdefault(a, []).append((b, polyline_length(lane.polyline)))

        INF = 1e18
        dist_map: Dict[str, float] = {start_node: 0.0}
        prev: Dict[str, str] = {}
        pq: List[Tuple[float, str]] = [(0.0, start_node)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist_map.get(u, INF):
                continue
            if u == dest_node:
                break
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist_map.get(v, INF):
                    dist_map[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dest_node not in dist_map:
            return []

        path = [dest_node]
        cur = dest_node
        while cur != start_node:
            cur = prev.get(cur)
            if cur is None:
                return []
            path.append(cur)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Vehicle dynamics (private)
    # ------------------------------------------------------------------

    def _sample_vehicle_target_speed(
        self,
        cur_v: float,
        profile: Dict[str, float],
        speed_limit_mps: float,
        allow_cruise: bool,
    ) -> float:
        """Sample a new target speed from the vehicle's stochastic profile."""
        speed_min = max(1.5, float(profile.get("speed_min_mps", 0.0) or 0.0))
        speed_cap = max(
            speed_min + 0.5,
            min(max(speed_limit_mps * 1.15, speed_min + 1.0), 42.0),
        )
        speed_max = max(
            speed_min + 0.5,
            min(float(profile.get("speed_max_mps", speed_cap) or speed_cap), speed_cap),
        )
        mean = min(
            speed_max - 0.1,
            max(
                speed_min + 0.1,
                float(profile.get("speed_mean_mps", 0.5 * (speed_min + speed_max)))
                or (0.5 * (speed_min + speed_max)),
            ),
        )
        std = max(
            0.45,
            min(
                float(profile.get("speed_std_mps", 0.18 * (speed_max - speed_min)))
                or (0.18 * (speed_max - speed_min)),
                0.40 * (speed_max - speed_min),
            ),
        )

        if allow_cruise and random.random() < 0.30:
            cand = random.gauss(cur_v, max(0.25, 0.25 * std))
        else:
            cand = trunc_gauss(mean, std, speed_min, speed_max)

        # Avoid tiny target changes so speed updates remain visually perceptible.
        min_delta = max(0.5, 0.10 * max(speed_max - speed_min, 1.0))
        if abs(cand - cur_v) < min_delta and random.random() < 0.75:
            bump = max(0.8, 0.18 * (speed_max - speed_min))
            if cand >= cur_v:
                cand = min(speed_max, cur_v + bump)
            else:
                cand = max(speed_min, cur_v - bump)
        return max(speed_min, min(speed_max, cand))

    def _sample_speed_change_interval(self, profile: Dict[str, float]) -> float:
        """Sample the next speed-update interval in seconds."""
        interval_min = max(1.5, float(profile.get("speed_change_interval_min_s", 0.0) or 0.0))
        interval_max = max(interval_min + 0.5, float(profile.get("speed_change_interval_max_s", 0.0) or 0.0))
        mean = min(
            interval_max - 0.1,
            max(
                interval_min + 0.1,
                float(profile.get("speed_change_interval_mean_s", 0.5 * (interval_min + interval_max)))
                or (0.5 * (interval_min + interval_max)),
            ),
        )
        std = max(0.35, 0.22 * (interval_max - interval_min))
        return trunc_gauss(mean, std, interval_min, interval_max)

    def _ensure_vehicle_dynamics_profile(
        self, veh: Vehicle, seg_speed_limit_mps: float
    ) -> Dict[str, float]:
        """Lazily seed and return the dynamics profile for *veh*.

        The profile is stored in ``self._veh_dyn`` and mirrored back to the
        Vehicle fields for GUI display and CSV export.
        """
        speed_limit_mps = max(4.0, float(seg_speed_limit_mps or DEFAULT_SPEED_LIMIT_MPS))
        cur_v = max(0.0, float(getattr(veh, "v", 0.0) or 0.0))
        profile = self._veh_dyn.setdefault(str(veh.id), {})

        seeded_min = float(
            profile.get("speed_min_mps", 0.0) or float(getattr(veh, "speed_min_mps", 0.0) or 0.0)
        )
        seeded_max = float(
            profile.get("speed_max_mps", 0.0) or float(getattr(veh, "speed_max_mps", 0.0) or 0.0)
        )
        if seeded_min <= 0.0 or seeded_max <= 0.0 or seeded_max <= seeded_min:
            if seeded_max <= 0.0:
                seeded_max = max(5.0, min(speed_limit_mps * random.uniform(0.82, 1.02), speed_limit_mps * 1.08, 39.0))
            if seeded_min <= 0.0 or seeded_min >= seeded_max:
                seeded_min = max(1.5, min(seeded_max - 0.8, seeded_max * random.uniform(0.50, 0.78)))
        profile["speed_min_mps"] = max(1.5, min(seeded_min, seeded_max - 0.5))
        profile["speed_max_mps"] = max(
            profile["speed_min_mps"] + 0.5,
            min(seeded_max, max(speed_limit_mps * 1.15, profile["speed_min_mps"] + 1.0), 42.0),
        )

        if float(profile.get("speed_mean_mps", 0.0) or 0.0) <= 0.0:
            seeded_mean = float(getattr(veh, "speed_mean_mps", 0.0) or 0.0)
            if seeded_mean <= 0.0:
                center = 0.5 * (float(profile["speed_min_mps"]) + float(profile["speed_max_mps"]))
                seeded_mean = trunc_gauss(
                    center,
                    0.12 * (float(profile["speed_max_mps"]) - float(profile["speed_min_mps"])),
                    float(profile["speed_min_mps"]),
                    float(profile["speed_max_mps"]),
                )
            profile["speed_mean_mps"] = seeded_mean

        if float(profile.get("speed_std_mps", 0.0) or 0.0) <= 0.0:
            seeded_std = float(getattr(veh, "speed_std_mps", 0.0) or 0.0)
            if seeded_std <= 0.0:
                seeded_std = max(0.7, min(5.0, 0.20 * (float(profile["speed_max_mps"]) - float(profile["speed_min_mps"]))))
            profile["speed_std_mps"] = seeded_std

        if float(profile.get("speed_change_interval_mean_s", 0.0) or 0.0) <= 0.0:
            v = float(getattr(veh, "speed_change_interval_mean_s", 0.0) or 0.0)
            profile["speed_change_interval_mean_s"] = v if v > 0.0 else random.uniform(5.0, 11.0)

        if float(profile.get("speed_change_interval_min_s", 0.0) or 0.0) <= 0.0:
            v = float(getattr(veh, "speed_change_interval_min_s", 0.0) or 0.0)
            profile["speed_change_interval_min_s"] = (
                v if v > 0.0 else max(2.0, 0.45 * float(profile["speed_change_interval_mean_s"]))
            )

        if float(profile.get("speed_change_interval_max_s", 0.0) or 0.0) <= 0.0:
            v = float(getattr(veh, "speed_change_interval_max_s", 0.0) or 0.0)
            profile["speed_change_interval_max_s"] = (
                v if v > 0.0
                else max(
                    float(profile["speed_change_interval_min_s"]) + 1.0,
                    1.75 * float(profile["speed_change_interval_mean_s"]),
                )
            )

        profile["cruise_hold_probability"] = 0.0
        profile["accel_response_s"] = ACCEL_RESPONSE_TAU_S

        mass = float(getattr(veh, "weight_kg", 1500.0) or 1500.0)
        if float(profile.get("max_accel_mps2", 0.0) or 0.0) <= 0.0:
            default_acc = 0.85 if mass >= 7000.0 else (1.35 if mass >= 2400.0 else 2.10)
            profile["max_accel_mps2"] = float(getattr(veh, "max_accel_mps2", 0.0) or default_acc)
        if float(profile.get("max_decel_mps2", 0.0) or 0.0) <= 0.0:
            default_dec = 1.20 if mass >= 7000.0 else (2.10 if mass >= 2400.0 else 3.10)
            profile["max_decel_mps2"] = float(getattr(veh, "max_decel_mps2", 0.0) or default_dec)

        if "target_speed_mps" not in profile:
            seeded_target = float(getattr(veh, "target_speed_mps", 0.0) or 0.0)
            if seeded_target <= 0.0:
                seeded_target = self._sample_vehicle_target_speed(
                    cur_v=max(cur_v, float(profile["speed_mean_mps"])),
                    profile=profile,
                    speed_limit_mps=speed_limit_mps,
                    allow_cruise=False,
                )
            profile["target_speed_mps"] = seeded_target

        if "last_speed_change_t" not in profile:
            profile["last_speed_change_t"] = float(getattr(veh, "last_speed_change_t", -1e9) or -1e9)
        if "next_speed_change_t" not in profile:
            seeded_next = float(getattr(veh, "next_speed_change_t", 0.0) or 0.0)
            if seeded_next <= self.t:
                seeded_next = self.t + self._sample_speed_change_interval(profile)
            profile["next_speed_change_t"] = seeded_next

        # Mirror profile back to the Vehicle for GUI display and CSV export.
        veh.speed_min_mps = float(profile["speed_min_mps"])
        veh.speed_max_mps = float(profile["speed_max_mps"])
        veh.speed_mean_mps = float(profile["speed_mean_mps"])
        veh.speed_std_mps = float(profile["speed_std_mps"])
        veh.target_speed_mps = float(profile["target_speed_mps"])
        veh.speed_change_interval_mean_s = float(profile["speed_change_interval_mean_s"])
        veh.speed_change_interval_min_s = float(profile["speed_change_interval_min_s"])
        veh.speed_change_interval_max_s = float(profile["speed_change_interval_max_s"])
        veh.next_speed_change_t = float(profile["next_speed_change_t"])
        veh.last_speed_change_t = float(profile["last_speed_change_t"])
        veh.cruise_hold_probability = float(profile["cruise_hold_probability"])
        veh.accel_response_s = float(profile["accel_response_s"])
        veh.max_accel_mps2 = float(profile["max_accel_mps2"])
        veh.max_decel_mps2 = float(profile["max_decel_mps2"])
        return profile

    def _update_vehicle_longitudinal_dynamics(
        self, veh: Vehicle, dt: float, speed_limit_mps: float
    ) -> float:
        """Advance speed towards the stochastic target; return the longitudinal acceleration.

        Anomaly overrides (each defaults to its no-op value, so the legacy
        path is byte-for-byte unchanged when no override is set):

        * ``veh.frozen``                — pin v to 0 and skip target tracking.
        * ``veh.a_cmd_override_mps2`` + ``a_cmd_override_until_t``
                                       — replace the target-tracking command
                                         with a scripted acceleration until the
                                         expiry time.  The clamp to
                                         ±max_decel/+max_accel is bypassed so
                                         hard-braking events (e.g. -7 m/s²) can
                                         actually reach the integrator.
        """
        # Frozen → vehicle is a static obstacle: hold v=0 and skip dynamics.
        if bool(getattr(veh, "frozen", False)):
            veh.v = 0.0
            veh.target_speed_mps = 0.0
            veh.a_long_mps2 = 0.0
            return 0.0
        profile = self._ensure_vehicle_dynamics_profile(veh, speed_limit_mps)
        cur_v = max(0.0, float(getattr(veh, "v", 0.0) or 0.0))

        if self.t >= float(profile.get("next_speed_change_t", 0.0) or 0.0):
            profile["target_speed_mps"] = self._sample_vehicle_target_speed(
                cur_v=cur_v,
                profile=profile,
                speed_limit_mps=speed_limit_mps,
                allow_cruise=True,
            )
            profile["last_speed_change_t"] = self.t
            profile["next_speed_change_t"] = self.t + self._sample_speed_change_interval(profile)

        speed_cap = max(
            speed_limit_mps * 1.15,
            float(profile.get("speed_max_mps", speed_limit_mps) or speed_limit_mps),
            4.0,
        )
        target_v = max(
            float(profile.get("speed_min_mps", 0.0) or 0.0),
            min(float(profile.get("target_speed_mps", 0.0) or 0.0), speed_cap),
        )
        dv = target_v - cur_v

        # First-order speed tracking.
        if abs(dv) < ACCEL_CMD_DEADBAND_MPS:
            a_cmd = 0.0
        else:
            a_cmd = dv / ACCEL_RESPONSE_TAU_S

        max_acc = max(0.2, float(profile.get("max_accel_mps2", DEFAULT_MAX_ACCEL_MPS2) or DEFAULT_MAX_ACCEL_MPS2))
        max_dec = max(0.2, float(profile.get("max_decel_mps2", DEFAULT_MAX_DECEL_MPS2) or DEFAULT_MAX_DECEL_MPS2))
        a_now = max(-max_dec, min(max_acc, a_cmd))

        # Deadband near target for visible cruise periods.
        if abs(dv) < CRUISE_DEADBAND_MPS:
            a_now = 0.0

        # Anomaly: scripted acceleration override.  When active this bypasses
        # the [-max_dec, max_acc] clamp so hard-braking events (-7 m/s², etc.)
        # actually reach the integrator.  Window expires automatically once
        # ``self.t >= a_cmd_override_until_t``.
        a_until = float(getattr(veh, "a_cmd_override_until_t", 0.0) or 0.0)
        if a_until > self.t:
            a_now = float(getattr(veh, "a_cmd_override_mps2", 0.0) or 0.0)

        # Mirror updated profile to Vehicle for GUI / export.
        veh.speed_min_mps = float(profile["speed_min_mps"])
        veh.speed_max_mps = float(profile["speed_max_mps"])
        veh.speed_mean_mps = float(profile["speed_mean_mps"])
        veh.speed_std_mps = float(profile["speed_std_mps"])
        veh.target_speed_mps = target_v
        veh.speed_change_interval_mean_s = float(profile["speed_change_interval_mean_s"])
        veh.speed_change_interval_min_s = float(profile["speed_change_interval_min_s"])
        veh.speed_change_interval_max_s = float(profile["speed_change_interval_max_s"])
        veh.next_speed_change_t = float(profile["next_speed_change_t"])
        veh.last_speed_change_t = float(profile["last_speed_change_t"])
        veh.cruise_hold_probability = 0.0
        veh.accel_response_s = ACCEL_RESPONSE_TAU_S
        veh.max_accel_mps2 = float(profile["max_accel_mps2"])
        veh.max_decel_mps2 = float(profile["max_decel_mps2"])
        return a_now

    def _pick_next_lane_for_vehicle(self, vehicle_id: str, current_lane_id: str) -> str:
        """Follow the planned route if one exists; otherwise use the random policy.

        The random fallback receives *vehicle_id* so it can apply the
        segment-visit cap and avoid routing loops.
        """
        nxt_list = self._route_next.get(vehicle_id)
        if nxt_list is not None and len(nxt_list) > 0:
            return nxt_list.pop(0)
        return self._pick_next_lane(current_lane_id, vehicle_id=vehicle_id)

    def _peek_next_lane_for_vehicle(self, vehicle_id: str, current_lane_id: str) -> str:
        """Return the next lane the vehicle would enter without consuming the route.

        This is a read-only sibling of :meth:`_pick_next_lane_for_vehicle` used
        by the segment-transition look-ahead for collision braking.  It does
        NOT apply the segment-visit cap — that decision is only made when the
        vehicle actually commits to a transition in :meth:`_pick_next_lane_for_vehicle`.
        """
        nxt_list = self._route_next.get(vehicle_id)
        if nxt_list is not None and len(nxt_list) > 0:
            return nxt_list[0]  # peek only — do NOT pop
        # No vehicle_id passed → visit cap not applied (look-ahead only).
        return self._pick_next_lane(current_lane_id)

    def _find_next_segment_leader(
        self, veh: "Vehicle"
    ) -> Optional[Tuple["Vehicle", float]]:
        """Look for a vehicle on the *next* lane the ego would enter.

        Called during the segment-transition look-ahead so the ego can begin
        braking before it actually crosses the lane boundary.  The effective
        gap accounts for:

        * The remaining distance on the ego's current lane
          (``L_ego - s_ego``).
        * The leader's arc-length position on the next lane (``s_leader``).
        * The physical vehicle length (``IDM_VEHICLE_LENGTH_M``) as a bumper
          offset.

        Returns
        -------
        ``(leader_vehicle, bumper_to_bumper_gap_m)`` or ``None``.
        """
        ego_lane = self.world.lanes.get(veh.lane_id)
        if ego_lane is None:
            return None

        L_ego = polyline_length(ego_lane.polyline)
        remaining = max(0.0, L_ego - veh.s)

        # Check boundary exit: if the current lane ends at a boundary-exit node
        # there is no next segment — no look-ahead needed.
        meta_ego = self._lane_meta.get(veh.lane_id, {})
        end_node = meta_ego.get("to", "")
        _boundary_exits = getattr(getattr(self, "world", None), "boundary_exit_nodes", set()) or set()
        if end_node in _boundary_exits:
            return None

        next_lane_id = self._peek_next_lane_for_vehicle(str(veh.id), veh.lane_id)
        if not next_lane_id or next_lane_id == veh.lane_id:
            return None  # dead-end or no outgoing lane

        best_gap: Optional[float] = None
        best_leader: Optional[Vehicle] = None

        for other in self.world.vehicles.values():
            if other is veh:
                continue
            if other.lane_id != next_lane_id:
                continue
            # Leader is ahead in the next lane, at arc-length s_leader.
            # Effective gap = remaining distance on current lane
            #               + leader's s on next lane
            #               - one vehicle length (bumper to bumper).
            gap = remaining + other.s - IDM_VEHICLE_LENGTH_M
            if gap < 0.0:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_leader = other

        if best_leader is None:
            return None
        return (best_leader, best_gap)

    # ------------------------------------------------------------------
    # IDM car-following model
    # ------------------------------------------------------------------

    def _find_leader(self, veh: Vehicle) -> Optional[Tuple["Vehicle", float]]:
        """Return the closest vehicle ahead on the same lane, and the gap to it.

        The search is restricted to vehicles sharing the same ``lane_id`` as
        *veh*.  Within a lane, longitudinal position is represented by ``s``
        (arc-length along the lane polyline), so the gap is simply::

            gap = s_leader - s_ego - IDM_VEHICLE_LENGTH_M

        This is valid because every vehicle on the same lane follows the same
        1-D coordinate axis.

        Known limitation — lane-to-lane continuity not handled
        --------------------------------------------------------
        Leader detection is restricted to vehicles that share the same
        ``lane_id`` as the ego.  A leader that has just transitioned to the
        *next* lane (while the ego is still near the end of the current lane)
        will not be detected, so the ego will briefly revert to free-driving
        behaviour for the remainder of that tick.  Extending detection across
        lane boundaries would require traversing the lane graph and projecting
        positions onto a continuous path coordinate — left as future work.

        Returns
        -------
        (leader, gap_m) or ``None`` if no vehicle is ahead.
        """
        best_gap: Optional[float] = None
        best_leader: Optional[Vehicle] = None

        for other in self.world.vehicles.values():
            if other is veh:
                continue
            if other.lane_id != veh.lane_id:
                continue
            gap = other.s - veh.s - IDM_VEHICLE_LENGTH_M
            if gap < 0.0:
                continue  # other vehicle is behind (or too close to count as ahead)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_leader = other

        if best_leader is None:
            return None
        return (best_leader, best_gap)

    def _idm_acceleration(
        self,
        v: float,
        v_target: float,
        gap: float,
        v_leader: float,
        *,
        s0: float = IDM_S0_M,
        T: float = IDM_T_S,
        a_max: float = IDM_A_MAX_MPS2,
        b: float = IDM_B_MPS2,
        delta: int = IDM_DELTA,
    ) -> float:
        """Compute the IDM longitudinal acceleration for a vehicle with a leader.

        All IDM parameters default to the module-level constants defined in
        ``constants.py`` but can be overridden per-call — making the model
        fully configurable without touching this function.

        Parameters
        ----------
        v        : ego speed (m/s)
        v_target : desired free-driving speed (m/s)
        gap      : bumper-to-bumper distance to the leader (m)
        v_leader : leader's current speed (m/s)
        s0       : minimum standstill gap (m)
        T        : desired time headway (s)
        a_max    : maximum free-driving acceleration (m/s²)
        b        : comfortable braking deceleration (m/s²)
        delta    : acceleration exponent (dimensionless)

        Returns
        -------
        Longitudinal acceleration in m/s² (negative = braking).
        """
        # Guard against numerical edge cases.
        v = max(0.0, v)
        v_target = max(0.1, v_target)
        gap = max(gap, 0.1)  # never divide by zero

        # Relative speed: positive means ego is faster → approaching the leader.
        delta_v = v - v_leader

        # Desired dynamic gap (IDM "interaction distance").
        s_star = s0 + v * T + (v * delta_v) / (2.0 * math.sqrt(a_max * b))
        s_star = max(s0, s_star)  # s_star must always be at least the standstill gap

        # Core IDM formula.
        a = a_max * (1.0 - (v / v_target) ** delta - (s_star / gap) ** 2)

        # Clamp to [-2b, a_max] — prevents extreme values during transients.
        a = max(-2.0 * b, min(a_max, a))

        # Emergency override: if the gap has collapsed below half s0, apply
        # maximum braking regardless of the formula output.
        if gap < 0.5 * s0:
            a = min(a, -2.0 * b)

        return a

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the simulation by *dt* seconds.

        Publishes the following event topics:
            * ``world.vehicle_state``      — position, speed, acceleration per vehicle.
            * ``world.vehicle_stuck``      — when a vehicle reaches a dead-end.
            * ``world.route_waypoint_reached`` — when a route waypoint is completed.
            * ``sensor.gps``               — GPS position measurement.
            * ``sensor.camera``            — Camera position detection.
            * ``sensor.das``               — DAS fiber measurement and trace.
        """
        self.t += dt

        # ---- Vehicle kinematics ----
        # Vehicles that exit via a boundary node are collected here and removed
        # from world.vehicles *after* the loop so we never mutate the dict
        # while iterating over it.
        _vids_to_remove: list = []

        for vid, veh in self.world.vehicles.items():
            if self._stuck.get(vid, False):
                continue

            lane = self.world.lanes.get(veh.lane_id)
            if lane is None:
                continue

            seg = self.world.segments.get(lane.segment_id)
            speed_limit_mps = (
                float(getattr(seg, "speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS) or DEFAULT_SPEED_LIMIT_MPS)
                if seg else DEFAULT_SPEED_LIMIT_MPS
            )
            a_now = self._update_vehicle_longitudinal_dynamics(veh, dt, speed_limit_mps)

            # --- IDM car-following override -----------------------------------
            # Check whether there is a vehicle directly ahead on the same lane.
            # If yes, the IDM acceleration replaces the free-driving value
            # whenever it is more restrictive (lower).
            #
            # To switch to *full* IDM control (IDM governs both free-driving
            # and following), replace the line below with simply:
            #     a_now = a_idm
            # For now we keep the min() blend: free-driving logic is unchanged
            # when the road is clear; IDM kicks in only to impose braking.
            #
            # Anomaly overrides honoured here:
            #   * ``veh.disable_idm``  — skip the blend entirely (used by
            #     scripted collision attackers and wrong-way drivers).
            #   * ``veh.min_gap_m``    — replace IDM's s0 (default 2 m) per-vehicle
            #     so tailgaters can settle below the normal standstill gap.
            #   * ``veh.time_headway_s`` — replace IDM's T (default 1.2 s) per-vehicle.
            leader_info = self._find_leader(veh)
            if leader_info is not None and not bool(getattr(veh, "disable_idm", False)):
                leader, gap = leader_info
                _s0 = float(getattr(veh, "min_gap_m", 0.0) or 0.0) or IDM_S0_M
                _T = float(getattr(veh, "time_headway_s", 0.0) or 0.0) or IDM_T_S
                a_idm = self._idm_acceleration(
                    v=max(0.0, float(getattr(veh, "v", 0.0) or 0.0)),
                    v_target=float(getattr(veh, "target_speed_mps", speed_limit_mps) or speed_limit_mps),
                    gap=gap,
                    v_leader=max(0.0, float(getattr(leader, "v", 0.0) or 0.0)),
                    s0=_s0,
                    T=_T,
                )
                a_now = min(a_now, a_idm)

            # ── Segment-transition look-ahead ─────────────────────────────────
            # When the ego is approaching the end of its current lane, check
            # whether there is a vehicle already occupying the next segment.  If
            # so, apply IDM braking as if that vehicle were a leader at the start
            # of the next lane — preventing a crash at the segment boundary.
            #
            # This mirrors the same-lane IDM block above but operates across the
            # lane boundary.  It is intentionally conservative (uses a generous
            # look-ahead horizon) because the cost of a false positive (mild
            # unnecessary braking) is far lower than a missed collision.
            #
            # Anomaly override: ``disable_idm`` suppresses this check too so
            # collision-scenario vehicles can deliberately run through the boundary.
            if not bool(getattr(veh, "disable_idm", False)):
                next_seg_leader_info = self._find_next_segment_leader(veh)
                if next_seg_leader_info is not None:
                    ns_leader, ns_gap = next_seg_leader_info
                    _s0 = float(getattr(veh, "min_gap_m", 0.0) or 0.0) or IDM_S0_M
                    _T = float(getattr(veh, "time_headway_s", 0.0) or 0.0) or IDM_T_S
                    a_idm_nxt = self._idm_acceleration(
                        v=max(0.0, float(getattr(veh, "v", 0.0) or 0.0)),
                        v_target=float(getattr(veh, "target_speed_mps", speed_limit_mps) or speed_limit_mps),
                        gap=ns_gap,
                        v_leader=max(0.0, float(getattr(ns_leader, "v", 0.0) or 0.0)),
                        s0=_s0,
                        T=_T,
                    )
                    a_now = min(a_now, a_idm_nxt)
            # ------------------------------------------------------------------

            veh.a_long_mps2 = a_now

            L = polyline_length(lane.polyline)
            v_prev = max(0.0, float(getattr(veh, "v", 0.0) or 0.0))
            # Step-level speed cap.  Anomaly: when ``ignore_speed_limit`` is set
            # we replace the segment-derived 1.12× cap with the vehicle's own
            # ``speed_max_mps`` envelope, allowing speeders to pull well past
            # the posted limit (still bounded so the integrator stays sane).
            if bool(getattr(veh, "ignore_speed_limit", False)):
                _veh_cap = max(4.0, float(getattr(veh, "speed_max_mps", 0.0) or speed_limit_mps * 1.12))
                v_next = max(0.0, min(v_prev + a_now * dt, _veh_cap))
            else:
                v_next = max(0.0, min(v_prev + a_now * dt, max(speed_limit_mps * 1.12, 4.0)))

            # ---- Numerical safety guard 1: velocity clamp ----
            # NOT part of the IDM model.  This is a discrete-time correction
            # that prevents integration error from accumulating into an overlap.
            # IDM computes a physically correct acceleration, but because
            # vehicles are updated sequentially within the loop, a follower
            # processed after its leader can still advance into the exclusion
            # zone if the timestep is large relative to the gap.
            # The clamp constrains v_next so that the trapezoid advance cannot
            # exceed the gap that remains after the leader has already moved:
            #     0.5 * (v_prev + v_next) * dt <= gap
            #     => v_next <= 2 * gap / dt - v_prev
            # In normal operating conditions (vehicles not spawned on top of
            # each other) IDM brakes long before this clamp becomes active.
            #
            # Anomaly: when ``allow_collision`` is set this clamp is skipped
            # so the follower can actually drive *into* the leader.  The
            # collision is detected and reported below.
            if leader_info is not None and not bool(getattr(veh, "allow_collision", False)):
                _gap = max(0.0, leader_info[1])
                v_safe_max = max(0.0, 2.0 * _gap / dt - v_prev)
                v_next = min(v_next, v_safe_max)

            veh.s += 0.5 * (v_prev + v_next) * dt
            veh.v = v_next

            # ---- Numerical safety guard 2: position clamp ----
            # NOT part of the IDM model.  Last-resort backstop for the one
            # remaining edge case that guard 1 cannot cover: when the gap is so
            # small that even v_next = 0 leaves the follower with enough
            # carry-over momentum (0.5 * v_prev * dt) to overshoot.  This only
            # triggers under pathological initial conditions (e.g. two vehicles
            # spawned closer together than IDM_VEHICLE_LENGTH_M) that are
            # physically unreachable during a running simulation.
            #
            # Anomaly: when ``allow_collision`` is set, the clamp is skipped
            # and we instead emit a one-shot ``world.collision`` event the
            # first time the bumper-to-bumper gap turns negative, then freeze
            # both vehicles so the post-impact jam develops naturally (and so
            # the same collision is not re-reported on subsequent ticks).
            if leader_info is not None:
                leader_obj = leader_info[0]
                _max_s = leader_obj.s - IDM_VEHICLE_LENGTH_M
                if bool(getattr(veh, "allow_collision", False)):
                    if veh.s > _max_s:
                        pair_key = frozenset({str(vid), str(leader_obj.id)})
                        if pair_key not in self._collided_pairs:
                            self._collided_pairs.add(pair_key)
                            # World-frame impact location: midway between bumpers.
                            _x_l, _y_l = point_at_s(lane.polyline, leader_obj.s)
                            _x_f, _y_f = point_at_s(lane.polyline, veh.s)
                            self.bus.publish(
                                "world.collision",
                                {
                                    "t": self.t,
                                    "vehicle_ids": [str(vid), str(leader_obj.id)],
                                    "follower_id": str(vid),
                                    "leader_id": str(leader_obj.id),
                                    "lane_id": veh.lane_id,
                                    "segment_id": lane.segment_id,
                                    "x": 0.5 * (_x_l + _x_f),
                                    "y": 0.5 * (_y_l + _y_f),
                                    "rel_speed_mps": max(
                                        0.0,
                                        float(getattr(veh, "v", 0.0) or 0.0)
                                        - float(getattr(leader_obj, "v", 0.0) or 0.0),
                                    ),
                                },
                            )
                            # Freeze both: post-impact, neither moves.
                            veh.frozen = True
                            leader_obj.frozen = True
                            veh.v = 0.0
                            leader_obj.v = 0.0
                            veh.target_speed_mps = 0.0
                            leader_obj.target_speed_mps = 0.0
                            # Park the follower exactly at the collision point.
                            veh.s = _max_s
                else:
                    if veh.s > _max_s:
                        veh.s = _max_s
                        veh.v = min(veh.v, max(0.0, float(getattr(leader_obj, "v", 0.0) or 0.0)))

            # Check for point-destination arrival.
            dest_lane = self._route_dest_lane.get(vid)
            if dest_lane and veh.lane_id == dest_lane:
                dest_s = float(self._route_dest_s.get(vid, 0.0))
                if veh.s >= dest_s:
                    veh.s = dest_s
                    self.bus.publish(
                        "world.route_waypoint_reached",
                        {"t": self.t, "vehicle_id": vid, "lane_id": dest_lane, "s": dest_s},
                    )
                    q = self._route_queue.get(vid)
                    route_finished = False
                    if q:
                        q.pop(0)
                        if q:
                            nxt_lane, nxt_s = q[0]
                            if not self.set_route_to_point(vid, nxt_lane, nxt_s):
                                self._route_queue.pop(vid, None)
                                self.clear_vehicle_route(vid)
                                route_finished = True
                        else:
                            self._route_queue.pop(vid, None)
                            self.clear_vehicle_route(vid)
                            route_finished = True
                    else:
                        self.clear_vehicle_route(vid)
                        route_finished = True

                    # Anomaly/visualization: emit ``world.route_complete`` exactly
                    # once when the last waypoint of the planned route is reached.
                    # Distinct from ``world.vehicle_stuck`` (dead-end fallback)
                    # so the GUI can render the blue-X "completed normally"
                    # marker only for cleanly finished routes.
                    if route_finished and not self._route_completed.get(vid, False):
                        self._route_completed[vid] = True
                        _xc, _yc = point_at_s(lane.polyline, dest_s)
                        self.bus.publish(
                            "world.route_complete",
                            {
                                "t": self.t,
                                "vehicle_id": vid,
                                "lane_id": dest_lane,
                                "segment_id": lane.segment_id,
                                "s": dest_s,
                                "x": _xc,
                                "y": _yc,
                            },
                        )

            # Check for end-of-lane.
            if L > 1e-6 and veh.s >= L:
                overflow = veh.s - L

                # If a node-destination has been reached, clear it.
                dest = self._route_dest_node.get(vid)
                meta0 = self._lane_meta.get(veh.lane_id, {})
                if dest and meta0.get("to") == dest and len(self._route_next.get(vid, [])) == 0:
                    self.clear_route(vid)

                nxt = self._pick_next_lane_for_vehicle(vid, veh.lane_id)

                # ── Loop-cap clean exit ───────────────────────────────────
                # _pick_next_lane returned the sentinel because every reachable
                # segment has been visited twice.  Remove the vehicle cleanly —
                # no "stuck" event, no further sensor detections — exactly as
                # if it had reached a boundary exit.
                if nxt == _LOOP_CAP_EXIT:
                    veh.s = L
                    _lc_x, _lc_y = point_at_s(lane.polyline, L)
                    if not self._route_completed.get(vid, False):
                        self._route_completed[vid] = True
                        self.bus.publish(
                            "world.route_complete",
                            {
                                "t": self.t,
                                "vehicle_id": vid,
                                "segment_id": lane.segment_id,
                                "x": _lc_x,
                                "y": _lc_y,
                                "reason": "loop_cap_exit",
                            },
                        )
                    _vids_to_remove.append(vid)
                    continue  # skip boundary-exit / stuck checks for this vehicle

                # ── Boundary-exit pre-check ───────────────────────────────
                # A declared boundary node must fire a clean exit even when
                # _pick_next_lane_for_vehicle() found a lane to follow (e.g.
                # the reverse segment on a two-way dead-end arm).  Without
                # this check, vehicles arriving at a bidirectional turnaround
                # take the reverse segment back into the network instead of
                # exiting the simulation.  Forcing nxt="" here drops into the
                # existing boundary-exit logic below without duplicating it.
                if nxt != "":
                    _bnd_exits = (
                        getattr(getattr(self, "world", None), "boundary_exit_nodes", None)
                        or set()
                    )
                    _end_meta = self._lane_meta.get(veh.lane_id, {})
                    if _end_meta.get("to", "") in _bnd_exits:
                        nxt = ""  # treat as dead-end → boundary-exit path below

                if nxt == "":
                    # Dead-end.  Decide whether this is a genuine stuck event
                    # or a clean boundary exit (vehicle left the map tile).
                    veh.s = L
                    meta = self._lane_meta.get(veh.lane_id, {})
                    end_node = meta.get("to", "")
                    x, y = point_at_s(lane.polyline, L)

                    # Check if this node is a declared map-boundary exit point.
                    _boundary_exits = getattr(
                        getattr(self, "world", None), "boundary_exit_nodes", set()
                    ) or set()
                    _is_boundary_exit = end_node in _boundary_exits

                    if _is_boundary_exit:
                        # Vehicle drove off the edge of the tile — treat as a
                        # completed trip, not an anomaly.  Do NOT set _stuck so
                        # the vehicle isn't marked with an X in the UI.
                        if not self._route_completed.get(vid, False):
                            self._route_completed[vid] = True
                            self.bus.publish(
                                "world.route_complete",
                                {
                                    "t": self.t,
                                    "vehicle_id": vid,
                                    "node_id": end_node,
                                    "segment_id": lane.segment_id,
                                    "x": x,
                                    "y": y,
                                    "reason": "boundary_exit",
                                },
                            )
                            # Schedule removal so followers can advance to the
                            # exit node without being blocked.  Actual deletion
                            # happens after the kinematics loop to avoid
                            # mutating world.vehicles while iterating over it.
                            _vids_to_remove.append(vid)
                    else:
                        # Genuine dead-end inside the map — vehicle is stuck.
                        self._stuck[vid] = True
                        self.bus.publish(
                            "world.vehicle_stuck",
                            {
                                "t": self.t,
                                "vehicle_id": vid,
                                "node_id": end_node,
                                "segment_id": lane.segment_id,
                                "x": x,
                                "y": y,
                            },
                        )
                else:
                    veh.lane_id = nxt
                    veh.s = max(0.0, overflow)
                    # Record that this vehicle entered the new segment.
                    _new_seg = getattr(self.world.lanes.get(nxt), "segment_id", "")
                    if _new_seg:
                        _vc = self._seg_visit_count.setdefault(vid, {})
                        _vc[_new_seg] = _vc.get(_new_seg, 0) + 1
                    lane = self.world.lanes.get(veh.lane_id)
                    if lane is None:
                        continue
                    L = polyline_length(lane.polyline)

            # ---- Lateral motion ------------------------------------------------
            # Default ("ou") path matches the previous Ornstein–Uhlenbeck drift
            # exactly.  Anomaly modes (weave/straddle/ramp/drift/random_walk)
            # only kick in when ``lateral_mode`` is non-default; an optional
            # [start_t, until_t) window can wrap *any* non-"ou" mode so the
            # deviation is transient.
            seg = self.world.segments.get(lane.segment_id)
            lane_width = float(getattr(seg, "lane_width", DEFAULT_LANE_WIDTH_M) or DEFAULT_LANE_WIDTH_M)
            max_lat = 0.5 * lane_width

            mode = str(getattr(veh, "lateral_mode", "ou") or "ou").lower()
            t_start = float(getattr(veh, "lateral_window_start_t", 0.0) or 0.0)
            t_until = float(getattr(veh, "lateral_window_until_t", 0.0) or 0.0)
            window_active = (
                mode != "ou"
                and (t_until <= 0.0 or (t_start <= self.t < t_until))
            )

            # Frozen vehicles (obstacles, post-collision wrecks, stalled cars)
            # must not drift laterally — they are physically static.  Skip the
            # entire lateral update so ``lateral_offset_m`` stays at whatever
            # value the scenario author chose (defaults to 0).
            if bool(getattr(veh, "frozen", False)):
                pass
            elif mode == "ou" or not window_active:
                # Default OU drift — UNCHANGED from pre-anomaly behaviour.
                rho = max(0.0, min(0.999, math.exp(-dt / LATERAL_DRIFT_TAU_S)))
                sigma_lat = 0.5 * max_lat
                veh.lateral_offset_m = (
                    rho * float(getattr(veh, "lateral_offset_m", 0.0))
                    + random.gauss(0.0, sigma_lat * math.sqrt(max(0.0, 1.0 - rho * rho)))
                )
                veh.lateral_offset_m = max(-max_lat, min(max_lat, veh.lateral_offset_m))

            elif mode == "straddle":
                # Pin the lateral offset at the configured fixed value (no clamp
                # to ±½ lane: a straddle by definition rides the lane edge).
                veh.lateral_offset_m = float(getattr(veh, "lateral_fixed_offset_m", 0.0) or 0.0)

            elif mode == "weave":
                # Sinusoidal lateral oscillation.  Clamp relaxed to ±lane_width
                # so amplitudes >½ lane are visible (drunk-driver behaviour).
                amp = float(getattr(veh, "lateral_weave_amplitude_m", 0.0) or 0.0)
                period = max(1e-3, float(getattr(veh, "lateral_weave_period_s", 0.0) or 0.0))
                phase = float(getattr(veh, "lateral_weave_phase_rad", 0.0) or 0.0)
                # Phase the sine on the *windowed* time so weaves with a window
                # start cleanly at zero crossing rather than mid-cycle.
                t_local = self.t - (t_start if t_until > 0.0 else 0.0)
                offset = amp * math.sin(2.0 * math.pi * t_local / period + phase)
                veh.lateral_offset_m = max(-lane_width, min(lane_width, offset))

            elif mode == "ramp":
                # Sharp instantaneous deviation to the fixed offset.  When a
                # window is active the offset jumps back to 0 outside it (this
                # is handled by the `window_active` check above).
                veh.lateral_offset_m = float(getattr(veh, "lateral_fixed_offset_m", 0.0) or 0.0)

            elif mode == "drift":
                # Slow linear drift; integrates `lateral_drift_rate_mps` while
                # active.  Clamp at ±lane_width so the value stays bounded.
                rate = float(getattr(veh, "lateral_drift_rate_mps", 0.0) or 0.0)
                cur = float(getattr(veh, "lateral_offset_m", 0.0) or 0.0)
                veh.lateral_offset_m = max(-lane_width, min(lane_width, cur + rate * dt))

            elif mode == "random_walk":
                # OU-style smooth random perturbation with user-specified sigma.
                # Seeded per-vehicle when ``lateral_random_seed != 0`` so the
                # trajectory is reproducible.
                sigma = float(getattr(veh, "lateral_random_sigma_m", 0.0) or 0.0) or (0.5 * max_lat)
                seed = int(getattr(veh, "lateral_random_seed", 0) or 0)
                rng = self._lateral_rng.get(str(vid))
                if rng is None and seed != 0:
                    rng = random.Random(seed)
                    self._lateral_rng[str(vid)] = rng
                rho = max(0.0, min(0.999, math.exp(-dt / LATERAL_DRIFT_TAU_S)))
                cur = float(getattr(veh, "lateral_offset_m", 0.0) or 0.0)
                noise_sigma = sigma * math.sqrt(max(0.0, 1.0 - rho * rho))
                noise = (rng.gauss(0.0, noise_sigma) if rng is not None
                         else random.gauss(0.0, noise_sigma))
                veh.lateral_offset_m = max(-lane_width, min(lane_width, rho * cur + noise))

            else:
                # Unknown mode → fall back to OU so misconfigured scenes don't
                # break the simulation.
                rho = max(0.0, min(0.999, math.exp(-dt / LATERAL_DRIFT_TAU_S)))
                sigma_lat = 0.5 * max_lat
                veh.lateral_offset_m = (
                    rho * float(getattr(veh, "lateral_offset_m", 0.0))
                    + random.gauss(0.0, sigma_lat * math.sqrt(max(0.0, 1.0 - rho * rho)))
                )
                veh.lateral_offset_m = max(-max_lat, min(max_lat, veh.lateral_offset_m))

            x, y = _pose_with_lateral(lane.polyline, min(veh.s, max(0.0, L)), float(getattr(veh, "lateral_offset_m", 0.0)))
            heading = _tangent_heading(lane.polyline, veh.s)
            veh.heading_rad = heading
            veh.ax_world_mps2 = float(getattr(veh, "a_long_mps2", 0.0) or 0.0) * math.cos(heading)
            veh.ay_world_mps2 = float(getattr(veh, "a_long_mps2", 0.0) or 0.0) * math.sin(heading)

            self.bus.publish(
                "world.vehicle_state",
                {
                    "t": self.t,
                    "vehicle_id": vid,
                    "lane_id": veh.lane_id,
                    "segment_id": lane.segment_id,
                    "x": x,
                    "y": y,
                    "v": veh.v,
                    "heading_rad": heading,
                    "a_long_mps2": getattr(veh, "a_long_mps2", 0.0),
                    "ax_world_mps2": getattr(veh, "ax_world_mps2", 0.0),
                    "ay_world_mps2": getattr(veh, "ay_world_mps2", 0.0),
                },
            )

        # ---- Remove boundary-exited vehicles ----
        # Done here, outside the kinematics loop, so dict mutation is safe.
        for _vid_rm in _vids_to_remove:
            self.world.vehicles.pop(_vid_rm, None)
            # Clean up per-vehicle routing state so stale entries don't linger.
            self._route_completed.pop(_vid_rm, None)
            self._route_next.pop(_vid_rm, None)
            self._route_dest_node.pop(_vid_rm, None)
            self._route_dest_lane.pop(_vid_rm, None)
            self._route_dest_s.pop(_vid_rm, None)

        # ---- GPS sensors ----
        for sid, g in self.world.gps.items():
            if not self._should_emit(f"gps:{sid}", g.update_hz):
                continue
            for vid, veh in self.world.vehicles.items():
                lane = self.world.lanes.get(veh.lane_id)
                if lane is None:
                    continue
                x_true, y_true = _pose_with_lateral(
                    lane.polyline, veh.s, float(getattr(veh, "lateral_offset_m", 0.0))
                )
                dx = x_true - g.x
                dy = y_true - g.y
                r = math.hypot(dx, dy)
                if r > g.radius_m:
                    continue

                # Noise grows mildly with range.
                dist_ratio = r / max(1e-6, g.radius_m)
                sigma = max(0.20, g.sigma_m * (1.0 + 0.35 * dist_ratio ** 2))
                nx = random.gauss(0.0, sigma)
                ny = random.gauss(0.0, sigma)
                confidence = max(0.05, min(0.99, 1.0 / (1.0 + (sigma / max(1e-6, g.sigma_m)) ** 2)))

                self.bus.publish(
                    "sensor.gps",
                    {
                        "t": self.t,
                        "sensor_id": sid,
                        "sensor_type": "GPS",
                        "vehicle_id": vid,
                        "lane_id": veh.lane_id,
                        "segment_id": lane.segment_id,
                        "x": x_true + nx,
                        "y": y_true + ny,
                        "x_error": nx,
                        "y_error": ny,
                        "x_true": x_true,
                        "y_true": y_true,
                        "sigma_m": sigma,
                        "range_m": r,
                        "speed_mps": float(getattr(veh, "v", 0.0) or 0.0),
                        "heading_rad": float(getattr(veh, "heading_rad", 0.0) or 0.0),
                        "a_long_mps2": float(getattr(veh, "a_long_mps2", 0.0) or 0.0),
                        "ax_true": float(getattr(veh, "ax_world_mps2", 0.0) or 0.0),
                        "ay_true": float(getattr(veh, "ay_world_mps2", 0.0) or 0.0),
                        "confidence": confidence,
                    },
                )

        # ---- Camera sensors ----
        for sid, c in self.world.cameras.items():
            if not self._should_emit(f"cam:{sid}", c.update_hz):
                continue
            for vid, veh in self.world.vehicles.items():
                lane = self.world.lanes.get(veh.lane_id)
                if lane is None:
                    continue
                x_true, y_true = _pose_with_lateral(
                    lane.polyline, veh.s, float(getattr(veh, "lateral_offset_m", 0.0))
                )
                dx = x_true - c.x
                dy = y_true - c.y
                r = math.hypot(dx, dy)
                if r > c.range_m:
                    continue

                ang = math.atan2(dy, dx)
                da = wrap_angle(ang - c.heading_rad)
                half_fov = math.radians(c.fov_deg) / 2.0
                if abs(da) > half_fov:
                    continue

                range_boost = max(0.0, 1.0 - r / max(1e-6, c.range_m))
                angle_boost = max(0.0, 1.0 - abs(da) / max(1e-6, half_fov))
                conf = max(0.02, min(0.98, 0.15 + 0.83 * (range_boost ** 1.35) * (0.55 + 0.45 * angle_boost)))

                if random.random() > conf:
                    continue

                # Camera uncertainty model per spec (SENSOR_ERROR_FORMULAS_HE.md):
                #   σ_cam = σ₀ + k_r · r + k_θ · |θ|
                # where r is range and θ (= da) is the off-axis viewing angle.
                # k_r is derived from the pinhole pixel-error model; k_θ accounts
                # for additional geometric distortion at the edges of the FOV.
                # σ₀ = 0.10 m is a small irreducible floor (quantisation + calibration).
                fpx = (CAMERA_PIXEL_WIDTH / 2.0) / max(1e-6, math.tan(math.radians(c.fov_deg) / 2.0))
                k_r = CAMERA_SIGMA_PX / max(1e-6, fpx)   # rad/pixel → m/m
                k_theta = 0.15                             # m per radian of off-axis angle
                sigma_0 = 0.10                             # irreducible floor [m]
                sigma = max(0.08, sigma_0 + k_r * r + k_theta * abs(da))
                nx = random.gauss(0.0, sigma)
                ny = random.gauss(0.0, sigma)

                self.bus.publish(
                    "sensor.camera",
                    {
                        "t": self.t,
                        "sensor_id": sid,
                        "sensor_type": "CAMERA",
                        "vehicle_id": vid,
                        "lane_id": veh.lane_id,
                        "segment_id": lane.segment_id,
                        "x": x_true + nx,
                        "y": y_true + ny,
                        "x_error": nx,
                        "y_error": ny,
                        "x_true": x_true,
                        "y_true": y_true,
                        "sigma_m": sigma,
                        "range_m": r,
                        "speed_mps": float(getattr(veh, "v", 0.0) or 0.0),
                        "heading_rad": float(getattr(veh, "heading_rad", 0.0) or 0.0),
                        "a_long_mps2": float(getattr(veh, "a_long_mps2", 0.0) or 0.0),
                        "ax_true": float(getattr(veh, "ax_world_mps2", 0.0) or 0.0),
                        "ay_true": float(getattr(veh, "ay_world_mps2", 0.0) or 0.0),
                        "confidence": conf,
                    },
                )

        # ---- DAS sensors ----
        # DAS is modelled as a 1-D fiber measurement along the road.  Each
        # vehicle generates a Gaussian amplitude peak in the trace whose
        # channel index is determined by the vehicle's arc-length projected
        # onto the fiber polyline.
        for sid, d in self.world.das.items():
            if not self._should_emit(f"das:{sid}", d.update_hz):
                continue

            n = max(1, d.channel_end - d.channel_start + 1)
            ideal_trace = [0.0] * n
            seg = self.world.segments.get(d.segment_id)
            per_vehicle = []

            if seg:
                n0 = self.world.nodes.get(seg.n0)
                n1 = self.world.nodes.get(seg.n1)
                base_poly = list(
                    getattr(seg, "points", []) or ([n0.p(), n1.p()] if (n0 and n1) else [])
                )
                fiber_poly = (
                    offset_polyline(base_poly, float(getattr(d, "fiber_offset_m", 2.0)))
                    if len(base_poly) >= 2 else []
                )
                fiber_len = max(1e-6, polyline_length(fiber_poly)) if fiber_poly else 1.0

                for vid, veh in self.world.vehicles.items():
                    lane = self.world.lanes.get(veh.lane_id)
                    if not lane or lane.segment_id != d.segment_id:
                        continue

                    x_true, y_true = _pose_with_lateral(
                        lane.polyline, veh.s, float(getattr(veh, "lateral_offset_m", 0.0))
                    )
                    if fiber_poly:
                        s_fiber, _, r = polyline_nearest_s(fiber_poly, (x_true, y_true))
                    else:
                        s_fiber, r = 0.0, 1e9

                    idx = int(max(0.0, min(n - 1, (s_fiber / fiber_len) * (n - 1))))
                    W = float(getattr(veh, "weight_kg", 1500.0) or 1500.0)
                    d0 = float(getattr(d, "d0_m", 0.7) or 0.7)
                    # Amplitude follows the Flamant-Boussinesq quasi-static point-load
                    # model (Hen et al. 2023, Nissan & Nissan 2023, Traffic_Monitoring
                    # 2025).  The DAS measures strain (or strain-rate) whose amplitude
                    # is determined by vehicle weight F and distance r to the fiber:
                    #
                    #   A = W / (r + d0)²
                    #
                    # Velocity does NOT appear as a direct amplitude factor in any of
                    # the three reference articles.  The velocity information is encoded
                    # in the diagonal slope of the waterfall plot (changing r over time),
                    # not in the instantaneous amplitude.
                    A = W / ((float(r) + d0) ** 2)

                    per_vehicle.append({
                        "vid": vid,
                        "lane": lane,
                        "x_true": x_true,
                        "y_true": y_true,
                        "r": float(r),
                        "s_fiber": float(s_fiber),
                        "A": A,
                        "W": W,
                        "lat": float(getattr(veh, "lateral_offset_m", 0.0)),
                    })
                    # Spread amplitude into neighbouring channels (σ² = 9 channels).
                    for k in range(-8, 9):
                        j = idx + k
                        if 0 <= j < n:
                            ideal_trace[j] += math.exp(-(k * k) / 18.0) * A

            # Waterfall trace: keep existing noise_std-based additive noise so the
            # trace display is unchanged.  This is decoupled from the new SNR model.
            trace_sigma = max(1e-6, float(getattr(d, "noise_std", 30.0) or 30.0))
            trace = [v + random.gauss(0.0, trace_sigma) for v in ideal_trace]

            # --- New error-based DAS model ---
            # SNR threshold depends on traffic load level of the segment.
            #   light  → snr_th = 6.0
            #   medium → snr_th = 8.0   (also used as default when level is "none")
            #   heavy  → snr_th = 10.0
            _traffic_level = (getattr(seg, "traffic_level", "none") or "none").strip().lower() if seg else "none"
            _snr_th_map = {"light": 6.0, "medium": 8.0, "heavy": 10.0}
            snr_th = _snr_th_map.get(_traffic_level, 8.0)

            if seg:
                # Detection/association is assumed solved: one guaranteed measurement
                # per active vehicle, no missed detections, no peak merging, no
                # multi-vehicle ambiguity.  This block is a clean noise-only model.
                _k_das: float = 2.0      # σ_das = k / √SNR  (m)
                _sigma_lane: float = 0.2  # lane-position uncertainty (m)

                for item in per_vehicle:
                    # SNR = A / snr_th  (traffic-load-dependent threshold)
                    snr = item["A"] / snr_th if snr_th > 1e-12 else 0.0

                    # DAS position sigma: σ_das = max(0.3, k / √SNR)
                    sigma_das = max(0.3, _k_das / math.sqrt(max(1e-6, snr)))

                    # Reliability: SNR / (SNR + snr_th), clamped to [0, 1]
                    reliability = max(0.0, min(1.0, snr / (snr + snr_th) if snr_th > 1e-12 else 0.0))

                    # Fiber tangent at the true projected arc-length — informational only.
                    # Used for fiber_position_m (GUI trajectory plot) and fiber_angle_rad.
                    tx, ty = (
                        tangent_at_s(fiber_poly, item["s_fiber"]) if fiber_poly else (1.0, 0.0)
                    )
                    fiber_angle = math.atan2(ty, tx)

                    # --- DAS position: add Gaussian noise directly to true (x, y) ---
                    # Two independent terms:
                    #   e_das  ~ N(0, sigma_das²)  — SNR-derived measurement error
                    #   e_lane ~ N(0, sigma_lane²) — intra-lane position uncertainty
                    x_meas = item["x_true"] + random.gauss(0.0, sigma_das) + random.gauss(0.0, _sigma_lane)
                    y_meas = item["y_true"] + random.gauss(0.0, sigma_das) + random.gauss(0.0, _sigma_lane)

                    # --- Velocity: 2D finite-difference from consecutive DAS (x, y) ---
                    # vx = (x_meas[k] - x_meas[k-1]) / dt
                    # vy = (y_meas[k] - y_meas[k-1]) / dt
                    #
                    # Total position noise per axis: sigma_total = hypot(sigma_das, sigma_lane).
                    # Propagated velocity uncertainty (two independent noisy positions):
                    #   σ_v = √2 · sigma_total / dt
                    # At 30 Hz (dt≈0.033 s) this is large, which is physically correct —
                    # the Kalman filter will appropriately down-weight this estimate.
                    _sigma_total = math.hypot(sigma_das, _sigma_lane)
                    _prev_key = (sid, item["vid"])
                    _prev = self._das_prev_s.get(_prev_key)
                    if _prev is not None:
                        _t_prev, _x_prev, _y_prev = _prev
                        _dt_fd = max(1e-6, self.t - _t_prev)
                        z_vx = (x_meas - _x_prev) / _dt_fd
                        z_vy = (y_meas - _y_prev) / _dt_fd
                        vel_sigma = max(0.10, math.sqrt(2.0) * _sigma_total / _dt_fd)
                    else:
                        # First sample for this (sensor, vehicle) pair — no previous
                        # position to difference against.  Emit no velocity measurement.
                        z_vx = 0.0
                        z_vy = 0.0
                        vel_sigma = 0.0  # sigma_v == 0.0 signals "not available" to Kalman
                    self._das_prev_s[_prev_key] = (self.t, x_meas, y_meas)

                    # speed and heading derived from the 2D DAS velocity vector.
                    speed_mps = math.hypot(z_vx, z_vy) if vel_sigma > 0.0 else 0.0
                    heading_meas = math.atan2(z_vy, z_vx) if vel_sigma > 0.0 else fiber_angle

                    self.bus.publish(
                        "sensor.das",
                        {
                            "t": self.t,
                            "sensor_id": sid,
                            "sensor_type": "DAS",
                            "vehicle_id": item["vid"],
                            "segment_id": d.segment_id,
                            "x": x_meas,
                            "y": y_meas,
                            "x_true": item["x_true"],
                            "y_true": item["y_true"],
                            "speed_mps": speed_mps,
                            "heading_rad": heading_meas,
                            "z_vx": z_vx,
                            "z_vy": z_vy,
                            "sigma_m": sigma_das,
                            "sigma_v": vel_sigma,
                            "confidence": reliability,
                            "fiber_distance_m": item["r"],
                            "fiber_position_m": item["s_fiber"],
                            "fiber_angle_rad": fiber_angle,
                            "lateral_offset_m": item["lat"],
                            "vehicle_weight_kg": item["W"],
                            "das_amplitude": item["A"],
                            "snr": snr,
                            "snr_th": snr_th,
                            "sigma_das": sigma_das,
                            "reliability": reliability,
                            "detected": True,
                            "channel_start": d.channel_start,
                            "channel_end": d.channel_end,
                            "trace": trace,
                        },
                    )

        # Maintain traffic density targets after emitting sensor events.
        self._enforce_segment_traffic()

    # ------------------------------------------------------------------
    # Traffic density control
    # ------------------------------------------------------------------

    def _enforce_segment_traffic(self) -> None:
        """Spawn or despawn vehicles to maintain per-segment traffic annotations.

        This is a macro density-control layer, not a car-following model.
        Segments with ``traffic_level == "none"`` (the default) are ignored.
        """
        if not self._traffic_enabled:
            return

        seg_lanes: Dict[str, List[str]] = {}
        for lid, lane in self.world.lanes.items():
            seg_id = getattr(lane, "segment_id", "")
            if seg_id:
                seg_lanes.setdefault(str(seg_id), []).append(str(lid))

        seg_vids: Dict[str, List[str]] = {}
        for vid, v in self.world.vehicles.items():
            lane = self.world.lanes.get(v.lane_id)
            if not lane:
                continue
            seg_id = getattr(lane, "segment_id", "")
            if seg_id:
                seg_vids.setdefault(str(seg_id), []).append(str(vid))

        for seg_id, seg in self.world.segments.items():
            level = (getattr(seg, "traffic_level", "none") or "none").strip().lower()
            if level == "none":
                continue
            lanes = seg_lanes.get(str(seg_id), [])
            if not lanes:
                continue

            density = float(self._traffic_density_vpkm_per_lane.get(level, 0.0))
            if density <= 0:
                continue

            rep_lane = self.world.lanes.get(lanes[0])
            seg_len_km = max(1e-6, (polyline_length(rep_lane.polyline) if rep_lane else 0.0) / 1000.0)
            target = int(round(density * seg_len_km * max(1, len(lanes))))
            if level == "heavy":
                target = max(1, target)

            cur = seg_vids.get(str(seg_id), [])
            if len(cur) < target:
                for _ in range(target - len(cur)):
                    lid = random.choice(lanes)
                    lane = self.world.lanes.get(lid)
                    if not lane:
                        continue
                    L = polyline_length(lane.polyline)
                    s = random.random() * max(1.0, 0.80 * L)
                    vlim = float(getattr(seg, "speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS) or DEFAULT_SPEED_LIMIT_MPS)
                    wkg, vv = sample_vehicle_profile(vlim)
                    base = len(self.world.vehicles) + 1
                    new_vid = f"veh{base:03d}"
                    j = 0
                    while new_vid in self.world.vehicles:
                        j += 1
                        new_vid = f"veh{base + j:03d}"
                    self.world.vehicles[new_vid] = Vehicle(
                        id=new_vid, lane_id=lid, s=float(s), v=float(vv), weight_kg=float(wkg)
                    )
            elif len(cur) > target:
                random.shuffle(cur)
                for vid in cur[: len(cur) - target]:
                    self.world.vehicles.pop(vid, None)

    # ------------------------------------------------------------------
    # Snapshot / restore  (for timeline scrubbing)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict:
        """Return a deep copy of the current simulation state for timeline replay."""
        return {
            "t": float(self.t),
            "world": copy.deepcopy(self.world),
            "lane_meta": copy.deepcopy(self._lane_meta),
            "route_queue": copy.deepcopy(self._route_queue),
            "route_dest_lane": copy.deepcopy(self._route_dest_lane),
            "route_dest_s": copy.deepcopy(self._route_dest_s),
            "route_dest_node": copy.deepcopy(self._route_dest_node),
            "route_next": copy.deepcopy(self._route_next),
            "stuck": copy.deepcopy(self._stuck),
            "rng_state": random.getstate(),
        }

    def restore(self, snap: Dict) -> None:
        """Restore a snapshot produced by :meth:`snapshot`."""
        self.t = float(snap.get("t", 0.0))
        self.world = snap.get("world", self.world)
        self._lane_meta = snap.get("lane_meta", self._lane_meta)
        self._route_queue = snap.get("route_queue", self._route_queue)
        self._route_dest_lane = snap.get("route_dest_lane", self._route_dest_lane)
        self._route_dest_s = snap.get("route_dest_s", self._route_dest_s)
        self._route_dest_node = snap.get("route_dest_node", self._route_dest_node)
        self._route_next = snap.get("route_next", self._route_next)
        self._stuck = snap.get("stuck", self._stuck)
        rs = snap.get("rng_state")
        if rs is not None:
            random.setstate(rs)

    # ------------------------------------------------------------------
    # Serialisable get/set_state  (lighter than snapshot; no deep copies)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a JSON-serialisable snapshot of the simulation state."""
        return {
            "t": self.t,
            "vehicles": {
                vid: {
                    "lane_id": v.lane_id,
                    "s": v.s,
                    "v": v.v,
                    "weight_kg": v.weight_kg,
                }
                for vid, v in self.world.vehicles.items()
            },
            "stuck": dict(self._stuck),
            "route_next": {k: list(v) for k, v in self._route_next.items()},
            "route_dest_node": dict(self._route_dest_node),
            "route_dest_lane": dict(self._route_dest_lane),
            "route_dest_s": dict(self._route_dest_s),
            "route_queue": {k: list(v) for k, v in self._route_queue.items()},
        }

    def set_state(self, st: dict) -> None:
        """Restore a state snapshot created by :meth:`get_state`.

        Assumes the lane topology has already been built via ``rebuild_lanes()``.
        """
        self.t = float(st.get("t", 0.0))
        for vid, vv in st.get("vehicles", {}).items():
            if vid in self.world.vehicles:
                v = self.world.vehicles[vid]
                v.lane_id = vv.get("lane_id", v.lane_id)
                v.s = float(vv.get("s", v.s))
                v.v = float(vv.get("v", v.v))
                v.weight_kg = float(vv.get("weight_kg", v.weight_kg))
        for vid in list(self.world.vehicles.keys()):
            if vid not in st.get("vehicles", {}):
                self.world.vehicles.pop(vid, None)
        self._stuck = dict(st.get("stuck", {}))
        self._route_next = {k: list(v) for k, v in st.get("route_next", {}).items()}
        self._route_dest_node = dict(st.get("route_dest_node", {}))
        self._route_dest_lane = dict(st.get("route_dest_lane", {}))
        self._route_dest_s = {k: float(v) for k, v in st.get("route_dest_s", {}).items()}
        self._route_queue = {k: list(v) for k, v in st.get("route_queue", {}).items()}
