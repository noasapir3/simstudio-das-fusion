from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math

Point = Tuple[float, float]

@dataclass
class Node:
    id: str
    x: float
    y: float
    def p(self) -> Point:
        return (self.x, self.y)

@dataclass
class Segment:
    id: str
    n0: str
    n1: str
    # Number of lanes in this direction (1..7). Two-way roads are represented
    # explicitly as two segments (A->B and B->A) if needed.
    lanes: int = 1
    lane_width: float = 3.6
    speed_limit_mps: float = 13.9
    # Optional polyline geometry (including endpoints) for curved roads.
    # If empty, geometry is inferred from node positions [n0, n1].
    points: List[Point] = field(default_factory=list)
    # Backward-compatible legacy field (ignored by the current engine).
    one_way: bool = True

    # Optional traffic annotation (used by the traffic-load randomizer/runtime).
    # Values: "none"|"light"|"medium"|"heavy".
    traffic_level: str = "none"

@dataclass
class LaneGeom:
    id: str
    segment_id: str
    offset_index: int
    polyline: List[Point]

@dataclass
class Vehicle:
    id: str
    lane_id: str
    s: float = 0.0
    # Speed along lane centerline (m/s)
    v: float = 10.0
    # Hidden longitudinal acceleration used by the simulator dynamics (m/s^2).
    a_long_mps2: float = 0.0
    # Approx mass used by sensor noise models and visualization (kg)
    weight_kg: float = 1500.0
    # Smooth lateral offset from lane centerline (m). Updated at runtime.
    lateral_offset_m: float = 0.0
    # Per-vehicle stochastic speed profile.
    speed_mean_mps: float = 0.0
    speed_std_mps: float = 0.0
    # Explicit macro controls for the feasible speed envelope of this vehicle.
    # If left as 0, the simulator seeds realistic values automatically.
    speed_min_mps: float = 0.0
    speed_max_mps: float = 0.0
    target_speed_mps: float = 0.0
    # Time between target-speed updates. Users may pin a min/max range; if left as 0
    # the simulator seeds realistic values automatically.
    speed_change_interval_mean_s: float = 0.0
    speed_change_interval_min_s: float = 0.0
    speed_change_interval_max_s: float = 0.0
    next_speed_change_t: float = 0.0
    last_speed_change_t: float = -1e9
    # Probability that the next update keeps the car in a near-constant-speed cruise window.
    cruise_hold_probability: float = 0.0
    accel_response_s: float = 0.0
    max_accel_mps2: float = 0.0
    max_decel_mps2: float = 0.0
    # Cached world-frame kinematics for export / fusion.
    heading_rad: float = 0.0
    ax_world_mps2: float = 0.0
    ay_world_mps2: float = 0.0

    # -----------------------------------------------------------------
    # Anomaly-enabling per-vehicle overrides (Phase 1).
    # All default values preserve pre-existing simulator behavior; they are
    # consumed by sim_core.Simulation when non-default.  Set at spawn time
    # via scenario JSON or mutated at runtime by the ScenarioDirector
    # (see scenario_script.py) so that a vehicle can *temporarily* deviate
    # from realistic dynamics for a bounded anomaly window.
    # -----------------------------------------------------------------

    #: If True, per-vehicle speed cap inside step() is bypassed (speeding anomaly).
    ignore_speed_limit: bool = False

    #: Per-vehicle IDM minimum standstill gap (m).  0 → use constants.IDM_S0_M.
    #: Lowering this enables tailgating (e.g. 0.5 allows ~sub-5 m following).
    min_gap_m: float = 0.0

    #: Per-vehicle IDM desired time headway (s).  0 → use constants.IDM_T_S.
    time_headway_s: float = 0.0

    #: If True, IDM car-following is skipped for this vehicle.  Useful for
    #: wrong-way driving, scripted tailgating where we want explicit control
    #: of the gap, and stalled/obstacle scenarios where the vehicle is frozen.
    disable_idm: bool = False

    #: If True, v is pinned to 0 (stalled-vehicle / obstacle anomaly).
    frozen: bool = False

    #: If True, numerical overlap guards in step() are disabled so this
    #: vehicle *can* collide with its leader.  The follower still runs IDM
    #: unless disable_idm is also set.
    allow_collision: bool = False

    #: Scripted acceleration override (m/s²).  When a_cmd_override_until_t > t
    #: the longitudinal dynamics bypass target-speed tracking and apply this
    #: acceleration directly (subject only to v>=0).  Enables panic braking
    #: (-7 m/s²) and aggressive acceleration windows.
    a_cmd_override_mps2: float = 0.0

    #: Absolute sim time (s) at which the acceleration override expires.
    #: 0 or negative = no active override.
    a_cmd_override_until_t: float = 0.0

    #: Lateral behaviour mode.  Supported values:
    #:   "ou"          — default Ornstein–Uhlenbeck drift (zero mean-reverting,
    #:                   clamped to ±½ lane width).  Pre-existing behaviour.
    #:   "weave"       — sinusoidal lateral oscillation with configurable
    #:                   amplitude + period; clamp relaxed to ±lane_width.
    #:   "straddle"    — lateral_offset_m held at lateral_fixed_offset_m.
    #:   "ramp"        — instant jump (sharp deviation) to lateral_fixed_offset_m.
    #:                   When combined with lateral_window_until_t the offset
    #:                   returns to 0 after the window — i.e. "leave + return".
    #:   "drift"       — slow linear drift at lateral_drift_rate_mps (m/s),
    #:                   clamped to ±lane_width.
    #:   "random_walk" — smooth random perturbation; OU process with the
    #:                   user-specified ``lateral_random_sigma_m``.  Seeded by
    #:                   ``lateral_random_seed`` for deterministic replay.
    #: All non-"ou" modes can be wrapped in a [start_t, until_t) interval via
    #: ``lateral_window_start_t`` / ``lateral_window_until_t``.
    lateral_mode: str = "ou"

    #: Used when lateral_mode == "straddle".  Sign convention matches
    #: lateral_offset_m: positive = left of travel direction (matches the
    #: right-hand rule used in _pose_with_lateral).
    lateral_fixed_offset_m: float = 0.0

    #: Used when lateral_mode == "weave".  Peak lateral displacement (m).
    lateral_weave_amplitude_m: float = 0.0

    #: Used when lateral_mode == "weave".  Oscillation period (s).
    lateral_weave_period_s: float = 0.0

    #: Phase offset for weave (rad).  Lets batch-generated runs decorrelate
    #: weave trajectories across seeds without parameter changes.
    lateral_weave_phase_rad: float = 0.0

    # -----------------------------------------------------------------
    # Phase 1b — additional lateral anomaly modes.
    # All defaults are no-op: pre-existing scenarios with `lateral_mode="ou"`
    # (or unset) are completely unaffected.
    # -----------------------------------------------------------------

    #: Optional time-window for ``lateral_mode``.  When ``lateral_window_until_t > 0``
    #: the lateral mode is only active while ``self.t in [start_t, until_t)``.
    #: Outside the window the vehicle reverts to the default OU drift, so the
    #: anomaly can be transient.  ``until_t = 0`` (default) means "always on".
    lateral_window_start_t: float = 0.0
    lateral_window_until_t: float = 0.0

    #: Used by ``lateral_mode == "drift"`` (slow lateral drift, m/s).
    #: Positive = drift to the left of travel direction.
    lateral_drift_rate_mps: float = 0.0

    #: Used by ``lateral_mode == "random_walk"``.  Process std-deviation (m).
    #: 0 → falls back to the same value as the default OU drift.
    lateral_random_sigma_m: float = 0.0

    #: Per-vehicle deterministic seed for ``lateral_mode == "random_walk"``.
    #: When 0 the simulator uses the global RNG (non-deterministic across
    #: parallel runs).  Any non-zero value yields a reproducible trajectory.
    lateral_random_seed: int = 0

    # -----------------------------------------------------------------
    # Phase 2 — object typing for "obstacle in lane" anomaly class.
    # -----------------------------------------------------------------
    # A frozen Vehicle with ``object_type="pedestrian"`` + appropriate
    # ``weight_kg`` reproduces a pedestrian's DAS signature (W/(r+d0)²)
    # correctly without needing a parallel sensor code path.  Future
    # renderers / label exporters can key off this string to distinguish
    # stalled cars from pedestrians / debris / animals.
    #
    # Recognised values (not enforced, free-form string):
    #   "vehicle"    — default car/bus/truck (weight_kg ≈ 1500 / 12000).
    #   "pedestrian" — human obstacle (weight_kg ≈ 80).
    #   "debris"     — fallen load / tyre etc. (weight_kg ≈ 200).
    #   "animal"     — deer/dog (weight_kg ≈ 30–300).
    #   "barrier"    — cone / traffic barrier (weight_kg small, effectively
    #                  invisible on DAS but visible on camera).
    object_type: str = "vehicle"

@dataclass
class GPSSensor:
    id: str
    x: float = 0.0
    y: float = 0.0
    sigma_m: float = 2.5
    update_hz: float = 5.0
    radius_m: float = 25.0

@dataclass
class CameraSensor:
    id: str
    x: float = 0.0
    y: float = 0.0
    heading_rad: float = 0.0
    fov_deg: float = 70.0
    range_m: float = 220.0
    update_hz: float = 10.0

@dataclass
class DASSensor:
    id: str
    segment_id: str = ""
    channel_start: int = 0
    channel_end: int = 1200
    update_hz: float = 30.0
    # DAS intrinsic noise floor in the same amplitude units as A = W / (r + d0)^2.
    # The noise_std is applied directly as the trace noise: trace_sigma = noise_std.
    # SNR is then: SNR = A / noise_std = (W / (r+d0)^2) / noise_std.
    # Kalman position uncertainty follows: sigma_DAS = k_das / sqrt(SNR).
    #
    # Calibration guide:
    #   noise_std = 30   → trace_sigma = 30  → SNR ≈ 7  for 1500 kg car at r=2 m  (standard)
    #   noise_std = 60   → trace_sigma = 60  → SNR ≈ 3.5 (noisy installation)
    #   noise_std = 15   → trace_sigma = 15  → SNR ≈ 14  (clean, buried fiber)
    noise_std: float = 30.0
    # Fiber is modeled as the lane polyline shifted sideways by this offset (m).
    fiber_offset_m: float = 2.0
    # Small distance floor to avoid singular amplitude at r=0 (m).
    d0_m: float = 0.7


@dataclass
class World:
    nodes: Dict[str, Node] = field(default_factory=dict)
    segments: Dict[str, Segment] = field(default_factory=dict)
    lanes: Dict[str, LaneGeom] = field(default_factory=dict)
    vehicles: Dict[str, Vehicle] = field(default_factory=dict)
    gps: Dict[str, GPSSensor] = field(default_factory=dict)
    cameras: Dict[str, CameraSensor] = field(default_factory=dict)
    das: Dict[str, DASSensor] = field(default_factory=dict)
    # Set to False in scenario JSON to disable automatic vehicle spawning.
    # When False, only the vehicles explicitly defined in the scenario are
    # present — essential for clean, repeatable validation scenarios.
    auto_spawn: bool = True
    # Node IDs at the physical boundary of the map tile — roads that continue
    # beyond the simulation area.  When a vehicle reaches one of these nodes
    # it fires ``world.route_complete`` (clean exit) instead of
    # ``world.vehicle_stuck`` (blocked/anomaly).  Define them in the .sim.json
    # as:  "boundary_exit_nodes": ["n123", "n456", ...]
    boundary_exit_nodes: "Set[str]" = field(default_factory=set)

def make_lane_polylines(n0: Point, n1: Point, lanes: int, lane_width: float):
    x0,y0 = n0
    x1,y1 = n1
    dx, dy = (x1-x0, y1-y0)
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return [[n0, n1] for _ in range(lanes)]
    nx, ny = (-dy/L, dx/L)
    mid = (lanes-1)/2.0
    offsets = [(i-mid)*lane_width for i in range(lanes)]
    polys = []
    for off in offsets:
        polys.append([(x0 + nx*off, y0 + ny*off), (x1 + nx*off, y1 + ny*off)])
    return polys
