"""Tests for Phase 1 per-vehicle anomaly overrides.

Each test enables exactly one override on a minimal straight-road world and
verifies that the simulator now produces an anomaly-compatible behaviour that
is impossible without the override.

Covered overrides (see Vehicle dataclass in ``models.py``):
    * ``a_cmd_override_mps2`` + ``a_cmd_override_until_t``  → sudden braking
    * ``min_gap_m`` + ``time_headway_s``                    → tailgating (<5 m)
    * ``frozen``                                            → stalled vehicle
    * ``ignore_speed_limit``                                → speeding
    * ``lateral_mode = "weave"``                            → lateral oscillation
    * ``lateral_mode = "straddle"``                         → fixed lane-edge offset
    * ``allow_collision``                                   → true overlap + world.collision
    * baseline (no overrides)                               → pre-existing behaviour

These tests guard against regressions in both directions:
    1. An override stops working (new behaviour silently broken).
    2. An override leaks into the default path (legacy scenarios drift).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simstudio.bus import EventBus
from simstudio.constants import IDM_VEHICLE_LENGTH_M
from simstudio.models import Node, Segment, Vehicle, World
from simstudio.sim_core import Simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_straight_world() -> World:
    """Single 500 m one-way segment with 1 lane; no auto-spawn."""
    w = World(auto_spawn=False)
    w.nodes["A"] = Node("A", 0.0, 0.0)
    w.nodes["B"] = Node("B", 500.0, 0.0)
    w.segments["S"] = Segment(
        "S", "A", "B", lanes=1, lane_width=3.6, speed_limit_mps=13.9
    )
    return w


def _build(vehicle_specs):
    """Construct (sim, world) with the given [(vid, s, overrides_dict), ...]."""
    world = _make_straight_world()
    sim = Simulation(EventBus(), world)
    sim.rebuild_lanes()
    lane_id = next(iter(world.lanes.keys()))
    for vid, s0, overrides in vehicle_specs:
        veh = Vehicle(
            id=vid,
            lane_id=lane_id,
            s=s0,
            v=13.9,
            weight_kg=1500.0,
            speed_min_mps=13.9,
            speed_max_mps=13.9,
            target_speed_mps=13.9,
            speed_mean_mps=13.9,
            speed_std_mps=0.0,
        )
        for k, v in overrides.items():
            setattr(veh, k, v)
        world.vehicles[vid] = veh
    return sim, world


def _step_for(sim, seconds: float, dt: float = 0.05) -> None:
    for _ in range(int(seconds / dt)):
        sim.step(dt)


# ---------------------------------------------------------------------------
# Longitudinal anomalies
# ---------------------------------------------------------------------------


def test_sudden_braking_override_bypasses_max_decel_clamp():
    """a_cmd_override of -7 m/s² must reach the integrator, ignoring
    max_decel_mps2 (default ~2.6) which would otherwise clip it."""
    sim, world = _build([("V1", 20.0, {})])
    v = world.vehicles["V1"]
    v.a_cmd_override_mps2 = -7.0
    v.a_cmd_override_until_t = 2.5
    _step_for(sim, 2.5)
    # At -7 m/s² from 13.9 m/s, v hits 0 in ≈ 2 s → must be ≤ ~1 m/s by 2.5 s.
    assert v.v < 1.0, f"expected near-zero v after hard brake, got {v.v:.3f}"


def test_tailgating_via_min_gap_and_time_headway():
    """Lowering IDM s0 to 0.5 m and T to 0.3 s must let the follower sit at
    <5 m gap without overlapping."""
    sim, world = _build(
        [
            ("L", 50.0, {}),
            ("F", 10.0, {"min_gap_m": 0.5, "time_headway_s": 0.3}),
        ]
    )
    leader = world.vehicles["L"]
    leader.target_speed_mps = 8.0
    leader.speed_min_mps = 8.0
    leader.speed_max_mps = 8.0
    leader.v = 8.0
    _step_for(sim, 30.0)
    gap = leader.s - world.vehicles["F"].s - IDM_VEHICLE_LENGTH_M
    assert 0.0 < gap < 5.0, f"expected tailgate gap in (0, 5), got {gap:.2f}"


def test_frozen_vehicle_holds_zero_velocity():
    """frozen=True must pin v=0 and s at spawn, regardless of IDM / targets."""
    sim, world = _build([("V1", 20.0, {"frozen": True})])
    world.vehicles["V1"].v = 0.0
    _step_for(sim, 5.0)
    v = world.vehicles["V1"]
    assert v.v == 0.0
    assert abs(v.s - 20.0) < 1e-6


def test_ignore_speed_limit_allows_exceeding_cap():
    """With ignore_speed_limit=True, the step-level 1.12× speed_limit clamp
    must be bypassed so the vehicle can reach its own speed_max_mps."""
    sim, world = _build([("V1", 20.0, {"ignore_speed_limit": True})])
    v = world.vehicles["V1"]
    v.speed_max_mps = 30.0
    v.target_speed_mps = 30.0
    v.speed_min_mps = 28.0
    v.speed_mean_mps = 30.0
    _step_for(sim, 40.0)
    # Speed limit is 13.9 → 1.12× = 15.57. Must far exceed that.
    assert v.v > 20.0, f"expected v > 20 m/s, got {v.v:.2f}"


# ---------------------------------------------------------------------------
# Lateral anomalies
# ---------------------------------------------------------------------------


def test_weave_mode_crosses_lane_marking():
    """Weave with amplitude > ½ lane_width must produce lateral excursions
    beyond the default ±½ lane clamp (i.e. visibly cross the lane line)."""
    sim, world = _build(
        [
            (
                "V1",
                20.0,
                {
                    "lateral_mode": "weave",
                    "lateral_weave_amplitude_m": 2.5,
                    "lateral_weave_period_s": 4.0,
                },
            )
        ]
    )
    v = world.vehicles["V1"]
    offsets = []
    for _ in range(80):  # 4 s
        sim.step(0.05)
        offsets.append(v.lateral_offset_m)
    assert max(offsets) > 1.8, f"max lateral offset {max(offsets):.2f} did not cross ½ lane"
    assert min(offsets) < -1.8, f"min lateral offset {min(offsets):.2f} did not cross ½ lane"


def test_straddle_mode_pins_offset():
    """Straddle mode holds lateral_offset_m at the configured fixed value."""
    sim, world = _build(
        [("V1", 20.0, {"lateral_mode": "straddle", "lateral_fixed_offset_m": 1.8})]
    )
    _step_for(sim, 2.0)
    assert abs(world.vehicles["V1"].lateral_offset_m - 1.8) < 1e-6


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------


def test_allow_collision_emits_event_and_freezes_both():
    """A vehicle flagged allow_collision that drives into a frozen leader
    must emit exactly one world.collision event and both vehicles must
    become frozen so post-impact jams develop naturally."""
    sim, world = _build(
        [
            ("L", 30.0, {}),
            ("F", 10.0, {"allow_collision": True, "disable_idm": True}),
        ]
    )
    leader = world.vehicles["L"]
    follower = world.vehicles["F"]
    leader.target_speed_mps = 0.0
    leader.v = 0.0
    leader.frozen = True
    follower.target_speed_mps = 13.9
    follower.v = 13.9

    collisions = []
    sim.bus.subscribe("world.collision", lambda ev: collisions.append(ev))
    _step_for(sim, 5.0)

    assert len(collisions) == 1, f"expected 1 collision event, got {len(collisions)}"
    assert follower.frozen, "follower must be frozen post-impact"
    assert leader.frozen, "leader must remain frozen post-impact"


# ---------------------------------------------------------------------------
# Baseline (no overrides) — must match pre-Phase-1 behaviour
# ---------------------------------------------------------------------------


def test_baseline_idm_still_prevents_overlap():
    """With no overrides, default IDM + numerical guards must keep the
    follower at gap ≥ s0 (2 m) while still closing toward a slower leader."""
    sim, world = _build([("L", 40.0, {}), ("F", 10.0, {})])
    leader = world.vehicles["L"]
    leader.target_speed_mps = 5.0
    leader.v = 5.0
    leader.speed_min_mps = 5.0
    leader.speed_max_mps = 5.0
    _step_for(sim, 30.0)
    gap = leader.s - world.vehicles["F"].s - IDM_VEHICLE_LENGTH_M
    assert gap >= 2.0, f"default IDM must hold gap ≥ s0=2, got {gap:.2f}"
    assert gap < 10.0, f"follower should pursue the leader, gap={gap:.2f}"


# ---------------------------------------------------------------------------
# Phase 2 — "obstacle in lane" anomaly via frozen typed Vehicle.
# ---------------------------------------------------------------------------
# A frozen Vehicle with weight_kg=80 and object_type="pedestrian" stands in
# for a pedestrian: IDM sees it as a v=0 leader, DAS produces a small-
# amplitude signature proportional to 80 (≈5 % of a car), and the simulator
# continues to run normally around it.  No new dataclass needed — the
# object_type string is the only distinguishing label for ground truth.


def test_obstacle_as_frozen_pedestrian_causes_follower_to_brake():
    sim, world = _build([])
    lane_id = next(iter(world.lanes.keys()))

    # Pedestrian at s=60, frozen (Phase 1), with pedestrian weight.
    ped = Vehicle(
        id="PED",
        lane_id=lane_id,
        s=60.0,
        v=0.0,
        weight_kg=80.0,
        speed_min_mps=0.0,
        speed_max_mps=0.0,
        target_speed_mps=0.0,
        speed_mean_mps=0.0,
        speed_std_mps=0.0,
        frozen=True,
        object_type="pedestrian",
    )
    world.vehicles["PED"] = ped

    # Approaching car at 13.9 m/s.
    car = Vehicle(
        id="CAR",
        lane_id=lane_id,
        s=10.0,
        v=13.9,
        weight_kg=1500.0,
        speed_min_mps=13.9,
        speed_max_mps=13.9,
        target_speed_mps=13.9,
        speed_mean_mps=13.9,
        speed_std_mps=0.0,
    )
    world.vehicles["CAR"] = car

    _step_for(sim, 10.0)

    gap = ped.s - car.s - IDM_VEHICLE_LENGTH_M
    assert gap >= 1.5, f"car must stop short of pedestrian; gap={gap:.2f}"
    assert car.v < 1.0, f"car must be stopped after 10 s; v={car.v:.3f}"
    assert ped.v == 0.0 and ped.s == 60.0, "pedestrian must remain frozen at spawn"
    assert ped.object_type == "pedestrian", "obstacle label must be preserved"


def test_object_type_defaults_to_vehicle_and_is_free_form():
    """object_type is a free-form label, defaults to 'vehicle', does not affect dynamics."""
    sim, world = _build([("V1", 20.0, {})])
    v = world.vehicles["V1"]
    assert v.object_type == "vehicle"
    # Arbitrary labels should be permitted (no enum enforcement).
    v.object_type = "animal"
    _step_for(sim, 1.0)
    # Dynamics unchanged — a 'vehicle' vs 'animal' label is purely semantic.
    assert v.v > 0.0
    assert v.object_type == "animal"
