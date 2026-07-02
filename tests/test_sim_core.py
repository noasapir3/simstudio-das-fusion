"""Tests for simstudio.sim_core — simulation engine behaviours."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simstudio.bus import EventBus
from simstudio.models import Node, Segment, Vehicle, World, GPSSensor, CameraSensor, DASSensor
from simstudio.sim_core import Simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_square_world() -> World:
    """A simple 4-node square road network (500 m sides, one-way clockwise)."""
    world = World()
    world.nodes["n1"] = Node(id="n1", x=0.0,   y=0.0)
    world.nodes["n2"] = Node(id="n2", x=500.0, y=0.0)
    world.nodes["n3"] = Node(id="n3", x=500.0, y=500.0)
    world.nodes["n4"] = Node(id="n4", x=0.0,   y=500.0)
    world.segments["s1"] = Segment(id="s1", n0="n1", n1="n2")
    world.segments["s2"] = Segment(id="s2", n0="n2", n1="n3")
    world.segments["s3"] = Segment(id="s3", n0="n3", n1="n4")
    world.segments["s4"] = Segment(id="s4", n0="n4", n1="n1")
    return world


def _make_sim(seed: int = 0) -> Simulation:
    random.seed(seed)
    world = _build_square_world()
    bus = EventBus()
    sim = Simulation(bus, world)
    sim.rebuild_lanes()
    return sim


# ---------------------------------------------------------------------------
# rebuild_lanes
# ---------------------------------------------------------------------------

def test_rebuild_lanes_creates_lanes():
    sim = _make_sim()
    assert len(sim.world.lanes) > 0


def test_rebuild_lanes_populates_meta():
    sim = _make_sim()
    assert len(sim._lane_meta) == len(sim.world.lanes)
    for lid, meta in sim._lane_meta.items():
        assert "from" in meta and "to" in meta


def test_rebuild_lanes_clears_stuck():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=5.0)
    sim._stuck["v0"] = True
    sim.rebuild_lanes()
    assert sim._stuck == {}


# ---------------------------------------------------------------------------
# Vehicle kinematics
# ---------------------------------------------------------------------------

def test_vehicle_advances_position():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=10.0)
    s_before = sim.world.vehicles["v0"].s
    sim.step(0.1)
    assert sim.world.vehicles["v0"].s > s_before


def test_vehicle_speed_non_negative():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=0.0)
    for _ in range(100):
        sim.step(0.1)
    assert sim.world.vehicles["v0"].v >= 0.0


def test_vehicle_changes_lane_at_junction():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    # Place vehicle right at end of lane so it must switch
    from simstudio.geometry import polyline_length
    L = polyline_length(sim.world.lanes[lid].polyline)
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=L - 0.1, v=20.0)
    for _ in range(10):
        sim.step(0.1)
    # Vehicle should have moved to a different lane
    assert sim.world.vehicles["v0"].lane_id != lid or sim.world.vehicles.get("v0") is None


def test_stuck_vehicle_does_not_move():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=10.0)
    sim._stuck["v0"] = True
    s_before = sim.world.vehicles["v0"].s
    sim.step(1.0)
    assert sim.world.vehicles["v0"].s == s_before


# ---------------------------------------------------------------------------
# Event publishing
# ---------------------------------------------------------------------------

def test_vehicle_state_events_emitted():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=10.0)
    events = []
    sim.bus.subscribe("world.vehicle_state", events.append)
    sim.step(0.1)
    assert len(events) == 1
    assert events[0].payload["vehicle_id"] == "v0"


def test_gps_event_emitted_within_range():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=50.0, v=10.0)
    sim.world.gps["g0"] = GPSSensor(id="g0", x=0.0, y=0.0, sigma_m=2.0, update_hz=100.0, radius_m=1000.0)
    events = []
    sim.bus.subscribe("sensor.gps", events.append)
    for _ in range(5):
        sim.step(0.1)
    assert len(events) > 0


def test_gps_event_not_emitted_outside_range():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=50.0, v=10.0)
    # GPS placed 10 000 m away with tiny radius
    sim.world.gps["g0"] = GPSSensor(id="g0", x=10000.0, y=10000.0, sigma_m=2.0, update_hz=100.0, radius_m=1.0)
    events = []
    sim.bus.subscribe("sensor.gps", events.append)
    for _ in range(10):
        sim.step(0.1)
    assert len(events) == 0


def test_event_payload_has_required_fields():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=10.0)
    sim.world.gps["g0"] = GPSSensor(id="g0", x=0.0, y=0.0, sigma_m=2.0, update_hz=100.0, radius_m=9999.0)
    gps_events = []
    sim.bus.subscribe("sensor.gps", gps_events.append)
    sim.step(0.1)
    required = {"t", "vehicle_id", "x", "y", "sigma_m", "confidence", "x_true", "y_true"}
    for ev in gps_events:
        for field in required:
            assert field in ev.payload, f"Missing field '{field}' in GPS event"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_shortest_route_lanes_same_node():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    meta = sim._lane_meta[lid]
    dest = meta["to"]
    result = sim.shortest_route_lanes(lid, dest)
    assert result == [lid]


def test_shortest_route_lanes_unreachable():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    result = sim.shortest_route_lanes(lid, "nonexistent_node")
    assert result == []


def test_set_route_to_node_returns_true():
    sim = _make_sim()
    lids = list(sim.world.lanes.keys())
    lid = lids[0]
    # Route to the 'to' node of the second lane
    dest = sim._lane_meta[lids[-1]]["to"]
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=10.0)
    ok = sim.set_route_to_node("v0", dest)
    assert ok is True


def test_clear_route_removes_all_state():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    meta = sim._lane_meta[lid]
    sim._route_next["v0"] = [lid]
    sim._route_dest_node["v0"] = meta["to"]
    sim._route_queue["v0"] = [(lid, 10.0)]
    sim.clear_route("v0")
    assert "v0" not in sim._route_next
    assert "v0" not in sim._route_dest_node
    assert "v0" not in sim._route_queue


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------

def test_snapshot_restore_preserves_time():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=10.0)
    for _ in range(20):
        sim.step(0.1)
    snap = sim.snapshot()
    t_snap = sim.t
    for _ in range(10):
        sim.step(0.1)
    sim.restore(snap)
    assert abs(sim.t - t_snap) < 1e-9


def test_snapshot_restore_vehicle_position():
    sim = _make_sim()
    lid = next(iter(sim.world.lanes))
    sim.world.vehicles["v0"] = Vehicle(id="v0", lane_id=lid, s=0.0, v=10.0)
    for _ in range(20):
        sim.step(0.1)
    snap = sim.snapshot()
    s_snap = snap["world"].vehicles["v0"].s
    for _ in range(50):
        sim.step(0.1)
    sim.restore(snap)
    assert abs(sim.world.vehicles["v0"].s - s_snap) < 1e-9


if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
