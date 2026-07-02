"""Author one example scenario JSON per supported anomaly class, then load
and run each via the existing headless code path to confirm the anomaly is
physically realisable with the current simulator (no new features).

Output:
    outputs/test_scenarios/anomalies/<class>.json    — authored scene
    (stdout)                                          — pass/fail per class

Run:
    PYTHONPATH=src python scripts/make_anomaly_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the repo root.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from simstudio.bus import EventBus
from simstudio.constants import IDM_VEHICLE_LENGTH_M
from simstudio.project_io import load_world
from simstudio.sim_core import Simulation

OUT_DIR = ROOT / "outputs" / "test_scenarios" / "anomalies"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Scene template — one 500 m segment, one DAS sensor, no auto-spawn.
# ---------------------------------------------------------------------------


def _base_scene() -> dict:
    return {
        "nodes": {
            "A": {"x": 0.0, "y": 0.0},
            "B": {"x": 500.0, "y": 0.0},
        },
        "segments": {
            "S": {
                "id": "S",
                "n0": "A",
                "n1": "B",
                "lanes": 1,
                "lane_width": 3.6,
                "speed_limit_mps": 13.9,
                "points": [],
                "one_way": True,
                "traffic_level": "none",
            }
        },
        "lanes": {
            "S_fwd_lane1": {
                "id": "S_fwd_lane1",
                "segment_id": "S",
                "offset_index": 0,
                "polyline": [[0.0, 0.0], [500.0, 0.0]],
            }
        },
        "vehicles": {},
        "gps": {},
        "cameras": {},
        "das": {
            "das1": {
                "id": "das1",
                "segment_id": "S",
                "channel_start": 0,
                "channel_end": 500,
                "update_hz": 30.0,
                "noise_std": 30.0,
                "fiber_offset_m": 2.0,
                "d0_m": 0.7,
            }
        },
        "auto_spawn": False,
    }


def _vehicle(vid: str, s: float, v: float = 13.9, **overrides) -> dict:
    """Minimal vehicle dict with the speed/accel envelope pinned, plus any
    anomaly-enabling override fields the caller specifies."""
    veh = {
        "id": vid,
        "lane_id": "S_fwd_lane1",
        "s": s,
        "v": v,
        "weight_kg": 1500.0,
        "speed_min_mps": v,
        "speed_max_mps": v,
        "target_speed_mps": v,
        "speed_mean_mps": v,
        "speed_std_mps": 0.0,
        "max_accel_mps2": 1.5,
        "max_decel_mps2": 2.6,
        "accel_response_s": 0.8,
    }
    veh.update(overrides)
    return veh


# ---------------------------------------------------------------------------
# One scene per anomaly class.
# Each scene is *self-contained* and already contains the override that
# makes the anomaly happen from t=0 (no runtime injection required).
# ---------------------------------------------------------------------------


def scene_sudden_braking() -> dict:
    scene = _base_scene()
    # A single car set to panic-brake at -7 m/s² for its first 3 seconds of life.
    scene["vehicles"]["V1"] = _vehicle(
        "V1",
        s=50.0,
        v=13.9,
        a_cmd_override_mps2=-7.0,
        a_cmd_override_until_t=3.0,
    )
    scene["_description"] = (
        "Sudden braking: V1 bypasses IDM and decelerates at -7 m/s^2 for 3 s "
        "(anomaly field: a_cmd_override_mps2 / a_cmd_override_until_t)."
    )
    return scene


def scene_tailgating() -> dict:
    scene = _base_scene()
    # Leader crawling at 8 m/s; follower with IDM s0=0.5, T=0.3 sits at <5 m gap.
    scene["vehicles"]["LEADER"] = _vehicle("LEADER", s=50.0, v=8.0)
    for k in ("speed_min_mps", "speed_max_mps", "target_speed_mps", "speed_mean_mps"):
        scene["vehicles"]["LEADER"][k] = 8.0
    scene["vehicles"]["FOLLOWER"] = _vehicle(
        "FOLLOWER",
        s=10.0,
        v=13.9,
        min_gap_m=0.5,
        time_headway_s=0.3,
    )
    scene["_description"] = (
        "Tailgating: follower with min_gap_m=0.5, time_headway_s=0.3 closes "
        "to a sub-5 m gap behind a slow leader."
    )
    return scene


def scene_stalled_vehicle() -> dict:
    scene = _base_scene()
    # A frozen car blocks the lane at s=60; an approaching car brakes via IDM.
    scene["vehicles"]["STALLED"] = _vehicle(
        "STALLED", s=60.0, v=0.0, frozen=True,
    )
    for k in ("speed_min_mps", "speed_max_mps", "target_speed_mps", "speed_mean_mps"):
        scene["vehicles"]["STALLED"][k] = 0.0
    scene["vehicles"]["APPROACH"] = _vehicle("APPROACH", s=10.0, v=13.9)
    scene["_description"] = (
        "Stalled vehicle: frozen=True holds STALLED at v=0; APPROACH brakes "
        "automatically under IDM."
    )
    return scene


def scene_obstacle_in_lane() -> dict:
    scene = _base_scene()
    # Pedestrian: frozen Vehicle, 80 kg, object_type='pedestrian'.
    scene["vehicles"]["PED"] = _vehicle(
        "PED",
        s=60.0,
        v=0.0,
        weight_kg=80.0,
        frozen=True,
        object_type="pedestrian",
    )
    for k in ("speed_min_mps", "speed_max_mps", "target_speed_mps", "speed_mean_mps"):
        scene["vehicles"]["PED"][k] = 0.0
    scene["vehicles"]["CAR"] = _vehicle("CAR", s=10.0, v=13.9)
    scene["_description"] = (
        "Obstacle in lane: frozen Vehicle with weight_kg=80 and "
        "object_type='pedestrian' — DAS signature scales with W, IDM treats "
        "it as a stopped leader."
    )
    return scene


def scene_weaving() -> dict:
    scene = _base_scene()
    scene["vehicles"]["V1"] = _vehicle(
        "V1",
        s=20.0,
        v=13.9,
        lateral_mode="weave",
        lateral_weave_amplitude_m=2.5,
        lateral_weave_period_s=4.0,
    )
    scene["_description"] = (
        "Weaving: lateral_mode='weave' with amplitude 2.5 m, period 4 s — "
        "vehicle oscillates beyond the ±half-lane clamp used in normal driving."
    )
    return scene


def scene_lane_straddling() -> dict:
    scene = _base_scene()
    scene["vehicles"]["V1"] = _vehicle(
        "V1",
        s=20.0,
        v=13.9,
        lateral_mode="straddle",
        lateral_fixed_offset_m=1.8,
    )
    scene["_description"] = (
        "Lane straddling: lateral_mode='straddle' pins lateral_offset_m at "
        "+1.8 m (on the lane edge) indefinitely."
    )
    return scene


def scene_collision() -> dict:
    scene = _base_scene()
    # Stopped leader; follower charges through with allow_collision=True, disable_idm=True.
    scene["vehicles"]["LEADER"] = _vehicle(
        "LEADER", s=30.0, v=0.0, frozen=True,
    )
    for k in ("speed_min_mps", "speed_max_mps", "target_speed_mps", "speed_mean_mps"):
        scene["vehicles"]["LEADER"][k] = 0.0
    scene["vehicles"]["FOLLOWER"] = _vehicle(
        "FOLLOWER",
        s=10.0,
        v=13.9,
        allow_collision=True,
        disable_idm=True,
    )
    scene["_description"] = (
        "Collision: stopped leader + follower with allow_collision=True and "
        "disable_idm=True — simulator emits world.collision and freezes both."
    )
    return scene


def scene_speeding() -> dict:
    scene = _base_scene()
    # Segment speed limit is 13.9; vehicle allowed to reach 30 m/s.
    scene["vehicles"]["V1"] = _vehicle(
        "V1",
        s=20.0,
        v=13.9,
        ignore_speed_limit=True,
    )
    # Raise the vehicle's own envelope so it actually pulls away from the limit.
    scene["vehicles"]["V1"]["speed_min_mps"] = 28.0
    scene["vehicles"]["V1"]["speed_max_mps"] = 30.0
    scene["vehicles"]["V1"]["target_speed_mps"] = 30.0
    scene["vehicles"]["V1"]["speed_mean_mps"] = 30.0
    scene["_description"] = (
        "Speeding: ignore_speed_limit=True bypasses the 1.12× segment-limit "
        "clamp; vehicle pulls to its own 30 m/s target."
    )
    return scene


# ---------------------------------------------------------------------------
# Verification harness: load each scene from disk and run it.
# ---------------------------------------------------------------------------


SCENES = [
    ("sudden_braking", scene_sudden_braking, 3.0),
    ("tailgating", scene_tailgating, 30.0),
    ("stalled_vehicle", scene_stalled_vehicle, 10.0),
    ("obstacle_in_lane", scene_obstacle_in_lane, 10.0),
    ("weaving", scene_weaving, 4.0),
    ("lane_straddling", scene_lane_straddling, 2.0),
    ("collision", scene_collision, 5.0),
    ("speeding", scene_speeding, 40.0),
]


def _run(path: Path, seconds: float):
    world = load_world(path)
    sim = Simulation(EventBus(), world)
    sim.rebuild_lanes()
    collisions = []
    sim.bus.subscribe("world.collision", lambda ev: collisions.append(ev))
    lateral_trace = []
    n = int(seconds / 0.05)
    for _ in range(n):
        sim.step(0.05)
        if world.vehicles:
            v = next(iter(world.vehicles.values()))
            lateral_trace.append(v.lateral_offset_m)
    return world, collisions, lateral_trace


def _check_sudden_braking(w, cols, lat):
    v = w.vehicles["V1"].v
    assert v < 1.0, f"V1 must be near-stopped after 3 s, got v={v:.2f}"
    return f"v_final={v:.2f} m/s (target < 1.0)"


def _check_tailgating(w, cols, lat):
    leader = w.vehicles["LEADER"]; follower = w.vehicles["FOLLOWER"]
    gap = leader.s - follower.s - IDM_VEHICLE_LENGTH_M
    assert 0 < gap < 5.0, f"gap must be in (0, 5), got {gap:.2f}"
    return f"gap={gap:.2f} m (target 0 < gap < 5)"


def _check_stalled(w, cols, lat):
    stalled = w.vehicles["STALLED"]; approach = w.vehicles["APPROACH"]
    assert stalled.v == 0.0 and abs(stalled.s - 60.0) < 1e-6, \
        f"STALLED drifted: s={stalled.s:.2f} v={stalled.v:.2f}"
    gap = stalled.s - approach.s - IDM_VEHICLE_LENGTH_M
    assert gap >= 1.5, f"APPROACH must stop short of stalled car, gap={gap:.2f}"
    return f"stalled_s={stalled.s:.1f} approach_v={approach.v:.2f} gap={gap:.2f}"


def _check_obstacle(w, cols, lat):
    ped = w.vehicles["PED"]; car = w.vehicles["CAR"]
    assert ped.object_type == "pedestrian", f"label lost: {ped.object_type}"
    assert car.v < 1.0, f"CAR must stop before pedestrian, v={car.v:.2f}"
    gap = ped.s - car.s - IDM_VEHICLE_LENGTH_M
    return f"label={ped.object_type} car_v={car.v:.2f} gap={gap:.2f}"


def _check_weaving(w, cols, lat):
    assert max(lat) > 1.8, f"max lateral only {max(lat):.2f} m"
    assert min(lat) < -1.8, f"min lateral only {min(lat):.2f} m"
    return f"lat_range=[{min(lat):.2f}, {max(lat):.2f}] (target |x|>1.8)"


def _check_straddle(w, cols, lat):
    v = w.vehicles["V1"]
    assert abs(v.lateral_offset_m - 1.8) < 1e-6, f"offset={v.lateral_offset_m:.2f}"
    return f"lateral_offset_m={v.lateral_offset_m:.2f} (target 1.80)"


def _check_collision(w, cols, lat):
    assert len(cols) == 1, f"expected 1 collision event, got {len(cols)}"
    assert w.vehicles["FOLLOWER"].frozen and w.vehicles["LEADER"].frozen, \
        "both vehicles should be frozen post-impact"
    return f"events={len(cols)} both_frozen=True"


def _check_speeding(w, cols, lat):
    v = w.vehicles["V1"].v
    # Speed limit is 13.9 → 1.12× = 15.57. Must exceed that by far.
    assert v > 20.0, f"V1 only reached {v:.2f} m/s"
    return f"v_final={v:.2f} m/s (speed_limit=13.9, 1.12x=15.57)"


CHECKS = {
    "sudden_braking": _check_sudden_braking,
    "tailgating": _check_tailgating,
    "stalled_vehicle": _check_stalled,
    "obstacle_in_lane": _check_obstacle,
    "weaving": _check_weaving,
    "lane_straddling": _check_straddle,
    "collision": _check_collision,
    "speeding": _check_speeding,
}


def main() -> int:
    failures = 0
    for name, builder, seconds in SCENES:
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(builder(), indent=2))
        try:
            world, cols, lat = _run(path, seconds)
            msg = CHECKS[name](world, cols, lat)
            print(f"  PASS  {name:<20s}  {msg}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name:<20s}  {e}")
    print()
    print(f"Authored {len(SCENES)} scenario JSONs under {OUT_DIR.relative_to(ROOT)}")
    print(f"Passed {len(SCENES) - failures}/{len(SCENES)} end-to-end load+run+verify")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
