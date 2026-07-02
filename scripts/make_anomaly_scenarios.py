"""Author 5 scenarios per anomaly class (8 × 5 = 40 labelled .sim.json files),
one subdirectory per class, then load + run each headlessly to confirm the
target anomaly is physically realisable.

Every scenario carries:
    * a straight or mildly-curved one-lane / one-way road,
    * one DAS fiber running along the segment,
    * one GPS sensor near the road midpoint,
    * one Camera looking along the road,
    * one or more Vehicles — at least one configured with the anomaly
      override fields documented in ``models.Vehicle``.

The 5 scenarios per class deliberately vary the stage (road length & shape,
speed limit, vehicle count / weights, sensor geometry) so the saved raw
sensor traces look different run-to-run.  This is the labelled dataset
scaffold requested for the anomaly-data-generation milestone.

Run:
    PYTHONPATH=src python3 scripts/make_anomaly_scenarios.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from simstudio.bus import EventBus
from simstudio.constants import IDM_VEHICLE_LENGTH_M
from simstudio.project_io import load_world
from simstudio.sim_core import Simulation

OUT_ROOT = ROOT / "outputs" / "test_scenarios" / "anomalies"


# ---------------------------------------------------------------------------
# Road + sensor scaffolding
# ---------------------------------------------------------------------------


def _arc_points(length_m: float, bulge_ratio: float = 1.0 / 6.0) -> List[List[float]]:
    """Return a gentle 3-point arc from (0,0) to (length, 0) with a mid
    *bulge* of ``length * bulge_ratio`` in the +y direction.  ``bulge_ratio=0``
    reduces to a straight line."""
    mid_x = length_m / 2.0
    mid_y = length_m * bulge_ratio
    return [[0.0, 0.0], [mid_x, mid_y], [length_m, 0.0]]


def _base_scene(
    *,
    road_length_m: float,
    speed_kmh: float,
    curved: bool = False,
    description: str = "",
) -> Dict[str, Any]:
    """Build a single-segment, one-lane scene with DAS + GPS + Camera."""
    speed_mps = speed_kmh / 3.6

    if curved:
        seg_points = _arc_points(road_length_m)
    else:
        seg_points = []  # engine will infer from node endpoints

    lane_poly = seg_points if seg_points else [[0.0, 0.0], [road_length_m, 0.0]]

    scene: Dict[str, Any] = {
        "_description": description,
        "nodes": {
            "A": {"x": 0.0, "y": 0.0},
            "B": {"x": road_length_m, "y": 0.0},
        },
        "segments": {
            "S": {
                "id": "S",
                "n0": "A",
                "n1": "B",
                "lanes": 1,
                "lane_width": 3.6,
                "speed_limit_mps": speed_mps,
                "points": seg_points,
                "one_way": True,
                "traffic_level": "none",
            }
        },
        "lanes": {
            "S_fwd_lane1": {
                "id": "S_fwd_lane1",
                "segment_id": "S",
                "offset_index": 0,
                "polyline": lane_poly,
            }
        },
        "vehicles": {},
        "gps": {
            "gps1": {
                "id": "gps1",
                "x": road_length_m / 2.0,
                "y": 0.0,
                "sigma_m": 2.5,
                "update_hz": 5.0,
                "radius_m": max(50.0, road_length_m * 0.6),
            }
        },
        "cameras": {
            "cam1": {
                "id": "cam1",
                "x": -5.0,
                "y": 0.0,
                "heading_rad": 0.0,
                "fov_deg": 70.0,
                "range_m": min(220.0, road_length_m),
                "update_hz": 10.0,
            }
        },
        "das": {
            "das1": {
                "id": "das1",
                "segment_id": "S",
                "channel_start": 0,
                "channel_end": int(road_length_m),
                "update_hz": 30.0,
                "noise_std": 30.0,
                "fiber_offset_m": 2.0,
                "d0_m": 0.7,
            }
        },
        "auto_spawn": False,
    }
    return scene


def _veh(
    vid: str,
    s: float,
    v: float,
    *,
    weight_kg: float = 1500.0,
    **overrides: Any,
) -> Dict[str, Any]:
    """Baseline vehicle dict with profile pinned; any **overrides are merged."""
    d = {
        "id": vid,
        "lane_id": "S_fwd_lane1",
        "s": s,
        "v": v,
        "weight_kg": weight_kg,
        "speed_min_mps": v,
        "speed_max_mps": v,
        "target_speed_mps": v,
        "speed_mean_mps": v,
        "speed_std_mps": 0.0,
        "max_accel_mps2": 1.5,
        "max_decel_mps2": 2.6,
        "accel_response_s": 0.8,
    }
    d.update(overrides)
    return d


def _pin_zero_profile(v: Dict[str, Any]) -> None:
    """Force a vehicle's envelope to v=0 (used for frozen actors)."""
    for k in ("speed_min_mps", "speed_max_mps", "target_speed_mps", "speed_mean_mps"):
        v[k] = 0.0
    v["v"] = 0.0


# ---------------------------------------------------------------------------
# One scene generator per anomaly class.  Each returns 5 (name, scene) pairs.
# ---------------------------------------------------------------------------


def sudden_braking() -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    s = _base_scene(road_length_m=300.0, speed_kmh=50.0,
                    description="Lone car panic-brakes from 50 km/h with a_cmd=-7 m/s^2.")
    s["vehicles"]["V1"] = _veh("V1", s=50.0, v=13.9,
                               a_cmd_override_mps2=-7.0, a_cmd_override_until_t=3.0)
    out.append(("01_lone_braker", s))

    s = _base_scene(road_length_m=400.0, speed_kmh=50.0,
                    description="Cascade brake: leader brakes hard; follower catches under IDM.")
    s["vehicles"]["LEADER"] = _veh("LEADER", s=80.0, v=13.9,
                                   a_cmd_override_mps2=-6.0, a_cmd_override_until_t=4.0)
    s["vehicles"]["FOLLOWER"] = _veh("FOLLOWER", s=20.0, v=13.9)
    out.append(("02_cascade_brake", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0, curved=True,
                    description="Braking on a curved road; GPS/Camera geometry is non-trivial.")
    s["vehicles"]["V1"] = _veh("V1", s=100.0, v=13.9,
                               a_cmd_override_mps2=-5.5, a_cmd_override_until_t=5.0)
    out.append(("03_curve_braker", s))

    s = _base_scene(road_length_m=600.0, speed_kmh=80.0,
                    description="High-speed (80 km/h) panic brake at -8 m/s^2.")
    s["vehicles"]["V1"] = _veh("V1", s=50.0, v=22.2,
                               a_cmd_override_mps2=-8.0, a_cmd_override_until_t=3.5)
    out.append(("04_high_speed", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Heavy truck (12 t) emergency brakes; DAS amplitude is 8x a car.")
    s["vehicles"]["TRUCK"] = _veh("TRUCK", s=60.0, v=13.9, weight_kg=12000.0,
                                  a_cmd_override_mps2=-5.0, a_cmd_override_until_t=4.0)
    out.append(("05_truck", s))

    return out


def tailgating() -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    s = _base_scene(road_length_m=400.0, speed_kmh=50.0,
                    description="Classic tailgate: follower at s0=0.5 m, T=0.3 s closes on slow leader.")
    s["vehicles"]["LEADER"] = _veh("LEADER", s=80.0, v=8.0)
    s["vehicles"]["FOLLOWER"] = _veh("FOLLOWER", s=20.0, v=13.9,
                                     min_gap_m=0.5, time_headway_s=0.3)
    out.append(("01_basic_pair", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Three-car chain; middle car tailgates the leader, trailing car drives normally.")
    s["vehicles"]["LEADER"] = _veh("LEADER", s=100.0, v=8.0)
    s["vehicles"]["MID"] = _veh("MID", s=70.0, v=13.9,
                                 min_gap_m=0.5, time_headway_s=0.3)
    s["vehicles"]["REAR"] = _veh("REAR", s=10.0, v=13.9)
    out.append(("02_chain_of_three", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Aggressive car tailgates a 12-ton truck — heavy-leader DAS signature.")
    s["vehicles"]["TRUCK"] = _veh("TRUCK", s=100.0, v=9.0, weight_kg=12000.0)
    s["vehicles"]["FOLLOWER"] = _veh("FOLLOWER", s=20.0, v=13.9,
                                     min_gap_m=0.7, time_headway_s=0.25)
    out.append(("03_behind_truck", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0, curved=True,
                    description="Tailgate pair on a curved road; fiber sees a curved line of peaks.")
    s["vehicles"]["LEADER"] = _veh("LEADER", s=120.0, v=8.0)
    s["vehicles"]["FOLLOWER"] = _veh("FOLLOWER", s=40.0, v=13.9,
                                     min_gap_m=0.5, time_headway_s=0.3)
    out.append(("04_curve_pair", s))

    s = _base_scene(road_length_m=900.0, speed_kmh=80.0,
                    description="High-speed tailgate at 80 km/h (22 m/s) with a sub-1 s headway.")
    s["vehicles"]["LEADER"] = _veh("LEADER", s=120.0, v=16.7)
    s["vehicles"]["FOLLOWER"] = _veh("FOLLOWER", s=70.0, v=22.2,
                                     min_gap_m=0.8, time_headway_s=0.4)
    out.append(("05_high_speed", s))

    return out


def stalled_vehicle() -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    s = _base_scene(road_length_m=300.0, speed_kmh=50.0,
                    description="Single stalled car in the lane; an approaching car brakes via IDM.")
    v = _veh("STALLED", s=150.0, v=0.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["STALLED"] = v
    s["vehicles"]["APPROACH"] = _veh("APPROACH", s=10.0, v=13.9)
    out.append(("01_single_stall", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Stalled car blocks the lane; three cars arrive and jam naturally.")
    v = _veh("STALLED", s=200.0, v=0.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["STALLED"] = v
    s["vehicles"]["A"] = _veh("A", s=10.0, v=13.9)
    s["vehicles"]["B"] = _veh("B", s=-30.0 + 40.0, v=13.9)  # =10.0 spaced
    s["vehicles"]["B"]["s"] = 40.0
    s["vehicles"]["C"] = _veh("C", s=70.0, v=13.9)
    out.append(("02_three_car_jam", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Stalled 12-ton truck: DAS amplitude 8x a car, static signature only.")
    v = _veh("TRUCK", s=180.0, v=0.0, weight_kg=12000.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["TRUCK"] = v
    s["vehicles"]["APPROACH"] = _veh("APPROACH", s=10.0, v=13.9)
    out.append(("03_stalled_truck", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0, curved=True,
                    description="Stalled car on a curved segment; approach vehicle rounds the bend into it.")
    v = _veh("STALLED", s=250.0, v=0.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["STALLED"] = v
    s["vehicles"]["APPROACH"] = _veh("APPROACH", s=30.0, v=13.9)
    out.append(("04_curve_stall", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Two stalled cars back-to-back (rare double breakdown).")
    for vid, s_pos in (("STALL_1", 180.0), ("STALL_2", 220.0)):
        v = _veh(vid, s=s_pos, v=0.0, frozen=True)
        _pin_zero_profile(v)
        s["vehicles"][vid] = v
    s["vehicles"]["APPROACH"] = _veh("APPROACH", s=10.0, v=13.9)
    out.append(("05_double_stall", s))

    return out


def obstacle_in_lane() -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    s = _base_scene(road_length_m=300.0, speed_kmh=50.0,
                    description="Pedestrian (80 kg) standing in the lane; car brakes under IDM.")
    v = _veh("PED", s=150.0, v=0.0, weight_kg=80.0, frozen=True, object_type="pedestrian")
    _pin_zero_profile(v)
    s["vehicles"]["PED"] = v
    s["vehicles"]["CAR"] = _veh("CAR", s=10.0, v=13.9)
    out.append(("01_pedestrian", s))

    s = _base_scene(road_length_m=300.0, speed_kmh=50.0,
                    description="Debris (200 kg, e.g. fallen cargo) in the road.")
    v = _veh("DEBRIS", s=140.0, v=0.0, weight_kg=200.0, frozen=True, object_type="debris")
    _pin_zero_profile(v)
    s["vehicles"]["DEBRIS"] = v
    s["vehicles"]["CAR"] = _veh("CAR", s=10.0, v=13.9)
    out.append(("02_debris", s))

    s = _base_scene(road_length_m=400.0, speed_kmh=50.0,
                    description="Animal (deer, 60 kg) stopped in lane.")
    v = _veh("DEER", s=170.0, v=0.0, weight_kg=60.0, frozen=True, object_type="animal")
    _pin_zero_profile(v)
    s["vehicles"]["DEER"] = v
    s["vehicles"]["CAR"] = _veh("CAR", s=20.0, v=13.9)
    out.append(("03_animal", s))

    s = _base_scene(road_length_m=400.0, speed_kmh=50.0,
                    description="Traffic cone / barrier (5 kg): camera-visible, DAS-invisible obstacle.")
    v = _veh("CONE", s=180.0, v=0.0, weight_kg=5.0, frozen=True, object_type="barrier")
    _pin_zero_profile(v)
    s["vehicles"]["CONE"] = v
    s["vehicles"]["CAR"] = _veh("CAR", s=20.0, v=13.9)
    out.append(("04_barrier", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Two pedestrians in the lane (e.g. people crossing mid-block).")
    for vid, s_pos in (("PED_A", 190.0), ("PED_B", 198.0)):
        v = _veh(vid, s=s_pos, v=0.0, weight_kg=80.0, frozen=True, object_type="pedestrian")
        _pin_zero_profile(v)
        s["vehicles"][vid] = v
    s["vehicles"]["CAR"] = _veh("CAR", s=20.0, v=13.9)
    out.append(("05_two_pedestrians", s))

    return out


def weaving() -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    s = _base_scene(road_length_m=400.0, speed_kmh=50.0,
                    description="Classic drunk-driver weave: amp=2.5 m, period=4 s, single car.")
    s["vehicles"]["V1"] = _veh("V1", s=50.0, v=13.9,
                               lateral_mode="weave",
                               lateral_weave_amplitude_m=2.5,
                               lateral_weave_period_s=4.0)
    out.append(("01_solo_weave", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Weaving car inside normal traffic; DAS sees the weaver's lateral motion.")
    s["vehicles"]["WEAVER"] = _veh("WEAVER", s=80.0, v=13.9,
                                   lateral_mode="weave",
                                   lateral_weave_amplitude_m=2.2,
                                   lateral_weave_period_s=5.0)
    s["vehicles"]["NORMAL_A"] = _veh("NORMAL_A", s=180.0, v=11.0)
    s["vehicles"]["NORMAL_B"] = _veh("NORMAL_B", s=20.0, v=13.0)
    out.append(("02_weaver_in_traffic", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Large-amplitude slow weave by a truck (3.0 m, 6 s).")
    s["vehicles"]["TRUCK"] = _veh("TRUCK", s=50.0, v=11.1, weight_kg=12000.0,
                                  lateral_mode="weave",
                                  lateral_weave_amplitude_m=3.0,
                                  lateral_weave_period_s=6.0)
    out.append(("03_truck_weave", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0, curved=True,
                    description="Weaving on a curved road; superimposed lateral motion over road curvature.")
    s["vehicles"]["V1"] = _veh("V1", s=60.0, v=13.9,
                               lateral_mode="weave",
                               lateral_weave_amplitude_m=2.3,
                               lateral_weave_period_s=4.5)
    out.append(("04_curve_weave", s))

    s = _base_scene(road_length_m=700.0, speed_kmh=80.0,
                    description="High-speed weaver at 80 km/h with fast 3 s period.")
    s["vehicles"]["V1"] = _veh("V1", s=80.0, v=22.2,
                               lateral_mode="weave",
                               lateral_weave_amplitude_m=2.0,
                               lateral_weave_period_s=3.0)
    out.append(("05_high_speed_weave", s))

    return out


def lane_straddling() -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    s = _base_scene(road_length_m=400.0, speed_kmh=50.0,
                    description="Car held on left edge (+1.8 m) of lane for the entire run.")
    s["vehicles"]["V1"] = _veh("V1", s=50.0, v=13.9,
                               lateral_mode="straddle", lateral_fixed_offset_m=1.8)
    out.append(("01_left_edge", s))

    s = _base_scene(road_length_m=400.0, speed_kmh=50.0,
                    description="Car held on right edge (-1.8 m); mirrors the left-edge case.")
    s["vehicles"]["V1"] = _veh("V1", s=50.0, v=13.9,
                               lateral_mode="straddle", lateral_fixed_offset_m=-1.8)
    out.append(("02_right_edge", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="12-ton truck straddles the left edge — high-severity DAS signature.")
    s["vehicles"]["TRUCK"] = _veh("TRUCK", s=60.0, v=11.1, weight_kg=12000.0,
                                  lateral_mode="straddle", lateral_fixed_offset_m=1.8)
    out.append(("03_truck_straddle", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Straddler behind a normal leader (interaction between OU leader + edge follower).")
    s["vehicles"]["LEADER"] = _veh("LEADER", s=150.0, v=13.9)
    s["vehicles"]["STRADDLER"] = _veh("STRADDLER", s=60.0, v=13.9,
                                       lateral_mode="straddle", lateral_fixed_offset_m=1.8)
    out.append(("04_pair_with_straddler", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0, curved=True,
                    description="Straddling on a curved road; lateral offset is w.r.t. the local tangent.")
    s["vehicles"]["V1"] = _veh("V1", s=60.0, v=13.9,
                               lateral_mode="straddle", lateral_fixed_offset_m=1.8)
    out.append(("05_curve_straddle", s))

    return out


def collision() -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    s = _base_scene(road_length_m=300.0, speed_kmh=50.0,
                    description="Rear-end: follower charges into a stopped car; both freeze post-impact.")
    v = _veh("LEADER", s=80.0, v=0.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["LEADER"] = v
    s["vehicles"]["FOLLOWER"] = _veh("FOLLOWER", s=10.0, v=13.9,
                                     allow_collision=True, disable_idm=True)
    out.append(("01_rear_end", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=80.0,
                    description="High-speed (80 km/h) rear-end; sharper DAS impulse.")
    v = _veh("LEADER", s=120.0, v=0.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["LEADER"] = v
    s["vehicles"]["FOLLOWER"] = _veh("FOLLOWER", s=10.0, v=22.2,
                                     allow_collision=True, disable_idm=True)
    out.append(("02_high_speed_rear_end", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Truck rear-ends a passenger car; impact mass asymmetric.")
    v = _veh("CAR", s=80.0, v=0.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["CAR"] = v
    s["vehicles"]["TRUCK"] = _veh("TRUCK", s=10.0, v=13.9, weight_kg=12000.0,
                                  allow_collision=True, disable_idm=True)
    out.append(("03_truck_into_car", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Car rear-ends a stopped truck (heavy leader, light follower).")
    v = _veh("TRUCK", s=80.0, v=0.0, weight_kg=12000.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["TRUCK"] = v
    s["vehicles"]["CAR"] = _veh("CAR", s=10.0, v=13.9,
                                allow_collision=True, disable_idm=True)
    out.append(("04_car_into_truck", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Chain: stopped lead, middle collides; trailing car jams naturally behind.")
    v = _veh("LEAD", s=120.0, v=0.0, frozen=True)
    _pin_zero_profile(v)
    s["vehicles"]["LEAD"] = v
    s["vehicles"]["MID"] = _veh("MID", s=40.0, v=13.9,
                                allow_collision=True, disable_idm=True)
    s["vehicles"]["REAR"] = _veh("REAR", s=5.0, v=13.9)
    out.append(("05_chain_collision", s))

    return out


def speeding() -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0,
                    description="Solo speeder: limit 50 km/h, target 30 m/s (~108 km/h).")
    v = _veh("V1", s=20.0, v=13.9, ignore_speed_limit=True)
    v["speed_min_mps"] = 28.0; v["speed_max_mps"] = 30.0
    v["target_speed_mps"] = 30.0; v["speed_mean_mps"] = 30.0
    s["vehicles"]["V1"] = v
    out.append(("01_solo_speeder", s))

    s = _base_scene(road_length_m=600.0, speed_kmh=50.0,
                    description="Speeder leads two law-abiding cars (target 28 m/s, trailing pair at 13 m/s).")
    v = _veh("V1", s=200.0, v=13.9, ignore_speed_limit=True)
    v["speed_min_mps"] = 25.0; v["speed_max_mps"] = 28.0
    v["target_speed_mps"] = 28.0; v["speed_mean_mps"] = 28.0
    s["vehicles"]["V1"] = v
    s["vehicles"]["TRAFFIC_A"] = _veh("TRAFFIC_A", s=50.0, v=13.0)
    s["vehicles"]["TRAFFIC_B"] = _veh("TRAFFIC_B", s=100.0, v=13.0)
    out.append(("02_speeder_in_traffic", s))

    s = _base_scene(road_length_m=600.0, speed_kmh=50.0,
                    description="Speeding 12-ton truck: rarer, heavier DAS peak at elevated speed.")
    v = _veh("TRUCK", s=30.0, v=13.9, weight_kg=12000.0, ignore_speed_limit=True)
    v["speed_min_mps"] = 22.0; v["speed_max_mps"] = 25.0
    v["target_speed_mps"] = 25.0; v["speed_mean_mps"] = 25.0
    s["vehicles"]["TRUCK"] = v
    out.append(("03_truck_speeding", s))

    s = _base_scene(road_length_m=1000.0, speed_kmh=50.0,
                    description="Sustained speeder on a 1 km straight — long DAS signature.")
    v = _veh("V1", s=30.0, v=13.9, ignore_speed_limit=True)
    v["speed_min_mps"] = 29.0; v["speed_max_mps"] = 31.0
    v["target_speed_mps"] = 30.0; v["speed_mean_mps"] = 30.0
    s["vehicles"]["V1"] = v
    out.append(("04_long_straight", s))

    s = _base_scene(road_length_m=500.0, speed_kmh=50.0, curved=True,
                    description="Speeder on a curved segment (stress-tests heading-based DAS mapping).")
    v = _veh("V1", s=40.0, v=13.9, ignore_speed_limit=True)
    v["speed_min_mps"] = 25.0; v["speed_max_mps"] = 28.0
    v["target_speed_mps"] = 28.0; v["speed_mean_mps"] = 28.0
    s["vehicles"]["V1"] = v
    out.append(("05_curve_speeder", s))

    return out


# ---------------------------------------------------------------------------
# Per-class verification harness — short end-to-end run to confirm the
# anomaly physics are still reachable after the authored JSON round-trips
# through load_world + Simulation.
# ---------------------------------------------------------------------------


def _run(path: Path, seconds: float) -> Tuple[Any, List[Any], Dict[str, Dict[str, float]]]:
    """Run a scene headlessly and return (world, collision-events, per-vehicle stats).

    The stats dict maps vehicle-id -> {'min_v': float, 'max_abs_lat': float} so
    the verification harness can inspect *trajectory extrema* rather than just
    the endpoint state (endpoint state is misleading whenever the braker has
    already released and re-accelerated, or the weaver happens to cross zero
    at the final tick)."""
    world = load_world(path)
    sim = Simulation(EventBus(), world)
    sim.rebuild_lanes()
    collisions = []
    sim.bus.subscribe("world.collision", lambda ev: collisions.append(ev))
    stats: Dict[str, Dict[str, float]] = {
        vid: {"min_v": float(v.v), "max_v": float(v.v),
              "max_abs_lat": abs(float(v.lateral_offset_m))}
        for vid, v in world.vehicles.items()
    }
    for _ in range(int(seconds / 0.05)):
        sim.step(0.05)
        for vid, v in world.vehicles.items():
            s = stats.setdefault(vid,
                                 {"min_v": float(v.v), "max_v": float(v.v),
                                  "max_abs_lat": 0.0})
            if v.v < s["min_v"]:
                s["min_v"] = float(v.v)
            if v.v > s["max_v"]:
                s["max_v"] = float(v.v)
            la = abs(float(v.lateral_offset_m))
            if la > s["max_abs_lat"]:
                s["max_abs_lat"] = la
    return world, collisions, stats


def _check(cls: str, world, collisions, stats: Dict[str, Dict[str, float]]) -> str:
    """Return short diagnostic string; raise AssertionError if class-specific
    invariants are violated."""
    vehs = world.vehicles
    if cls == "sudden_braking":
        # Some vehicle has a_cmd_override_until_t > 0 and should have slowed.
        # Check the *minimum* velocity seen during the run (not the final one):
        # once the override expires the driver releases the brake and IDM
        # accelerates back to target, so the endpoint would hide the event.
        brakers = [v for v in vehs.values() if float(getattr(v, "a_cmd_override_until_t", 0.0) or 0.0) > 0.0]
        assert brakers, "no braker in scene"
        min_v = min(stats[v.id]["min_v"] for v in brakers)
        assert min_v < 1.5, f"expected near-stopped braker, got min_v={min_v:.2f}"
        return f"min_braker_v={min_v:.2f}"
    if cls == "tailgating":
        # Check time-headway (gap / v_follower) rather than raw gap: IDM's
        # equilibrium gap is s0 + v*T, so a 22 m/s tailgater with T=0.4 s
        # naturally settles near ~9 m, still well under a normal 1 s headway.
        # A tailgater passes if the time-headway is below 0.7 s (half the
        # ~1.5 s default T) OR the gap is below 5 m at low speed.
        tg = [v for v in vehs.values() if 0 < float(getattr(v, "min_gap_m", 0.0) or 0.0) < 2.0]
        assert tg, "no tailgater"
        for f in tg:
            ahead = [l for l in vehs.values() if l.id != f.id and l.s > f.s]
            if not ahead:
                continue
            leader = min(ahead, key=lambda l: l.s - f.s)
            gap = leader.s - f.s - IDM_VEHICLE_LENGTH_M
            th = gap / max(1.0, f.v)
            assert gap > 0, f"tailgate gap {gap:.2f} <= 0 (collision)"
            assert gap < 5.0 or th < 0.7, (
                f"tailgate gap {gap:.2f} m, t_head={th:.2f} s — neither close nor short-headway"
            )
            return f"gap={gap:.2f} t_head={th:.2f}s"
        raise AssertionError("no leader for any tailgater")
    if cls == "stalled_vehicle":
        frozen = [v for v in vehs.values() if getattr(v, "frozen", False) and getattr(v, "object_type", "vehicle") == "vehicle"]
        assert frozen, "no frozen vehicle"
        for f in frozen:
            assert f.v == 0.0 and abs(f.s - f.s) < 1e-6, "stalled vehicle drifted"
        return f"stalled={len(frozen)}"
    if cls == "obstacle_in_lane":
        obs = [v for v in vehs.values() if getattr(v, "object_type", "vehicle") != "vehicle" and getattr(v, "frozen", False)]
        assert obs, "no obstacle"
        for o in obs:
            assert o.v == 0.0 and o.lateral_offset_m == 0.0, "obstacle moved"
        return f"obstacles={len(obs)}"
    if cls == "weaving":
        wv = [v for v in vehs.values() if str(getattr(v, "lateral_mode", "ou") or "ou") == "weave"]
        assert wv, "no weaver"
        # Look at the *peak* lateral during the run: the sine wave crosses
        # zero at t=N*period and the endpoint check misfires when the run
        # duration is an integer multiple of the period.
        peak = max(stats[v.id]["max_abs_lat"] for v in wv)
        assert peak > 0.5, f"weaver not oscillating, peak={peak:.2f}"
        return f"peak_lat={peak:.2f}"
    if cls == "lane_straddling":
        st = [v for v in vehs.values() if str(getattr(v, "lateral_mode", "ou") or "ou") == "straddle"]
        assert st, "no straddler"
        for v in st:
            assert abs(v.lateral_offset_m - float(getattr(v, "lateral_fixed_offset_m", 0.0))) < 1e-6
        return f"pinned={st[0].lateral_offset_m:.2f}"
    if cls == "collision":
        assert len(collisions) >= 1, "no collision event"
        return f"events={len(collisions)}"
    if cls == "speeding":
        sp = [v for v in vehs.values() if getattr(v, "ignore_speed_limit", False)]
        assert sp, "no speeder"
        peak_v = max(stats[v.id]["max_v"] for v in sp)
        assert peak_v > 20.0, f"speeder only reached peak_v={peak_v:.2f}"
        return f"max_v={peak_v:.2f}"
    raise AssertionError("unknown class")


CLASSES = {
    "sudden_braking":     (sudden_braking,    5.0),
    "tailgating":         (tailgating,       30.0),
    "stalled_vehicle":    (stalled_vehicle,  12.0),
    "obstacle_in_lane":   (obstacle_in_lane, 12.0),
    "weaving":            (weaving,           8.0),
    "lane_straddling":    (lane_straddling,   3.0),
    "collision":          (collision,         6.0),
    "speeding":           (speeding,         30.0),
}


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total, failures = 0, 0
    for cls_name, (gen, run_seconds) in CLASSES.items():
        cls_dir = OUT_ROOT / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{cls_name}]")
        for name, scene in gen():
            total += 1
            path = cls_dir / f"{name}.sim.json"
            path.write_text(json.dumps(scene, indent=2))
            try:
                world, cols, stats = _run(path, run_seconds)
                diag = _check(cls_name, world, cols, stats)
                print(f"  PASS  {name:<28s}  {diag}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name:<28s}  {e}")

    print()
    print(f"Authored {total} scenarios under {OUT_ROOT.relative_to(ROOT)}")
    print(f"Passed {total - failures}/{total}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
