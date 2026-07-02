"""Tests for simstudio.kalman — Kalman filter and RMSE utilities."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simstudio.kalman import (
    LegacyKalmanFilter,
    build_kalman_rmse,
    build_kalman_rows,
    build_kalman_rows_tracked,
)


# ---------------------------------------------------------------------------
# LegacyKalmanFilter — basic state / covariance sanity checks
# ---------------------------------------------------------------------------

def test_initial_state_is_zero():
    kf = LegacyKalmanFilter(dt=0.1)
    assert list(kf.x.reshape(-1)) == [0.0] * 6


def test_initial_covariance_is_diagonal():
    kf = LegacyKalmanFilter(dt=0.1)
    import numpy as np
    off_diag = kf.P - np.diag(np.diag(kf.P))
    assert (abs(off_diag) < 1e-9).all()


def test_initialize_state_sets_position():
    kf = LegacyKalmanFilter(dt=0.1)
    kf.initialize_state(x=10.0, y=20.0)
    assert abs(float(kf.x.reshape(-1)[0]) - 10.0) < 1e-9
    assert abs(float(kf.x.reshape(-1)[3]) - 20.0) < 1e-9


def test_initialize_sets_initialized_flag():
    kf = LegacyKalmanFilter(dt=0.1)
    assert not kf._initialized
    kf.initialize_state(0.0, 0.0)
    assert kf._initialized


def test_predict_advances_position():
    kf = LegacyKalmanFilter(dt=0.1)
    kf.initialize_state(x=0.0, y=0.0, vx=10.0, vy=0.0)
    kf.predict(dt=1.0)
    # After 1 s at 10 m/s, x should be ~10 m.
    assert abs(float(kf.x.reshape(-1)[0]) - 10.0) < 1e-6


def test_predict_increases_covariance():
    import numpy as np
    kf = LegacyKalmanFilter(dt=0.1)
    kf.initialize_state(0.0, 0.0)
    p_before = float(kf.P[0, 0])
    kf.predict(dt=0.1)
    assert float(kf.P[0, 0]) >= p_before


def test_update_reduces_uncertainty():
    import numpy as np
    kf = LegacyKalmanFilter(dt=0.1)
    kf.initialize_state(0.0, 0.0, pos_sigma=10.0)
    p_before = float(kf.P[0, 0])
    kf.update_positions([("p_gps", 0.5, 0.5, 1.0, 1.0)])
    assert float(kf.P[0, 0]) < p_before


def test_update_moves_estimate_towards_measurement():
    kf = LegacyKalmanFilter(dt=0.1)
    kf.initialize_state(x=0.0, y=0.0)
    kf.update_positions([("p_gps", 100.0, 0.0, 1.0, 1.0)])
    # Estimate should move towards the measurement.
    assert float(kf.x.reshape(-1)[0]) > 0.0


def test_covariance_stays_positive_definite():
    import numpy as np
    kf = LegacyKalmanFilter(dt=0.1)
    kf.initialize_state(0.0, 0.0)
    for i in range(30):
        kf.predict(dt=0.1)
        kf.update_positions([("p_gps", float(i), 0.0, 1.5, 0.9)])
    eigenvalues = np.linalg.eigvalsh(kf.P)
    assert (eigenvalues > 0).all(), f"Non-positive eigenvalue: {eigenvalues}"


def test_rebuild_matrices_updates_dt():
    kf = LegacyKalmanFilter(dt=0.1)
    kf._rebuild_matrices(0.5)
    assert abs(kf.dt - 0.5) < 1e-9


def test_empty_position_update_is_noop():
    kf = LegacyKalmanFilter(dt=0.1)
    kf.initialize_state(5.0, 5.0)
    x_before = kf.x.copy()
    kf.update_positions([])
    assert (kf.x == x_before).all()


# ---------------------------------------------------------------------------
# build_kalman_rmse
# ---------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self, topic: str, payload: dict):
        self.topic = topic
        self.payload = payload


def _make_straight_run_events(n_steps: int = 60, dt: float = 1.0 / 30.0):
    """Simulate a vehicle driving straight at 10 m/s; produce truth + GPS events."""
    events = []
    for i in range(n_steps):
        t = i * dt
        x_true = 10.0 * t
        y_true = 0.0
        events.append(_FakeEvent("world.vehicle_state", {
            "t": t, "vehicle_id": "v0",
            "x": x_true, "y": y_true,
            "v": 10.0, "heading_rad": 0.0,
            "ax_world_mps2": 0.0, "ay_world_mps2": 0.0,
        }))
        events.append(_FakeEvent("sensor.gps", {
            "t": t, "vehicle_id": "v0",
            "x": x_true + 0.5, "y": y_true + 0.3,
            "sigma_m": 1.5, "confidence": 0.9,
            "x_true": x_true, "y_true": y_true,
            "speed_mps": 10.0, "heading_rad": 0.0,
            "sensor_id": "g0",
        }))
    return events


def test_build_kalman_rmse_returns_finite():
    events = _make_straight_run_events()
    rmse_x, rmse_y, rmse_pos, n = build_kalman_rmse(events)
    assert n > 0
    assert rmse_x is not None and math.isfinite(rmse_x)
    assert rmse_y is not None and math.isfinite(rmse_y)
    assert rmse_pos is not None and math.isfinite(rmse_pos)


def test_build_kalman_rmse_position_error_reasonable():
    """RMSE should be well below 10 m for a clean GPS track."""
    events = _make_straight_run_events(n_steps=120)
    _, _, rmse_pos, _ = build_kalman_rmse(events)
    assert rmse_pos < 10.0, f"RMSE too large: {rmse_pos:.2f} m"


def test_build_kalman_rmse_empty_returns_none():
    rmse_x, rmse_y, rmse_pos, n = build_kalman_rmse([])
    assert n == 0
    assert rmse_x is None and rmse_y is None and rmse_pos is None


def test_build_kalman_rmse_no_truth_returns_none():
    # Only sensor events, no truth — RMSE cannot be computed.
    events = [_FakeEvent("sensor.gps", {
        "t": 0.0, "vehicle_id": "v0",
        "x": 5.0, "y": 0.0, "sigma_m": 1.0, "confidence": 0.9,
        "x_true": 5.0, "y_true": 0.0,
        "speed_mps": 0.0, "heading_rad": 0.0, "sensor_id": "g0",
    })]
    _, _, _, n = build_kalman_rmse(events)
    assert n == 0


# ---------------------------------------------------------------------------
# Phase 4 — build_kalman_rows_tracked (shadow implementation)
#
# These tests verify that the tracker-driven analogue of build_kalman_rows
# is (a) a drop-in for Phase 5 (schema match) and (b) functionally identical
# to the oracle-driven path in Phase 1 (1:1 track ↔ vehicle).  They are
# deliberately tight: if the shadow ever drifts, the test suite should
# reject the change rather than silently normalise it.
# ---------------------------------------------------------------------------


def _make_two_vehicle_events(n_steps: int = 40, dt: float = 1.0 / 30.0):
    """Two vehicles on non-overlapping positions, each with GPS + truth.

    vA drives east along y=0;  vB drives east along y=50 (offset).
    Both emit truth and GPS at every step, and they share timestamps.
    The two tracks must stay completely separate in the tracker-driven
    row stream (no cross-measurement contamination).
    """
    events = []
    for i in range(n_steps):
        t = i * dt
        # Vehicle A
        xa = 10.0 * t
        events.append(_FakeEvent("world.vehicle_state", {
            "t": t, "vehicle_id": "vA",
            "x": xa, "y": 0.0,
            "v": 10.0, "heading_rad": 0.0,
            "ax_world_mps2": 0.0, "ay_world_mps2": 0.0,
        }))
        events.append(_FakeEvent("sensor.gps", {
            "t": t, "vehicle_id": "vA",
            "x": xa + 0.4, "y": 0.2,
            "sigma_m": 1.5, "confidence": 0.9,
            "x_true": xa, "y_true": 0.0,
            "speed_mps": 10.0, "heading_rad": 0.0,
            "sensor_id": "gA",
        }))
        # Vehicle B (different y so any mix-up would be obvious)
        xb = 8.0 * t + 20.0
        events.append(_FakeEvent("world.vehicle_state", {
            "t": t, "vehicle_id": "vB",
            "x": xb, "y": 50.0,
            "v": 8.0, "heading_rad": 0.0,
            "ax_world_mps2": 0.0, "ay_world_mps2": 0.0,
        }))
        events.append(_FakeEvent("sensor.gps", {
            "t": t, "vehicle_id": "vB",
            "x": xb - 0.3, "y": 50.25,
            "sigma_m": 1.5, "confidence": 0.9,
            "x_true": xb, "y_true": 50.0,
            "speed_mps": 8.0, "heading_rad": 0.0,
            "sensor_id": "gB",
        }))
    return events


def test_tracked_rows_schema_is_superset_of_legacy():
    """Phase 7: the tracked row is a 13-column superset of the legacy 12.

    Legacy  :  (t, vid,            sources, x, y, vx, vy, ax, ay, σp, σv, err)
    Tracked :  (t, gid, vid_oracle, sources, x, y, vx, vy, ax, ay, σp, σv, err)

    Stripping the ``global_track_id`` column (index 1) from the tracked
    output must yield exactly the legacy schema — the existing Phase 5
    GUI contract and the XLSX export path rely on that structure.
    """
    events = _make_straight_run_events(n_steps=20)
    legacy = build_kalman_rows(events)
    tracked = build_kalman_rows_tracked(events)
    assert legacy and tracked, "both paths must produce at least one row"
    assert len(legacy[0]) == 12, "legacy schema must remain 12 columns"
    assert len(tracked[0]) == 13, "tracked schema must be 13 columns (Phase 7: gid prepended)"
    # Every column is a string in both schemas.
    for col_idx in range(12):
        assert isinstance(legacy[0][col_idx], str)
    for col_idx in range(13):
        assert isinstance(tracked[0][col_idx], str)
    # gid column (index 1) must be a non-empty string — the tracker
    # assigns a stable identifier, never an empty sentinel.
    assert tracked[0][1]


def _strip_gid(tracked_rows):
    """Drop the Phase 7 ``global_track_id`` column so tracked rows can be
    compared bit-for-bit with the 12-column legacy output."""
    return [r[:1] + r[2:] for r in tracked_rows]


def test_tracked_single_vehicle_converges_to_legacy_exactly():
    """On a one-vehicle scenario the tracker is 1:1 with the oracle, so
    the two row streams must match bit-for-bit — after stripping the
    tracker-only ``global_track_id`` column from the tracked output.

    This is the core convergence claim that survived Phase 7: the
    grouping indirection does not change any KF behaviour.  Drift here
    means a bug in the grouping logic or the KF state reset.
    """
    events = _make_straight_run_events(n_steps=60)
    legacy = build_kalman_rows(events)
    tracked = build_kalman_rows_tracked(events)
    stripped = _strip_gid(tracked)
    assert stripped == legacy, (
        f"tracker-driven rows (gid-stripped) must equal legacy rows on "
        f"single-vehicle input; diff at first mismatch: "
        f"legacy={legacy[:2]}  stripped={stripped[:2]}"
    )
    # Every tracked row must carry a non-empty gid — the tracker does
    # assign an identity to every surviving track on this scene.
    assert all(r[1] for r in tracked)


def test_tracked_two_vehicles_no_contamination():
    """Two distinct vehicles ⇒ two distinct tracks ⇒ two disjoint row
    groups.  A measurement from vA must never influence vB's Kalman
    state, and vice-versa.  We verify this two ways:

    1. The tracker-driven rows (gid-stripped) equal the legacy rows
       bit-for-bit.
    2. Each per-vehicle sub-stream — keyed on the Phase 7 oracle column
       (``vehicle_id_oracle``, index 2 in the tracked row) — matches the
       per-vehicle sub-stream of the legacy path for that same vehicle.
    3. gid values are distinct between the two vehicles (i.e. the
       tracker did not merge them into a single track).
    """
    events = _make_two_vehicle_events(n_steps=40)
    legacy = build_kalman_rows(events)
    tracked = build_kalman_rows_tracked(events)

    # Bit-exact equality of the 12-col projection — Phase 1 is 1:1.
    assert _strip_gid(tracked) == legacy

    # Split tracked rows by the oracle column (index 2 post-Phase-7).
    def _split_by_oracle(rows):
        out = {}
        for r in rows:
            out.setdefault(r[2], []).append(r)
        return out

    def _split_by_vid_legacy(rows):
        out = {}
        for r in rows:
            out.setdefault(r[1], []).append(r)
        return out

    tracked_by_oracle = _split_by_oracle(tracked)
    legacy_by_vid     = _split_by_vid_legacy(legacy)
    assert set(tracked_by_oracle.keys()) == {"vA", "vB"}
    assert _strip_gid(tracked_by_oracle["vA"]) == legacy_by_vid["vA"]
    assert _strip_gid(tracked_by_oracle["vB"]) == legacy_by_vid["vB"]
    # Same number of updates per vehicle.
    assert len(tracked_by_oracle["vA"]) == len(tracked_by_oracle["vB"]) > 0

    # gid uniqueness: every row for vA must carry the same gid, every
    # row for vB must carry the same gid, and those two gids must differ.
    gids_a = {r[1] for r in tracked_by_oracle["vA"]}
    gids_b = {r[1] for r in tracked_by_oracle["vB"]}
    assert len(gids_a) == 1 and len(gids_b) == 1
    assert gids_a != gids_b, "tracker must not merge distinct vehicles into one gid"


def test_tracked_handles_generator_input():
    """Callers may pass a generator.  The function materialises it
    internally (one list() call), so a second pass works correctly.

    Phase 7: row width is 13 (gid prepended after t).
    """
    def _gen():
        for ev in _make_straight_run_events(n_steps=10):
            yield ev
    rows = build_kalman_rows_tracked(_gen())
    assert rows and all(isinstance(r, tuple) and len(r) == 13 for r in rows)


# ---------------------------------------------------------------------------
# Phase 8 — identity & safety tests: no oracle vehicle_id in fusion
#
# These tests prove that the Kalman/export pipeline uses global_track_id as
# the primary identity key, and that vehicle_id is relegated to the clearly-
# labelled vehicle_id_oracle column only.  They are deliberately independent
# of any GUI or tkinter import.
# ---------------------------------------------------------------------------


def test_global_track_id_is_primary_key_not_oracle_vid():
    """Column 1 of every tracked row must be a tracker-generated id (T######),
    never the oracle vehicle_id.  Column 2 must be the oracle id, clearly
    separated.  This proves the Kalman output uses real tracker identity.
    """
    events = _make_straight_run_events(n_steps=30)
    rows = build_kalman_rows_tracked(events)
    assert rows, "must produce at least one row"
    for row in rows:
        gid    = row[1]   # global_track_id column
        oracle = row[2]   # vehicle_id_oracle column
        # Tracker-generated ids follow the T###### format.
        assert gid.startswith("T"), (
            f"global_track_id column must be a tracker id (T…), got: {gid!r}"
        )
        # The primary identity key must never equal the oracle vid —
        # they are conceptually different objects.
        assert gid != oracle, (
            f"global_track_id must differ from vehicle_id_oracle; "
            f"got gid={gid!r} oracle={oracle!r}"
        )
        # Oracle column must carry the original vehicle_id from the events.
        assert oracle == "v0", (
            f"vehicle_id_oracle should be 'v0' (the oracle id), got {oracle!r}"
        )


def test_kalman_tracked_sensor_events_without_vehicle_id():
    """Kalman grouping is driven by the tracker's geometric association —
    NOT by oracle vehicle_id.  Sensor events without a vehicle_id field
    must still produce correctly tracked Kalman rows.

    Ground-truth events (world.vehicle_state) keep vehicle_id so the
    tracker can produce hypothesis bookkeeping and RMSE lookup.  Only
    sensor payloads lose the field, which is the realistic future state
    of the pipeline (Phase 3+).
    """
    events: list = []
    dt = 1.0 / 30.0
    for i in range(20):
        t = i * dt
        x_true = 10.0 * t
        # Ground-truth stream: oracle vehicle_id retained (used for RMSE only).
        events.append(_FakeEvent("world.vehicle_state", {
            "t": t, "vehicle_id": "v0",
            "x": x_true, "y": 0.0,
            "v": 10.0, "heading_rad": 0.0,
            "ax_world_mps2": 0.0, "ay_world_mps2": 0.0,
        }))
        # Sensor event deliberately omits vehicle_id — tracker must associate
        # purely by geometry.
        events.append(_FakeEvent("sensor.gps", {
            "t": t,
            # No vehicle_id key.
            "x": x_true + 0.3, "y": 0.05,
            "sigma_m": 1.5, "confidence": 0.9,
            "x_true": x_true, "y_true": 0.0,
            "speed_mps": 10.0, "heading_rad": 0.0,
            "sensor_id": "g0",
        }))

    rows = build_kalman_rows_tracked(events)
    assert rows, (
        "tracked rows must be produced even when sensor events carry no vehicle_id"
    )
    # Every row must have a valid tracker-generated gid — the recorder
    # captured assignments by (source, x, y) during Pass 1, so the fallback
    # oracle-lookup branch must never have been needed.
    for row in rows:
        assert row[1].startswith("T"), (
            f"expected T-prefixed gid (tracker-generated), got {row[1]!r}"
        )
    # There must be exactly one track — one vehicle, one identity.
    gids = {row[1] for row in rows}
    assert len(gids) == 1, (
        f"expected 1 track for a single-vehicle run, got {len(gids)}: {gids}"
    )


def test_cross_sensor_camera_das_camera_single_track():
    """Sensor-fusion continuity: Camera → DAS → Camera on a single vehicle
    must produce exactly one global_track_id throughout.

    This is the core sensor-fusion claim of the project: the tracker must
    preserve identity even when the measurement source changes mid-route.
    A new track must NOT be created when the source switches.

    Layout:
      Phase 1 (t = 0 … 1 s)  — Camera only
      Phase 2 (t = 1 … 2 s)  — DAS only   (Camera dropped out)
      Phase 3 (t = 2 … 3 s)  — Camera again
    """
    events: list = []
    dt = 1.0 / 30.0
    speed = 10.0  # m/s, constant velocity east

    def _truth(t: float) -> _FakeEvent:
        return _FakeEvent("world.vehicle_state", {
            "t": t, "vehicle_id": "v0",
            "x": speed * t, "y": 0.0,
            "v": speed, "heading_rad": 0.0,
            "ax_world_mps2": 0.0, "ay_world_mps2": 0.0,
        })

    # Phase 1: Camera only (30 frames).
    for i in range(30):
        t = i * dt
        x = speed * t
        events.append(_truth(t))
        events.append(_FakeEvent("sensor.camera", {
            "t": t, "vehicle_id": "v0",
            "x": x + 0.20, "y": 0.10,
            "sigma_m": 1.3, "confidence": 0.9,
            "x_true": x, "y_true": 0.0,
            "speed_mps": speed, "heading_rad": 0.0,
            "sensor_id": "cam0",
        }))

    # Phase 2: DAS only (30 frames).
    for i in range(30):
        t = 1.0 + i * dt
        x = speed * t
        events.append(_truth(t))
        events.append(_FakeEvent("sensor.das", {
            "t": t, "vehicle_id": "v0",
            "x": x - 0.10, "y": 0.0,
            "sigma_m": 2.0, "confidence": 0.8,
            "x_true": x, "y_true": 0.0,
            "speed_mps": speed, "heading_rad": 0.0,
            "sensor_id": "das0",
            "snr": 15.0, "fiber_angle_rad": 0.0,
            "z_vx": speed, "z_vy": 0.0, "sigma_v": 0.5,
        }))

    # Phase 3: Camera again (30 frames).
    for i in range(30):
        t = 2.0 + i * dt
        x = speed * t
        events.append(_truth(t))
        events.append(_FakeEvent("sensor.camera", {
            "t": t, "vehicle_id": "v0",
            "x": x + 0.15, "y": 0.05,
            "sigma_m": 1.3, "confidence": 0.9,
            "x_true": x, "y_true": 0.0,
            "speed_mps": speed, "heading_rad": 0.0,
            "sensor_id": "cam0",
        }))

    rows = build_kalman_rows_tracked(events)
    assert rows, "must produce Kalman rows for a 3-phase multi-sensor run"

    # All rows must share exactly one global_track_id.
    gids = {row[1] for row in rows}
    assert len(gids) == 1, (
        f"Camera→DAS→Camera must keep exactly 1 global_track_id throughout; "
        f"got {len(gids)}: {gids}"
    )

    # Both camera and DAS contributions must be visible in the sources column.
    # mtype "p_camera" → "camera";  "p_fiber" → "fiber"
    all_sources: set = set()
    for row in rows:
        all_sources.update(row[3].split("+"))
    assert "camera" in all_sources, (
        "camera measurements must appear in at least one Kalman row"
    )
    assert "fiber" in all_sources, (
        "DAS (fiber) measurements must appear in at least one Kalman row"
    )


def test_segment_transition_does_not_split_track():
    """A vehicle moving from one road segment to an adjacent one must keep
    the same global_track_id — no false identity split at the boundary.

    The tracker's cost function applies a segment-jump penalty only to
    non-adjacent segments; same-or-adjacent transitions must associate
    normally.  Even without a world graph (no adjacency info), the
    Euclidean proximity of the measurement must win over the penalty and
    keep the single track alive.
    """
    events: list = []
    dt = 1.0 / 30.0
    speed = 5.0  # m/s

    # Phase 1: vehicle on segment s1 (20 frames).
    for i in range(20):
        t = i * dt
        x = speed * t
        events.append(_FakeEvent("world.vehicle_state", {
            "t": t, "vehicle_id": "v0",
            "x": x, "y": 0.0,
            "v": speed, "heading_rad": 0.0,
            "ax_world_mps2": 0.0, "ay_world_mps2": 0.0,
        }))
        events.append(_FakeEvent("sensor.gps", {
            "t": t, "vehicle_id": "v0",
            "x": x + 0.10, "y": 0.05,
            "sigma_m": 1.5, "confidence": 0.9,
            "x_true": x, "y_true": 0.0,
            "speed_mps": speed, "heading_rad": 0.0,
            "sensor_id": "g0",
            "segment_id": "s1",
        }))

    # Phase 2: vehicle has crossed into adjacent segment s2 (20 frames).
    for i in range(20):
        t = (20 + i) * dt
        x = speed * t
        events.append(_FakeEvent("world.vehicle_state", {
            "t": t, "vehicle_id": "v0",
            "x": x, "y": 0.0,
            "v": speed, "heading_rad": 0.0,
            "ax_world_mps2": 0.0, "ay_world_mps2": 0.0,
        }))
        events.append(_FakeEvent("sensor.gps", {
            "t": t, "vehicle_id": "v0",
            "x": x + 0.10, "y": 0.05,
            "sigma_m": 1.5, "confidence": 0.9,
            "x_true": x, "y_true": 0.0,
            "speed_mps": speed, "heading_rad": 0.0,
            "sensor_id": "g0",
            "segment_id": "s2",   # transitioned to adjacent segment
        }))

    rows = build_kalman_rows_tracked(events)
    assert rows, "must produce Kalman rows across a segment transition"

    # Must be exactly one global_track_id — no false split at s1→s2 boundary.
    gids = {row[1] for row in rows}
    assert len(gids) == 1, (
        f"Segment s1→s2 transition must not create a new track; "
        f"got {len(gids)} distinct gids: {gids}"
    )


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
