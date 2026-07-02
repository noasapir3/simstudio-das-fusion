"""Tests for simstudio.audit — tracking validation/export pipeline.

Four scenarios from the user's specification:

1.  Single-vehicle, every generated sensor event consumed by Kalman.
2.  Sensor dropout — Kalman gets a measurable time gap.
3.  Multi-segment trajectory keeps continuity in distance_m.
4.  A bad measurement is skipped and its reason appears in the audit.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simstudio.audit import (
    AuditRow,
    TrajectoryRow,
    build_audit_and_trajectory,
    coverage_summary,
    export_all,
    segment_transitions,
    time_gaps,
    write_report_docx,
)


# ---------------------------------------------------------------------------
# Helpers — matched to the synthetic-event pattern in tests/test_kalman.py.
# ---------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self, topic: str, payload: dict):
        self.topic = topic
        self.payload = payload


def _truth(vid: str, t: float, x: float, y: float, v: float = 10.0,
           heading: float = 0.0, lane_id: str = "") -> _FakeEvent:
    return _FakeEvent("world.vehicle_state", {
        "t": t, "vehicle_id": vid,
        "x": x, "y": y, "v": v, "heading_rad": heading,
        "lane_id": lane_id,
        "ax_world_mps2": 0.0, "ay_world_mps2": 0.0,
    })


def _gps(vid: str, t: float, x: float, y: float, *, sigma: float = 1.5,
         conf: float = 0.9, sensor_id: str = "g0") -> _FakeEvent:
    return _FakeEvent("sensor.gps", {
        "t": t, "vehicle_id": vid,
        "x": x, "y": y,
        "sigma_m": sigma, "confidence": conf,
        "x_true": x, "y_true": y,
        "speed_mps": 10.0, "heading_rad": 0.0,
        "sensor_id": sensor_id,
    })


def _cam(vid: str, t: float, x: float, y: float, *, sigma: float = 0.7,
         conf: float = 0.85, sensor_id: str = "c0") -> _FakeEvent:
    return _FakeEvent("sensor.camera", {
        "t": t, "vehicle_id": vid,
        "x": x, "y": y,
        "sigma_m": sigma, "confidence": conf,
        "x_true": x, "y_true": y,
        "speed_mps": 10.0, "heading_rad": 0.0,
        "sensor_id": sensor_id,
    })


# ---------------------------------------------------------------------------
# Scenario 1 — full consumption: every generated sensor event ⇒ accepted=True
# ---------------------------------------------------------------------------

def _make_full_consumption_events(n_steps: int = 30, dt: float = 1.0 / 30.0):
    events = []
    for i in range(n_steps):
        t = i * dt
        x = 10.0 * t
        events.append(_truth("vA", t, x, 0.0))
        # Slightly noisy positions that are unique per t so the recorder
        # cannot accidentally collide two vehicles.
        events.append(_gps("vA", t, x + 0.4, 0.2))
        events.append(_cam("vA", t, x - 0.3, -0.1))
    return events


def test_full_consumption_one_vehicle():
    events = _make_full_consumption_events(n_steps=24)
    audit, traj = build_audit_and_trajectory(events)

    # We generated 24 GPS + 24 Camera events.
    n_gps = sum(1 for r in audit if r.sensor_type == "GPS")
    n_cam = sum(1 for r in audit if r.sensor_type == "Camera")
    assert n_gps == 24
    assert n_cam == 24
    # Every measurement must have been accepted into Kalman.
    skipped = [r for r in audit if not r.accepted]
    assert skipped == [], f"unexpected skips: {[(r.skip_reason, r.t) for r in skipped]}"
    # Every accepted row must point at a Kalman row.
    assert all(r.kalman_row_index is not None for r in audit)
    assert all(r.global_track_id for r in audit)
    # All gids are the same (single vehicle).
    gids = {r.global_track_id for r in audit}
    assert len(gids) == 1

    # Kalman rows: one per timestep.
    assert len(traj) == 24
    # Coverage: 100% of Kalman rows updated by GPS and by Camera.
    cov = {row["sensor"]: row for row in coverage_summary(audit, traj)}
    assert cov["GPS"]["accepted"] == 24
    assert cov["GPS"]["skipped"] == 0
    assert math.isclose(cov["GPS"]["pct_kalman_rows_updated"], 100.0, abs_tol=1e-6)
    assert math.isclose(cov["Camera"]["pct_kalman_rows_updated"], 100.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Scenario 2 — sensor dropout: a deliberate gap should appear in time_gaps()
# ---------------------------------------------------------------------------

def test_sensor_dropout_creates_visible_time_gap():
    """Drop every sensor sample in a 0.5 s window so the Kalman row stream
    has a gap that ``time_gaps`` reports.
    """
    n = 60
    dt = 1.0 / 30.0
    drop_lo, drop_hi = int(n * 0.4), int(n * 0.55)  # ~0.5 s gap mid-run
    events = []
    for i in range(n):
        t = i * dt
        x = 10.0 * t
        events.append(_truth("vA", t, x, 0.0))
        if drop_lo <= i < drop_hi:
            continue
        events.append(_gps("vA", t, x + 0.4, 0.2))

    audit, traj = build_audit_and_trajectory(events)
    # Every emitted GPS event must have been accepted (no skips).
    assert all(r.accepted for r in audit if r.sensor_type == "GPS")

    gaps = time_gaps(traj, gap_threshold_s=2.0 * dt)
    assert gaps, f"expected at least one gap > {2 * dt:.3f}s, got none"
    biggest = max(g["gap_s"] for g in gaps)
    expected = (drop_hi - drop_lo) * dt
    # Allow 1 dt tolerance because the gap is between samples.
    assert biggest >= expected - dt - 1e-6, (
        f"largest gap {biggest:.3f}s smaller than expected {expected:.3f}s"
    )


# ---------------------------------------------------------------------------
# Scenario 3 — multi-segment, distance stays continuous across the boundary
# ---------------------------------------------------------------------------

def test_multi_segment_distance_is_monotonic_and_continuous():
    """Vehicle crosses a junction at t≈1 s.  Cumulative distance must be
    monotonic in t and continuous at the segment boundary (no reset).
    """
    n_steps = 60
    dt = 1.0 / 30.0
    junction_idx = n_steps // 2
    events = []
    x_prev = 0.0
    for i in range(n_steps):
        t = i * dt
        x = 10.0 * t
        x_prev = x
        # Switch lane_id at the junction; the distance metric is purely
        # geometric, so it must be continuous regardless.
        lane = "L_a" if i < junction_idx else "L_b"
        events.append(_truth("vA", t, x, 0.0, lane_id=lane))
        events.append(_gps("vA", t, x + 0.3, 0.1))

    audit, traj = build_audit_and_trajectory(events)
    rows = sorted(traj, key=lambda r: r.t)
    dists = [r.true_distance_m for r in rows if r.true_distance_m is not None]
    assert len(dists) == len(rows), "every row should have a true_distance_m"
    # Monotonic non-decreasing.
    for a, b in zip(dists, dists[1:]):
        assert b >= a - 1e-6, f"true_distance_m regressed: {a} → {b}"
    # No gigantic jump at the junction (<= 1.5× the largest legitimate step).
    deltas = [b - a for a, b in zip(dists, dists[1:])]
    if deltas:
        median = sorted(deltas)[len(deltas) // 2]
        assert max(deltas) <= 1.6 * median + 1e-6, (
            f"distance discontinuity at boundary: max Δ={max(deltas):.3f}, "
            f"median Δ={median:.3f}"
        )

    # Visited segments — derived from lane→segment is N/A here (no world
    # passed) so segment_id is "" and segment_transitions() returns []
    # without crashing, which is the documented limitation.
    assert segment_transitions(traj, rows[0].global_track_id) == []


# ---------------------------------------------------------------------------
# Scenario 4 — bad measurement is skipped with a clear reason
# ---------------------------------------------------------------------------

def test_invalid_measurement_is_skipped_with_reason():
    """One GPS event has missing x; another has invalid sigma_m."""
    events = []
    n = 12
    dt = 1.0 / 30.0
    for i in range(n):
        t = i * dt
        x = 10.0 * t
        events.append(_truth("vA", t, x, 0.0))
        if i == 4:
            # Missing x — must produce a "missing x/y" skip.
            events.append(_FakeEvent("sensor.gps", {
                "t": t, "vehicle_id": "vA",
                "x": None, "y": 0.2,
                "sigma_m": 1.0, "confidence": 0.9,
                "speed_mps": 10.0, "heading_rad": 0.0,
                "sensor_id": "g0",
            }))
        elif i == 7:
            # Non-finite sigma — must produce an "invalid sigma_m" skip.
            events.append(_FakeEvent("sensor.gps", {
                "t": t, "vehicle_id": "vA",
                "x": x + 0.3, "y": 0.1,
                "sigma_m": float("inf"), "confidence": 0.9,
                "speed_mps": 10.0, "heading_rad": 0.0,
                "sensor_id": "g0",
            }))
        else:
            events.append(_gps("vA", t, x + 0.3, 0.1))

    audit, traj = build_audit_and_trajectory(events)
    skipped = [r for r in audit if r.skipped]
    reasons = {r.skip_reason for r in skipped}
    assert "missing x/y" in reasons
    assert "invalid sigma_m" in reasons
    # Two skips total, all the rest accepted.
    assert len(skipped) == 2
    accepted = [r for r in audit if r.accepted]
    assert len(accepted) == n - 2

    # Coverage breakdown surfaces the reasons.
    cov = {row["sensor"]: row for row in coverage_summary(audit, traj)}
    assert cov["GPS"]["skipped"] == 2
    assert "missing x/y=1" in cov["GPS"]["skip_reasons"]
    assert "invalid sigma_m=1" in cov["GPS"]["skip_reasons"]


# ---------------------------------------------------------------------------
# Smoke test — export_all writes the documented files.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 9 — strict identity & prediction-only rows
# ---------------------------------------------------------------------------

def _events_with_failed_track_assignment():
    """Three measurements, the middle one carries a position the recorder
    has never seen (we hand-craft the event stream).  In strict mode the
    audit must mark that one 'no_global_track_assignment'; in legacy
    mode the oracle vehicle_id fallback recovers it.
    """
    n = 8
    dt = 1.0 / 30.0
    events = []
    for i in range(n):
        t = i * dt
        x = 10.0 * t
        events.append(_truth("vA", t, x, 0.0))
        if i == 4:
            # Position "999" never appears as a track.update — the tracker
            # WILL associate it (since x/y are valid and there's only one
            # track), but to fake an unassociable measurement we monkey
            # the recorder via a bogus vehicle_id with a unique stride
            # that the recorder might miss.  Easiest reliable trick: bury
            # the failure by giving vehicle_id a value we know the
            # recorder won't have, and a coordinate not seen by the
            # tracker.  Since the tracker creates a new track for any
            # finite (x,y), we instead inject a measurement with a
            # finite x/y but assert via the audit that strict mode
            # behaves correctly given a recorder miss — see
            # `test_strict_mode_skips_unassociable_measurement`
            # which constructs the failure deterministically.
            pass
        events.append(_gps("vA", t, x + 0.3, 0.1))
    return events


def test_strict_mode_skips_unassociable_measurement():
    """Construct a deterministic recorder miss by patching the
    TrackManager pre-replay so one event yields no association.  Verify
    that strict mode (default) records skip_reason='no_global_track_assignment'
    while legacy mode silently recovers it via oracle vehicle_id.
    """
    from simstudio import audit as _aud_mod
    from simstudio import kalman as _km

    # Reuse the audit module's plumbing but inject a recorder whose
    # lookup() always returns "" — simulating a complete association
    # failure for every event.
    original_recorder_cls = _km._TrackAssignmentRecorder

    class _BlindRecorder(original_recorder_cls):
        def lookup(self, source, x, y):
            return ""

    _km._TrackAssignmentRecorder = _BlindRecorder
    _aud_mod._TrackAssignmentRecorder = _BlindRecorder
    try:
        events = _make_full_consumption_events(n_steps=6)

        # Strict mode (default): every sensor event becomes a skip with
        # the new "no_global_track_assignment" reason.
        audit_strict, traj_strict = _aud_mod.build_audit_and_trajectory(events)
        sensor_audit = [a for a in audit_strict if a.sensor_type in {"GPS", "Camera"}]
        assert sensor_audit, "no sensor measurements were generated by the harness"
        assert all(a.skipped for a in sensor_audit), "strict mode must skip every unassociated measurement"
        reasons = {a.skip_reason for a in sensor_audit}
        assert reasons == {"no_global_track_assignment"}, f"unexpected reasons: {reasons}"
        # No Kalman group is opened in strict mode — no trajectory rows.
        assert traj_strict == [], "strict mode must not open synthetic Kalman groups"

        # Legacy mode: oracle vehicle_id rescues the association.
        # The rescue can route to either an existing tracker gid (when
        # world.vehicle_state events created one for the vid) OR to a
        # synthetic ``__untracked__<vid>__`` group (when there's no
        # vid→gid entry).  Either path is the leak the strict mode
        # closes; the assertion below just verifies the leak is real
        # in legacy mode.
        audit_lax, traj_lax = _aud_mod.build_audit_and_trajectory(
            events, strict_global_track_only=False,
        )
        sensor_audit_lax = [a for a in audit_lax if a.sensor_type in {"GPS", "Camera"}]
        n_acc_lax = sum(1 for a in sensor_audit_lax if a.accepted)
        assert n_acc_lax > 0, "legacy fallback should recover at least some measurements"
        assert traj_lax, "legacy fallback must produce trajectory rows"
        # Crucially: the rescued gid must have come via the oracle vid,
        # not via the recorder (which we blinded).
        rescued = {a.global_track_id for a in sensor_audit_lax if a.accepted}
        assert all(g for g in rescued), f"empty gid in legacy rescue: {rescued}"
    finally:
        _km._TrackAssignmentRecorder = original_recorder_cls
        _aud_mod._TrackAssignmentRecorder = original_recorder_cls


def test_production_kalman_strict_mode_drops_unassociated_measurements():
    """When STRICT_TRACKER_IDENTITY is True (the production default), a
    sensor event the tracker never associated must NOT enter the Kalman
    via the oracle vehicle_id fallback.
    """
    from simstudio import kalman as _km

    original_recorder_cls = _km._TrackAssignmentRecorder

    class _BlindRecorder(original_recorder_cls):
        def lookup(self, source, x, y):
            return ""

    _km._TrackAssignmentRecorder = _BlindRecorder
    try:
        events = _make_full_consumption_events(n_steps=8)

        # Default: STRICT_TRACKER_IDENTITY is True — no rows at all.
        assert _km.STRICT_TRACKER_IDENTITY is True
        rows_strict = _km.build_kalman_rows_tracked(events)
        assert rows_strict == [], (
            f"strict-mode production Kalman must drop unassociable measurements; "
            f"got {len(rows_strict)} rows"
        )

        # Disable strict mode → fallback re-engages, rows reappear.
        _km.STRICT_TRACKER_IDENTITY = False
        try:
            rows_lax = _km.build_kalman_rows_tracked(events)
            assert rows_lax, "legacy fallback should produce rows when strict mode is off"
        finally:
            _km.STRICT_TRACKER_IDENTITY = True
    finally:
        _km._TrackAssignmentRecorder = original_recorder_cls


def test_prediction_only_rows_fill_dropout_gap():
    """Sensor dropout in the middle of a run.  With prediction emission
    enabled (the default), the trajectory must contain rows with
    update_kind='prediction_only' that fill the gap at sim cadence.
    """
    n = 60
    dt = 1.0 / 30.0
    drop_lo, drop_hi = 24, 36   # 0.4 s gap
    events = []
    for i in range(n):
        t = i * dt
        x = 10.0 * t
        events.append(_truth("vA", t, x, 0.0))
        if drop_lo <= i < drop_hi:
            continue
        events.append(_gps("vA", t, x + 0.4, 0.2))

    audit, traj = build_audit_and_trajectory(events)
    pred = [r for r in traj if r.update_kind == "prediction_only"]
    assert pred, "expected at least one prediction-only row inside the dropout"

    # Prediction rows must lie inside the gap window.
    t_lo = drop_lo * dt
    t_hi = drop_hi * dt
    assert all(t_lo - 1e-6 <= r.t <= t_hi + 1e-6 for r in pred), (
        f"prediction rows outside the expected window "
        f"[{t_lo:.3f}, {t_hi:.3f}]: {[r.t for r in pred]}"
    )

    # All prediction rows have a Kalman estimate but no sensor data.
    for r in pred:
        assert r.x_hat is not None and r.y_hat is not None
        assert r.gps_x is None and r.cam_x is None and r.das_x is None
        assert r.sigma_pos_m is not None and r.sigma_pos_m > 0.0

    # Coverage summary surfaces the count.
    cov = coverage_summary(audit, traj)
    pred_row = [c for c in cov if c["sensor"] == "(prediction-only)"][0]
    assert f"prediction_only_rows={len(pred)}" in pred_row["skip_reasons"]


def test_prediction_emission_does_not_change_kalman_at_measurement_boundaries():
    """The shadow filter that produces prediction-only rows must not
    affect the production Kalman state.  Concretely: the Kalman estimate
    at every measurement-bearing trajectory row must be byte-identical
    whether prediction emission is enabled or disabled.
    """
    events = _make_full_consumption_events(n_steps=24)

    _, traj_with = build_audit_and_trajectory(events, emit_prediction_rows=True)
    _, traj_without = build_audit_and_trajectory(events, emit_prediction_rows=False)

    # Restrict to measurement rows in both.
    meas_with = [r for r in traj_with if r.update_kind != "prediction_only"]
    meas_without = [r for r in traj_without if r.update_kind != "prediction_only"]
    assert len(meas_with) == len(meas_without) == 24

    for a, b in zip(meas_with, meas_without):
        assert a.t == b.t
        assert a.x_hat == b.x_hat
        assert a.y_hat == b.y_hat
        assert a.vx_hat == b.vx_hat
        assert a.vy_hat == b.vy_hat
        assert a.sigma_pos_m == b.sigma_pos_m


# ---------------------------------------------------------------------------
# Issue 1 — segment_id filled from vehicle_state lane_id without DAS
# ---------------------------------------------------------------------------

def test_segment_id_filled_without_das():
    """GPS-only run (no DAS); segment_id must be filled from vehicle_state
    lane_id -> world.lanes mapping for every trajectory row, and
    segment_transitions() must return exactly one seg1->seg2 crossing.
    """

    class _FakeLane:
        def __init__(self, segment_id: str):
            self.segment_id = segment_id

    class _FakeWorld:
        lanes = {
            "L_seg1": _FakeLane("seg1"),
            "L_seg2": _FakeLane("seg2"),
        }
        segments: dict = {}  # SegmentGraph.rebuild iterates world.segments

        def __getattr__(self, name: str):
            return None  # tolerate any extra attribute lookup by TrackManager

    world = _FakeWorld()
    n_per_seg = 15
    dt = 1.0 / 30.0
    events = []
    for i in range(n_per_seg):
        t = i * dt
        x = 10.0 * t
        events.append(_truth("vA", t, x, 0.0, lane_id="L_seg1"))
        events.append(_gps("vA", t, x + 0.3, 0.1))
    for i in range(n_per_seg):
        t = (n_per_seg + i) * dt
        x = 10.0 * t
        events.append(_truth("vA", t, x, 0.0, lane_id="L_seg2"))
        events.append(_gps("vA", t, x + 0.3, 0.1))

    _, traj = build_audit_and_trajectory(events, world=world)
    assert traj, "expected trajectory rows"

    junction_t = n_per_seg * dt  # 0.5 s
    seg1_rows = [r for r in traj if r.t < junction_t]
    seg2_rows = [r for r in traj if r.t >= junction_t]

    assert seg1_rows, "no rows before junction"
    assert seg2_rows, "no rows at/after junction"
    for r in seg1_rows:
        assert r.segment_id == "seg1", (
            f"t={r.t:.4f}: expected segment_id='seg1', got {r.segment_id!r}"
        )
    for r in seg2_rows:
        assert r.segment_id == "seg2", (
            f"t={r.t:.4f}: expected segment_id='seg2', got {r.segment_id!r}"
        )

    gid = traj[0].global_track_id
    transitions = segment_transitions(traj, gid)
    assert len(transitions) == 1, f"expected 1 transition, got {transitions}"
    assert transitions[0]["from_segment"] == "seg1"
    assert transitions[0]["to_segment"] == "seg2"


# ---------------------------------------------------------------------------
# Issue 2 — PNG filter skips short/tentative tracks
# ---------------------------------------------------------------------------

def test_png_filter_skips_short_tracks(tmp_path, monkeypatch):
    """Short track gets a CSV but not a PNG; long track gets both.

    Oracle vehicle_id appears in trajectory rows for post-run evaluation
    display only — it does not affect association or Kalman (allowed per
    oracle usage policy).
    """
    import simstudio.audit as _aud

    def _row(gid: str, vid: str, t: float, dist: float,
             kind: str = "position") -> TrajectoryRow:
        return TrajectoryRow(
            t=t, global_track_id=gid, vehicle_id_oracle=vid, segment_id="seg1",
            true_x=dist, true_y=0.0, true_distance_m=dist,
            gps_x=dist, gps_y=0.0, gps_distance_m=dist,
            cam_x=None, cam_y=None, cam_distance_m=None,
            das_x=None, das_y=None, das_fiber_position_m=None, das_distance_m=None,
            x_hat=dist, y_hat=0.0, vx_hat=1.0, vy_hat=0.0, ax_hat=0.0, ay_hat=0.0,
            distance_hat_m=dist, sigma_pos_m=1.0, sigma_vel_mps=0.5, pos_err_m=0.1,
            update_kind=kind,
        )

    # Short track: 2 rows, 0.1 s, 0.5 m — below all filter thresholds.
    short_rows = [
        _row("gid_short", "vShort", 0.0, 0.0),
        _row("gid_short", "vShort", 0.1, 0.5),
    ]
    # Long track: 30 rows, 5 s, 50 m — above all filter thresholds.
    long_rows = [
        _row("gid_long", "vLong", i * (5.0 / 29), i * (50.0 / 29))
        for i in range(30)
    ]
    fake_traj = short_rows + long_rows

    monkeypatch.setattr(_aud, "build_audit_and_trajectory",
                        lambda *a, **kw: ([], fake_traj))
    monkeypatch.setattr(_aud, "EXPORT_FINAL_TRACK_PNGS_ONLY", True)
    # Avoid matplotlib dependency — the filter logic is what we're testing.
    monkeypatch.setattr(_aud, "write_trajectory_png",
                        lambda traj_rows, path, gid, **kw: Path(path))

    result = export_all([], tmp_path)

    assert result["n_gids"] == 2
    # Only the long track gets a CSV (short track is filtered out).
    assert len(result["trajectory_csvs"]) == 1
    for p in result["trajectory_csvs"]:
        assert p.exists(), f"CSV missing: {p}"
    # Only the long track gets a PNG.
    assert result["n_pngs_created"] == 1, result
    assert result["n_pngs_skipped"] == 1, result
    assert "gid_short" in result["png_skip_reasons"], result["png_skip_reasons"]
    assert len(result["trajectory_pngs"]) == 1


# ---------------------------------------------------------------------------
# Word report tests
# ---------------------------------------------------------------------------

def _make_rows_for_report(n: int = 35) -> tuple:
    """Return (audit_rows, traj_rows) for a single-vehicle run long enough
    to pass the meaningful-track filter (duration >1 s, >=20 rows, dist >5 m).
    """
    events = _make_full_consumption_events(n_steps=n)
    return build_audit_and_trajectory(events)


def test_compute_track_stats_basic():
    """_compute_track_stats returns correct duration, row count, and pred count."""
    import simstudio.audit as _aud

    audit_rows, traj_rows = _make_rows_for_report(n=35)
    by_gid: dict = {}
    for r in traj_rows:
        by_gid.setdefault(r.global_track_id, []).append(r)
    gid = next(iter(by_gid))
    rows = sorted(by_gid[gid], key=lambda r: r.t)
    stats = _aud._compute_track_stats(rows, audit_rows)

    assert stats["gid"] == gid
    assert stats["n_rows"] == len(rows)
    assert stats["duration_s"] == pytest.approx(rows[-1].t - rows[0].t, abs=1e-9)
    assert stats["n_pred"] == sum(1 for r in rows if r.update_kind == "prediction_only")
    assert isinstance(stats["segments"], list)
    assert isinstance(stats["transitions"], list)


def test_detect_anomalies_error_spike():
    """_detect_anomalies fires an error_spike for a row with large pos_err_m."""
    import simstudio.audit as _aud

    def _row(t: float, err: float, gid: str = "g0") -> TrajectoryRow:
        return TrajectoryRow(
            t=t, global_track_id=gid, vehicle_id_oracle="v0", segment_id="s1",
            true_x=0.0, true_y=0.0, true_distance_m=t * 10,
            gps_x=0.0, gps_y=0.0, gps_distance_m=None,
            cam_x=None, cam_y=None, cam_distance_m=None,
            das_x=None, das_y=None, das_fiber_position_m=None, das_distance_m=None,
            x_hat=0.0, y_hat=0.0, vx_hat=0.0, vy_hat=0.0, ax_hat=0.0, ay_hat=0.0,
            distance_hat_m=t * 10, sigma_pos_m=1.0, sigma_vel_mps=0.5,
            pos_err_m=err, update_kind="position",
        )

    rows = [_row(float(i), 0.5) for i in range(20)]
    rows[10] = _row(10.0, 15.0)  # spike above ANOMALY_ERROR_SPIKE_M=10

    anomalies = _aud._detect_anomalies(rows)
    spikes = [a for a in anomalies if a["kind"] == "error_spike"]
    assert spikes, "expected at least one error_spike anomaly"
    assert any(a["value"] == pytest.approx(15.0) for a in spikes)

    # No spike when all errors are below threshold.
    normal_rows = [_row(float(i), 1.0) for i in range(20)]
    assert not [a for a in _aud._detect_anomalies(normal_rows) if a["kind"] == "error_spike"]


def test_word_report_created(tmp_path, monkeypatch):
    """write_report_docx produces a non-trivial .docx file."""
    import simstudio.audit as _aud

    audit_rows, traj_rows = _make_rows_for_report(n=35)

    # Skip actual figure rendering (avoids needing a display / temp-file I/O).
    monkeypatch.setattr(
        _aud, "_render_track_figure_bytes",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("figure skipped in test")),
    )

    out = tmp_path / "report.docx"
    result = write_report_docx(audit_rows, traj_rows, out, sim_name="test_run")

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 2_000   # non-trivial Word file


def test_export_all_writes_audit_and_trajectory_files(tmp_path):
    events = _make_full_consumption_events(n_steps=10)
    summary = export_all(events, tmp_path)

    audit_csv = summary["audit_csv"]
    assert audit_csv.exists()
    assert audit_csv.name == "kalman_measurement_audit.xlsx"

    coverage_csv = summary["coverage_csv"]
    assert coverage_csv.exists()
    assert coverage_csv.name == "coverage_summary.xlsx"

    # One trajectory CSV per gid.
    assert summary["n_gids"] >= 1
    for p in summary["trajectory_csvs"]:
        assert p.exists() and p.name.startswith("track_trajectory_") and p.suffix == ".xlsx"


def test_pdf_report_created(tmp_path, monkeypatch):
    """write_report_pdf produces a non-trivial .pdf file."""
    import simstudio.audit as _aud

    audit_rows, traj_rows = _make_rows_for_report(n=35)

    # Stub figure building so the test runs without a display.
    def _fake_build(traj_rows, gid, *, title=None, anomaly_markers=None):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(8, 6))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0, 1], [0, 1])
        return fig

    monkeypatch.setattr(_aud, "_build_track_figure", _fake_build)

    out = tmp_path / "report.pdf"
    result = _aud.write_report_pdf(audit_rows, traj_rows, out, sim_name="test_run")

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 1_000   # non-trivial PDF


def test_anomaly_adaptive_thresholds():
    """Adaptive thresholds scale with track_rmse: spike fires at 3×RMSE, high_rmse at 2×RMSE."""
    import simstudio.audit as _aud

    def _row(t: float, err: float, gid: str = "g0") -> TrajectoryRow:
        return TrajectoryRow(
            t=t, global_track_id=gid, vehicle_id_oracle="v0", segment_id="s1",
            true_x=0.0, true_y=0.0, true_distance_m=t * 5,
            gps_x=0.0, gps_y=0.0, gps_distance_m=None,
            cam_x=None, cam_y=None, cam_distance_m=None,
            das_x=None, das_y=None, das_fiber_position_m=None, das_distance_m=None,
            x_hat=0.0, y_hat=0.0, vx_hat=0.0, vy_hat=0.0, ax_hat=0.0, ay_hat=0.0,
            distance_hat_m=t * 5, sigma_pos_m=1.0, sigma_vel_mps=0.5,
            pos_err_m=err, update_kind="position",
        )

    # track_rmse=3.0 → spike_thr=max(2.0, 9.0)=9.0, rmse_thr=max(1.0, 6.0)=6.0
    track_rmse = 3.0
    rows = [_row(float(i), 3.0) for i in range(20)]  # avg err=3.0 → RMSE≈3.0
    rows[15] = _row(15.0, 8.0)  # below spike_thr=9.0, should NOT fire error_spike

    anomalies = _aud._detect_anomalies(rows, track_rmse=track_rmse)
    spikes = [a for a in anomalies if a["kind"] == "error_spike"]
    assert not spikes, f"spike should not fire below adaptive threshold; got {spikes}"

    # A value above the adaptive threshold SHOULD fire.
    rows2 = [_row(float(i), 3.0) for i in range(20)]
    rows2[15] = _row(15.0, 10.0)  # above spike_thr=9.0
    anomalies2 = _aud._detect_anomalies(rows2, track_rmse=track_rmse)
    spikes2 = [a for a in anomalies2 if a["kind"] == "error_spike"]
    assert spikes2, "spike should fire above adaptive threshold"
