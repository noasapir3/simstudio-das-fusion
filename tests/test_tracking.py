"""Tests for simstudio.tracking — Phase 1 skeleton.

These tests verify the *invisible-by-default* contract of :class:`TrackManager`:

* Tracks are created from events.
* No payload mutation occurs.
* Bad / missing payloads never raise.
* SegmentGraph adjacency reflects the world topology.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simstudio.bus import Event, EventBus
from simstudio.models import (
    LaneGeom,
    Node,
    Segment,
    World,
)
from simstudio.tracking import SegmentGraph, Track, TrackManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_world() -> World:
    """A 3-node, 2-segment world sharing node n2:  n1 --s1-- n2 --s2-- n3."""
    w = World()
    w.nodes["n1"] = Node(id="n1", x=0.0,   y=0.0)
    w.nodes["n2"] = Node(id="n2", x=100.0, y=0.0)
    w.nodes["n3"] = Node(id="n3", x=200.0, y=0.0)
    w.segments["s1"] = Segment(id="s1", n0="n1", n1="n2")
    w.segments["s2"] = Segment(id="s2", n0="n2", n1="n3")
    w.lanes["s1_fwd_lane1"] = LaneGeom(
        id="s1_fwd_lane1", segment_id="s1", offset_index=0,
        polyline=[(0.0, 0.0), (100.0, 0.0)],
    )
    w.lanes["s2_fwd_lane1"] = LaneGeom(
        id="s2_fwd_lane1", segment_id="s2", offset_index=0,
        polyline=[(100.0, 0.0), (200.0, 0.0)],
    )
    return w


def _ev(topic: str, ts: float, payload: dict) -> Event:
    return Event(topic=topic, ts=ts, payload=payload)


# ---------------------------------------------------------------------------
# SegmentGraph
# ---------------------------------------------------------------------------

def test_segment_graph_adjacency():
    w = _tiny_world()
    g = SegmentGraph(w)
    # s1 and s2 share node n2 → neighbours.
    assert "s2" in g.neighbours_of("s1")
    assert "s1" in g.neighbours_of("s2")
    # No self-loops.
    assert "s1" not in g.neighbours_of("s1")
    # Unknown segment is handled gracefully.
    assert g.neighbours_of("does-not-exist") == set()


def test_segment_graph_rebuild_idempotent():
    w = _tiny_world()
    g = SegmentGraph()
    g.rebuild(w)
    first = g.neighbours_of("s1")
    g.rebuild(w)
    assert g.neighbours_of("s1") == first


# ---------------------------------------------------------------------------
# TrackManager — basic ingestion
# ---------------------------------------------------------------------------

def test_track_manager_creates_track_on_vehicle_state():
    w = _tiny_world()
    tm = TrackManager(w)
    payload = {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 10.0, "y": 0.0, "v": 8.0, "heading_rad": 0.0}
    tm.on_event(_ev("world.vehicle_state", 0.1, payload))
    s = tm.summary()
    assert s["n_tracks"] == 1
    assert s["error_count"] == 0
    # segment inference from lane_id → segment_id works
    tr = next(iter(tm.tracks.values()))
    assert tr.last_segment_id == "s1"
    assert tr.n_state == 1
    assert tr.tentative is True  # not yet promoted


def test_track_manager_promotes_after_n_confirm():
    tm = TrackManager(_tiny_world())
    for i in range(TrackManager._N_CONFIRM):
        tm.on_event(_ev("world.vehicle_state", 0.01 * i,
                        {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": i, "y": 0.0}))
    tr = next(iter(tm.tracks.values()))
    assert tr.tentative is False
    assert tr.consecutive_updates >= TrackManager._N_CONFIRM


def test_track_manager_counts_per_sensor():
    tm = TrackManager(_tiny_world())
    tm.on_event(_ev("sensor.gps",    0.1, {"vehicle_id": "v1", "x": 1.0, "y": 0.0}))
    tm.on_event(_ev("sensor.camera", 0.2, {"vehicle_id": "v1", "x": 2.0, "y": 0.0}))
    tm.on_event(_ev("sensor.das",    0.3, {"vehicle_id": "v1", "x": 3.0, "y": 0.0, "segment_id": "s2"}))
    tr = next(iter(tm.tracks.values()))
    assert (tr.n_gps, tr.n_cam, tr.n_das) == (1, 1, 1)
    # DAS event payload carried segment_id='s2' → segments_visited reflects it.
    assert "s2" in tr.segments_visited
    assert tr.last_segment_id == "s2"


def test_track_manager_reuses_track_for_same_vehicle():
    tm = TrackManager(_tiny_world())
    tm.on_event(_ev("world.vehicle_state", 0.1, {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    tm.on_event(_ev("sensor.gps",          0.2, {"vehicle_id": "v1", "x": 1.0, "y": 0.0}))
    assert len(tm.tracks) == 1
    tm.on_event(_ev("world.vehicle_state", 0.3, {"vehicle_id": "v2", "lane_id": "s2_fwd_lane1"}))
    assert len(tm.tracks) == 2


# ---------------------------------------------------------------------------
# Defensive contract
# ---------------------------------------------------------------------------

def test_track_manager_does_not_mutate_payload():
    tm = TrackManager(_tiny_world())
    payload = {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 10.0, "y": 0.0}
    before = copy.deepcopy(payload)
    tm.on_event(_ev("world.vehicle_state", 0.1, payload))
    assert payload == before, "TrackManager must not mutate event payloads"


def test_track_manager_swallows_bad_payloads():
    tm = TrackManager(_tiny_world())
    # Missing vehicle_id → ignored.
    tm.on_event(_ev("world.vehicle_state", 0.1, {"lane_id": "s1_fwd_lane1"}))
    # None payload → shape-tolerant, must not raise.
    tm.on_event(Event(topic="sensor.gps", ts=0.2, payload=None))  # type: ignore[arg-type]
    # Unknown topic → silently ignored.
    tm.on_event(_ev("some.other.topic", 0.3, {"vehicle_id": "v9"}))
    # Non-dict payload → ignored.
    tm.on_event(Event(topic="sensor.das", ts=0.4, payload="not-a-dict"))  # type: ignore[arg-type]
    assert tm.summary()["error_count"] == 0
    assert tm.summary()["n_tracks"] == 0


def test_track_manager_reset_clears_state():
    tm = TrackManager(_tiny_world())
    tm.on_event(_ev("world.vehicle_state", 0.1, {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    tm.on_event(_ev("world.vehicle_state", 0.2, {"vehicle_id": "v2", "lane_id": "s2_fwd_lane1"}))
    assert tm.summary()["n_tracks"] == 2
    tm.reset()
    s = tm.summary()
    assert s["n_tracks"] == 0
    assert s["next_track_ix"] == 1
    # Graph is preserved after reset.
    assert s["graph_size"] == 2


# ---------------------------------------------------------------------------
# Bus integration
# ---------------------------------------------------------------------------

def test_track_manager_integrates_with_event_bus():
    """End-to-end: subscribing TrackManager to the real EventBus works and
    does not interfere with other subscribers receiving the same event."""
    tm = TrackManager(_tiny_world())
    bus = EventBus()

    other_calls = []
    bus.subscribe("world.vehicle_state", lambda ev: other_calls.append(ev.payload.get("vehicle_id")))
    bus.subscribe("world.vehicle_state", lambda ev: tm.on_event(ev))

    bus.publish("world.vehicle_state", {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 0.0, "y": 0.0})
    bus.publish("world.vehicle_state", {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 1.0, "y": 0.0})

    assert other_calls == ["v1", "v1"]
    assert tm.summary()["n_tracks"] == 1
    assert tm.summary()["error_count"] == 0


# ---------------------------------------------------------------------------
# Track dataclass defaults
# ---------------------------------------------------------------------------

def test_track_defaults_are_sane():
    t = Track(global_track_id="T000001")
    assert t.tentative is True
    assert t.n_gps == t.n_cam == t.n_das == t.n_state == 0
    assert t.hypothesis_vehicle_ids == set()
    assert t.segments_visited == ()
    assert t.kf is None


# ---------------------------------------------------------------------------
# Phase 2a: ``track.*`` publishing
# ---------------------------------------------------------------------------

def _collect(bus: EventBus) -> dict:
    """Subscribe capture-lists to every track.* topic on *bus*."""
    seen = {"track.birth": [], "track.update": []}
    bus.subscribe("track.birth",  lambda ev: seen["track.birth"].append(ev))
    bus.subscribe("track.update", lambda ev: seen["track.update"].append(ev))
    return seen


def test_no_publishing_without_bus():
    """Default (bus=None) behaviour stays identical to Phase 1."""
    tm = TrackManager(_tiny_world())  # no bus
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 0.0, "y": 0.0}))
    s = tm.summary()
    assert s["bus_attached"] is False
    assert s["n_published_birth"] == 0
    assert s["n_published_update"] == 0


def test_birth_emitted_exactly_once_per_track():
    bus = EventBus()
    seen = _collect(bus)
    tm = TrackManager(_tiny_world(), bus=bus)
    # Three events for the same vehicle should yield exactly ONE birth.
    for i in range(3):
        tm.on_event(_ev("world.vehicle_state", 0.1 * i,
                        {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": i, "y": 0.0}))
    assert len(seen["track.birth"]) == 1
    assert seen["track.birth"][0].payload["vehicle_id_oracle"] == "v1"
    assert seen["track.birth"][0].payload["global_track_id"].startswith("T")


def test_update_emitted_per_event():
    bus = EventBus()
    seen = _collect(bus)
    tm = TrackManager(_tiny_world(), bus=bus)
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 1.0, "y": 0.0, "v": 8.0}))
    tm.on_event(_ev("sensor.gps", 0.2,
                    {"vehicle_id": "v1", "x": 1.5, "y": 0.0}))
    tm.on_event(_ev("sensor.camera", 0.3,
                    {"vehicle_id": "v1", "x": 2.0, "y": 0.0}))
    tm.on_event(_ev("sensor.das", 0.4,
                    {"vehicle_id": "v1", "x": 2.5, "y": 0.0, "segment_id": "s2"}))
    # Four events of interest → four update emissions.
    assert len(seen["track.update"]) == 4
    sources = [ev.payload["source"] for ev in seen["track.update"]]
    assert sources == ["state", "gps", "cam", "das"]
    # Payload shape check on one representative event.
    last = seen["track.update"][-1].payload
    for key in ("global_track_id", "t", "vehicle_id_oracle", "x", "y",
                "segment_id", "lane_id", "tentative",
                "n_gps", "n_cam", "n_das", "n_state"):
        assert key in last


def test_published_payload_is_independent_of_input():
    """Mutating the TrackManager's state afterwards must not alter the
    already-published event payload."""
    bus = EventBus()
    seen = _collect(bus)
    tm = TrackManager(_tiny_world(), bus=bus)
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 1.0, "y": 0.0}))
    first_payload = seen["track.update"][0].payload
    first_x = first_payload["x"]
    # Second event advances the vehicle.
    tm.on_event(_ev("world.vehicle_state", 0.2,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 99.0, "y": 0.0}))
    # The captured first payload must not have been mutated.
    assert first_payload["x"] == first_x == 1.0


def test_published_payload_does_not_leak_input_reference():
    """TrackManager must not put the caller's payload dict onto the bus."""
    bus = EventBus()
    seen = _collect(bus)
    tm = TrackManager(_tiny_world(), bus=bus)
    input_payload = {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1", "x": 1.0, "y": 0.0}
    tm.on_event(_ev("world.vehicle_state", 0.1, input_payload))
    # The published update payload must be a distinct object.
    assert seen["track.update"][0].payload is not input_payload
    # Mutating the original input after publishing must not affect anything.
    input_payload["x"] = -999.0
    assert seen["track.update"][0].payload["x"] == 1.0


def test_bus_publishing_survives_broken_subscriber():
    """If a downstream subscriber raises, the tracker's error_count increments
    but on_event itself does not raise."""
    bus = EventBus()
    bus.subscribe("track.update", lambda ev: (_ for _ in ()).throw(RuntimeError("boom")))
    tm = TrackManager(_tiny_world(), bus=bus)
    # Should NOT raise.
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    # The raise happens inside _publish → counted in error_count.
    assert tm.summary()["error_count"] >= 1
    # And the track was still created.
    assert tm.summary()["n_tracks"] == 1


def test_attach_bus_late_binding():
    tm = TrackManager(_tiny_world())  # no bus at construction
    bus = EventBus()
    seen = _collect(bus)
    tm.attach_bus(bus)
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    assert len(seen["track.birth"]) == 1
    assert len(seen["track.update"]) == 1


def test_reset_clears_publish_counters():
    bus = EventBus()
    tm = TrackManager(_tiny_world(), bus=bus)
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    assert tm.summary()["n_published_birth"] == 1
    tm.reset()
    s = tm.summary()
    assert s["n_published_birth"] == 0
    assert s["n_published_update"] == 0


# ---------------------------------------------------------------------------
# Phase 2b: Tracks-tab contract
# ---------------------------------------------------------------------------

# Column keys the GUI's Tracks tab row-builder reads out of each
# ``track.update`` payload.  Kept in sync with `_populate_tracks_table` in
# ``simstudio.gui.app``.  If a column key is renamed on either side without
# updating the other, this test fails fast.
_TRACK_UPDATE_COLUMN_KEYS = (
    "t", "global_track_id", "vehicle_id_oracle", "source",
    "x", "y", "v", "segment_id", "lane_id", "tentative",
    "n_gps", "n_cam", "n_das", "n_state",
)


def test_track_update_payload_contains_every_tab_column():
    """Every column the Tracks tab renders must exist in each update payload."""
    bus = EventBus()
    seen = _collect(bus)
    tm = TrackManager(_tiny_world(), bus=bus)
    tm.on_event(_ev("world.vehicle_state", 0.1, {
        "vehicle_id": "v1", "lane_id": "s1_fwd_lane1",
        "x": 5.0, "y": 0.0, "v": 3.0, "heading_rad": 0.0,
    }))
    tm.on_event(_ev("sensor.gps", 0.2, {"vehicle_id": "v1", "x": 6.0, "y": 0.0}))
    for ev in seen["track.update"]:
        for key in _TRACK_UPDATE_COLUMN_KEYS:
            assert key in ev.payload, f"missing column key '{key}' in track.update payload"


def test_track_birth_payload_contains_vehicle_id_oracle():
    """The renamed key must also appear in track.birth so downstream
    diagnostics can cross-reference births with updates."""
    bus = EventBus()
    seen = _collect(bus)
    tm = TrackManager(_tiny_world(), bus=bus)
    tm.on_event(_ev("world.vehicle_state", 0.1, {
        "vehicle_id": "v1", "lane_id": "s1_fwd_lane1",
    }))
    assert seen["track.birth"]
    for ev in seen["track.birth"]:
        assert "vehicle_id_oracle" in ev.payload
        assert ev.payload["vehicle_id_oracle"] == "v1"


def test_track_update_source_values_are_restricted_set():
    """The 'source' column enumerates the known producers only."""
    bus = EventBus()
    seen = _collect(bus)
    tm = TrackManager(_tiny_world(), bus=bus)
    tm.on_event(_ev("world.vehicle_state", 0.1, {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    tm.on_event(_ev("sensor.gps", 0.2, {"vehicle_id": "v1", "x": 1.0, "y": 0.0}))
    tm.on_event(_ev("sensor.camera", 0.3, {"vehicle_id": "v1", "x": 2.0, "y": 0.0}))
    tm.on_event(_ev("sensor.das", 0.4, {"vehicle_id": "v1", "x": 3.0, "y": 0.0, "segment_id": "s2"}))
    allowed = {"state", "gps", "cam", "das"}
    for ev in seen["track.update"]:
        assert ev.payload["source"] in allowed


def test_ring_buffer_drops_oldest_when_full():
    """A bounded deque (the pattern used by the GUI) must preserve the
    newest events when overflowed — the tracker itself never blocks
    publishing, so dropping is the correct back-pressure policy."""
    from collections import deque
    buf: deque = deque(maxlen=5)
    bus = EventBus()
    bus.subscribe("track.update", lambda ev, _b=buf: _b.append(ev))
    tm = TrackManager(_tiny_world(), bus=bus)
    for i in range(20):
        tm.on_event(_ev("world.vehicle_state", 0.01 * i, {
            "vehicle_id": "v1", "lane_id": "s1_fwd_lane1",
            "x": float(i), "y": 0.0,
        }))
    assert len(buf) == 5  # bounded
    # The newest five events have x values 15..19.
    xs = [ev.payload["x"] for ev in buf]
    assert xs == [15.0, 16.0, 17.0, 18.0, 19.0]
    # Tracker's own state is independent of the buffer: it processed all 20.
    tr = next(iter(tm.tracks.values()))
    assert tr.n_state == 20


# ---------------------------------------------------------------------------
# Phase 3 — per-track diagnostic view
# ---------------------------------------------------------------------------

# The canonical column order agreed with the user: vehicle identifiers
# first, then current segment, then state, then time/duration fields, then
# per-source counters, then segment history, then hypothesis/id-switch
# indicators, and finally ``n_updates_total`` (lowest display priority).
_EXPECTED_DIAG_COLUMNS = (
    "global_track_id",
    "vehicle_id_oracle",
    "segment_id",
    "state",
    "t_born",
    "t_last",
    "duration_s",
    "n_gps",
    "n_cam",
    "n_das",
    "n_state",
    "n_segments",
    "segments_visited",
    "hypothesis_count",
    "id_switches",
    "n_updates_total",
)


def test_diagnostic_columns_contract():
    """DIAGNOSTIC_COLUMNS is the public contract the GUI relies on; any
    change must be intentional and test-visible."""
    assert TrackManager.DIAGNOSTIC_COLUMNS == _EXPECTED_DIAG_COLUMNS


def test_diagnostic_rows_empty_when_no_tracks():
    """Empty trackers must return an empty list (no placeholder rows,
    per the Phase 3 design agreement)."""
    tm = TrackManager(_tiny_world())
    assert tm.diagnostic_rows() == []


def test_diagnostic_rows_shape_matches_columns():
    """Every row is a dict whose keys are exactly DIAGNOSTIC_COLUMNS."""
    tm = TrackManager(_tiny_world())
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1",
                     "x": 1.0, "y": 0.0, "v": 5.0}))
    tm.on_event(_ev("sensor.gps", 0.2,
                    {"vehicle_id": "v1", "x": 2.0, "y": 0.0}))
    tm.on_event(_ev("sensor.das", 0.3,
                    {"vehicle_id": "v1", "x": 3.0, "y": 0.0, "segment_id": "s2"}))
    rows = tm.diagnostic_rows()
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == set(TrackManager.DIAGNOSTIC_COLUMNS)
    # Spot-check some values.
    assert row["vehicle_id_oracle"] == "v1"
    assert row["n_state"] == 1
    assert row["n_gps"] == 1
    assert row["n_das"] == 1
    assert row["n_updates_total"] == 3
    # duration_s is last_t - t_born, non-negative.
    assert row["duration_s"] >= 0.0
    # segments_visited is a comma-joined string; n_segments matches it.
    assert isinstance(row["segments_visited"], str)
    if row["segments_visited"]:
        assert row["n_segments"] == len(row["segments_visited"].split(","))
    # state is one of the two Phase-1 labels.
    assert row["state"] in ("tentative", "confirmed")


def test_diagnostic_rows_sorted_by_t_born_ascending():
    """Rows must come out oldest-first so the GUI can display them in
    birth order without further sorting."""
    tm = TrackManager(_tiny_world())
    # Deliberately create tracks out of t_born order.
    tm.on_event(_ev("world.vehicle_state", 5.0,
                    {"vehicle_id": "vB", "lane_id": "s1_fwd_lane1"}))
    tm.on_event(_ev("world.vehicle_state", 1.0,
                    {"vehicle_id": "vA", "lane_id": "s1_fwd_lane1"}))
    tm.on_event(_ev("world.vehicle_state", 3.0,
                    {"vehicle_id": "vC", "lane_id": "s1_fwd_lane1"}))
    rows = tm.diagnostic_rows()
    # Sorted by t_born ASC: vA (t=1) → vC (t=3) → vB (t=5), regardless
    # of the order in which the tracks were born.
    assert [r["vehicle_id_oracle"] for r in rows] == ["vA", "vC", "vB"]
    tbs = [r["t_born"] for r in rows]
    assert tbs == sorted(tbs)
    assert tbs == [1.0, 3.0, 5.0]


def test_diagnostic_rows_reflect_state_promotion():
    """Once a track passes _N_CONFIRM updates its ``state`` flips to
    'confirmed' in the diagnostic view."""
    tm = TrackManager(_tiny_world())
    # A single event leaves the track tentative.
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    rows = tm.diagnostic_rows()
    assert rows[0]["state"] == "tentative"
    # Pump enough updates to promote it.
    for i in range(TrackManager._N_CONFIRM):
        tm.on_event(_ev("world.vehicle_state", 0.2 + 0.01 * i,
                        {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    rows = tm.diagnostic_rows()
    assert rows[0]["state"] == "confirmed"


def test_diagnostic_rows_reset_clears_view():
    """After ``reset()`` the diagnostic view is empty again."""
    tm = TrackManager(_tiny_world())
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    assert len(tm.diagnostic_rows()) == 1
    tm.reset()
    assert tm.diagnostic_rows() == []


def test_diagnostic_rows_never_mutate_track_state():
    """Calling ``diagnostic_rows`` is a pure read: counters, set sizes,
    and the segments_visited tuple must be untouched afterwards."""
    tm = TrackManager(_tiny_world())
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1",
                     "x": 1.0, "y": 0.0}))
    tm.on_event(_ev("sensor.das", 0.2,
                    {"vehicle_id": "v1", "x": 2.0, "y": 0.0, "segment_id": "s2"}))
    tr = next(iter(tm.tracks.values()))
    snapshot = {
        "n_state": tr.n_state,
        "n_das": tr.n_das,
        "hyp": set(tr.hypothesis_vehicle_ids),
        "segs": tuple(tr.segments_visited),
        "last_segment_id": tr.last_segment_id,
        "tentative": tr.tentative,
    }
    _ = tm.diagnostic_rows()
    _ = tm.diagnostic_rows()
    _ = tm.diagnostic_rows()
    assert tr.n_state == snapshot["n_state"]
    assert tr.n_das == snapshot["n_das"]
    assert tr.hypothesis_vehicle_ids == snapshot["hyp"]
    assert tuple(tr.segments_visited) == snapshot["segs"]
    assert tr.last_segment_id == snapshot["last_segment_id"]
    assert tr.tentative == snapshot["tentative"]


def test_diagnostic_rows_row_dicts_are_independent_copies():
    """Mutating a returned row must not leak back into the tracker."""
    tm = TrackManager(_tiny_world())
    tm.on_event(_ev("world.vehicle_state", 0.1,
                    {"vehicle_id": "v1", "lane_id": "s1_fwd_lane1"}))
    rows_a = tm.diagnostic_rows()
    rows_a[0]["n_state"] = 9999
    rows_a[0]["segments_visited"] = "TAMPERED"
    rows_b = tm.diagnostic_rows()
    assert rows_b[0]["n_state"] != 9999
    assert rows_b[0]["segments_visited"] != "TAMPERED"


def test_ambiguity_guard_prevents_wrong_merge():
    """Two vehicles equidistant from a measurement must never be wrongly merged.

    Geometry:
        vA at (0, 0)        vB at (4, 0)
                  meas₁ at (2, 0)          ← equidistant; ambiguity fires
                  meas₂ at (0.3, 0)        ← clearly closer to vA; vA wins

    With _AMBIG_RATIO=2.0 and both candidates at cost=2.0:
        second_c (2.0) < _AMBIG_RATIO * best_c (4.0)  → True → new track opened.

    For meas₂ candidates are vA=0.3, new_track≈1.7, vB=3.7:
        second_c (1.7) < 2.0 * 0.3 (0.6) → False → vA wins, no new track.
    """
    tm = TrackManager(_tiny_world())

    # Seed oracle positions so the tracker knows where each vehicle sits.
    tm.on_event(_ev("world.vehicle_state", 0.0, {
        "vehicle_id": "vA", "x": 0.0, "y": 0.0,
        "v": 0.0, "heading_rad": 0.0, "lane_id": "s1_fwd_lane1",
    }))
    tm.on_event(_ev("world.vehicle_state", 0.0, {
        "vehicle_id": "vB", "x": 4.0, "y": 0.0,
        "v": 0.0, "heading_rad": 0.0, "lane_id": "s1_fwd_lane1",
    }))

    gid_A = tm._vehicle_to_track["vA"]
    gid_B = tm._vehicle_to_track["vB"]
    assert gid_A != gid_B, "precondition: two distinct tracks created"

    # ── Case 1: equidistant measurement → ambiguity guard fires → new track ──
    tm.on_event(_ev("sensor.gps", 0.0, {"x": 2.0, "y": 0.0}))

    assert tm.summary()["n_tracks"] == 3, (
        "equidistant GPS should open a 3rd tentative track, not merge with either"
    )
    assert tm.tracks[gid_A].n_gps == 0, "vA must not absorb the ambiguous measurement"
    assert tm.tracks[gid_B].n_gps == 0, "vB must not absorb the ambiguous measurement"

    # ── Case 2: clearly closer measurement → vA wins, no new track ──
    tm.on_event(_ev("sensor.gps", 0.1, {"x": 0.3, "y": 0.0}))

    assert tm.summary()["n_tracks"] == 3, (
        "unambiguous GPS close to vA should assign to vA, not open a 4th track"
    )
    assert tm.tracks[gid_A].n_gps == 1, "vA should receive the clearly-closer measurement"
    assert tm.tracks[gid_B].n_gps == 0, "vB must remain unaffected"


if __name__ == "__main__":
    import sys as _sys
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    _passed = _failed = 0
    for _fn in _tests:
        try:
            _fn()
            print(f"PASS  {_fn.__name__}")
            _passed += 1
        except Exception as _exc:
            print(f"FAIL  {_fn.__name__}: {_exc}")
            _failed += 1
    print(f"\n{_passed} passed, {_failed} failed")
    _sys.exit(0 if _failed == 0 else 1)
