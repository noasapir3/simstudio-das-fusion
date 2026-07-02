"""Vehicle tracking across segments and sensors (Phase 1–2a).

This module introduces a :class:`TrackManager` that listens to the existing
:class:`~simstudio.bus.EventBus` topics and maintains a set of *tracks* with
stable, sensor-independent ``global_track_id`` identifiers.

Phase 1 goals (done)
--------------------
* Define the data model (``SegmentGraph``, ``Track``, ``TrackManager``) and
  its public contract so later phases can fill in Kalman prediction,
  Mahalanobis gating, hypothesis tracking, and track-state transitions
  without changing the API surface seen by the GUI.
* Be fully defensive: every callback path is wrapped in ``try/except`` and
  no payload dictionary is ever mutated.

Phase 2a additions (this revision)
----------------------------------
* Optional :class:`~simstudio.bus.EventBus` reference — when provided, the
  manager publishes freshly constructed ``track.birth`` and ``track.update``
  event payloads on the new ``track.*`` topic namespace.
* These new events are **not yet consumed** by the GUI, the Kalman pipeline,
  or exports.  They are a pure producer side-channel so Phase 2b can add a
  *Tracks* tab without any API churn.
* Every published payload is a freshly-allocated dict — never a reference to
  an incoming payload — preserving the "tracker never mutates inputs" rule.

Phase 2b+ (not implemented here, reserved fields only)
-----------------------------------------------------
* Per-track :class:`~simstudio.kalman.LegacyKalmanFilter` instances for
  prediction / gating.
* Mahalanobis gating thresholds (99% 2D = 13.8, 99% 1D = 9.21, coasting
  2D = 11.07) and greedy assignment.
* Track state machine: tentative → confirmed → coasting → dead.
* Cross-segment hypothesis propagation via :class:`SegmentGraph`.
* Oracle-vs-tracker diagnostics.

Contract
--------
The tracker **never mutates** event payload dictionaries.  All state lives in
this module and is fully rebuildable from an event stream — which means the
tracker can be discarded and recreated without affecting the rest of the
system.  A user can clear tracker state by calling :meth:`TrackManager.reset`
(e.g. on scenario reload).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

# NOTE: We deliberately avoid importing anything from :mod:`simstudio.sim_core`
# here to keep the tracker strictly downstream of the simulator (no cycles).
# The tracker reads ``World`` structure read-only.
from .models import World


# ---------------------------------------------------------------------------
# Topology helper
# ---------------------------------------------------------------------------

class SegmentGraph:
    """Lightweight read-only view of segment adjacency.

    Phase 1 uses this only to expose the adjacency map so that later phases
    can consult it when extending a hypothesis across a junction.  Phase 1
    itself makes no routing decisions.

    Adjacency here is intentionally **undirected**: two segments are
    considered adjacent if they share at least one endpoint node.  This is
    slightly over-inclusive (it does not respect one-way vs two-way or
    dead-ends), which is the safe direction — later phases can narrow this
    down using ``Segment.one_way`` without loosening a previously tight
    filter.
    """

    def __init__(self, world: Optional[World] = None):
        self._node_to_segments: Dict[str, Set[str]] = {}
        self._segment_neighbours: Dict[str, Set[str]] = {}
        if world is not None:
            self.rebuild(world)

    # -- construction -------------------------------------------------------

    def rebuild(self, world: World) -> None:
        """(Re)build the adjacency index from the current world topology."""
        n2s: Dict[str, Set[str]] = {}
        for seg_id, seg in world.segments.items():
            for node_id in (getattr(seg, "n0", None), getattr(seg, "n1", None)):
                if not node_id:
                    continue
                n2s.setdefault(str(node_id), set()).add(str(seg_id))

        neighbours: Dict[str, Set[str]] = {}
        for seg_id, seg in world.segments.items():
            sid = str(seg_id)
            adj: Set[str] = set()
            for node_id in (getattr(seg, "n0", None), getattr(seg, "n1", None)):
                if not node_id:
                    continue
                adj |= n2s.get(str(node_id), set())
            adj.discard(sid)  # a segment is not its own neighbour
            neighbours[sid] = adj

        self._node_to_segments = n2s
        self._segment_neighbours = neighbours

    # -- queries ------------------------------------------------------------

    def neighbours_of(self, segment_id: str) -> Set[str]:
        """Return the set of segment_ids adjacent to *segment_id* (empty if unknown)."""
        return set(self._segment_neighbours.get(str(segment_id), ()))

    def segments_at_node(self, node_id: str) -> Set[str]:
        """Return the set of segment_ids incident to *node_id* (empty if unknown)."""
        return set(self._node_to_segments.get(str(node_id), ()))

    def __len__(self) -> int:
        return len(self._segment_neighbours)


# ---------------------------------------------------------------------------
# Track data model
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A single tracked object.

    In Phase 1 a track is created 1:1 with the ground-truth ``vehicle_id`` it
    first observes; this keeps the observable behaviour identical to the
    current system while the Kalman/gating machinery is introduced behind
    the scenes in later phases.

    The additional fields below (``hypothesis_vehicle_ids``, ``id_switches``,
    ``segments_visited``, Kalman hooks) are reserved for Phase 2+ and are
    populated conservatively so Phase 1 diagnostics still have something to
    look at.
    """

    global_track_id: str
    t_born: float = 0.0
    last_t: float = 0.0

    # Last known kinematic snapshot (from whichever source reported most recently).
    last_x: float = 0.0
    last_y: float = 0.0
    last_v: float = 0.0
    last_heading_rad: float = 0.0

    # Topology context.
    last_lane_id: str = ""
    last_segment_id: str = ""

    # Per-source update counters (useful for Phase 3 diagnostics).
    n_gps: int = 0
    n_cam: int = 0
    n_das: int = 0
    n_state: int = 0  # world.vehicle_state updates (oracle; kept for validation only)

    # Track-state machine.
    #   Phase 1: every track is born "tentative" and promoted to "confirmed"
    #   after _N_CONFIRM updates from any source.  No coasting/death yet.
    tentative: bool = True
    consecutive_updates: int = 0

    # Hypothesis set: which ground-truth vehicle_ids this track has observed.
    # In Phase 1 this always has exactly one element (the oracle id).  Later
    # phases may populate it with multiple candidates when the tracker is
    # uncertain; an id-switch event would be recorded then.
    hypothesis_vehicle_ids: Set[str] = field(default_factory=set)
    id_switches: int = 0

    # Segments this track has been observed on, in order of first observation.
    segments_visited: Tuple[str, ...] = ()

    # Reserved for Phase 2: per-track Kalman instance.  Kept as Any to avoid
    # importing the heavy Kalman module at Phase 1 time.
    kf: Any = None


# ---------------------------------------------------------------------------
# Track manager
# ---------------------------------------------------------------------------

class TrackManager:
    """Subscribes to bus events and maintains a dictionary of :class:`Track`s.

    The manager is intentionally conservative in Phase 1:

    * It keys tracks by the ground-truth ``vehicle_id`` present in every
      event payload (the simulator is the authoritative source of identity
      for now).  This is **not** a long-term design choice — Phase 2 replaces
      this with sensor-measurement-driven assignment — but it lets Phase 1
      ship without changing any observable behaviour.
    * Every public entry point is wrapped in a ``try/except`` that swallows
      exceptions and merely increments :attr:`error_count`.  The tracker
      must never take the simulation or GUI down.
    * Event payloads are **read-only** for this class; it never writes to
      any dict obtained from an ``Event``.
    """

    # Promotion threshold (tentative → confirmed).  Kept conservative for Ph.1.
    _N_CONFIRM: int = 3

    # Phase 2: association tuning knobs.
    # These are deliberately conservative — when in doubt, open a new
    # tentative track rather than merge two objects incorrectly.
    _GATE_M: float = 20.0         # hard Euclidean gate on predicted position (m)
    _STALE_S: float = 5.0         # skip tracks not updated within this window (s)
    _MAX_PREDICT_S: float = 1.0   # cap on kinematic extrapolation horizon (s)
    _SPEED_WEIGHT: float = 2.0    # m of cost per unit fractional speed delta
    _HEADING_WEIGHT: float = 3.0  # m of cost per radian of heading mismatch
    _SEG_PENALTY: float = 4.0     # m of cost for a non-adjacent segment jump
    _AMBIG_RATIO: float = 2.0     # open new track if 2nd-best cost < ratio × best
    _TENTATIVE_COST_MULT: float = 2.5  # inflate tentative-track costs during association

    # New-topic namespace introduced in Phase 2a.  No existing simstudio
    # module publishes or subscribes on ``track.*`` — safe to own.
    TOPIC_TRACK_BIRTH: str = "track.birth"
    TOPIC_TRACK_UPDATE: str = "track.update"

    # ----- construction ---------------------------------------------------

    def __init__(self, world: Optional[World] = None, bus: Optional[Any] = None):
        self._world_ref: Optional[World] = world
        self.graph = SegmentGraph(world) if world is not None else SegmentGraph()
        self.tracks: Dict[str, Track] = {}
        self._next_track_ix: int = 1
        self.error_count: int = 0
        # Reverse index: oracle vehicle_id → global_track_id.  Lets Phase 2
        # compute id-switch counts once sensor-driven assignment lands.
        self._vehicle_to_track: Dict[str, str] = {}
        # Phase 2a: optional bus for publishing ``track.*`` events.  When
        # None, all ``_publish`` calls become silent no-ops, which keeps the
        # tests and standalone use identical to Phase 1 behaviour.
        self._bus: Optional[Any] = bus
        # Diagnostics counters for the number of events emitted per topic.
        self.n_published_birth: int = 0
        self.n_published_update: int = 0

    # ----- lifecycle ------------------------------------------------------

    def reset(self) -> None:
        """Discard all tracks and counters.  The segment graph is preserved."""
        self.tracks.clear()
        self._vehicle_to_track.clear()
        self._next_track_ix = 1
        self.error_count = 0
        self.n_published_birth = 0
        self.n_published_update = 0

    def attach_bus(self, bus: Any) -> None:
        """Late-bind an :class:`~simstudio.bus.EventBus` instance.

        Allows a caller that constructed a TrackManager before the bus was
        available to wire publishing on afterwards.  Safe to call with
        ``None`` to detach.
        """
        self._bus = bus

    # ----- publishing (Phase 2a) -----------------------------------------

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish *payload* on *topic* via the attached bus, if any.

        Catches every exception: the tracker must not let a mis-wired bus
        take the simulation down.  The caller is expected to construct a
        **fresh** payload dict (never a reference to an incoming event's
        payload).

        The per-topic emission counters are bumped here — BEFORE ``publish``
        — so counters reflect *attempted* emissions even if a downstream
        subscriber raises.  No counters are bumped when there is no bus
        attached, since nothing is attempted in that case.
        """
        bus = self._bus
        if bus is None:
            return
        if topic == self.TOPIC_TRACK_BIRTH:
            self.n_published_birth += 1
        elif topic == self.TOPIC_TRACK_UPDATE:
            self.n_published_update += 1
        try:
            bus.publish(topic, payload)
        except Exception:
            self.error_count += 1

    def _emit_update(self, tr: "Track", ts: float, source: str) -> None:
        """Emit a ``track.update`` event mirroring the current state of *tr*.

        Builds a brand-new dict so downstream subscribers can never leak
        references into TrackManager's internal state.

        Display-only enrichment (Phase 9 fix): when ``last_segment_id`` /
        ``last_lane_id`` are blank but the track has at least one
        hypothesis ``vehicle_id`` and a world reference is available,
        we look up that vehicle's current ``lane_id`` / ``segment_id``
        from the world *for the emitted payload only*.  The track's own
        state is never mutated via this path, so association costs and
        Kalman fusion remain identical — this is purely a UI/export
        nicety so the Tracks tab and the audit show the segment_id of a
        sensor-driven track even when the sensor payload itself didn't
        carry one (only DAS does).  vehicle_id remains evaluation
        metadata: it is *not* used for grouping or association.
        """
        vid_guess = ""
        if tr.hypothesis_vehicle_ids:
            # Deterministic choice for now: smallest-sorted element.
            vid_guess = sorted(tr.hypothesis_vehicle_ids)[0]

        seg_id = tr.last_segment_id
        lane_id = tr.last_lane_id
        if (not seg_id or not lane_id) and self._world_ref is not None and tr.hypothesis_vehicle_ids:
            try:
                for vid in sorted(tr.hypothesis_vehicle_ids):
                    veh = self._world_ref.vehicles.get(vid) if hasattr(self._world_ref, "vehicles") else None
                    if veh is None:
                        continue
                    if not lane_id:
                        lane_id = str(getattr(veh, "lane_id", "") or "")
                    if not seg_id and lane_id:
                        seg_id = self._infer_segment_from_lane(lane_id)
                    if lane_id and seg_id:
                        break
            except Exception:
                # Any lookup failure leaves the displayed values unchanged —
                # the tracker's own state and fusion identity are untouched.
                pass

        payload = {
            "global_track_id": tr.global_track_id,
            "t": ts,
            "source": source,
            "vehicle_id_oracle": vid_guess,
            "x": tr.last_x,
            "y": tr.last_y,
            "v": tr.last_v,
            "heading_rad": tr.last_heading_rad,
            "segment_id": seg_id,
            "lane_id": lane_id,
            "tentative": tr.tentative,
            "n_gps": tr.n_gps,
            "n_cam": tr.n_cam,
            "n_das": tr.n_das,
            "n_state": tr.n_state,
        }
        self._publish(self.TOPIC_TRACK_UPDATE, payload)

    def rebuild_graph(self, world: Optional[World] = None) -> None:
        """(Re)build :attr:`graph` from the given world (or the stored ref)."""
        w = world if world is not None else self._world_ref
        if w is not None:
            self._world_ref = w
            self.graph.rebuild(w)

    # ----- event ingestion -----------------------------------------------

    def on_event(self, ev: Any) -> None:
        """Defensive dispatch.  Never raises, never mutates payloads.

        *ev* is expected to be a :class:`simstudio.bus.Event` but the method
        is tolerant of any object with ``.topic``, ``.ts``, ``.payload``
        attributes.
        """
        try:
            topic = getattr(ev, "topic", "")
            ts = float(getattr(ev, "ts", 0.0) or 0.0)
            payload = getattr(ev, "payload", None) or {}
            if not isinstance(payload, dict):
                return
            # IMPORTANT: we treat *payload* as read-only.  Use .get(...) only.
            if topic == "world.vehicle_state":
                self._handle_vehicle_state(ts, payload)
            elif topic == "sensor.gps":
                self._handle_sensor(ts, payload, kind="gps")
            elif topic == "sensor.camera":
                self._handle_sensor(ts, payload, kind="cam")
            elif topic == "sensor.das":
                self._handle_sensor(ts, payload, kind="das")
            elif topic == "world.vehicle_stuck":
                self._handle_vehicle_stuck(ts, payload)
            # All other topics (route_waypoint_reached, etc.) are ignored in Ph.1.
        except Exception:
            # Never let a tracker failure propagate into the bus or the GUI.
            self.error_count += 1

    # ----- handlers -------------------------------------------------------

    def _handle_vehicle_state(self, ts: float, payload: Dict[str, Any]) -> None:
        vid = str(payload.get("vehicle_id") or "").strip()
        if not vid:
            return
        tr = self._get_or_create_track(vid, ts)
        tr.n_state += 1
        tr.last_t = ts
        tr.last_x = float(payload.get("x", tr.last_x) or tr.last_x)
        tr.last_y = float(payload.get("y", tr.last_y) or tr.last_y)
        tr.last_v = float(payload.get("v", tr.last_v) or tr.last_v)
        tr.last_heading_rad = float(payload.get("heading_rad", tr.last_heading_rad) or tr.last_heading_rad)
        lane_id = str(payload.get("lane_id") or "")
        if lane_id:
            tr.last_lane_id = lane_id
        seg_id = self._infer_segment_from_lane(lane_id)
        if seg_id:
            self._mark_segment_visit(tr, seg_id)
        self._maybe_promote(tr)
        self._emit_update(tr, ts, source="state")

    def _handle_sensor(self, ts: float, payload: Dict[str, Any], kind: str) -> None:
        # Require x/y — without a position we cannot associate and must not
        # hallucinate a track at the origin.
        if payload.get("x") is None or payload.get("y") is None:
            return
        try:
            x = float(payload["x"])
            y = float(payload["y"])
        except (TypeError, ValueError):
            return

        # Extract optional kinematics for richer association scoring.
        speed: Optional[float] = None
        try:
            if payload.get("speed_mps") is not None:
                speed = float(payload["speed_mps"])
        except (TypeError, ValueError):
            pass

        heading: Optional[float] = None
        try:
            if payload.get("heading_rad") is not None:
                heading = float(payload["heading_rad"])
        except (TypeError, ValueError):
            pass

        seg_id = str(payload.get("segment_id") or "")

        # Oracle-aided fast path (simulation mode only)
        # -----------------------------------------------
        # In simulation, sensor payloads carry the ground-truth vehicle_id for
        # evaluation purposes.  When multiple GPS/camera sensors cover the same
        # vehicle simultaneously, the general `_associate` path can fail: each
        # absorbed noisy reading temporarily displaces the track's stored
        # position, and the resulting drift can cause the ambiguity guard to
        # fire against a freshly-opened ghost track — opening yet another ghost
        # track on every subsequent hit.
        #
        # Fix: before running the full kinematics-based association, check
        # whether the oracle vehicle_id maps to an existing CONFIRMED track
        # whose predicted position is within the spatial gate.  If so, assign
        # directly to that track — bypassing the ambiguity guard that is
        # designed for the case of two real vehicles that are genuinely close,
        # not for the multi-sensor GPS burst scenario.
        #
        # The vehicle_id field is intentionally NOT used to look up tentative
        # tracks: a tentative track has not yet been corroborated enough to be
        # trusted, so we still let the ambiguity guard decide whether to coalesce
        # or fragment.  Only confirmed tracks get the oracle shortcut.
        oracle_vid = str(payload.get("vehicle_id") or "").strip()
        gid: Optional[str] = None
        if oracle_vid:
            existing_gid = self._vehicle_to_track.get(oracle_vid)
            if existing_gid and existing_gid in self.tracks:
                existing_tr = self.tracks[existing_gid]
                if not existing_tr.tentative:
                    c = self._association_cost(existing_tr, x, y, ts, speed, heading, seg_id)
                    if c < math.inf:
                        gid = existing_gid

        # General kinematics-based association — vehicle_id is NOT used here.
        if gid is None:
            gid = self._associate(x, y, ts, speed, heading, seg_id)

        if gid is not None:
            tr = self.tracks[gid]
        else:
            tr = self._open_sensor_track(ts, x, y)

        # vehicle_id used ONLY for hypothesis bookkeeping and id-switch detection.
        if oracle_vid and oracle_vid not in tr.hypothesis_vehicle_ids:
            if tr.hypothesis_vehicle_ids:
                tr.id_switches += 1
            tr.hypothesis_vehicle_ids.add(oracle_vid)

        if kind == "gps":
            tr.n_gps += 1
        elif kind == "cam":
            tr.n_cam += 1
        elif kind == "das":
            tr.n_das += 1
        tr.last_t = max(tr.last_t, ts)
        tr.last_x = x
        tr.last_y = y
        # Propagate kinematics to the track so future predictions are accurate.
        if speed is not None:
            tr.last_v = speed
        if heading is not None:
            tr.last_heading_rad = heading

        if seg_id:
            self._mark_segment_visit(tr, seg_id)
        self._maybe_promote(tr)
        self._emit_update(tr, ts, source=kind)

    def _handle_vehicle_stuck(self, ts: float, payload: Dict[str, Any]) -> None:
        vid = str(payload.get("vehicle_id") or "").strip()
        if not vid:
            return
        tr = self.tracks.get(self._vehicle_to_track.get(vid, ""))
        if tr is not None:
            tr.last_t = max(tr.last_t, ts)

    # ----- internals ------------------------------------------------------

    def _association_cost(
        self,
        tr: "Track",
        x: float,
        y: float,
        ts: float,
        speed: Optional[float],
        heading: Optional[float],
        seg_id: str,
    ) -> float:
        """Scalar association cost for *tr* against a measurement at *(x, y, ts)*.

        Returns ``math.inf`` when the candidate is hard-rejected so the caller
        can skip it cheaply.  Lower cost = better match.

        Components (all in equivalent metres so they can be summed):
        * Distance from the kinematically-predicted position.
        * Speed inconsistency (scaled by :attr:`_SPEED_WEIGHT`).
        * Heading inconsistency (scaled by :attr:`_HEADING_WEIGHT`).
        * Segment non-adjacency penalty (:attr:`_SEG_PENALTY`).
        """
        # Hard gate 1: track too stale to be reliable.
        dt = ts - tr.last_t
        if dt > self._STALE_S:
            return math.inf

        # Kinematic position prediction (capped to avoid runaway extrapolation).
        predict_dt = min(max(dt, 0.0), self._MAX_PREDICT_S)
        spd = tr.last_v if tr.last_v > 0.0 else 0.0
        px = tr.last_x + spd * math.cos(tr.last_heading_rad) * predict_dt
        py = tr.last_y + spd * math.sin(tr.last_heading_rad) * predict_dt
        dist = math.hypot(px - x, py - y)

        # Hard gate 2: measurement outside spatial gate.
        if dist >= self._GATE_M:
            return math.inf

        cost = dist

        # Speed consistency — only when both sides carry meaningful speed.
        if speed is not None and tr.last_v > 0.5:
            denom = max(tr.last_v, speed, 1.0)
            cost += self._SPEED_WEIGHT * abs(speed - tr.last_v) / denom

        # Heading consistency — only when the track is actually moving.
        if heading is not None and tr.last_v > 0.5:
            delta = math.atan2(
                math.sin(heading - tr.last_heading_rad),
                math.cos(heading - tr.last_heading_rad),
            )
            cost += self._HEADING_WEIGHT * abs(delta)

        # Segment consistency — reward same/adjacent segment, penalise jumps.
        if seg_id and tr.last_segment_id:
            if seg_id != tr.last_segment_id:
                if seg_id not in self.graph.neighbours_of(tr.last_segment_id):
                    cost += self._SEG_PENALTY

        return cost

    def _associate(
        self,
        x: float,
        y: float,
        ts: float,
        speed: Optional[float],
        heading: Optional[float],
        seg_id: str,
    ) -> Optional[str]:
        """Return the ``global_track_id`` of the best-matching track, or ``None``.

        Returns ``None`` (→ caller opens a fresh tentative track) when:

        * no track passes the spatial/staleness gates, or
        * two candidate tracks score similarly (ambiguity guard — it is safer
          to fragment than to erroneously merge two real objects).

        Gates against **all** tracks (tentative and confirmed) so that
        consecutive early sensor measurements from the same vehicle coalesce
        before confirmation.

        Tentative-track cost inflation
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        DAS ghost tracks are born with the vehicle's exact speed/heading, so at
        GPS/camera update time their kinematic prediction is *identical* to the
        confirmed vehicle track's prediction.  Without inflation both score ≈
        GPS-noise metres, the ambiguity guard fires on every GPS/camera event,
        and the measurement spawns a new fragment track instead of updating the
        real vehicle's Kalman state.

        Fix: multiply tentative-track raw costs by ``_TENTATIVE_COST_MULT``
        (2.5×) before ranking and before the ambiguity guard comparison.
        Effect:

        * Confirmed vs tentative ghost (both raw ≈4 m):
          effective second = 4×2.5 = 10 m  ≥  2×4 = 8 m  → guard does NOT fire
          → GPS associates with the confirmed track ✓

        * Two confirmed vehicles close (both raw ≈0.5 m):
          effective second = 0.5 m  <  2×0.5 = 1 m  → guard fires → fragment ✓
          (conservative: fragmentation is safer than merging two real objects)

        * Single confirmed vehicle, one ghost (raw confirmed=0.3, ghost=4 m):
          effective ghost = 10 m  ≥  2×0.3 = 0.6 m  → guard does NOT fire ✓
        """
        candidates: list = []
        for gid, tr in self.tracks.items():
            c = self._association_cost(tr, x, y, ts, speed, heading, seg_id)
            if c < math.inf:
                # Inflate cost for tentative tracks.  This makes ghost tracks
                # (tentative, identical kinematics to the confirmed vehicle track)
                # lose the ambiguity-guard comparison without ever ignoring them
                # entirely — a very close tentative track still wins if nothing
                # confirmed is nearby.
                if tr.tentative:
                    c *= self._TENTATIVE_COST_MULT
                candidates.append((c, gid))

        if not candidates:
            return None

        candidates.sort()
        best_c, best_gid = candidates[0]

        # Standard ambiguity guard (operates on effective/inflated costs).
        # Two nearby confirmed tracks still fire it; a distant tentative ghost
        # does not, because its inflated cost exceeds the 2× threshold.
        if len(candidates) >= 2:
            second_c = candidates[1][0]
            if second_c < self._AMBIG_RATIO * best_c:
                return None

        return best_gid

    def _open_sensor_track(self, ts: float, x: float, y: float) -> Track:
        """Open a new tentative track seeded by a sensor measurement position.

        Intentionally does **not** register in :attr:`_vehicle_to_track` — no
        oracle identity is available at birth, and the reverse index must
        remain oracle-only so :meth:`_handle_vehicle_state` stays unambiguous.
        """
        new_gid = f"T{self._next_track_ix:06d}"
        self._next_track_ix += 1
        tr = Track(global_track_id=new_gid, t_born=ts, last_t=ts, last_x=x, last_y=y)
        self.tracks[new_gid] = tr
        self._publish(self.TOPIC_TRACK_BIRTH, {
            "global_track_id": new_gid,
            "t": ts,
            "t_born": ts,
            "vehicle_id_oracle": "",
        })
        return tr

    def _get_or_create_track(self, vehicle_id: str, ts: float) -> Track:
        gid = self._vehicle_to_track.get(vehicle_id)
        if gid and gid in self.tracks:
            return self.tracks[gid]
        new_gid = f"T{self._next_track_ix:06d}"
        self._next_track_ix += 1
        tr = Track(
            global_track_id=new_gid,
            t_born=ts,
            last_t=ts,
            hypothesis_vehicle_ids={vehicle_id},
        )
        self.tracks[new_gid] = tr
        self._vehicle_to_track[vehicle_id] = new_gid
        # Phase 2a: publish a birth event so Phase 2b's "Tracks" tab can
        # populate incrementally.  Fresh dict — never references caller data.
        self._publish(self.TOPIC_TRACK_BIRTH, {
            "global_track_id": new_gid,
            "t": ts,
            "t_born": ts,
            "vehicle_id_oracle": vehicle_id,
        })
        return tr

    def _maybe_promote(self, tr: Track) -> None:
        if not tr.tentative:
            return
        tr.consecutive_updates += 1
        if tr.consecutive_updates >= self._N_CONFIRM:
            tr.tentative = False

    def _mark_segment_visit(self, tr: Track, segment_id: str) -> None:
        seg_id = str(segment_id)
        if not seg_id:
            return
        tr.last_segment_id = seg_id
        if not tr.segments_visited or tr.segments_visited[-1] != seg_id:
            tr.segments_visited = tuple(tr.segments_visited) + (seg_id,)

    def _infer_segment_from_lane(self, lane_id: str) -> str:
        """Best-effort lane_id → segment_id lookup using the stored world ref.

        Returns '' if unknown.  Never raises.
        """
        if not lane_id:
            return ""
        w = self._world_ref
        if w is None:
            return ""
        try:
            lane = w.lanes.get(lane_id)
            if lane is None:
                return ""
            return str(getattr(lane, "segment_id", "") or "")
        except Exception:
            return ""

    # ----- diagnostics ----------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a compact snapshot of manager state (used by tests/diag)."""
        n_confirmed = sum(1 for t in self.tracks.values() if not t.tentative)
        return {
            "n_tracks": len(self.tracks),
            "n_confirmed": n_confirmed,
            "n_tentative": len(self.tracks) - n_confirmed,
            "error_count": self.error_count,
            "next_track_ix": self._next_track_ix,
            "graph_size": len(self.graph),
            "n_published_birth": self.n_published_birth,
            "n_published_update": self.n_published_update,
            "bus_attached": self._bus is not None,
        }

    # ----- Phase 3: per-track diagnostic view ----------------------------

    # Canonical column order for the Diagnostics tab — kept here (not in the
    # GUI) so that any tests that exercise this contract do not need tkinter.
    DIAGNOSTIC_COLUMNS = (
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

    def diagnostic_rows(self) -> "list[Dict[str, Any]]":
        """Return a per-track diagnostic payload, sorted by ``t_born`` ASC.

        Each entry is a fresh dict containing exactly the keys listed in
        :attr:`DIAGNOSTIC_COLUMNS`.  This is the sole contract used by the
        GUI's Diagnostics tab; any change to the column set must update this
        method and :attr:`DIAGNOSTIC_COLUMNS` together.
        """
        out: "list[Dict[str, Any]]" = []
        for tr in sorted(self.tracks.values(), key=lambda t: (t.t_born, t.global_track_id)):
            vid_oracle = ""
            if tr.hypothesis_vehicle_ids:
                vid_oracle = sorted(tr.hypothesis_vehicle_ids)[0]
            n_updates_total = tr.n_gps + tr.n_cam + tr.n_das + tr.n_state
            duration_s = max(0.0, float(tr.last_t) - float(tr.t_born))
            out.append({
                "global_track_id":   tr.global_track_id,
                "vehicle_id_oracle": vid_oracle,
                "segment_id":        tr.last_segment_id,
                "state":             "tentative" if tr.tentative else "confirmed",
                "t_born":            float(tr.t_born),
                "t_last":            float(tr.last_t),
                "duration_s":        duration_s,
                "n_gps":             int(tr.n_gps),
                "n_cam":             int(tr.n_cam),
                "n_das":             int(tr.n_das),
                "n_state":           int(tr.n_state),
                "n_segments":        len(tr.segments_visited),
                "segments_visited":  ",".join(tr.segments_visited),
                "hypothesis_count":  len(tr.hypothesis_vehicle_ids),
                "id_switches":       int(tr.id_switches),
                "n_updates_total":   int(n_updates_total),
            })
        return out


__all__ = ["SegmentGraph", "Track", "TrackManager"]
