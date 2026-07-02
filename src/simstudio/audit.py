"""Tracking validation / audit / trajectory exporter.

This module is purely additive: it does NOT modify the Kalman or tracker
implementations.  Instead it mirrors the production flow used by
``simstudio.kalman.build_kalman_rows_tracked`` and emits two parallel data
products that the existing pipeline never produced:

* :func:`build_audit_and_trajectory` returns

  - ``audit_rows`` — one :class:`AuditRow` per sensor measurement with the
    decision (accepted / skipped + reason), the resolved
    ``global_track_id``, the consumer Kalman timestep, and the kind of
    Kalman update it produced.

  - ``traj_rows`` — one :class:`TrajectoryRow` per ``(gid, t)`` Kalman
    batch carrying ground truth, raw sensor measurements, the Kalman
    estimate, and a continuous *distance traveled* coordinate computed
    from ``world.vehicle_state`` events (continuous across segment
    transitions).

Helper functions :func:`coverage_summary` and :func:`time_gaps`, plus CSV
and PNG writers, give the rest of the application a one-call path to the
required artefacts.

Design rules
------------
1. Mirror the *exact* skip / fallback ladder of
   :func:`simstudio.kalman.build_kalman_rows_tracked`.  Any difference is
   a bug in this module — the reference is the original.
2. Never mutate anything in the kalman or tracking modules.
3. Defensive everywhere — a malformed event must never raise; instead it
   becomes a skipped audit row with a clear reason.
"""

from __future__ import annotations

import bisect
import csv
import logging
import math

_log = logging.getLogger(__name__)
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Re-use the helpers that already exist in simstudio.kalman so this module
# stays in sync with the production Kalman pipeline.
from .kalman import (
    LegacyKalmanFilter,
    _Meas,
    _Truth,
    _TrackAssignmentRecorder,
    _collect_end_times,
    _collect_truth,
    _nearest_truth,
    _sensor_velocity_sigma,
)

_TOPIC_MTYPE: Dict[str, str] = {
    "sensor.gps":    "p_gps",
    "sensor.camera": "p_camera",
    "sensor.das":    "p_fiber",
}
_TOPIC_SOURCE: Dict[str, str] = {
    "sensor.gps":    "gps",
    "sensor.camera": "cam",
    "sensor.das":    "das",
}
_SENSOR_LABEL: Dict[str, str] = {
    "p_gps":    "GPS",
    "p_camera": "Camera",
    "p_fiber":  "DAS",
}

# ---------------------------------------------------------------------------
# PNG-export filter thresholds (Issue 2)
# ---------------------------------------------------------------------------

PNG_MIN_DURATION_S: float = 1.0
PNG_MIN_ROWS: int = 20
PNG_MIN_DISTANCE_M: float = 5.0
# Minimum number of Kalman timesteps that had a real sensor update (i.e.
# update_kind != "prediction_only").  Filters pure single-hit DAS ghosts
# (1 fiber hit + prediction coast) while keeping GPS-only tracks where a
# vehicle received just 2–3 sparse pings.  Set to 1 to disable.
PNG_MIN_SENSOR_HITS: int = 2
EXPORT_FINAL_TRACK_PNGS_ONLY: bool = True

# ---------------------------------------------------------------------------
# Word report constants
# ---------------------------------------------------------------------------

EXPORT_WORD_REPORT: bool = True
EXPORT_PDF_REPORT: bool = False   # Word report is the single deliverable
# Anomaly detection baselines — adaptive logic scales these up for noisy tracks.
# error spike fires if pos_err_m  > max(ANOMALY_ERROR_SPIKE_M,  3 × track_RMSE)
# high RMSE   fires if rolling_RMSE > max(ANOMALY_RMSE_M,       2 × track_RMSE)
ANOMALY_ERROR_SPIKE_M: float = 2.0
ANOMALY_RMSE_M: float = 1.0
ANOMALY_PRED_GAP_S: float = 1.0
ANOMALY_DIST_JUMP_M: float = 3.0
ANOMALY_DROPOUT_S: float = 0.5
ANOMALY_SIGMA_M: float = 3.0
ANOMALY_MAX_PER_TRACK: int = 5   # cap on anomalies shown per track in the report


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AuditRow:
    """One row per sensor measurement that *could have* entered Kalman."""

    t: float
    sensor_type: str          # "GPS" / "Camera" / "DAS"
    sensor_id: str
    vehicle_id_oracle: str
    global_track_id: str
    x: Optional[float]
    y: Optional[float]
    speed_mps: float
    heading_rad: float
    sigma_m: float
    sigma_v: float            # DAS only; 0 otherwise
    confidence: float
    snr: float                # DAS only; 0 otherwise
    segment_id: str            # from event payload; "" when unavailable
    accepted: bool
    skipped: bool
    skip_reason: str
    update_kind: str          # "position" | "position+velocity" | "skipped" | ""
    kalman_t: Optional[float]
    kalman_row_index: Optional[int]
    # ── DAS physics fields (DAS only; 0.0 / NaN otherwise) ──────────────────
    # These are read directly from the bus event payload published by sim_core.
    # They were previously discarded during audit construction; adding them here
    # enables exact weight back-calculation and spatial coverage analysis.
    das_amplitude: float      # A = W / (r + d0)^2 [raw amplitude units]
    fiber_distance_m: float   # r = perpendicular vehicle-to-fiber distance [m]
    snr_th: float             # SNR threshold used (traffic-level dependent)
    lateral_offset_m: float   # vehicle lateral offset from lane centerline [m]


@dataclass
class TrajectoryRow:
    """One row per ``(gid, t)`` event — usable for graphing.

    Two flavours exist:

    * ``update_kind ∈ {"position", "position+velocity"}`` — emitted at
      every measurement-bearing timestep; carries the raw sensor
      measurements that fed the Kalman update.
    * ``update_kind = "prediction_only"`` — emitted at simulation
      cadence inside time gaps where no sensor measurement was
      available.  All sensor fields are ``None``.  Produced by a
      *shadow* filter so the production Kalman state is unchanged.
    """

    t: float
    global_track_id: str
    vehicle_id_oracle: str
    segment_id: str

    true_x: Optional[float]
    true_y: Optional[float]
    true_distance_m: Optional[float]

    gps_x: Optional[float]
    gps_y: Optional[float]
    gps_distance_m: Optional[float]

    cam_x: Optional[float]
    cam_y: Optional[float]
    cam_distance_m: Optional[float]

    das_x: Optional[float]
    das_y: Optional[float]
    das_fiber_position_m: Optional[float]
    das_distance_m: Optional[float]

    x_hat: Optional[float]
    y_hat: Optional[float]
    vx_hat: Optional[float]
    vy_hat: Optional[float]
    ax_hat: Optional[float]
    ay_hat: Optional[float]

    distance_hat_m: Optional[float]
    sigma_pos_m: Optional[float]
    sigma_vel_mps: Optional[float]
    pos_err_m: Optional[float]

    # ── Kalman innovation (Phase 2 — populated by build_audit_and_trajectory) ─
    # Pre-update residual ν = z − H·x̂⁻ and innovation covariance diagonal.
    # NaN / None on prediction-only rows (no measurement, no innovation).
    # NIS_x = ν_x² / S_xx follows χ²(1) when the filter is consistent;
    # persistent NIS_x > 3.84 (95th-pct) is the CUSUM trigger signal.
    innovation_x:   Optional[float] = None   # x-position residual ν_x [m]
    innovation_y:   Optional[float] = None   # y-position residual ν_y [m]
    innovation_S_x: Optional[float] = None   # innovation variance S_xx [m²]
    innovation_S_y: Optional[float] = None   # innovation variance S_yy [m²]
    NIS_x:          Optional[float] = None   # ν_x² / S_xx  (χ²(1) statistic)

    update_kind: str = "measurement"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_audit_and_trajectory(
    events: Iterable,
    world: Any = None,
    *,
    strict_global_track_only: bool = True,
    emit_prediction_rows: bool = True,
    prediction_cadence_s: Optional[float] = None,
) -> Tuple[List[AuditRow], List[TrajectoryRow]]:
    """Build the audit + per-batch trajectory tables for *events*.

    Mirrors :func:`simstudio.kalman.build_kalman_rows_tracked` step by step:

    * Pass 1 — replay through TrackManager + recorder.
    * Pass 2 — assign each sensor event to a ``global_track_id`` and
      run the per-gid Kalman with the same skip rules; record the
      decision in :class:`AuditRow`, record the post-update state in
      :class:`TrajectoryRow`.

    Parameters
    ----------
    strict_global_track_only : bool, default ``True``
        When ``True`` (the production setting), measurements that the
        tracker / recorder failed to assign to a ``global_track_id`` are
        recorded in the audit as
        ``skip_reason="no_global_track_assignment"`` — the oracle
        ``vehicle_id`` is **never** used to recover the association.
        Set to ``False`` to reproduce the historical fall-back behaviour
        (oracle-vid → gid, then synthetic ``__untracked__<vid>__``).
        This flag is independent of the production-Kalman switch in
        :data:`simstudio.kalman.STRICT_TRACKER_IDENTITY`.
    emit_prediction_rows : bool, default ``True``
        When ``True``, additional :class:`TrajectoryRow`s with
        ``update_kind = "prediction_only"`` are emitted between
        consecutive measurement batches at *prediction_cadence_s*
        spacing.  These rows are produced by stepping a *shadow*
        :class:`~simstudio.kalman.LegacyKalmanFilter` synchronised with
        the production filter at every measurement boundary — so the
        production filter's state evolution is unchanged.  The shadow's
        per-step covariance during long gaps will differ slightly from
        a single monolithic ``predict(dt)`` because Q scales
        non-linearly with dt; the *mean* trajectory and the post-gap
        covariance at the next measurement match exactly.
    prediction_cadence_s : float | None, default ``None``
        Spacing of the prediction-only rows.  When ``None``, the median
        ``Δt`` of the per-vehicle ``world.vehicle_state`` stream is
        used; this matches the simulator's actual cadence.  Falls back
        to ``1/30 s`` if no truth stream is present.
    """
    events = list(events)

    # Pass 1 — tracker replay.
    from .tracking import TrackManager as _TM

    recorder = _TrackAssignmentRecorder()
    tm = _TM(world=world, bus=recorder)
    for ev in events:
        tm.on_event(ev)

    gid_to_vid: Dict[str, str] = {}
    vid_to_gid: Dict[str, str] = {}
    for gid, tr in tm.tracks.items():
        gid_s = str(gid)
        if tr.hypothesis_vehicle_ids:
            rep = sorted(tr.hypothesis_vehicle_ids)[0]
            gid_to_vid[gid_s] = str(rep)
        else:
            gid_to_vid[gid_s] = ""
        for vid in tr.hypothesis_vehicle_ids:
            vid_to_gid.setdefault(str(vid), gid_s)

    end_times = _collect_end_times(events)
    truths = _collect_truth(events)

    audit_rows: List[AuditRow] = []
    # Per-gid measurement queue, each item paired with its audit row so the
    # post-update pass can mark accepted=True without rebuilding.
    per_gid: Dict[str, List[Tuple[_Meas, AuditRow]]] = {}

    for ev in events:
        topic = getattr(ev, "topic", "")
        if topic not in _TOPIC_MTYPE:
            continue
        p = getattr(ev, "payload", None) or {}
        if not isinstance(p, dict):
            continue

        is_das = (topic == "sensor.das")
        sensor_label = _SENSOR_LABEL[_TOPIC_MTYPE[topic]]
        sensor_id = str(p.get("sensor_id", "") or "")
        vid_oracle = str(p.get("vehicle_id", "") or "")
        try:
            t = float(p.get("t", 0.0) or 0.0)
        except (TypeError, ValueError):
            t = 0.0

        x_raw = p.get("x")
        y_raw = p.get("y")
        speed_mps = _safe_float(p.get("speed_mps", 0.0))
        heading_rad = _safe_float(p.get("heading_rad", 0.0))
        sigma_m_raw = p.get("sigma_m", 2.0)
        # Detect a non-finite or non-numeric raw value *before* defaulting so
        # the "invalid sigma_m" skip path actually fires.  _safe_float would
        # otherwise silently fall back to the default and mask the problem.
        try:
            sigma_m_probe = float(sigma_m_raw) if sigma_m_raw is not None else 2.0
        except (TypeError, ValueError):
            sigma_m_probe = float("nan")
        sigma_m = sigma_m_probe
        sigma_v_raw = p.get("sigma_v", 0.0) if is_das else 0.0
        sigma_v = _safe_float(sigma_v_raw)
        confidence = _safe_float(p.get("confidence", 0.55), default=0.55)
        snr_val = _safe_float(p.get("snr", 0.0)) if is_das else 0.0
        # DAS physics fields — read from bus payload; zero for non-DAS sensors
        das_amplitude_val    = _safe_float(p.get("das_amplitude",    0.0)) if is_das else 0.0
        fiber_distance_val   = _safe_float(p.get("fiber_distance_m", 0.0)) if is_das else 0.0
        snr_th_val           = _safe_float(p.get("snr_th",           0.0)) if is_das else 0.0
        lateral_offset_val   = _safe_float(p.get("lateral_offset_m", 0.0)) if is_das else 0.0

        a = AuditRow(
            t=t,
            sensor_type=sensor_label,
            sensor_id=sensor_id,
            vehicle_id_oracle=vid_oracle,
            global_track_id="",
            x=None,
            y=None,
            speed_mps=speed_mps,
            heading_rad=heading_rad,
            sigma_m=sigma_m,
            sigma_v=sigma_v,
            confidence=confidence,
            snr=snr_val,
            segment_id=str(p.get("segment_id", "") or ""),
            accepted=False,
            skipped=False,
            skip_reason="",
            update_kind="",
            kalman_t=None,
            kalman_row_index=None,
            das_amplitude=das_amplitude_val,
            fiber_distance_m=fiber_distance_val,
            snr_th=snr_th_val,
            lateral_offset_m=lateral_offset_val,
        )

        # ---- skip ladder (same order as build_kalman_rows_tracked) ----

        if x_raw is None or y_raw is None:
            a.skipped = True
            a.skip_reason = "missing x/y"
            a.update_kind = "skipped"
            audit_rows.append(a)
            continue
        try:
            x_v = float(x_raw)
            y_v = float(y_raw)
        except (TypeError, ValueError):
            a.skipped = True
            a.skip_reason = "non-numeric x/y"
            a.update_kind = "skipped"
            audit_rows.append(a)
            continue
        a.x = x_v
        a.y = y_v

        source_kind = _TOPIC_SOURCE[topic]
        gid = recorder.lookup(source_kind, x_v, y_v)
        if not gid:
            # Strict mode (default, Phase 9): never recover association
            # via the oracle vehicle_id — the measurement is dropped from
            # fusion exactly like in the production Kalman path when
            # STRICT_TRACKER_IDENTITY is on.  vehicle_id_oracle remains in
            # the audit row as debug metadata only.
            if strict_global_track_only:
                a.skipped = True
                a.skip_reason = "no_global_track_assignment"
                a.update_kind = "skipped"
                audit_rows.append(a)
                continue
            # Legacy / diagnostic mode: oracle-vid fallback.
            if not vid_oracle:
                a.skipped = True
                a.skip_reason = "track association failed (no global_track_id, no vehicle_id fallback)"
                a.update_kind = "skipped"
                audit_rows.append(a)
                continue
            gid = vid_to_gid.get(vid_oracle, f"__untracked__{vid_oracle}")
            gid_to_vid.setdefault(gid, vid_oracle)
        a.global_track_id = str(gid)

        if vid_oracle and vid_oracle in end_times and t >= end_times[vid_oracle]:
            a.skipped = True
            a.skip_reason = "vehicle/track ended"
            a.update_kind = "skipped"
            audit_rows.append(a)
            continue

        if not (sigma_m > 0.0 and math.isfinite(sigma_m)):
            a.skipped = True
            a.skip_reason = "invalid sigma_m"
            a.update_kind = "skipped"
            audit_rows.append(a)
            continue
        if not (math.isfinite(x_v) and math.isfinite(y_v)):
            a.skipped = True
            a.skip_reason = "non-finite x/y"
            a.update_kind = "skipped"
            audit_rows.append(a)
            continue
        if not (0.0 <= confidence <= 1.0 and math.isfinite(confidence)):
            a.skipped = True
            a.skip_reason = "invalid confidence"
            a.update_kind = "skipped"
            audit_rows.append(a)
            continue

        m = _Meas(
            t=t,
            vehicle_id=gid_to_vid.get(gid, ""),
            mtype=_TOPIC_MTYPE[topic],
            x=x_v,
            y=y_v,
            sigma=sigma_m,
            confidence=confidence,
            x_true=(None if p.get("x_true") is None else _safe_float(p.get("x_true"))),
            y_true=(None if p.get("y_true") is None else _safe_float(p.get("y_true"))),
            speed_mps=speed_mps,
            heading_rad=heading_rad,
            source_id=sensor_id,
            snr=snr_val,
            fiber_angle_rad=_safe_float(p.get("fiber_angle_rad", 0.0)) if is_das else 0.0,
            z_vx=_safe_float(p.get("z_vx", 0.0)) if is_das else 0.0,
            z_vy=_safe_float(p.get("z_vy", 0.0)) if is_das else 0.0,
            sigma_v=sigma_v,
        )
        per_gid.setdefault(gid, []).append((m, a))
        audit_rows.append(a)

    # Sort per-gid measurements by time / type / source_id for batched updates.
    for gid in per_gid:
        per_gid[gid].sort(key=lambda pair: (pair[0].t, pair[0].mtype, pair[0].source_id))

    # Sort gids by representative oracle vid so output ordering matches
    # build_kalman_rows_tracked exactly on 1:1 scenarios.
    sorted_gids = sorted(per_gid.keys(), key=lambda g: gid_to_vid.get(g, g))

    # Determine the prediction cadence (only used when
    # emit_prediction_rows=True).  Inferred from world.vehicle_state Δt
    # so it matches the simulator's actual cadence.
    cadence = _infer_cadence_s(events) if (emit_prediction_rows and prediction_cadence_s is None) else prediction_cadence_s

    traj_rows: List[TrajectoryRow] = []

    for gid in sorted_gids:
        pairs = per_gid[gid]
        if not pairs:
            continue
        oracle_vid = gid_to_vid.get(gid, "")
        truth_list = truths.get(oracle_vid, [])

        kf = LegacyKalmanFilter(dt=1.0 / 30.0)
        i = 0
        last_t: Optional[float] = None
        while i < len(pairs):
            t = pairs[i][0].t
            batch: List[Tuple[_Meas, AuditRow]] = []
            while i < len(pairs) and abs(pairs[i][0].t - t) < 1e-9:
                batch.append(pairs[i])
                i += 1

            # Phase 9: emit prediction-only TrajectoryRows for the
            # interval (last_t, t) BEFORE processing this measurement
            # batch.  Uses a shadow filter cloned from kf so the
            # production filter's state evolution is untouched (kf will
            # still take its monolithic predict(dt) below).
            if (emit_prediction_rows and last_t is not None
                    and cadence is not None and cadence > 0
                    and t - float(last_t) > 1.5 * cadence):
                _emit_prediction_rows_for_gap(
                    traj_rows=traj_rows,
                    kf=kf,
                    gid=str(gid),
                    oracle_vid=oracle_vid,
                    truth_list=truth_list,
                    t_start=float(last_t),
                    t_end=t,
                    cadence=cadence,
                )

            truth = _nearest_truth(truth_list, t)
            if not kf._initialized:
                seed_m = batch[0][0]
                vx0 = seed_m.speed_mps * math.cos(seed_m.heading_rad)
                vy0 = seed_m.speed_mps * math.sin(seed_m.heading_rad)
                if truth is not None:
                    kf.initialize_state(truth.x, truth.y, vx=vx0, vy=vy0,
                                        ax=0.0, ay=0.0, pos_sigma=2.5)
                else:
                    y_init = 0.0 if seed_m.mtype == "p_fiber" else seed_m.y
                    kf.initialize_state(seed_m.x, y_init, vx=vx0, vy=vy0,
                                        ax=0.0, ay=0.0,
                                        pos_sigma=max(1.0, seed_m.sigma))
                last_t = t
            else:
                dt = max(1.0 / 60.0, t - float(last_t))
                kf.predict(dt=dt)
                last_t = t

            das_pos = [(m.mtype, m.x, m.y, m.sigma, m.confidence)
                       for (m, _a) in batch if m.mtype == "p_fiber"]
            other_pos = [(m.mtype, m.x, m.y, m.sigma, m.confidence)
                         for (m, _a) in batch if m.mtype != "p_fiber"]
            if das_pos:
                kf.update_position_x_only(das_pos)
            if other_pos:
                kf.update_positions(other_pos)

            das_vel: list = []
            other_vel: list = []
            for (m, _a) in batch:
                if m.mtype == "p_fiber":
                    if m.sigma_v > 0.0:
                        das_vel.append(("v_das", m.z_vx, m.z_vy, m.sigma_v, m.confidence))
                else:
                    if m.speed_mps <= 0.0:
                        continue
                    vel_sigma = _sensor_velocity_sigma(m.mtype, m.sigma, m.confidence)
                    vx_m = m.speed_mps * math.cos(m.heading_rad)
                    vy_m = m.speed_mps * math.sin(m.heading_rad)
                    other_vel.append((m.mtype.replace("p_", "v_"), vx_m, vy_m,
                                      vel_sigma, m.confidence))
            if das_vel:
                kf.update_velocities(das_vel)
            if other_vel:
                kf.update_velocities(other_vel)

            xh, vxh, axh, yh, vyh, ayh = [float(v) for v in kf.x.reshape(-1)]
            sigma_pos = math.sqrt(max(0.0, 0.5 * (kf.P[0, 0] + kf.P[3, 3])))
            sigma_vel = math.sqrt(max(0.0, 0.5 * (kf.P[1, 1] + kf.P[4, 4])))
            pos_err = (math.hypot(xh - truth.x, yh - truth.y)
                       if truth is not None else None)

            row_idx = len(traj_rows)
            had_vel_update = bool(das_vel) or bool(other_vel)
            for (m, audit) in batch:
                audit.accepted = True
                audit.skipped = False
                audit.kalman_t = t
                audit.kalman_row_index = row_idx
                if m.mtype == "p_fiber":
                    audit.update_kind = ("position+velocity"
                                         if m.sigma_v > 0.0 else "position")
                else:
                    audit.update_kind = ("position+velocity"
                                         if m.speed_mps > 0.0 else "position")

            gps_x = gps_y = None
            cam_x = cam_y = None
            das_x = das_y = None
            das_fp = None
            for (m, _a) in batch:
                if m.mtype == "p_gps":
                    gps_x, gps_y = m.x, m.y
                elif m.mtype == "p_camera":
                    cam_x, cam_y = m.x, m.y
                elif m.mtype == "p_fiber":
                    das_x, das_y = m.x, m.y

            def _nan_to_none(v: float) -> Optional[float]:
                return None if math.isnan(v) else v

            traj_rows.append(TrajectoryRow(
                t=t,
                global_track_id=str(gid),
                vehicle_id_oracle=oracle_vid,
                segment_id="",
                true_x=(truth.x if truth else None),
                true_y=(truth.y if truth else None),
                true_distance_m=None,
                gps_x=gps_x, gps_y=gps_y, gps_distance_m=None,
                cam_x=cam_x, cam_y=cam_y, cam_distance_m=None,
                das_x=das_x, das_y=das_y,
                das_fiber_position_m=das_fp,
                das_distance_m=None,
                x_hat=xh, y_hat=yh,
                vx_hat=vxh, vy_hat=vyh,
                ax_hat=axh, ay_hat=ayh,
                distance_hat_m=None,
                sigma_pos_m=sigma_pos, sigma_vel_mps=sigma_vel,
                pos_err_m=pos_err,
                innovation_x=_nan_to_none(kf.last_innovation_x),
                innovation_y=_nan_to_none(kf.last_innovation_y),
                innovation_S_x=_nan_to_none(kf.last_S_x),
                innovation_S_y=_nan_to_none(kf.last_S_y),
                NIS_x=_nan_to_none(kf.last_NIS_x),
                update_kind=("position+velocity" if had_vel_update else "position"),
            ))

    # Final pass: distances, segment_id, das fiber position.
    _augment_distances(traj_rows, events, world=world)

    audit_rows.sort(key=lambda r: (r.t, r.sensor_type, r.sensor_id))
    return audit_rows, traj_rows


# ---------------------------------------------------------------------------
# Distance / segment augmentation
# ---------------------------------------------------------------------------


def _augment_distances(
    traj_rows: List[TrajectoryRow],
    events: List[Any],
    world: Any = None,
) -> None:
    """Fill ``true_distance_m``, ``distance_hat_m``, segment_id, sensor distances.

    The "distance traveled" coordinate is the cumulative path length along
    the per-vehicle ``world.vehicle_state`` trail.  This is continuous
    across segment transitions automatically because it is purely a
    function of consecutive positions.
    """
    # Per-vehicle (sorted-by-time) trail with cumulative distance + lane_id.
    per_vid: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events:
        if getattr(ev, "topic", "") != "world.vehicle_state":
            continue
        p = getattr(ev, "payload", None) or {}
        if not isinstance(p, dict):
            continue
        vid = str(p.get("vehicle_id", "") or "")
        if not vid:
            continue
        per_vid.setdefault(vid, []).append({
            "t": _safe_float(p.get("t", 0.0)),
            "x": _safe_float(p.get("x", 0.0)),
            "y": _safe_float(p.get("y", 0.0)),
            "lane_id": str(p.get("lane_id", "") or ""),
        })
    for vid, trail in per_vid.items():
        trail.sort(key=lambda r: r["t"])
        prev = None
        cum = 0.0
        for s in trail:
            if prev is not None:
                cum += math.hypot(s["x"] - prev["x"], s["y"] - prev["y"])
            s["cum_dist"] = cum
            prev = s

    # Optional lane_id → segment_id lookup.  Also build a centroid cache for
    # the fallback nearest-lane lookup (priorities 2-3 in the segment_id fill).
    lane_to_seg: Dict[str, str] = {}
    lane_centroids: List[Tuple[str, str, float, float]] = []  # (lane_id, seg, cx, cy)
    if world is not None:
        try:
            for lid, lane in getattr(world, "lanes", {}).items():
                seg = str(getattr(lane, "segment_id", "") or "")
                lane_to_seg[str(lid)] = seg
                if seg:
                    pts = (getattr(lane, "polyline", None)
                           or getattr(lane, "points", None) or [])
                    cx, cy = 0.0, 0.0
                    if pts:
                        try:
                            xs = [float(p.x if hasattr(p, "x") else p[0]) for p in pts]
                            ys = [float(p.y if hasattr(p, "y") else p[1]) for p in pts]
                            cx = sum(xs) / len(xs)
                            cy = sum(ys) / len(ys)
                        except Exception:
                            cx = _safe_float(getattr(lane, "x", 0.0))
                            cy = _safe_float(getattr(lane, "y", 0.0))
                    else:
                        cx = _safe_float(getattr(lane, "x", 0.0))
                        cy = _safe_float(getattr(lane, "y", 0.0))
                    lane_centroids.append((str(lid), seg, cx, cy))
        except Exception:  # pragma: no cover - defensive only
            lane_to_seg = {}
            lane_centroids = []

    def _nearest_seg(lx: float, ly: float) -> str:
        """Segment id of the lane whose centroid is nearest to (lx, ly)."""
        if not lane_centroids:
            return ""
        best_seg = ""
        best_d2 = float("inf")
        for _, s, cx, cy in lane_centroids:
            d2 = (lx - cx) ** 2 + (ly - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_seg = s
        return best_seg

    # Fiber arc-length lookup for DAS, keyed by event payload.
    fiber_index: Dict[Tuple[str, float, float], float] = {}
    for ev in events:
        if getattr(ev, "topic", "") != "sensor.das":
            continue
        p = getattr(ev, "payload", None) or {}
        if not isinstance(p, dict):
            continue
        try:
            xk = float(p.get("x"))
            yk = float(p.get("y"))
            tk = float(p.get("t", 0.0))
        except (TypeError, ValueError):
            continue
        sid = str(p.get("sensor_id", "") or "")
        fiber_index[(sid, xk, yk)] = _safe_float(p.get("fiber_position_m", 0.0))
        # Also index by (any, x, y) so that callers without sensor_id can hit it.
        fiber_index[("", xk, yk)] = _safe_float(p.get("fiber_position_m", 0.0))

    by_gid: Dict[str, List[TrajectoryRow]] = {}
    for r in traj_rows:
        by_gid.setdefault(r.global_track_id, []).append(r)

    for gid, rows in by_gid.items():
        rows.sort(key=lambda r: r.t)
        oracle_vid = rows[0].vehicle_id_oracle
        states = per_vid.get(oracle_vid, [])
        t_arr = [s["t"] for s in states] if states else []

        # Cumulative Kalman distance.
        prev_xy: Optional[Tuple[float, float]] = None
        cum_hat = 0.0
        for r in rows:
            if r.x_hat is None or r.y_hat is None:
                continue
            if prev_xy is not None:
                cum_hat += math.hypot(r.x_hat - prev_xy[0],
                                      r.y_hat - prev_xy[1])
            r.distance_hat_m = cum_hat
            prev_xy = (r.x_hat, r.y_hat)

        # Per-row truth distance + lane→segment.
        for r in rows:
            if states:
                idx = bisect.bisect_left(t_arr, r.t)
                if idx == 0:
                    s = states[0]
                elif idx >= len(states):
                    s = states[-1]
                else:
                    a, b = states[idx - 1], states[idx]
                    s = a if abs(a["t"] - r.t) <= abs(b["t"] - r.t) else b
                r.true_distance_m = s["cum_dist"]
                r.segment_id = lane_to_seg.get(s["lane_id"], "") or ""
            # Priority 2: nearest lane by Kalman estimate (post-run, display only).
            if not r.segment_id and r.x_hat is not None and r.y_hat is not None:
                r.segment_id = _nearest_seg(r.x_hat, r.y_hat)
            # Priority 3: nearest lane by oracle position (post-run, display only).
            if not r.segment_id and r.true_x is not None and r.true_y is not None:
                r.segment_id = _nearest_seg(r.true_x, r.true_y)
            # Sensor distances: snap to the nearest true sample.
            for sx_attr, sy_attr, dst_attr in (
                ("gps_x", "gps_y", "gps_distance_m"),
                ("cam_x", "cam_y", "cam_distance_m"),
                ("das_x", "das_y", "das_distance_m"),
            ):
                sx = getattr(r, sx_attr)
                sy = getattr(r, sy_attr)
                if sx is None or sy is None:
                    continue
                if states:
                    best = min(states,
                               key=lambda s: math.hypot(sx - s["x"], sy - s["y"]))
                    setattr(r, dst_attr, best["cum_dist"])
                else:
                    setattr(r, dst_attr, math.hypot(sx, sy))

            # DAS fiber arc-length (best-effort lookup by raw x/y).
            if r.das_x is not None and r.das_y is not None:
                key = ("", float(r.das_x), float(r.das_y))
                if key in fiber_index:
                    r.das_fiber_position_m = fiber_index[key]

        # Priority 4: per-gid hysteresis — carry the last established segment_id
        # forward through rows that still have an empty segment_id so boundary
        # rows never flicker blank before the next non-empty id arrives.
        _last_seg = ""
        for r in rows:
            if r.segment_id:
                _last_seg = r.segment_id
            elif _last_seg:
                r.segment_id = _last_seg


# ---------------------------------------------------------------------------
# Coverage / gap helpers
# ---------------------------------------------------------------------------


def coverage_summary(
    audit_rows: List[AuditRow],
    traj_rows: List[TrajectoryRow],
) -> List[Dict[str, Any]]:
    """Per-sensor coverage: generated / accepted / skipped / pct of Kalman rows."""
    by_sensor: Dict[str, List[AuditRow]] = {"GPS": [], "Camera": [], "DAS": []}
    for a in audit_rows:
        by_sensor.setdefault(a.sensor_type, []).append(a)

    n_kalman_rows = len(traj_rows)
    out: List[Dict[str, Any]] = []
    for sensor in ("GPS", "Camera", "DAS"):
        rows = by_sensor.get(sensor, [])
        n_total = len(rows)
        n_acc = sum(1 for r in rows if r.accepted)
        n_skip = n_total - n_acc
        skip_breakdown: Dict[str, int] = {}
        for r in rows:
            if not r.accepted:
                skip_breakdown[r.skip_reason] = skip_breakdown.get(r.skip_reason, 0) + 1
        # Distinct (gid, kalman_t) batches updated by this sensor.
        batches = {(r.global_track_id, r.kalman_t)
                   for r in rows if r.accepted and r.kalman_t is not None}
        pct = 100.0 * len(batches) / max(1, n_kalman_rows)
        out.append({
            "sensor": sensor,
            "generated": n_total,
            "accepted": n_acc,
            "skipped": n_skip,
            "pct_kalman_rows_updated": pct,
            "skip_reasons": "; ".join(f"{k}={v}" for k, v in sorted(skip_breakdown.items())),
        })
    # Prediction-only rows: TrajectoryRows whose update_kind is the
    # synthetic "prediction_only" emitted by the shadow filter inside
    # sensor gaps.  This is now an explicit, first-class concept — see
    # build_audit_and_trajectory's emit_prediction_rows parameter.
    n_pred_only = sum(1 for tr in traj_rows
                      if getattr(tr, "update_kind", "") == "prediction_only")
    out.append({
        "sensor": "(prediction-only)",
        "generated": "",
        "accepted": "",
        "skipped": "",
        "pct_kalman_rows_updated": 100.0 * n_pred_only / max(1, n_kalman_rows),
        "skip_reasons": f"prediction_only_rows={n_pred_only}",
    })
    return out


def time_gaps(
    traj_rows: List[TrajectoryRow],
    gap_threshold_s: float = 0.5,
) -> List[Dict[str, Any]]:
    """Return time gaps in *measurement-bearing* rows larger than the threshold.

    Prediction-only rows are intentionally ignored here — by design they
    fill the visual trajectory inside the gap; *measurement* gaps are
    what the user wants to evaluate (these are the windows where the
    Kalman is coasting).
    """
    by_gid: Dict[str, List[TrajectoryRow]] = {}
    for r in traj_rows:
        if getattr(r, "update_kind", "") == "prediction_only":
            continue
        by_gid.setdefault(r.global_track_id, []).append(r)
    gaps: List[Dict[str, Any]] = []
    for gid, rows in by_gid.items():
        rows.sort(key=lambda r: r.t)
        for i in range(1, len(rows)):
            dt = rows[i].t - rows[i - 1].t
            if dt > gap_threshold_s:
                gaps.append({
                    "global_track_id": gid,
                    "t_start": rows[i - 1].t,
                    "t_end":   rows[i].t,
                    "gap_s":   dt,
                })
    return gaps


def segment_transitions(
    traj_rows: List[TrajectoryRow],
    gid: str,
) -> List[Dict[str, Any]]:
    """Return the (t, from_segment, to_segment) transitions for *gid*."""
    rows = sorted([r for r in traj_rows if r.global_track_id == gid],
                  key=lambda r: r.t)
    out: List[Dict[str, Any]] = []
    last_seg: Optional[str] = None
    for r in rows:
        seg = r.segment_id or ""
        if seg and seg != (last_seg or ""):
            if last_seg is not None:
                out.append({"t": r.t, "from_segment": last_seg, "to_segment": seg})
            last_seg = seg
    return out


# ---------------------------------------------------------------------------
# CSV / PNG writers
# ---------------------------------------------------------------------------


_AUDIT_FIELDS: Tuple[str, ...] = (
    "t", "sensor_type", "sensor_id",
    "vehicle_id_oracle", "global_track_id", "segment_id",
    "x", "y", "speed_mps", "heading_rad",
    "sigma_m", "sigma_v", "confidence", "snr",
    "accepted", "skipped", "skip_reason",
    "update_kind", "kalman_t", "kalman_row_index",
    # DAS physics fields (non-DAS rows will have 0.0)
    "das_amplitude", "fiber_distance_m", "snr_th", "lateral_offset_m",
)

_TRAJ_FIELDS: Tuple[str, ...] = (
    "t", "global_track_id", "vehicle_id_oracle", "segment_id",
    "update_kind",
    "true_x", "true_y", "true_distance_m",
    "gps_x", "gps_y", "gps_distance_m",
    "cam_x", "cam_y", "cam_distance_m",
    "das_x", "das_y", "das_fiber_position_m", "das_distance_m",
    "x_hat", "y_hat", "vx_hat", "vy_hat", "ax_hat", "ay_hat",
    "distance_hat_m", "sigma_pos_m", "sigma_vel_mps", "pos_err_m",
    # Phase 2 — Kalman innovation fields
    "innovation_x", "innovation_y", "innovation_S_x", "innovation_S_y", "NIS_x",
)


def _fmt_v(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.4f}"
    return v


def _xlsx_write_rows(ws: Any, header: List[str], data_rows: List[List[Any]]) -> None:
    """Write *header* + *data_rows* into an openpyxl worksheet with basic styling.

    The header row is bold and auto-filtered; columns are auto-sized to the
    widest value observed in the first 500 data rows (capped at 60 chars).
    """
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF", size=10)
    cell_font = Font(size=10)

    ws.append(header)
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for row in data_rows:
        ws.append(row)

    # Auto-size columns based on the widest content (sample first 500 rows).
    col_widths = [len(str(h)) for h in header]
    for row in data_rows[:500]:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)) if val is not None else 0)

    for i, width in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = min(width + 2, 62)

    ws.auto_filter.ref = ws.dimensions


def write_audit_csv(audit_rows: List[AuditRow], path: Any) -> Path:
    """Write the Kalman measurement audit table as an Excel (.xlsx) file."""
    import openpyxl
    p = Path(path).with_suffix(".xlsx")
    p.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Kalman Audit"
    header = list(_AUDIT_FIELDS)
    data: List[List[Any]] = []
    for a in audit_rows:
        d = asdict(a)
        data.append([_fmt_v(d.get(k)) for k in _AUDIT_FIELDS])
    _xlsx_write_rows(ws, header, data)
    wb.save(p)
    return p


def write_trajectory_csv(
    traj_rows: List[TrajectoryRow],
    path: Any,
    gid: Optional[str] = None,
) -> Path:
    """Write per-track trajectory data as an Excel (.xlsx) file."""
    import openpyxl
    p = Path(path).with_suffix(".xlsx")
    p.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Trajectory"
    header = list(_TRAJ_FIELDS)
    data: List[List[Any]] = []
    for r in traj_rows:
        if gid is not None and r.global_track_id != gid:
            continue
        d = asdict(r)
        data.append([_fmt_v(d.get(k)) for k in _TRAJ_FIELDS])
    _xlsx_write_rows(ws, header, data)
    wb.save(p)
    return p


def write_coverage_csv(
    coverage: List[Dict[str, Any]],
    path: Any,
) -> Path:
    """Write the per-sensor coverage summary as an Excel (.xlsx) file."""
    import openpyxl
    p = Path(path).with_suffix(".xlsx")
    p.parent.mkdir(parents=True, exist_ok=True)
    if not coverage:
        coverage = [{"sensor": "(no data)", "generated": 0, "accepted": 0,
                     "skipped": 0, "pct_kalman_rows_updated": 0.0,
                     "skip_reasons": ""}]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Coverage"
    keys = list(coverage[0].keys())
    data: List[List[Any]] = [[_fmt_v(row.get(k)) for k in keys] for row in coverage]
    _xlsx_write_rows(ws, keys, data)
    wb.save(p)
    return p


# ---------------------------------------------------------------------------
# Human-readable column labels for the consolidated audit workbook
# ---------------------------------------------------------------------------

_AUDIT_DISPLAY: Dict[str, str] = {
    "t":                  "Time (s)",
    "sensor_type":        "Sensor Type",
    "sensor_id":          "Sensor ID",
    "vehicle_id_oracle":  "True Vehicle ID",
    "global_track_id":    "Track ID",
    "segment_id":         "Road Segment",
    "x":                  "Measured X (m)",
    "y":                  "Measured Y (m)",
    "speed_mps":          "Measured Speed (m/s)",
    "heading_rad":        "Heading (rad)",
    "sigma_m":            "Position Uncertainty σ (m)",
    "sigma_v":            "Velocity Uncertainty σ (m/s)",
    "confidence":         "Detection Confidence",
    "snr":                "Signal-to-Noise Ratio",
    "accepted":           "Accepted by Kalman (1/0)",
    "skipped":            "Skipped (1/0)",
    "skip_reason":        "Skip Reason",
    "update_kind":        "Kalman Update Type",
    "kalman_t":           "Kalman Time (s)",
    "kalman_row_index":   "Kalman Row Index",
    "das_amplitude":      "DAS Amplitude A (raw units)",
    "fiber_distance_m":   "Fiber Distance r (m)",
    "snr_th":             "SNR Threshold",
    "lateral_offset_m":   "Lateral Offset from Lane Centre (m)",
}

_TRAJ_DISPLAY: Dict[str, str] = {
    "t":                  "Time (s)",
    "global_track_id":    "Track ID",
    "vehicle_id_oracle":  "True Vehicle ID",
    "segment_id":         "Road Segment",
    "update_kind":        "Update Type",
    "true_x":             "True X (m)",
    "true_y":             "True Y (m)",
    "true_distance_m":    "True Distance Along Road (m)",
    "gps_x":              "GPS X (m)",
    "gps_y":              "GPS Y (m)",
    "gps_distance_m":     "GPS Distance Along Road (m)",
    "cam_x":              "Camera X (m)",
    "cam_y":              "Camera Y (m)",
    "cam_distance_m":     "Camera Distance Along Road (m)",
    "das_x":              "DAS X (m)",
    "das_y":              "DAS Y (m)",
    "das_fiber_position_m": "DAS Fiber Position (m)",
    "das_distance_m":     "DAS Distance Along Road (m)",
    "x_hat":              "Kalman X Estimate (m)",
    "y_hat":              "Kalman Y Estimate (m)",
    "vx_hat":             "Kalman Velocity X (m/s)",
    "vy_hat":             "Kalman Velocity Y (m/s)",
    "ax_hat":             "Kalman Acceleration X (m/s²)",
    "ay_hat":             "Kalman Acceleration Y (m/s²)",
    "distance_hat_m":     "Kalman Distance Along Road (m)",
    "sigma_pos_m":        "Kalman Position Uncertainty σ (m)",
    "sigma_vel_mps":      "Kalman Velocity Uncertainty σ (m/s)",
    "pos_err_m":          "2D Position Error |true − estimate| (m)",
    # Phase 2 — Kalman innovation
    "innovation_x":       "Pre-update X Residual ν_x = x_meas − x_pred (m)",
    "innovation_y":       "Pre-update Y Residual ν_y = y_meas − y_pred (m)",
    "innovation_S_x":     "Innovation Variance S_xx (m²) — expected spread of ν_x",
    "innovation_S_y":     "Innovation Variance S_yy (m²) — expected spread of ν_y",
    "NIS_x":              "NIS_x = ν_x² / S_xx — follows χ²(1) when filter is consistent; >3.84 = anomaly signal",
}

_COV_DISPLAY: Dict[str, str] = {
    "sensor":                    "Sensor Type",
    "generated":                 "Measurements Generated",
    "accepted":                  "Measurements Accepted by Kalman",
    "skipped":                   "Measurements Skipped",
    "pct_kalman_rows_updated":   "% of Kalman Timesteps Updated",
    "skip_reasons":              "Skip Reason Breakdown",
}


def _readme_sheet(ws: Any, track_ids: List[str]) -> None:
    """Populate a 'Read Me' worksheet with a plain-language guide."""
    from openpyxl.styles import Font, PatternFill, Alignment

    title_font  = Font(bold=True, size=13, color="1F4E78")
    section_font = Font(bold=True, size=10, color="1F4E78")
    body_font   = Font(size=10)
    alt_fill    = PatternFill("solid", fgColor="EBF3FC")

    def _write(row_idx: int, col_a: str, col_b: str = "",
               *, bold: bool = False, fill: bool = False, big: bool = False) -> None:
        a = ws.cell(row=row_idx, column=1, value=col_a)
        b = ws.cell(row=row_idx, column=2, value=col_b)
        f = Font(bold=True, size=13, color="1F4E78") if big else (
            Font(bold=True, size=10) if bold else body_font)
        a.font = f
        b.font = body_font
        if fill:
            for c in (a, b):
                c.fill = alt_fill

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 72

    r = 1
    ws.cell(row=r, column=1, value="SimStudio Tracking Audit Report").font = title_font
    r += 1
    ws.cell(row=r, column=1, value="Auto-generated by SimStudio audit.py").font = Font(italic=True, size=9, color="555555")
    r += 2

    # --- Sheet guide ---------------------------------------------------------
    _write(r, "SHEET GUIDE", "", bold=True, big=True)
    r += 1
    _write(r, "Sheet name", "What it contains", bold=True)
    r += 1
    sheet_guide = [
        ("Coverage",      "How many measurements each sensor generated, how many were accepted by the "
                          "Kalman filter, and the fraction of Kalman timesteps that received an update."),
        ("Kalman Audit",  "Every individual sensor measurement, with the Kalman filter's decision "
                          "(accepted or skipped) and the reason for any skip.  "
                          "DAS rows also include the raw physics values: amplitude, fiber distance, "
                          "and lateral offset."),
    ]
    for i, (name, desc) in enumerate(sheet_guide):
        _write(r, name, desc, fill=(i % 2 == 0))
        r += 1
    for i, tid in enumerate(track_ids):
        _write(r, f"Track {tid}",
               f"Kalman state estimates and raw sensor measurements for vehicle track {tid}.  "
               "Columns show the ground truth position, what each sensor reported, and what "
               "the Kalman filter estimated.  pos_err_m is the 2-D distance between the "
               "filter estimate and ground truth.",
               fill=((len(sheet_guide) + i) % 2 == 0))
        r += 1

    r += 1
    # --- Column glossary -----------------------------------------------------
    _write(r, "KEY TERMS", "", bold=True, big=True)
    r += 1
    _write(r, "Term", "Meaning", bold=True)
    r += 1
    glossary = [
        ("Track ID",           "Unique identifier assigned to each detected vehicle track (e.g. T000001)."),
        ("True Vehicle ID",    "Ground-truth vehicle label from the simulator (oracle information)."),
        ("update_kind",        "'measurement_update' = Kalman corrected by a sensor; "
                               "'prediction_only' = no sensor available, filter coasted on physics model."),
        ("sigma_m",            "Position uncertainty (one standard deviation) in metres — lower is more confident."),
        ("SNR",                "Signal-to-Noise Ratio for DAS.  Higher SNR → better measurement quality."),
        ("Accepted (1/0)",     "1 = the Kalman filter used this measurement; 0 = it was skipped."),
        ("DAS Amplitude A",    "Raw DAS amplitude: A = W / (r + d₀)².  Proportional to vehicle weight."),
        ("Fiber Distance r",   "Perpendicular distance from vehicle to fibre-optic cable [m]."),
        ("Lateral Offset",     "Signed distance from vehicle centre to lane centreline [m].  "
                               "Large absolute values indicate lane-straddling or weaving."),
        ("pos_err_m",          "2-D Euclidean distance between Kalman estimate and ground truth [m]."),
    ]
    for i, (term, meaning) in enumerate(glossary):
        _write(r, term, meaning, fill=(i % 2 == 0))
        r += 1

    ws.sheet_view.showGridLines = False


def write_audit_workbook(
    audit_rows: List[AuditRow],
    traj_rows: List[TrajectoryRow],
    coverage: List[Dict[str, Any]],
    path: Any,
) -> Path:
    """Write a consolidated, human-readable multi-sheet audit Excel workbook.

    Sheets produced
    ---------------
    * **Read Me**       — plain-language guide to every sheet and key term.
    * **Coverage**      — per-sensor measurement acceptance / skip counters.
    * **Kalman Audit**  — every sensor measurement with the filter's decision.
    * **Track T000001** — per-timestep trajectory for each track (one sheet each).

    The workbook uses the same dark-blue header style as the main SimStudio
    export so both files feel consistent when opened together.

    Parameters
    ----------
    audit_rows:
        List of :class:`AuditRow` objects from :func:`build_audit_and_trajectory`.
    traj_rows:
        List of :class:`TrajectoryRow` objects from :func:`build_audit_and_trajectory`.
    coverage:
        Coverage summary list from :func:`coverage_summary`.
    path:
        Output file path.  The ``.xlsx`` extension is enforced automatically.

    Returns
    -------
    Path
        Absolute path to the written workbook.
    """
    import openpyxl

    p = Path(path).with_suffix(".xlsx")
    p.parent.mkdir(parents=True, exist_ok=True)

    # Collect meaningful track IDs — apply the same quality gate used for PNG
    # export so the workbook only contains real vehicle trajectories.
    # Ghost tracks (single DAS hit + prediction coast) and GPS fragments
    # (2–3 sparse GPS pings) are excluded here; they are documented in the
    # Ghost-Track Filter Analysis section of the Word report instead.
    _traj_by_gid: Dict[str, List[Any]] = {}
    for _r in traj_rows:
        if _r.global_track_id:
            _traj_by_gid.setdefault(_r.global_track_id, []).append(_r)

    track_ids: List[str] = []
    for _gid in sorted(_traj_by_gid.keys()):
        _ok, _reason = _is_meaningful_track_for_png(_traj_by_gid[_gid])
        if _ok:
            track_ids.append(_gid)

    wb = openpyxl.Workbook()

    # ── Read Me ──────────────────────────────────────────────────────────────
    ws_rm = wb.active
    ws_rm.title = "Read Me"
    _readme_sheet(ws_rm, track_ids)

    # ── Coverage ─────────────────────────────────────────────────────────────
    ws_cov = wb.create_sheet("Coverage")
    if not coverage:
        coverage = [{"sensor": "(no data)", "generated": 0, "accepted": 0,
                     "skipped": 0, "pct_kalman_rows_updated": 0.0,
                     "skip_reasons": ""}]
    cov_keys = list(coverage[0].keys())
    cov_header = [_COV_DISPLAY.get(k, k) for k in cov_keys]
    cov_data = [[_fmt_v(row.get(k)) for k in cov_keys] for row in coverage]
    _xlsx_write_rows(ws_cov, cov_header, cov_data)

    # ── Kalman Audit ─────────────────────────────────────────────────────────
    ws_aud = wb.create_sheet("Kalman Audit")
    aud_header = [_AUDIT_DISPLAY.get(k, k) for k in _AUDIT_FIELDS]
    aud_data: List[List[Any]] = []
    for a in audit_rows:
        d = asdict(a)
        aud_data.append([_fmt_v(d.get(k)) for k in _AUDIT_FIELDS])
    _xlsx_write_rows(ws_aud, aud_header, aud_data)

    # ── One sheet per track ───────────────────────────────────────────────────
    # Use the pre-grouped dict (already built above) to avoid an O(N_tracks × N_traj_rows)
    # scan through all traj_rows for each track.
    traj_header = [_TRAJ_DISPLAY.get(k, k) for k in _TRAJ_FIELDS]
    for tid in track_ids:
        # Excel sheet names are max 31 chars; truncate if needed.
        sheet_name = f"Track {tid}"[:31]
        ws_trk = wb.create_sheet(sheet_name)
        trk_data: List[List[Any]] = [
            [_fmt_v(asdict(r).get(k)) for k in _TRAJ_FIELDS]
            for r in _traj_by_gid.get(tid, [])
        ]
        _xlsx_write_rows(ws_trk, traj_header, trk_data)

    wb.save(p)
    _log.info("Audit workbook written: %s  (%d tracks, %d audit rows)",
              p, len(track_ids), len(audit_rows))
    return p


def _build_track_figure(
    traj_rows: List[TrajectoryRow],
    gid: str,
    *,
    title: Optional[str] = None,
    anomaly_markers: Optional[List[float]] = None,
) -> "Figure":  # type: ignore[name-defined]
    """Build and return a matplotlib Figure for one track (two panels).

    Panel 1: time vs. distance (ground truth, Kalman, raw sensor markers).
    Panel 2: tracking error over time, with optional dotted red anomaly vlines.

    Does not save to disk — callers decide the output format.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
    except Exception as exc:
        raise RuntimeError("matplotlib is required for figure rendering") from exc

    rows = sorted([r for r in traj_rows if r.global_track_id == gid], key=lambda r: r.t)
    if not rows:
        raise ValueError(f"no trajectory rows for global_track_id={gid!r}")

    fig = Figure(figsize=(12.5, 8.2), dpi=140)
    FigureCanvas(fig)  # attach non-interactive canvas
    gs = fig.add_gridspec(2, 1, height_ratios=[2.25, 1.0], hspace=0.18)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[1, 0], sharex=ax_traj)

    # --- Panel 1: time-distance trajectory ---------------------------------
    t_truth = [r.t for r in rows if r.true_distance_m is not None]
    d_truth = [r.true_distance_m for r in rows if r.true_distance_m is not None]
    if t_truth:
        ax_traj.plot(t_truth, d_truth, "-", lw=2.0, color="#1f77b4", label="ground truth", alpha=0.95, zorder=5)

    t_hat = [r.t for r in rows if r.distance_hat_m is not None]
    d_hat = [r.distance_hat_m for r in rows if r.distance_hat_m is not None]
    if t_hat:
        ax_traj.plot(t_hat, d_hat, "-", lw=1.8, color="#ff7f0e", label="Kalman estimate", zorder=4)

    def _scatter(attr: str, color: str, marker: str, label: str, ms: float, alpha: float) -> None:
        ts = [r.t for r in rows if getattr(r, attr) is not None]
        ds = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        if ts:
            ax_traj.plot(ts, ds, marker, color=color, ms=ms, alpha=alpha, label=label, lw=0, zorder=3)

    _scatter("gps_distance_m", "#2ca02c", "o", "GPS", ms=3.5, alpha=0.70)
    _scatter("cam_distance_m", "#d62728", "s", "Camera", ms=3.3, alpha=0.65)
    _scatter("das_distance_m", "#9467bd", "^", "DAS", ms=5.5, alpha=0.45)

    # --- Panel 2: distance error (matches the distance metric in panel 1) ----
    # Primary: |distance_hat_m − true_distance_m| — same unit as the top panel.
    # Fallback: pos_err_m (2D Euclidean) if distance values are unavailable.
    t_err_dist = [r.t for r in rows if r.distance_hat_m is not None and r.true_distance_m is not None]
    err_dist = [
        abs(float(r.distance_hat_m) - float(r.true_distance_m))
        for r in rows
        if r.distance_hat_m is not None and r.true_distance_m is not None
    ]
    t_err_pos = [r.t for r in rows if r.pos_err_m is not None]
    err_pos = [r.pos_err_m for r in rows if r.pos_err_m is not None]
    # Choose which series to use for the bottom panel.
    t_primary = t_err_dist if t_err_dist else t_err_pos
    err_primary = err_dist if t_err_dist else err_pos
    primary_label = "|distance_hat − distance_true|" if t_err_dist else "instantaneous 2D error"
    rolling_rmse_t: List[float] = []
    rolling_rmse: List[float] = []
    if t_primary:
        ax_err.plot(t_primary, err_primary, "-", lw=1.0, color="#111111", alpha=0.55,
                    label=primary_label)
        window_s = 1.0
        half_w = 0.5 * window_s
        # O(N log N) rolling RMSE via bisect — replaces the previous O(N²) loop.
        for i, t0 in enumerate(t_primary):
            lo_idx = bisect.bisect_left(t_primary, t0 - half_w)
            hi_idx = bisect.bisect_right(t_primary, t0 + half_w)
            vals = err_primary[lo_idx:hi_idx]
            if vals:
                rolling_rmse_t.append(t0)
                rolling_rmse.append((sum(v * v for v in vals) / len(vals)) ** 0.5)
        if rolling_rmse_t:
            ax_err.plot(rolling_rmse_t, rolling_rmse, "-", lw=2.0, color="#ff7f0e",
                        alpha=0.95, label="rolling RMSE (1.0 s)")

    # Anomaly markers: dotted red vertical lines on the error panel.
    if anomaly_markers:
        for t_a in anomaly_markers:
            ax_err.axvline(t_a, color="#cc0000", lw=1.2, ls=":", alpha=0.80, zorder=6)

    # --- Segment context on both panels ------------------------------------
    transitions = segment_transitions(traj_rows, gid)
    # Layout rules to prevent overlap:
    #   • Summary text   → top-left  (0.01, 0.985)  va="top"
    #   • Vline badges   → y=0.90 in xaxis_transform  — well below summary box
    #   • Transition legend → bottom-left (0.01, 0.030) va="bottom"
    #     (opposite end of the axes from the summary, never overlaps even for
    #     very long segment chains)
    # ── Time-aware badge placement ────────────────────────────────────────────
    # Four vertical levels.  Level assignment is driven by *time proximity*:
    # any badge whose timestamp falls within `_collision_s` of an already-
    # placed badge gets a different level.  Badges that are far apart in time
    # can safely reuse a level — the simple modulo approach failed because it
    # didn't account for this and placed badges 4 indices apart at the same
    # height even when their timestamps were only seconds apart.
    _BADGE_Y_LEVELS = [0.91, 0.78, 0.65, 0.52]

    # Collision window: badges closer than this (in seconds) must differ in y.
    # 8 % of the visible time range approximates the rendered width of one badge.
    _t_vals = [float(ev.get("t", 0.0)) for ev in transitions]
    _t_span = (max(_t_vals) - min(_t_vals)) if len(_t_vals) > 1 else 1.0
    _collision_s = max(2.0, _t_span * 0.08)

    # Greedy assignment: for each badge, pick the lowest level not already
    # occupied by a nearby badge.
    _badge_y: List[float] = []
    for _i_tr, _t_cur in enumerate(_t_vals):
        _occupied = {
            _badge_y[_j]
            for _j, _t_prev in enumerate(_t_vals[:_i_tr])
            if abs(_t_cur - _t_prev) < _collision_s
        }
        _chosen = next(
            (lvl for lvl in _BADGE_Y_LEVELS if lvl not in _occupied),
            _BADGE_Y_LEVELS[_i_tr % len(_BADGE_Y_LEVELS)],  # fallback if all occupied
        )
        _badge_y.append(_chosen)

    trans_legend_parts: List[str] = []
    for _i_tr, ev in enumerate(transitions):
        badge_num = _i_tr + 1
        t_ev = _t_vals[_i_tr]
        old_seg = str(ev.get("from_segment", ev.get("from", "")) or "?")
        new_seg = str(ev.get("to_segment", ev.get("to", "")) or "?")
        badge_y = _badge_y[_i_tr]
        # Dashed vline on both panels.
        for ax in (ax_traj, ax_err):
            ax.axvline(t_ev, color="#8aa9c7", lw=1.0, ls="--", alpha=0.85, zorder=1)
        for ax in (ax_traj, ax_err):
            ax.text(
                t_ev, badge_y, f"[{badge_num}]",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5,
                color="#3f6f99", fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "#8aa9c7",
                      "alpha": 0.92, "pad": 1.2, "boxstyle": "round,pad=0.25"},
                zorder=9,
            )
        trans_legend_parts.append(f"[{badge_num}] {old_seg}→{new_seg}")
    # Transition legend — bottom-left, anchored at the BOTTOM of the axes so
    # it can never collide with the summary box at the top regardless of how
    # long the segment-chain text is.
    if trans_legend_parts:
        _row_size = 3
        _rows = [
            "   ".join(trans_legend_parts[i: i + _row_size])
            for i in range(0, len(trans_legend_parts), _row_size)
        ]
        ax_traj.text(
            0.01, 0.030, "\n".join(_rows),
            transform=ax_traj.transAxes, fontsize=7.8,
            color="#3f6f99", va="bottom", ha="left",
            bbox={"facecolor": "white", "edgecolor": "#8aa9c7",
                  "alpha": 0.90, "pad": 2.2, "boxstyle": "round,pad=0.35"},
            zorder=10,
        )

    segments: List[str] = []
    for r in rows:
        seg = (r.segment_id or "").strip()
        if seg and (not segments or segments[-1] != seg):
            segments.append(seg)
    seg_text = " → ".join(segments) if segments else "unknown"

    vehs = sorted({(r.vehicle_id_oracle or "").strip() for r in rows if (r.vehicle_id_oracle or "").strip()})
    vehicle_text = ",".join(vehs[:3]) + ("…" if len(vehs) > 3 else "") if vehs else "unknown"
    rmse = (sum(float(e) ** 2 for e in err_primary) / len(err_primary)) ** 0.5 if err_primary else None
    max_err = max(err_primary) if err_primary else None
    prediction_only = sum(1 for r in rows if str(r.update_kind) == "prediction_only")
    summary_parts = [
        f"vehicle={vehicle_text}",
        f"segments={seg_text}",
        f"transitions={len(transitions)}",
        f"prediction_only={prediction_only}",
    ]

    fig.suptitle(title or f"Tracking trajectory — global_track_id = {gid}", fontsize=13)
    ax_traj.text(0.01, 0.985, " | ".join(summary_parts), transform=ax_traj.transAxes, fontsize=8.3,
                 color="#333333", va="top", ha="left",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0})

    ax_traj.set_xlabel("time (s)")
    ax_traj.set_ylabel("distance traveled (m)")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(loc="lower right", framealpha=0.88, fontsize=9)

    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("distance error (m)")
    ax_err.grid(True, alpha=0.25)
    if rmse is not None or max_err is not None:
        err_summary = []
        if rmse is not None:
            err_summary.append(f"RMSE={rmse:.2f} m")
        if max_err is not None:
            err_summary.append(f"max error={max_err:.2f} m")
        ax_err.text(0.99, 0.95, " | ".join(err_summary), transform=ax_err.transAxes, fontsize=8.4,
                    color="#333333", va="top", ha="right",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.0})
    if t_primary:
        ax_err.legend(loc="upper left", framealpha=0.88, fontsize=8)
    else:
        ax_err.text(0.5, 0.5, "No error values available", transform=ax_err.transAxes,
                    ha="center", va="center", color="#777777")

    return fig


def write_trajectory_png(
    traj_rows: List[TrajectoryRow],
    path: Any,
    gid: str,
    title: Optional[str] = None,
) -> Path:
    """Export a per-track trajectory figure to PNG.

    The figure has two stacked panels:
    1. time vs. distance, with ground truth, Kalman estimate and raw sensor
       markers;
    2. tracking error over time.

    Delegates all plotting to :func:`_build_track_figure`.
    """
    fig = _build_track_figure(traj_rows, gid, title=title)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    fig.clear()  # release matplotlib artist objects; prevents memory build-up during batch export
    return p


# ---------------------------------------------------------------------------
# 2-D XY spatial trajectory figure
# ---------------------------------------------------------------------------

def _build_xy_figure(
    traj_rows: List[TrajectoryRow],
    gid: str,
    *,
    title: Optional[str] = None,
) -> "Figure":  # type: ignore[name-defined]
    """Build a two-panel XY spatial figure for one track.

    Panel 1 (top): X vs Y — the true 2-D path vs. the Kalman-estimated path,
    with raw sensor observations overlaid and the Kalman path colour-coded by
    positional error magnitude.

    Panel 2 (bottom): 2-D positional error (pos_err_m) vs. time — the same
    metric used to colour the top panel, shown over time so that brief spikes
    are easy to spot.  This keeps both panels talking about the same quantity
    (2-D Euclidean distance between estimate and truth), in contrast to the
    distance-trajectory figure which shows along-track distance in both panels.

    Color scheme mirrors :func:`_build_track_figure` for visual consistency.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
    except Exception as exc:
        raise RuntimeError("matplotlib is required for figure rendering") from exc

    rows = sorted([r for r in traj_rows if r.global_track_id == gid], key=lambda r: r.t)
    if not rows:
        raise ValueError(f"no trajectory rows for global_track_id={gid!r}")

    fig = Figure(figsize=(10.5, 10.5), dpi=140)
    FigureCanvas(fig)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.22)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[1, 0])

    # ---- Panel 1: XY spatial path -----------------------------------------
    hat_rows = [r for r in rows if r.x_hat is not None and r.y_hat is not None]

    gt_x = [r.true_x for r in rows if r.true_x is not None and r.true_y is not None]
    gt_y = [r.true_y for r in rows if r.true_x is not None and r.true_y is not None]
    if gt_x:
        ax_xy.plot(gt_x, gt_y, "-", lw=2.2, color="#1f77b4", label="ground truth",
                   alpha=0.90, zorder=5)
        ax_xy.plot(gt_x[0], gt_y[0], "o", color="#1f77b4", ms=7, zorder=7,
                   markeredgecolor="white", markeredgewidth=1.2)
        ax_xy.plot(gt_x[-1], gt_y[-1], "s", color="#1f77b4", ms=7, zorder=7,
                   markeredgecolor="white", markeredgewidth=1.2)

    if hat_rows:
        hx = [r.x_hat for r in hat_rows]
        hy = [r.y_hat for r in hat_rows]
        ax_xy.plot(hx, hy, "-", lw=1.8, color="#ff7f0e", label="Kalman estimate",
                   alpha=0.85, zorder=4)
        ax_xy.plot(hx[0], hy[0], "o", color="#ff7f0e", ms=7, zorder=7,
                   markeredgecolor="white", markeredgewidth=1.2)
        ax_xy.plot(hx[-1], hy[-1], "s", color="#ff7f0e", ms=7, zorder=7,
                   markeredgecolor="white", markeredgewidth=1.2)


    # Sensor observations
    gps_x = [r.gps_x for r in rows if r.gps_x is not None and r.gps_y is not None]
    gps_y = [r.gps_y for r in rows if r.gps_x is not None and r.gps_y is not None]
    if gps_x:
        ax_xy.plot(gps_x, gps_y, "o", color="#2ca02c", ms=4.5, alpha=0.75,
                   label="GPS", lw=0, zorder=3)

    cam_x = [r.cam_x for r in rows if r.cam_x is not None and r.cam_y is not None]
    cam_y = [r.cam_y for r in rows if r.cam_x is not None and r.cam_y is not None]
    if cam_x:
        ax_xy.plot(cam_x, cam_y, "s", color="#d62728", ms=4.0, alpha=0.65,
                   label="Camera", lw=0, zorder=3)

    das_x = [r.das_x for r in rows if r.das_x is not None and r.das_y is not None]
    das_y = [r.das_y for r in rows if r.das_x is not None and r.das_y is not None]
    if das_x:
        ax_xy.plot(das_x, das_y, "^", color="#9467bd", ms=5.0, alpha=0.50,
                   label="DAS", lw=0, zorder=3)

    # Segment-transition on the XY panel.
    # Layout: badges at y=0.90 (below summary box), legend at bottom-left.
    transitions = segment_transitions(traj_rows, gid)
    xy_trans_legend_parts: List[str] = []
    for _i_tr, ev_t in enumerate(transitions):
        badge_num = _i_tr + 1
        t_ev = float(ev_t.get("t", 0.0))
        old_seg = str(ev_t.get("from_segment", ev_t.get("from", "")) or "?")
        new_seg = str(ev_t.get("to_segment", ev_t.get("to", "")) or "?")
        nearest = min(hat_rows or rows, key=lambda r: abs(r.t - t_ev))
        px = nearest.x_hat if nearest.x_hat is not None else nearest.true_x
        if px is not None:
            ax_xy.axvline(px, color="#8aa9c7", lw=1.0, ls="--", alpha=0.85, zorder=1)
            ax_xy.text(
                px, 0.90, f"[{badge_num}]",
                transform=ax_xy.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5,
                color="#3f6f99", fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "#8aa9c7",
                      "alpha": 0.92, "pad": 1.2, "boxstyle": "round,pad=0.25"},
                zorder=9,
            )
        xy_trans_legend_parts.append(f"[{badge_num}] {old_seg}→{new_seg}")

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(loc="best", framealpha=0.88, fontsize=9)

    # ---- Panel 2: 2D positional error vs time -----------------------------
    t_err = [r.t for r in rows if r.pos_err_m is not None]
    err = [r.pos_err_m for r in rows if r.pos_err_m is not None]
    rolling_t: List[float] = []
    rolling_r: List[float] = []
    if t_err:
        ax_err.plot(t_err, err, "-", lw=1.0, color="#111111", alpha=0.55,
                    label="instantaneous 2D error")
        window_s = 1.0
        half_w = 0.5 * window_s
        # O(N log N) rolling RMSE via bisect — replaces the previous O(N²) loop.
        for t0 in t_err:
            lo_idx = bisect.bisect_left(t_err, t0 - half_w)
            hi_idx = bisect.bisect_right(t_err, t0 + half_w)
            vals = err[lo_idx:hi_idx]
            if vals:
                rolling_t.append(t0)
                rolling_r.append((sum(v * v for v in vals) / len(vals)) ** 0.5)
        if rolling_t:
            ax_err.plot(rolling_t, rolling_r, "-", lw=2.0, color="#ff7f0e",
                        alpha=0.95, label="rolling RMSE (1.0 s)")

    # Mirror segment-transition vlines + badges onto the error panel.
    # Badges at y=0.90 — same level as the XY panel badges for visual consistency.
    for _i_tr, ev_t in enumerate(transitions):
        badge_num = _i_tr + 1
        t_ev = float(ev_t.get("t", 0.0))
        ax_err.axvline(t_ev, color="#8aa9c7", lw=1.0, ls="--", alpha=0.85, zorder=1)
        ax_err.text(
            t_ev, 0.90, f"[{badge_num}]",
            transform=ax_err.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5,
            color="#3f6f99", fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "#8aa9c7",
                  "alpha": 0.92, "pad": 1.2, "boxstyle": "round,pad=0.25"},
            zorder=9,
        )

    rmse = (sum(e ** 2 for e in err) / len(err)) ** 0.5 if err else None
    max_err = max(err) if err else None
    if rmse is not None or max_err is not None:
        parts = []
        if rmse is not None:
            parts.append(f"RMSE={rmse:.2f} m")
        if max_err is not None:
            parts.append(f"max error={max_err:.2f} m")
        ax_err.text(0.99, 0.95, " | ".join(parts), transform=ax_err.transAxes,
                    fontsize=8.4, color="#333333", va="top", ha="right",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.0})

    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("2D positional error (m)")
    ax_err.grid(True, alpha=0.25)
    if t_err:
        ax_err.legend(loc="upper left", framealpha=0.88, fontsize=8)
    else:
        ax_err.text(0.5, 0.5, "No positional error values available",
                    transform=ax_err.transAxes, ha="center", va="center", color="#777777")

    # ---- Title & summary inset --------------------------------------------
    vehs = sorted({(r.vehicle_id_oracle or "").strip() for r in rows
                   if (r.vehicle_id_oracle or "").strip()})
    vehicle_text = ",".join(vehs[:3]) + ("…" if len(vehs) > 3 else "") if vehs else "unknown"
    # Build segment sequence text (same logic as the distance-trajectory figure).
    segments_xy: List[str] = []
    for _r in rows:
        _seg = (_r.segment_id or "").strip()
        if _seg and (not segments_xy or segments_xy[-1] != _seg):
            segments_xy.append(_seg)
    seg_text_xy = " → ".join(segments_xy) if segments_xy else "unknown"
    prediction_only_xy = sum(1 for r in rows if str(r.update_kind) == "prediction_only")
    summary_parts = [
        f"vehicle={vehicle_text}",
        f"segments={seg_text_xy}",
        f"transitions={len(transitions)}",
        f"prediction_only={prediction_only_xy}",
    ]
    ax_xy.text(0.01, 0.985, " | ".join(summary_parts), transform=ax_xy.transAxes,
               fontsize=8.3, color="#333333", va="top", ha="left",
               bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0})
    # Transition legend — bottom-left, anchored at the bottom so it can never
    # collide with the summary box at the top regardless of segment-chain length.
    if xy_trans_legend_parts:
        _row_size = 3
        _rows = [
            "   ".join(xy_trans_legend_parts[i: i + _row_size])
            for i in range(0, len(xy_trans_legend_parts), _row_size)
        ]
        ax_xy.text(
            0.01, 0.030, "\n".join(_rows),
            transform=ax_xy.transAxes, fontsize=7.8,
            color="#3f6f99", va="bottom", ha="left",
            bbox={"facecolor": "white", "edgecolor": "#8aa9c7",
                  "alpha": 0.90, "pad": 2.2, "boxstyle": "round,pad=0.35"},
            zorder=10,
        )

    fig.suptitle(title or f"XY spatial trajectory — global_track_id = {gid}", fontsize=13)
    fig.tight_layout()
    return fig


def write_xy_png(
    traj_rows: List[TrajectoryRow],
    path: Any,
    gid: str,
    title: Optional[str] = None,
) -> Path:
    """Export the 2-D XY spatial trajectory figure for *gid* to a PNG file.

    Complements :func:`write_trajectory_png`: that plot shows distance vs.
    time; this one shows the actual X/Y plane so you can see the spatial
    separation between the ground-truth path and the Kalman estimate directly.
    """
    fig = _build_xy_figure(traj_rows, gid, title=title)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    fig.clear()  # release matplotlib artist objects; prevents memory build-up during batch export
    return p


# ---------------------------------------------------------------------------
# Post-run analysis helpers (all oracle-aware, never touch tracking/Kalman)
# ---------------------------------------------------------------------------


def _compute_track_stats(
    rows: List[TrajectoryRow],
    audit_rows: List[AuditRow],
) -> Dict[str, Any]:
    """Return a summary-stats dict for a single track's rows.

    Uses oracle ground-truth fields for error / distance metrics —
    post-run evaluation only, does not affect tracking or Kalman.
    """
    if not rows:
        return {}
    rows = sorted(rows, key=lambda r: r.t)
    gid = rows[0].global_track_id
    meas_rows = [r for r in rows if r.update_kind != "prediction_only"]
    errors = [r.pos_err_m for r in meas_rows if r.pos_err_m is not None]
    rmse = (sum(e ** 2 for e in errors) / len(errors)) ** 0.5 if errors else None

    segs: List[str] = []
    for r in rows:
        s = r.segment_id or ""
        if s and (not segs or segs[-1] != s):
            segs.append(s)

    transitions = segment_transitions(rows, gid)
    gaps = [g for g in time_gaps(rows, gap_threshold_s=0.1)
            if g["global_track_id"] == gid]
    return {
        "gid": gid,
        "oracle_vid": rows[0].vehicle_id_oracle or "",
        "duration_s": rows[-1].t - rows[0].t,
        "true_distance_m": max((r.true_distance_m or 0.0) for r in rows),
        "distance_hat_m": max((r.distance_hat_m or 0.0) for r in rows),
        "rmse_m": rmse,
        "max_err_m": max(errors) if errors else None,
        "n_gps": sum(1 for r in rows if r.gps_x is not None),
        "n_cam": sum(1 for r in rows if r.cam_x is not None),
        "n_das": sum(1 for r in rows if r.das_x is not None),
        "n_pred": sum(1 for r in rows if r.update_kind == "prediction_only"),
        "n_rows": len(rows),
        "segments": segs,
        "n_transitions": len(transitions),
        "transitions": transitions,
        "max_gap_s": max((g["gap_s"] for g in gaps), default=0.0),
    }


def _detect_anomalies(
    rows: List[TrajectoryRow],
    track_rmse: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return a list of anomaly dicts for one track.  Post-run, oracle-aware.

    Eight detectors (A1–A8) — all rule-based, shallow, no ML.

    Parameters
    ----------
    track_rmse : float | None
        If provided, thresholds for A1 and A2 are raised adaptively so that
        naturally noisy tracks don't flood the report:
            A1 spike threshold = max(ANOMALY_ERROR_SPIKE_M, 3 × track_rmse)
            A2 RMSE  threshold = max(ANOMALY_RMSE_M,       2 × track_rmse)
    """
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r.t)
    gid = rows[0].global_track_id
    out: List[Dict[str, Any]] = []

    # Adaptive thresholds (A1, A2 only).
    _spike_thr = (max(ANOMALY_ERROR_SPIKE_M, 3.0 * track_rmse)
                  if track_rmse else ANOMALY_ERROR_SPIKE_M)
    _rmse_thr  = (max(ANOMALY_RMSE_M, 2.0 * track_rmse)
                  if track_rmse else ANOMALY_RMSE_M)

    # A1: Instantaneous error spike.
    for r in rows:
        if r.pos_err_m is not None and r.pos_err_m > _spike_thr:
            out.append({"kind": "error_spike", "t_start": r.t, "t_end": r.t,
                        "value": r.pos_err_m, "gid": gid})

    # A2: High 1-second rolling RMSE.
    _ws = 1.0
    for r in rows:
        if r.pos_err_m is None:
            continue
        near = [r2.pos_err_m for r2 in rows
                if abs(r2.t - r.t) <= 0.5 * _ws and r2.pos_err_m is not None]
        if len(near) >= 3:
            rmse = (sum(e ** 2 for e in near) / len(near)) ** 0.5
            if rmse > _rmse_thr:
                out.append({"kind": "high_rmse",
                            "t_start": r.t - 0.5 * _ws, "t_end": r.t + 0.5 * _ws,
                            "value": rmse, "gid": gid})

    # A3: Long prediction-only gap.
    in_pred = False
    pred_start: Optional[float] = None
    for r in rows:
        is_pred = (r.update_kind == "prediction_only")
        if is_pred and not in_pred:
            in_pred = True
            pred_start = r.t
        elif not is_pred and in_pred:
            gap = r.t - (pred_start or r.t)
            if gap > ANOMALY_PRED_GAP_S:
                out.append({"kind": "long_coasting",
                            "t_start": pred_start, "t_end": r.t,
                            "value": gap, "gid": gid})
            in_pred = False

    # A4: Sudden jump in estimated distance.
    for prev, curr in zip(rows, rows[1:]):
        if prev.distance_hat_m is None or curr.distance_hat_m is None:
            continue
        delta = abs(curr.distance_hat_m - prev.distance_hat_m)
        if delta > ANOMALY_DIST_JUMP_M:
            out.append({"kind": "distance_jump",
                        "t_start": prev.t, "t_end": curr.t,
                        "value": delta, "gid": gid})

    # A5: Individual sensor dropout.
    for src_attr, label in [("gps_x", "GPS"), ("cam_x", "Camera"), ("das_x", "DAS")]:
        drop_start: Optional[float] = None
        last_seen: Optional[float] = None
        for r in rows:
            present = getattr(r, src_attr) is not None
            if present:
                if drop_start is not None and last_seen is not None:
                    gap = r.t - drop_start
                    if gap > ANOMALY_DROPOUT_S:
                        out.append({"kind": "sensor_dropout",
                                    "t_start": drop_start, "t_end": r.t,
                                    "value": gap, "sensor": label, "gid": gid})
                drop_start = None
                last_seen = r.t
            elif last_seen is not None and drop_start is None:
                drop_start = r.t

    # A6: Position uncertainty (sigma) inflation.
    for r in rows:
        if r.sigma_pos_m is not None and r.sigma_pos_m > ANOMALY_SIGMA_M:
            out.append({"kind": "sigma_inflation",
                        "t_start": r.t, "t_end": r.t,
                        "value": r.sigma_pos_m, "gid": gid})

    # A7: Error rise after a segment transition.
    for tr in segment_transitions(rows, gid):
        t_tr = tr["t"]
        b_errs = [r.pos_err_m for r in rows
                  if t_tr - 1.0 <= r.t < t_tr and r.pos_err_m is not None]
        a_errs = [r.pos_err_m for r in rows
                  if t_tr <= r.t <= t_tr + 1.0 and r.pos_err_m is not None]
        if b_errs and a_errs:
            mb = sum(b_errs) / len(b_errs)
            ma = sum(a_errs) / len(a_errs)
            if mb > 0 and ma > 1.5 * mb:
                out.append({"kind": "post_transition_error_rise",
                            "t_start": t_tr, "t_end": t_tr + 1.0,
                            "value": ma / mb,
                            "from_segment": tr["from_segment"],
                            "to_segment": tr["to_segment"],
                            "gid": gid})

    # A8: Track-level weak sensor support (only one sensor type ever present).
    active_types: set = set()
    for r in rows:
        if r.gps_x is not None:
            active_types.add("GPS")
        if r.cam_x is not None:
            active_types.add("Camera")
        if r.das_x is not None:
            active_types.add("DAS")
    if 0 < len(active_types) < 2:
        out.append({"kind": "weak_support",
                    "t_start": rows[0].t, "t_end": rows[-1].t,
                    "value": len(active_types),
                    "active_sensors": sorted(active_types),
                    "gid": gid})

    return out


def _deduplicate_anomalies(
    anomalies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge nearby anomalies of the same kind (within 0.5 s) into one."""
    by_key: Dict[Any, List[Dict[str, Any]]] = {}
    for a in anomalies:
        key = (a["kind"], a.get("sensor", ""), a.get("from_segment", ""))
        by_key.setdefault(key, []).append(a)

    result: List[Dict[str, Any]] = []
    for group in by_key.values():
        group.sort(key=lambda a: a["t_start"])
        merged: List[Dict[str, Any]] = []
        for a in group:
            if merged and a["t_start"] <= merged[-1]["t_end"] + 0.5:
                merged[-1]["t_end"] = max(merged[-1]["t_end"], a["t_end"])
                if a.get("value", 0) > merged[-1].get("value", 0):
                    merged[-1]["value"] = a["value"]
            else:
                merged.append(dict(a))
        result.extend(merged)
    result.sort(key=lambda a: a["t_start"])
    return result


def _build_narrative(
    anomaly: Dict[str, Any],
    rows: List[TrajectoryRow],
    audit_rows: List[AuditRow],
) -> str:
    """Return a shallow, rule-based narrative paragraph for one anomaly."""
    kind = anomaly["kind"]
    t_s = float(anomaly.get("t_start") or 0.0)
    t_e = float(anomaly.get("t_end") or t_s)
    ctx = 1.0

    def _win(lo: float, hi: float) -> List[TrajectoryRow]:
        return [r for r in rows if lo <= r.t <= hi]

    def _sens(rs: List[TrajectoryRow]) -> str:
        types: set = set()
        for r in rs:
            if r.gps_x is not None:
                types.add("GPS")
            if r.cam_x is not None:
                types.add("Camera")
            if r.das_x is not None:
                types.add("DAS")
        return ", ".join(sorted(types)) if types else "none"

    def _me(rs: List[TrajectoryRow]) -> Optional[float]:
        e = [r.pos_err_m for r in rs if r.pos_err_m is not None]
        return sum(e) / len(e) if e else None

    def _segs(rs: List[TrajectoryRow]) -> str:
        seen: List[str] = []
        for r in sorted(rs, key=lambda r: r.t):
            s = r.segment_id or ""
            if s and (not seen or seen[-1] != s):
                seen.append(s)
        return " → ".join(seen) if seen else "unknown"

    bef = _win(t_s - ctx, t_s)
    dur = _win(t_s, t_e)
    aft = _win(t_e, t_e + ctx)

    _KIND_OPENING = {
        "error_spike":
            f"Error spike of {anomaly.get('value', 0):.1f} m at t={t_s:.2f} s.",
        "high_rmse":
            f"Rolling RMSE of {anomaly.get('value', 0):.1f} m in t={t_s:.2f}–{t_e:.2f} s.",
        "long_coasting":
            f"Prediction-only gap of {anomaly.get('value', 0):.2f} s "
            f"(t={t_s:.2f}–{t_e:.2f} s).",
        "distance_jump":
            f"Estimated distance jumped {anomaly.get('value', 0):.1f} m "
            f"at t={t_s:.2f}–{t_e:.2f} s.",
        "sensor_dropout":
            f"{anomaly.get('sensor', 'Sensor')} dropout of "
            f"{anomaly.get('value', 0):.2f} s (t={t_s:.2f}–{t_e:.2f} s).",
        "sigma_inflation":
            f"Position uncertainty rose to σ={anomaly.get('value', 0):.1f} m "
            f"at t={t_s:.2f} s.",
        "post_transition_error_rise":
            f"Error rose ×{anomaly.get('value', 0):.1f} after "
            f"{anomaly.get('from_segment', '?')} → {anomaly.get('to_segment', '?')} "
            f"transition at t={t_s:.2f} s.",
        "weak_support":
            "Track had weak sensor support throughout "
            f"(only {', '.join(anomaly.get('active_sensors', [])) or 'none'} active).",
    }
    lines = [_KIND_OPENING.get(kind, f"Anomaly [{kind}] at t={t_s:.2f}–{t_e:.2f} s.")]

    if bef:
        be = _me(bef)
        lines.append(
            f"  Before: sensors={_sens(bef)}, segment={_segs(bef)}"
            + (f", mean error={be:.1f} m" if be is not None else "") + "."
        )
    if dur and kind not in ("weak_support",):
        de = _me(dur)
        n_pred = sum(1 for r in dur if r.update_kind == "prediction_only")
        lines.append(
            f"  During: sensors={_sens(dur)}"
            + (f", mean error={de:.1f} m" if de is not None else "")
            + (f", {n_pred}/{len(dur)} rows prediction-only" if n_pred else "") + "."
        )
    if aft:
        ae = _me(aft)
        de = _me(dur) if dur else None
        lines.append(
            f"  After: sensors={_sens(aft)}, segment={_segs(aft)}"
            + (f", mean error={ae:.1f} m" if ae is not None else "") + "."
        )
        if de is not None and ae is not None:
            if ae < de:
                lines.append(f"  → Error recovered to {ae:.1f} m after the anomaly.")
            elif ae > de * 1.5:
                lines.append(
                    f"  → Error continued to worsen after the anomaly ({ae:.1f} m)."
                )
    return "\n".join(lines)


def _render_track_figure_bytes(
    traj_rows: List[TrajectoryRow],
    gid: str,
    *,
    anomaly_markers: Optional[List[float]] = None,
) -> bytes:
    """Render the 2-panel trajectory figure as PNG bytes in memory."""
    import io
    fig = _build_track_figure(traj_rows, gid, anomaly_markers=anomaly_markers)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    # Explicitly close to free memory — _build_track_figure uses a non-pyplot
    # Figure, so we call clf() to release axes/artists rather than plt.close().
    try:
        fig.clf()
    except Exception:
        pass
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PNG meaningfulness filter (Issue 2)
# ---------------------------------------------------------------------------


def _is_meaningful_track_for_png(rows: List[TrajectoryRow]) -> Tuple[bool, str]:
    """Return (should_export, skip_reason_or_empty).

    Checks: duration, row count, distance, has Kalman estimate.
    All checks use only the traj_rows data (oracle_vid is used only for
    post-run evaluation, not for association).
    """
    if not rows:
        return False, "no_rows"
    meas_rows = [r for r in rows if r.update_kind != "prediction_only"]
    if not meas_rows:
        return False, "no_kalman_estimate"
    # Require a minimum number of real sensor updates.  GPS fragments and DAS
    # ghosts pass the duration / row / distance checks because the Kalman runs
    # many prediction-only steps between sparse sensor pings, inflating counts.
    # This check is applied first so that downstream checks are not mislead by
    # prediction-only row inflation.
    if len(meas_rows) < PNG_MIN_SENSOR_HITS:
        return False, f"too_few_sensor_hits({len(meas_rows)}<{PNG_MIN_SENSOR_HITS})"
    duration = max(r.t for r in rows) - min(r.t for r in rows)
    if duration < PNG_MIN_DURATION_S:
        return False, f"too_short({duration:.2f}s<{PNG_MIN_DURATION_S}s)"
    if len(rows) < PNG_MIN_ROWS:
        return False, f"too_few_rows({len(rows)}<{PNG_MIN_ROWS})"
    dist = max((r.distance_hat_m or r.true_distance_m or 0.0) for r in rows)
    if dist < PNG_MIN_DISTANCE_M:
        return False, f"too_small_distance({dist:.1f}m<{PNG_MIN_DISTANCE_M}m)"
    return True, ""


# ---------------------------------------------------------------------------
# Shared report data preparation (used by docx and pdf writers)
# ---------------------------------------------------------------------------


def _prepare_report_data(
    audit_rows: List[AuditRow],
    traj_rows: List[TrajectoryRow],
) -> Dict[str, Any]:
    """Return the pre-computed data structures shared by both report writers.

    Computes meaningful/skipped track lists, per-track stats + anomalies, and
    global summary values exactly once — both write_report_docx and
    write_report_pdf call this instead of duplicating the logic.
    """
    by_gid: Dict[str, List[TrajectoryRow]] = {}
    for r in traj_rows:
        if r.global_track_id:
            by_gid.setdefault(r.global_track_id, []).append(r)

    meaningful_gids: List[str] = []
    skipped_gids: List[Tuple[str, str]] = []
    for gid, gid_rows in sorted(by_gid.items()):
        ok, reason = _is_meaningful_track_for_png(gid_rows)
        if ok:
            meaningful_gids.append(gid)
        else:
            skipped_gids.append((gid, reason))

    stats_by_gid: Dict[str, Dict[str, Any]] = {}
    anomalies_by_gid: Dict[str, List[Dict[str, Any]]] = {}
    for gid in meaningful_gids:
        gid_rows = sorted(by_gid[gid], key=lambda r: r.t)
        stats_by_gid[gid] = _compute_track_stats(gid_rows, audit_rows)
        track_rmse = stats_by_gid[gid].get("rmse_m")
        raw = _deduplicate_anomalies(_detect_anomalies(gid_rows, track_rmse=track_rmse))
        anomalies_by_gid[gid] = raw[:ANOMALY_MAX_PER_TRACK]

    cov = coverage_summary(audit_rows, traj_rows)
    cov_map = {c["sensor"]: c for c in cov}
    all_errors = [r.pos_err_m for r in traj_rows if r.pos_err_m is not None]
    global_rmse = ((sum(e ** 2 for e in all_errors) / len(all_errors)) ** 0.5
                   if all_errors else None)
    global_max_err = max(all_errors) if all_errors else None
    n_pred_only = sum(1 for r in traj_rows if r.update_kind == "prediction_only")
    transitions_by_gid: Dict[str, List[Dict[str, Any]]] = {
        gid: stats_by_gid[gid].get("transitions", []) for gid in meaningful_gids
    }
    all_transitions = [tr for gid in meaningful_gids for tr in transitions_by_gid[gid]]
    total_anomalies = sum(len(v) for v in anomalies_by_gid.values())

    return {
        "by_gid": by_gid,
        "meaningful_gids": meaningful_gids,
        "skipped_gids": skipped_gids,
        "stats_by_gid": stats_by_gid,
        "anomalies_by_gid": anomalies_by_gid,
        "cov": cov,
        "cov_map": cov_map,
        "global_rmse": global_rmse,
        "global_max_err": global_max_err,
        "n_pred_only": n_pred_only,
        "all_transitions": all_transitions,
        "transitions_by_gid": transitions_by_gid,
        "total_anomalies": total_anomalies,
    }


# ---------------------------------------------------------------------------
# Word report — shared cell-styling helpers
# ---------------------------------------------------------------------------

def _docx_set_cell_bg(cell: Any, hex6: str) -> None:
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    shd = parse_xml(
        f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{hex6}"/>'
    )
    cell._tc.get_or_add_tcPr().append(shd)


def _docx_hdr_cell(cell: Any, text: str, bg: str = "2E75B6") -> None:
    from docx.shared import Pt, RGBColor
    _docx_set_cell_bg(cell, bg)
    cell.text = text
    for run in cell.paragraphs[0].runs:
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def _docx_key_cell(cell: Any, text: str) -> None:
    from docx.shared import RGBColor
    _docx_set_cell_bg(cell, "D6E4F7")
    cell.text = text
    for run in cell.paragraphs[0].runs:
        run.bold = True
        run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)


# ---------------------------------------------------------------------------
# Style patcher — overwrites Word's theme defaults to match the visual template
# ---------------------------------------------------------------------------

def _patch_doc_styles(doc: Any) -> None:
    """Replace python-docx / Office-theme defaults with the exact style XML
    from the visual template (Heading 1/2 colours, sizes, spacing; Normal body
    text colour and spacing; page margins).  Called once immediately after
    Document() is created.
    """
    from lxml import etree
    from docx.shared import Inches, Pt, RGBColor

    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def _replace(elem: Any, local_tag: str, new_xml: str) -> None:
        """Remove any existing child with this tag, then insert the new one."""
        tag = f"{{{W}}}{local_tag}"
        for old in elem.findall(tag):
            elem.remove(old)
        if new_xml:
            elem.append(etree.fromstring(new_xml))

    # ── Heading 1: 14 pt, bold, navy #1F3864, 18 pt before / 4 pt after ──
    _replace(
        doc.styles["Heading 1"].element, "pPr",
        '<w:pPr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:spacing w:before="360" w:after="80"/>'
        '<w:outlineLvl w:val="0"/>'
        '</w:pPr>',
    )
    _replace(
        doc.styles["Heading 1"].element, "rPr",
        '<w:rPr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:b/><w:bCs/>'
        '<w:color w:val="1F3864"/>'
        '<w:sz w:val="28"/><w:szCs w:val="28"/>'
        '</w:rPr>',
    )

    # ── Heading 2: 12 pt, bold, blue #2E75B6, 12 pt before / 3 pt after ──
    _replace(
        doc.styles["Heading 2"].element, "pPr",
        '<w:pPr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:spacing w:before="240" w:after="60"/>'
        '<w:outlineLvl w:val="1"/>'
        '</w:pPr>',
    )
    _replace(
        doc.styles["Heading 2"].element, "rPr",
        '<w:rPr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:b/><w:bCs/>'
        '<w:color w:val="2E75B6"/>'
        '<w:sz w:val="24"/><w:szCs w:val="24"/>'
        '</w:rPr>',
    )

    # ── Normal: Arial 11 pt, grey #595959, 3 pt before / 4 pt after ──────
    n = doc.styles["Normal"]
    n.font.name = "Arial"
    n.font.size = Pt(11)
    n.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    n.paragraph_format.space_before = Pt(3)
    n.paragraph_format.space_after = Pt(4)

    # ── List Paragraph: 2 pt before / 2 pt after ─────────────────────────
    lp = doc.styles["List Paragraph"]
    lp.paragraph_format.space_before = Pt(2)
    lp.paragraph_format.space_after = Pt(2)

    # ── Page margins: 1" all sides (matching template) ────────────────────
    sec = doc.sections[0]
    sec.top_margin    = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin   = Inches(1)
    sec.right_margin  = Inches(1)


def _docx_alt_row(row: Any, is_alt: bool) -> None:
    if is_alt:
        for cell in row.cells:
            _docx_set_cell_bg(cell, "F2F2F2")


_SEV_STYLE: Dict[str, tuple] = {
    "critical": ("✖", "C00000"),
    "warning":  ("⚠", "C55A11"),
    "info":     ("ℹ", "2E75B6"),
}


# ---------------------------------------------------------------------------
# 8.2 chart helpers — embedded matplotlib figures
# ---------------------------------------------------------------------------

def _render_error_timeline_bytes(traj_rows: List[Any], gid: str) -> bytes:
    """Error-over-time line chart with sensor-active vs prediction-only bands."""
    import io as _io
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rows = sorted([r for r in traj_rows if r.global_track_id == gid], key=lambda r: r.t)
    if not rows:
        raise ValueError("no rows")

    t_vals  = [r.t for r in rows]
    err_vals = [r.pos_err_m if r.pos_err_m is not None else float("nan") for r in rows]

    fig, ax = plt.subplots(figsize=(8, 2.8))

    # Shade prediction-only windows in light red, sensor-active in light green.
    i = 0
    while i < len(rows):
        kind = rows[i].update_kind
        j = i + 1
        while j < len(rows) and rows[j].update_kind == kind:
            j += 1
        color = "#fde8e8" if kind == "prediction_only" else "#e8f5e9"
        ax.axvspan(rows[i].t, rows[j - 1].t, color=color, alpha=0.6, linewidth=0)
        i = j

    ax.plot(t_vals, err_vals, color="#1a5276", linewidth=1.1, label="Position error")
    ax.axhline(0.5, color="#c0392b", linewidth=0.8, linestyle="--", label="0.5 m threshold")

    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Position error (m)", fontsize=8)
    ax.set_title(f"Error over time — {gid}", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)

    pred_patch   = mpatches.Patch(color="#fde8e8", label="Prediction-only")
    sensor_patch = mpatches.Patch(color="#e8f5e9", label="Sensor active")
    ax.legend(handles=[pred_patch, sensor_patch,
                        plt.Line2D([0],[0], color="#1a5276", lw=1.1, label="Error"),
                        plt.Line2D([0],[0], color="#c0392b", lw=0.8, ls="--", label="0.5 m")],
              fontsize=7, ncol=2, loc="upper right")

    fig.tight_layout(pad=0.5)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_gantt_bytes(traj_rows: List[Any], gid: str) -> bytes:
    """Sensor Gantt chart — horizontal bars per sensor showing active windows."""
    import io as _io
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rows = sorted([r for r in traj_rows if r.global_track_id == gid], key=lambda r: r.t)
    if not rows:
        raise ValueError("no rows")

    sensors = ["GPS", "Camera", "DAS"]
    colors  = {"GPS": "#2980b9", "Camera": "#27ae60", "DAS": "#8e44ad"}

    # Build active intervals per sensor
    def _intervals(attr: str) -> List[tuple]:
        intervals: List[tuple] = []
        in_seg = False
        seg_start = 0.0
        for r in rows:
            active = getattr(r, attr, None) is not None
            if active and not in_seg:
                seg_start = r.t
                in_seg = True
            elif not active and in_seg:
                intervals.append((seg_start, r.t))
                in_seg = False
        if in_seg:
            intervals.append((seg_start, rows[-1].t))
        return intervals

    sensor_attrs = {"GPS": "gps_x", "Camera": "cam_x", "DAS": "das_x"}

    fig, ax = plt.subplots(figsize=(8, 1.8))
    t0 = rows[0].t
    t1 = rows[-1].t

    for yi, sname in enumerate(sensors):
        for start, end in _intervals(sensor_attrs[sname]):
            ax.broken_barh([(start, end - start)], (yi - 0.4, 0.8),
                           facecolors=colors[sname], alpha=0.85)

    ax.set_yticks(range(len(sensors)))
    ax.set_yticklabels(sensors, fontsize=8)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(f"Sensor activity — {gid}", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout(pad=0.5)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Word report writer — self-contained Python-docx renderer
# ---------------------------------------------------------------------------


def write_report_docx(
    audit_rows: List[AuditRow],
    traj_rows: List[TrajectoryRow],
    out_path: Any,
    *,
    sim_name: str = "",
    events: Optional[List[Any]] = None,
    world: Any = None,
    min_track_updates: int = 5,
) -> Path:
    """Write a tracking-audit Word report (.docx) matching the enhanced template.

    Structure
    ---------
    Cover  (title, subtitle, meta lines)
    1.    Executive Summary  (narrative + ranked-findings table + colored bullets)
    2-N.  Analysis sections from audit_analysis (sensor quality, Kalman, …)
    N+1.  Track Overview  (compact summary table — no embedded charts)
    N+2.  Ghost-Track Filter Analysis  (filter stats, per-ghost-track detail)
    N+3.  Sensor σ Calibration Check
    N+4.  Per-Axis Error Breakdown (X vs Y)
    N+5.  Sensor Coverage Timeline
    N+6.  Cold-Start Latency
    N+7.  Speed vs Position Error

    Ghost-track filter
    ------------------
    ``min_track_updates`` (default 5) controls how many tracker update events a
    track must have before it is considered real.  Tracks below the threshold are
    ghost tracks produced by single unassociated DAS fiber hits and are suppressed
    from the export.  The Ghost-Track Filter Analysis section documents every
    suppressed track and flags any that had camera or GPS hits (potential real
    vehicles dropped by the filter).

    Performance notes
    -----------------
    No matplotlib figures are created or embedded in the docx.  All visuals are
    pure python-docx tables, which is ~100× cheaper than PNG rendering.
    """
    try:
        from docx import Document as _Doc
        from docx.shared import Pt, RGBColor
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required — run: pip install python-docx"
        ) from exc

    from datetime import datetime as _dt

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # ── Gather data ────────────────────────────────────────────────────
    rd = _prepare_report_data(audit_rows, traj_rows)
    by_gid           = rd["by_gid"]
    meaningful_gids  = rd["meaningful_gids"]
    skipped_gids     = rd["skipped_gids"]
    stats_by_gid     = rd["stats_by_gid"]
    cov_map          = rd["cov_map"]
    global_rmse      = rd["global_rmse"]
    global_max_err   = rd["global_max_err"]
    n_pred_only      = rd["n_pred_only"]
    all_transitions  = rd["all_transitions"]

    # ── Ghost-track filter statistics ──────────────────────────────────
    # For each track ID seen in track.update events, record: update count,
    # the oracle vehicle ID, sensor-type breakdown, and tentative flag.
    # Tracks below ``min_track_updates`` are "ghost tracks" (single-fiber
    # DAS hits that never gated onto an existing track).
    from collections import Counter as _Counter

    _gt_update_count: dict = {}       # track_id  -> int
    _gt_last_n_gps:   dict = {}       # track_id  -> int (from last update)
    _gt_last_n_cam:   dict = {}
    _gt_last_n_das:   dict = {}
    _gt_oracle_vid:   dict = {}       # track_id  -> str
    _gt_tentative:    dict = {}       # track_id  -> bool
    _gt_sources:      dict = {}       # track_id  -> Counter(source)

    if events:
        for _ev in events:
            if getattr(_ev, "topic", "") == "track.update":
                _p = getattr(_ev, "payload", None) or {}
                if not isinstance(_p, dict):
                    continue
                _tid = str(_p.get("global_track_id", ""))
                if not _tid:
                    continue
                _gt_update_count[_tid] = _gt_update_count.get(_tid, 0) + 1
                _gt_last_n_gps[_tid]   = int(_p.get("n_gps", 0))
                _gt_last_n_cam[_tid]   = int(_p.get("n_cam", 0))
                _gt_last_n_das[_tid]   = int(_p.get("n_das", 0))
                _gt_tentative[_tid]    = bool(_p.get("tentative", True))
                _src = str(_p.get("source", "unknown"))
                if _tid not in _gt_sources:
                    _gt_sources[_tid] = _Counter()
                _gt_sources[_tid][_src] += 1
                _vid = str(_p.get("vehicle_id_oracle", ""))
                if _vid and _tid not in _gt_oracle_vid:
                    _gt_oracle_vid[_tid] = _vid

    _gt_total       = len(_gt_update_count)
    _gt_retained_ids = sorted(
        [tid for tid, cnt in _gt_update_count.items() if cnt >= min_track_updates]
    )
    _gt_filtered_ids = sorted(
        [tid for tid, cnt in _gt_update_count.items() if cnt < min_track_updates]
    )
    _gt_n_retained  = len(_gt_retained_ids)
    _gt_n_filtered  = len(_gt_filtered_ids)
    _gt_filter_rate = (
        f"{100 * _gt_n_filtered // max(_gt_total, 1)}%" if _gt_total else "n/a"
    )

    # Aggregate sensor breakdown across all ghost tracks
    _gt_ghost_sensor_totals: dict = _Counter()
    for _tid in _gt_filtered_ids:
        for _src, _cnt in _gt_sources.get(_tid, {}).items():
            _gt_ghost_sensor_totals[_src] += _cnt

    from . import audit_analysis as _aa
    exec_summary, analysis_sections = _aa.build_full_analysis(
        audit_rows=audit_rows,
        traj_rows=traj_rows,
        by_gid=by_gid,
        stats_by_gid=stats_by_gid,
        anomalies_by_gid=rd["anomalies_by_gid"],
        skipped_gids=skipped_gids,
        coverage=rd["cov"],
        transitions_by_gid=rd["transitions_by_gid"],
        events=events,
        world=world,
    )

    enhanced = _compute_enhanced_diagnostics(
        audit_rows, traj_rows, meaningful_gids, by_gid, stats_by_gid
    )

    # ── Build document ─────────────────────────────────────────────────
    doc = _Doc()
    _patch_doc_styles(doc)

    # ── Helpers ────────────────────────────────────────────────────────
    def _bullet_colored(text: str, rgb: tuple) -> None:
        p_ = doc.add_paragraph(style="List Bullet")
        run = p_.add_run(text)
        run.font.color.rgb = RGBColor(*rgb)

    def _bullet_severity(text: str) -> None:
        low = text.lower()
        if any(w in low for w in ("critical", "fail", "zero", "never",
                                   "missing", "only", "⚠")):
            _bullet_colored(text, (0xC0, 0x00, 0x00))
        elif any(w in low for w in ("warning", "caution", "suspicious",
                                    "underestim", "bias", "gap", "sparse")):
            _bullet_colored(text, (0xB8, 0x86, 0x0B))
        elif any(w in low for w in ("good", "healthy", "✓", " ok", "well",
                                    "correct", "calibr")):
            _bullet_colored(text, (0x37, 0x56, 0x23))
        else:
            doc.add_paragraph(text, style="List Bullet")

    def _data_table(
        headers: List[str],
        rows: List[List[str]],
        *,
        status_col: int = -1,
    ) -> None:
        """Blue-header table with alternating rows.
        status_col: column index whose text is colored by keyword (Critical /
        Warning / OK).  Pass -1 (default) to skip per-cell coloring.
        """
        if not rows:
            return
        tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
        tbl.style = "Table Grid"
        for j, h in enumerate(headers):
            _docx_hdr_cell(tbl.rows[0].cells[j], h)
        for i, rv in enumerate(rows, start=1):
            for j, v in enumerate(rv):
                tbl.rows[i].cells[j].text = str(v)
            _docx_alt_row(tbl.rows[i], i % 2 == 0)
            if 0 <= status_col < len(rv):
                val = str(rv[status_col])
                if any(w in val for w in ("Critical", "⚠ Crit", "Over-trust")):
                    rgb = (0xC0, 0x00, 0x00)
                elif any(w in val for w in ("Warning", "⚠ Warn", "Conserv", "High")):
                    rgb = (0xB8, 0x86, 0x0B)
                elif any(w in val for w in ("✓", "OK", "ok", "calibr", "Low")):
                    rgb = (0x37, 0x56, 0x23)
                else:
                    rgb = None
                if rgb:
                    cell = tbl.rows[i].cells[status_col]
                    for par in cell.paragraphs:
                        for run in par.runs:
                            run.font.color.rgb = RGBColor(*rgb)

    def _insight_box(body: str, bg: str = "EEF4FF", label: str = "") -> None:
        """1-cell callout table matching the enhanced template insight boxes."""
        tbl = doc.add_table(rows=1, cols=1)
        tbl.style = "Table Grid"
        cell = tbl.rows[0].cells[0]
        _docx_set_cell_bg(cell, bg)
        par = cell.paragraphs[0]
        if label:
            run_lbl = par.add_run(label + "\n")
            run_lbl.bold = True
            run_lbl.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
        run_body = par.add_run(body)
        run_body.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
        doc.add_paragraph()  # breathing room after box

    def _render_section(sec_num: int, sec: Any) -> None:
        """Render one Section with a numbered heading and colored insight boxes."""
        doc.add_heading(f"{sec_num}. {getattr(sec, 'title', '')}", 1)
        if getattr(sec, "summary", ""):
            p_ = doc.add_paragraph(sec.summary)
            if p_.runs:
                p_.runs[0].font.color.rgb = RGBColor(0x59, 0x59, 0x59)
        tbl_data = getattr(sec, "table", None)
        if tbl_data and tbl_data.get("rows"):
            _data_table(tbl_data["headers"], tbl_data["rows"])
            doc.add_paragraph()
        for b in getattr(sec, "bullets", []):
            _bullet_severity(b)
        for interp in getattr(sec, "interpretations", []):
            sev = getattr(interp, "severity", "info")
            if sev == "critical":
                box_bg, box_label = "FFF5F5", f"⚠ CRITICAL — {interp.label}"
            elif sev == "warning":
                box_bg, box_label = "FFF3CD", f"⚠ WARNING — {interp.label}"
            else:
                box_bg, box_label = "EEF4FF", f"ℹ {interp.label}"
            parts = []
            for attr, pfx in [
                ("technical",   "What happened: "),
                ("meaning",     "What it means: "),
                ("why_matters", "Why it matters: "),
            ]:
                val = getattr(interp, attr, "")
                if val:
                    parts.append(pfx + val)
            _insight_box("  ".join(parts), box_bg, box_label)
            for lst_label, attr in [
                ("Evidence:",        "evidence"),
                ("Possible causes:", "hypotheses"),
                ("Next checks:",     "next_checks"),
            ]:
                items = list(getattr(interp, attr, []))
                if items:
                    lbl_p = doc.add_paragraph()
                    lbl_p.add_run(lst_label).bold = True
                    for item in items:
                        doc.add_paragraph(item, style="List Bullet")

    # ══════════════════════════════════════════════════════════════════
    # COVER
    # ══════════════════════════════════════════════════════════════════
    doc.add_paragraph()   # top spacer (matches template)

    title_para = doc.add_paragraph()
    title_run = title_para.add_run("TRACKING AUDIT REPORT")
    title_run.font.name  = "Arial"
    title_run.font.size  = Pt(26)
    title_run.font.bold  = True
    title_run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    title_para.paragraph_format.space_before = Pt(0)
    title_para.paragraph_format.space_after  = Pt(0)

    doc.add_paragraph()   # blank between title and subtitle

    sub_para = doc.add_paragraph()
    sub_run = sub_para.add_run("Enhanced Diagnostic Analysis")
    sub_run.font.name  = "Arial"
    sub_run.font.size  = Pt(14)
    sub_run.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    sub_para.paragraph_format.space_before = Pt(0)
    sub_para.paragraph_format.space_after  = Pt(0)

    doc.add_paragraph()   # spacers before meta lines
    doc.add_paragraph()
    doc.add_paragraph()

    sim_line = doc.add_paragraph()
    sim_run = sim_line.add_run(f"Simulation: {sim_name or 'n/a'}")
    sim_run.font.name = "Arial"
    sim_run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    sim_line.paragraph_format.space_before = Pt(0)
    sim_line.paragraph_format.space_after  = Pt(2)

    n_sensor_types = len(
        [s for s in ("GPS", "Camera", "DAS")
         if cov_map.get(s, {}).get("accepted", 0) > 0]
    )
    gen_line = doc.add_paragraph()
    gen_run = gen_line.add_run(
        f"Generated: {_dt.now().strftime('%Y-%m-%d')}"
        f"  |  {len(meaningful_gids)} Tracks"
        f"  |  {n_sensor_types} Sensor Types"
    )
    gen_run.font.name = "Arial"
    gen_run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    gen_line.paragraph_format.space_before = Pt(0)
    gen_line.paragraph_format.space_after  = Pt(6)

    # ══════════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════
    doc.add_heading("1. Executive Summary", 1)
    doc.add_paragraph()

    p_ = doc.add_paragraph(exec_summary.plain_english_paragraph)
    if p_.runs:
        p_.runs[0].font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    doc.add_paragraph()
    doc.add_paragraph()

    # Ranked findings table  (mirrors Table 0 in the enhanced template)
    rank_rows: List[List[str]] = []
    for i, b in enumerate(exec_summary.bullets, 1):
        low = b.lower()
        if any(w in low for w in ("critical", "fail", "zero", "never", "missing")):
            status, impact = "⚠ Critical", "High"
        elif any(w in low for w in ("warning", "suspicious", "low", "bias", "gap")):
            status, impact = "⚠ Warning", "Medium"
        else:
            status, impact = "✓ OK", "Low"
        rank_rows.append([f"#{i}", b, impact, status])

    if rank_rows:
        _data_table(["Rank", "Finding", "Impact", "Status"],
                    rank_rows, status_col=3)

    # ══════════════════════════════════════════════════════════════════
    # 2-N. ANALYSIS SECTIONS  (sensor quality, Kalman health, …)
    # ══════════════════════════════════════════════════════════════════
    for i, sec in enumerate(analysis_sections, 2):
        doc.add_paragraph()
        _render_section(i, sec)

    next_num = len(analysis_sections) + 2

    # ══════════════════════════════════════════════════════════════════
    # TRACK OVERVIEW  (compact table — no embedded images)
    # ══════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_heading(f"{next_num}. Track Overview", 1)
    next_num += 1

    if global_rmse is not None:
        doc.add_paragraph(
            f"{len(meaningful_gids)} track(s) recorded.  "
            f"Global RMSE: {global_rmse:.2f} m.  "
            f"Global max error: {global_max_err:.2f} m.  "
            f"Prediction-only rows: {n_pred_only}."
        )
    else:
        doc.add_paragraph(f"{len(meaningful_gids)} track(s) recorded.")

    tr_hdr = [
        "Track", "Vehicle", "Duration (s)", "RMSE (m)", "Max err (m)",
        "GPS", "Camera", "DAS", "Pred-only", "% Pred", "Segments",
    ]
    tr_data: List[List[str]] = []
    for gid in meaningful_gids:
        st = stats_by_gid[gid]
        n_tot = st["n_rows"]
        pct = f"{100 * st['n_pred'] // max(n_tot, 1)}%" if n_tot else "n/a"
        tr_data.append([
            str(gid),
            st.get("oracle_vid", "") or "",
            f"{st['duration_s']:.2f}",
            f"{st['rmse_m']:.2f}" if st.get("rmse_m") is not None else "n/a",
            f"{st['max_err_m']:.2f}" if st.get("max_err_m") is not None else "n/a",
            str(st["n_gps"]),
            str(st["n_cam"]),
            str(st["n_das"]),
            str(st["n_pred"]),
            pct,
            " → ".join(st["segments"]) if st["segments"] else "—",
        ])
    _data_table(tr_hdr, tr_data)

    if all_transitions:
        doc.add_paragraph()
        doc.add_heading("Segment Transitions", 2)
        doc.add_paragraph(f"{len(all_transitions)} transition(s) across all tracks.")
        tr_trans: List[List[str]] = [
            [str(gid), f"{tr['t']:.2f}", tr["from_segment"], tr["to_segment"]]
            for gid in meaningful_gids
            for tr in stats_by_gid[gid].get("transitions", [])
        ]
        _data_table(["Track", "Time (s)", "From segment", "To segment"], tr_trans)

    if skipped_gids:
        doc.add_paragraph()
        doc.add_heading("Filtered Tracks", 2)
        doc.add_paragraph(
            f"{len(skipped_gids)} track(s) excluded "
            "(below duration, row-count, or distance thresholds)."
        )
        _data_table(
            ["Track ID", "Skip reason"],
            [(str(g), r) for g, r in sorted(skipped_gids)],
        )

    # ══════════════════════════════════════════════════════════════════
    # GHOST-TRACK FILTER ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_heading(f"{next_num}. Ghost-Track Filter Analysis", 1)
    next_num += 1

    doc.add_paragraph(
        "The simulator's tracker assigns a new tentative track to every fiber-optic "
        "DAS hit that falls outside the association gate of all existing tracks.  In "
        "dense sensor deployments this produces large numbers of single-hit "
        "\"ghost tracks\" — tracks that were never observed a second time and therefore "
        "carry no useful trajectory information.  Before export, a ghost-track filter "
        f"suppresses any track that received fewer than {min_track_updates} tracker "
        "updates (the MIN_TRACK_UPDATES threshold).  The table and breakdown below "
        "document every filtered track so the analyst can verify that no real vehicle "
        "was incorrectly discarded."
    )
    doc.add_paragraph()

    # ── Summary box ──
    if _gt_total == 0:
        _insight_box(
            "No track.update events were found in the event log.  "
            "Ghost-track statistics cannot be computed for this run.",
            "FFF3CD",
            "⚠ WARNING — No track events",
        )
    else:
        _summary_body = (
            f"Total unique track IDs seen: {_gt_total}  |  "
            f"Retained (≥ {min_track_updates} updates): {_gt_n_retained}  |  "
            f"Ghost-filtered (< {min_track_updates} updates): {_gt_n_filtered}  |  "
            f"Filter rate: {_gt_filter_rate}"
        )
        if _gt_n_filtered == 0:
            _insight_box(_summary_body, "EEF4FF", "ℹ Ghost-Track Summary — No tracks filtered")
        elif _gt_n_filtered > _gt_n_retained:
            _insight_box(
                _summary_body + "  —  More tracks were filtered than retained. "
                "This typically indicates very dense DAS coverage (≥ 1 sensor per "
                "lane segment) or an association gate (_GATE_M) that is too narrow. "
                "Consider raising _GATE_M in tracking.py or reducing DAS density.",
                "FFF5F5",
                "⚠ CRITICAL — Ghost tracks dominate",
            )
        else:
            _insight_box(_summary_body, "EEF4FF", "ℹ Ghost-Track Summary")

        # ── Summary stats table ──
        _summary_rows = [
            ["Total tracks seen",                  str(_gt_total)],
            [f"Retained  (≥ {min_track_updates} updates)", str(_gt_n_retained)],
            [f"Filtered  (< {min_track_updates} updates)", str(_gt_n_filtered)],
            ["Filter rate",                         _gt_filter_rate],
            ["MIN_TRACK_UPDATES threshold",         str(min_track_updates)],
        ]
        _data_table(["Metric", "Value"], _summary_rows)
        doc.add_paragraph()

        # ── Sensor-type breakdown of ghost tracks ──
        if _gt_ghost_sensor_totals:
            doc.add_heading("Sensor Source Breakdown — Filtered (Ghost) Tracks", 2)
            doc.add_paragraph(
                "Each sensor update that arrived for a ghost track is tallied below "
                "by source type.  A DAS-dominant breakdown confirms that the ghosts "
                "originate from unassociated fiber-optic hits, not from GPS or camera "
                "measurement noise."
            )
            _gs_rows = [
                [src, str(cnt)]
                for src, cnt in sorted(
                    _gt_ghost_sensor_totals.items(), key=lambda x: -x[1]
                )
            ]
            _data_table(["Sensor source", "Update count"], _gs_rows)
            doc.add_paragraph()

        # ── Per-ghost-track detail table ──
        if _gt_filtered_ids:
            doc.add_heading("Per-Filtered-Track Detail", 2)
            doc.add_paragraph(
                "Every track that was suppressed by the ghost-track filter is listed "
                "here with its full sensor breakdown.  Tracks that show a non-zero "
                "camera or GPS hit count despite low update totals may indicate a "
                "real vehicle that was dropped too aggressively — consider lowering "
                "the MIN_TRACK_UPDATES threshold or investigating the association gate."
            )
            _detail_rows: List[List[str]] = []
            for _tid in _gt_filtered_ids:
                _upd   = _gt_update_count.get(_tid, 0)
                _ngps  = _gt_last_n_gps.get(_tid, 0)
                _ncam  = _gt_last_n_cam.get(_tid, 0)
                _ndas  = _gt_last_n_das.get(_tid, 0)
                _vid   = _gt_oracle_vid.get(_tid, "—")
                _tent  = "Yes" if _gt_tentative.get(_tid, True) else "No"
                _srcs  = ", ".join(
                    f"{s}×{c}" for s, c in sorted(
                        (_gt_sources.get(_tid, {})).items(), key=lambda x: -x[1]
                    )
                ) or "—"
                # Flag rows that might be real vehicles (camera or GPS hit)
                _flag  = "⚠" if (_ncam > 0 or _ngps > 0) else ""
                _detail_rows.append([
                    _flag + _tid, str(_upd), str(_ngps), str(_ncam), str(_ndas),
                    _srcs, _vid, _tent,
                ])
            _data_table(
                [
                    "Track ID", "Updates", "GPS hits", "Cam hits", "DAS hits",
                    "Sources (type×count)", "Oracle vehicle", "Tentative",
                ],
                _detail_rows,
            )
            doc.add_paragraph()
            # Highlight flagged rows with an insight box
            _flagged = [r for r in _detail_rows if r[0].startswith("⚠")]
            if _flagged:
                _flagged_ids = ", ".join(r[0].replace("⚠", "").strip() for r in _flagged)
                _insight_box(
                    f"{len(_flagged)} filtered track(s) had at least one GPS or camera "
                    f"measurement: {_flagged_ids}.  These may be real vehicles that fell "
                    "below the update threshold due to a brief sensor gap.  Review "
                    "manually before concluding they are ghost tracks.",
                    "FFF3CD",
                    "⚠ WARNING — Possible real vehicles in filtered set",
                )
        else:
            _insight_box(
                "No tracks were suppressed by the ghost-track filter in this run.  "
                "All tracks had at least the required number of updates.",
                "F0FFF0",
                "✓ Ghost-Track Filter — Nothing filtered",
            )

    # ══════════════════════════════════════════════════════════════════
    # SENSOR σ CALIBRATION CHECK
    # ══════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_heading(f"{next_num}. Sensor σ Calibration Check", 1)
    next_num += 1
    doc.add_paragraph(
        "Compares each sensor's declared measurement noise (sigma_m) against actual "
        "error from ground truth.  A well-calibrated sensor has mean actual error ≈ mean "
        "declared σ.  If actual error > 1.2×σ the filter over-trusts that sensor and the "
        "R-matrix entry should be increased."
    )
    cal = enhanced["sigma_calibration"]
    _data_table(cal["headers"], cal["rows"],
                status_col=len(cal["headers"]) - 1)
    doc.add_paragraph()
    for flag in cal["flags"]:
        if flag["sev"] == "warning":
            _insight_box(flag["text"], "FFF5F5", "⚠ CALIBRATION WARNING")
        elif flag["sev"] == "ok":
            _bullet_colored(flag["text"], (0x37, 0x56, 0x23))
        else:
            _bullet_colored(flag["text"], (0x2E, 0x75, 0xB6))

    # ══════════════════════════════════════════════════════════════════
    # PER-AXIS ERROR BREAKDOWN
    # ══════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_heading(f"{next_num}. Per-Axis Error Breakdown (X vs Y)", 1)
    next_num += 1
    doc.add_paragraph(
        "Scalar RMSE hides directional bias.  A dominant Y error (lateral) usually "
        "points to insufficient lateral process noise (Q_yy too small) or missing "
        "sensors that constrain cross-track position.  A dominant X error suggests "
        "speed-model mismatch."
    )
    for ax in enhanced["axis_errors"]:
        lbl = f"Track {ax['gid']}" + (f" — {ax['vid']}" if ax["vid"] else "")
        doc.add_heading(lbl, 2)
        _data_table(ax["headers"], ax["rows"])
        col = (0xB8, 0x86, 0x0B) if ax["verdict"].startswith("⚠") else (0x37, 0x56, 0x23)
        _bullet_colored(ax["verdict"], col)

    # ══════════════════════════════════════════════════════════════════
    # SENSOR COVERAGE TIMELINE
    # ══════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_heading(f"{next_num}. Sensor Coverage Timeline", 1)
    next_num += 1
    doc.add_paragraph(
        "Each row = one second of simulation.  "
        "✓ = at least one accepted measurement;  — = sensor silent (filter coasted).  "
        "Seconds where ALL sensors are silent are the primary source of position drift."
    )
    for tl in enhanced["coverage_timeline"]:
        lbl = f"Track {tl['gid']}" + (f" — {tl['vid']}" if tl["vid"] else "")
        doc.add_heading(lbl, 2)
        _data_table(tl["headers"], tl["rows"])
        tot = tl["total_seconds"]
        pct2 = (100 * tl["dead_seconds"] // max(tot, 1))
        doc.add_paragraph(
            f"Dead seconds (all sensors silent): "
            f"{tl['dead_seconds']} / {tot} ({pct2}%)"
        ).italic = True

    # ══════════════════════════════════════════════════════════════════
    # COLD-START LATENCY
    # ══════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_heading(f"{next_num}. Cold-Start Latency", 1)
    next_num += 1
    doc.add_paragraph(
        "Time from the first trajectory row to the first moment the position error "
        "drops below 0.5 m.  A long cold-start indicates the filter is initialized "
        "far from the true position or the initial covariance P₀ is too small."
    )
    cs = enhanced["cold_start"]
    _data_table(cs["headers"], cs["rows"])

    # ══════════════════════════════════════════════════════════════════
    # SPEED VS POSITION ERROR
    # ══════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_heading(f"{next_num}. Speed vs Position Error", 1)
    doc.add_paragraph(
        "Position error bucketed by estimated vehicle speed.  If error grows with "
        "speed, the process noise Q or motion model is misspecified for high-speed "
        "dynamics.  If error is flat, sensor dropout is the dominant driver."
    )
    spd = enhanced["speed_vs_error"]
    _data_table(spd["headers"], spd["rows"])
    col = (0xB8, 0x86, 0x0B) if spd["verdict"].startswith("⚠") else (0x37, 0x56, 0x23)
    _bullet_colored(spd["verdict"], col)

    # ── Save ──────────────────────────────────────────────────────────
    doc.save(str(p))
    _log.info("Word report written to %s", p)
    return p


def _compute_enhanced_diagnostics(
    audit_rows: List["AuditRow"],
    traj_rows: List["TrajectoryRow"],
    meaningful_gids: List[str],
    by_gid: Dict[str, List["TrajectoryRow"]],
    stats_by_gid: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute enhanced diagnostic data for the JS renderer.

    Returns a JSON-serialisable dict with five diagnostic blocks:
    sigma_calibration, axis_errors, coverage_timeline, cold_start,
    speed_vs_error.  No document object is touched — pure data computation.
    """
    import math as _math

    # Ground-truth lookup: (gid, t_rounded) → (tx, ty)
    _truth: Dict[Tuple[str, float], Tuple[float, float]] = {}
    for _r in traj_rows:
        if _r.global_track_id and _r.true_x is not None and _r.true_y is not None:
            _truth[(_r.global_track_id, round(_r.t, 6))] = (_r.true_x, _r.true_y)

    def _nearest_truth(gid: str, t: float) -> Optional[Tuple[float, float]]:
        key = (gid, round(t, 6))
        if key in _truth:
            return _truth[key]
        best_xy, best_dt = None, 0.15
        for (g, rt), xy in _truth.items():
            if g == gid:
                dt = abs(rt - t)
                if dt < best_dt:
                    best_xy, best_dt = xy, dt
        return best_xy

    # ==================================================================
    # 1. SENSOR σ CALIBRATION CHECK
    # ==================================================================
    _ss: Dict[str, Dict[str, Any]] = {}
    for ar in audit_rows:
        if not ar.accepted or ar.x is None or ar.y is None:
            continue
        truth_xy = _nearest_truth(ar.global_track_id, ar.t)
        if truth_xy is None:
            continue
        actual_err = _math.sqrt((ar.x - truth_xy[0]) ** 2 + (ar.y - truth_xy[1]) ** 2)
        st = _ss.setdefault(ar.sensor_type, {
            "n": 0, "sum_err": 0.0, "sum_sq": 0.0,
            "max_err": 0.0, "sum_sigma": 0.0, "n_over": 0,
        })
        st["n"] += 1
        st["sum_err"] += actual_err
        st["sum_sq"] += actual_err ** 2
        if actual_err > st["max_err"]:
            st["max_err"] = actual_err
        st["sum_sigma"] += ar.sigma_m
        if actual_err > 1.2 * ar.sigma_m:
            st["n_over"] += 1

    _cal_hdr = [
        "Sensor", "N meas.", "Mean actual err (m)", "Max actual err (m)",
        "RMSE (m)", "Mean declared σ (m)", "Actual / σ ratio",
        "% over-trusted", "Verdict",
    ]
    _cal_rows: List[List[str]] = []
    _cal_flags: List[Dict[str, str]] = []
    for _stype in ("GPS", "Camera", "DAS"):
        st = _ss.get(_stype)
        if not st or st["n"] == 0:
            _cal_rows.append([_stype, "0"] + ["n/a"] * 7)
            continue
        n = st["n"]
        mean_err = st["sum_err"] / n
        rmse = _math.sqrt(st["sum_sq"] / n)
        mean_sig = st["sum_sigma"] / n
        ratio = mean_err / mean_sig if mean_sig > 0 else float("inf")
        pct_over = 100.0 * st["n_over"] / n
        if ratio > 1.2:
            verdict = "⚠ Over-trusted"
        elif ratio < 0.8:
            verdict = "ℹ Conservative"
        else:
            verdict = "✓ Well-calibrated"
        _cal_rows.append([
            _stype, str(n),
            f"{mean_err:.3f}", f"{st['max_err']:.3f}", f"{rmse:.3f}",
            f"{mean_sig:.3f}", f"{ratio:.2f}",
            f"{pct_over:.1f}%", verdict,
        ])
        if mean_sig > 0:
            if ratio > 1.2:
                _cal_flags.append({
                    "sev": "warning",
                    "text": (
                        f"{_stype}: actual error is {ratio:.2f}× declared σ "
                        f"(mean err {mean_err:.2f} m vs σ {mean_sig:.2f} m). "
                        f"Increase R_{_stype.lower()} in the Kalman noise matrix."
                    ),
                })
            elif ratio < 0.8:
                _cal_flags.append({
                    "sev": "info",
                    "text": (
                        f"{_stype}: σ is conservative (actual/σ = {ratio:.2f}). "
                        f"You can reduce R_{_stype.lower()} to give this sensor more weight."
                    ),
                })
            else:
                _cal_flags.append({
                    "sev": "ok",
                    "text": f"{_stype}: calibration healthy (actual/σ = {ratio:.2f}).",
                })

    sigma_calibration = {
        "headers": _cal_hdr,
        "rows": _cal_rows,
        "flags": _cal_flags,
    }

    # ==================================================================
    # 2. PER-AXIS ERROR BREAKDOWN (X vs Y)
    # ==================================================================
    axis_errors: List[Dict[str, Any]] = []
    for gid in meaningful_gids:
        _grows = sorted(by_gid[gid], key=lambda r: r.t)
        _xe = [r.x_hat - r.true_x for r in _grows
               if r.x_hat is not None and r.true_x is not None]
        _ye = [r.y_hat - r.true_y for r in _grows
               if r.y_hat is not None and r.true_y is not None]
        if not _xe:
            continue
        _n = len(_xe)
        mean_xe = sum(abs(e) for e in _xe) / _n
        mean_ye = sum(abs(e) for e in _ye) / _n
        rmse_x = _math.sqrt(sum(e ** 2 for e in _xe) / _n)
        rmse_y = _math.sqrt(sum(e ** 2 for e in _ye) / _n)
        bias_x = sum(_xe) / _n
        bias_y = sum(_ye) / _n

        _dom = "Y (lateral)" if mean_ye > mean_xe else "X (longitudinal)"
        _denom = min(mean_ye, mean_xe) if min(mean_ye, mean_xe) > 1e-9 else 1e-9
        _ratio_ax = max(mean_ye, mean_xe) / _denom
        if _ratio_ax > 1.5:
            _advice = (
                "Consider increasing Q_yy (lateral process noise) so the filter "
                "tracks lateral motion more responsively."
                if _dom == "Y (lateral)"
                else "Consider increasing Q_xx (longitudinal process noise) or "
                     "reviewing the vehicle speed model."
            )
            verdict = f"⚠ {_dom} error dominates by {_ratio_ax:.1f}×. {_advice}"
        else:
            verdict = "✓ X and Y errors are balanced — no strong axis bias."

        axis_errors.append({
            "gid": gid,
            "vid": stats_by_gid[gid].get("oracle_vid", ""),
            "headers": ["Axis", "Mean |error| (m)", "RMSE (m)", "Signed bias (m)", "Dominant?"],
            "rows": [
                ["X (longitudinal)",
                 f"{mean_xe:.3f}", f"{rmse_x:.3f}", f"{bias_x:+.3f}",
                 "Yes" if mean_xe > mean_ye else "—"],
                ["Y (lateral)",
                 f"{mean_ye:.3f}", f"{rmse_y:.3f}", f"{bias_y:+.3f}",
                 "Yes" if mean_ye >= mean_xe else "—"],
            ],
            "verdict": verdict,
        })

    # ==================================================================
    # 3. SENSOR COVERAGE TIMELINE (1-second bins)
    # ==================================================================
    coverage_timeline: List[Dict[str, Any]] = []
    for gid in meaningful_gids:
        _grows = sorted(by_gid[gid], key=lambda r: r.t)
        if not _grows:
            continue
        _t0 = int(_grows[0].t)
        _t1 = int(_grows[-1].t) + 1
        _bins: Dict[int, Dict[str, bool]] = {
            _sec: {"GPS": False, "Camera": False, "DAS": False}
            for _sec in range(_t0, _t1 + 1)
        }
        for _r in _grows:
            _sec = int(_r.t)
            if _sec not in _bins:
                continue
            if _r.gps_x is not None:
                _bins[_sec]["GPS"] = True
            if _r.cam_x is not None:
                _bins[_sec]["Camera"] = True
            if _r.das_x is not None:
                _bins[_sec]["DAS"] = True

        _tl_hdr = ["Second (s)", "GPS", "Camera", "DAS", "Status"]
        _tl_data: List[List[str]] = []
        _dead = 0
        for _sec in sorted(_bins.keys()):
            _b = _bins[_sec]
            _any = _b["GPS"] or _b["Camera"] or _b["DAS"]
            if not _any:
                _dead += 1
            _tl_data.append([
                f"{_sec}–{_sec+1}",
                "✓" if _b["GPS"] else "—",
                "✓" if _b["Camera"] else "—",
                "✓" if _b["DAS"] else "—",
                "✓ sensor active" if _any else "✗ prediction only",
            ])

        coverage_timeline.append({
            "gid": gid,
            "vid": stats_by_gid[gid].get("oracle_vid", ""),
            "headers": _tl_hdr,
            "rows": _tl_data,
            "dead_seconds": _dead,
            "total_seconds": len(_bins),
        })

    # ==================================================================
    # 4. COLD-START LATENCY
    # ==================================================================
    _cs_hdr = [
        "Track", "Vehicle", "Track start (s)",
        "First error < 0.5 m at (s)", "Cold-start latency (s)", "Initial error (m)",
    ]
    _cs_rows_data: List[List[str]] = []
    for gid in meaningful_gids:
        _grows = sorted(by_gid[gid], key=lambda r: r.t)
        _err_rows = [r for r in _grows if r.pos_err_m is not None]
        _vid = stats_by_gid[gid].get("oracle_vid", "")
        if not _err_rows:
            _cs_rows_data.append([gid, _vid, "n/a", "n/a", "n/a", "n/a"])
            continue
        _t_start = _err_rows[0].t
        _init_err = _err_rows[0].pos_err_m
        _conv = next((r for r in _err_rows if r.pos_err_m < 0.5), None)
        if _conv is not None:
            _lat = _conv.t - _t_start
            _cs_rows_data.append([
                gid, _vid,
                f"{_t_start:.3f}", f"{_conv.t:.3f}",
                f"{_lat:.3f} s", f"{_init_err:.3f}",
            ])
        else:
            _cs_rows_data.append([
                gid, _vid,
                f"{_t_start:.3f}", "never < 0.5 m",
                "n/a", f"{_init_err:.3f}",
            ])

    cold_start = {"headers": _cs_hdr, "rows": _cs_rows_data}

    # ==================================================================
    # 5. SPEED VS POSITION ERROR
    # ==================================================================
    _buckets: Dict[str, List[float]] = {
        "0–5 m/s":   [],
        "5–10 m/s":  [],
        "10–15 m/s": [],
        "15–20 m/s": [],
        "20+ m/s":   [],
    }
    for _r in traj_rows:
        if _r.vx_hat is None or _r.vy_hat is None or _r.pos_err_m is None:
            continue
        _spd = _math.sqrt(_r.vx_hat ** 2 + _r.vy_hat ** 2)
        if _spd < 5:
            _buckets["0–5 m/s"].append(_r.pos_err_m)
        elif _spd < 10:
            _buckets["5–10 m/s"].append(_r.pos_err_m)
        elif _spd < 15:
            _buckets["10–15 m/s"].append(_r.pos_err_m)
        elif _spd < 20:
            _buckets["15–20 m/s"].append(_r.pos_err_m)
        else:
            _buckets["20+ m/s"].append(_r.pos_err_m)

    _spd_hdr = ["Speed bucket", "N rows", "Mean error (m)", "Max error (m)", "Trend vs prev"]
    _spd_data: List[List[str]] = []
    _prev_mean: Optional[float] = None
    for _label, _errs in _buckets.items():
        if not _errs:
            _spd_data.append([_label, "0", "—", "—", "—"])
            _prev_mean = None
            continue
        _me = sum(_errs) / len(_errs)
        _mx = max(_errs)
        if _prev_mean is None:
            _trend = "—"
        elif _me > _prev_mean * 1.1:
            _trend = "↑ growing"
        elif _me < _prev_mean * 0.9:
            _trend = "↓ shrinking"
        else:
            _trend = "→ stable"
        _spd_data.append([_label, str(len(_errs)), f"{_me:.3f}", f"{_mx:.3f}", _trend])
        _prev_mean = _me

    _populated = [
        (lbl, sum(errs) / len(errs))
        for lbl, errs in _buckets.items()
        if errs
    ]
    if len(_populated) >= 2 and _populated[-1][1] > _populated[0][1] * 1.2:
        _spd_verdict = (
            f"⚠ Error grows from {_populated[0][1]:.2f} m "
            f"(at {_populated[0][0]}) to {_populated[-1][1]:.2f} m "
            f"(at {_populated[-1][0]}) — process noise Q likely needs "
            "to be increased for higher speeds, especially the acceleration term."
        )
    elif _populated:
        _spd_verdict = (
            "✓ Error does not strongly correlate with speed — "
            "sensor dropout is the primary error driver, not model mismatch."
        )
    else:
        _spd_verdict = "No speed data available."

    speed_vs_error = {
        "headers": _spd_hdr,
        "rows": _spd_data,
        "verdict": _spd_verdict,
    }

    return {
        "sigma_calibration": sigma_calibration,
        "axis_errors": axis_errors,
        "coverage_timeline": coverage_timeline,
        "cold_start": cold_start,
        "speed_vs_error": speed_vs_error,
    }


# ---------------------------------------------------------------------------
# PDF report (matplotlib PdfPages, vector quality)
# ---------------------------------------------------------------------------


def write_report_pdf(
    audit_rows: List[AuditRow],
    traj_rows: List[TrajectoryRow],
    path: Any,
    *,
    sim_name: str = "",
    events: Optional[List[Any]] = None,
    world: Any = None,
) -> Path:
    """Write a multi-page PDF tracking report using matplotlib PdfPages.

    Pages:
        1.   Executive Summary (plain-English scorecard, new)
        2.   Simulation overview (styled table figure)
        3..N. Per-track: trajectory figure with anomaly vlines
        N+1. Segment transitions table (if any)
        N+2. Anomaly detail (text page per anomaly, up to cap)
        N+3. Filtered tracks table (if any)
        N+4… Diagnostic analysis sections (sensor coverage, sensor noise,
              spike investigations, fragmentation, segment transitions,
              camera/FOV, DAS, Kalman behavior, recommendations).
    """
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    except Exception as exc:
        raise RuntimeError("matplotlib is required for PDF export") from exc

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rd = _prepare_report_data(audit_rows, traj_rows)
    by_gid          = rd["by_gid"]
    meaningful_gids = rd["meaningful_gids"]
    skipped_gids    = rd["skipped_gids"]
    stats_by_gid    = rd["stats_by_gid"]
    anomalies_by_gid = rd["anomalies_by_gid"]
    cov_map         = rd["cov_map"]
    global_rmse     = rd["global_rmse"]
    global_max_err  = rd["global_max_err"]
    n_pred_only     = rd["n_pred_only"]
    all_transitions = rd["all_transitions"]
    transitions_by_gid = rd["transitions_by_gid"]
    total_anomalies = rd["total_anomalies"]

    # New analysis layer.
    from . import audit_analysis as _aa
    exec_summary, analysis_sections = _aa.build_full_analysis(
        audit_rows=audit_rows,
        traj_rows=traj_rows,
        by_gid=by_gid,
        stats_by_gid=stats_by_gid,
        anomalies_by_gid=anomalies_by_gid,
        skipped_gids=skipped_gids,
        coverage=rd["cov"],
        transitions_by_gid=transitions_by_gid,
        events=events,
        world=world,
    )

    def _text_fig(lines: List[str], *, title: str = "") -> "Figure":
        """Single-page text figure — used for anomaly/summary pages."""
        fig = Figure(figsize=(11, 8.5))
        FigureCanvas(fig)
        ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        y = 0.97
        for line in lines:
            bold = line.startswith("**") and line.endswith("**")
            txt = line[2:-2] if bold else line
            ax.text(0.0, y, txt, transform=ax.transAxes, fontsize=9,
                    va="top", fontweight="bold" if bold else "normal",
                    wrap=True, family="monospace" if line.startswith("  ") else "sans-serif")
            y -= 0.045
            if y < 0.02:
                break
        return fig

    def _table_fig(col_headers: List[str], rows_data: List[List[str]], title: str = "") -> "Figure":
        """Render a table as a matplotlib figure page."""
        fig = Figure(figsize=(11, 8.5))
        FigureCanvas(fig)
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.88])
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        if not rows_data:
            ax.text(0.5, 0.5, "(none)", ha="center", va="center", transform=ax.transAxes, color="#888888")
            return fig
        cell_data = [col_headers] + rows_data
        n_cols = len(col_headers)
        col_widths = [1.0 / n_cols] * n_cols
        tbl = ax.table(cellText=cell_data, colWidths=col_widths, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        for j in range(n_cols):
            cell = tbl[0, j]
            cell.set_facecolor("#2E75B6")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        for i in range(1, len(cell_data)):
            bg = "#EBF3FB" if i % 2 == 0 else "white"
            for j in range(n_cols):
                tbl[i, j].set_facecolor(bg)
        return fig

    # --- Build PDF ----------------------------------------------------------
    with PdfPages(str(p)) as pdf:

        # Page 1: Executive Summary (plain English).
        es_lines: List[str] = []
        es_lines.append("**Executive Summary**")
        es_lines.append("")
        # Wrap the paragraph so it fits on the page.
        import textwrap as _tw
        for chunk in _tw.wrap(exec_summary.plain_english_paragraph, width=110):
            es_lines.append(chunk)
        es_lines.append("")
        es_lines.append("**Scorecard**")
        for line in exec_summary.bullets:
            es_lines.append(f"  • {line}")
        es_lines.append("")
        es_lines.append(
            "Note: claims below are tagged confirmed / likely / hypothesis / "
            "next-check so the reader can tell facts from speculation."
        )
        fig_es = _text_fig(es_lines, title="Tracking Audit Report — Executive Summary")
        pdf.savefig(fig_es, bbox_inches="tight")

        # Page 2: overview table.
        overview_rows = [
            ["Simulation", sim_name or "—"],
            ["Meaningful tracks", str(len(meaningful_gids))],
            ["Skipped tracks", str(len(skipped_gids))],
            ["Segment transitions", str(len(all_transitions))],
            ["Total anomalies (capped)", str(total_anomalies)],
            ["Global RMSE", f"{global_rmse:.3f} m" if global_rmse is not None else "n/a"],
            ["Global max error", f"{global_max_err:.3f} m" if global_max_err is not None else "n/a"],
            ["Prediction-only rows", str(n_pred_only)],
            ["GPS measurements", str(cov_map.get("gps", {}).get("n_used", "n/a"))],
            ["Camera measurements", str(cov_map.get("camera", {}).get("n_used", "n/a"))],
            ["DAS measurements", str(cov_map.get("das", {}).get("n_used", "n/a"))],
        ]
        fig_ov = _table_fig(["Field", "Value"], overview_rows, title="Simulation Tracking Overview")
        pdf.savefig(fig_ov, bbox_inches="tight")

        # Pages 2-N: per-track trajectory figures.
        for gid in meaningful_gids:
            gid_rows = sorted(by_gid[gid], key=lambda r: r.t)
            anomalies = anomalies_by_gid.get(gid, [])
            a_times = [float(a.get("t", 0.0)) for a in anomalies if a.get("t") is not None]
            try:
                fig_track = _build_track_figure(
                    traj_rows, gid,
                    anomaly_markers=a_times or None,
                )
                pdf.savefig(fig_track, bbox_inches="tight")
            except Exception as exc:
                fig_err = _text_fig([f"Figure unavailable: {exc}"], title=f"Track {gid}")
                pdf.savefig(fig_err, bbox_inches="tight")

        # Segment transitions page.
        if all_transitions:
            tr_rows = [
                [str(gid), f"{tr['t']:.2f}", tr["from_segment"], tr["to_segment"]]
                for gid in meaningful_gids
                for tr in stats_by_gid[gid].get("transitions", [])
            ]
            fig_tr = _table_fig(
                ["Track", "Time (s)", "From segment", "To segment"],
                tr_rows,
                title="Segment Transitions",
            )
            pdf.savefig(fig_tr, bbox_inches="tight")

        # Anomaly detail page.
        if total_anomalies > 0:
            anom_lines: List[str] = []
            for gid in meaningful_gids:
                for a in anomalies_by_gid.get(gid, []):
                    gid_rows = sorted(by_gid[gid], key=lambda r: r.t)
                    narrative = _build_narrative(a, gid_rows, audit_rows)
                    t_anchor = a.get("t_start", a.get("t", None))
                    try:
                        t_str = f"{float(t_anchor):.2f}s"
                    except (TypeError, ValueError):
                        t_str = "?"
                    anom_lines.append(
                        f"**Track {gid} — {a.get('kind', '?')} @ t={t_str}**"
                    )
                    anom_lines.append(f"  {narrative}")
                    anom_lines.append("")
            fig_anom = _text_fig(anom_lines, title="Anomaly Detail")
            pdf.savefig(fig_anom, bbox_inches="tight")

        # Skipped tracks page.
        if skipped_gids:
            sk_rows = [[str(g), r] for g, r in sorted(skipped_gids)]
            fig_sk = _table_fig(["Track ID", "Skip reason"], sk_rows, title="Filtered / Skipped Tracks")
            pdf.savefig(fig_sk, bbox_inches="tight")

        # New analysis layer — render each Section.
        for section in analysis_sections:
            _pdf_render_section(
                pdf=pdf,
                section=section,
                text_fig=_text_fig,
                table_fig=_table_fig,
            )

    _log.info("PDF report written to %s", p)
    return p


def _pdf_render_section(
    *,
    pdf: Any,
    section: Any,
    text_fig: Any,
    table_fig: Any,
) -> None:
    """Render one :class:`audit_analysis.Section` into the PDF.

    A header / summary / bullets text page is followed by an optional
    table page, then one text page per Interpretation containing the
    technical / meaning / evidence / hypotheses / next-checks blocks.
    """
    import textwrap as _tw

    lines: List[str] = []
    lines.append(f"**{section.title}**")
    lines.append("")
    if section.summary:
        for ch in _tw.wrap(section.summary, width=110):
            lines.append(ch)
        lines.append("")
    for b in section.bullets:
        for ch in _tw.wrap(f"• {b}", width=108, subsequent_indent="  "):
            lines.append(ch)
    fig_intro = text_fig(lines, title=section.title)
    pdf.savefig(fig_intro, bbox_inches="tight")

    # Table.
    if section.table and section.table.get("rows"):
        fig_tbl = table_fig(
            section.table["headers"],
            [[str(v) for v in row] for row in section.table["rows"]],
            title=f"{section.title} — table",
        )
        pdf.savefig(fig_tbl, bbox_inches="tight")

    # Interpretations.
    for interp in section.interpretations:
        sev_marker = {"info": "[INFO]", "warning": "[WARN]", "critical": "[CRIT]"}.get(
            interp.severity, "[NOTE]"
        )
        ilines: List[str] = []
        ilines.append(f"**{sev_marker}  {interp.label}**")
        ilines.append("")
        ilines.append("**What happened**")
        for ch in _tw.wrap(interp.technical, width=108):
            ilines.append(f"  {ch}")
        ilines.append("")
        ilines.append("**What it probably means**")
        for ch in _tw.wrap(interp.meaning, width=108):
            ilines.append(f"  {ch}")
        ilines.append("")
        ilines.append("**Why it matters**")
        for ch in _tw.wrap(interp.why_matters, width=108):
            ilines.append(f"  {ch}")
        if interp.evidence:
            ilines.append("")
            ilines.append("**Confirmed evidence**")
            for e in interp.evidence:
                for ch in _tw.wrap(f"• {e}", width=108, subsequent_indent="  "):
                    ilines.append(f"  {ch}")
        if interp.hypotheses:
            ilines.append("")
            ilines.append("**Possible explanations (hypotheses)**")
            for h in interp.hypotheses:
                for ch in _tw.wrap(f"• {h}", width=108, subsequent_indent="  "):
                    ilines.append(f"  {ch}")
        if interp.next_checks:
            ilines.append("")
            ilines.append("**Recommended next checks**")
            for n in interp.next_checks:
                for ch in _tw.wrap(f"• {n}", width=108, subsequent_indent="  "):
                    ilines.append(f"  {ch}")
        fig_i = text_fig(ilines, title=f"{section.title} — {interp.label}")
        pdf.savefig(fig_i, bbox_inches="tight")


# ---------------------------------------------------------------------------
# One-shot convenience entry point used by the GUI
# ---------------------------------------------------------------------------


def export_all(
    events: Iterable,
    out_dir: Any,
    world: Any = None,
    gids: Optional[Iterable[str]] = None,
    progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Write all audit artefacts to *out_dir* and return their paths.

    Files produced:

    * ``tracking_audit.xlsx``           — **consolidated workbook** (human-readable):
      one sheet per artefact (Coverage, Kalman Audit, one Track sheet each).
    * ``kalman_measurement_audit.xlsx`` — raw measurement audit (machine-friendly).
    * ``coverage_summary.xlsx``         — per-sensor counters.
    * ``track_trajectory_<gid>.png``    — one PNG per chosen gid (best-effort;
      skipped if matplotlib is unavailable).
    * ``track_xy_<gid>.png``            — 2-D XY spatial PNG per chosen gid.

    Parameters
    ----------
    progress_cb : callable(str) | None
        Optional callback invoked with a short status string at each major
        stage so callers (e.g. the GUI status bar) can show progress.
    """
    import concurrent.futures as _cf

    def _progress(msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(msg)
            except Exception:
                pass
        _log.info("export_all: %s", msg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Materialise events once — needed twice (build_audit_and_trajectory
    # plus the new analysis layer for oracle-segment timeline / DAS SNR).
    events = list(events)

    _progress("Replaying tracker… (building audit + trajectory rows)")
    audit_rows, traj_rows = build_audit_and_trajectory(events, world=world)

    # Pre-group traj_rows by gid once — avoids O(N_tracks × N_traj_rows)
    # repeated scans in write_audit_workbook and the PNG rendering loop.
    _traj_by_gid_export: Dict[str, List[Any]] = {}
    for _r in traj_rows:
        if _r.global_track_id:
            _traj_by_gid_export.setdefault(_r.global_track_id, []).append(_r)

    _progress("Writing measurement audit workbook…")
    audit_path = write_audit_csv(audit_rows, out / "kalman_measurement_audit.xlsx")
    coverage = coverage_summary(audit_rows, traj_rows)
    coverage_path = write_coverage_csv(coverage, out / "coverage_summary.xlsx")

    # Consolidated human-readable workbook (all tables in one file).
    _progress("Writing consolidated tracking_audit.xlsx…")
    try:
        audit_workbook_path = write_audit_workbook(
            audit_rows, traj_rows, coverage,
            out / "tracking_audit.xlsx",
        )
    except Exception as exc:
        _log.error("Consolidated audit workbook failed: %s: %s", type(exc).__name__, exc)
        audit_workbook_path = None

    if gids is None:
        gids = sorted({r.global_track_id for r in traj_rows
                       if r.global_track_id})
    else:
        gids = list(gids)

    traj_paths: List[Path] = []
    png_paths: List[Path] = []
    n_pngs_created = 0
    n_pngs_skipped = 0
    png_skip_reasons: Dict[str, str] = {}

    # ── Determine which gids pass the quality filter ─────────────────────────
    # Pre-filter using the already-grouped dict to avoid an O(N_traj_rows)
    # scan per gid.
    meaningful_gids: List[str] = []
    for gid in gids:
        gid_rows = _traj_by_gid_export.get(gid, [])
        if EXPORT_FINAL_TRACK_PNGS_ONLY:
            ok, reason = _is_meaningful_track_for_png(gid_rows)
            if not ok:
                n_pngs_skipped += 1
                png_skip_reasons[str(gid)] = reason
                continue
        meaningful_gids.append(gid)

    # ── Render PNGs — parallelised across tracks ──────────────────────────────
    # Each worker renders trajectory + XY figures for one track.
    # write_trajectory_png / write_xy_png use the OO matplotlib Agg backend
    # which is thread-safe when each thread owns its own Figure objects.
    # Passing pre-filtered per-gid rows avoids repeated O(N_total) scans
    # inside _build_track_figure / segment_transitions.
    _progress(f"Rendering PNGs for {len(meaningful_gids)} tracks…")

    def _render_one_gid(gid: str) -> Tuple[str, Optional[Path], Optional[Exception], Optional[Exception]]:
        """Render trajectory + XY PNGs for one gid; return status tuple."""
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(gid))
        gid_rows = _traj_by_gid_export.get(gid, [])
        traj_exc: Optional[Exception] = None
        xy_exc: Optional[Exception] = None
        traj_p: Optional[Path] = None
        try:
            traj_p = write_trajectory_png(
                gid_rows, out / f"track_trajectory_{safe}.png", gid=gid,
            )
        except Exception as exc:
            traj_exc = exc
        try:
            write_xy_png(
                gid_rows, out / f"track_xy_{safe}.png", gid=gid,
            )
        except Exception as exc:
            xy_exc = exc
        return gid, traj_p, traj_exc, xy_exc

    n_workers = min(4, max(1, len(meaningful_gids)))
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_render_one_gid, gid): gid for gid in meaningful_gids}
        done_count = 0
        for fut in _cf.as_completed(futures):
            gid, traj_p, traj_exc, xy_exc = fut.result()
            done_count += 1
            if traj_exc is not None:
                _log.error("PNG export failed for global_track_id=%r: %s: %s",
                           gid, type(traj_exc).__name__, traj_exc)
            else:
                if traj_p is not None:
                    png_paths.append(traj_p)
                n_pngs_created += 1
            if xy_exc is not None:
                _log.error("XY PNG export failed for global_track_id=%r: %s: %s",
                           gid, type(xy_exc).__name__, xy_exc)
            if done_count % 3 == 0 or done_count == len(meaningful_gids):
                _progress(f"Rendering PNGs… {done_count}/{len(meaningful_gids)} tracks done")

    _log.info(
        "PNG export: %d created, %d skipped out of %d tracks",
        n_pngs_created, n_pngs_skipped, len(gids),
    )
    for _gid_s, _reason in png_skip_reasons.items():
        _log.info("  skipped gid=%s reason=%s", _gid_s, _reason)

    # Word report (post-run, oracle-aware).
    _progress("Writing Word report…")
    report_path: Optional[Path] = None
    if EXPORT_WORD_REPORT:
        try:
            report_path = write_report_docx(
                audit_rows,
                traj_rows,
                out / "tracking_audit_report.docx",
                sim_name=str(out_dir),
                events=events,
                world=world,
            )
        except Exception as exc:
            _log.error("Word report failed: %s: %s", type(exc).__name__, exc)

    # PDF report (matplotlib PdfPages).
    report_pdf_path: Optional[Path] = None
    if EXPORT_PDF_REPORT:
        try:
            report_pdf_path = write_report_pdf(
                audit_rows,
                traj_rows,
                out / "tracking_audit_report.pdf",
                sim_name=str(out_dir),
                events=events,
                world=world,
            )
        except Exception as exc:
            _log.error("PDF report failed: %s: %s", type(exc).__name__, exc)

    return {
        "audit_workbook": audit_workbook_path,   # consolidated multi-sheet Excel
        "audit_csv": audit_path,
        "coverage_csv": coverage_path,
        "trajectory_csvs": traj_paths,
        "trajectory_pngs": png_paths,
        "n_audit_rows": len(audit_rows),
        "n_traj_rows": len(traj_rows),
        "n_gids": len(gids),
        "n_pngs_created": n_pngs_created,
        "n_pngs_skipped": n_pngs_skipped,
        "png_skip_reasons": png_skip_reasons,
        "report_docx": report_path,
        "report_pdf": report_pdf_path,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _infer_cadence_s(events: List[Any], default: float = 1.0 / 30.0) -> float:
    """Infer simulation cadence from world.vehicle_state Δt (median).

    Falls back to *default* when no truth stream is present.
    """
    per_vid: Dict[str, List[float]] = {}
    for ev in events:
        if getattr(ev, "topic", "") != "world.vehicle_state":
            continue
        p = getattr(ev, "payload", None) or {}
        if not isinstance(p, dict):
            continue
        vid = str(p.get("vehicle_id", "") or "")
        if not vid:
            continue
        try:
            per_vid.setdefault(vid, []).append(float(p.get("t", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    deltas: List[float] = []
    for ts in per_vid.values():
        ts.sort()
        for a, b in zip(ts, ts[1:]):
            d = b - a
            if d > 1e-6 and math.isfinite(d):
                deltas.append(d)
    if not deltas:
        return default
    deltas.sort()
    return deltas[len(deltas) // 2]


def _emit_prediction_rows_for_gap(
    *,
    traj_rows: List[TrajectoryRow],
    kf: LegacyKalmanFilter,
    gid: str,
    oracle_vid: str,
    truth_list: List[_Truth],
    t_start: float,
    t_end: float,
    cadence: float,
) -> None:
    """Emit ``TrajectoryRow``s with ``update_kind="prediction_only"``.

    A *shadow* :class:`LegacyKalmanFilter` is cloned from *kf* (same
    state and covariance, same sigma_q) and stepped forward in
    *cadence*-sized increments until ``t_end``.  *kf* itself is never
    touched here — the production filter still takes its monolithic
    ``predict(dt)`` at the next measurement update outside this helper.

    Mathematical note: stepping the shadow with N small predicts gives a
    covariance slightly larger than ``predict(N*dt)`` because Q scales
    non-linearly with dt.  The mean trajectory is exact (state matrix F
    is exact for constant-acceleration), and the *next* measurement
    update happens on the production filter, so production behaviour is
    unchanged.  The visible per-step σ inside the gap is therefore an
    upper bound on the production filter's σ — adequate for the
    intended dropout-evaluation use case.
    """
    if cadence <= 0 or not math.isfinite(cadence):
        return
    if t_end - t_start <= 1.5 * cadence:
        return  # nothing to emit at this cadence

    # Clone state into a shadow filter.
    shadow = LegacyKalmanFilter(dt=cadence, sigma_q=kf.sigma_q)
    shadow.x = kf.x.copy()
    shadow.P = kf.P.copy()
    shadow._initialized = True

    # Step at cadence; emit one prediction-only row per step.
    t_pred = t_start + cadence
    while t_pred < t_end - 1e-9:
        shadow.predict(dt=cadence)
        xh, vxh, axh, yh, vyh, ayh = [float(v) for v in shadow.x.reshape(-1)]
        sigma_pos = math.sqrt(max(0.0, 0.5 * (shadow.P[0, 0] + shadow.P[3, 3])))
        sigma_vel = math.sqrt(max(0.0, 0.5 * (shadow.P[1, 1] + shadow.P[4, 4])))
        truth = _nearest_truth(truth_list, t_pred)
        pos_err = (math.hypot(xh - truth.x, yh - truth.y)
                   if truth is not None else None)
        traj_rows.append(TrajectoryRow(
            t=t_pred,
            global_track_id=gid,
            vehicle_id_oracle=oracle_vid,
            segment_id="",
            true_x=(truth.x if truth else None),
            true_y=(truth.y if truth else None),
            true_distance_m=None,
            gps_x=None, gps_y=None, gps_distance_m=None,
            cam_x=None, cam_y=None, cam_distance_m=None,
            das_x=None, das_y=None,
            das_fiber_position_m=None,
            das_distance_m=None,
            x_hat=xh, y_hat=yh,
            vx_hat=vxh, vy_hat=vyh,
            ax_hat=axh, ay_hat=ayh,
            distance_hat_m=None,
            sigma_pos_m=sigma_pos, sigma_vel_mps=sigma_vel,
            pos_err_m=pos_err,
            update_kind="prediction_only",
        ))
        t_pred += cadence


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Best-effort numeric coercion that never raises."""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


__all__ = [
    "AuditRow",
    "TrajectoryRow",
    "build_audit_and_trajectory",
    "coverage_summary",
    "time_gaps",
    "segment_transitions",
    "write_audit_csv",
    "write_trajectory_csv",
    "write_coverage_csv",
    "write_audit_workbook",
    "write_trajectory_png",
    "write_xy_png",
    "write_report_docx",
    "write_report_pdf",
    "export_all",
]
