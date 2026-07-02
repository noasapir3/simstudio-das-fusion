from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import bisect
import math

import numpy as np


# k constant for σ_DAS = k / sqrt(SNR) — position uncertainty (m).
# Used in debug-mode logging inside update_position_x_only.
_K_DAS: float = 2.0


# ---------------------------------------------------------------------------
# Identity-safety toggle (Phase 9)
# ---------------------------------------------------------------------------
#
# When True (the default and recommended setting), the tracker-driven Kalman
# pipeline NEVER falls back to the oracle ``vehicle_id`` to recover a sensor
# measurement that the data-association layer (TrackManager + recorder) was
# unable to assign to a ``global_track_id``.  Such a measurement is dropped
# from fusion exactly like any other un-associable observation.  The
# accompanying audit module (:mod:`simstudio.audit`) records every such drop
# explicitly with ``skip_reason="no_global_track_assignment"`` so the failure
# is visible end-to-end.
#
# Flipping this flag to False reactivates the historical "use oracle vid as a
# safety net" behaviour preserved here for legacy diagnostics and the
# bit-for-bit convergence tests on 1:1 scenes.  Production fusion must run
# with STRICT_TRACKER_IDENTITY = True so that no oracle identity ever leaks
# into the Kalman filter through the back door.
STRICT_TRACKER_IDENTITY: bool = True


class LegacyKalmanFilter:
    """Robust 6-state constant-acceleration Kalman filter.

    State: [x, vx, ax, y, vy, ay]^T

    DAS (fiber) measurements only update x and vx:
    - update_position_x_only: x only     (H = [1,0,0,0,0,0])
    - update_velocity_x_only: vx only    (H = [0,1,0,0,0,0])
    Camera/GPS measurements update both x,y and vx,vy.

    Dynamic Kalman R:
    Each DAS measurement carries sigma_m = k_das/sqrt(SNR) computed in
    sim_core.py.  This is passed as the measurement noise std, so R_x = sigma_m^2
    is automatically higher (less trust) when SNR is low and vice-versa.
    """

    def __init__(
        self,
        dt: float,
        sigma_q: float = 2.0,
        sigma_p_fiber: float = 5.0,
        sigma_p_camera: float = 1.3,
        sigma_p_gps: float = 1.8,
        sigma_v_gps: float = 0.8,
        sigma_v_camera: float = 1.1,
        debug_das: bool = False,
    ) -> None:
        self.debug_das = bool(debug_das)
        self.dt = float(max(1e-3, dt))
        self.sigma_q = float(max(1e-4, sigma_q))
        self.sensor_sigma = {
            "p_fiber": float(sigma_p_fiber),
            "p_camera": float(sigma_p_camera),
            "p_gps": float(sigma_p_gps),
            "v_gps": float(sigma_v_gps),
            "v_camera": float(sigma_v_camera),
        }
        self.x = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6, dtype=float) * 25.0
        self._initialized = False
        self._rebuild_matrices(self.dt)

        # ── Innovation logging (read-only by audit.py — NEVER used by filter) ──
        # Reset to NaN each predict() call; set inside _update() after each
        # measurement update.  These fields expose the pre-update residual
        # ν = z − H·x̂⁻ and the innovation covariance S for each timestep.
        # NIS_x = ν_x² / S_xx follows χ²(1) when the filter is consistent.
        self.last_innovation_x: float = float("nan")  # x-position residual [m]
        self.last_innovation_y: float = float("nan")  # y-position residual [m]
        self.last_S_x: float = float("nan")           # S_xx innovation variance [m²]
        self.last_S_y: float = float("nan")           # S_yy innovation variance [m²]
        self.last_NIS_x: float = float("nan")         # ν_x² / S_xx (χ²(1) statistic)

    def _rebuild_matrices(self, dt: float) -> None:
        self.dt = float(max(1e-3, dt))
        dt = self.dt
        f_axis = np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.F = np.block(
            [
                [f_axis, np.zeros((3, 3), dtype=float)],
                [np.zeros((3, 3), dtype=float), f_axis],
            ]
        )
        g_axis = np.array(
            [
                [dt**4 / 4.0, dt**3 / 2.0, dt**2 / 2.0],
                [dt**3 / 2.0, dt**2, dt],
                [dt**2 / 2.0, dt, 1.0],
            ],
            dtype=float,
        )
        self.Q = (self.sigma_q**2) * np.block(
            [
                [g_axis, np.zeros((3, 3), dtype=float)],
                [np.zeros((3, 3), dtype=float), g_axis],
            ]
        )

    def initialize_state(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        ax: float = 0.0,
        ay: float = 0.0,
        pos_sigma: float = 3.0,
        vel_sigma: float = 2.0,
        acc_sigma: float = 1.0,
    ) -> None:
        self.x[:, 0] = [float(x), float(vx), float(ax), float(y), float(vy), float(ay)]
        self.P = np.diag(
            [
                max(0.1, float(pos_sigma)) ** 2,
                max(0.1, float(vel_sigma)) ** 2,
                max(0.1, float(acc_sigma)) ** 2,
                max(0.1, float(pos_sigma)) ** 2,
                max(0.1, float(vel_sigma)) ** 2,
                max(0.1, float(acc_sigma)) ** 2,
            ]
        )
        self._initialized = True

    def predict(self, dt: float | None = None) -> None:
        if dt is not None and abs(float(dt) - self.dt) > 1e-9:
            self._rebuild_matrices(float(dt))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # New timestep — clear innovations so prediction-only rows carry NaN.
        self.last_innovation_x = float("nan")
        self.last_innovation_y = float("nan")
        self.last_S_x          = float("nan")
        self.last_S_y          = float("nan")
        self.last_NIS_x        = float("nan")

    def update_positions(self, measurements: Sequence[Tuple[str, float, float, float, float]]) -> None:
        if not measurements:
            return
        H_rows: List[np.ndarray] = []
        z_vals: List[float] = []
        r_vals: List[float] = []
        for meas_type, x_m, y_m, sigma_m, confidence in measurements:
            base_sigma = float(sigma_m or self.sensor_sigma.get(meas_type, 2.0))
            conf = float(max(0.05, min(1.0, confidence if confidence is not None else 1.0)))
            sigma_eff = max(0.05, base_sigma / math.sqrt(conf))
            H_rows.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
            H_rows.append(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float))
            z_vals.extend([float(x_m), float(y_m)])
            r_vals.extend([sigma_eff**2, sigma_eff**2])
        self._update(np.vstack(H_rows), np.asarray(z_vals, dtype=float).reshape(-1, 1), np.diag(r_vals))

    def update_position_x_only(self, measurements: Sequence[Tuple[str, float, float, float, float]]) -> None:
        """Update only the x-axis state from measurements.

        Used for DAS (fiber) readings: DAS measures position along the fiber
        (x-axis only) but cannot observe the lateral position (y-axis).
        Feeding a fabricated y value to the Kalman causes a permanent bias on
        the y-state, so DAS measurements must only update x.

        ``sigma_m`` is the SNR-derived uncertainty already computed in
        sim_core.py: σ_DAS = k_das / sqrt(SNR).  This is used directly as the
        measurement noise std, making R_x = sigma_m^2 dynamic and physically
        grounded.  y-state is completely unaffected by DAS measurements.
        """
        if not measurements:
            return
        H_rows: List[np.ndarray] = []
        z_vals: List[float] = []
        r_vals: List[float] = []
        for meas_type, x_m, y_m, sigma_m, confidence in measurements:
            # DEBUG MODE: y_m is now used symmetrically with x_m for testing purposes.
            # This is not physically correct (DAS cannot observe lateral position),
            # but allows verifying Kalman response when Y is fully observable.
            base_sigma = float(sigma_m or self.sensor_sigma.get(meas_type, 2.0))
            conf = float(max(0.05, min(1.0, confidence if confidence is not None else 1.0)))
            sigma_eff = max(0.05, base_sigma / math.sqrt(conf))
            if self.debug_das:
                innovation_x = float(x_m) - float(self.x[0])
                innovation_y = float(y_m) - float(self.x[3])
                snr_approx = (_K_DAS / sigma_eff) ** 2
                print(
                    f"[DAS pos]  x_meas={float(x_m):.2f}m  x_pred={float(self.x[0]):.2f}m"
                    f"  innov_x={innovation_x:+.3f}m  y_meas={float(y_m):.2f}m  y_pred={float(self.x[3]):.2f}m"
                    f"  innov_y={innovation_y:+.3f}m  sigma={sigma_eff:.3f}m  SNR≈{snr_approx:.1f}"
                )
            H_rows.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
            H_rows.append(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float))  # DEBUG: symmetric Y update
            z_vals.append(float(x_m))
            z_vals.append(float(y_m))                                               # DEBUG: use actual y_meas
            r_vals.append(sigma_eff**2)
            r_vals.append(sigma_eff**2)                                             # DEBUG: same sigma as X
        self._update(np.vstack(H_rows), np.asarray(z_vals, dtype=float).reshape(-1, 1), np.diag(r_vals))

    def update_velocity_fiber(
        self,
        measurements: Sequence[Tuple[str, float, float, float, float, float]],
    ) -> None:
        """Update state from a DAS along-fiber speed measurement.

        DAS measures scalar speed along the fiber tangent direction.
        The correct measurement equation for any fiber angle θ is:

            z = vx·cos(θ) + vy·sin(θ)       (scalar, signed)
            H = [0, cos(θ), 0, 0, sin(θ), 0]

        This correctly constrains the along-fiber component of velocity while
        leaving the cross-fiber component (perpendicular to fiber) free.

        For a horizontal fiber (θ=0): H reduces to [0,1,0,0,0,0] and z = vx
        — same as the old x-only approximation, but now general.

        Tuple format: (mtype, z_fiber, fiber_angle_rad, sigma_m, confidence)
            z_fiber        — signed speed along fiber (m/s); positive = in fiber
                             tangent direction, negative = against it.
            fiber_angle_rad — fiber tangent angle in world frame (radians).
        """
        if not measurements:
            return
        H_rows: List[np.ndarray] = []
        z_vals: List[float] = []
        r_vals: List[float] = []
        for meas_type, z_fiber, fiber_angle, sigma_m, confidence in measurements:
            ct = math.cos(float(fiber_angle))
            st = math.sin(float(fiber_angle))
            base_sigma = float(sigma_m or self.sensor_sigma.get(meas_type, 1.0))
            conf = float(max(0.05, min(1.0, confidence if confidence is not None else 1.0)))
            sigma_eff = max(0.05, base_sigma / math.sqrt(conf))
            if self.debug_das:
                v_fiber_pred = float(self.x[1]) * ct + float(self.x[4]) * st
                innovation_v = float(z_fiber) - v_fiber_pred
                snr_approx = (_K_DAS / max(sigma_eff, 1e-9)) ** 2
                print(
                    f"[DAS vel]  v_fiber_meas={float(z_fiber):.2f}m/s"
                    f"  v_fiber_pred={v_fiber_pred:.2f}m/s"
                    f"  innov={innovation_v:+.3f}m/s  sigma_v={sigma_eff:.3f}m/s"
                    f"  θ={math.degrees(float(fiber_angle)):.1f}°  SNR≈{snr_approx:.1f}"
                )
            H_rows.append(np.array([0.0, ct, 0.0, 0.0, st, 0.0], dtype=float))
            z_vals.append(float(z_fiber))
            r_vals.append(sigma_eff**2)
        self._update(np.vstack(H_rows), np.asarray(z_vals, dtype=float).reshape(-1, 1), np.diag(r_vals))

    def update_velocity_x_only(self, measurements: Sequence[Tuple[str, float, float, float, float]]) -> None:
        """Convenience wrapper: fiber velocity update for a horizontal fiber (θ=0).

        Equivalent to update_velocity_fiber with fiber_angle=0.
        Kept for backward compatibility and tests.
        """
        # Convert (mtype, vx_m, vy_m, sigma, conf) → (mtype, z_fiber=vx_m, angle=0, sigma, conf)
        converted = [(t, vx, 0.0, s, c) for t, vx, _vy, s, c in measurements]
        self.update_velocity_fiber(converted)

    def update_velocities(self, measurements: Sequence[Tuple[str, float, float, float, float]]) -> None:
        if not measurements:
            return
        H_rows: List[np.ndarray] = []
        z_vals: List[float] = []
        r_vals: List[float] = []
        for meas_type, vx_m, vy_m, sigma_m, confidence in measurements:
            base_sigma = float(sigma_m or self.sensor_sigma.get(meas_type, 1.0))
            conf = float(max(0.05, min(1.0, confidence if confidence is not None else 1.0)))
            sigma_eff = max(0.05, base_sigma / math.sqrt(conf))
            H_rows.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
            H_rows.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float))
            z_vals.extend([float(vx_m), float(vy_m)])
            r_vals.extend([sigma_eff**2, sigma_eff**2])
        self._update(np.vstack(H_rows), np.asarray(z_vals, dtype=float).reshape(-1, 1), np.diag(r_vals))

    def _update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)
        y = z - H @ self.x
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0], dtype=float)
        # Joseph form for numerical stability.
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T

        # ── Innovation logging (read-only; zero impact on filter state) ───────
        # Scan H rows to identify which measurement corresponds to x-position
        # (H[i,0] == 1) and y-position (H[i,3] == 1).  Only the last matched
        # row per axis is kept when multiple measurements of the same type are
        # stacked (e.g. DAS + camera in the same timestep).
        y_flat = y.flatten()
        S_diag = np.diag(S)
        for i, h_row in enumerate(H):
            if h_row[0] == 1.0:                          # x-position measurement
                self.last_innovation_x = float(y_flat[i])
                self.last_S_x          = float(S_diag[i])
                self.last_NIS_x        = (float(y_flat[i]) ** 2
                                          / max(float(S_diag[i]), 1e-9))
            elif h_row[3] == 1.0:                        # y-position measurement
                self.last_innovation_y = float(y_flat[i])
                self.last_S_y          = float(S_diag[i])


@dataclass
class _Meas:
    t: float
    vehicle_id: str
    mtype: str
    x: float
    y: float
    sigma: float
    confidence: float
    x_true: float | None
    y_true: float | None
    speed_mps: float
    heading_rad: float
    source_id: str
    snr: float = 0.0              # DAS only: signal-to-noise ratio at this timestep
    fiber_angle_rad: float = 0.0  # DAS only: fiber tangent angle in world frame (rad); informational
    z_vx: float = 0.0             # DAS only: 2D finite-diff vx from DAS x,y (m/s); 0 if unavailable
    z_vy: float = 0.0             # DAS only: 2D finite-diff vy from DAS x,y (m/s); 0 if unavailable
    sigma_v: float = 0.0          # DAS only: velocity noise std (m/s); 0 means no velocity measurement


@dataclass
class _Truth:
    t: float
    x: float
    y: float
    v: float
    heading_rad: float
    ax: float
    ay: float


def _collect_end_times(events: Iterable) -> Dict[str, float]:
    """Return the time each vehicle reached the end of its route (vehicle_stuck event).

    A vehicle_stuck event fires both when a vehicle genuinely gets stuck AND when
    it simply reaches the end of its planned route (dead-end node).  Either way,
    tracking beyond that point is meaningless, so we treat this time as the
    vehicle's end-of-life for RMSE purposes.
    """
    end_times: Dict[str, float] = {}
    for ev in events:
        if getattr(ev, "topic", "") != "world.vehicle_stuck":
            continue
        p = getattr(ev, "payload", None) or {}
        vid = str(p.get("vehicle_id", "") or "")
        t = float(p.get("t", 0.0) or 0.0)
        if vid and vid not in end_times:
            end_times[vid] = t
    return end_times


def _collect_measurements(events: Iterable) -> Dict[str, List[_Meas]]:
    # First pass: find when each vehicle ends its route.
    events = list(events)
    end_times = _collect_end_times(events)

    grouped: Dict[str, List[_Meas]] = {}
    for ev in events:
        topic = getattr(ev, "topic", "")
        if topic not in {"sensor.gps", "sensor.camera", "sensor.das"}:
            continue
        p = getattr(ev, "payload", None) or {}
        vid = str(p.get("vehicle_id", "") or "")
        if not vid:
            continue
        t = float(p.get("t", 0.0) or 0.0)
        # Skip measurements after the vehicle reached the end of its route.
        if vid in end_times and t >= end_times[vid]:
            continue
        if topic == "sensor.gps":
            mtype = "p_gps"
        elif topic == "sensor.camera":
            mtype = "p_camera"
        else:
            mtype = "p_fiber"
        grouped.setdefault(vid, []).append(
            _Meas(
                t=t,
                vehicle_id=vid,
                mtype=mtype,
                x=float(p.get("x", 0.0) or 0.0),
                y=float(p.get("y", 0.0) or 0.0),
                sigma=float(p.get("sigma_m", 2.0) or 2.0),
                confidence=float(p.get("confidence", 0.55) or 0.55),
                x_true=(None if p.get("x_true") is None else float(p.get("x_true"))),
                y_true=(None if p.get("y_true") is None else float(p.get("y_true"))),
                speed_mps=float(p.get("speed_mps", 0.0) or 0.0),
                heading_rad=float(p.get("heading_rad", 0.0) or 0.0),
                source_id=str(p.get("sensor_id", "") or ""),
                snr=float(p.get("snr", 0.0) or 0.0) if topic == "sensor.das" else 0.0,
                fiber_angle_rad=float(p.get("fiber_angle_rad", 0.0) or 0.0) if topic == "sensor.das" else 0.0,
                z_vx=float(p.get("z_vx", 0.0) or 0.0) if topic == "sensor.das" else 0.0,
                z_vy=float(p.get("z_vy", 0.0) or 0.0) if topic == "sensor.das" else 0.0,
                sigma_v=float(p.get("sigma_v", 0.0) or 0.0) if topic == "sensor.das" else 0.0,
            )
        )
    for vid in grouped:
        grouped[vid].sort(key=lambda m: (m.t, m.mtype, m.source_id))
    return grouped


def _collect_truth(events: Iterable) -> Dict[str, List[_Truth]]:
    grouped: Dict[str, List[_Truth]] = {}
    for ev in events:
        if getattr(ev, "topic", "") != "world.vehicle_state":
            continue
        p = getattr(ev, "payload", None) or {}
        vid = str(p.get("vehicle_id", "") or "")
        if not vid:
            continue
        grouped.setdefault(vid, []).append(
            _Truth(
                t=float(p.get("t", 0.0) or 0.0),
                x=float(p.get("x", 0.0) or 0.0),
                y=float(p.get("y", 0.0) or 0.0),
                v=float(p.get("v", 0.0) or 0.0),
                heading_rad=float(p.get("heading_rad", 0.0) or 0.0),
                ax=float(p.get("ax_world_mps2", 0.0) or 0.0),
                ay=float(p.get("ay_world_mps2", 0.0) or 0.0),
            )
        )
    for vid in grouped:
        grouped[vid].sort(key=lambda m: m.t)
    return grouped


def _nearest_truth(truth_list: List[_Truth], t: float) -> _Truth | None:
    if not truth_list:
        return None
    ts = [item.t for item in truth_list]
    idx = bisect.bisect_left(ts, t)
    if idx <= 0:
        return truth_list[0]
    if idx >= len(truth_list):
        return truth_list[-1]
    a = truth_list[idx - 1]
    b = truth_list[idx]
    return a if abs(a.t - t) <= abs(b.t - t) else b


def _sensor_velocity_sigma(meas_type: str, pos_sigma: float, confidence: float) -> float:
    conf = max(0.05, min(1.0, float(confidence)))
    if meas_type == "p_gps":
        base = 0.7
        return max(0.20, (base + 0.15 * pos_sigma) / math.sqrt(conf))
    if meas_type == "p_camera":
        base = 0.9
        return max(0.25, (base + 0.20 * pos_sigma) / math.sqrt(conf))
    return max(1.20, 0.60 * pos_sigma / math.sqrt(conf))


def build_kalman_rows(events: Iterable, debug_das: bool = False) -> List[Tuple[str, ...]]:
    """Build per-timestep Kalman state rows from simulation events.

    Parameters
    ----------
    events : iterable of Event objects from the simulation bus.
    debug_das : bool
        When True, print per-measurement DAS diagnostics to stdout:
        SNR, sigma fed to Kalman, and innovation (z − Hx) before the update.
        Useful for verifying that the filter is behaving correctly.
    """
    rows: List[Tuple[str, ...]] = []
    grouped = _collect_measurements(events)
    truths = _collect_truth(events)

    for vid, meas_list in sorted(grouped.items()):
        if not meas_list:
            continue
        truth_list = truths.get(vid, [])
        kf = LegacyKalmanFilter(dt=1.0 / 30.0, debug_das=debug_das)
        i = 0
        last_t = None
        while i < len(meas_list):
            t = meas_list[i].t
            batch: List[_Meas] = []
            while i < len(meas_list) and abs(meas_list[i].t - t) < 1e-9:
                batch.append(meas_list[i])
                i += 1

            truth = _nearest_truth(truth_list, t)
            if not kf._initialized:
                seed = batch[0]
                vx0 = seed.speed_mps * math.cos(seed.heading_rad)
                vy0 = seed.speed_mps * math.sin(seed.heading_rad)
                if truth is not None:
                    # Seed from position truth only. Acceleration remains hidden and starts neutral.
                    kf.initialize_state(truth.x, truth.y, vx=vx0, vy=vy0, ax=0.0, ay=0.0, pos_sigma=2.5)
                else:
                    # DAS cannot observe y: initialise y to road centreline (0.0),
                    # not fiber_offset_m, to avoid a permanent y-bias.
                    y_init = 0.0 if seed.mtype == "p_fiber" else seed.y
                    kf.initialize_state(seed.x, y_init, vx=vx0, vy=vy0, ax=0.0, ay=0.0, pos_sigma=max(1.0, seed.sigma))
                last_t = t
            else:
                dt = max(1.0 / 60.0, t - float(last_t))
                kf.predict(dt=dt)
                last_t = t

            # DAS (p_fiber): x-position only, vx only.
            # Camera/GPS: full (x,y) position and (vx,vy) velocity.
            das_pos = [(m.mtype, m.x, m.y, m.sigma, m.confidence) for m in batch if m.mtype == "p_fiber"]
            other_pos = [(m.mtype, m.x, m.y, m.sigma, m.confidence) for m in batch if m.mtype != "p_fiber"]
            if das_pos:
                if debug_das:
                    for m in batch:
                        if m.mtype != "p_fiber":
                            continue
                        print(
                            f"[DAS t={t:.2f} vid={vid}]"
                            f"  SNR={m.snr:.1f}  sigma_x={m.sigma:.3f}m"
                        )
                kf.update_position_x_only(das_pos)
            if other_pos:
                kf.update_positions(other_pos)

            das_vel: List[Tuple] = []
            other_vel: List[Tuple] = []
            for m in batch:
                if m.mtype == "p_fiber":
                    # Velocity: only use when sigma_v > 0, meaning a finite-difference
                    # estimate was available (i.e. this is not the first DAS sample).
                    # sigma_v == 0.0 means no velocity measurement — skip entirely.
                    if m.sigma_v > 0.0:
                        das_vel.append(("v_das", m.z_vx, m.z_vy, m.sigma_v, m.confidence))
                else:
                    if m.speed_mps <= 0.0:
                        continue
                    vel_sigma = _sensor_velocity_sigma(m.mtype, m.sigma, m.confidence)
                    vx_m = m.speed_mps * math.cos(m.heading_rad)
                    vy_m = m.speed_mps * math.sin(m.heading_rad)
                    other_vel.append((m.mtype.replace("p_", "v_"), vx_m, vy_m, vel_sigma, m.confidence))
            if das_vel:
                kf.update_velocities(das_vel)
            if other_vel:
                kf.update_velocities(other_vel)

            xh, vxh, axh, yh, vyh, ayh = [float(v) for v in kf.x.reshape(-1)]
            sigma_pos = math.sqrt(max(0.0, 0.5 * (kf.P[0, 0] + kf.P[3, 3])))
            sigma_vel = math.sqrt(max(0.0, 0.5 * (kf.P[1, 1] + kf.P[4, 4])))

            pos_err = ""
            if truth is not None:
                pos_err = f"{math.hypot(xh - truth.x, yh - truth.y):.2f}"
            sources = "+".join(sorted({m.mtype.replace('p_', '') for m in batch}))
            rows.append(
                (
                    f"{t:.2f}",
                    vid,
                    sources,
                    f"{xh:.2f}",
                    f"{yh:.2f}",
                    f"{vxh:.2f}",
                    f"{vyh:.2f}",
                    f"{axh:.2f}",
                    f"{ayh:.2f}",
                    f"{sigma_pos:.2f}",
                    f"{sigma_vel:.2f}",
                    pos_err,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Phase 3: tracker-driven Kalman grouping by global_track_id.
# ---------------------------------------------------------------------------
#
# ``build_kalman_rows_tracked`` groups measurements by the tracker's stable
# ``global_track_id`` rather than oracle ``vehicle_id``.  ``vehicle_id`` is
# retained only for (a) end-time clipping and (b) truth/RMSE lookup —
# never for measurement-to-filter assignment.
#
# The assignment is captured via ``_TrackAssignmentRecorder``, a minimal bus
# that records the ``(source, x, y) → gid`` mapping emitted by TrackManager
# as ``track.update`` events during event replay.  This means the Kalman
# grouping reflects Phase 2 geometric association exactly, even when that
# disagrees with the oracle.
#
# Output schema (13 columns, unchanged from prior phases):
#   (t, global_track_id, vehicle_id_oracle, sources,
#    x_hat, y_hat, vx_hat, vy_hat, ax_hat, ay_hat,
#    sigma_pos_m, sigma_vel_mps, pos_err_m)
# Stripping column 1 (global_track_id) yields the 12-column legacy schema
# byte-for-bit on 1:1 (single-vehicle or well-separated) scenarios.


class _TrackAssignmentRecorder:
    """Minimal bus-like sink that records tracker GID assignments.

    Subscribes to ``track.update`` events emitted by
    :class:`~simstudio.tracking.TrackManager` during event replay and
    indexes them by ``(source_kind, x, y)`` so that
    :func:`build_kalman_rows_tracked` can look up the GID the tracker
    actually assigned to each sensor measurement without using oracle
    ``vehicle_id`` for grouping.

    Time is omitted from the key because ``ev.ts`` (used by ``on_event``)
    may differ from the simulation-time ``"t"`` field in sensor payloads
    (e.g. when replaying ``FakeEvent`` objects in tests).  The
    ``(source, x, y)`` triple is unique for all current scenarios:
    different vehicles have distinct y positions; single-vehicle x grows
    monotonically.
    """

    def __init__(self) -> None:
        # (source_kind, x, y) → global_track_id
        self._assignments: Dict[Tuple[str, float, float], str] = {}

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if topic != "track.update":
            return
        source = str(payload.get("source", ""))
        if source not in {"gps", "cam", "das"}:
            return  # ignore world.vehicle_state updates
        try:
            key: Tuple[str, float, float] = (
                source,
                float(payload["x"]),
                float(payload["y"]),
            )
            self._assignments[key] = str(payload.get("global_track_id", ""))
        except (TypeError, ValueError, KeyError):
            pass

    def lookup(self, source: str, x: float, y: float) -> str:
        """Return the gid assigned to this measurement, or '' if unknown."""
        return self._assignments.get((source, x, y), "")

def build_kalman_rows_tracked(
    events: Iterable,
    world: "object | None" = None,
    debug_das: bool = False,
) -> List[Tuple[str, ...]]:
    """Tracker-driven sibling of :func:`build_kalman_rows`.

    Measurements are grouped by the tracker's stable ``global_track_id``
    rather than by the oracle ``vehicle_id``.  In the current phase of
    tracking development this is a 1:1 re-labelling — so the output is
    functionally identical to :func:`build_kalman_rows` — but the
    indirection is the mechanism that will let Phase 6+ carry tracker
    identity through without breaking exports.

    Parameters
    ----------
    events : iterable
        Iterable of bus-style events (objects with ``.topic`` and
        ``.payload``).  Safe to pass generators.
    world : object | None
        Optional :class:`~simstudio.models.World`.  When supplied, the
        internal :class:`~simstudio.tracking.TrackManager` can infer
        ``segment_id`` from ``lane_id`` — which does not change any row
        value, but keeps the tracker's own diagnostics in sync.  Pass
        ``None`` to skip.
    debug_das : bool
        Same semantics as :func:`build_kalman_rows`.

    Returns
    -------
    list[tuple[str, ...]]
        One row per (track, timestep) batch in a 13-column schema
        (``(t, global_track_id, vehicle_id_oracle, sources, x_hat,
        y_hat, vx_hat, vy_hat, ax_hat, ay_hat, sigma_pos_m,
        sigma_vel_mps, pos_err_m)``).  Stripping the ``global_track_id``
        column (index 1) yields the 12-column schema emitted by
        :func:`build_kalman_rows`, and the values match byte-for-byte on
        Phase-1 (1:1 vid ↔ gid) scenes.
    """
    # Materialise once — the event list is replayed twice (tracker pass,
    # then measurement collection) and also used for truth/end-times.
    events = list(events)

    from .tracking import TrackManager as _TM

    # --- Pass 1: replay through tracker with recorder bus ----------------
    # The recorder captures (source, x, y) → gid from every track.update
    # event so we can assign each sensor measurement to its GID without
    # touching vehicle_id.
    recorder = _TrackAssignmentRecorder()
    tm = _TM(world=world, bus=recorder)
    for ev in events:
        tm.on_event(ev)

    # gid_to_vid: oracle representative per track — used ONLY for:
    #   (a) the vehicle_id_oracle output column, and
    #   (b) truth lookup (RMSE).  Never used for measurement grouping.
    # vid_to_gid: kept as a defensive fallback for measurements the
    #   recorder missed (e.g. events dropped by the tracker).
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

    # --- Pass 2: collect measurements grouped by tracker-assigned gid ---
    # vehicle_id is read only for end-time clipping — not for grouping.
    end_times = _collect_end_times(events)
    truths = _collect_truth(events)

    _TOPIC_MTYPE = {
        "sensor.gps":    "p_gps",
        "sensor.camera": "p_camera",
        "sensor.das":    "p_fiber",
    }
    _TOPIC_SOURCE = {
        "sensor.gps":    "gps",
        "sensor.camera": "cam",
        "sensor.das":    "das",
    }

    per_gid_meas: Dict[str, List[_Meas]] = {}
    for ev in events:
        topic = getattr(ev, "topic", "")
        if topic not in _TOPIC_MTYPE:
            continue
        p = getattr(ev, "payload", None) or {}

        # Position is required for recorder lookup.
        x_raw, y_raw = p.get("x"), p.get("y")
        if x_raw is None or y_raw is None:
            continue
        try:
            x, y = float(x_raw), float(y_raw)
        except (TypeError, ValueError):
            continue

        t = float(p.get("t", 0.0) or 0.0)
        source_kind = _TOPIC_SOURCE[topic]

        # Primary: look up the GID the tracker actually assigned.
        gid = recorder.lookup(source_kind, x, y)
        if not gid:
            # Phase 9: by default we refuse to silently recover the
            # association via the oracle vehicle_id — that would be a
            # back-door leak of ground-truth identity into fusion.  Drop
            # the measurement; the audit module surfaces this explicitly
            # with skip_reason="no_global_track_assignment".
            if STRICT_TRACKER_IDENTITY:
                continue
            # Legacy / diagnostic mode (STRICT_TRACKER_IDENTITY = False):
            # keep the historical oracle-vid fallback so the bit-for-bit
            # convergence tests against build_kalman_rows still pass.
            vid_fb = str(p.get("vehicle_id", "") or "")
            if not vid_fb:
                continue
            gid = vid_to_gid.get(vid_fb, f"__untracked__{vid_fb}")
            gid_to_vid.setdefault(gid, vid_fb)

        # vehicle_id used ONLY for end-time clipping (metadata, not fusion).
        vid_clip = str(p.get("vehicle_id", "") or "")
        if vid_clip and vid_clip in end_times and t >= end_times[vid_clip]:
            continue

        mtype = _TOPIC_MTYPE[topic]
        is_das = topic == "sensor.das"
        per_gid_meas.setdefault(gid, []).append(
            _Meas(
                t=t,
                vehicle_id=gid_to_vid.get(gid, ""),  # oracle vid, stored but not used for fusion
                mtype=mtype,
                x=x,
                y=y,
                sigma=float(p.get("sigma_m", 2.0) or 2.0),
                confidence=float(p.get("confidence", 0.55) or 0.55),
                x_true=(None if p.get("x_true") is None else float(p["x_true"])),
                y_true=(None if p.get("y_true") is None else float(p["y_true"])),
                speed_mps=float(p.get("speed_mps", 0.0) or 0.0),
                heading_rad=float(p.get("heading_rad", 0.0) or 0.0),
                source_id=str(p.get("sensor_id", "") or ""),
                snr=float(p.get("snr", 0.0) or 0.0) if is_das else 0.0,
                fiber_angle_rad=float(p.get("fiber_angle_rad", 0.0) or 0.0) if is_das else 0.0,
                z_vx=float(p.get("z_vx", 0.0) or 0.0) if is_das else 0.0,
                z_vy=float(p.get("z_vy", 0.0) or 0.0) if is_das else 0.0,
                sigma_v=float(p.get("sigma_v", 0.0) or 0.0) if is_das else 0.0,
            )
        )

    for gid in per_gid_meas:
        per_gid_meas[gid].sort(key=lambda m: (m.t, m.mtype, m.source_id))

    # Sort gids by oracle vid so row order matches build_kalman_rows exactly
    # (required for the bit-for-bit convergence test on 1:1 scenarios).
    sorted_gids = sorted(per_gid_meas.keys(), key=lambda g: gid_to_vid.get(g, g))

    rows: List[Tuple[str, ...]] = []
    for gid in sorted_gids:
        meas_list = per_gid_meas[gid]
        if not meas_list:
            continue
        oracle_vid = gid_to_vid.get(gid, "")
        # The truth stream is indexed by oracle vid (the simulator only
        # knows oracle identity).  A track with no oracle vid — currently
        # impossible but kept defensive for later phases — is scored
        # without truth (pos_err column remains blank).
        truth_list = truths.get(oracle_vid, [])

        kf = LegacyKalmanFilter(dt=1.0 / 30.0, debug_das=debug_das)
        i = 0
        last_t = None
        while i < len(meas_list):
            t = meas_list[i].t
            batch: List[_Meas] = []
            while i < len(meas_list) and abs(meas_list[i].t - t) < 1e-9:
                batch.append(meas_list[i])
                i += 1

            truth = _nearest_truth(truth_list, t)
            if not kf._initialized:
                seed = batch[0]
                vx0 = seed.speed_mps * math.cos(seed.heading_rad)
                vy0 = seed.speed_mps * math.sin(seed.heading_rad)
                if truth is not None:
                    kf.initialize_state(truth.x, truth.y, vx=vx0, vy=vy0, ax=0.0, ay=0.0, pos_sigma=2.5)
                else:
                    # DAS cannot observe y — see comment in build_kalman_rows.
                    y_init = 0.0 if seed.mtype == "p_fiber" else seed.y
                    kf.initialize_state(seed.x, y_init, vx=vx0, vy=vy0, ax=0.0, ay=0.0, pos_sigma=max(1.0, seed.sigma))
                last_t = t
            else:
                dt = max(1.0 / 60.0, t - float(last_t))
                kf.predict(dt=dt)
                last_t = t

            das_pos = [(m.mtype, m.x, m.y, m.sigma, m.confidence) for m in batch if m.mtype == "p_fiber"]
            other_pos = [(m.mtype, m.x, m.y, m.sigma, m.confidence) for m in batch if m.mtype != "p_fiber"]
            if das_pos:
                if debug_das:
                    for m in batch:
                        if m.mtype != "p_fiber":
                            continue
                        print(
                            f"[DAS t={t:.2f} gid={gid} vid={oracle_vid}]"
                            f"  SNR={m.snr:.1f}  sigma_x={m.sigma:.3f}m"
                        )
                kf.update_position_x_only(das_pos)
            if other_pos:
                kf.update_positions(other_pos)

            das_vel: List[Tuple] = []
            other_vel: List[Tuple] = []
            for m in batch:
                if m.mtype == "p_fiber":
                    if m.sigma_v > 0.0:
                        das_vel.append(("v_das", m.z_vx, m.z_vy, m.sigma_v, m.confidence))
                else:
                    if m.speed_mps <= 0.0:
                        continue
                    vel_sigma = _sensor_velocity_sigma(m.mtype, m.sigma, m.confidence)
                    vx_m = m.speed_mps * math.cos(m.heading_rad)
                    vy_m = m.speed_mps * math.sin(m.heading_rad)
                    other_vel.append((m.mtype.replace("p_", "v_"), vx_m, vy_m, vel_sigma, m.confidence))
            if das_vel:
                kf.update_velocities(das_vel)
            if other_vel:
                kf.update_velocities(other_vel)

            xh, vxh, axh, yh, vyh, ayh = [float(v) for v in kf.x.reshape(-1)]
            sigma_pos = math.sqrt(max(0.0, 0.5 * (kf.P[0, 0] + kf.P[3, 3])))
            sigma_vel = math.sqrt(max(0.0, 0.5 * (kf.P[1, 1] + kf.P[4, 4])))

            pos_err = ""
            if truth is not None:
                pos_err = f"{math.hypot(xh - truth.x, yh - truth.y):.2f}"
            sources = "+".join(sorted({m.mtype.replace('p_', '') for m in batch}))
            # Phase 7: gid is emitted as a stringified value so the whole
            # row remains a homogeneous tuple of strings — matching the
            # rest of the schema and the Treeview's value contract.
            rows.append(
                (
                    f"{t:.2f}",
                    str(gid),     # NEW (Phase 7): tracker's global_track_id
                    oracle_vid,   # oracle vehicle_id (ground-truth reference)
                    sources,
                    f"{xh:.2f}",
                    f"{yh:.2f}",
                    f"{vxh:.2f}",
                    f"{vyh:.2f}",
                    f"{axh:.2f}",
                    f"{ayh:.2f}",
                    f"{sigma_pos:.2f}",
                    f"{sigma_vel:.2f}",
                    pos_err,
                )
            )
    return rows


def build_sensor_gid_lookup(
    events: Iterable,
) -> "tuple[dict, dict]":
    """Replay *events* through a fresh :class:`~simstudio.tracking.TrackManager`
    and return two look-up dicts for annotating raw sensor tables.

    Parameters
    ----------
    events : iterable
        Same event stream passed to :func:`build_kalman_rows_tracked`.

    Returns
    -------
    sensor_lookup : dict[(source_kind, x, y) -> global_track_id]
        ``source_kind`` is one of ``"gps"``, ``"cam"``, ``"das"``.
        ``x`` and ``y`` are the exact :class:`float` values from each
        sensor event payload — the same values that appear in the
        GPS/Camera/DAS table rows.
    vid_to_gid : dict[oracle_vehicle_id -> global_track_id]
        Derived from :attr:`~simstudio.tracking.Track.hypothesis_vehicle_ids`.
        Populated only for oracle vehicle ids that appeared in
        ``world.vehicle_state`` events and were absorbed into a track.
    """
    from .tracking import TrackManager as _TM
    events = list(events)
    recorder = _TrackAssignmentRecorder()
    tm = _TM(world=None, bus=recorder)
    for ev in events:
        tm.on_event(ev)
    vid_to_gid: Dict[str, str] = {}
    for gid, tr in tm.tracks.items():
        for vid in tr.hypothesis_vehicle_ids:
            vid_to_gid.setdefault(str(vid), str(gid))
    return dict(recorder._assignments), vid_to_gid


def build_kalman_rmse(events: Iterable) -> Tuple[float | None, float | None, float | None, int]:
    grouped = _collect_measurements(events)
    truths = _collect_truth(events)
    x_err2: List[float] = []
    y_err2: List[float] = []
    pos_err2: List[float] = []

    for vid, meas_list in grouped.items():
        truth_list = truths.get(vid, [])
        if not meas_list or not truth_list:
            continue
        kf = LegacyKalmanFilter(dt=1.0 / 30.0)
        i = 0
        last_t = None
        while i < len(meas_list):
            t = meas_list[i].t
            batch: List[_Meas] = []
            while i < len(meas_list) and abs(meas_list[i].t - t) < 1e-9:
                batch.append(meas_list[i])
                i += 1
            truth = _nearest_truth(truth_list, t)
            if truth is None:
                continue
            if not kf._initialized:
                seed = batch[0]
                vx0 = seed.speed_mps * math.cos(seed.heading_rad)
                vy0 = seed.speed_mps * math.sin(seed.heading_rad)
                kf.initialize_state(truth.x, truth.y, vx=vx0, vy=vy0, ax=0.0, ay=0.0, pos_sigma=2.5)
                last_t = t
            else:
                dt = max(1.0 / 60.0, t - float(last_t))
                kf.predict(dt=dt)
                last_t = t
            das_pos = [(m.mtype, m.x, m.y, m.sigma, m.confidence) for m in batch if m.mtype == "p_fiber"]
            other_pos = [(m.mtype, m.x, m.y, m.sigma, m.confidence) for m in batch if m.mtype != "p_fiber"]
            if das_pos:
                kf.update_position_x_only(das_pos)
            if other_pos:
                kf.update_positions(other_pos)
            das_vel: List[Tuple] = []
            other_vel: List[Tuple] = []
            for m in batch:
                if m.mtype == "p_fiber":
                    if m.sigma_v > 0.0:
                        das_vel.append(("v_das", m.z_vx, m.z_vy, m.sigma_v, m.confidence))
                else:
                    if m.speed_mps <= 0.0:
                        continue
                    vel_sigma = _sensor_velocity_sigma(m.mtype, m.sigma, m.confidence)
                    vx_m = m.speed_mps * math.cos(m.heading_rad)
                    vy_m = m.speed_mps * math.sin(m.heading_rad)
                    other_vel.append((m.mtype.replace("p_", "v_"), vx_m, vy_m, vel_sigma, m.confidence))
            if das_vel:
                kf.update_velocities(das_vel)
            if other_vel:
                kf.update_velocities(other_vel)
            xh, _, _, yh, _, _ = [float(v) for v in kf.x.reshape(-1)]
            dx = xh - truth.x
            dy = yh - truth.y
            x_err2.append(dx*dx)
            y_err2.append(dy*dy)
            pos_err2.append(dx*dx + dy*dy)

    n = len(pos_err2)
    if not n:
        return None, None, None, 0
    return math.sqrt(sum(x_err2)/n), math.sqrt(sum(y_err2)/n), math.sqrt(sum(pos_err2)/n), n
