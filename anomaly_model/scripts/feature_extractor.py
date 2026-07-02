"""
anomaly_model/feature_extractor.py
====================================
Phase 1 — Feature Extraction Layer
SimStudio Anomaly Detection Model

PURPOSE
-------
Read-only post-processing module.  Consumes the standard SimStudio export
files (trajectory CSVs, audit CSV, Excel workbook, scenario JSON) and
produces a flat feature DataFrame with one row per (scenario, track).

This file never imports from simstudio.* and never touches the simulator,
GUI, Kalman filter, or tracking logic.  It works entirely on the files that
already exist on disk after a simulation run.

DATA SOURCES CONSUMED (all optional — missing sources are silently skipped)
----------------------------------------------------------------------------
1.  track_trajectory_T*.csv   — Kalman states, ground truth, per-sensor meas.
2.  kalman_measurement_audit.csv — per-measurement SNR, confidence, sigma, flags
3.  coverage_summary.csv      — sensor-level accepted/skipped counts
4.  simstudio_export_*.xlsx   — Vehicles, DAS, Kalman, RMSE, Issues sheets
5.  scenario *.json / *.sim.json — vehicle weights, anomaly flags, sensor config

OUTPUT
------
pd.DataFrame with columns defined in FEATURE_CATALOG (see bottom of file).
One row per (scenario_id, global_track_id).

WEIGHT ESTIMATION NOTE
-----------------------
das_amplitude and fiber_distance_m are published on the internal event bus
but are NOT currently written to the audit CSV or Excel export.  The audit
CSV does contain `snr` for DAS rows.  Since:

    SNR = A / snr_th = W / ((r + d0)^2 * snr_th)

we can back-estimate weight only if we also know r (fiber_distance_m) and d0.
These can be read from the scenario JSON (fiber_offset_m, d0_m) but r varies
per timestep with the vehicle's lateral position.

Current approach:
  - Use mean SNR as a direct proxy for vehicle weight (monotone with W for
    fixed geometry).
  - Read declared weight_kg from scenario JSON when available.
  - Compute W_est only when scenario JSON provides fiber_offset_m and d0_m,
    using the approximation r ≈ fiber_offset_m (lane-center assumption).
  - Flag when W_est differs significantly from declared weight_kg.

This limitation is documented in anomaly_model_flow.md.

USAGE
-----
    from anomaly_model.feature_extractor import ScenarioLoader, extract_features

    # Single scenario folder
    row = ScenarioLoader("/path/to/scenario_output/").extract()

    # Batch over many scenario folders
    df = extract_features([folder1, folder2, ...], scenario_id_fn=lambda p: p.name)
    df.to_csv("anomaly_model/outputs/features.csv", index=False)
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DAS physics constants (must match sim_core.py defaults)
K_DAS = 2.0          # sigma_das = K_DAS / sqrt(SNR)
D0_DEFAULT = 0.7     # singularity floor [m]
SNR_TH_LIGHT  = 6.0
SNR_TH_MEDIUM = 8.0
SNR_TH_HEAVY  = 10.0
SNR_TH_DEFAULT = SNR_TH_MEDIUM

# Physical plausibility limits
V_MAX_NORMAL_MPS   = 40.0   # ~144 km/h  – beyond this is physically suspect
A_MAX_NORMAL_MPS2  =  8.0   # emergency braking / hard acceleration upper bound
LATERAL_DEV_NORMAL_M = 1.0  # OU lateral deviation seldom exceeds half lane width (~1.8 m)

# Anomaly score thresholds used for binary flag columns
LOW_SNR_THRESHOLD   = 5.0    # DAS SNR below this is "low quality"
LOW_CONF_THRESHOLD  = 0.30   # camera confidence below this is "unreliable"
HIGH_PRED_FRACTION  = 0.40   # >40 % prediction-only rows is concerning
OVERCONF_RATIO      = 2.0    # pos_err / sigma_pos > 2 → filter overconfident
DISAGREE_THRESHOLD_M = 5.0   # sensor disagreement > 5 m is notable
TTC_DANGER_S         = 2.0   # TTC below 2 s is universally considered dangerous

# Weight class boundaries [kg] — used to classify W_est
WEIGHT_CLASSES = {
    "pedestrian":  (0,   150),
    "motorcycle":  (150, 900),
    "car":         (900, 3000),
    "van_truck":   (3000, 8000),
    "bus_hgv":     (8000, 1e9),
}

# ---------------------------------------------------------------------------
# XLSX column-name mappings
# ---------------------------------------------------------------------------
# The SimStudio tracking_audit.xlsx uses human-readable headers.
# These dicts translate them to the internal snake_case names used
# throughout this module.  Any header not in the map is kept as-is.

_TRACK_SHEET_COL_MAP: dict = {
    "Time (s)":                                                                 "t",
    "Track ID":                                                                 "global_track_id",
    "True Vehicle ID":                                                          "vehicle_id_oracle",
    "Road Segment":                                                             "segment_id",
    "Update Type":                                                              "update_kind",
    "True X (m)":                                                               "true_x",
    "True Y (m)":                                                               "true_y",
    "True Distance Along Road (m)":                                             "true_s",
    "GPS X (m)":                                                                "gps_x",
    "GPS Y (m)":                                                                "gps_y",
    "GPS Distance Along Road (m)":                                              "gps_s",
    "Camera X (m)":                                                             "cam_x",
    "Camera Y (m)":                                                             "cam_y",
    "Camera Distance Along Road (m)":                                           "cam_s",
    "DAS X (m)":                                                                "das_x",
    "DAS Y (m)":                                                                "das_y",
    "DAS Fiber Position (m)":                                                   "das_fiber_position_m",
    "DAS Distance Along Road (m)":                                              "das_s",
    "Kalman X Estimate (m)":                                                    "x_hat",
    "Kalman Y Estimate (m)":                                                    "y_hat",
    "Kalman Velocity X (m/s)":                                                  "vx_hat",
    "Kalman Velocity Y (m/s)":                                                  "vy_hat",
    "Kalman Acceleration X (m/s²)":                                        "ax_hat",
    "Kalman Acceleration Y (m/s²)":                                        "ay_hat",
    "Kalman Distance Along Road (m)":                                           "hat_s",
    "Kalman Position Uncertainty σ (m)":                                   "sigma_pos_m",
    "Kalman Velocity Uncertainty σ (m/s)":                                 "sigma_v_mps",
    "2D Position Error |true − estimate| (m)":                             "pos_err_m",
    "Pre-update X Residual ν_x = x_meas − x_pred (m)":              "innovation_x",
    "Pre-update Y Residual ν_y = y_meas − y_pred (m)":              "innovation_y",
    "Innovation Variance S_xx (m²) — expected spread of ν_x":   "innovation_S_x",
    "Innovation Variance S_yy (m²) — expected spread of ν_y":   "innovation_S_y",
    # The NIS_x column header is long — match by prefix in the reader below
}

_AUDIT_SHEET_COL_MAP: dict = {
    "Time (s)":                                 "t",
    "Sensor Type":                              "sensor_type",
    "Sensor ID":                                "sensor_id",
    "True Vehicle ID":                          "vehicle_id_oracle",
    "Track ID":                                 "global_track_id",
    "Road Segment":                             "segment_id",
    "Measured X (m)":                           "x_meas",
    "Measured Y (m)":                           "y_meas",
    "Measured Speed (m/s)":                     "speed_meas",
    "Heading (rad)":                            "heading",
    "Position Uncertainty σ (m)":          "sigma_m",
    "Velocity Uncertainty σ (m/s)":        "sigma_v",
    "Detection Confidence":                     "confidence",
    "Signal-to-Noise Ratio":                    "snr",
    "Accepted by Kalman (1/0)":                 "accepted",
    "Skipped (1/0)":                            "skipped",
    "Skip Reason":                              "skip_reason",
    "Kalman Update Type":                       "kalman_update_kind",
    "Kalman Time (s)":                          "kalman_t",
    "Kalman Row Index":                         "kalman_row_idx",
    "DAS Amplitude A (raw units)":              "das_amplitude",
    "Fiber Distance r (m)":                     "fiber_distance_m",
    "SNR Threshold":                            "snr_th",
    "Lateral Offset from Lane Centre (m)":      "lateral_offset_m",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.mean()) if len(v) else float("nan")

def _safe_std(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.std(ddof=1)) if len(v) >= 2 else float("nan")

def _safe_max(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.max()) if len(v) else float("nan")

def _safe_min(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.min()) if len(v) else float("nan")

def _safe_frac(num: int, denom: int) -> float:
    return float(num) / denom if denom else float("nan")

def _classify_weight(w_kg: float) -> str:
    for cls, (lo, hi) in WEIGHT_CLASSES.items():
        if lo <= w_kg < hi:
            return cls
    return "unknown"

def _consecutive_runs(bools: np.ndarray) -> List[int]:
    """Return lengths of all consecutive True runs in a boolean array."""
    runs, current = [], 0
    for b in bools:
        if b:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs

def _rolling_rmse(errors: np.ndarray, window: int = 10) -> np.ndarray:
    """Compute rolling RMSE over `window` samples."""
    out = np.full(len(errors), float("nan"))
    sq = errors ** 2
    for i in range(window - 1, len(errors)):
        out[i] = math.sqrt(float(np.mean(sq[i - window + 1: i + 1])))
    return out


# ---------------------------------------------------------------------------
# Excel reader
# ---------------------------------------------------------------------------

def _read_excel_sheet(wb_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    """
    Read a styled SimStudio Excel sheet into a DataFrame.
    The styled workbook has a title row, blank row, then a header row,
    then a literal 'values' sentinel row, then data rows.
    Falls back to scanning for the first row with only string values.
    Returns None if the sheet is missing or empty.
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
        if sheet_name not in wb.sheetnames:
            wb.close()
            return None
        ws = wb[sheet_name]
        all_rows = list(ws.iter_rows(values_only=True))
        wb.close()

        if not all_rows:
            return None

        # Find header row: first row where cell[0] is a non-None string
        # and at least 2 cells are non-None strings
        header_row_idx = None
        for i, row in enumerate(all_rows):
            non_none = [v for v in row if v is not None]
            str_vals  = [v for v in non_none if isinstance(v, str)]
            if len(str_vals) >= 2 and isinstance(row[0], str) and row[0] not in (
                sheet_name, "values", "key", None
            ):
                header_row_idx = i
                break

        if header_row_idx is None:
            return None

        headers = [str(v) if v is not None else f"col_{i}"
                   for i, v in enumerate(all_rows[header_row_idx])]

        # Sentinel strings to skip (sheet titles, placeholder rows)
        _SENTINELS = {"values", "key", None}

        # Data rows: all rows after header that have at least one non-None value
        # and whose first cell is not a known sentinel or section title.
        # We keep BOTH numeric-first (Vehicles, DAS) and string-first (RMSE)
        # rows, because some sheets like RMSE use strings as row keys.
        data = []
        for row in all_rows[header_row_idx + 1:]:
            first = row[0]
            # Skip fully empty rows and known sentinels
            if first is None:
                continue
            if isinstance(first, str) and first.strip() in _SENTINELS:
                continue
            # Skip pure chart-data rows that appear after real data
            # (identified as having only 1-2 non-None values total)
            non_none = sum(1 for v in row if v is not None)
            if non_none < 1:
                continue
            data.append(list(row[: len(headers)]))

        if not data:
            return None

        df = pd.DataFrame(data, columns=headers)
        # Drop all-None columns (from merged cells / chart ranges)
        df = df.loc[:, df.columns.notna()]
        df = df.dropna(axis=1, how="all")
        return df

    except Exception as exc:
        warnings.warn(f"Could not read Excel sheet '{sheet_name}' from {wb_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Scenario JSON reader
# ---------------------------------------------------------------------------

def _load_scenario_json(path: Path) -> Dict:
    """Load a scenario JSON/sim.json file. Returns {} on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _empty_scenario_meta() -> Dict:
    """Return a scenario-meta dict with NaN defaults (no JSON available).
    Only sc_speed_limit_mps is exposed as a model feature; the geometry
    fields are kept for internal W_est computation only.
    """
    return {
        "sc_speed_limit_mps":    float("nan"),
        "sc_das_fiber_offset_m": float("nan"),
        "sc_das_d0_m":           D0_DEFAULT,
        "sc_das_snr_th":         SNR_TH_DEFAULT,
        # Internal only — not exposed as a model feature
        "sc_seg_headings":       {},
    }

def _extract_scenario_meta(sc: Dict) -> Dict:
    """
    Pull vehicle-level anomaly flags and physical parameters from a
    scenario dict.  Returns a flat dict of scenario-level metadata.

    Fields returned (all prefixed 'sc_'):
      sc_n_vehicles              total number of vehicles in scenario
      sc_n_frozen                how many vehicles are frozen (stalled/obstacle)
      sc_n_speeding              how many have ignore_speed_limit=True
      sc_n_collision             how many have allow_collision=True
      sc_n_weaving               how many have lateral_mode='weave'
      sc_n_straddling            how many have lateral_mode='straddle'
      sc_n_braking_override      how many have a_cmd_override_mps2 < -2
      sc_n_tailgating            how many have min_gap_m > 0 and < 1.5
      sc_weight_kg_list          comma-separated list of vehicle weights
      sc_speed_limit_mps         min speed limit across all segments (most restrictive)
      sc_has_das                 bool — DAS sensors present
      sc_das_fiber_offset_m      fiber_offset_m of first DAS sensor
      sc_das_d0_m                d0_m of first DAS sensor
      sc_das_noise_std           noise_std of first DAS sensor
      sc_das_snr_th              SNR threshold implied by traffic_level (default medium)
      sc_anomaly_type            heuristic label inferred from flags
    """
    vehicles = sc.get("vehicles", {})
    segments = sc.get("segments", {})
    das_sensors = sc.get("das", {})

    n_frozen   = sum(1 for v in vehicles.values() if v.get("frozen", False))
    n_speed    = sum(1 for v in vehicles.values() if v.get("ignore_speed_limit", False))
    n_coll     = sum(1 for v in vehicles.values() if v.get("allow_collision", False))
    n_weave    = sum(1 for v in vehicles.values()
                     if v.get("lateral_mode", "ou") == "weave")
    n_strad    = sum(1 for v in vehicles.values()
                     if v.get("lateral_mode", "ou") == "straddle")
    n_brake    = sum(1 for v in vehicles.values()
                     if v.get("a_cmd_override_mps2", 0.0) < -2.0)
    n_tail     = sum(1 for v in vehicles.values()
                     if 0 < v.get("min_gap_m", 0.0) < 1.5)

    weights = [v.get("weight_kg", float("nan")) for v in vehicles.values()]
    weight_str = ",".join(str(w) for w in weights if not math.isnan(w))

    speed_limits = [s.get("speed_limit_mps", float("nan"))
                    for s in segments.values()]
    speed_limit = float(min([s for s in speed_limits if not math.isnan(s)],
                            default=float("nan")))

    # DAS config
    has_das = len(das_sensors) > 0
    first_das = next(iter(das_sensors.values()), {}) if has_das else {}
    fiber_offset = first_das.get("fiber_offset_m", float("nan"))
    d0 = first_das.get("d0_m", D0_DEFAULT)
    noise_std = first_das.get("noise_std", float("nan"))

    # Infer anomaly type from flags (heuristic, for labelling purposes)
    if n_coll > 0:
        anomaly_type = "collision"
    elif n_speed > 0:
        anomaly_type = "speeding"
    elif n_brake > 0:
        anomaly_type = "sudden_braking"
    elif n_weave > 0:
        anomaly_type = "weaving"
    elif n_strad > 0:
        anomaly_type = "lane_straddling"
    elif n_tail > 0:
        anomaly_type = "tailgating"
    elif n_frozen > 0:
        # distinguish stalled vehicle from obstacle
        obj_types = {v.get("object_type", "vehicle") for v in vehicles.values()
                     if v.get("frozen", False)}
        if obj_types - {"vehicle"}:
            anomaly_type = "obstacle_in_lane"
        else:
            anomaly_type = "stalled_vehicle"
    else:
        anomaly_type = "normal"

    # Segment headings — internal only, used to compute kin_road_heading_diff_mean_rad.
    # Each segment has a `points` list of [x, y] pairs; we take the heading from
    # the first to last point.  The key is the segment ID string.
    seg_headings: Dict = {}
    for seg_id, seg in segments.items():
        pts = seg.get("points", [])
        if len(pts) >= 2:
            p0, p1 = pts[0], pts[-1]
            try:
                seg_headings[str(seg_id)] = math.atan2(
                    float(p1[1]) - float(p0[1]),
                    float(p1[0]) - float(p0[0]),
                )
            except (IndexError, TypeError, ValueError):
                pass

    return {
        # ── Only this field is exposed as a model feature ──────────────────
        "sc_speed_limit_mps":    speed_limit,
        # ── Sensor geometry kept for internal W_est computation only ───────
        "sc_das_fiber_offset_m": fiber_offset,
        "sc_das_d0_m":           d0,
        "sc_das_snr_th":         SNR_TH_DEFAULT,
        # ── Segment headings: internal, not a model feature ────────────────
        "sc_seg_headings":       seg_headings,
    }


# ---------------------------------------------------------------------------
# Per-track feature computation
# ---------------------------------------------------------------------------

def _features_from_trajectory(traj: pd.DataFrame, track_id: str,
                               seg_headings: Optional[Dict] = None) -> Dict:
    """
    Extract kinematic, Kalman quality, and sensor-disagreement features
    from a single track trajectory DataFrame.

    All feature names are prefixed by their group:
      kin_   kinematic (speed, accel, lateral)
      kf_    Kalman filter quality (RMSE, sigma, consistency)
      cov_   sensor coverage / prediction gaps
      dis_   sensor disagreement
    """
    feats: Dict = {"global_track_id": track_id}

    if traj is None or traj.empty:
        return feats

    # ── Deduplicate by timestamp ──────────────────────────────────────────────
    # The batch-path Tracks sheet emits multiple rows per timestep (one per
    # Kalman measurement update), and live-mode _all_events can accumulate
    # vehicle_state events from multiple simulation runs if Start is pressed
    # more than once in the same GUI session.  In both cases, two consecutive
    # rows at the same t (or at an artificially tiny dt) can have very different
    # Kalman-derived or oracle heading values, producing phantom heading-rate
    # spikes of hundreds to tens-of-thousands of rad/s.
    # Fix: keep only the LAST row at each unique timestamp so that every
    # Δt used in rate computation reflects a genuine simulation step.
    if "t" in traj.columns and len(traj) > 1:
        traj = (traj.sort_values("t", kind="mergesort")
                    .drop_duplicates(subset=["t"], keep="last")
                    .reset_index(drop=True))

    # ── Speed ────────────────────────────────────────────────────────────────
    speed = None   # shared below
    if "vx_hat" in traj.columns and "vy_hat" in traj.columns:
        speed = np.sqrt(traj["vx_hat"].fillna(0.0) ** 2 +
                        traj["vy_hat"].fillna(0.0) ** 2)
        s = pd.Series(speed)
        feats["kin_speed_mean_mps"]    = _safe_mean(s)
        feats["kin_speed_max_mps"]     = _safe_max(s)
        feats["kin_speed_min_mps"]     = _safe_min(s)
        feats["kin_speed_std_mps"]     = _safe_std(s)
        feats["kin_speed_p90_mps"]     = float(s.quantile(0.90)) if len(s) else float("nan")
        feats["kin_speed_over_limit_frac"]  = float("nan")  # filled later when limit known
        feats["kin_speed_excess_max_mps"]   = float("nan")  # filled later when limit known
        feats["kin_speed_excess_mean_mps"]  = float("nan")  # filled later when limit known
        # Coefficient of variation — low CV = very consistent speed (tailgating proxy)
        mean_s = feats["kin_speed_mean_mps"]
        feats["kin_speed_cv"] = (feats["kin_speed_std_mps"] / mean_s
                                 if mean_s and mean_s > 0.5 else float("nan"))

        # ── Lateral velocity component (direct vy_hat) ────────────────────────
        # kin_lateral_vel_mean_mps / kin_lateral_vel_max_mps
        #   Pure lateral speed from the Kalman filter.  Independent of y_hat
        #   coordinate frame.  High max value = sudden sideways lurch (weaving
        #   onset, lane-change, or collision side-swipe).
        #   Helps detect: Weaving (W), Lane-straddling (LS).
        lat_vel = traj["vy_hat"].abs().dropna()
        feats["kin_lateral_vel_mean_mps"] = _safe_mean(lat_vel)
        feats["kin_lateral_vel_max_mps"]  = _safe_max(lat_vel)

        # ── Speed jump between consecutive ticks ──────────────────────────────
        # kin_speed_jump_max_mps
        #   Max |Δspeed| between adjacent 50 Hz rows.  Sudden speed changes that
        #   exceed physical limits (≈ 6–8 m/s²·Δt ≈ 0.12–0.16 m/s per tick)
        #   indicate ghost tracks, sensor outliers, or emergency stops.
        #   Helps detect: Hard braking (HB), Ghost track (GEN).
        spd_series = pd.Series(speed).values
        jumps = np.abs(np.diff(spd_series))
        feats["kin_speed_jump_max_mps"] = float(np.nanmax(jumps)) if len(jumps) else float("nan")

        # ── Heading change rate ───────────────────────────────────────────────
        # kin_heading_change_rate_max_rad_per_s
        #   Maximum |Δheading / Δt| while moving (speed > 0.5 m/s).
        #   High values = tight curves, abrupt swerves, or weaving manoeuvres.
        #   Helps detect: Weaving (W), Lane-straddling (LS).
        if "t" in traj.columns:
            t_arr_h = traj["t"].values
            vx_h = traj["vx_hat"].fillna(0.0).values
            vy_h = traj["vy_hat"].fillna(0.0).values
            spd_h = speed  # already computed
            heading_vals = np.where(spd_h > 0.5, np.arctan2(vy_h, vx_h), np.nan)
            dt_h = np.diff(t_arr_h)
            dh   = np.diff(heading_vals)
            # Wrap angular difference to [-π, π]
            dh = (dh + np.pi) % (2 * np.pi) - np.pi
            # Require at least 5 ms between heading samples.  This floors out
            # any residual near-zero dt pairs that survive deduplication (e.g.
            # floating-point rounding in long simulations) and prevents them
            # from producing arbitrarily large phantom rates.
            valid_h = (dt_h >= 0.005) & np.isfinite(dh)
            if valid_h.any():
                rates_h = np.abs(dh[valid_h] / dt_h[valid_h])
                # Use P99 instead of MAX.  The Kalman velocity estimate can
                # produce direction flips (~π rad in 0.02 s = 157 rad/s) at
                # any speed when the x/y components have high noise relative
                # to the vehicle speed.  These artifact pairs can represent
                # up to ~1.5% of all valid pairs in a collision scenario.
                # Taking the 99th percentile instead of the maximum removes
                # single or small clusters of spurious spikes while still
                # faithfully detecting sustained aggressive manoeuvres.
                feats["kin_heading_change_rate_max_rad_per_s"] = float(
                    np.percentile(rates_h, 99))
            else:
                feats["kin_heading_change_rate_max_rad_per_s"] = float("nan")
        else:
            feats["kin_heading_change_rate_max_rad_per_s"] = float("nan")

    # ── Stall / stopped-vehicle detection ────────────────────────────────────
    if speed is not None:
        STOP_THRESH_MPS = 0.5   # below this = effectively stopped
        is_stopped = (pd.Series(speed) < STOP_THRESH_MPS).values
        n_total = len(is_stopped)
        feats["kin_stopped_frac"] = float(is_stopped.sum()) / max(n_total, 1)

        stop_runs = _consecutive_runs(is_stopped)
        feats["kin_stop_event_count"]        = len(stop_runs)
        feats["kin_max_stopped_steps"]       = int(max(stop_runs)) if stop_runs else 0

        # Convert longest stop to seconds if we have timestamps
        if "t" in traj.columns:
            t_arr = traj["t"].dropna().values
            dt_median = float(np.median(np.diff(t_arr))) if len(t_arr) >= 2 else 0.02
            feats["kin_max_stopped_duration_s"] = (
                feats["kin_max_stopped_steps"] * dt_median
            )
        else:
            feats["kin_max_stopped_duration_s"] = float("nan")

    # ── Acceleration & deceleration ──────────────────────────────────────────
    if "ax_hat" in traj.columns and "ay_hat" in traj.columns:
        accel_mag = np.sqrt(traj["ax_hat"].fillna(0.0) ** 2 +
                            traj["ay_hat"].fillna(0.0) ** 2)
        feats["kin_accel_mean_abs_mps2"] = _safe_mean(pd.Series(accel_mag))
        feats["kin_accel_max_abs_mps2"]  = _safe_max(pd.Series(accel_mag))
        feats["kin_accel_std_mps2"]      = _safe_std(pd.Series(accel_mag))
        feats["kin_phys_impossible_v"]   = int(
            (pd.Series(speed) > V_MAX_NORMAL_MPS).sum()) if speed is not None else 0
        feats["kin_phys_impossible_a"]   = int(
            (pd.Series(accel_mag) > A_MAX_NORMAL_MPS2).sum())

        # kin_lateral_accel_max_mps2
        #   Maximum absolute lateral (y-axis) acceleration from Kalman filter.
        #   Large lateral acceleration = aggressive steering / weaving / collision
        #   side-impact.  Independent of road curvature because it uses the raw
        #   Kalman y-axis, which is the simulator's lateral axis.
        #   Helps detect: Weaving (W), Lane-straddling (LS), Collision (CA).
        feats["kin_lateral_accel_max_mps2"] = _safe_max(
            traj["ay_hat"].abs().dropna()
        )

    # Longitudinal deceleration — project Kalman-smoothed ax_hat/ay_hat onto the
    # direction of travel.  This is cleaner than differentiating noisy speed estimates.
    # Positive result = decelerating (speed decreasing).
    if (speed is not None and "ax_hat" in traj.columns
            and "ay_hat" in traj.columns):
        spd_arr = pd.Series(speed).values
        # Unit velocity vector (guard against zero speed)
        vx = traj["vx_hat"].fillna(0.0).values
        vy = traj["vy_hat"].fillna(0.0).values
        ax = traj["ax_hat"].fillna(0.0).values
        ay = traj["ay_hat"].fillna(0.0).values
        spd_safe = np.where(spd_arr > 0.1, spd_arr, np.nan)
        v_unit_x = vx / spd_safe
        v_unit_y = vy / spd_safe
        # Longitudinal acceleration: positive = forward accel, negative = braking
        a_long   = ax * v_unit_x + ay * v_unit_y
        a_long   = a_long[~np.isnan(a_long)]

        HARD_DECEL_THRESH = 3.0   # m/s² magnitude of braking
        if len(a_long):
            decel_vals   = -a_long                         # positive = braking
            hard_braking = decel_vals > HARD_DECEL_THRESH

            feats["kin_decel_max_mps2"]    = float(decel_vals.max())
            feats["kin_high_decel_frac"]   = float(hard_braking.sum()) / len(decel_vals)
            feats["kin_decel_event_count"] = int(np.sum(np.diff(
                hard_braking.astype(int)) == 1)) if len(hard_braking) > 1 else 0
            # Jerk from smoothed longitudinal acceleration
            jerk = np.diff(a_long)
            if len(jerk):
                feats["kin_jerk_max_mps3"]      = float(np.abs(jerk).max())
                feats["kin_jerk_mean_abs_mps3"] = float(np.abs(jerk).mean())
            else:
                feats["kin_jerk_max_mps3"]      = float("nan")
                feats["kin_jerk_mean_abs_mps3"] = float("nan")
        else:
            feats["kin_decel_max_mps2"]    = float("nan")
            feats["kin_high_decel_frac"]   = float("nan")
            feats["kin_decel_event_count"] = 0
            feats["kin_jerk_max_mps3"]     = float("nan")
            feats["kin_jerk_mean_abs_mps3"]= float("nan")

    # ── Lateral deviation — enhanced for weaving & straddling ────────────────
    if "y_hat" in traj.columns:
        y = traj["y_hat"].dropna()
        if len(y) >= 2:
            y_vals = y.values
            y_mean = float(y_vals.mean())
            lat_dev = np.abs(y_vals - y_mean)

            feats["kin_lateral_dev_mean_m"] = float(lat_dev.mean())
            feats["kin_lateral_dev_max_m"]  = float(lat_dev.max())
            feats["kin_lateral_dev_std_m"]  = float(lat_dev.std())
            feats["kin_lateral_peak_to_peak_m"] = float(y_vals.max() - y_vals.min())

            # Sinusoidal score: high std relative to mean → oscillation (weaving)
            if feats["kin_lateral_dev_mean_m"] > 0:
                feats["kin_lateral_oscillation_ratio"] = (
                    feats["kin_lateral_dev_std_m"] /
                    max(feats["kin_lateral_dev_mean_m"], 0.01)
                )
            else:
                feats["kin_lateral_oscillation_ratio"] = 0.0

            # Lateral movement speed — P99 rate of y_hat change across consecutive rows.
            # Does NOT require knowing lane center; measures how fast the vehicle
            # is moving sideways regardless of where it is.
            # Uses P99 (not MAX) because the Kalman y-position estimate can produce
            # large position jumps on measurement updates; P99 removes these single
            # outlier pairs while preserving genuine sustained lateral movement.
            # Mirrors the dt >= 0.005 floor applied to heading-rate computation.
            if "t" in traj.columns and len(y) >= 2:
                t_vals = traj.loc[y.index, "t"].values if "t" in traj.columns else None
                if t_vals is not None and len(t_vals) >= 2:
                    dt = np.diff(t_vals)
                    dy = np.diff(y_vals)
                    valid = dt >= 0.005
                    if valid.any():
                        lat_speed = np.abs(dy[valid] / dt[valid])
                        feats["kin_lateral_speed_max_mps"] = float(
                            np.percentile(lat_speed, 99))
                    else:
                        feats["kin_lateral_speed_max_mps"] = float("nan")
                else:
                    feats["kin_lateral_speed_max_mps"] = float("nan")
            else:
                feats["kin_lateral_speed_max_mps"] = float("nan")

    # ── Road-heading alignment ───────────────────────────────────────────────
    # kin_road_heading_diff_mean_rad
    #   Mean absolute angular difference between the vehicle's heading (from
    #   Kalman vx_hat/vy_hat) and the road segment's design heading (from the
    #   scenario JSON `points` array), computed only when speed > 0.5 m/s.
    #   Low = vehicle is travelling along the road as intended.
    #   High = vehicle is crossing the lane or going against traffic.
    #   Requires: trajectory has segment_id column, seg_headings dict provided.
    #   Helps detect: Weaving (W), Wrong-way driving, Lane-straddling (LS).
    if (seg_headings and "segment_id" in traj.columns
            and "vx_hat" in traj.columns and "vy_hat" in traj.columns
            and speed is not None):
        seg_ids = traj["segment_id"].astype(str).values
        vx_rh = traj["vx_hat"].fillna(0.0).values
        vy_rh = traj["vy_hat"].fillna(0.0).values
        spd_rh = speed
        road_diffs: List[float] = []
        for i in range(len(traj)):
            if spd_rh[i] < 0.5:
                continue
            road_h = seg_headings.get(seg_ids[i])
            if road_h is None:
                continue
            veh_h = math.atan2(vy_rh[i], vx_rh[i])
            diff  = abs(math.atan2(
                math.sin(veh_h - road_h),
                math.cos(veh_h - road_h),
            ))
            road_diffs.append(diff)
        feats["kin_road_heading_diff_mean_rad"] = (
            float(np.mean(road_diffs)) if road_diffs else float("nan")
        )
    else:
        feats["kin_road_heading_diff_mean_rad"] = float("nan")

    # ── Kalman quality ───────────────────────────────────────────────────────
    if "pos_err_m" in traj.columns:
        err = traj["pos_err_m"].dropna()
        feats["kf_pos_err_mean_m"]   = _safe_mean(err)
        feats["kf_pos_err_max_m"]    = _safe_max(err)
        feats["kf_pos_err_std_m"]    = _safe_std(err)
        feats["kf_pos_err_p90_m"]    = float(err.quantile(0.90)) if len(err) else float("nan")

        # Rolling RMSE spike detection: count windows where rolling RMSE
        # exceeds 2× the track-global RMSE
        if len(err) >= 10:
            track_rmse = float(np.sqrt((err**2).mean()))
            feats["kf_track_rmse_m"] = track_rmse
            rolling = _rolling_rmse(err.values, window=10)
            spikes = np.sum(rolling > 2.0 * track_rmse)
            feats["kf_rolling_rmse_spike_count"] = int(spikes)
        else:
            feats["kf_track_rmse_m"] = float("nan")
            feats["kf_rolling_rmse_spike_count"] = 0

    if "sigma_pos_m" in traj.columns:
        sig = traj["sigma_pos_m"].dropna()
        feats["kf_sigma_pos_mean_m"] = _safe_mean(sig)
        feats["kf_sigma_pos_max_m"]  = _safe_max(sig)

    # Filter consistency: pos_err / sigma_pos (should be ≈ 1 if well-calibrated)
    if "pos_err_m" in traj.columns and "sigma_pos_m" in traj.columns:
        both = traj[["pos_err_m", "sigma_pos_m"]].dropna()
        if len(both):
            ratio = both["pos_err_m"] / both["sigma_pos_m"].clip(lower=0.01)
            feats["kf_consistency_ratio_mean"] = float(ratio.mean())
            feats["kf_consistency_ratio_max"]  = float(ratio.max())
            feats["kf_overconfident_frac"] = float(
                (ratio > OVERCONF_RATIO).sum()) / len(ratio)

    # ── Sigma growth rate during prediction-only gaps ─────────────────────────
    # During sensor-dropout periods, sigma_pos grows as the filter propagates
    # uncertainty forward without measurement corrections.  A healthy filter
    # grows at a rate set by process noise.  Abnormally fast growth (or growth
    # even while sensors are present) indicates filter divergence.
    #
    # Feature: kf_sigma_growth_rate_during_gaps_m_per_s
    #   — mean rate of sigma_pos increase (m/s) across all prediction-only runs.
    #   — Only increasing steps are included (plateau / reset steps ignored).
    #   — NaN if no prediction-only period is ≥ 2 rows.
    #
    # Helps detect: Sensor dropout (SD) — abnormally fast uncertainty growth.
    if ("sigma_pos_m" in traj.columns
            and "update_kind" in traj.columns
            and "t" in traj.columns):
        sigma_arr = traj["sigma_pos_m"].values
        t_arr     = traj["t"].values
        gap_mask  = (traj["update_kind"] == "prediction_only").values
        growth_rates: List[float] = []
        i = 0
        while i < len(gap_mask):
            if gap_mask[i]:
                j = i
                while j < len(gap_mask) and gap_mask[j]:
                    j += 1
                # Prediction-only run from index i to j-1 (inclusive)
                if j - i >= 2:
                    s_run = sigma_arr[i:j]
                    t_run = t_arr[i:j]
                    dt    = np.diff(t_run)
                    ds    = np.diff(s_run)
                    valid = (dt > 0) & np.isfinite(ds) & np.isfinite(dt)
                    if valid.any():
                        rates = ds[valid] / dt[valid]
                        growth_rates.extend(rates[rates > 0].tolist())  # only positive growth
                i = j
            else:
                i += 1
        feats["kf_sigma_growth_rate_during_gaps_m_per_s"] = (
            float(np.mean(growth_rates)) if growth_rates else float("nan")
        )

    # ── Post-dropout position jump ────────────────────────────────────────────
    # kf_post_dropout_jump_m
    #   At every transition from a prediction_only run back to a measurement
    #   update, compute the 2D jump in Kalman position estimate.  Large jumps
    #   mean the filter had drifted significantly during the gap, which is a
    #   signature of either very long dropouts or a ghost track that disappeared
    #   and re-appeared at a different position.
    #   Helps detect: Sensor dropout (SD), Ghost track (GEN).
    if ("update_kind" in traj.columns
            and "x_hat" in traj.columns
            and "y_hat" in traj.columns):
        uk_vals = traj["update_kind"].values
        xh_vals = traj["x_hat"].values
        yh_vals = traj["y_hat"].values
        post_jumps: List[float] = []
        for i in range(1, len(uk_vals)):
            if uk_vals[i - 1] == "prediction_only" and uk_vals[i] != "prediction_only":
                dx = xh_vals[i] - xh_vals[i - 1]
                dy = yh_vals[i] - yh_vals[i - 1]
                if np.isfinite(dx) and np.isfinite(dy):
                    post_jumps.append(math.sqrt(dx ** 2 + dy ** 2))
        feats["kf_post_dropout_jump_m"] = (
            float(np.mean(post_jumps)) if post_jumps else float("nan")
        )

    # ── Prediction-only coverage ──────────────────────────────────────────────
    if "update_kind" in traj.columns:
        is_pred = (traj["update_kind"] == "prediction_only").values
        n_total = len(is_pred)
        n_pred  = int(is_pred.sum())
        feats["cov_total_rows"]         = n_total
        feats["cov_pred_only_count"]    = n_pred
        feats["cov_pred_only_frac"]     = _safe_frac(n_pred, n_total)
        runs = _consecutive_runs(is_pred)
        feats["cov_max_consec_pred"]    = max(runs) if runs else 0
        feats["cov_n_dropout_events"]   = len(runs)

    # ── Duration ─────────────────────────────────────────────────────────────
    if "t" in traj.columns:
        t = traj["t"].dropna()
        feats["cov_duration_s"] = float(t.max() - t.min()) if len(t) >= 2 else 0.0

    # ── Sensor disagreement ──────────────────────────────────────────────────
    for src_a, src_b, label in [
        ("das_x", "cam_x",   "dis_das_cam_mean_m"),
        ("das_x", "x_hat",   "dis_das_hat_mean_m"),
        ("cam_x", "x_hat",   "dis_cam_hat_mean_m"),
        ("gps_x", "x_hat",   "dis_gps_hat_mean_m"),
    ]:
        if src_a in traj.columns and src_b in traj.columns:
            both = traj[[src_a, src_b]].dropna()
            if len(both):
                diff = (both[src_a] - both[src_b]).abs()
                feats[label] = float(diff.mean())
                feats[label.replace("_mean_", "_max_")] = float(diff.max())
            else:
                feats[label] = float("nan")
                feats[label.replace("_mean_", "_max_")] = float("nan")

    # DAS-makes-fusion-worse indicator:
    # compare mean pos_err on DAS-only rows vs camera-only rows
    if all(c in traj.columns for c in ["das_x", "cam_x", "pos_err_m"]):
        das_only = traj[
            traj["das_x"].notna() & traj["cam_x"].isna()
        ]["pos_err_m"].dropna()
        cam_only = traj[
            traj["cam_x"].notna() & traj["das_x"].isna()
        ]["pos_err_m"].dropna()
        feats["dis_das_only_err_mean_m"] = _safe_mean(das_only)
        feats["dis_cam_only_err_mean_m"] = _safe_mean(cam_only)
        if len(das_only) >= 3 and len(cam_only) >= 3:
            feats["dis_das_worse_than_cam"] = int(
                _safe_mean(das_only) > _safe_mean(cam_only))
        else:
            feats["dis_das_worse_than_cam"] = float("nan")

    return feats


def _features_from_audit(audit: pd.DataFrame, track_id: str,
                          fiber_offset_m: float = float("nan"),
                          d0_m: float = D0_DEFAULT,
                          snr_th: float = SNR_TH_DEFAULT) -> Dict:
    """
    Extract sensor quality features from the audit CSV,
    filtered to a specific track.

    Feature prefix:
      das_   DAS-specific quality
      cam_   Camera-specific quality
      gps_   GPS-specific quality
      aud_   Cross-sensor audit stats
    """
    feats: Dict = {}

    if audit is None or audit.empty:
        return feats

    trk = audit[audit["global_track_id"] == track_id]
    if trk.empty:
        return feats

    # ── DAS quality ──────────────────────────────────────────────────────────
    das = trk[trk["sensor_type"] == "DAS"]
    feats["das_n_measurements"]    = len(das)
    if len(das):
        snr = das["snr"].dropna()
        feats["das_snr_mean"]          = _safe_mean(snr)
        feats["das_snr_min"]           = _safe_min(snr)
        feats["das_snr_max"]           = _safe_max(snr)
        feats["das_snr_std"]           = _safe_std(snr)
        feats["das_snr_low_frac"]      = _safe_frac(
            int((snr < LOW_SNR_THRESHOLD).sum()), len(snr))
        sig = das["sigma_m"].dropna()
        feats["das_sigma_mean_m"]      = _safe_mean(sig)
        feats["das_sigma_max_m"]       = _safe_max(sig)
        conf = das["confidence"].dropna()
        feats["das_confidence_mean"]   = _safe_mean(conf)
        sig_v = das["sigma_v"].dropna()
        feats["das_sigma_v_mean_mps"]  = _safe_mean(sig_v)
        n_acc = int(das["accepted"].sum()) if "accepted" in das.columns else 0
        feats["das_accept_frac"]       = _safe_frac(n_acc, len(das))
        feats["das_skip_frac"]         = 1.0 - feats["das_accept_frac"]

        # ── Per-measurement W_est variability ─────────────────────────────
        # Compute W_est for every DAS row individually (not just from mean SNR).
        # A consistent vehicle produces a stable W_est series.
        # Lateral movement or sensor issues cause W_est to fluctuate.
        if (not math.isnan(fiber_offset_m) and "snr" in das.columns):
            r = abs(fiber_offset_m)
            per_row_snr = das["snr"].dropna()
            if len(per_row_snr) >= 2:
                w_series = per_row_snr * snr_th * (r + d0_m) ** 2
                w_mean = float(w_series.mean())
                w_std  = float(w_series.std(ddof=1))
                feats["das_W_est_std_kg"] = w_std
                feats["das_W_est_cv"]     = (w_std / w_mean
                                              if w_mean > 1.0 else float("nan"))
            else:
                feats["das_W_est_std_kg"] = float("nan")
                feats["das_W_est_cv"]     = float("nan")
        else:
            feats["das_W_est_std_kg"] = float("nan")
            feats["das_W_est_cv"]     = float("nan")
    else:
        for k in ["das_snr_mean", "das_snr_min", "das_snr_max", "das_snr_std",
                  "das_snr_low_frac", "das_sigma_mean_m", "das_sigma_max_m",
                  "das_confidence_mean", "das_sigma_v_mean_mps",
                  "das_accept_frac", "das_skip_frac",
                  "das_W_est_std_kg", "das_W_est_cv"]:
            feats[k] = float("nan")

    # ── Camera quality ───────────────────────────────────────────────────────
    cam = trk[trk["sensor_type"] == "Camera"]
    feats["cam_n_measurements"] = len(cam)
    if len(cam):
        conf = cam["confidence"].dropna()
        feats["cam_confidence_mean"] = _safe_mean(conf)
        feats["cam_confidence_min"]  = _safe_min(conf)
        feats["cam_confidence_std"]  = _safe_std(conf)
        feats["cam_low_conf_frac"]   = _safe_frac(
            int((conf < LOW_CONF_THRESHOLD).sum()), len(conf))
        sig = cam["sigma_m"].dropna()
        feats["cam_sigma_mean_m"]    = _safe_mean(sig)
        n_acc = int(cam["accepted"].sum()) if "accepted" in cam.columns else 0
        feats["cam_accept_frac"]     = _safe_frac(n_acc, len(cam))
    else:
        for k in ["cam_confidence_mean", "cam_confidence_min",
                  "cam_confidence_std", "cam_low_conf_frac",
                  "cam_sigma_mean_m", "cam_accept_frac"]:
            feats[k] = float("nan")

    # ── GPS quality ──────────────────────────────────────────────────────────
    gps = trk[trk["sensor_type"] == "GPS"]
    feats["gps_n_measurements"] = len(gps)
    if len(gps):
        sig = gps["sigma_m"].dropna()
        feats["gps_sigma_mean_m"] = _safe_mean(sig)

    # ── Skip analysis ────────────────────────────────────────────────────────
    if "skipped" in trk.columns:
        feats["aud_skip_total"]     = int(trk["skipped"].sum())
        feats["aud_skip_frac"]      = _safe_frac(
            feats["aud_skip_total"], len(trk))
    if "skip_reason" in trk.columns:
        skip_reasons = trk[trk["skipped"] == 1]["skip_reason"].dropna()
        feats["aud_skip_reason_top"] = (
            skip_reasons.value_counts().index[0]
            if len(skip_reasons) else ""
        )

        # aud_skip_chi2_frac
        #   Fraction of skipped measurements whose skip_reason contains "chi"
        #   (chi-squared gate rejection).  The Kalman filter rejects measurements
        #   that are statistically inconsistent with the filter's predicted state.
        #   High fraction = filter consistently disagrees with the sensors, which
        #   can indicate: wrong vehicle dynamics model, sensor misidentification,
        #   or a ghost track whose position drifts away from real measurements.
        #   Helps detect: Ghost track (GEN), Wrong weight (WW).
        if len(skip_reasons):
            chi_mask = skip_reasons.str.contains("chi", case=False, na=False)
            feats["aud_skip_chi2_frac"] = float(chi_mask.sum()) / len(skip_reasons)
        else:
            feats["aud_skip_chi2_frac"] = float("nan")

    return feats


def _features_from_coverage(traj: pd.DataFrame, audit: Optional[pd.DataFrame],
                             track_id: str,
                             d0_m: float = D0_DEFAULT) -> Dict:
    """
    Expanded coverage analysis — answering four research questions:

    A. What fraction of tracking time does each sensor contribute?
    B. Does sensor redundancy (multi-sensor overlap) actually reduce error?
    C. Where in time and space do coverage gaps hurt most?
    D. What is the effective update rate of each sensor?

    These features go well beyond simple counting and enable research
    conclusions about sensor layout, density, and anomaly-detection value.

    Feature prefix: cov_  (replaces and extends the basic cov_ group
    already computed in _features_from_trajectory)
    """
    feats: Dict = {}
    if traj is None or traj.empty:
        return feats

    n = len(traj)

    # ── A. Per-sensor active fractions ──────────────────────────────────────
    has_das = traj["das_x"].notna() if "das_x" in traj.columns else pd.Series([False] * n)
    has_cam = traj["cam_x"].notna() if "cam_x" in traj.columns else pd.Series([False] * n)
    has_gps = traj["gps_x"].notna() if "gps_x" in traj.columns else pd.Series([False] * n)

    feats["cov_das_active_frac"]  = _safe_frac(int(has_das.sum()), n)
    feats["cov_cam_active_frac"]  = _safe_frac(int(has_cam.sum()), n)
    feats["cov_gps_active_frac"]  = _safe_frac(int(has_gps.sum()), n)

    # Sensor combination fractions
    sensor_count = has_das.astype(int) + has_cam.astype(int) + has_gps.astype(int)
    feats["cov_multi_sensor_frac"]   = _safe_frac(int((sensor_count >= 2).sum()), n)
    feats["cov_single_sensor_frac"]  = _safe_frac(int((sensor_count == 1).sum()), n)
    feats["cov_no_sensor_frac"]      = _safe_frac(int((sensor_count == 0).sum()), n)

    # Pairwise overlap fractions
    feats["cov_das_cam_overlap_frac"] = _safe_frac(int((has_das & has_cam).sum()), n)
    feats["cov_das_gps_overlap_frac"] = _safe_frac(int((has_das & has_gps).sum()), n)
    feats["cov_cam_gps_overlap_frac"] = _safe_frac(int((has_cam & has_gps).sum()), n)
    feats["cov_all_sensors_frac"]     = _safe_frac(int((sensor_count == 3).sum()), n)

    # Dominant sensor: which sensor is active for the most rows
    fracs = {
        "DAS":    feats["cov_das_active_frac"],
        "Camera": feats["cov_cam_active_frac"],
        "GPS":    feats["cov_gps_active_frac"],
    }
    non_nan = {k: v for k, v in fracs.items() if not math.isnan(v)}
    if non_nan:
        feats["cov_dominant_sensor"] = max(non_nan, key=non_nan.get)
    else:
        feats["cov_dominant_sensor"] = "none"

    # ── B. Fusion benefit: does redundancy reduce error? ─────────────────────
    if "pos_err_m" in traj.columns:
        err = traj["pos_err_m"]

        # Mean error in windows where each sensor is the ONLY active one
        das_only_mask = has_das & ~has_cam & ~has_gps
        cam_only_mask = has_cam & ~has_das & ~has_gps
        gps_only_mask = has_gps & ~has_das & ~has_cam
        multi_mask    = sensor_count >= 2
        no_mask       = sensor_count == 0

        feats["cov_err_das_only_m"]    = _safe_mean(err[das_only_mask])
        feats["cov_err_cam_only_m"]    = _safe_mean(err[cam_only_mask])
        feats["cov_err_gps_only_m"]    = _safe_mean(err[gps_only_mask])
        feats["cov_err_multi_sensor_m"] = _safe_mean(err[multi_mask])
        feats["cov_err_no_sensor_m"]   = _safe_mean(err[no_mask])

        # Fusion benefit index: how much does multi-sensor reduce error vs best single sensor?
        single_errs = [
            v for v in [feats["cov_err_das_only_m"],
                        feats["cov_err_cam_only_m"],
                        feats["cov_err_gps_only_m"]]
            if not math.isnan(v)
        ]
        multi_err = feats["cov_err_multi_sensor_m"]
        if single_errs and not math.isnan(multi_err):
            best_single = min(single_errs)
            # Positive = fusion is better; negative = fusion is worse (surprising!)
            feats["cov_fusion_benefit_vs_best_single"] = best_single - multi_err
        else:
            feats["cov_fusion_benefit_vs_best_single"] = float("nan")

        # DAS contribution: does adding DAS on top of camera help?
        cam_no_das = has_cam & ~has_das
        cam_with_das = has_cam & has_das
        err_cam_no_das   = _safe_mean(err[cam_no_das])
        err_cam_with_das = _safe_mean(err[cam_with_das])
        feats["cov_err_cam_no_das_m"]    = err_cam_no_das
        feats["cov_err_cam_with_das_m"]  = err_cam_with_das
        if not math.isnan(err_cam_no_das) and not math.isnan(err_cam_with_das):
            # Positive = DAS helps; negative = DAS hurts
            feats["cov_das_fusion_benefit_m"] = err_cam_no_das - err_cam_with_das
        else:
            feats["cov_das_fusion_benefit_m"] = float("nan")

    # ── C. Temporal and spatial gap structure ────────────────────────────────
    if "update_kind" in traj.columns:
        is_pred = (traj["update_kind"] == "prediction_only").values
        n_pred  = int(is_pred.sum())

        # Temporal position of gaps: early / middle / late thirds of track
        thirds = n // 3
        if thirds > 0:
            early  = is_pred[:thirds]
            middle = is_pred[thirds: 2 * thirds]
            late   = is_pred[2 * thirds:]
            feats["cov_gap_frac_early"]  = _safe_frac(int(early.sum()),  len(early))
            feats["cov_gap_frac_middle"] = _safe_frac(int(middle.sum()), len(middle))
            feats["cov_gap_frac_late"]   = _safe_frac(int(late.sum()),   len(late))
        else:
            feats["cov_gap_frac_early"] = feats["cov_gap_frac_middle"] = feats["cov_gap_frac_late"] = float("nan")

        # Gap clustering: std of gap row indices relative to track length
        gap_idx = np.where(is_pred)[0]
        if len(gap_idx) >= 2:
            feats["cov_gap_clustering_cv"] = float(
                np.std(gap_idx) / (np.mean(gap_idx) + 1e-6))  # coefficient of variation
        else:
            feats["cov_gap_clustering_cv"] = float("nan")

    # Spatial position of gaps: road arc-length where gaps occur
    if "das_fiber_position_m" in traj.columns and "update_kind" in traj.columns:
        gap_rows = traj[traj["update_kind"] == "prediction_only"]
        non_gap  = traj[traj["update_kind"] != "prediction_only"]
        # Use das_fiber_position_m as spatial proxy when available
        gap_pos = gap_rows["das_fiber_position_m"].dropna()
        all_pos = traj["das_fiber_position_m"].dropna()
        if len(gap_pos) >= 2 and len(all_pos) >= 2:
            track_len = float(all_pos.max() - all_pos.min())
            if track_len > 0:
                feats["cov_gap_spatial_pos_mean_norm"] = float(
                    (gap_pos.mean() - all_pos.min()) / track_len)  # 0=start, 1=end
                feats["cov_gap_spatial_spread_norm"]   = float(
                    gap_pos.std() / track_len)  # spread of gap positions
            else:
                feats["cov_gap_spatial_pos_mean_norm"] = float("nan")
                feats["cov_gap_spatial_spread_norm"]   = float("nan")

    # High-speed gaps: prediction-only rows where estimated speed was high
    # (being blind at high speed is more dangerous than being blind at low speed)
    if "update_kind" in traj.columns and "vx_hat" in traj.columns and "vy_hat" in traj.columns:
        speed = np.sqrt(traj["vx_hat"].fillna(0.0)**2 + traj["vy_hat"].fillna(0.0)**2)
        gap_mask = traj["update_kind"] == "prediction_only"
        if gap_mask.sum() > 0:
            speed_at_gaps = speed[gap_mask.values]
            feats["cov_high_speed_gap_frac"] = _safe_frac(
                int((speed_at_gaps > V_MAX_NORMAL_MPS * 0.5).sum()),  # >50% of v_max
                int(gap_mask.sum()))
            feats["cov_mean_speed_at_gaps_mps"] = float(speed_at_gaps.mean())
        else:
            feats["cov_high_speed_gap_frac"]    = 0.0
            feats["cov_mean_speed_at_gaps_mps"] = float("nan")

    # ── D. Effective update rates per sensor ─────────────────────────────────
    if "t" in traj.columns:
        duration = feats.get("cov_duration_s", float("nan"))  # set by _features_from_trajectory
        if math.isnan(duration):
            t_col = traj["t"].dropna()
            duration = float(t_col.max() - t_col.min()) if len(t_col) >= 2 else float("nan")

        if not math.isnan(duration) and duration > 0:
            feats["cov_das_hz_est"] = feats["cov_das_active_frac"] * n / duration
            feats["cov_cam_hz_est"] = feats["cov_cam_active_frac"] * n / duration
            feats["cov_gps_hz_est"] = feats["cov_gps_active_frac"] * n / duration
        else:
            feats["cov_das_hz_est"] = feats["cov_cam_hz_est"] = feats["cov_gps_hz_est"] = float("nan")

    # ── Per-sensor Kalman update share (from audit CSV) ───────────────────────
    if audit is not None and not audit.empty:
        trk_audit = audit[audit["global_track_id"] == track_id]
        accepted  = trk_audit[trk_audit.get("accepted", pd.Series(dtype=bool)) == 1] \
            if "accepted" in trk_audit.columns else trk_audit
        total_accepted = max(len(accepted), 1)
        for sensor, col_name in [("DAS", "cov_das_update_share"),
                                  ("Camera", "cov_cam_update_share"),
                                  ("GPS", "cov_gps_update_share")]:
            n_sensor = int((accepted["sensor_type"] == sensor).sum()) \
                if "sensor_type" in accepted.columns else 0
            feats[col_name] = _safe_frac(n_sensor, total_accepted)

        # Which sensor type contributed the most accepted updates?
        shares = {
            "DAS":    feats.get("cov_das_update_share", 0.0),
            "Camera": feats.get("cov_cam_update_share", 0.0),
            "GPS":    feats.get("cov_gps_update_share", 0.0),
        }
        valid_shares = {k: v for k, v in shares.items() if not math.isnan(v)}
        feats["cov_dominant_kalman_updater"] = (
            max(valid_shares, key=valid_shares.get) if valid_shares else "none"
        )

        # New DAS physics fields now available in audit CSV
        das_rows = trk_audit[trk_audit["sensor_type"] == "DAS"] \
            if "sensor_type" in trk_audit.columns else pd.DataFrame()
        if len(das_rows) and "das_amplitude" in das_rows.columns:
            amp = das_rows["das_amplitude"].dropna()
            feats["cov_das_amplitude_mean"] = _safe_mean(amp)
            feats["cov_das_amplitude_std"]  = _safe_std(amp)
            # High amplitude variance → vehicle weaving (changing fiber distance)
            if not math.isnan(feats["cov_das_amplitude_mean"]) and feats["cov_das_amplitude_mean"] > 0:
                feats["cov_das_amplitude_cv"] = (
                    feats["cov_das_amplitude_std"] / feats["cov_das_amplitude_mean"])
            else:
                feats["cov_das_amplitude_cv"] = float("nan")

            # ── DAS physics consistency: Amplitude ∝ 1/(r + d0)² ────────────
            # Feature: das_amplitude_vs_dist_pearson_r
            #   Pearson r between A and 1/(fiber_distance_m + d0)².
            #   Under the DAS physics model, amplitude is proportional to
            #   vehicle weight / (fiber_distance + d0)².  Holding weight ≈
            #   constant, amplitude should closely track the inverse-square
            #   law with distance.  Low |r| = signal is NOT following the
            #   expected physics, which can indicate: vehicle physically
            #   unusual (wrong weight), sensor malfunction, or shadowing.
            #   We use d0_m from the scenario JSON (defaults to D0_DEFAULT).
            #
            # Helps detect: Wrong weight (WW), Sensor dropout (SD).
            if "fiber_distance_m" in das_rows.columns:
                phys = das_rows[["das_amplitude", "fiber_distance_m"]].dropna()
                if len(phys) >= 5:
                    A_vals  = phys["das_amplitude"].values.astype(float)
                    r_vals  = phys["fiber_distance_m"].values.astype(float)
                    inv_r2  = 1.0 / np.maximum((r_vals + d0_m) ** 2, 1e-9)
                    x_c     = inv_r2 - inv_r2.mean()
                    y_c     = A_vals - A_vals.mean()
                    denom   = math.sqrt(float((x_c ** 2).sum()) *
                                        float((y_c ** 2).sum()))
                    feats["das_amplitude_vs_dist_pearson_r"] = (
                        float(np.dot(x_c, y_c) / denom) if denom > 0 else float("nan")
                    )
                else:
                    feats["das_amplitude_vs_dist_pearson_r"] = float("nan")
            else:
                feats["das_amplitude_vs_dist_pearson_r"] = float("nan")
        else:
            feats["das_amplitude_vs_dist_pearson_r"] = float("nan")

        if len(das_rows) and "fiber_distance_m" in das_rows.columns:
            r = das_rows["fiber_distance_m"].dropna()
            feats["cov_das_fiber_dist_mean_m"] = _safe_mean(r)
            feats["cov_das_fiber_dist_std_m"]  = _safe_std(r)
            feats["cov_das_fiber_dist_max_m"]  = _safe_max(r)
            # High std in fiber distance → lateral oscillation
            feats["cov_das_fiber_dist_cv"] = (
                feats["cov_das_fiber_dist_std_m"] / max(feats["cov_das_fiber_dist_mean_m"], 0.01)
                if not math.isnan(feats.get("cov_das_fiber_dist_mean_m", float("nan"))) else float("nan")
            )

        if len(das_rows) and "lateral_offset_m" in das_rows.columns:
            lat = das_rows["lateral_offset_m"].dropna()
            feats["cov_das_lateral_offset_mean_m"]     = _safe_mean(lat)
            feats["cov_das_lateral_offset_std_m"]      = _safe_std(lat)
            feats["cov_das_lateral_offset_max_m"]      = _safe_max(lat.abs())
            # Lane-relative features for weaving / straddling
            feats["cov_das_lateral_offset_abs_mean_m"] = float(lat.abs().mean()) if len(lat) else float("nan")
            feats["cov_das_lateral_offset_range_m"]    = float(lat.max() - lat.min()) if len(lat) >= 2 else float("nan")
            feats["cov_das_lateral_frac_outside_half_m"] = float((lat.abs() > 0.5).mean()) if len(lat) else float("nan")
            feats["cov_das_lateral_frac_outside_1m"]     = float((lat.abs() > 1.0).mean()) if len(lat) else float("nan")
            # Noise-robust sign changes on a 5-sample smoothed series (weaving detection)
            if len(lat) >= 10:
                smoothed = lat.rolling(5, center=True, min_periods=3).mean().dropna().values
                if len(smoothed) >= 2:
                    signs = np.sign(smoothed)
                    feats["cov_das_lateral_sign_changes"] = int(np.sum(np.diff(signs) != 0))
                else:
                    feats["cov_das_lateral_sign_changes"] = 0
            else:
                feats["cov_das_lateral_sign_changes"] = 0

            # Lane-relative consistent-side fraction: fraction of DAS rows where the
            # vehicle is consistently on one side of the lane centre (|offset| > 0.3 m
            # and same sign as the mean offset).  High = straddling one lane edge.
            # Uses lateral_offset_m (lane-relative), so no lane-centre estimation needed.
            lat_mean = float(lat.mean()) if len(lat) else float("nan")
            if not math.isnan(lat_mean) and abs(lat_mean) > 0.3 and len(lat) >= 5:
                same_side = int((np.sign(lat.values) == np.sign(lat_mean)).sum())
                feats["cov_das_lateral_consistent_side_frac"] = same_side / len(lat)
            else:
                feats["cov_das_lateral_consistent_side_frac"] = float("nan")

            # Lateral change rate: max |Δlateral_offset_m / Δt| between consecutive
            # DAS measurements.  High = sharp lateral lurch; helps catch weaving onset.
            if "t" in das_rows.columns and len(das_rows) >= 2:
                das_sorted = das_rows.sort_values("t")
                lat_vals = das_sorted["lateral_offset_m"].dropna().values
                t_vals   = das_sorted.loc[das_sorted["lateral_offset_m"].notna(), "t"].values
                if len(lat_vals) >= 2:
                    dt = np.diff(t_vals)
                    dl = np.abs(np.diff(lat_vals))
                    valid = dt > 0
                    if valid.any():
                        feats["cov_das_lateral_change_rate_max"] = float(
                            (dl[valid] / dt[valid]).max())
                    else:
                        feats["cov_das_lateral_change_rate_max"] = float("nan")
                else:
                    feats["cov_das_lateral_change_rate_max"] = float("nan")
            else:
                feats["cov_das_lateral_change_rate_max"] = float("nan")

    return feats


def _features_from_excel(xl_path: Path, track_id: str,
                          vehicle_id: Optional[str] = None) -> Dict:
    """
    Extract features from the SimStudio Excel export that are NOT available
    in the CSVs.

    Currently reads:
      - Issues sheet: vehicle_stuck events (count, timestamps)
      - RMSE sheet: per-source RMSE breakdown
      - Vehicles sheet: ground-truth speed/acceleration time-series
        (higher resolution than Kalman estimates, for physical plausibility)
    """
    feats: Dict = {}

    if xl_path is None or not xl_path.exists():
        return feats

    # ── Issues / events ──────────────────────────────────────────────────────
    issues_df = _read_excel_sheet(xl_path, "Issues")
    if issues_df is not None and not issues_df.empty:
        # Columns: t, type, details
        if "type" in issues_df.columns:
            stuck = issues_df[issues_df["type"] == "vehicle_stuck"]
            feats["xl_n_vehicle_stuck_events"] = len(stuck)
            if len(stuck) and "t" in stuck.columns:
                feats["xl_first_stuck_t"] = float(stuck["t"].min())
        else:
            feats["xl_n_vehicle_stuck_events"] = 0

    return feats


def _estimate_weight(das_snr_mean: float, snr_th: float,
                     fiber_offset_m: float, d0_m: float) -> float:
    """
    Back-estimate vehicle weight from mean DAS SNR using the lane-center
    approximation: r ≈ fiber_offset_m.

    Formula: SNR = W / ((r + d0)^2 * snr_th)
    → W_est = SNR * snr_th * (r + d0)^2

    This is an approximation — true r varies with lateral position.
    Confidence increases when the vehicle drives close to lane center
    (small lateral deviation).
    """
    if any(math.isnan(v) for v in [das_snr_mean, snr_th, fiber_offset_m, d0_m]):
        return float("nan")
    r_approx = abs(fiber_offset_m)
    return das_snr_mean * snr_th * (r_approx + d0_m) ** 2


# ---------------------------------------------------------------------------
# XLSX trajectory / audit readers
# ---------------------------------------------------------------------------

def _read_track_sheet_from_xlsx(wb_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    """
    Read one 'Track T000001'-style sheet from tracking_audit.xlsx and
    return a DataFrame with internal snake_case column names.

    The NIS_x column header is very long (contains the full formula);
    we detect it by the 'NIS_x' prefix rather than an exact string match.

    Returns None if the sheet doesn't exist or cannot be parsed.
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
        if sheet_name not in wb.sheetnames:
            wb.close()
            return None
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
    except Exception as exc:
        warnings.warn(f"Cannot open {wb_path} sheet '{sheet_name}': {exc}")
        return None

    if len(rows) < 2:
        return None

    # Row 0 is the header
    raw_headers = [str(h) if h is not None else f"col_{i}"
                   for i, h in enumerate(rows[0])]

    # Map headers: exact match first, then NIS_x prefix catch-all
    mapped = []
    for h in raw_headers:
        if h in _TRACK_SHEET_COL_MAP:
            mapped.append(_TRACK_SHEET_COL_MAP[h])
        elif h.startswith("NIS_x"):
            mapped.append("NIS_x")
        else:
            mapped.append(h)

    data = [list(r[:len(mapped)]) for r in rows[1:] if any(v is not None for v in r)]
    if not data:
        return None

    df = pd.DataFrame(data, columns=mapped)

    # Cast numeric columns
    for col in df.columns:
        if col not in ("global_track_id", "vehicle_id_oracle", "segment_id",
                       "update_kind"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _read_audit_sheet_from_xlsx(wb_path: Path) -> Optional[pd.DataFrame]:
    """
    Read the 'Kalman Audit' sheet from tracking_audit.xlsx and return a
    DataFrame with internal snake_case column names.

    Falls back to reading kalman_measurement_audit.xlsx in the same folder
    if the consolidated workbook doesn't have a Kalman Audit sheet.
    """
    def _load_sheet(path: Path, sheet: str) -> Optional[pd.DataFrame]:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            if sheet not in wb.sheetnames:
                wb.close()
                return None
            ws = wb[sheet]
            rows = list(ws.iter_rows(values_only=True))
            wb.close()
        except Exception:
            return None
        if len(rows) < 2:
            return None

        raw_headers = [str(h) if h is not None else f"col_{i}"
                       for i, h in enumerate(rows[0])]
        mapped = [_AUDIT_SHEET_COL_MAP.get(h, h) for h in raw_headers]
        data = [list(r[:len(mapped)]) for r in rows[1:] if any(v is not None for v in r)]
        if not data:
            return None
        df = pd.DataFrame(data, columns=mapped)
        # Cast numeric columns; keep string columns intact
        str_cols = {"sensor_type", "sensor_id", "vehicle_id_oracle",
                    "global_track_id", "segment_id", "skip_reason",
                    "kalman_update_kind"}
        for col in df.columns:
            if col not in str_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # Try consolidated workbook first
    df = _load_sheet(wb_path, "Kalman Audit")
    if df is not None:
        return df

    # Fall back to separate kalman_measurement_audit.xlsx
    fallback = wb_path.parent / "kalman_measurement_audit.xlsx"
    if fallback.exists():
        import openpyxl
        try:
            wb2 = openpyxl.load_workbook(fallback, read_only=True, data_only=True)
            sheet2 = wb2.sheetnames[0] if wb2.sheetnames else None
            wb2.close()
            if sheet2:
                return _load_sheet(fallback, sheet2)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Inter-vehicle proximity features
# ---------------------------------------------------------------------------

#: Proximity thresholds for flagging dangerous following distances
_TAILGATE_DIST_M  = 5.0   # < 5 m = tailgating
_CLOSE_DIST_M     = 10.0  # < 10 m = close following
_OVERLAP_DIST_M   = 3.0   # < 3 m ≈ physical vehicle-width overlap (collision threshold)

def _features_inter_vehicle(
    traj: pd.DataFrame,
    all_trajs: pd.DataFrame,
    track_id: str,
) -> Dict:
    """
    Compute inter-vehicle proximity features for one track given the
    combined trajectory of all tracks in the scenario.

    Uses 2-D Euclidean position (x_hat, y_hat) and Kalman velocity
    (vx_hat, vy_hat) from the 50 Hz trajectory CSVs.  All timestamps are
    snapped to integer 50 Hz tick indices to avoid float-join edge cases.

    Distance-only features  (always computed when positions available)
    ────────────────────────────────────────────────────────────────────
    iv_n_other_vehicles         Number of other concurrent tracks
    iv_min_dist_m               Minimum 2D distance to any other vehicle
    iv_mean_min_dist_m          Mean of per-tick minimum distances
    iv_p10_min_dist_m           10th pct of per-tick min dist (sustained close)
    iv_close_proximity_frac     Frac of ticks with nearest < 10 m
    iv_tailgate_proximity_frac  Frac of ticks with nearest < 5 m
    iv_overlap_frac             Frac of ticks with nearest < 3 m (collision)
    iv_overlap_duration_s       Total seconds nearest < 3 m

    Velocity-dependent features  (require vx_hat / vy_hat in both dataframes)
    ────────────────────────────────────────────────────────────────────────────
    iv_ttc_min_s
        Minimum Time-to-Collision across the track [s].
        TTC = distance / closing_speed, where
        closing_speed = −(Δpos · Δvel) / dist  (positive when vehicles approach).
        Only computed when closing (closing_speed > 0.1 m/s); infinite otherwise.
        TTC < 2 s is the universally accepted danger threshold.
        Helps detect: Tailgating (TG), Collision / accident (CA).

    iv_ttc_below_2s_frac
        Fraction of shared ticks where TTC < 2 s (chronic tailgating).
        Helps detect: Tailgating (TG).

    iv_closing_speed_max_mps
        Maximum closing speed (m/s) observed at any tick.
        High value while close = kinetic energy of an imminent impact.
        Helps detect: Tailgating (TG), Collision / accident (CA).

    iv_time_headway_mean_s
        Mean time headway = distance / own_speed at each tick [s].
        < 1.5 s is considered dangerously close regardless of absolute distance.
        Helps detect: Tailgating (TG).

    iv_speed_proximity_ratio_max
        max(own_speed / max(min_dist, 0.5)) — crude inverse-TTC proxy.
        Kept for backward compatibility alongside the true TTC metrics.

    iv_speed_excess_over_others_mps
        Own mean speed − mean speed of all concurrent vehicles [m/s].
        Helps detect: Speeding (S), Stalled vehicle (SV).

    iv_speed_ratio_to_others
        Own mean speed / mean speed of concurrent vehicles (ratio).
        Helps detect: Speeding (S), Stalled vehicle (SV).

    iv_speed_pearson_r_traffic
        Pearson correlation between own per-tick speed and per-tick mean
        speed of all other vehicles.  Normal drivers correlated with traffic
        (0.6–0.9); anomalous vehicles decouple (near 0 or negative).
        Helps detect: Stalled vehicle (SV), Speeding (S), Ghost track (GEN).

    iv_decel_at_min_dist_mps2
        Longitudinal deceleration of this vehicle (m/s²) at the timestep of
        minimum inter-vehicle distance.  Positive = braking.
        Collision signature: large deceleration coincides with or immediately
        follows the closest approach.
        Near-miss signature: deceleration precedes the closest approach.
        Helps detect: Collision / accident (CA), Hard braking (HB).

    iv_others_mean_speed_when_stopped
        Mean speed of other vehicles at ticks when this vehicle is stopped.
        High value = vehicle stopped anomalously while surrounding traffic moves.
        Helps detect: Stalled vehicle (SV).
    """
    feats: Dict = {}
    NAN = float("nan")

    # ── Guard: need position + time in both dataframes ───────────────────────
    req_pos = {"x_hat", "y_hat", "t", "global_track_id"}
    if not req_pos.issubset(traj.columns) or not req_pos.issubset(all_trajs.columns):
        _iv_nan_all(feats, NAN)
        return feats

    others = all_trajs[all_trajs["global_track_id"] != track_id].copy()
    feats["iv_n_other_vehicles"] = int(others["global_track_id"].nunique())

    if others.empty or feats["iv_n_other_vehicles"] == 0:
        _iv_nan_all(feats, NAN, skip={"iv_n_other_vehicles"})
        return feats

    # ── Snap timestamps to 50 Hz tick indices ───────────────────────────────
    DT = 0.02   # 50 Hz nominal step

    has_vel = ({"vx_hat", "vy_hat"}.issubset(traj.columns) and
               {"vx_hat", "vy_hat"}.issubset(others.columns))
    has_decel = has_vel and {"ax_hat", "ay_hat"}.issubset(traj.columns)

    # Columns to pull from own trajectory
    own_cols = ["t", "x_hat", "y_hat"]
    oth_cols = ["t", "x_hat", "y_hat"]
    if has_vel:
        own_cols += ["vx_hat", "vy_hat"]
        oth_cols += ["vx_hat", "vy_hat"]

    own = traj[own_cols].dropna(subset=["x_hat", "y_hat"]).copy()
    oth = others[oth_cols + ["global_track_id"]].dropna(subset=["x_hat", "y_hat"]).copy()

    own["tick"] = (own["t"] / DT).round().astype(int)
    oth["tick"] = (oth["t"] / DT).round().astype(int)

    # Merge on tick: one own row × all other rows at that tick
    merged = own.merge(
        oth.drop(columns=["global_track_id"]),
        on="tick", suffixes=("_own", "_oth"),
    )
    if merged.empty:
        _iv_nan_all(feats, NAN, skip={"iv_n_other_vehicles"})
        return feats

    # ── 2D Euclidean distance ────────────────────────────────────────────────
    merged["dist_m"] = np.sqrt(
        (merged["x_hat_own"] - merged["x_hat_oth"]) ** 2 +
        (merged["y_hat_own"] - merged["y_hat_oth"]) ** 2
    )

    # Per-tick minimum distance (closest other vehicle at each moment)
    min_dist_per_tick = merged.groupby("tick")["dist_m"].min()

    feats["iv_min_dist_m"]              = float(min_dist_per_tick.min())
    feats["iv_mean_min_dist_m"]         = float(min_dist_per_tick.mean())
    feats["iv_p10_min_dist_m"]          = float(min_dist_per_tick.quantile(0.10))
    feats["iv_close_proximity_frac"]    = float((min_dist_per_tick < _CLOSE_DIST_M).mean())
    feats["iv_tailgate_proximity_frac"] = float((min_dist_per_tick < _TAILGATE_DIST_M).mean())

    # Collision / physical overlap (< one vehicle width)
    overlap_mask = min_dist_per_tick < _OVERLAP_DIST_M
    feats["iv_overlap_frac"]       = float(overlap_mask.mean())
    feats["iv_overlap_duration_s"] = float(overlap_mask.sum()) * DT

    # ── Velocity-dependent features ──────────────────────────────────────────
    if has_vel:
        # Relative position vector and velocity vector at each merged row
        dx  = (merged["x_hat_own"] - merged["x_hat_oth"]).values
        dy  = (merged["y_hat_own"] - merged["y_hat_oth"]).values
        dvx = (merged["vx_hat_own"].fillna(0) - merged["vx_hat_oth"].fillna(0)).values
        dvy = (merged["vy_hat_own"].fillna(0) - merged["vy_hat_oth"].fillna(0)).values
        dist_safe = np.maximum(merged["dist_m"].values, 1e-3)

        # Closing speed: rate of decrease of distance.
        # d(dist)/dt = (Δpos · Δvel) / dist → closing = −d(dist)/dt.
        # Clamped to ≥ 0: we only care about approaching pairs.
        closing = np.maximum(-(dx * dvx + dy * dvy) / dist_safe, 0.0)
        merged["closing_speed_mps"] = closing

        # TTC = distance / closing_speed  (inf when not closing)
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(closing > 0.1, merged["dist_m"].values / closing, np.inf)
        merged["ttc_s"] = ttc

        # Own speed at each merged row
        merged["own_speed_mps"] = np.sqrt(
            merged["vx_hat_own"].fillna(0) ** 2 +
            merged["vy_hat_own"].fillna(0) ** 2
        )
        # Time headway = dist / own_speed (only meaningful when moving)
        merged["time_headway_s"] = (
            merged["dist_m"] / merged["own_speed_mps"].clip(lower=0.5)
        )

        # Per-tick aggregates
        min_ttc_per_tick      = merged.groupby("tick")["ttc_s"].min()
        max_closing_per_tick  = merged.groupby("tick")["closing_speed_mps"].max()
        min_headway_per_tick  = merged.groupby("tick")["time_headway_s"].min()

        # iv_ttc_min_s: minimum TTC across the entire track.
        # Cap at 999 s (inf would confuse imputer); NaN if all infinite.
        finite_ttc = min_ttc_per_tick[np.isfinite(min_ttc_per_tick)]
        feats["iv_ttc_min_s"] = (
            float(finite_ttc.min()) if len(finite_ttc) else float("nan")
        )
        feats["iv_ttc_below_2s_frac"] = float(
            (min_ttc_per_tick < TTC_DANGER_S).mean()
        )
        feats["iv_closing_speed_max_mps"] = float(max_closing_per_tick.max())
        feats["iv_time_headway_mean_s"]   = float(min_headway_per_tick.mean())

        # ── Speed-proximity ratio (backward-compat inverse-TTC proxy) ────────
        speed_by_tick = merged.groupby("tick")["own_speed_mps"].first()
        common = min_dist_per_tick.index.intersection(speed_by_tick.index)
        if len(common):
            ratio = speed_by_tick[common] / min_dist_per_tick[common].clip(lower=0.5)
            feats["iv_speed_proximity_ratio_max"] = float(ratio.max())
        else:
            feats["iv_speed_proximity_ratio_max"] = NAN

        # ── Speed relative to traffic (mean speed comparison) ────────────────
        # Per-vehicle mean speed of others
        oth_speeds = []
        for other_id in others["global_track_id"].unique():
            o = others[others["global_track_id"] == other_id]
            s = np.sqrt(o["vx_hat"].fillna(0.0) ** 2 + o["vy_hat"].fillna(0.0) ** 2)
            if len(s):
                oth_speeds.append(float(s.mean()))

        own_spd_arr = np.sqrt(
            traj["vx_hat"].fillna(0.0) ** 2 + traj["vy_hat"].fillna(0.0) ** 2
        )
        own_mean_speed = float(own_spd_arr.mean()) if len(own_spd_arr) else NAN

        if oth_speeds and not math.isnan(own_mean_speed):
            others_mean = float(np.mean(oth_speeds))
            feats["iv_speed_excess_over_others_mps"] = own_mean_speed - others_mean
            feats["iv_speed_ratio_to_others"] = (
                own_mean_speed / others_mean if others_mean > 0.5 else NAN
            )
        else:
            feats["iv_speed_excess_over_others_mps"] = NAN
            feats["iv_speed_ratio_to_others"]         = NAN

        # ── Pearson correlation between own speed and traffic mean speed ──────
        # Feature: iv_speed_pearson_r_traffic
        #   Normal traffic: everyone slows/speeds together → high positive r.
        #   Speeding/stalled anomaly: vehicle decoupled from traffic → low r.
        #   Computed over ticks where both own and at least one other are present.
        #   Requires ≥ 10 shared ticks for a meaningful estimate.
        #
        # Helps detect: Stalled vehicle (SV), Speeding (S), General (GEN).
        own_tick_speed = (
            traj.assign(
                tick=((traj["t"] / DT).round().astype(int)),
                spd=np.sqrt(traj["vx_hat"].fillna(0) ** 2 +
                            traj["vy_hat"].fillna(0) ** 2),
            )
            .groupby("tick")["spd"].mean()
        )
        oth_tick_speed = (
            others.assign(
                tick=((others["t"] / DT).round().astype(int)),
                spd=np.sqrt(others["vx_hat"].fillna(0) ** 2 +
                            others["vy_hat"].fillna(0) ** 2),
            )
            .groupby("tick")["spd"].mean()
        )
        common_ticks = own_tick_speed.index.intersection(oth_tick_speed.index)
        if len(common_ticks) >= 10:
            x = own_tick_speed[common_ticks].values.astype(float)
            y = oth_tick_speed[common_ticks].values.astype(float)
            x_c = x - x.mean()
            y_c = y - y.mean()
            denom = math.sqrt(float((x_c ** 2).sum()) * float((y_c ** 2).sum()))
            feats["iv_speed_pearson_r_traffic"] = (
                float(np.dot(x_c, y_c) / denom) if denom > 0 else NAN
            )
        else:
            feats["iv_speed_pearson_r_traffic"] = NAN

        # ── Deceleration at closest approach ─────────────────────────────────
        # Feature: iv_decel_at_min_dist_mps2
        #   Longitudinal deceleration at the tick of minimum inter-vehicle
        #   distance.  Positive = braking.  Helps distinguish collision
        #   (high decel coincides with min dist) from tailgating (close but
        #   speed and decel are not extreme).
        #
        # Requires ax_hat, ay_hat to be present in the trajectory.
        # Helps detect: Collision / accident (CA), Hard braking (HB).
        if has_decel and not min_dist_per_tick.empty:
            tick_of_min = int(min_dist_per_tick.idxmin())
            # Build a tick-indexed lookup for this vehicle's ax/ay/vx/vy
            decel_lookup = (
                traj[["t", "ax_hat", "ay_hat", "vx_hat", "vy_hat"]]
                .dropna()
                .assign(tick=lambda df: (df["t"] / DT).round().astype(int))
                .drop_duplicates("tick")
                .set_index("tick")
            )
            if tick_of_min in decel_lookup.index:
                r = decel_lookup.loc[tick_of_min]
                vx_v = float(r["vx_hat"])
                vy_v = float(r["vy_hat"])
                ax_v = float(r["ax_hat"])
                ay_v = float(r["ay_hat"])
                spd_v = math.sqrt(vx_v ** 2 + vy_v ** 2)
                if spd_v > 0.1:
                    # Longitudinal component: positive = forward accel, neg = braking
                    a_long = ax_v * (vx_v / spd_v) + ay_v * (vy_v / spd_v)
                    feats["iv_decel_at_min_dist_mps2"] = float(-a_long)
                else:
                    feats["iv_decel_at_min_dist_mps2"] = NAN
            else:
                feats["iv_decel_at_min_dist_mps2"] = NAN
        else:
            feats["iv_decel_at_min_dist_mps2"] = NAN

        # ── Relative speed at closest approach ───────────────────────────────
        # iv_rel_speed_at_min_dist_mps
        #   |Δv| (magnitude of relative velocity vector) at the tick of minimum
        #   inter-vehicle distance.  High relative speed at the closest point =
        #   high kinetic-energy impact signature.
        #   Helps detect: Collision / accident (CA), Tailgating (TG).
        #
        # iv_collision_risk_proxy
        #   max(|Δv|² / max(dist, 0.5)) over all shared ticks.
        #   Dimensionally equivalent to twice the specific kinetic energy of the
        #   relative motion, scaled by proximity.  Peaks sharply just before a
        #   collision and is well-behaved (non-infinite) even when dist → 0
        #   because of the 0.5 m floor.
        #   Helps detect: Collision / accident (CA).
        rel_spd_arr = np.sqrt(dvx ** 2 + dvy ** 2)
        merged["rel_speed_mps"] = rel_spd_arr

        if not min_dist_per_tick.empty:
            tick_of_min = int(min_dist_per_tick.idxmin())
            at_min_tick = merged[merged["tick"] == tick_of_min]
            if len(at_min_tick):
                feats["iv_rel_speed_at_min_dist_mps"] = float(
                    at_min_tick["rel_speed_mps"].mean()
                )
            else:
                feats["iv_rel_speed_at_min_dist_mps"] = NAN
        else:
            feats["iv_rel_speed_at_min_dist_mps"] = NAN

        risk = (rel_spd_arr ** 2) / np.maximum(merged["dist_m"].values, 0.5)
        risk_by_tick = pd.Series(risk, index=merged["tick"].values).groupby(
            level=0
        ).max()
        feats["iv_collision_risk_proxy"] = float(risk_by_tick.max()) if len(risk_by_tick) else NAN

        # ── Others' speed when this vehicle is stopped ────────────────────────
        # Feature: iv_others_mean_speed_when_stopped
        #   Mean speed of other vehicles on ticks when this vehicle is stopped.
        #   High value = solo anomalous stop while traffic flows normally.
        #   Helps detect: Stalled vehicle (SV).
        own_spd_full = np.sqrt(
            traj["vx_hat"].fillna(0.0) ** 2 + traj["vy_hat"].fillna(0.0) ** 2
        )
        traj_tick = (traj["t"] / DT).round().astype(int).values
        stopped_ticks = set(traj_tick[own_spd_full.values < 0.5])

        if stopped_ticks:
            oth_at_stop = others[
                others["t"].notna() &
                ((others["t"] / DT).round().astype(int)).isin(stopped_ticks)
            ]
            if len(oth_at_stop):
                oth_spd_stop = np.sqrt(
                    oth_at_stop["vx_hat"].fillna(0.0) ** 2 +
                    oth_at_stop["vy_hat"].fillna(0.0) ** 2
                )
                feats["iv_others_mean_speed_when_stopped"] = float(oth_spd_stop.mean())
            else:
                feats["iv_others_mean_speed_when_stopped"] = NAN
        else:
            feats["iv_others_mean_speed_when_stopped"] = NAN

    else:
        # Velocities not available — set all velocity-dependent features to NaN
        for k in ["iv_ttc_min_s", "iv_ttc_below_2s_frac",
                  "iv_closing_speed_max_mps", "iv_time_headway_mean_s",
                  "iv_speed_proximity_ratio_max",
                  "iv_speed_excess_over_others_mps", "iv_speed_ratio_to_others",
                  "iv_speed_pearson_r_traffic", "iv_decel_at_min_dist_mps2",
                  "iv_others_mean_speed_when_stopped",
                  "iv_rel_speed_at_min_dist_mps", "iv_collision_risk_proxy"]:
            feats[k] = NAN

    return feats


def _iv_nan_all(feats: Dict, NAN: float,
                skip: Optional[set] = None) -> None:
    """Fill all inter-vehicle features with NaN (used on early-exit paths)."""
    skip = skip or set()
    for k in [
        "iv_min_dist_m", "iv_mean_min_dist_m", "iv_p10_min_dist_m",
        "iv_close_proximity_frac", "iv_tailgate_proximity_frac",
        "iv_overlap_frac", "iv_overlap_duration_s",
        "iv_ttc_min_s", "iv_ttc_below_2s_frac",
        "iv_closing_speed_max_mps", "iv_time_headway_mean_s",
        "iv_speed_proximity_ratio_max",
        "iv_speed_excess_over_others_mps", "iv_speed_ratio_to_others",
        "iv_speed_pearson_r_traffic", "iv_decel_at_min_dist_mps2",
        "iv_others_mean_speed_when_stopped",
        "iv_rel_speed_at_min_dist_mps", "iv_collision_risk_proxy",
    ]:
        if k not in skip:
            feats[k] = NAN


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class ScenarioLoader:
    """
    Load and extract features from a single scenario output folder.

    A scenario folder is expected to contain:
      - track_trajectory_T*.csv    (one per track)
      - kalman_measurement_audit.csv
      - coverage_summary.csv       (optional)
      - simstudio_export_*.xlsx    (optional, first match used)
      - *.json / *.sim.json        (optional scenario definition)

    Parameters
    ----------
    folder : str | Path
        Path to the scenario output folder.
    scenario_id : str, optional
        Label for this scenario. Defaults to folder name.
    scenario_json : str | Path, optional
        Explicit path to scenario JSON. If None, auto-detected in folder.
    """

    def __init__(
        self,
        folder: str | Path,
        scenario_id: Optional[str] = None,
        scenario_json: Optional[str | Path] = None,
    ):
        self.folder = Path(folder)
        self.scenario_id = scenario_id or self.folder.name

        # Locate scenario JSON
        if scenario_json:
            self.json_path = Path(scenario_json)
        else:
            candidates = (
                list(self.folder.glob("*.sim.json")) +
                list(self.folder.glob("*.json"))
            )
            # Exclude files that look like data outputs
            candidates = [p for p in candidates
                          if "coverage" not in p.name and
                             "export" not in p.name]
            self.json_path = candidates[0] if candidates else None

        # Locate Excel export (prefer simstudio_export_*.xlsx for xl_ features;
        # tracking_audit.xlsx is the consolidated audit workbook)
        xl_candidates = list(self.folder.glob("simstudio_export_*.xlsx"))
        self.xl_path = xl_candidates[0] if xl_candidates else None

        # Consolidated audit workbook (produced by run_all_simulations.py)
        audit_wb = self.folder / "tracking_audit.xlsx"
        self._audit_wb_path: Optional[Path] = audit_wb if audit_wb.exists() else None

        # Cache parsed data
        self._audit_df: Optional[pd.DataFrame] = None
        self._scenario_meta: Optional[Dict] = None

    # ── Internal loaders ─────────────────────────────────────────────────────

    def _get_audit(self) -> Optional[pd.DataFrame]:
        if self._audit_df is not None:
            return self._audit_df

        # Primary: CSV (legacy format)
        p = self.folder / "kalman_measurement_audit.csv"
        if p.exists():
            try:
                self._audit_df = pd.read_csv(p, low_memory=False)
                return self._audit_df
            except Exception:
                pass

        # Fallback: read from tracking_audit.xlsx → 'Kalman Audit' sheet
        if self._audit_wb_path:
            self._audit_df = _read_audit_sheet_from_xlsx(self._audit_wb_path)

        return self._audit_df

    def _get_scenario_meta(self) -> Dict:
        if self._scenario_meta is not None:
            return self._scenario_meta
        if self.json_path and self.json_path.exists():
            sc = _load_scenario_json(self.json_path)
            self._scenario_meta = _extract_scenario_meta(sc)
        else:
            # No JSON available — return NaN-filled template so columns
            # are always present in the batch DataFrame.
            self._scenario_meta = _empty_scenario_meta()
        return self._scenario_meta

    def _get_trajectory_files(self) -> List[Path]:
        return sorted(self.folder.glob("track_trajectory_T*.csv"))

    def _get_xlsx_track_ids(self) -> List[str]:
        """
        Return the list of track IDs available in tracking_audit.xlsx
        (sheet names like 'Track T000001' → 'T000001').
        Returns [] if the workbook is absent or has no Track sheets.
        """
        if not self._audit_wb_path:
            return []
        try:
            import openpyxl
            wb = openpyxl.load_workbook(
                self._audit_wb_path, read_only=True, data_only=True)
            ids = [s.replace("Track ", "")
                   for s in wb.sheetnames if s.startswith("Track T")]
            wb.close()
            return sorted(ids)
        except Exception:
            return []

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self) -> pd.DataFrame:
        """
        Extract all features for this scenario.

        Returns
        -------
        pd.DataFrame
            One row per track in this scenario.
        """
        traj_files  = self._get_trajectory_files()
        xlsx_ids    = self._get_xlsx_track_ids() if not traj_files else []
        use_xlsx    = (not traj_files) and bool(xlsx_ids)

        audit      = self._get_audit()
        sc_meta    = self._get_scenario_meta()


        if not traj_files and not use_xlsx:
            row = {
                "scenario_id":          self.scenario_id,
                "global_track_id":      float("nan"),
                "vehicle_id_oracle":    float("nan"),
                "sc_speed_limit_mps":   sc_meta.get("sc_speed_limit_mps", float("nan")),
                "das_W_est_kg":         float("nan"),
                "das_W_est_std_kg":     float("nan"),
                "das_W_est_cv":         float("nan"),
            }
            return pd.DataFrame([row])

        rows = []

        # Build an iterator that yields (track_id, traj_df) regardless of source
        def _iter_tracks():
            if use_xlsx:
                for tid in xlsx_ids:
                    df = _read_track_sheet_from_xlsx(
                        self._audit_wb_path, f"Track {tid}")
                    if df is not None and not df.empty:
                        yield tid, df
            else:
                for traj_path in traj_files:
                    try:
                        df = pd.read_csv(traj_path, low_memory=False)
                    except Exception:
                        continue
                    tid = traj_path.stem.replace("track_trajectory_", "")
                    yield tid, df

        # ── Two-pass approach for inter-vehicle features ───────────────────
        # Pass 1: load all tracks into memory and build the combined DataFrame
        # that _features_inter_vehicle() needs to compute proximity features.
        all_track_data: List[Tuple[str, pd.DataFrame]] = []
        for tid, df in _iter_tracks():
            df = df.copy()
            df["global_track_id"] = tid   # ensure column present for join
            all_track_data.append((tid, df))

        if all_track_data:
            all_trajs_combined = pd.concat(
                [df for _, df in all_track_data], ignore_index=True)
        else:
            all_trajs_combined = pd.DataFrame()

        # Pass 2: extract features per track (now has access to all_trajs_combined)
        for track_id, traj in all_track_data:

            # Infer oracle vehicle_id from trajectory (mode of the column)
            vehicle_id = None
            if "vehicle_id_oracle" in traj.columns:
                vc = traj["vehicle_id_oracle"].dropna().mode()
                vehicle_id = str(vc.iloc[0]) if len(vc) else None

            # ── Gather all feature groups ─────────────────────────────────
            row: Dict = {
                "scenario_id":      self.scenario_id,
                "global_track_id":  track_id,
                "vehicle_id_oracle": vehicle_id,
            }

            row["sc_speed_limit_mps"] = sc_meta.get("sc_speed_limit_mps", float("nan"))
            row.update(_features_from_trajectory(
                traj, track_id,
                seg_headings=sc_meta.get("sc_seg_headings", {}),
            ))
            row.update(_features_from_audit(
                audit, track_id,
                fiber_offset_m = sc_meta.get("sc_das_fiber_offset_m", float("nan")),
                d0_m           = sc_meta.get("sc_das_d0_m", D0_DEFAULT),
                snr_th         = sc_meta.get("sc_das_snr_th", SNR_TH_DEFAULT),
            ))
            row.update(_features_from_coverage(
                traj, audit, track_id,
                d0_m=sc_meta.get("sc_das_d0_m", D0_DEFAULT),
            ))
            row.update(_features_from_excel(self.xl_path, track_id, vehicle_id))
            row.update(_features_inter_vehicle(traj, all_trajs_combined, track_id))

            # ── Speed-over-limit (needs speed_limit from scenario) ────────
            if "kin_speed_max_mps" in row and not math.isnan(
                row.get("sc_speed_limit_mps", float("nan"))
            ):
                limit = row["sc_speed_limit_mps"]
                if "vx_hat" in traj.columns and "vy_hat" in traj.columns:
                    speed = np.sqrt(
                        traj["vx_hat"].fillna(0.0)**2 +
                        traj["vy_hat"].fillna(0.0)**2
                    )
                    over = speed > limit * 1.05
                    row["kin_speed_over_limit_frac"] = float(over.sum()) / max(len(speed), 1)
                    excess = (speed[over] - limit).values if over.sum() else np.array([0.0])
                    row["kin_speed_excess_max_mps"]  = float(excess.max())
                    row["kin_speed_excess_mean_mps"] = float(excess.mean())

            # ── DAS weight estimation (mean only; variability computed in _features_from_audit)
            w_est = _estimate_weight(
                das_snr_mean   = row.get("das_snr_mean", float("nan")),
                snr_th         = sc_meta.get("sc_das_snr_th", SNR_TH_DEFAULT),
                fiber_offset_m = sc_meta.get("sc_das_fiber_offset_m", float("nan")),
                d0_m           = sc_meta.get("sc_das_d0_m", D0_DEFAULT),
            )
            row["das_W_est_kg"]    = w_est
            row["das_W_est_class"] = _classify_weight(w_est) if not math.isnan(w_est) else "unknown"

            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure consistent column order per FEATURE_CATALOG
        return df

    def describe(self) -> str:
        """Return a human-readable summary of what was found in this folder."""
        traj_files = self._get_trajectory_files()
        audit_exists = (self.folder / "kalman_measurement_audit.csv").exists()
        lines = [
            f"ScenarioLoader: {self.scenario_id}",
            f"  Folder         : {self.folder}",
            f"  Scenario JSON  : {self.json_path or 'not found'}",
            f"  Excel export   : {self.xl_path or 'not found'}",
            f"  Trajectory CSVs: {len(traj_files)} file(s)",
            f"  Audit CSV      : {'✓' if audit_exists else '✗ (not found)'}",
        ]
        sc_meta = self._get_scenario_meta()
        if sc_meta:
            lines.append(f"  Anomaly type   : {sc_meta.get('sc_anomaly_type', 'unknown')}")
            lines.append(f"  Vehicles       : {sc_meta.get('sc_n_vehicles', '?')}")
            lines.append(f"  DAS present    : {bool(sc_meta.get('sc_has_das', 0))}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# New-feature coverage logging
# ---------------------------------------------------------------------------

# Features added in the second round of improvements.  Checked at the end of
# each batch extraction to verify how many tracks actually received a value.
_NEW_FEATURES: List[str] = [
    # Inter-vehicle — velocity-dependent (require vx_hat/vy_hat in trajectories)
    "iv_ttc_min_s",                        # TTC: Tailgating / Collision
    "iv_ttc_below_2s_frac",                # chronic tailgating fraction
    "iv_closing_speed_max_mps",            # approach rate
    "iv_time_headway_mean_s",              # headway proxy for tailgating
    "iv_speed_pearson_r_traffic",          # coupling with traffic flow
    "iv_decel_at_min_dist_mps2",           # deceleration at closest approach
    "iv_rel_speed_at_min_dist_mps",        # |Δv| at closest approach
    "iv_collision_risk_proxy",             # |Δv|² / dist — kinetic energy proxy
    # Kinematic — lateral motion
    "kin_lateral_vel_mean_mps",            # mean |vy_hat|
    "kin_lateral_vel_max_mps",             # max |vy_hat|
    "kin_lateral_accel_max_mps2",          # max |ay_hat|
    "kin_heading_change_rate_max_rad_per_s",  # max angular rate
    "kin_speed_jump_max_mps",             # max |Δspeed| per tick
    "kin_road_heading_diff_mean_rad",      # deviation from road heading
    # Kalman quality
    "kf_sigma_growth_rate_during_gaps_m_per_s",
    "kf_post_dropout_jump_m",             # position jump at gap end
    # Audit
    "aud_skip_chi2_frac",                 # fraction of chi-gate rejections
    # DAS physics consistency
    "das_amplitude_vs_dist_pearson_r",
]


def _log_new_feature_coverage(df: pd.DataFrame) -> None:
    """
    Print a compact coverage table for the 8 new priority features,
    showing how many tracks received a non-NaN value vs. total tracks.
    """
    n_total = len(df)
    if n_total == 0:
        return
    print("\n  ┌─ New-feature coverage ──────────────────────────────────────────")
    for feat in _NEW_FEATURES:
        if feat in df.columns:
            n_valid = int(df[feat].notna().sum())
            pct = 100 * n_valid / n_total
            bar_len = int(pct / 5)          # 20-char bar at 5 % per char
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  │  {feat:<48s}  {n_valid:>4}/{n_total}  {bar}  {pct:5.1f}%")
        else:
            print(f"  │  {feat:<48s}  (column absent)")
    print("  └────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_features(
    folders: List[str | Path],
    scenario_id_fn=None,
    scenario_json_map: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract features from multiple scenario folders and return a single
    combined DataFrame.

    Parameters
    ----------
    folders : list of str | Path
        Scenario output folders to process.
    scenario_id_fn : callable, optional
        Function (path) → str for scenario ID. Defaults to folder name.
    scenario_json_map : dict, optional
        Maps folder path strings to explicit JSON paths, for scenarios where
        the JSON is not co-located with the outputs.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Concatenated feature DataFrame, one row per (scenario, track).
    """
    if scenario_id_fn is None:
        scenario_id_fn = lambda p: Path(p).name

    all_frames = []
    for folder in folders:
        folder = Path(folder)
        sc_id  = scenario_id_fn(folder)
        sc_json = (scenario_json_map or {}).get(str(folder))

        loader = ScenarioLoader(folder, scenario_id=sc_id,
                                scenario_json=sc_json)
        if verbose:
            print(f"  Loading: {sc_id} ... ", end="", flush=True)

        df = loader.extract()
        all_frames.append(df)

        if verbose:
            n_tracks = len(df)
            n_feats  = len(df.columns)
            anomaly  = df["sc_anomaly_type"].iloc[0] if "sc_anomaly_type" in df.columns else "?"
            print(f"{n_tracks} track(s), {n_feats} features. [{anomaly}]")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    if verbose:
        _log_new_feature_coverage(combined)

    return combined


# ---------------------------------------------------------------------------
# Feature catalog (documentation)
# ---------------------------------------------------------------------------

FEATURE_CATALOG = {
    # ── Identifiers ──────────────────────────────────────────────────────────
    "scenario_id":              "Folder name or user-supplied scenario label",
    "global_track_id":          "Track ID (e.g. T000001)",
    "vehicle_id_oracle":        "Ground-truth vehicle identifier",

    # ── Scenario metadata (from JSON) ─────────────────────────────────────────
    "sc_n_vehicles":            "Number of vehicles in the scenario",
    "sc_n_frozen":              "Vehicles with frozen=True (stalled/obstacle)",
    "sc_n_speeding":            "Vehicles with ignore_speed_limit=True",
    "sc_n_collision":           "Vehicles with allow_collision=True",
    "sc_n_weaving":             "Vehicles with lateral_mode='weave'",
    "sc_n_straddling":          "Vehicles with lateral_mode='straddle'",
    "sc_n_braking_override":    "Vehicles with a_cmd_override_mps2 < -2",
    "sc_n_tailgating":          "Vehicles with min_gap_m < 1.5 m",
    "sc_weight_kg_list":        "Comma-separated list of vehicle weights [kg]",
    "sc_speed_limit_mps":       "Min speed limit across segments [m/s]",
    "sc_has_das":               "1 if DAS sensors present, else 0",
    "sc_das_fiber_offset_m":    "DAS fiber lateral offset from road [m]",
    "sc_das_d0_m":              "DAS singularity floor d0 [m]",
    "sc_das_noise_std":         "DAS trace noise standard deviation",
    "sc_das_snr_th":            "SNR threshold (traffic-level dependent)",
    "sc_anomaly_type":          "Heuristic anomaly label from JSON flags",

    # ── Kinematic features (from trajectory CSV, Kalman estimates) ────────────
    "kin_speed_mean_mps":       "Mean estimated speed [m/s]",
    "kin_speed_max_mps":        "Max estimated speed [m/s]",
    "kin_speed_min_mps":        "Min estimated speed [m/s]",
    "kin_speed_std_mps":        "Std dev of estimated speed [m/s]",
    "kin_speed_cv":             "Coefficient of variation of speed (std/mean); low = constant speed (tailgating proxy)",
    "kin_speed_over_limit_frac":"Fraction of timesteps where speed > speed_limit × 1.05",
    "kin_accel_mean_abs_mps2":  "Mean absolute acceleration magnitude [m/s²]",
    "kin_accel_max_abs_mps2":   "Max absolute acceleration magnitude [m/s²]",
    "kin_accel_std_mps2":       "Std dev of acceleration magnitude [m/s²]",
    "kin_phys_impossible_v":    "Rows where estimated speed > 40 m/s",
    "kin_phys_impossible_a":    "Rows where estimated accel > 8 m/s²",
    # Stall / stopped-vehicle features
    "kin_stopped_frac":             "Fraction of timesteps where speed < 0.5 m/s (stopped)",
    "kin_stop_event_count":         "Number of distinct stop events (transitions to stopped state)",
    "kin_max_stopped_steps":        "Longest continuous stopped period [timesteps]",
    "kin_max_stopped_duration_s":   "Longest continuous stopped period [seconds]",
    # Longitudinal deceleration (directional — captures hard braking)
    "kin_decel_max_mps2":       "Max deceleration rate [m/s²] (positive = braking hard)",
    "kin_high_decel_frac":      "Fraction of timesteps with deceleration > 3 m/s²",
    "kin_decel_event_count":    "Number of distinct hard-braking events (decel > 3 m/s²)",
    "kin_jerk_max_mps3":        "Max jerk magnitude [m/s³] — spike = impact / sudden stop",
    "kin_jerk_mean_abs_mps3":   "Mean absolute jerk [m/s³] — high = erratic driving",
    # Lateral features — enhanced for weaving & straddling
    "kin_lateral_dev_mean_m":   "Mean lateral deviation of y_hat from its mean [m]",
    "kin_lateral_dev_max_m":    "Max lateral deviation of y_hat [m]",
    "kin_lateral_dev_std_m":    "Std dev of lateral deviation [m]",
    "kin_lateral_peak_to_peak_m":"Range (max−min) of lateral position [m] — large = big swing",
    "kin_lateral_oscillation_ratio": "std(lat_dev) / mean(lat_dev): high = weaving",
    "kin_lateral_zero_crossings":    "Number of times lateral position crosses its mean — high = weaving",
    "kin_lateral_oscillation_rate_hz": "Lateral crossings per second [Hz] — periodic = weaving",
    "kin_lateral_bias_m":            "Absolute mean lateral offset from lane centre [m] — high = straddling",
    "kin_lateral_consistent_side_frac": "Fraction of time on same side as mean offset — near 1.0 = straddling, ~0.5 = weaving",

    # ── Kalman quality features ───────────────────────────────────────────────
    "kf_pos_err_mean_m":        "Mean position error vs ground truth [m]",
    "kf_pos_err_max_m":         "Max position error [m]",
    "kf_pos_err_std_m":         "Std dev of position error [m]",
    "kf_pos_err_p90_m":         "90th percentile position error [m]",
    "kf_track_rmse_m":          "Overall track RMSE [m]",
    "kf_rolling_rmse_spike_count": "Count of 10-step windows where rolling RMSE > 2× track RMSE",
    "kf_sigma_pos_mean_m":      "Mean Kalman position uncertainty σ [m]",
    "kf_sigma_pos_max_m":       "Max Kalman position uncertainty σ [m]",
    "kf_consistency_ratio_mean":"Mean(pos_err / sigma_pos): >2 = overconfident filter",
    "kf_consistency_ratio_max": "Max(pos_err / sigma_pos)",
    "kf_overconfident_frac":    "Fraction of rows where pos_err > 2 × sigma_pos",

    # ── Coverage: basic prediction-gap counters ───────────────────────────────
    "cov_total_rows":               "Total trajectory rows (measurement + prediction)",
    "cov_pred_only_count":          "Number of prediction-only (no measurement) rows",
    "cov_pred_only_frac":           "Fraction of rows that are prediction-only",
    "cov_max_consec_pred":          "Max consecutive prediction-only rows (longest gap)",
    "cov_n_dropout_events":         "Number of distinct measurement dropout events",
    "cov_duration_s":               "Track duration [s]",

    # ── Coverage Group A: per-sensor active fractions ─────────────────────────
    "cov_das_active_frac":          "Fraction of track rows where DAS has a measurement",
    "cov_cam_active_frac":          "Fraction of track rows where Camera has a measurement",
    "cov_gps_active_frac":          "Fraction of track rows where GPS has a measurement",
    "cov_multi_sensor_frac":        "Fraction of rows where ≥2 sensors are active simultaneously",
    "cov_single_sensor_frac":       "Fraction of rows where exactly 1 sensor is active",
    "cov_no_sensor_frac":           "Fraction of rows where no sensor provides a measurement (pure prediction)",
    "cov_das_cam_overlap_frac":     "Fraction of rows where both DAS and Camera are active",
    "cov_das_gps_overlap_frac":     "Fraction of rows where both DAS and GPS are active",
    "cov_cam_gps_overlap_frac":     "Fraction of rows where both Camera and GPS are active",
    "cov_all_sensors_frac":         "Fraction of rows where all three sensors (DAS+Cam+GPS) are active",
    "cov_dominant_sensor":          "Sensor active for the largest fraction of rows (DAS/Camera/GPS/none)",

    # ── Coverage Group B: fusion benefit (does redundancy reduce error?) ──────
    "cov_err_das_only_m":           "Mean position error [m] on rows where DAS is the sole active sensor",
    "cov_err_cam_only_m":           "Mean position error [m] on rows where Camera is the sole active sensor",
    "cov_err_gps_only_m":           "Mean position error [m] on rows where GPS is the sole active sensor",
    "cov_err_multi_sensor_m":       "Mean position error [m] on rows where ≥2 sensors are active (fusion benefit)",
    "cov_err_no_sensor_m":          "Mean position error [m] on prediction-only rows (no sensor active)",
    "cov_fusion_benefit_vs_best_single": "best_single_err − multi_sensor_err [m]: positive = fusion helps",
    "cov_err_cam_no_das_m":         "Mean position error [m] when Camera is active but DAS is not",
    "cov_err_cam_with_das_m":       "Mean position error [m] when Camera and DAS are both active",
    "cov_das_fusion_benefit_m":     "err_cam_no_das − err_cam_with_das [m]: positive = DAS helps Camera",

    # ── Coverage Group C: temporal and spatial gap structure ──────────────────
    "cov_gap_frac_early":           "Fraction of the first third of the track that is prediction-only",
    "cov_gap_frac_middle":          "Fraction of the middle third of the track that is prediction-only",
    "cov_gap_frac_late":            "Fraction of the last third of the track that is prediction-only",
    "cov_gap_clustering_cv":        "Coefficient of variation of gap row indices — high = gaps are clustered",
    "cov_gap_spatial_pos_mean_norm":"Normalised mean spatial position of gaps (0=track-start, 1=track-end)",
    "cov_gap_spatial_spread_norm":  "Normalised std dev of gap spatial positions (spread along the road)",
    "cov_high_speed_gap_frac":      "Fraction of prediction-only rows where estimated speed > 50% of v_max",
    "cov_mean_speed_at_gaps_mps":   "Mean estimated speed [m/s] during prediction-only (gap) rows",

    # ── Coverage Group D: effective sensor update rates ───────────────────────
    "cov_das_hz_est":               "Estimated DAS update rate [Hz] = active_frac × n_rows / duration",
    "cov_cam_hz_est":               "Estimated Camera update rate [Hz]",
    "cov_gps_hz_est":               "Estimated GPS update rate [Hz]",
    "cov_das_update_share":         "Fraction of accepted Kalman updates that came from DAS",
    "cov_cam_update_share":         "Fraction of accepted Kalman updates that came from Camera",
    "cov_gps_update_share":         "Fraction of accepted Kalman updates that came from GPS",
    "cov_dominant_kalman_updater":  "Sensor type that contributed the most accepted Kalman updates",

    # ── Coverage Group D: DAS physics from audit (new fields) ─────────────────
    "cov_das_amplitude_mean":       "Mean DAS raw amplitude A = W/(r+d0)² across accepted DAS rows",
    "cov_das_amplitude_std":        "Std dev of DAS amplitude — high variance may indicate weaving",
    "cov_das_amplitude_cv":         "Coefficient of variation of DAS amplitude (std/mean)",
    "cov_das_fiber_dist_mean_m":    "Mean perpendicular vehicle-to-fiber distance r [m]",
    "cov_das_fiber_dist_std_m":     "Std dev of fiber distance — high = lateral oscillation",
    "cov_das_fiber_dist_max_m":     "Maximum fiber distance observed [m]",
    "cov_das_fiber_dist_cv":        "Coefficient of variation of fiber distance (std/mean)",
    "cov_das_lateral_offset_mean_m":"Mean vehicle lateral offset from lane centerline [m]",
    "cov_das_lateral_offset_std_m": "Std dev of lateral offset — high = lane-straddling/weaving",
    "cov_das_lateral_offset_max_m": "Maximum absolute lateral offset from lane centerline [m]",

    # ── Sensor disagreement ───────────────────────────────────────────────────
    "dis_das_cam_mean_m":       "Mean |DAS_x - Camera_x| when both present [m]",
    "dis_das_cam_max_m":        "Max |DAS_x - Camera_x| [m]",
    "dis_das_hat_mean_m":       "Mean |DAS_x - x_hat| [m]",
    "dis_das_hat_max_m":        "Max |DAS_x - x_hat| [m]",
    "dis_cam_hat_mean_m":       "Mean |Camera_x - x_hat| [m]",
    "dis_cam_hat_max_m":        "Max |Camera_x - x_hat| [m]",
    "dis_gps_hat_mean_m":       "Mean |GPS_x - x_hat| [m]",
    "dis_gps_hat_max_m":        "Max |GPS_x - x_hat| [m]",
    "dis_das_only_err_mean_m":  "Mean pos_err on DAS-only rows [m]",
    "dis_cam_only_err_mean_m":  "Mean pos_err on camera-only rows [m]",
    "dis_das_worse_than_cam":   "1 if DAS-only error > camera-only error (DAS degrades fusion)",

    # ── DAS sensor quality ────────────────────────────────────────────────────
    "das_n_measurements":       "Total DAS measurements for this track",
    "das_snr_mean":             "Mean DAS SNR",
    "das_snr_min":              "Min DAS SNR",
    "das_snr_max":              "Max DAS SNR",
    "das_snr_std":              "Std dev of DAS SNR",
    "das_snr_low_frac":         "Fraction of DAS rows with SNR < 5 (low quality)",
    "das_sigma_mean_m":         "Mean DAS position uncertainty σ [m]",
    "das_sigma_max_m":          "Max DAS position uncertainty σ [m]",
    "das_confidence_mean":      "Mean DAS reliability (SNR-derived confidence)",
    "das_sigma_v_mean_mps":     "Mean DAS velocity uncertainty σ_v [m/s]",
    "das_accept_frac":          "Fraction of DAS measurements accepted by Kalman",
    "das_skip_frac":            "Fraction of DAS measurements skipped",

    # ── DAS weight estimation ─────────────────────────────────────────────────
    "das_W_est_kg":             "Estimated vehicle weight [kg] from SNR (lane-center approx)",
    "das_W_est_class":          "Weight class of W_est (motorcycle/car/bus/...)",
    "das_declared_weight_kg":   "Declared vehicle weight from scenario JSON [kg]",
    "das_declared_weight_class":"Weight class of declared weight",
    "das_weight_ratio":         "W_est / declared_weight (1.0 = perfect match)",
    "das_weight_anomaly":       "1 if |W_est/declared - 1| > 0.5 (50% discrepancy)",

    # ── Camera sensor quality ─────────────────────────────────────────────────
    "cam_n_measurements":       "Total camera measurements for this track",
    "cam_confidence_mean":      "Mean camera detection confidence",
    "cam_confidence_min":       "Min camera detection confidence",
    "cam_confidence_std":       "Std dev of camera confidence",
    "cam_low_conf_frac":        "Fraction of camera rows with confidence < 0.30",
    "cam_sigma_mean_m":         "Mean camera position uncertainty σ [m]",
    "cam_accept_frac":          "Fraction of camera measurements accepted by Kalman",

    # ── GPS sensor quality ────────────────────────────────────────────────────
    "gps_n_measurements":       "Total GPS measurements for this track",
    "gps_sigma_mean_m":         "Mean GPS position uncertainty σ [m]",

    # ── Audit cross-sensor ────────────────────────────────────────────────────
    "aud_skip_total":           "Total measurements skipped across all sensor types",
    "aud_skip_frac":            "Fraction of all measurements skipped",
    "aud_skip_reason_top":      "Most common skip reason (string)",

    # ── Excel-derived features ────────────────────────────────────────────────
    "xl_n_vehicle_stuck_events":"Number of vehicle_stuck events in Issues sheet",
    "xl_first_stuck_t":         "Timestamp of first vehicle_stuck event [s]",
    "xl_rmse_gps_rmse_pos_m":   "GPS position RMSE from Excel RMSE sheet [m]",
    "xl_rmse_camera_rmse_pos_m":"Camera position RMSE from Excel RMSE sheet [m]",
    "xl_rmse_das_rmse_pos_m":   "DAS position RMSE from Excel RMSE sheet [m]",
    "xl_true_speed_mean_mps":   "Mean true (ground truth) speed from Vehicles sheet [m/s]",
    "xl_true_speed_max_mps":    "Max true speed [m/s]",
    "xl_true_speed_std_mps":    "Std dev of true speed [m/s]",
    "xl_true_accel_max_abs_mps2":"Max absolute true acceleration [m/s²]",
    "xl_true_min_accel_mps2":   "Most negative true acceleration (hardest braking) [m/s²]",
}


# ---------------------------------------------------------------------------
# Live GUI bridge — build features directly from in-memory event objects
# ---------------------------------------------------------------------------

def events_to_feature_dataframe(
    events,
    scenario_json_path=None,
    scenario_id: str = "live_simulation",
) -> "pd.DataFrame":
    """
    Build the same feature DataFrame that ``ScenarioLoader.extract()`` produces
    from files on disk, but entirely from the in-memory event objects that the
    SimStudio GUI accumulates in ``_all_events`` during a live session.

    No files are read or written.  The function re-uses every existing
    ``_features_from_*`` helper unchanged, so features computed here are
    numerically identical to those produced by the batch script.

    Parameters
    ----------
    events : iterable of Event objects
        Each object must expose ``.topic`` (str) and ``.payload`` (dict).
        Typically ``list(app._all_events)``.
    scenario_json_path : str | Path | None
        Optional path to the ``.sim.json`` that was loaded.  Used only for
        scenario-level metadata (speed limit, DAS parameters).  If ``None``
        those features are left as NaN.
    scenario_id : str
        Label embedded in the ``scenario_id`` column.

    Returns
    -------
    pd.DataFrame
        One row per vehicle, same columns as ``ScenarioLoader.extract()``.
        Columns that require the Kalman post-processing pipeline
        (``kf_pos_err_*``, ``kf_sigma_*``, Excel-only ``xl_*`` features)
        are present but filled with NaN.
    """
    from collections import defaultdict

    # ── 1. Partition events by topic and vehicle_id ───────────────────────────
    veh_states:    dict = defaultdict(list)
    das_payloads:  dict = defaultdict(list)
    gps_payloads:  dict = defaultdict(list)
    cam_payloads:  dict = defaultdict(list)
    collision_vids: set = set()

    for ev in events:
        topic = getattr(ev, "topic", "")
        p     = getattr(ev, "payload", None) or {}
        vid   = str(p.get("vehicle_id", "") or "")
        if topic == "world.vehicle_state":
            if vid:
                veh_states[vid].append(p)
        elif topic == "sensor.das":
            if vid:
                das_payloads[vid].append(p)
        elif topic == "sensor.gps":
            if vid:
                gps_payloads[vid].append(p)
        elif topic == "sensor.camera":
            if vid:
                cam_payloads[vid].append(p)
        elif topic == "world.collision":
            for v in (p.get("vehicle_ids") or
                      [p.get("follower_id"), p.get("leader_id")]):
                if v:
                    collision_vids.add(str(v))

    all_vids = sorted(veh_states.keys())
    if not all_vids:
        return pd.DataFrame()

    # ── 2. Scenario-level metadata ────────────────────────────────────────────
    if scenario_json_path is not None:
        _jp = Path(scenario_json_path)
        if _jp.exists():
            sc      = _load_scenario_json(_jp)
            sc_meta = _extract_scenario_meta(sc)
        else:
            sc_meta = _empty_scenario_meta()
    else:
        sc_meta = _empty_scenario_meta()

    fiber_offset_m = sc_meta.get("sc_das_fiber_offset_m", float("nan"))
    d0_m           = sc_meta.get("sc_das_d0_m",           D0_DEFAULT)
    snr_th_val     = sc_meta.get("sc_das_snr_th",         SNR_TH_DEFAULT)

    # ── 3. Build per-vehicle trajectory DataFrames ────────────────────────────
    # Column layout mirrors track_trajectory_T*.csv so _features_from_trajectory
    # can be called unchanged.
    #
    # Extra columns added for live mode compatibility:
    #   das_x / cam_x / gps_x — NaN where sensor has no nearby event, x_hat
    #       otherwise.  Used by _features_from_coverage() to compute per-sensor
    #       active-fraction and gap features without reading CSV files.
    #   update_kind — "prediction_only" where no sensor is active, otherwise
    #       "measurement_update".  Drives cov_pred_only_frac / cov_gap_* features.
    def _ma5(arr):
        """5-sample moving-average — reduces oracle acceleration noise."""
        n = len(arr)
        if n < 5:
            return arr.copy()
        out = np.empty(n)
        for i in range(n):
            lo = max(0, i - 2)
            hi = min(n, i + 3)
            out[i] = float(np.mean(arr[lo:hi]))
        return out

    def _sensor_active_mask(t_arr: np.ndarray, sensor_times: list,
                            window: float = 0.12) -> np.ndarray:
        """Exclusive-assignment mask: each sensor event claims its single nearest
        trajectory timestep (within `window`).  This matches the batch Kalman
        extractor semantics where each measurement triggers at most one trajectory
        row, so active-fraction ≈ n_sensor_events / n_rows rather than blowing
        up to 100% when the sensor fires faster than the window width."""
        mask = np.zeros(len(t_arr), dtype=bool)
        if not sensor_times or len(t_arr) == 0:
            return mask
        t_sorted = np.sort(t_arr)
        # For speed, work on sorted t_arr and map back via argsort
        order = np.argsort(t_arr)
        t_s   = t_arr[order]          # sorted view
        m_s   = np.zeros(len(t_arr), dtype=bool)
        for ts in sorted(sensor_times):
            # Binary search for nearest trajectory timestep
            idx = np.searchsorted(t_s, ts)
            best = -1
            best_dist = window + 1.0
            for cand in (idx - 1, idx):
                if 0 <= cand < len(t_s):
                    d = abs(t_s[cand] - ts)
                    if d < best_dist:
                        best_dist = d
                        best = cand
            if best >= 0 and best_dist <= window:
                m_s[best] = True      # may be claimed by multiple events — that's fine
        # Undo the sort
        mask[order] = m_s
        return mask

    all_track_data: list = []   # [(vid, traj_df), ...]

    for vid in all_vids:
        states = sorted(veh_states[vid],
                        key=lambda p: float(p.get("t", 0) or 0))

        # Deduplicate by timestamp: if _all_events accumulated vehicle_state
        # events from more than one simulation run (possible when the GUI's
        # Start button is pressed again without resetting the scene), two
        # runs produce events with nearly-identical but not-quite-equal
        # wall-clock timestamps.  After sorting these interleave at sub-ms
        # intervals, and the heading values at those tiny-dt pairs can differ
        # by up to π rad, producing phantom rates of 10,000+ rad/s.
        # Keeping only the LAST event at each exact timestamp is safe because
        # sim_core fires exactly one world.vehicle_state per vehicle per step.
        _seen_t: dict = {}
        for _p in states:
            _seen_t[float(_p.get("t", 0) or 0)] = _p
        states = [_seen_t[_k] for _k in sorted(_seen_t.keys())]

        if len(states) < 3:
            continue

        t_arr  = np.array([float(p.get("t",            0) or 0) for p in states])
        v_arr  = np.array([float(p.get("v",            0) or 0) for p in states])
        h_arr  = np.array([float(p.get("heading_rad",  0) or 0) for p in states])
        x_arr  = np.array([float(p.get("x",            0) or 0) for p in states])
        y_arr  = np.array([float(p.get("y",            0) or 0) for p in states])
        # Smooth world-frame oracle accelerations (match training data quality)
        ax_raw = np.array([float(p.get("ax_world_mps2", 0) or 0) for p in states])
        ay_raw = np.array([float(p.get("ay_world_mps2", 0) or 0) for p in states])
        ax_sm  = _ma5(ax_raw)
        ay_sm  = _ma5(ay_raw)

        # World-frame velocity components (used by _features_from_trajectory)
        vx_hat = v_arr * np.cos(h_arr)
        vy_hat = v_arr * np.sin(h_arr)

        # Sensor active masks — derived from event timestamps
        _das_times = [float(p.get("t", 0) or 0) for p in das_payloads[vid]]
        _gps_times = [float(p.get("t", 0) or 0) for p in gps_payloads[vid]]
        _cam_times = [float(p.get("t", 0) or 0) for p in cam_payloads[vid]]
        das_active = _sensor_active_mask(t_arr, _das_times)
        gps_active = _sensor_active_mask(t_arr, _gps_times)
        cam_active = _sensor_active_mask(t_arr, _cam_times)
        any_active = das_active | gps_active | cam_active

        traj = pd.DataFrame({
            "t":                 t_arr,
            "x_hat":             x_arr,
            "y_hat":             y_arr,
            "vx_hat":            vx_hat,
            "vy_hat":            vy_hat,
            "heading_hat":       h_arr,
            "ax_hat":            ax_sm,    # smoothed → used for jerk/decel
            "ay_hat":            ay_sm,
            "vehicle_id_oracle": vid,
            "global_track_id":   vid,
            # Kalman quality columns — NaN because live mode has no KF pipeline
            "pos_err_m":         np.nan,
            "sigma_pos_m":       np.nan,
            "sigma_v_mps":       np.nan,
            # Sensor presence columns — NaN when sensor inactive, x_hat when active.
            # _features_from_coverage() uses notna() to determine active fractions.
            "das_x":  np.where(das_active, x_arr, np.nan),
            "cam_x":  np.where(cam_active, x_arr, np.nan),
            "gps_x":  np.where(gps_active, x_arr, np.nan),
            # Kalman update kind — "prediction_only" when no sensor has a reading.
            # Drives cov_pred_only_frac, cov_gap_* features.
            "update_kind": np.where(any_active,
                                    "measurement_update", "prediction_only"),
        })
        all_track_data.append((vid, traj))

    if not all_track_data:
        return pd.DataFrame()

    # Combined DataFrame used by _features_inter_vehicle
    all_trajs_combined = pd.concat(
        [df for _, df in all_track_data], ignore_index=True
    )

    # ── 4. Build audit DataFrame (all sensors, all vehicles, all rows) ────────
    # Mirrors kalman_measurement_audit.csv so _features_from_audit and
    # _features_from_coverage can be called unchanged.
    #
    # IMPORTANT — "skipped" semantics:
    #   In the batch Kalman pipeline, "skipped=1" means the Kalman chi-squared
    #   gate REJECTED the measurement.  In normal operation this is essentially
    #   never triggered (training mean=0, std=0.001), so aud_skip_total≈0.
    #   In live mode we have no Kalman gate — setting skipped based on sensor
    #   reliability/confidence would inject z-scores of 10,000–80,000 for
    #   perfectly normal vehicles.  Always use skipped=0 in live mode.
    #
    # IMPORTANT — Camera sensor_type casing:
    #   _features_from_audit() and _features_from_coverage() filter for
    #   sensor_type == "Camera" (title-case), matching the batch CSV convention.
    audit_rows: list = []
    for vid in all_vids:
        for p in das_payloads[vid]:
            _rel = float(p.get("reliability", p.get("confidence", 0)) or 0)
            audit_rows.append({
                "global_track_id":  vid,
                "vehicle_id_oracle": vid,
                "sensor_type":       "DAS",
                "t":                 float(p.get("t",           0) or 0),
                "sigma_m":           float(p.get("sigma_m",     0) or 0),
                "sigma_v":           float(p.get("sigma_v",     0) or 0),
                "snr":               float(p.get("snr",         0) or 0),
                "snr_th":            float(p.get("snr_th",      snr_th_val) or snr_th_val),
                "confidence":        _rel,
                "accepted":          int(_rel > 0.5),
                "skipped":           0,   # no Kalman gate in live mode — never skip
                "das_amplitude":     float(p.get("das_amplitude",    0) or 0),
                "fiber_distance_m":  float(p.get("fiber_distance_m", 0) or 0),
                "lateral_offset_m":  float(p.get("lateral_offset_m", 0) or 0),
            })
        for p in gps_payloads[vid]:
            audit_rows.append({
                "global_track_id":   vid,
                "vehicle_id_oracle": vid,
                "sensor_type":       "GPS",
                "t":                 float(p.get("t",         0) or 0),
                "sigma_m":           float(p.get("sigma_m",   0) or 0),
                "sigma_v":           float("nan"),
                "snr":               float("nan"),
                "confidence":        float(p.get("confidence", 0) or 0),
                "accepted":          1,
                "skipped":           0,   # GPS always accepted by Kalman in normal operation
                "das_amplitude":     float("nan"),
                "fiber_distance_m":  float("nan"),
                "lateral_offset_m":  float("nan"),
            })
        for p in cam_payloads[vid]:
            _conf = float(p.get("confidence", 0) or 0)
            audit_rows.append({
                "global_track_id":   vid,
                "vehicle_id_oracle": vid,
                "sensor_type":       "Camera",  # title-case to match _features_from_audit
                "t":                 float(p.get("t",       0) or 0),
                "sigma_m":           float(p.get("sigma_m", 0) or 0),
                "sigma_v":           float("nan"),
                "snr":               float("nan"),
                "confidence":        _conf,   # used for cam_confidence_mean, cam_low_conf_frac
                "accepted":          1,       # Kalman gate ≈ always accepts in normal op.
                "skipped":           0,       # no Kalman gate in live mode — never skip
                "das_amplitude":     float("nan"),
                "fiber_distance_m":  float("nan"),
                "lateral_offset_m":  float("nan"),
            })

    audit_df: Optional[pd.DataFrame] = (
        pd.DataFrame(audit_rows) if audit_rows else None
    )

    # ── 5. Extract features per vehicle using the existing helpers ────────────
    rows: list = []
    for track_id, traj in all_track_data:
        row: dict = {
            "scenario_id":       scenario_id,
            "global_track_id":   track_id,
            "vehicle_id_oracle": track_id,
            # Collision flag from world.collision events — not computable
            # from kinematic data alone; must be set here from the event stream.
            "iv_collision_detected": 1.0 if track_id in collision_vids else 0.0,
        }

        row["sc_speed_limit_mps"] = sc_meta.get("sc_speed_limit_mps", float("nan"))

        # ── Kinematic / Kalman / coverage features ────────────────────────────
        row.update(_features_from_trajectory(
            traj, track_id,
            seg_headings=sc_meta.get("sc_seg_headings", {}),
        ))

        # ── DAS / GPS / Camera sensor features ───────────────────────────────
        row.update(_features_from_audit(
            audit_df, track_id,
            fiber_offset_m=fiber_offset_m,
            d0_m=d0_m,
            snr_th=snr_th_val,
        ))

        # ── Sensor coverage / prediction-gap features ─────────────────────────
        row.update(_features_from_coverage(
            traj, audit_df, track_id,
            d0_m=d0_m,
        ))

        # ── Inter-vehicle proximity / TTC / collision features ────────────────
        row.update(_features_inter_vehicle(traj, all_trajs_combined, track_id))

        # _features_from_excel() reads Kalman RMSE sheets from the tracking
        # workbook — not available in live mode, so skip it.  Those xl_* / kf_*
        # columns stay absent (will be imputed to training median during scoring).

        # ── Speed-over-limit (deferred until speed_limit known) ───────────────
        sl = row.get("sc_speed_limit_mps", float("nan"))
        if not math.isnan(sl if sl is not None else float("nan")):
            if "vx_hat" in traj.columns and "vy_hat" in traj.columns:
                speed_ts = np.sqrt(
                    traj["vx_hat"].fillna(0.0) ** 2 +
                    traj["vy_hat"].fillna(0.0) ** 2
                )
                over = speed_ts > sl * 1.05
                row["kin_speed_over_limit_frac"] = float(over.sum()) / max(len(speed_ts), 1)
                excess = (speed_ts[over] - sl).values if over.sum() else np.array([0.0])
                row["kin_speed_excess_max_mps"]  = float(excess.max())
                row["kin_speed_excess_mean_mps"] = float(excess.mean())

        # ── DAS weight estimation (mean only; variability done in _from_audit) ─
        w_est = _estimate_weight(
            das_snr_mean   = row.get("das_snr_mean",   float("nan")),
            snr_th         = snr_th_val,
            fiber_offset_m = fiber_offset_m,
            d0_m           = d0_m,
        )
        row["das_W_est_kg"]    = w_est
        row["das_W_est_class"] = (
            _classify_weight(w_est) if not math.isnan(w_est) else "unknown"
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract anomaly-detection features from SimStudio scenario folders."
    )
    parser.add_argument(
        "folders", nargs="+", metavar="FOLDER",
        help="One or more scenario output folders to process."
    )
    parser.add_argument(
        "--output", "-o", default="anomaly_model/outputs/features.csv",
        help="Output CSV path (default: anomaly_model/outputs/features.csv)"
    )
    parser.add_argument(
        "--catalog", action="store_true",
        help="Print the feature catalog and exit."
    )
    args = parser.parse_args()

    if args.catalog:
        print(f"\n{'Feature':<40} {'Description'}")
        print("-" * 90)
        for feat, desc in FEATURE_CATALOG.items():
            print(f"  {feat:<38} {desc}")
        sys.exit(0)

    print(f"\nSimStudio Feature Extractor — Phase 1")
    print(f"Processing {len(args.folders)} folder(s)...\n")

    df = extract_features(args.folders, verbose=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nDone. {len(df)} rows × {len(df.columns)} features → {out}")
