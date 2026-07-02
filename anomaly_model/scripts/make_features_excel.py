"""
anomaly_model/make_features_excel.py
======================================
Convert a features CSV into a visually styled Excel workbook.

Sheets produced
---------------
  1. Features        – full data table, colour-coded by feature group,
                       frozen panes, auto-filter, alternating row shading.
  2. Summary         – high-level stats (tiles, scenarios, tracks, feature
                       counts, per-group breakdowns, key numeric stats).
  3. Feature Catalog – 6-column human-readable description of every column:
                       Feature | Group | Helps identify | Description |
                       Output range | Collection method

Usage
-----
    # Normal features (default)
    python anomaly_model/make_features_excel.py

    # Anomaly features
    python anomaly_model/make_features_excel.py \
        --input  anomaly_model/outputs/features_anomaly.csv \
        --output anomaly_model/outputs/features_anomaly.xlsx

    # Preview
    python anomaly_model/make_features_excel.py \
        --input  anomaly_model/outputs/features_preview.csv \
        --output anomaly_model/outputs/features_preview.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent
_OUTPUT_DIR = _HERE / "outputs"

DEFAULT_INPUT  = _OUTPUT_DIR / "features_normal.csv"
DEFAULT_OUTPUT = _OUTPUT_DIR / "features_normal.xlsx"

# ---------------------------------------------------------------------------
# Colour palette — one distinct colour per feature group
# ---------------------------------------------------------------------------
#  Each value is (header_bg_hex, header_font_hex, band_bg_hex)

GROUP_PALETTE: Dict[str, Tuple[str, str, str]] = {
    "id":   ("2C3E50", "FFFFFF", "ECF0F1"),   # dark slate   – identifiers
    "lbl":  ("27AE60", "FFFFFF", "EAFAF1"),   # emerald      – label
    "sc_":  ("2980B9", "FFFFFF", "EBF5FB"),   # blue         – scenario meta
    "kin_": ("E67E22", "FFFFFF", "FEF9E7"),   # orange       – kinematic
    "kf_":  ("8E44AD", "FFFFFF", "F5EEF8"),   # purple       – Kalman quality
    "cov_": ("16A085", "FFFFFF", "E8F8F5"),   # teal         – coverage
    "iv_":  ("00897B", "FFFFFF", "E0F2F1"),   # deep teal    – inter-vehicle
    "dis_": ("C0392B", "FFFFFF", "FDEDEC"),   # red          – disagreement
    "das_": ("1A5276", "FFFFFF", "EBF5FB"),   # navy         – DAS quality
    "cam_": ("B7950B", "FFFFFF", "FEFDE7"),   # gold         – camera
    "gps_": ("117A65", "FFFFFF", "E8F8F5"),   # dark teal    – GPS
    "aud_": ("6C3483", "FFFFFF", "F5EEF8"),   # dark violet  – audit
    "xl_":  ("784212", "FFFFFF", "FDF2E9"),   # brown        – Excel-derived
}

GROUP_NAMES: Dict[str, str] = {
    "id":   "Identifiers",
    "lbl":  "Label",
    "sc_":  "Scenario Metadata (sc_)",
    "kin_": "Kinematic (kin_)",
    "kf_":  "Kalman Quality (kf_)",
    "cov_": "Coverage (cov_)",
    "iv_":  "Inter-Vehicle (iv_)",
    "dis_": "Sensor Disagreement (dis_)",
    "das_": "DAS Sensor Quality (das_)",
    "cam_": "Camera Sensor Quality (cam_)",
    "gps_": "GPS Sensor Quality (gps_)",
    "aud_": "Audit Cross-Sensor (aud_)",
    "xl_":  "Excel-Derived (xl_)",
}

# ---------------------------------------------------------------------------
# Rich feature catalog
# Each entry: feature_name → {helps_identify, description, output_range,
#                              collection_method}
# FEATURE_CATALOG (simple str dict) is derived from this for backward compat.
# ---------------------------------------------------------------------------

# Shorthand anomaly labels used in "helps_identify"
_S   = "Speeding"
_HB  = "Hard braking"
_WW  = "Wrong weight"
_TG  = "Tailgating"
_LW  = "Lane weaving"
_LS  = "Lane straddling"
_SD  = "Sensor dropout"
_SV  = "Stalled vehicle"
_GT  = "Ghost / stale track"
_CA  = "Collision / accident"
_GEN = "General / multiple"
_ID  = "—"           # identifier / metadata

# Shorthand collection methods
_TRJ  = "Trajectory CSV (Kalman estimates)"
_AUD  = "Kalman measurement audit CSV"
_DAS  = "Audit CSV — DAS rows (lateral_offset_m, SNR, amplitude)"
_CAM  = "Audit CSV — camera rows"
_GPS  = "Audit CSV — GPS rows"
_JSON = "Scenario JSON (vehicle / sensor config)"
_XLS  = "SimStudio export Excel (RMSE / Issues sheets)"
_IV   = "Cross-track Euclidean distance at matched 50 Hz ticks"
_META = "Extraction pipeline (derived / injected)"

FEATURE_CATALOG_RICH: Dict[str, Dict[str, str]] = {
    # ── Identifiers ──────────────────────────────────────────────────────────
    "scenario_id":              {"helps_identify": _ID,  "description": "Folder name or user-supplied scenario label",                      "output_range": "String",         "collection_method": _META},
    "global_track_id":          {"helps_identify": _ID,  "description": "Track ID (e.g. T000001)",                                          "output_range": "String",         "collection_method": _TRJ},
    "vehicle_id_oracle":        {"helps_identify": _ID,  "description": "Ground-truth vehicle identifier",                                  "output_range": "String",         "collection_method": _TRJ},
    "label":                    {"helps_identify": _ID,  "description": "Ground-truth label (normal / anomaly)",                            "output_range": "normal/anomaly", "collection_method": _META},

    # ── Scenario metadata ─────────────────────────────────────────────────────
    "sc_speed_limit_mps":       {"helps_identify": _S,   "description": "Min speed limit across segments [m/s]",                          "output_range": "≥ 0 m/s",        "collection_method": _JSON},

    # ── Kinematic ─────────────────────────────────────────────────────────────
    "kin_speed_mean_mps":       {"helps_identify": _S,   "description": "Mean estimated speed [m/s]",                                      "output_range": "0–40 m/s",       "collection_method": _TRJ},
    "kin_speed_max_mps":        {"helps_identify": _S,   "description": "Max estimated speed [m/s]",                                       "output_range": "0–50+ m/s",      "collection_method": _TRJ},
    "kin_speed_std_mps":        {"helps_identify": _S,   "description": "Std dev of estimated speed [m/s]",                                "output_range": "0–15 m/s",       "collection_method": _TRJ},
    "kin_speed_p90_mps":        {"helps_identify": _S,   "description": "90th-percentile estimated speed [m/s]; robust peak speed proxy",  "output_range": "0–50+ m/s",      "collection_method": _TRJ},
    "kin_speed_over_limit_frac":{"helps_identify": _S,   "description": "Fraction of timesteps where speed > speed_limit × 1.05",          "output_range": "0–1",            "collection_method": _TRJ},
    "kin_speed_excess_max_mps": {"helps_identify": _S,   "description": "Max speed excess above speed limit on over-limit timesteps [m/s]", "output_range": "0–30+ m/s",      "collection_method": _TRJ},
    "kin_speed_excess_mean_mps":{"helps_identify": _S,   "description": "Mean speed excess above speed limit on over-limit timesteps [m/s]","output_range": "0–20 m/s",       "collection_method": _TRJ},
    "kin_accel_mean_abs_mps2":  {"helps_identify": f"{_HB}, {_CA}", "description": "Mean absolute acceleration magnitude [m/s²]",                "output_range": "0–5 m/s²",       "collection_method": _TRJ},
    "kin_accel_max_abs_mps2":   {"helps_identify": f"{_HB}, {_CA}", "description": "Max absolute acceleration magnitude [m/s²]",               "output_range": "0–10+ m/s²",     "collection_method": _TRJ},
    "kin_accel_std_mps2":       {"helps_identify": f"{_HB}, {_CA}", "description": "Std dev of acceleration magnitude [m/s²]",                 "output_range": "0–5 m/s²",       "collection_method": _TRJ},
    "kin_phys_impossible_v":    {"helps_identify": _S,   "description": "Row count where estimated speed > 40 m/s (physically suspect)",    "output_range": "0–n",            "collection_method": _TRJ},
    "kin_phys_impossible_a":    {"helps_identify": f"{_HB}, {_CA}", "description": "Row count where estimated |accel| > 8 m/s² (physically suspect)", "output_range": "0–n",   "collection_method": _TRJ},
    "kin_lateral_dev_mean_m":   {"helps_identify": f"{_LW}, {_LS}", "description": "Mean lateral deviation of y_hat from its track mean [m]",       "output_range": "0–5 m",          "collection_method": _TRJ},
    "kin_lateral_dev_max_m":    {"helps_identify": f"{_LW}, {_LS}", "description": "Max lateral deviation of y_hat from track mean [m]",            "output_range": "0–10 m",         "collection_method": _TRJ},
    "kin_lateral_dev_std_m":    {"helps_identify": f"{_LW}, {_LS}", "description": "Std dev of lateral deviation [m]",                             "output_range": "0–5 m",          "collection_method": _TRJ},
    "kin_lateral_oscillation_ratio": {"helps_identify": _LW, "description": "std(lat_dev) / mean(lat_dev); high value indicates lane weaving",   "output_range": "0–5+ (ratio)",   "collection_method": _TRJ},
    "kin_stopped_frac":         {"helps_identify": _SV,  "description": "Fraction of timesteps where estimated speed < 0.5 m/s (stopped)",  "output_range": "0–1",            "collection_method": _TRJ},
    "kin_speed_min_mps":        {"helps_identify": _SV,  "description": "Minimum estimated speed [m/s]; near 0 = vehicle stopped or stalled", "output_range": "0–40 m/s",       "collection_method": _TRJ},
    "kin_speed_cv":             {"helps_identify": _S,   "description": "Coefficient of variation of speed (std/mean); high = erratic speed profile", "output_range": "0–∞ (ratio)", "collection_method": _TRJ},
    "kin_stop_event_count":     {"helps_identify": _SV,  "description": "Number of distinct stop events (transitions into speed < 0.5 m/s)",  "output_range": "0–n",            "collection_method": _TRJ},
    "kin_max_stopped_steps":    {"helps_identify": _SV,  "description": "Maximum consecutive timesteps where speed < 0.5 m/s",                "output_range": "0–n",            "collection_method": _TRJ},
    "kin_max_stopped_duration_s":{"helps_identify": _SV, "description": "Longest continuous stop duration [s]",                               "output_range": "0–∞ s",          "collection_method": _TRJ},
    "kin_decel_max_mps2":       {"helps_identify": f"{_HB}, {_CA}", "description": "Maximum deceleration (most negative accel) [m/s²]",         "output_range": "0–10+ m/s²",     "collection_method": _TRJ},
    "kin_high_decel_frac":      {"helps_identify": f"{_HB}, {_CA}", "description": "Fraction of timesteps where deceleration > 3 m/s²",        "output_range": "0–1",            "collection_method": _TRJ},
    "kin_decel_event_count":    {"helps_identify": f"{_HB}, {_CA}", "description": "Number of distinct hard-braking events (decel > 3 m/s²)",  "output_range": "0–n",            "collection_method": _TRJ},
    "kin_jerk_max_mps3":        {"helps_identify": f"{_HB}, {_CA}", "description": "Maximum jerk (rate of change of acceleration) [m/s³]",     "output_range": "0–20+ m/s³",     "collection_method": _TRJ},
    "kin_jerk_mean_abs_mps3":   {"helps_identify": f"{_HB}, {_CA}", "description": "Mean absolute jerk [m/s³]; high = aggressive/erratic driving", "output_range": "0–10 m/s³", "collection_method": _TRJ},
    "kin_lateral_peak_to_peak_m":{"helps_identify": _LW, "description": "Peak-to-peak lateral range of lane-relative offset [m]; high = wide weaving", "output_range": "0–6+ m", "collection_method": _DAS},
    "kin_lateral_speed_max_mps": {"helps_identify": f"{_LW}, {_LS}", "description": "Max rate of lateral position change |Δy_hat / Δt| [m/s]; captures rapid lane-change events", "output_range": "0–5+ m/s", "collection_method": _TRJ},
    "kin_lateral_vel_mean_mps":  {"helps_identify": f"{_LW}, {_LS}", "description": "Mean |vy_hat| [m/s] — average lateral velocity component from Kalman filter; sustained high value = persistent sideways motion", "output_range": "0–5+ m/s", "collection_method": _TRJ},
    "kin_lateral_vel_max_mps":   {"helps_identify": f"{_LW}, {_LS}", "description": "Max |vy_hat| [m/s] — peak lateral speed from Kalman filter; high = sudden sideways lurch (weaving onset, lane-change, collision side-swipe)", "output_range": "0–10+ m/s", "collection_method": _TRJ},
    "kin_lateral_accel_max_mps2":{"helps_identify": f"{_LW}, {_LS}, {_CA}", "description": "Max |ay_hat| [m/s²] — peak lateral acceleration from Kalman filter; large value = aggressive steering / weaving manoeuvre / side-impact", "output_range": "0–10+ m/s²", "collection_method": _TRJ},
    "kin_heading_change_rate_max_rad_per_s": {"helps_identify": f"{_LW}, {_LS}", "description": "Max |Δheading / Δt| [rad/s] while speed > 0.5 m/s; high = abrupt swerve, tight curve, or lane-change manoeuvre", "output_range": "0–5+ rad/s", "collection_method": _TRJ},
    "kin_speed_jump_max_mps":    {"helps_identify": f"{_HB}, {_GEN}", "description": "Max |Δspeed| between consecutive 50 Hz ticks [m/s]; sudden jump exceeding physical limits = ghost track, sensor outlier, or emergency stop", "output_range": "0–20+ m/s", "collection_method": _TRJ},
    "kin_road_heading_diff_mean_rad": {"helps_identify": f"{_LW}, {_LS}", "description": "Mean absolute angular difference between vehicle heading (from vy/vx) and road segment design heading [rad]; high = crossing lane or wrong-way driving; requires scenario JSON points array", "output_range": "0–π rad", "collection_method": f"{_TRJ} + {_JSON}"},

    # ── Kalman quality ────────────────────────────────────────────────────────
    "kf_pos_err_mean_m":        {"helps_identify": _GEN, "description": "Mean position error vs ground truth [m]",                         "output_range": "0–20 m",         "collection_method": _TRJ},
    "kf_pos_err_max_m":         {"helps_identify": _GEN, "description": "Max position error [m]",                                          "output_range": "0–50+ m",        "collection_method": _TRJ},
    "kf_pos_err_std_m":         {"helps_identify": _GEN, "description": "Std dev of position error [m]",                                   "output_range": "0–15 m",         "collection_method": _TRJ},
    "kf_pos_err_p90_m":         {"helps_identify": _GEN, "description": "90th percentile position error [m]",                              "output_range": "0–30 m",         "collection_method": _TRJ},
    "kf_track_rmse_m":          {"helps_identify": _GEN, "description": "Overall track RMSE [m]",                                          "output_range": "0–20 m",         "collection_method": _TRJ},
    "kf_rolling_rmse_spike_count": {"helps_identify": _GEN, "description": "Count of 10-step windows where rolling RMSE > 2× track RMSE", "output_range": "0–n",            "collection_method": _TRJ},
    "kf_sigma_pos_mean_m":      {"helps_identify": _SD,  "description": "Mean Kalman position uncertainty σ [m]",                         "output_range": "0–10 m",         "collection_method": _TRJ},
    "kf_sigma_pos_max_m":       {"helps_identify": _SD,  "description": "Max Kalman position uncertainty σ [m]",                          "output_range": "0–20+ m",        "collection_method": _TRJ},
    "kf_consistency_ratio_mean":{"helps_identify": _GEN, "description": "Mean(pos_err / sigma_pos); > 2 = overconfident filter",          "output_range": "0–5+ (ratio)",   "collection_method": _TRJ},
    "kf_consistency_ratio_max": {"helps_identify": _GEN, "description": "Max(pos_err / sigma_pos)",                                       "output_range": "0–10+ (ratio)",  "collection_method": _TRJ},
    "kf_overconfident_frac":    {"helps_identify": _GEN, "description": "Fraction of rows where pos_err > 2 × sigma_pos",                 "output_range": "0–1",            "collection_method": _TRJ},
    "kf_sigma_growth_rate_during_gaps_m_per_s": {"helps_identify": _SD, "description": "Mean rate of sigma_pos increase [m/s] during prediction-only (no-measurement) periods; abnormally fast = filter diverging", "output_range": "0–∞ m/s", "collection_method": _TRJ},
    "kf_post_dropout_jump_m":    {"helps_identify": f"{_SD}, {_GEN}", "description": "Mean 2-D position jump |Δx_hat| at every transition from a prediction_only run back to a measurement row [m]; large = filter drifted during the gap (long dropout) or ghost track that reappeared at a different position", "output_range": "0–50+ m", "collection_method": _TRJ},

    # ── Coverage ──────────────────────────────────────────────────────────────
    "cov_total_rows":               {"helps_identify": _GEN, "description": "Total trajectory rows (measurement + prediction)",              "output_range": "≥ 1 (integer)",  "collection_method": _TRJ},
    "cov_pred_only_count":          {"helps_identify": _SD,  "description": "Number of prediction-only (no measurement) rows",              "output_range": "0–n",            "collection_method": _TRJ},
    "cov_pred_only_frac":           {"helps_identify": _SD,  "description": "Fraction of rows that are prediction-only",                    "output_range": "0–1",            "collection_method": _TRJ},
    "cov_max_consec_pred":          {"helps_identify": _SD,  "description": "Max consecutive prediction-only rows (longest dropout gap)",   "output_range": "0–n",            "collection_method": _TRJ},
    "cov_n_dropout_events":         {"helps_identify": _SD,  "description": "Number of distinct measurement dropout events",                "output_range": "0–n",            "collection_method": _TRJ},
    "cov_duration_s":               {"helps_identify": _GEN, "description": "Track duration [s]",                                           "output_range": "0–∞ s",          "collection_method": _TRJ},
    "cov_das_active_frac":          {"helps_identify": _SD,  "description": "Fraction of track rows where DAS has a measurement",           "output_range": "0–1",            "collection_method": _TRJ},
    "cov_cam_active_frac":          {"helps_identify": _SD,  "description": "Fraction of track rows where Camera has a measurement",        "output_range": "0–1",            "collection_method": _TRJ},
    "cov_gps_active_frac":          {"helps_identify": _SD,  "description": "Fraction of track rows where GPS has a measurement",           "output_range": "0–1",            "collection_method": _TRJ},
    "cov_multi_sensor_frac":        {"helps_identify": _SD,  "description": "Fraction of rows where ≥ 2 sensors are active simultaneously", "output_range": "0–1",            "collection_method": _TRJ},
    "cov_single_sensor_frac":       {"helps_identify": _SD,  "description": "Fraction of rows where exactly 1 sensor is active",            "output_range": "0–1",            "collection_method": _TRJ},
    "cov_no_sensor_frac":           {"helps_identify": _SD,  "description": "Fraction of rows where no sensor provides a measurement",      "output_range": "0–1",            "collection_method": _TRJ},
    "cov_das_cam_overlap_frac":     {"helps_identify": _SD,  "description": "Fraction of rows where both DAS and Camera are active",         "output_range": "0–1",            "collection_method": _TRJ},
    "cov_das_gps_overlap_frac":     {"helps_identify": _SD,  "description": "Fraction of rows where both DAS and GPS are active",            "output_range": "0–1",            "collection_method": _TRJ},
    "cov_cam_gps_overlap_frac":     {"helps_identify": _SD,  "description": "Fraction of rows where both Camera and GPS are active",         "output_range": "0–1",            "collection_method": _TRJ},
    "cov_all_sensors_frac":         {"helps_identify": _SD,  "description": "Fraction of rows where all three sensors are active",           "output_range": "0–1",            "collection_method": _TRJ},
    "cov_dominant_sensor":          {"helps_identify": _SD,  "description": "Sensor active for the largest fraction of rows",               "output_range": "das/cam/gps",    "collection_method": _TRJ},
    "cov_err_das_only_m":           {"helps_identify": _SD,  "description": "Mean position error on rows where DAS is the sole sensor [m]", "output_range": "0–20 m",         "collection_method": _TRJ},
    "cov_err_cam_only_m":           {"helps_identify": _SD,  "description": "Mean position error on rows where Camera is the sole sensor [m]","output_range": "0–20 m",        "collection_method": _TRJ},
    "cov_err_gps_only_m":           {"helps_identify": _SD,  "description": "Mean position error on rows where GPS is the sole sensor [m]", "output_range": "0–20 m",         "collection_method": _TRJ},
    "cov_err_multi_sensor_m":       {"helps_identify": _SD,  "description": "Mean position error on rows where ≥ 2 sensors are active [m]", "output_range": "0–10 m",         "collection_method": _TRJ},
    "cov_err_no_sensor_m":          {"helps_identify": _GEN, "description": "Mean position error on prediction-only rows [m]",              "output_range": "0–30+ m",        "collection_method": _TRJ},
    "cov_fusion_benefit_vs_best_single": {"helps_identify": _SD, "description": "best_single_err − multi_sensor_err [m]; positive = fusion helps", "output_range": "-∞ to +∞ m", "collection_method": _TRJ},
    "cov_err_cam_no_das_m":         {"helps_identify": _SD,  "description": "Mean position error when Camera active but DAS absent [m]",    "output_range": "0–20 m",         "collection_method": _TRJ},
    "cov_err_cam_with_das_m":       {"helps_identify": _SD,  "description": "Mean position error when Camera and DAS both active [m]",      "output_range": "0–10 m",         "collection_method": _TRJ},
    "cov_das_fusion_benefit_m":     {"helps_identify": _SD,  "description": "err_cam_no_das − err_cam_with_das [m]",                        "output_range": "-∞ to +∞ m",     "collection_method": _TRJ},
    "cov_gap_frac_early":           {"helps_identify": _SD,  "description": "Fraction of the first third of the track that is prediction-only",  "output_range": "0–1",       "collection_method": _TRJ},
    "cov_gap_frac_middle":          {"helps_identify": _SD,  "description": "Fraction of the middle third of the track that is prediction-only", "output_range": "0–1",        "collection_method": _TRJ},
    "cov_gap_frac_late":            {"helps_identify": _SD,  "description": "Fraction of the last third of the track that is prediction-only",   "output_range": "0–1",        "collection_method": _TRJ},
    "cov_gap_clustering_cv":        {"helps_identify": _SD,  "description": "Coefficient of variation of gap row indices",                  "output_range": "0–∞",            "collection_method": _TRJ},
    "cov_gap_spatial_pos_mean_norm":{"helps_identify": _SD,  "description": "Normalised mean spatial position of gaps (0=start, 1=end)",    "output_range": "0–1",            "collection_method": _TRJ},
    "cov_gap_spatial_spread_norm":  {"helps_identify": _SD,  "description": "Normalised std dev of gap spatial positions",                  "output_range": "0–1",            "collection_method": _TRJ},
    "cov_high_speed_gap_frac":      {"helps_identify": _S,   "description": "Fraction of prediction-only rows where speed > 50% of v_max",  "output_range": "0–1",            "collection_method": _TRJ},
    "cov_mean_speed_at_gaps_mps":   {"helps_identify": _S,   "description": "Mean estimated speed during prediction-only rows [m/s]",       "output_range": "0–40+ m/s",      "collection_method": _TRJ},
    "cov_das_hz_est":               {"helps_identify": _SD,  "description": "Estimated DAS update rate [Hz]",                               "output_range": "0–50 Hz",        "collection_method": _AUD},
    "cov_cam_hz_est":               {"helps_identify": _SD,  "description": "Estimated Camera update rate [Hz]",                            "output_range": "0–50 Hz",        "collection_method": _AUD},
    "cov_gps_hz_est":               {"helps_identify": _SD,  "description": "Estimated GPS update rate [Hz]",                               "output_range": "0–50 Hz",        "collection_method": _AUD},
    "cov_das_update_share":         {"helps_identify": _SD,  "description": "Fraction of accepted Kalman updates from DAS",                 "output_range": "0–1",            "collection_method": _AUD},
    "cov_cam_update_share":         {"helps_identify": _SD,  "description": "Fraction of accepted Kalman updates from Camera",              "output_range": "0–1",            "collection_method": _AUD},
    "cov_gps_update_share":         {"helps_identify": _SD,  "description": "Fraction of accepted Kalman updates from GPS",                 "output_range": "0–1",            "collection_method": _AUD},
    "cov_dominant_kalman_updater":  {"helps_identify": _SD,  "description": "Sensor that contributed the most accepted Kalman updates",     "output_range": "das/cam/gps",    "collection_method": _AUD},
    "cov_das_amplitude_mean":       {"helps_identify": _WW,  "description": "Mean DAS raw amplitude A across accepted DAS rows",            "output_range": "0–∞ (raw units)","collection_method": _DAS},
    "cov_das_amplitude_std":        {"helps_identify": _WW,  "description": "Std dev of DAS amplitude",                                    "output_range": "0–∞",            "collection_method": _DAS},
    "cov_das_amplitude_cv":         {"helps_identify": _WW,  "description": "Coefficient of variation of DAS amplitude (std/mean)",         "output_range": "0–∞ (ratio)",    "collection_method": _DAS},
    "cov_das_fiber_dist_mean_m":    {"helps_identify": f"{_LW}, {_LS}", "description": "Mean perpendicular vehicle-to-fiber distance r [m]",             "output_range": "0–10+ m",        "collection_method": _DAS},
    "cov_das_fiber_dist_std_m":     {"helps_identify": f"{_LW}, {_LS}", "description": "Std dev of fiber distance [m]",                                   "output_range": "0–5 m",          "collection_method": _DAS},
    "cov_das_fiber_dist_max_m":     {"helps_identify": f"{_LW}, {_LS}", "description": "Maximum fiber distance observed [m]",                             "output_range": "0–10+ m",        "collection_method": _DAS},
    "cov_das_fiber_dist_cv":        {"helps_identify": f"{_LW}, {_LS}", "description": "Coefficient of variation of fiber distance (std/mean)",            "output_range": "0–∞ (ratio)",    "collection_method": _DAS},
    "cov_das_lateral_offset_mean_m":{"helps_identify": f"{_LW}, {_LS}", "description": "Mean signed vehicle lateral offset from lane centreline [m]",     "output_range": "−2 to +2 m",     "collection_method": _DAS},
    "cov_das_lateral_offset_std_m": {"helps_identify": f"{_LW}, {_LS}", "description": "Std dev of lane-relative lateral offset [m]",                     "output_range": "0–2 m",          "collection_method": _DAS},
    "cov_das_lateral_offset_max_m": {"helps_identify": f"{_LW}, {_LS}", "description": "Maximum absolute lateral offset from lane centreline [m]",        "output_range": "0–5+ m",         "collection_method": _DAS},
    # New DAS lateral features
    "cov_das_lateral_offset_abs_mean_m": {"helps_identify": f"{_LW}, {_LS}", "description": "Mean |lateral_offset_m| — average unsigned distance from lane centre [m]", "output_range": "0–3 m",     "collection_method": _DAS},
    "cov_das_lateral_offset_range_m":    {"helps_identify": _LW,  "description": "max(lateral_offset_m) − min(lateral_offset_m); total lateral excursion [m]",          "output_range": "0–6+ m",    "collection_method": _DAS},
    "cov_das_lateral_frac_outside_half_m":{"helps_identify": _LS, "description": "Fraction of DAS rows where |lateral_offset_m| > 0.5 m",                               "output_range": "0–1",       "collection_method": _DAS},
    "cov_das_lateral_frac_outside_1m":   {"helps_identify": _LS,  "description": "Fraction of DAS rows where |lateral_offset_m| > 1.0 m (strong straddling signal)",    "output_range": "0–1",       "collection_method": _DAS},
    "cov_das_lateral_sign_changes":      {"helps_identify": _LW,  "description": "Noise-robust sign-change count on smoothed lateral_offset_m; high = lane weaving",     "output_range": "0–n",       "collection_method": _DAS},
    "cov_das_lateral_consistent_side_frac": {"helps_identify": _LS, "description": "Fraction of DAS rows where |lateral_offset_m| > 0.3 m and the sign is consistent; high = persistent lane straddling", "output_range": "0–1", "collection_method": _DAS},
    "cov_das_lateral_change_rate_max":      {"helps_identify": _LW, "description": "Max |Δlateral_offset_m / Δt| across consecutive DAS rows [m/s]; high = rapid lane-crossing event",                   "output_range": "0–5+ m/s", "collection_method": _DAS},

    # ── Inter-vehicle ─────────────────────────────────────────────────────────
    "iv_n_other_vehicles":         {"helps_identify": _TG,           "description": "Number of other concurrent tracks present during this track's lifespan",                      "output_range": "0–n",        "collection_method": _IV},
    "iv_min_dist_m":               {"helps_identify": f"{_TG}, {_CA}", "description": "Minimum 2-D Euclidean distance to any other track at any shared timestamp [m]",           "output_range": "0–200+ m",   "collection_method": _IV},
    "iv_mean_min_dist_m":          {"helps_identify": f"{_TG}, {_CA}", "description": "Mean of per-tick minimum distance to nearest other track [m]",                            "output_range": "0–200+ m",   "collection_method": _IV},
    "iv_p10_min_dist_m":           {"helps_identify": _TG,           "description": "10th percentile of per-tick minimum distance; captures sustained close following",           "output_range": "0–100 m",    "collection_method": _IV},
    "iv_close_proximity_frac":     {"helps_identify": _TG,           "description": "Fraction of shared ticks where nearest vehicle is < 10 m",                                  "output_range": "0–1",        "collection_method": _IV},
    "iv_tailgate_proximity_frac":  {"helps_identify": _TG,           "description": "Fraction of shared ticks where nearest vehicle is < 5 m (tailgating threshold)",            "output_range": "0–1",        "collection_method": _IV},
    "iv_speed_proximity_ratio_max":{"helps_identify": f"{_TG}, {_CA}","description": "Max(own_speed / max(dist_to_nearest, 0.5)) — high when fast and very close",              "output_range": "0–∞ (ratio)","collection_method": _IV},
    "iv_overlap_duration_s":      {"helps_identify": _CA,            "description": "Total seconds where nearest vehicle is within 3 m (vehicle-width overlap threshold)",                                                        "output_range": "0–∞ s",       "collection_method": _IV},
    "iv_overlap_frac":            {"helps_identify": _CA,            "description": "Fraction of shared ticks where nearest vehicle is within 3 m; sustained = collision",                                                                 "output_range": "0–1",         "collection_method": _IV},
    "iv_ttc_min_s":               {"helps_identify": f"{_TG}, {_CA}","description": "Minimum Time-to-Collision across the track [s]. TTC = distance / closing_speed; < 2 s is universally the danger threshold",                        "output_range": "0–∞ s",       "collection_method": _IV},
    "iv_ttc_below_2s_frac":       {"helps_identify": _TG,            "description": "Fraction of shared ticks where TTC < 2 s; persistent sub-2-s TTC = chronic tailgating",                                                             "output_range": "0–1",         "collection_method": _IV},
    "iv_closing_speed_max_mps":   {"helps_identify": f"{_TG}, {_CA}","description": "Maximum closing speed to any other vehicle at any tick [m/s]; kinetic energy proxy for an imminent impact",                                          "output_range": "0–30+ m/s",   "collection_method": _IV},
    "iv_time_headway_mean_s":     {"helps_identify": _TG,            "description": "Mean time headway = dist / own_speed [s]; < 1.5 s is considered dangerous regardless of absolute distance",                                          "output_range": "0–∞ s",       "collection_method": _IV},
    "iv_speed_ratio_to_others":   {"helps_identify": f"{_S}, {_SV}", "description": "Own mean speed / mean speed of concurrent other vehicles; > 1 = faster, < 1 = slower",                                                              "output_range": "0–∞ (ratio)", "collection_method": _IV},
    "iv_speed_excess_over_others_mps": {"helps_identify": _S,        "description": "Own mean speed − mean speed of others [m/s]; positive = this vehicle is faster than traffic",                                                        "output_range": "-∞ to +∞ m/s","collection_method": _IV},
    "iv_speed_pearson_r_traffic": {"helps_identify": f"{_SV}, {_S}", "description": "Pearson r between own per-tick speed and mean speed of all other vehicles; low = vehicle decoupled from traffic flow",                               "output_range": "-1 to +1",    "collection_method": _IV},
    "iv_decel_at_min_dist_mps2":  {"helps_identify": f"{_CA}, {_HB}","description": "Longitudinal deceleration of this vehicle at the tick of minimum inter-vehicle distance [m/s²]; collision: high decel coincides with closest approach", "output_range": "-∞ to +∞ m/s²","collection_method": _IV},
    "iv_others_mean_speed_when_stopped": {"helps_identify": _SV,     "description": "Mean speed of all other vehicles during ticks when this vehicle is stopped [m/s]; high = anomalous solo stop while traffic flows",                    "output_range": "0–40+ m/s",   "collection_method": _IV},
    "iv_rel_speed_at_min_dist_mps": {"helps_identify": f"{_CA}, {_TG}", "description": "Magnitude of the relative velocity vector |Δv| at the tick of minimum inter-vehicle distance [m/s]; high relative speed at the closest point = high kinetic-energy collision signature", "output_range": "0–30+ m/s", "collection_method": _IV},
    "iv_collision_risk_proxy":   {"helps_identify": _CA,            "description": "max(|Δv|² / max(dist, 0.5)) across all shared ticks — dimensionally ≈ twice the specific kinetic energy of relative motion scaled by proximity; peaks sharply before a collision; stays bounded even at dist → 0 due to 0.5 m floor", "output_range": "0–∞ (m/s²/m)", "collection_method": _IV},

    # ── Sensor disagreement ───────────────────────────────────────────────────
    "dis_das_cam_mean_m":       {"helps_identify": _SD,  "description": "Mean |DAS_x − Camera_x| when both sensors present [m]",           "output_range": "0–20 m",         "collection_method": _TRJ},
    "dis_das_cam_max_m":        {"helps_identify": _SD,  "description": "Max |DAS_x − Camera_x| [m]",                                      "output_range": "0–50+ m",        "collection_method": _TRJ},
    "dis_das_hat_mean_m":       {"helps_identify": _SD,  "description": "Mean |DAS_x − x_hat| [m]",                                        "output_range": "0–20 m",         "collection_method": _TRJ},
    "dis_das_hat_max_m":        {"helps_identify": _SD,  "description": "Max |DAS_x − x_hat| [m]",                                         "output_range": "0–50+ m",        "collection_method": _TRJ},
    "dis_cam_hat_mean_m":       {"helps_identify": _SD,  "description": "Mean |Camera_x − x_hat| [m]",                                     "output_range": "0–20 m",         "collection_method": _TRJ},
    "dis_cam_hat_max_m":        {"helps_identify": _SD,  "description": "Max |Camera_x − x_hat| [m]",                                      "output_range": "0–50+ m",        "collection_method": _TRJ},
    "dis_gps_hat_mean_m":       {"helps_identify": _SD,  "description": "Mean |GPS_x − x_hat| [m]",                                        "output_range": "0–20 m",         "collection_method": _TRJ},
    "dis_gps_hat_max_m":        {"helps_identify": _SD,  "description": "Max |GPS_x − x_hat| [m]",                                         "output_range": "0–50+ m",        "collection_method": _TRJ},
    "dis_das_only_err_mean_m":  {"helps_identify": _SD,  "description": "Mean pos_err on DAS-only rows [m]",                               "output_range": "0–20 m",         "collection_method": _TRJ},
    "dis_cam_only_err_mean_m":  {"helps_identify": _SD,  "description": "Mean pos_err on camera-only rows [m]",                            "output_range": "0–20 m",         "collection_method": _TRJ},
    "dis_das_worse_than_cam":   {"helps_identify": _SD,  "description": "1 if DAS-only position error exceeds camera-only error",          "output_range": "0 or 1",         "collection_method": _TRJ},

    # ── DAS quality ───────────────────────────────────────────────────────────
    "das_n_measurements":       {"helps_identify": _SD,  "description": "Total DAS measurements for this track",                           "output_range": "0–n",            "collection_method": _DAS},
    "das_snr_mean":             {"helps_identify": _WW,  "description": "Mean DAS SNR across accepted DAS rows",                           "output_range": "0–30+ (ratio)",  "collection_method": _DAS},
    "das_snr_min":              {"helps_identify": _WW,  "description": "Min DAS SNR",                                                     "output_range": "0–30+",          "collection_method": _DAS},
    "das_snr_max":              {"helps_identify": _WW,  "description": "Max DAS SNR",                                                     "output_range": "0–30+",          "collection_method": _DAS},
    "das_snr_std":              {"helps_identify": _WW,  "description": "Std dev of DAS SNR",                                              "output_range": "0–10+",          "collection_method": _DAS},
    "das_snr_low_frac":         {"helps_identify": _WW,  "description": "Fraction of DAS rows with SNR < 5 (low quality threshold)",      "output_range": "0–1",            "collection_method": _DAS},
    "das_sigma_mean_m":         {"helps_identify": _SD,  "description": "Mean DAS position uncertainty σ [m]",                            "output_range": "0–5 m",          "collection_method": _DAS},
    "das_sigma_max_m":          {"helps_identify": _SD,  "description": "Max DAS position uncertainty σ [m]",                             "output_range": "0–10+ m",        "collection_method": _DAS},
    "das_confidence_mean":      {"helps_identify": _WW,  "description": "Mean DAS reliability (SNR-derived confidence proxy)",             "output_range": "0–1",            "collection_method": _DAS},
    "das_sigma_v_mean_mps":     {"helps_identify": _SD,  "description": "Mean DAS velocity uncertainty σ_v [m/s]",                        "output_range": "0–5 m/s",        "collection_method": _DAS},
    "das_accept_frac":          {"helps_identify": _SD,  "description": "Fraction of DAS measurements accepted by Kalman filter",         "output_range": "0–1",            "collection_method": _DAS},
    "das_skip_frac":            {"helps_identify": _SD,  "description": "Fraction of DAS measurements skipped by Kalman filter",          "output_range": "0–1",            "collection_method": _DAS},
    "das_amplitude_vs_dist_pearson_r": {"helps_identify": f"{_WW}, {_SD}", "description": "Pearson r between DAS amplitude A and 1/(fiber_dist + d0)²; physics model predicts A ∝ W/(r+d0)²; low |r| = signal does not follow inverse-square law", "output_range": "-1 to +1", "collection_method": _DAS},
    "das_W_est_kg":             {"helps_identify": _WW,  "description": "Estimated vehicle weight [kg] back-computed from mean DAS SNR",             "output_range": "0–20 000 kg",    "collection_method": _DAS},
    "das_W_est_std_kg":         {"helps_identify": _WW,  "description": "Std dev of per-measurement W_est series [kg]; high = lateral movement or sensor inconsistency", "output_range": "0–∞ kg",   "collection_method": _DAS},
    "das_W_est_cv":             {"helps_identify": _WW,  "description": "Coefficient of variation of W_est (std/mean); normalised instability of weight signal",         "output_range": "0–∞ (ratio)", "collection_method": _DAS},
    "das_W_est_class":          {"helps_identify": _WW,  "description": "Weight class of W_est (pedestrian/motorcycle/car/van_truck/bus_hgv)",                          "output_range": "String",      "collection_method": _DAS},

    # ── Camera quality ────────────────────────────────────────────────────────
    "cam_n_measurements":       {"helps_identify": _SD,  "description": "Total camera measurements for this track",                        "output_range": "0–n",            "collection_method": _CAM},
    "cam_confidence_mean":      {"helps_identify": _SD,  "description": "Mean camera detection confidence",                                "output_range": "0–1",            "collection_method": _CAM},
    "cam_confidence_min":       {"helps_identify": _SD,  "description": "Min camera detection confidence",                                 "output_range": "0–1",            "collection_method": _CAM},
    "cam_confidence_std":       {"helps_identify": _SD,  "description": "Std dev of camera confidence",                                    "output_range": "0–0.5",          "collection_method": _CAM},
    "cam_low_conf_frac":        {"helps_identify": _SD,  "description": "Fraction of camera rows with confidence < 0.30",                 "output_range": "0–1",            "collection_method": _CAM},
    "cam_sigma_mean_m":         {"helps_identify": _SD,  "description": "Mean camera position uncertainty σ [m]",                         "output_range": "0–5 m",          "collection_method": _CAM},
    "cam_accept_frac":          {"helps_identify": _SD,  "description": "Fraction of camera measurements accepted by Kalman filter",      "output_range": "0–1",            "collection_method": _CAM},

    # ── GPS quality ───────────────────────────────────────────────────────────
    "gps_n_measurements":       {"helps_identify": _SD,  "description": "Total GPS measurements for this track",                           "output_range": "0–n",            "collection_method": _GPS},
    "gps_sigma_mean_m":         {"helps_identify": _SD,  "description": "Mean GPS position uncertainty σ [m]",                            "output_range": "0–10 m",         "collection_method": _GPS},

    # ── Audit cross-sensor ────────────────────────────────────────────────────
    "aud_skip_total":           {"helps_identify": _SD,  "description": "Total measurements skipped across all sensor types",              "output_range": "0–n",            "collection_method": _AUD},
    "aud_skip_frac":            {"helps_identify": _SD,  "description": "Fraction of all measurements that were skipped",                  "output_range": "0–1",            "collection_method": _AUD},
    "aud_skip_reason_top":      {"helps_identify": _SD,  "description": "Most common skip reason (string label from audit CSV)",          "output_range": "String / none",  "collection_method": _AUD},
    "aud_skip_chi2_frac":       {"helps_identify": f"{_GEN}, {_WW}", "description": "Fraction of skipped measurements whose skip_reason contains 'chi' (chi-squared gate rejection); high = Kalman filter consistently rejects sensor measurements as statistically inconsistent — indicates sensor misidentification, ghost track drift, or wrong dynamics model", "output_range": "0–1", "collection_method": _AUD},

    # ── Excel-derived ─────────────────────────────────────────────────────────
    "xl_n_vehicle_stuck_events":{"helps_identify": _SV,  "description": "Number of vehicle_stuck events in Issues sheet",                  "output_range": "0–n",            "collection_method": _XLS},
    "xl_first_stuck_t":         {"helps_identify": _SV,  "description": "Timestamp of first vehicle_stuck event [s]",                      "output_range": "0–∞ s / NaN",    "collection_method": _XLS},
}

# Backward-compatible simple dict (description only)
FEATURE_CATALOG = {k: v["description"] for k, v in FEATURE_CATALOG_RICH.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_fill(hex_colour: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_colour)


def _hex_font(hex_colour: str, bold: bool = False, size: int = 10) -> Font:
    return Font(color=hex_colour, bold=bold, size=size, name="Arial")


def _thin_border() -> Border:
    s = Side(style="thin", color="D0D0D0")
    return Border(left=s, right=s, top=s, bottom=s)


def _group_of(col_name: str) -> str:
    """Return the palette key for a column name."""
    if col_name == "label":
        return "lbl"
    if col_name in ("scenario_id", "global_track_id", "vehicle_id_oracle"):
        return "id"
    for prefix in ("kin_", "kf_", "cov_", "iv_", "dis_", "das_", "cam_", "gps_", "aud_", "xl_", "sc_"):
        if col_name.startswith(prefix):
            return prefix
    return "id"


def _set_col_width(ws, col_idx: int, width: float) -> None:
    ws.column_dimensions[get_column_letter(col_idx)].width = width


# ---------------------------------------------------------------------------
# Sheet 1: Features
# ---------------------------------------------------------------------------

def _build_features_sheet(ws, df: pd.DataFrame) -> None:
    ws.title = "Features"

    cols = list(df.columns)
    n_cols = len(cols)
    n_rows = len(df)

    # ── Header row ─────────────────────────────────────────────────────────
    for c_idx, col in enumerate(cols, start=1):
        group  = _group_of(col)
        bg, fg, _ = GROUP_PALETTE[group]
        cell = ws.cell(row=1, column=c_idx, value=col)
        cell.fill      = _hex_fill(bg)
        cell.font      = Font(color=fg, bold=True, size=9, name="Arial")
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border    = _thin_border()
    ws.row_dimensions[1].height = 42

    # ── Data rows ──────────────────────────────────────────────────────────
    for r_idx, (_, row_data) in enumerate(df.iterrows(), start=2):
        is_even = (r_idx % 2 == 0)
        for c_idx, col in enumerate(cols, start=1):
            group = _group_of(col)
            _, _, band = GROUP_PALETTE[group]
            val  = row_data[col]
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            if is_even:
                cell.fill = _hex_fill(band)
            cell.font      = Font(size=9, name="Arial")
            cell.alignment = Alignment(horizontal="left", vertical="center")
            cell.border    = _thin_border()
        ws.row_dimensions[r_idx].height = 15

    # ── Column widths ──────────────────────────────────────────────────────
    _set_col_width(ws, 1, 30)
    _set_col_width(ws, 2, 12)
    _set_col_width(ws, 3, 14)
    if len(cols) > 3 and cols[3] == "label":
        _set_col_width(ws, 4, 10)
    for c_idx in range(5, n_cols + 1):
        _set_col_width(ws, c_idx, 12)

    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{get_column_letter(n_cols)}{n_rows + 1}"
    ws.sheet_view.zoomScale = 90


# ---------------------------------------------------------------------------
# Sheet 2: Summary
# ---------------------------------------------------------------------------

def _build_summary_sheet(ws, df: pd.DataFrame, subtitle: str = "") -> None:
    ws.title = "Summary"

    TITLE_FILL = _hex_fill("1A252F")
    TITLE_FONT = Font(color="FFFFFF", bold=True, size=14, name="Arial")
    SEC_FILL   = _hex_fill("2C3E50")
    SEC_FONT   = Font(color="FFFFFF", bold=True, size=10, name="Arial")
    KEY_FONT   = Font(bold=True, size=10, name="Arial", color="2C3E50")
    VAL_FONT   = Font(size=10, name="Arial")
    ALT_FILL   = _hex_fill("F2F3F4")

    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 36
    ws.column_dimensions["D"].width = 22

    r = 1

    ws.merge_cells(f"A{r}:D{r}")
    c = ws.cell(row=r, column=1, value="SimStudio Anomaly Model — Feature Summary")
    c.fill = TITLE_FILL; c.font = TITLE_FONT
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[r].height = 28
    r += 1

    ws.merge_cells(f"A{r}:D{r}")
    c = ws.cell(row=r, column=1, value=subtitle or "features.csv")
    c.fill = _hex_fill("2980B9")
    c.font = Font(color="FFFFFF", italic=True, size=10, name="Arial")
    c.alignment = Alignment(horizontal="center")
    ws.row_dimensions[r].height = 18
    r += 2

    def section(title: str) -> None:
        nonlocal r
        ws.merge_cells(f"A{r}:D{r}")
        c = ws.cell(row=r, column=1, value=title)
        c.fill = SEC_FILL; c.font = SEC_FONT
        c.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[r].height = 18
        r += 1

    def kv(key: str, val, alt: bool = False) -> None:
        nonlocal r
        ck = ws.cell(row=r, column=1, value=key)
        cv = ws.cell(row=r, column=2, value=val)
        ck.font = KEY_FONT; cv.font = VAL_FONT
        if alt:
            ck.fill = ALT_FILL; cv.fill = ALT_FILL
        ck.border = _thin_border(); cv.border = _thin_border()
        ck.alignment = Alignment(horizontal="left", indent=1)
        cv.alignment = Alignment(horizontal="right")
        ws.row_dimensions[r].height = 15
        r += 1

    section("Dataset Overview")
    n_tracks    = len(df)
    n_scenarios = df["scenario_id"].nunique() if "scenario_id" in df.columns else "—"
    n_features  = len([c for c in df.columns
                       if c not in ("scenario_id", "global_track_id", "vehicle_id_oracle", "label")])
    kv("Total track rows",  n_tracks,        alt=False)
    kv("Unique scenarios",  n_scenarios,     alt=True)
    kv("Feature columns",   n_features,      alt=False)
    kv("Total columns",     len(df.columns), alt=True)
    if "label" in df.columns:
        for lbl in df["label"].unique():
            kv(f"  {lbl} rows", int((df["label"] == lbl).sum()), alt=False)
    r += 1

    section("Columns Per Feature Group")
    group_counts: Dict[str, int] = {}
    for col in df.columns:
        g = _group_of(col)
        group_counts[g] = group_counts.get(g, 0) + 1

    for g_key, g_name in GROUP_NAMES.items():
        count = group_counts.get(g_key, 0)
        if count == 0:
            continue
        bg, fg, _ = GROUP_PALETTE[g_key]
        ck = ws.cell(row=r, column=1, value=g_name)
        cv = ws.cell(row=r, column=2, value=count)
        ck.fill = _hex_fill(bg); cv.fill = _hex_fill(bg)
        ck.font = Font(color=fg, bold=True, size=9, name="Arial")
        cv.font = Font(color=fg, bold=True, size=9, name="Arial")
        ck.border = _thin_border(); cv.border = _thin_border()
        ck.alignment = Alignment(horizontal="left", indent=1)
        cv.alignment = Alignment(horizontal="right")
        ws.row_dimensions[r].height = 15
        r += 1
    r += 1

    section("Key Statistics")

    def stat_row(label: str, col_name: str, fmt: str = ".2f", alt: bool = False) -> None:
        nonlocal r
        if col_name not in df.columns:
            return
        s = df[col_name].dropna()
        mean_val = f"{s.mean():{fmt}}" if len(s) else "—"
        ck = ws.cell(row=r, column=1, value=label)
        cv = ws.cell(row=r, column=2, value=mean_val)
        ck.font = KEY_FONT; cv.font = VAL_FONT
        if alt:
            ck.fill = ALT_FILL; cv.fill = ALT_FILL
        ck.border = _thin_border(); cv.border = _thin_border()
        ck.alignment = Alignment(horizontal="left", indent=1)
        cv.alignment = Alignment(horizontal="right")
        ws.row_dimensions[r].height = 15
        r += 1

    stat_row("Mean speed (m/s)",          "kin_speed_mean_mps")
    stat_row("Max speed (m/s)",           "kin_speed_max_mps",   alt=True)
    stat_row("Speed excess max (m/s)",    "kin_speed_excess_max_mps")
    stat_row("Mean KF pos error (m)",     "kf_pos_err_mean_m",   alt=True)
    stat_row("Mean RMSE (m)",             "kf_track_rmse_m")
    stat_row("Min inter-vehicle dist (m)","iv_min_dist_m",        alt=True)
    stat_row("Lateral offset abs (m)",    "cov_das_lateral_offset_abs_mean_m")
    stat_row("Mean DAS SNR",              "das_snr_mean",         alt=True)
    stat_row("Mean cam confidence",       "cam_confidence_mean")
    stat_row("Coverage pred-only frac",   "cov_pred_only_frac",   alt=True)
    stat_row("Mean track duration (s)",   "cov_duration_s")

    ws.sheet_view.zoomScale = 100


# ---------------------------------------------------------------------------
# Sheet 3: Feature Catalog  (6 columns)
# ---------------------------------------------------------------------------

def _build_catalog_sheet(ws, df_cols: List[str]) -> None:
    ws.title = "Feature Catalog"

    # Column widths
    col_widths = [32, 20, 22, 58, 20, 40]
    col_letters = [get_column_letter(i) for i in range(1, 7)]
    for letter, width in zip(col_letters, col_widths):
        ws.column_dimensions[letter].width = width

    TITLE_FILL = _hex_fill("1A252F")
    TITLE_FONT = Font(color="FFFFFF", bold=True, size=13, name="Arial")
    HDR_FILL   = _hex_fill("2C3E50")
    HDR_FONT   = Font(color="FFFFFF", bold=True, size=10, name="Arial")

    # Title
    ws.merge_cells("A1:F1")
    c = ws.cell(row=1, column=1, value="Feature Catalog — All Columns")
    c.fill = TITLE_FILL; c.font = TITLE_FONT
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 26

    # Header row
    headers = ["Feature", "Group", "Helps identify", "Description",
               "Output range", "Collection method"]
    for col_i, h in enumerate(headers, start=1):
        cell = ws.cell(row=2, column=col_i, value=h)
        cell.fill = _hex_fill("2C3E50")
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = _thin_border()
    ws.row_dimensions[2].height = 18

    # Build ordered feature list: catalog-ordered first, then extra columns
    present = [c for c in FEATURE_CATALOG_RICH if c in df_cols]
    extra   = [c for c in df_cols if c not in FEATURE_CATALOG_RICH]

    for row_i, col_name in enumerate(present + extra, start=3):
        group = _group_of(col_name)
        bg, fg, band = GROUP_PALETTE[group]
        alt = (row_i % 2 == 0)
        meta = FEATURE_CATALOG_RICH.get(col_name, {})

        row_bg   = band if alt else bg
        row_fg   = "2C3E50" if alt else fg
        row_bold = not alt

        def _cell(col_i: int, value: str, monospace: bool = False) -> None:
            c = ws.cell(row=row_i, column=col_i, value=value)
            c.fill = _hex_fill(row_bg)
            c.font = Font(
                color=row_fg, size=9, bold=(row_bold and col_i == 1),
                name="Courier New" if monospace else "Arial"
            )
            c.border = _thin_border()
            c.alignment = Alignment(horizontal="left", indent=1,
                                    vertical="center", wrap_text=True)

        # Col A: feature name (monospace chip coloured by group)
        c1 = ws.cell(row=row_i, column=1, value=col_name)
        c1.fill = _hex_fill(bg if not alt else band)
        c1.font = Font(color=fg if not alt else "2C3E50", size=9,
                       bold=row_bold, name="Courier New")
        c1.border = _thin_border()
        c1.alignment = Alignment(horizontal="left", indent=1, vertical="center")

        # Col B: group badge (always group colour)
        c2 = ws.cell(row=row_i, column=2, value=GROUP_NAMES.get(group, group))
        c2.fill = _hex_fill(bg)
        c2.font = Font(color=fg, size=8, bold=True, name="Arial")
        c2.border = _thin_border()
        c2.alignment = Alignment(horizontal="center", vertical="center")

        # Cols C–F: rich metadata fields
        _cell(3, meta.get("helps_identify", "—"))
        _cell(4, meta.get("description", "(no description)"))
        _cell(5, meta.get("output_range", "—"))
        _cell(6, meta.get("collection_method", "—"))

        ws.row_dimensions[row_i].height = 15

    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:F{len(present) + len(extra) + 2}"
    ws.sheet_view.zoomScale = 100


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_excel(input_path: Path, output_path: Path) -> None:
    print(f"\nSimStudio — Feature Excel Builder")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")

    if not input_path.exists():
        print(f"\n[ERROR] Input CSV not found: {input_path}")
        print("Run `python anomaly_model/scripts/exp_scripts/run_extraction.py` first.")
        sys.exit(1)

    print("  Reading CSV …", end="", flush=True)
    df = pd.read_csv(input_path, low_memory=False)
    print(f" {len(df)} rows × {len(df.columns)} columns")

    wb = openpyxl.Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    subtitle = f"{input_path.name}  ·  {len(df)} tracks, {len(df.columns)} columns"

    print("  Building Features sheet …", end="", flush=True)
    ws_feat = wb.create_sheet("Features")
    _build_features_sheet(ws_feat, df)
    print(" done")

    print("  Building Summary sheet …", end="", flush=True)
    ws_summ = wb.create_sheet("Summary")
    _build_summary_sheet(ws_summ, df, subtitle)
    print(" done")

    print("  Building Feature Catalog sheet …", end="", flush=True)
    ws_cat = wb.create_sheet("Feature Catalog")
    _build_catalog_sheet(ws_cat, list(df.columns))
    print(" done")

    wb.active = ws_summ

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(output_path))
    print(f"\n  ✓  Saved: {output_path}")
    n_catalog = sum(1 for c in df.columns if c in FEATURE_CATALOG_RICH)
    n_extra   = sum(1 for c in df.columns if c not in FEATURE_CATALOG_RICH)
    print(f"     Catalog entries : {n_catalog} known + {n_extra} unlabelled")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_features_excel",
        description="Convert a features CSV into a styled 3-sheet Excel workbook.",
    )
    p.add_argument("--input",  "-i", default=str(DEFAULT_INPUT),
                   metavar="CSV",
                   help=f"Input CSV (default: {DEFAULT_INPUT})")
    p.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT),
                   metavar="XLSX",
                   help=f"Output XLSX (default: {DEFAULT_OUTPUT})")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    make_excel(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
