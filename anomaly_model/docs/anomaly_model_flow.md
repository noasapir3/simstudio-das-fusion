# Anomaly Model — Living Flow Document

**Project:** SimStudio — Fiber-Optic Traffic Monitoring  
**Module path:** `anomaly_model/`  
**Current phase:** Phase 3 — CUSUM Temporal Detector (complete)  
**Last updated:** Phase 3 — cusum_detector.py written and smoke-tested

---

## 1. What Currently Exists

```
anomaly_model/
├── __init__.py                  Module package marker + version (0.3.0)
├── feature_extractor.py         Phase 1: reads all exports, extracts 123 features
├── cusum_detector.py            Phase 3: Page's CUSUM on NIS_x (temporal detector)
├── anomaly_model_flow.md        This file — living documentation
└── outputs/
    ├── features_sample.csv      Feature extraction result on sample_audit_report
    └── feature_catalog.csv      Full catalog of 145 documented features
```

**Simulator files modified (surgical additions only):**
- `src/simstudio/audit.py` — 4 new DAS physics fields on `AuditRow`; new `write_audit_workbook()` function; `export_all()` now also writes a consolidated `tracking_audit.xlsx`

No Kalman filter, GUI, simulator, or tracking code was touched.

---

## 2. Data Sources Being Used

| Source | File Pattern | Status | Notes |
|--------|-------------|--------|-------|
| Track trajectory | `track_trajectory_T*.csv` | ✅ Used | Primary source: Kalman states, ground truth, per-sensor measurements |
| Measurement audit | `kalman_measurement_audit.xlsx` | ✅ Used | DAS SNR, camera confidence, skip reasons — now also includes DAS physics fields |
| Scenario definition | `*.json` / `*.sim.json` | ✅ Used | Vehicle weights, anomaly flags, sensor config |
| Excel export | `simstudio_export_*.xlsx` | ✅ Used | Issues, RMSE, Vehicles, DAS sheets |
| Coverage summary | `coverage_summary.xlsx` | ✅ Used | Sensor-level counts; used for per-track update-share features |
| JSONL recording | `events.jsonl` | 🔄 Bridged | `das_amplitude`, `fiber_distance_m`, `snr_th`, `lateral_offset_m` now extracted from JSONL and written to audit CSV during export |

### New DAS physics fields now in audit CSV

Added to `AuditRow` dataclass and `_AUDIT_FIELDS` tuple in `audit.py`. Written for DAS rows only; non-DAS rows get `0.0`:

| Field | Formula / Source | Why it matters |
|-------|-----------------|----------------|
| `das_amplitude` | `A = W / (r + d0)²` | Proportional to vehicle weight — basis for weight back-estimation |
| `fiber_distance_m` | Perpendicular vehicle-to-fiber distance `r` [m] | Varies with lateral position; high std → weaving |
| `snr_th` | Traffic-level SNR threshold used | Required to back-calculate `W_est = SNR × snr_th × (r + d0)²` |
| `lateral_offset_m` | Signed offset from lane centreline [m] | Large absolute value → straddling / weaving |

---

## 3. Features Currently Extracted

**123 columns total per (scenario, track) row.** Organized into 9 groups:

### Group 1: Identifiers (3 features)
`scenario_id`, `global_track_id`, `vehicle_id_oracle`

### Group 2: Scenario Metadata — from JSON (16 features, prefix `sc_`)
Read from the scenario `.json` / `.sim.json` file. All are NaN if no JSON is found.

| Feature | Description |
|---------|-------------|
| `sc_n_vehicles` | Total vehicles in scenario |
| `sc_n_frozen` | Vehicles with `frozen=True` (stalled / obstacle) |
| `sc_n_speeding` | Vehicles with `ignore_speed_limit=True` |
| `sc_n_collision` | Vehicles with `allow_collision=True` |
| `sc_n_weaving` | Vehicles with `lateral_mode='weave'` |
| `sc_n_straddling` | Vehicles with `lateral_mode='straddle'` |
| `sc_n_braking_override` | Vehicles with `a_cmd_override_mps2 < -2` |
| `sc_n_tailgating` | Vehicles with `min_gap_m < 1.5 m` |
| `sc_weight_kg_list` | Comma-separated vehicle weights |
| `sc_speed_limit_mps` | Min speed limit across all segments |
| `sc_has_das` | 1 if DAS sensors present |
| `sc_das_fiber_offset_m` | Fiber lateral offset from road `[m]` |
| `sc_das_d0_m` | Singularity floor `d0` |
| `sc_das_noise_std` | DAS trace noise std |
| `sc_das_snr_th` | SNR threshold (traffic-level dependent) |
| `sc_anomaly_type` | **Heuristic label** inferred from flags (see below) |

**Anomaly type labelling logic** (priority order):
1. `allow_collision` → `"collision"`
2. `ignore_speed_limit` → `"speeding"`
3. `a_cmd_override < -2` → `"sudden_braking"`
4. `lateral_mode='weave'` → `"weaving"`
5. `lateral_mode='straddle'` → `"lane_straddling"`
6. `min_gap_m < 1.5` → `"tailgating"`
7. `frozen=True` + non-vehicle object_type → `"obstacle_in_lane"`
8. `frozen=True` + vehicle object_type → `"stalled_vehicle"`
9. else → `"normal"`

### Group 3: Kinematic Features — from trajectory CSV (13 features, prefix `kin_`)
Computed from Kalman-estimated states (`vx_hat`, `vy_hat`, `ax_hat`, `ay_hat`, `y_hat`).

| Feature | What it detects |
|---------|----------------|
| `kin_speed_mean_mps`, `_max_`, `_std_` | Speed profile — high max → speeding |
| `kin_speed_over_limit_frac` | Fraction of time spent exceeding speed limit |
| `kin_accel_mean_abs_mps2`, `_max_`, `_std_` | Acceleration intensity — spike → braking/collision |
| `kin_phys_impossible_v` | Rows with speed > 40 m/s → Kalman divergence |
| `kin_phys_impossible_a` | Rows with accel > 8 m/s² → Kalman divergence |
| `kin_lateral_dev_mean_m`, `_max_`, `_std_` | Lateral displacement — high → weaving/straddling |
| `kin_lateral_oscillation_ratio` | `std(lat) / mean(lat)` — high sinusoidal ratio → weaving |

### Group 4: Kalman Quality Features — from trajectory CSV (11 features, prefix `kf_`)

| Feature | What it detects |
|---------|----------------|
| `kf_pos_err_mean_m`, `_max_`, `_std_`, `_p90_` | Overall tracking quality |
| `kf_track_rmse_m` | Track RMSE — primary summary metric |
| `kf_rolling_rmse_spike_count` | Sudden quality drops within a track |
| `kf_sigma_pos_mean_m`, `_max_` | Filter uncertainty level |
| `kf_consistency_ratio_mean`, `_max_` | `pos_err / sigma_pos` — >2 → overconfident filter |
| `kf_overconfident_frac` | Fraction of time the filter underestimates its own error |

### Group 5: Coverage & Sensor Utility — from trajectory + audit (39 features, prefix `cov_`)

This is the most information-dense group. It answers four research questions about sensor layout and utility.

#### 5A — Per-sensor active fractions (11 features)
What fraction of tracking time does each sensor contribute?

| Feature | Interpretation |
|---------|---------------|
| `cov_das_active_frac` | DAS coverage fraction — low → fiber misses much of the road |
| `cov_cam_active_frac` | Camera coverage fraction — low → camera range or occlusion issue |
| `cov_gps_active_frac` | GPS coverage fraction |
| `cov_multi_sensor_frac` | Fraction with ≥2 sensors simultaneously — high → good fusion opportunity |
| `cov_single_sensor_frac` | Fraction with exactly 1 sensor — single point of failure |
| `cov_no_sensor_frac` | Fraction with no sensor at all (pure Kalman prediction) |
| `cov_das_cam_overlap_frac` | DAS + Camera simultaneous coverage — fusion quality indicator |
| `cov_das_gps_overlap_frac` | DAS + GPS simultaneous coverage |
| `cov_cam_gps_overlap_frac` | Camera + GPS simultaneous coverage |
| `cov_all_sensors_frac` | All three sensors active simultaneously — ideal tracking |
| `cov_dominant_sensor` | Sensor active for the most rows (DAS / Camera / GPS / none) |

#### 5B — Fusion benefit (9 features)
Does sensor redundancy actually reduce tracking error?

| Feature | Interpretation |
|---------|---------------|
| `cov_err_das_only_m` | Mean position error when DAS is the sole active sensor |
| `cov_err_cam_only_m` | Mean position error when Camera is the sole active sensor |
| `cov_err_gps_only_m` | Mean position error when GPS is the sole active sensor |
| `cov_err_multi_sensor_m` | Mean position error under multi-sensor fusion |
| `cov_err_no_sensor_m` | Mean position error during prediction-only (no sensor) |
| `cov_fusion_benefit_vs_best_single` | `best_single_err − multi_err` — positive = fusion helps |
| `cov_err_cam_no_das_m` | Camera-only error (DAS absent) |
| `cov_err_cam_with_das_m` | Camera error when DAS is also present |
| `cov_das_fusion_benefit_m` | `cam_no_das_err − cam_with_das_err` — positive = DAS helps |

#### 5C — Temporal and spatial gap structure (8 features)
Where and when do coverage gaps occur? Being blind at high speed is riskier than being blind at low speed.

| Feature | Interpretation |
|---------|---------------|
| `cov_gap_frac_early` | Fraction of the first third of the track that is prediction-only |
| `cov_gap_frac_middle` | Fraction of the middle third prediction-only |
| `cov_gap_frac_late` | Fraction of the last third prediction-only |
| `cov_gap_clustering_cv` | Coefficient of variation of gap positions — high = clustered gaps |
| `cov_gap_spatial_pos_mean_norm` | Where on the road gaps concentrate (0=start, 1=end of track) |
| `cov_gap_spatial_spread_norm` | Spatial spread of gaps — high = scattered, low = all in one zone |
| `cov_high_speed_gap_frac` | Fraction of blind rows where estimated speed > 50% of v_max |
| `cov_mean_speed_at_gaps_mps` | Average speed during blind periods — high is dangerous |

#### 5D — Effective update rates and DAS physics (11–21 features depending on data)
What is the real information rate of each sensor, and what do the DAS physics measurements reveal?

| Feature | Interpretation |
|---------|---------------|
| `cov_das_hz_est` | Estimated DAS update rate [Hz] |
| `cov_cam_hz_est` | Estimated Camera update rate [Hz] |
| `cov_gps_hz_est` | Estimated GPS update rate [Hz] |
| `cov_das_update_share` | Fraction of accepted Kalman updates from DAS |
| `cov_cam_update_share` | Fraction of accepted Kalman updates from Camera |
| `cov_gps_update_share` | Fraction of accepted Kalman updates from GPS |
| `cov_dominant_kalman_updater` | Which sensor drives the most Kalman corrections |
| `cov_das_amplitude_mean/std/cv` | DAS amplitude statistics — high CV → vehicle weaving (changing fiber distance) |
| `cov_das_fiber_dist_mean/std/max/cv` | Vehicle-to-fiber distance statistics — high std → lateral oscillation |
| `cov_das_lateral_offset_mean/std/max` | Lateral offset from lane centerline — large max → straddling |

### Group 6: Sensor Disagreement — from trajectory CSV (11 features, prefix `dis_`)

| Feature | What it detects |
|---------|----------------|
| `dis_das_cam_mean_m`, `_max_` | DAS vs Camera disagreement — large → sensor conflict |
| `dis_das_hat_mean_m`, `_max_` | DAS vs Kalman estimate — large → DAS outlier |
| `dis_cam_hat_mean_m`, `_max_` | Camera vs Kalman estimate |
| `dis_gps_hat_mean_m`, `_max_` | GPS vs Kalman estimate |
| `dis_das_only_err_mean_m` | Kalman error in DAS-only windows |
| `dis_cam_only_err_mean_m` | Kalman error in camera-only windows |
| `dis_das_worse_than_cam` | **1 if DAS degrades fusion** (DAS-only error > cam-only error) |

### Group 7: DAS Sensor Quality — from audit CSV (11 features, prefix `das_`)

| Feature | What it detects |
|---------|----------------|
| `das_n_measurements` | Total DAS events for this track |
| `das_snr_mean`, `_min_`, `_max_`, `_std_` | DAS signal quality over time |
| `das_snr_low_frac` | Fraction with SNR < 5 (high-noise window) |
| `das_sigma_mean_m`, `_max_` | DAS position uncertainty |
| `das_confidence_mean` | DAS reliability score (SNR-derived) |
| `das_sigma_v_mean_mps` | DAS velocity measurement uncertainty |
| `das_accept_frac`, `_skip_frac_` | Filter acceptance rate for DAS |

### Group 8: DAS Weight Estimation (6 features, prefix `das_W_` / `das_declared_`)

| Feature | Description | Availability |
|---------|-------------|-------------|
| `das_W_est_kg` | Back-estimated weight from SNR + geometry | Requires DAS SNR data + scenario JSON |
| `das_W_est_class` | Weight class of estimate (motorcycle / car / bus_hgv / …) | Same |
| `das_declared_weight_kg` | Weight from scenario JSON | Requires scenario JSON |
| `das_declared_weight_class` | Weight class of declared weight | Requires scenario JSON |
| `das_weight_ratio` | `W_est / declared` — 1.0 = perfect match | Both sources needed |
| `das_weight_anomaly` | 1 if ratio differs by > 50% | Both sources needed |

**Formula:** `W_est = SNR_mean × snr_th × (fiber_offset_m + d0_m)²`  
**Assumption:** vehicle drives at lane center (`r ≈ fiber_offset_m`). Error grows with lateral deviation.  
**Phase 1 improvement:** `das_amplitude` and `fiber_distance_m` are now written to the audit CSV, enabling per-timestep weight estimation in Phase 2.

### Camera, GPS, Audit Quality, Excel (prefixes `cam_`, `gps_`, `aud_`, `xl_`)
Camera confidence, GPS sigma, measurement skip statistics, and Excel-derived true-speed / RMSE-by-source features.

---

## 4. Consolidated Audit Workbook

Every time "Export All" is run from the SimStudio GUI, the audit folder now also contains:

```
tracking_audit_<timestamp>/
├── tracking_audit.xlsx          ← NEW: consolidated human-readable workbook
├── kalman_measurement_audit.xlsx
├── coverage_summary.xlsx
├── track_trajectory_T000001.xlsx
└── ...
```

**`tracking_audit.xlsx` sheet structure:**

| Sheet | Contents |
|-------|----------|
| **Read Me** | Plain-language guide to every sheet, key column definitions, glossary |
| **Coverage** | Per-sensor measurement counts: generated, accepted, skipped, % Kalman rows updated |
| **Kalman Audit** | Every sensor measurement with human-readable column headers: accepted/skipped decision, skip reason, DAS physics values |
| **Track T000001** | Per-timestep Kalman trajectory for that track: ground truth, each sensor's reading, Kalman estimate, position error |
| **Track T000002** | …and so on for each track |

The workbook uses the same dark-blue header styling as `simstudio_export_*.xlsx` for visual consistency. It is self-contained — you can open it and understand each table without any other documentation.

---

## 5. What Each Feature Means

See `outputs/feature_catalog.csv` for the complete one-line description of all 145 defined features. Key interpretations:

| If this feature is high... | It likely means... |
|---------------------------|-------------------|
| `kin_speed_max_mps` >> `sc_speed_limit_mps` | Speeding anomaly |
| `kin_speed_over_limit_frac` > 0.5 | Sustained speeding |
| `kin_lateral_oscillation_ratio` > 1.5 | Weaving behavior |
| `kin_lateral_dev_max_m` > 2 × lane_width | Straddling or extreme weave |
| `kin_accel_max_abs_mps2` > 5 and sudden | Braking event or collision |
| `kf_overconfident_frac` > 0.3 | Filter poorly calibrated |
| `cov_no_sensor_frac` > 0.5 | Sensor dropout / camera out-of-range |
| `cov_max_consec_pred` > 20 | Long blind period — significant drift risk |
| `cov_fusion_benefit_vs_best_single` < 0 | Multi-sensor fusion is *worse* than best single sensor (unexpected) |
| `cov_das_fusion_benefit_m` < 0 | Adding DAS on top of Camera makes tracking worse |
| `cov_high_speed_gap_frac` > 0.3 | The system is often blind precisely when the vehicle is fastest |
| `cov_das_amplitude_cv` > 0.5 | DAS amplitude varies a lot → vehicle weaving (changing r over time) |
| `cov_das_fiber_dist_std_m` > 1 m | Vehicle lateral position changes significantly → weaving |
| `cov_das_lateral_offset_max_m` > 1.5 m | Vehicle strays well beyond lane centre |
| `dis_das_cam_mean_m` > 5 m | Sensor fusion conflict |
| `dis_das_worse_than_cam = 1` | DAS adds noise, not signal |
| `das_snr_low_frac` > 0.3 | Vehicle too far from fiber, or very light (motorcycle) |
| `das_W_est_kg` >> `das_declared_weight_kg` | DAS sees heavier load than declared |
| `xl_n_vehicle_stuck_events` > 0 | Vehicle reached dead-end or collided |
| `xl_true_min_accel_mps2` < -5 | Hard braking in ground truth |

---

## 6. Current Stage of the Anomaly Model

```
Phase 1 — Feature Extraction                   ✅ COMPLETE
Phase 2 — Innovation Export (kalman.py patch)  ✅ COMPLETE
Phase 3 — Temporal Detection (CUSUM)           ✅ COMPLETE
Phase 4 — Cross-Scenario Anomaly Scoring       ⬜ WAITING — needs 500+ partner simulations
Phase 5 — Report / GUI Integration             ⬜ PLANNED
```

---

## 7. What Was Added (complete changelog — Phases 1–3)

### Phase 3 additions (`anomaly_model/`)
- **`cusum_detector.py`** — Page's CUSUM temporal detector
  - `CUSUMDetector` class: `fit(traj_df) → CUSUMResult`
  - `AlarmEvent` dataclass: onset_t, offset_t, duration_s, peak_stat, onset_index
  - `CUSUMResult` dataclass: traj_df (with `cusum_stat` + `cusum_alarm` columns), alarms list, NIS diagnostics
  - `run_cusum(path, ...)` — single-file convenience wrapper
  - `batch_cusum(folders, ...)` — multi-scenario summary DataFrame
  - Default parameters: k=1.5, h=5.0, gap_reset_s=2.0
  - NaN carry-forward for prediction-only rows; gap reset for measurement gaps > 2 s
- **`__init__.py`** updated: version 0.3.0, Phase 3 in docstring

### Phase 2 additions (`src/simstudio/`)
- **`kalman.py`**: 5 read-only innovation logging fields added to `LegacyKalmanFilter`
  - `last_innovation_x/y`, `last_S_x/y`, `last_NIS_x` — NaN on predict-only rows
  - `predict()` resets all 5 fields to NaN each timestep
  - `_update()` populates them by scanning H rows (x-position → row with H[i,0]==1, y-position → H[i,3]==1)
- **`audit.py`** — `TrajectoryRow` extended with 5 new fields: `innovation_x/y`, `innovation_S_x/y`, `NIS_x`
  - `_TRAJ_FIELDS` and `_TRAJ_DISPLAY` updated (33 columns total)
  - `traj_rows.append()` wires `_nan_to_none(kf.last_*)` into each row

### Phase 1 new files (`anomaly_model/`)
- `__init__.py` — package marker
- `feature_extractor.py` — 123-feature extraction module (1,400+ lines)
- `anomaly_model_flow.md` — this file
- `outputs/features_sample.csv` — sample extraction result
- `outputs/feature_catalog.csv` — all 145 features documented

### Changes to `src/simstudio/audit.py` (surgical additions only)
- **`AuditRow` dataclass**: 4 new fields appended — `das_amplitude`, `fiber_distance_m`, `snr_th`, `lateral_offset_m`
- **`_AUDIT_FIELDS`**: 4 new field names added to the export tuple
- **DAS event reader** (~line 299): extracts the 4 new fields from JSONL payload when `is_das=True`
- **`_AUDIT_DISPLAY`, `_TRAJ_DISPLAY`, `_COV_DISPLAY`**: human-readable column name mappings
- **`_readme_sheet()`**: writes the "Read Me" worksheet with plain-language sheet guide and column glossary
- **`write_audit_workbook()`**: new public function — consolidates all audit tables into one multi-sheet Excel workbook
- **`export_all()`**: now also calls `write_audit_workbook()` and returns `"audit_workbook"` key
- **`__all__`**: `write_audit_workbook` added

### Verified behaviors
- ✅ Reads all 5 data sources (trajectory CSV, audit CSV, Excel, scenario JSON, coverage summary)
- ✅ Produces 123 columns per (scenario, track) row
- ✅ Handles missing files gracefully (NaN rather than crash)
- ✅ Column schema is uniform across metadata-only and full-data rows
- ✅ All 39 `cov_` features compute correctly on sample data
- ✅ All 39 `cov_` features documented in `FEATURE_CATALOG`
- ✅ `write_audit_workbook()` produces all expected sheets without error
- ✅ Both `audit.py` and `feature_extractor.py` pass `py_compile` syntax check

---

## 8. What Is Still Missing

### Critical for Phase 4
- **Partner's simulation dataset**: 500+ normal + anomaly scenarios from the partner. This is the sole prerequisite for Phase 4. Nothing in the codebase needs to change — just run the simulations and point `batch_cusum()` and `extract_features()` at the output folders.

### Important for analysis
- **Ground-truth anomaly onset timestamps**: The scenario JSON provides vehicle-level anomaly flags but not exact onset times (e.g., "weaving starts at t=12.3 s"). These are needed to evaluate CUSUM detection latency (how many seconds after the anomaly starts does the alarm fire?).
- **Weight estimation refinement**: `das_amplitude` and `fiber_distance_m` are now in the audit CSV, enabling per-timestep weight estimation `W_est(t)`. A variance-over-time feature could further strengthen the weaving detector.

### Nice to have
- Per-timestep `W_est(t)` time series: `std(W_est_t) / mean(W_est_t)` → lateral oscillation indicator without needing explicit lateral offset
- `track_coverage_detail.csv`: per-track, per-timestep sensor active flags for finer gap analysis

---

## 9. Next Planned Steps

### Phase 4 — Cross-Scenario Isolation Forest
**Goal:** Score each scenario as normal/anomalous from its feature vector.

**New file:** `anomaly_model/scenario_scorer.py`  
**Input:** `features.csv` (many scenarios)  
**Output:** `anomaly_score`, `anomaly_rank`, feature contribution breakdown per scenario

**Prerequisite:** Run simulations for all anomaly scenario JSONs to get real trajectory data.

### Phase 5 — Report and GUI Integration
**Goal:** Add anomaly scores and plots to the existing report and diagnostic view.

**Changes:** One new sheet ("Anomaly") in Excel export, one new matplotlib figure.

---

## 10. Assumptions, Limitations, and Risks

### Assumptions
1. **Lane-center approximation for weight estimation**: `W_est = SNR × snr_th × (fiber_offset_m + d0_m)²` assumes the vehicle's lateral position equals `fiber_offset_m`. Now that `fiber_distance_m` is in the audit CSV, Phase 2 can use the actual `r` per timestep instead.
2. **Single DAS sensor per scenario**: The extractor reads `first_das` for fiber geometry. Multi-sensor scenarios may need per-sensor feature extraction (planned).
3. **Anomaly type is single-label**: The heuristic assigns one anomaly type per scenario. Multi-anomaly scenarios (e.g., speeding + weaving) are assigned to the first matched category.
4. **Global track ID uniqueness**: Feature rows are keyed by `global_track_id`. If a track ID is reused across replays, rows may conflict in a batch DataFrame.

### Limitations
1. **Weight estimation requires both SNR data AND scenario JSON**: If either is missing, `das_W_est_kg` is NaN.
2. **No ground truth anomaly labels at timestep level**: The scenario JSON provides vehicle-level anomaly flags, not timestep-level labels. Ground truth onset timestamps are needed to evaluate CUSUM detection latency precisely.
3. **CUSUM is one-sided**: The detector is calibrated for *elevated* NIS_x (filter inconsistency). A suddenly over-confident filter (NIS_x → 0) would not be flagged — a two-sided scheme is a future extension if needed.

### Risks
1. **Partner dataset volume**: Phase 4 quality depends on having enough normal vs. anomaly examples per anomaly type for Isolation Forest to separate them. If a class has <20 examples, scores will be unreliable.
2. **CUSUM threshold calibration**: k=1.5 / h=5.0 was calibrated theoretically for χ²(1). After the partner dataset arrives, empirical ARL should be validated on known-normal runs — recalibration may be needed if the filter's NIS_x mean under H₀ differs from 1.0.
3. **`kin_speed_over_limit_frac` is NaN when no scenario JSON**: The speed limit comes from the JSON. If the JSON is missing (e.g., a production scenario export without the definition file), this feature is unavailable.

---

## Appendix: Feature Extraction Quick-Start

```python
# Single scenario
from anomaly_model.feature_extractor import ScenarioLoader
loader = ScenarioLoader("path/to/scenario_outputs/",
                        scenario_id="my_scenario",
                        scenario_json="path/to/scenario.json")
print(loader.describe())
df = loader.extract()   # DataFrame: 1 row per track, 123 columns

# Batch over many folders
from anomaly_model.feature_extractor import extract_features
df = extract_features(
    folders=[folder1, folder2, ...],
    scenario_id_fn=lambda p: p.name,
)
df.to_csv("anomaly_model/outputs/features_all.csv", index=False)

# Print feature catalog
from anomaly_model.feature_extractor import FEATURE_CATALOG
for name, desc in FEATURE_CATALOG.items():
    print(f"{name}: {desc}")
```
