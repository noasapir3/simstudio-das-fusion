# Anomaly Intel — How to Read the Charts

**SimStudio | Advanced Control Lab — Anomaly Detection Module**

This guide explains what each chart in the **Anomaly Intel** tab means, how to run an analysis, and how to interpret the results.

---

## 1. What Is Anomaly Intel?

The Anomaly Intel tab automatically scores every vehicle in the current simulation against a trained statistical baseline model. The model was trained on recordings from hundreds of *normal* driving scenarios. Any vehicle whose behavior deviates strongly from that baseline is flagged as **anomalous**.

The scoring method is **Z-score based**: for each feature (e.g., speed, jerk, inter-vehicle distance) the system computes how many standard deviations the vehicle's value is from the normal population mean. A vehicle is flagged when its **maximum Z-score across all features exceeds the threshold** (currently **15.8 σ**).

---

## 2. How to Run an Analysis

1. **Open a scenario** in SimStudio (File → Open `.sim.json`).
2. **Run the simulation** — press Play and let it run until the event you want to analyze happens (e.g., the collision occurs, the reckless driver accelerates, etc.). The more of the scenario you simulate, the more complete the analysis.
3. Switch to the **Anomaly Intel** tab.
4. Click **▶ Analyze Simulation**.
5. Results appear in the four sub-tabs below.

> **Important:** Run the simulation *before* clicking Analyze. The analysis operates on events that already happened in the current session. If you analyze before a collision occurs, `iv_collision_detected` will be 0 for all vehicles.

---

## 3. The Four Charts

### 3.1 Z-Score Heatmap

**What it shows:** A color-coded grid where each column is a vehicle and each row is a behavioral feature. Color indicates the signed Z-score:

| Color | Meaning |
|-------|---------|
| Dark red | Very high anomaly signal (feature far above normal) |
| Light red / pink | Moderate high signal |
| White / light grey | Normal range (near-zero Z-score) |
| Light blue | Feature is below normal (e.g., unusually slow vehicle) |
| Dark blue | Strongly below normal |

**How to use it:**
- Look for columns with many dark-red cells — those vehicles are abnormal across many dimensions simultaneously.
- Look for single bright-red cells — that vehicle has one extreme outlier feature (e.g., a collision flag, extreme jerk).
- Features are grouped by category (collision, proximity/TTC, kinematics, Kalman filter, sensor coverage, DAS).
- Only features with at least one vehicle showing |Z| > 0.5 are displayed; quiet features are hidden to reduce noise.

**Black cell borders** mark cells where |Z| exceeds the anomaly threshold — these are the direct drivers of a vehicle being flagged.

---

### 3.2 Score Ranking

**What it shows:** Two sub-plots for each vehicle:

**Left — Overall Z-Score bar chart:**  
Bars show the maximum Z-score per vehicle (the "worst" feature). The red dashed line is the anomaly threshold (15.8 σ). Vehicles above the line are **flagged anomalous**. The bar is labelled with the name of the top-contributing feature.

**Right — Isolation Forest Score (legacy):**  
If a legacy sklearn model is loaded, this shows the Isolation Forest anomaly score (higher = more anomalous). In the current model this panel shows "Z-Score only model" since the IF component has been removed in favor of the more interpretable Z-score method.

**How to use it:**
- Compare bar heights to quickly rank vehicles from most to least anomalous.
- The **top feature label** tells you *why* a vehicle was flagged — e.g., `iv_collision_detected` means the flag is driven by a collision event, while `kin_jerk_max_mps3` means sudden acceleration changes.

---

### 3.3 Feature Distributions

**What it shows:** For every feature the model uses, a small histogram shows the *training distribution* (blue, from normal scenarios) overlaid with colored dots showing where each vehicle falls.

**How to use it:**
- Vehicles plotted far to the right of the blue histogram are extreme outliers on that feature.
- Features where all vehicles cluster inside the histogram are uninteresting; skip them.
- Look for features where a vehicle is orders of magnitude outside the normal range.

**Common patterns:**
- `iv_collision_detected = 1.0` with training distribution at 0 → collision was detected.
- `kin_stopped_frac` near 1.0 with training distribution near 0 → vehicle was stopped for almost the entire scenario (consistent with a collision aftermath or obstacle).
- `iv_collision_risk_proxy` very high → vehicle had a close approach with high relative speed.

---

### 3.4 CUSUM Temporal Analysis

**What it shows:** A **CUSUM (Cumulative Sum) chart** tracking each vehicle's speed over time. CUSUM is a statistical process-control method that detects *sustained shifts* in behavior — it accumulates evidence of deviation until it crosses a detection threshold.

- The **upper CUSUM** (positive values) detects sustained speed *increases*.
- The **lower CUSUM** (negative values) detects sustained speed *decreases*.
- A **red marker (▲)** indicates the time point at which the CUSUM alarm fired.
- The **speed trace** (thin line) shows the raw vehicle speed alongside the CUSUM curves.

**How to use it:**
- Use this to answer *when* an anomaly started, not just *whether* it occurred.
- A vehicle that suddenly stops (e.g., after a collision) will show a strong lower CUSUM alarm — the cumulative evidence of "this vehicle is slower than normal" crosses the threshold quickly.
- Normal stochastic speed variation keeps CUSUM near zero; structural changes in behavior push it to one side.

---

## 4. Anomaly Decision Logic

```
For each vehicle:
  For each feature in the model (162 features):
    z = (live_value − training_mean) / training_std

  max_z = max(|z₁|, |z₂|, ..., |z₁₆₂|)

  if max_z > 15.8:
    vehicle = ANOMALOUS  ← top feature is the main driver
  else:
    vehicle = NORMAL
```

The threshold of **15.8 σ** was chosen as the 99th percentile of `max_z` across all vehicles in the normal training set. This means a false-positive rate of ~1% on clean scenarios.

---

## 5. Key Features and What They Mean

### Collision Features
| Feature | Description |
|---------|-------------|
| `iv_collision_detected` | 1 if a `world.collision` event occurred involving this vehicle; 0 otherwise. Z=1000 when triggered — always the dominant signal. |
| `iv_overlap_frac` | Fraction of sampled timesteps where this vehicle was within 5 m (center-to-center) of another vehicle. |
| `iv_overlap_duration_s` | Total seconds of overlap (estimated). |
| `iv_collision_risk_proxy` | max(Δv² / dist) over all timesteps — kinetic energy proxy. Peaks sharply just before impact. |

### Proximity / TTC
| Feature | Description |
|---------|-------------|
| `iv_min_dist_m` | Minimum inter-vehicle distance observed over the scenario. |
| `iv_mean_min_dist_m` | Mean of per-timestep minimum distances. |
| `iv_ttc_min_s` | Minimum time-to-collision (TTC) = dist / relative speed. |
| `iv_rel_speed_at_min_dist_mps` | Relative speed at the moment of closest approach. |

### Kinematics
| Feature | Description |
|---------|-------------|
| `kin_speed_mean_mps` | Mean speed over the scenario (m/s). |
| `kin_stopped_frac` | Fraction of time the vehicle was stationary (speed < 0.5 m/s). |
| `kin_jerk_max_mps3` | Peak rate of acceleration change (m/s³). High jerk = sudden braking or impact. |
| `kin_accel_max_abs_mps2` | Largest absolute acceleration observed. |
| `kin_decel_max_mps2` | Largest deceleration observed. |
| `kin_heading_change_rate_max_rad_per_s` | Maximum heading change rate — detects swerving. |

### Kalman Filter / Tracking
| Feature | Description |
|---------|-------------|
| `kf_pos_err_mean_m` | Mean position estimation error from Kalman filter. |
| `kf_pos_err_max_m` | Maximum position error — spikes after abrupt maneuvers. |
| `kf_snr_mean_db` | Mean signal-to-noise ratio of DAS fiber sensor detections. |

### Sensor Coverage
| Feature | Description |
|---------|-------------|
| `cov_total_rows` | Total measurement rows for this vehicle. |
| `cov_pred_only_frac` | Fraction of timesteps with only predicted (no sensor) detections. |
| `das_snr_mean` | Mean SNR across all DAS detections for this vehicle. |

---

## 6. What "Normal" Looks Like

The model was trained on scenarios with vehicles driving at typical speeds (10–28 m/s), maintaining safe following distances (> 10 m), braking smoothly, and completing routes without incident. A normal vehicle's max Z-score is typically 2–8 σ.

Anomalous signatures trained-in include:
- **Collision** (`allow_collision=True` in scenario JSON): sudden stop, frozen position post-impact, `iv_collision_detected=1`
- **Reckless driver**: extreme speed, high jerk, close following, high TTC alarm frequency
- **Frozen/stuck vehicle**: `kin_stopped_frac ≈ 1.0`, zero speed variation

---

## 7. Exporting Results

Click **📄 Export Report…** (in the Anomaly Intel toolbar or the tab control bar) to generate an Excel workbook with four sheets:

| Sheet | Contents |
|-------|---------|
| **Summary** | Verdict per vehicle (color-coded), top feature, Z-score and IF score |
| **Z-Score Heatmap** | Signal-only features × vehicles, coloured by signed Z-score |
| **All Feature Z-Scores** | Full 162-feature × vehicle matrix |
| **Charts** | All four matplotlib charts embedded at high resolution |

The file is saved in the same folder as the current simulation's export, or your working directory if no export path is set.

---

## 8. Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| "No simulation data" | You clicked Analyze before running the simulation | Press Play, let it run, then Analyze |
| Collision vehicle not flagged | Analysis ran before collision happened | Run simulation until collision occurs, then re-Analyze |
| veh004 flagged instead of colliding vehicles | Old model or old app.py with buggy `iv_collision_risk_proxy` | Make sure you have the latest `model_live.pkl` and `app.py` |
| Model not loaded (info panel blank) | `anomaly_model/outputs/model_live.pkl` missing | Run: `python anomaly_model/scripts/scenario_scorer.py --train-live` |
| Feature Distributions chart empty | pandas not installed, or model features list is empty | Check console for errors; re-load model |

---

*Last updated: May 2026 | SimStudio Anomaly Detection v2.0*
