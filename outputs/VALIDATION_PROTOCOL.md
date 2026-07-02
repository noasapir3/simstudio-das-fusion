# Simulator Validation Protocol
## SimStudio — Fiber Optic Traffic Simulator

---

## Document Structure

**Part A** — 7 isolated test scenarios
**Part B** — Separation table: internal consistency vs. physical fidelity

---

## Preliminary Note: Correct Metrics for DAS

DAS is **not** a 2D position sensor. It measures strain-rate along the fiber. Therefore:

| Wrong metric ❌ | Correct metric ✅ |
|---|---|
| 2D position error (RMSE_x, RMSE_y) | Along-fiber position only — `fiber_position_m` |
| Comparing measured y to true vehicle y | Waterfall slope — `Δchannel / Δtime = v` |
| Vehicle position vs. fiber position | Time of Closest Approach (TCA) to a fiber channel |
| — | Peak amplitude — `A = W / (r + d₀)²` |
| — | SNR = A / σ_trace |

---

# Part A — Test Scenarios

---

## Scenario 1 — Baseline: Single Vehicle, Constant Speed

### Purpose
Obtain the cleanest possible DAS signature and validate the basic waterfall structure.

### Isolated variable
Nothing changes — speed, mass, and fiber distance are all fixed. No other vehicles.

### Scenario definition
| Parameter | Value |
|---|---|
| Vehicles | 1 only |
| weight_kg | 1500 |
| Speed | 13.9 m/s (50 km/h), constant (speed_min = speed_max) |
| fiber_offset_m | 2.0 m |
| d0_m | 0.7 m |
| traffic_level | "light" |
| Road length | 200 m straight |
| DAS channels | 0–200 |
| Camera / GPS | disabled |

### Expected output
1. **Single diagonal stripe** in the waterfall — no additional stripes.
2. **Stripe slope**: `Δchannel / Δt = 13.9 m/s` → time/distance slope = `1/13.9 ≈ 0.072 s/m`.
3. **Amplitude profile along fiber**: at any given moment, the channel closest to the vehicle shows a peak. Peak amplitude: `A_peak = 1500 / (2.0 + 0.7)² ≈ 206`.

### Metrics to check
| Metric | Calculation | Expected value |
|---|---|---|
| Waterfall slope | `fit_slope = Δfiber_position_m / Δt` | 13.9 ± 0.3 m/s |
| Peak amplitude | max(SNR) × σ_trace | ≈ 206 units |
| SNR at closest approach | SNR at the nearest channel | ≈ 6.9 (= 206/30) |
| Number of stripes | visual / peak detection | exactly 1 |

### Connection to papers
- Hen et al. 2023, eq. (3): `ε̇_DAS = (1/L)[u̇_x(x+L/2) − u̇_x(x−L/2)]`
  The waterfall slope is directly derived from vehicle velocity — validated here.
- Nissan & Nissan 2023: spatial sampling 1 m⁻¹, temporal 30 Hz → minimum resolution needed to measure slope.

---

## Scenario 2 — Speed Variation Only

### Purpose
Validate that the waterfall slope changes proportionally with 1/v and nothing else.

### Isolated variable
`speed_mean_mps` — three separate runs.

### Scenario definition
| Run | Speed | Expected slope (s/m) |
|---|---|---|
| A | 8.33 m/s (30 km/h) | 0.120 |
| B | 13.9 m/s (50 km/h) | 0.072 |
| C | 22.2 m/s (80 km/h) | 0.045 |

> All other parameters identical to Scenario 1. Speed is constant within each run.

### Expected output
Measured slope = `1/v ± 5%`.
Peak amplitude should remain unchanged (W and r are fixed).

### Metrics to check
| Metric | Calculation |
|---|---|
| `slope_measured` | linear fit on `(t, fiber_position_m)` |
| Relative error | `|slope_measured − 1/v| / (1/v)` < 5% |
| `A_peak` | must remain ≈ 206 across all three runs |

### Connection to papers
- F-K filter in papers: filters out velocities outside the range [2.5, 25] m/s.
  This scenario directly tests that the simulator produces velocities within this range.
- Nissan & Nissan 2023: F-K keeps v ∈ [2.5, 25] m/s → all three runs fall within range.

---

## Scenario 3 — Fiber Distance Variation Only (Amplitude Decay)

### Purpose
Validate that amplitude decay follows A = W/(r+d₀)² exactly as a function of perpendicular fiber distance h.

### Isolated variable
`fiber_offset_m` — four separate runs.

### Scenario definition
| Run | fiber_offset_m (h) | Expected A_peak |
|---|---|---|
| A | 0.5 m | 1500/(0.5+0.7)² ≈ 1042 |
| B | 2.0 m | 1500/(2.0+0.7)² ≈ 206 |
| C | 5.0 m | 1500/(5.0+0.7)² ≈ 46 |
| D | 10.0 m | 1500/(10.0+0.7)² ≈ 13 |

> Constant speed 13.9 m/s, weight=1500, straight 200 m road.

### Expected output
- **A_peak ratios across runs**: D : C : B : A = 1 : 3.5 : 15.8 : 80 (from the formula).
- **SNR**: Run A very high, Run D SNR < 1 → noise dominates.
- **Waterfall slope**: must not change (v is constant).

### Metrics to check
| Metric | Calculation |
|---|---|
| A_peak ratio | max(snr × σ_trace) per run |
| Formula fit | regression: `log(A_peak) ~ −2·log(h+0.7)` — coefficient must be −2 |
| SNR threshold | Run D: SNR < 1 → does DAS still detect the vehicle? |

### Connection to papers
This is the **central test** of the Flamant-Boussinesq model:
- Hen et al. eq. (1): `u_x ∝ F/r²` → after differentiation: strain-rate ∝ W/(r+d₀)²
- Verifying the −2 exponent confirms cylindrical quasi-static wave decay rather than spherical (which would give −3/2).

---

## Scenario 4 — Mass Variation Only

### Purpose
Validate that amplitude scales linearly with W, analogous to load F in the Flamant-Boussinesq formula.

### Isolated variable
`weight_kg` — three separate runs. All other parameters fixed.

### Scenario definition
| Run | weight_kg | Expected A_peak | Expected SNR |
|---|---|---|---|
| A | 600 (motorcycle) | 600/7.29 ≈ 82 | 2.7 |
| B | 1500 (car) | 1500/7.29 ≈ 206 | 6.9 |
| C | 12000 (bus) | 12000/7.29 ≈ 1645 | 54.8 |

> h=2.0, d0=0.7, v=13.9, traffic_level="light"

### Expected output
- **A_peak ratio**: C : B : A = 8 : 1 : 0.4 (= 12000/1500 = 8, 600/1500 = 0.4)
- **Waterfall slope**: identical across all three runs (v is constant)
- **Motorcycle (Run A)**: SNR ≈ 2.7 — weak but detectable signal
- **Bus (Run C)**: SNR ≈ 55 — strong signal, visible even at larger distances

### Metrics to check
| Metric | Calculation |
|---|---|
| SNR ratio | `SNR_C / SNR_B ≈ 8`, `SNR_A / SNR_B ≈ 0.4` |
| Linearity | regression: `log(A_peak) ~ log(W)` — coefficient must be exactly 1 |

### Connection to papers
- Hen et al. eq. (1): `u_x ∝ F` — point load → strain is linear in F.
- Hen et al. 2023 demonstrates car vs. bus classification based on signal amplitude — this scenario validates that such discrimination is possible.

---

## Scenario 5 — Camera Only (DAS disabled)

### Purpose
Validate the camera noise model in isolation: `σ_cam = σ₀ + k_r · r + k_θ · |θ|`.

### Isolated variable
Vehicle distance from camera `r`.

### Scenario definition
- Camera at road start, FOV 70°, range 220 m
- Single vehicle passing along the road
- DAS completely disabled (remove from JSON)
- GPS disabled

### Expected output
When plotting `σ_cam_observed` as a function of `r`:
- Data points should fall on the line: `y = 0.10 + 0.00164·r + 0.15·|θ|`
- Mean residual vs. formula: < 0.01 m

### Metrics to check
| Metric | Calculation |
|---|---|
| `σ_residual` | `mean(σ_m − σ_expected)` ≈ 0 |
| `R²` of fit | ≥ 0.99 |
| `confidence` | verify camera stops detecting beyond range_m |

### Connection to papers
- Paper 3 (Kalman fusion): camera error model with range and detection-confidence dependence.
  ⚠️ The simulator is linear in r; Paper 3 presents quadratic dependence (σ² ∝ x²).
  This scenario lets you see whether the difference is significant in practice.

---

## Scenario 6 — DAS Only (Camera disabled)

### Purpose
Validate what DAS can provide **alone** — without fusion — and establish its accuracy limits.

### Scenario definition
- Single vehicle, speed 13.9 m/s, weight=1500
- DAS active on full segment
- Camera and GPS fully disabled

### Expected output
- **Along fiber (x-axis)**: `σ_x ≈ k/√SNR = 2/√6.9 ≈ 0.76 m`
- **Perpendicular to fiber (y-axis)**: DAS does not measure y directly — it projects to the nearest fiber point.
  y_measured = fiber_offset_m = 2.0 m always.
  y_true ≈ 0 m (road centerline).
  → **Systematic bias: 2.0 m** — this is expected and correct behavior, not a failure.

### Metrics to check
| Metric | What is measured | Expected value |
|---|---|---|
| `RMSE_fiber` | error along fiber | ≈ 0.76–1.0 m |
| `bias_y` | constant offset y_measured − y_true | ≈ 2.0 m (known and expected) |
| `slope_error` | velocity estimate error from DAS | < 10% |

> **Important**: DAS RMSE_y is **not** a quality metric — it is a result of the geometry. Do not use it as a failure indicator.

### Connection to papers
- Nissan & Nissan 2023: fiber location calibration with accuracy < 5 m.
  The simulator assumes fiber position is known — does not add uncertainty for that.
- σ_DAS = k/√SNR — defined in SENSOR_ERROR_FORMULAS_HE.md.

---

## Scenario 7 — Kalman Fusion: Camera FOV Transition

### Purpose
Validate that fusion outperforms each sensor individually, and that tracking is maintained after leaving camera FOV.

### Scenario definition
- Straight road 200 m
- Camera: range_m = 90 m only (covers x = [0, 90])
- DAS: full segment (x = [0, 200])
- Vehicle: v = 13.9 m/s, weight = 1500
- Measurement: pos_err_m from Kalman in both zones

### Three sub-scenarios for comparison

| Sub | Active sensors | What is tested |
|---|---|---|
| 7A | Camera only (range=90) | Camera performance within FOV |
| 7B | DAS only | DAS performance across full 200 m |
| 7C | Camera + DAS (fusion) | Does fusion < min(7A, 7B) in each zone? |

### Expected output

| Zone | Camera only (7A) | DAS only (7B) | Fusion (7C) |
|---|---|---|---|
| x ∈ [0, 90] | MAE < 0.5 m | MAE ≈ 0.76 m | MAE < 0.5 m |
| x ∈ [90, 200] | ❌ no data | MAE ≈ 0.76–1.5 m | MAE ≤ 1.5 m |

- **Paper 3**: validates tracking up to 100 m beyond FOV with MAE < 1.5 m.
- No dramatic improvement expected in x=[0,90]: camera is already precise there. Fusion benefit is primarily in x=[90,200].

### Metrics to check
| Metric | Calculation |
|---|---|
| `MAE_near (x<90)` | mean(pos_err_m) for rows with fiber_position_m < 90 |
| `MAE_far (x>90)` | mean(pos_err_m) for rows with fiber_position_m > 90 |
| `sources_dist` | percentage of rows: fiber / camera / camera+fiber |
| `transition_jump` | is there a sudden spike in pos_err_m at x=90? |

---

---

# Part B — Internal Consistency vs. Physical Fidelity

---

## Internal Consistency of the Simulator

> Question: **Does the code do what it claims to do?**
> No comparison to the real world is needed — only consistency with the equations defined in `sim_core.py`.

| Check | Scenario | What it proves |
|---|---|---|
| Waterfall slope = v | 1, 2 | DAS computes position correctly from vehicle.s |
| A_peak = W/(r+d₀)² | 1, 3, 4 | Amplitude formula is correctly implemented |
| σ_cam = σ₀ + k_r·r + k_θ·θ | 5 | Camera noise model is correctly implemented |
| σ_DAS = k/√SNR | 6 | DAS uncertainty formula is correctly implemented |
| Kalman reduces error | 7 | Filter converges and basic fusion works |
| y_DAS = fiber_offset_m | 6 | Fiber geometry is correctly implemented |

**Possible conclusion from internal validation**: "The simulator consistently implements the equations defined within it."

---

## Physical Fidelity to the Papers

> Question: **How faithfully does the model represent real DAS physics?**
> Requires comparison to papers and realistic reference values.

| Aspect | What the simulator does | What the papers say | Gap / note |
|---|---|---|---|
| Decay model | A = W/(r+d₀)² | Flamant-Boussinesq: u_x ∝ F/r² → strain ∝ W/r² | Good near-field approximation; not exact DAS |
| Frequency content | not filtered | Quasi-static waves < 1 Hz (Hen et al.) | Simulator does not implement F-K filter |
| DAS noise | traffic_level-based | SNR depends on soil conditions, fiber type, depth | Not modelled: coupling, depth, soil type |
| Kalman performance | MAE ≈ 1.44 m | Paper 3: MAE < 1.5 m beyond FOV | ✅ meets threshold |
| Camera error model | linear in r | Paper 3: quadratic in r, weighted by confidence | Minor gap at long ranges |
| Maximum speed | 100 km/h | F-K filter: 72–90 km/h | Simulator allows speeds outside paper range |
| Fiber orientation | fixed | In practice: coupling angle, curvature | Not modelled |

### What can be claimed professionally
> "The simulator implements a simplified Flamant-Boussinesq model and enables testing of sensor fusion algorithms. It is sufficient for validating Kalman filter performance under controlled conditions (Paper 3 MAE threshold ✅), but does not implement F-K filtering, frequency-dependent content, or realistic fiber coupling characteristics — and therefore raw DAS signal from the simulator cannot be directly compared to signal from a real DAS interrogator."

---

## What Is Still Missing for Full Validation

| Missing element | What is needed to test it |
|---|---|
| F-K filter response | Run FFT/FK on DAS output and compare to Fig. 2 in Hen et al. |
| Frequency content | Prove signal energy is < 1 Hz (quasi-static) |
| Ground coupling factor | Requires experimental data — not available in simulator |
| Multi-vehicle separability | Test minimum separation distance for two distinct signatures |
| Opposite-direction travel | Does the simulator produce a negative waterfall slope? |
| GPS accuracy | GPS not included in any scenario yet — not validated |
