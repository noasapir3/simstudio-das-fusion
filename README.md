# SimStudio — Fiber-Optic Sensing for Smart Cities

**IoT-enabled fiber-optic sensing for real-time vehicle tracking and road-anomaly detection**

*Final Engineering Project No. 3221 · School of Electrical Engineering, Tel Aviv University · 2025–2026*

**Students:** Noa Sapir · Nitai Dobrecki  |  **Supervisor:** Khen Cohen

---

## What is this?

Modern cities are laced with thousands of kilometers of telecommunication optical fiber. This project asks: **can that existing fiber act as a distributed urban traffic sensor?**

SimStudio is a complete desktop simulation, sensor-fusion, and anomaly-detection environment for fiber-optic **Distributed Acoustic Sensing (DAS)**. It imports real road networks from OpenStreetMap, drives physically grounded vehicles along them, synthesizes three sensing modalities — DAS strain on a buried fiber, camera detections, and GPS fixes — and fuses them into vehicle tracks with a Kalman filter. A read-only analytics layer then flags irregular driving behavior (speeding, hard braking, stalls, collisions) without any labeled training data.

## Headline results

| Target | Goal | Achieved |
|---|---|---|
| Cross-street tracking accuracy | ≥ 90% | **Sub-meter RMSE** (0.35 m mean, 0.49 m showcase) |
| Anomaly detection | ≥ 5 / 10 | **6 of 9** injected anomalies detected |
| Position tolerance | ≤ ±5 m | Max error **1.52 m** |
| Real-world validation | — | Klausner St. bus-stop event tracked at **0.23 m RMSE from DAS alone** |

## How it works

```
Simulation World ──► Sensing (DAS · Camera · GPS) ──► Event Bus ──► Track Manager
                                                                        │
        Anomaly Model (166 features · Z-score · CUSUM) ◄── Kalman Fusion (6-state CA, SNR-driven R)
                                                                        │
                                          Outputs: trajectories · XLSX · audit reports · plots
```

Key design decisions:

- **Event-driven pipeline** — simulator, tracker, fusion, and analytics are loosely coupled stages on a shared event bus; each is independently testable and replaceable.
- **Physics-grounded noise** — every sensor's uncertainty is derived from its physics (Flamant–Boussinesq strain amplitude, SNR-dependent σ), so the Kalman filter automatically trusts whichever sensor is currently most reliable.
- **Strict ground-truth separation** — the anomaly model consumes only sensor-derived data; ground truth is used solely for evaluation, so the same pipeline runs unchanged on recorded real DAS data.

## Repository structure

```
├── simstudio/          # Simulation engine, sensor models, event bus,
│                       #   track manager, Kalman fusion, export & audit, GUI editor
├── anomaly_model/      # Feature extractor (166 features), CUSUM detector,
│                       #   scenario scorer (Z-score / Isolation Forest / Random Forest)
├── maps/               # OpenStreetMap networks (Florentin, North Tel Aviv) + sensor layouts
├── scenarios/          # ~150 normal-traffic scenarios + 9 injected-anomaly scenarios
├── docs/               # Physical-models reference, validation protocol,
│                       #   anomaly-interpretation guide, changelog
├── tests/              # Test suite: simulator core, Kalman filter, tracker, export
└── results/            # Trajectories, Excel workbooks, audit reports, figures
```

## Quick start

```bash
git clone https://github.com/noasapir3/simstudio-das-fusion.git
cd simstudio-das-fusion
pip install -e .
python -m simstudio            # launch the desktop editor
```

Run a scenario end-to-end (simulate → track → fuse → export → score):

```bash
python -m simstudio.run scenarios/<scenario>.json
python -m anomaly_model.score results/<run>/
```

## Real-world validation

The pipeline was exercised on a **real DAS + camera recording** from Klausner Street, beside the Tel Aviv University campus. A bus-stop event reconstructed from the recording was tracked at 0.23 m RMSE from 245 DAS measurements alone — with the filter coasting on prediction for 49% of the run during the dwell — supporting the fidelity of the simulator's physical models. See Section 5.6 of the [project report](docs/Final_Project_Report.pdf).

## References

Built on DAS traffic-monitoring research at Tel Aviv University, including Cohen, Hen & Lellouch, *"A fiber-optic traffic monitoring network trained with video inputs,"* Scientific Reports 15:14928 (2025), [arXiv:2412.12743](https://arxiv.org/abs/2412.12743).

## Citation

```bibtex
@misc{sapir2026simstudio,
  title  = {SimStudio: IoT-Enabled Fiber-Optic Sensing for Real-Time
            Vehicle Tracking and Road-Anomaly Detection},
  author = {Sapir, Noa and Dobrecki, Nitai},
  year   = {2026},
  note   = {Final Engineering Project No. 3221, School of Electrical
            Engineering, Tel Aviv University. Supervisor: Khen Cohen}
}
```
