# North Tel Aviv — the normal-baseline campaign

This folder holds the batch campaign that the anomaly model's **normal baseline** was
learned from: 149 map regions tiled out of the north Tel Aviv road network, each with its
own scenario definitions and, on the working machine, the full export of every simulation
run on it.

The folder name `tiles_north_tlv_south/` is historical. It is not a set of background map
images — the "tiles" are geographic regions, and each one is a small self-contained
simulation world.

## What a region folder looks like

```
tiles_north_tlv_south/045/
├── region_info.json                 # bounding box, street count, tiling metadata
├── region_045.sim.json              # the world: nodes, segments, lanes, speed limits
├── region_045_clean.sim.json        # same world after the geometry-cleaning pass
├── region_045_sensors.sim.json      # the world with DAS / camera / GPS sensors placed
├── region_045_overview.png          # map preview of the region
└── simulations/
    ├── normal_medium_traffic/
    │   ├── region_045_normal_medium_traffic.sim.json          # the scenario (vehicles)
    │   └── ..._export_<YYYYMMDD_HHMMSS>_tracking_audit/       # one run's output
    ├── normal_heavy_traffic/
    └── normal_das_cam_only/ ...
```

## What is in this repository, and what is not

| content | files | size | tracked |
|---|---|---|---|
| `region_*.sim.json`, `region_*_sensors.sim.json`, `region_info.json` — scenario definitions | 2,368 | 65 MB | yes |
| `region_*_overview.png` — region previews | 148 | 15 MB | yes |
| `tracking_audit_report.docx` — per-run audit reports | 1,783 | 96 MB | no |
| `tracking_audit.xlsx` — per-run raw exports | 5,356 | 5.1 GB | no |
| `track_*.png` — per-track trajectory and x-y plots | 29,117 | 5.0 GB | no |

The full campaign is 11 GB across 38,703 files. GitHub rejects single files above 100 MB
and is not built for repositories of that size, so the per-run output stays out. What is
tracked is the part that makes it reproducible: **every scenario definition**. The
exports are a deterministic function of those definitions plus the pipeline, so anyone
who clones this repository can rebuild them.

## How to obtain the measurements

### 1. Re-run the campaign

The batch runner replays a region, a range, or the whole campaign. For each `.sim.json`
it builds a fresh `World`, a new `EventBus` and a new `TrackManager` — nothing is shared
between runs, so data cannot bleed across scenarios.

```bash
python scripts/run_all_simulations.py all --dry-run   # list what would run, run nothing
python scripts/run_all_simulations.py 045             # one region
python scripts/run_all_simulations.py 001-020         # a range
python scripts/run_all_simulations.py 010,025,047     # a comma list
python scripts/run_all_simulations.py all             # the whole campaign (hours, ~11 GB)
```

Each run stops when every vehicle has left the map, with a safety cap of 600 s of
simulated time (`--max-duration`). The output lands next to its scenario as
`<scenario>_export_<YYYYMMDD_HHMMSS>_tracking_audit/`, containing:

| file | what it holds |
|---|---|
| `tracking_audit.xlsx` | one sheet per track: time, true position, each sensor's reading, the Kalman estimate, its σ, the pre-update residuals and the innovation variances |
| `coverage_summary.xlsx` | per-sensor detections, misses and acceptance rates |
| `kalman_measurement_audit.xlsx` | every measurement offered to the filter and what happened to it |
| `tracking_audit_report.docx` | the same run written up as a readable report |
| `track_trajectory_*.png`, `track_xy_*.png` | per-track plots |

### 2. Extract the features

The extractor reduces each track to one row of 166 columns — the behavioural fingerprint
the model scores:

```bash
python anomaly_model/scripts/feature_extractor.py \
    maps/north_TLV/tiles_north_tlv_south/045/simulations/*/*_tracking_audit \
    --output anomaly_model/outputs/features_region_045.csv

python anomaly_model/scripts/feature_extractor.py --catalog   # what every column means
```

### 3. Score against the trained model

```bash
python anomaly_model/scripts/scenario_scorer.py \
    --features anomaly_model/outputs/features_region_045.csv
```

It prints `track_id`, `z_score_max`, `flagged` and `top_feature`, and writes the same as a
CSV. The scorer loads `anomaly_model/outputs/model_live.pkl`, needs only numpy and pandas,
and never imports the simulator — the separation that keeps the reported results free of
ground-truth leakage.

## Measurements that are already committed

You do not have to re-run anything to inspect the results the report and the presentation
cite:

- `anomaly_model/simulations/` — the nine evaluation scenarios with their injected
  anomalies, complete with exports, plots and audit reports.
- `anomaly_model/outputs/` — the extracted feature tables, the learned baseline and the
  trained model.
- `maps/klausner/` — the real-world Klausner Street recording: the bus event replayed
  through the unchanged pipeline, its exports, its features and the anomaly report.
