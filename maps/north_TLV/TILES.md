# `tiles_north_tlv_south/` — what it holds and why it is not in the repository

The folder `maps/north_TLV/tiles_north_tlv_south/` on the working machine is **11 GB**
across 38,703 files. Despite the folder name it is not a set of background images: it
holds the **normal-baseline batch campaign** — 149 map regions of north Tel Aviv, each
with its scenario definitions and the full export of every simulation run on it. This is
the campaign the anomaly model's normal baseline was learned from.

## What is in it

| content | files | size |
|---|---|---|
| `region_*.sim.json`, `region_*_sensors.sim.json`, `region_info.json` — scenario definitions | 2,368 | 65 MB |
| `region_*_overview.png` — one map preview per region | 148 | 15 MB |
| `tracking_audit_report.docx` — per-run audit reports | 1,783 | 96 MB |
| `tracking_audit.xlsx` — per-run raw exports | 5,356 | 5.1 GB |
| `track_*.png` — per-track trajectory and x-y plots | 29,117 | 5.0 GB |

## Why it is not pushed in full

GitHub rejects any single file above 100 MB and is not intended for repositories of this
size, so a full push of 11 GB would fail. The heavy part is per-run output that the
pipeline regenerates deterministically from the scenario definitions.

## How to regenerate it

The scenario definitions are the reproducible input. From the repository root:

```bash
python scripts/run_all_simulations.py        # replays the campaign region by region
python scripts/tile_north_tlv.py             # rebuilds the region tiling from OSM
```

The evaluation runs that the report and the presentation actually cite are committed in
full under `anomaly_model/simulations/` (193 files) and `anomaly_model/outputs/`
(53 files), and the Klausner real-world recording with its exports is under
`maps/klausner/`.
