"""
anomaly_model — SimStudio Anomaly Detection Research Module
============================================================
Read-only post-processing module.  Never imports from simstudio.*
and never modifies the simulator, GUI, Kalman filter, or tracking.

Folder layout
-------------
anomaly_model/
├── __init__.py                  ← this file (package marker)
├── feature_extractor.py         ← Phase 1: extract features from exported CSVs/Excel
├── cusum_detector.py            ← Phase 3: Page's CUSUM on NIS_x
├── make_features_excel.py       ← Excel export helper
│
├── scripts/                     ← runnable CLI entry points
│   ├── scenario_scorer.py       ← train / score the anomaly model
│   ├── run_extraction.py        ← batch feature extraction from simulation exports
│   └── run_anomaly_simulations.py  ← headless batch runner for anomaly .sim.json files
│
├── simulations/                 ← anomaly scenario data (region_XXX/ subdirectories)
├── outputs/                     ← generated artefacts (CSVs, PKLs, JSONs)
└── docs/                        ← documentation and reference data
    ├── anomaly_model_flow.md
    ├── feature_catalog.xlsx
    └── ANOMALY MODEL commands.docx

Importable modules
------------------
    from anomaly_model.feature_extractor import ScenarioLoader, extract_features
    from anomaly_model.cusum_detector    import run_cusum, batch_cusum, CUSUMDetector

CLI scripts (run from the repo root)
-------------------------------------
    python anomaly_model/scripts/scenario_scorer.py --train
    python anomaly_model/scripts/scenario_scorer.py --score --input anomaly_model/outputs/features_anomaly.csv
    python anomaly_model/scripts/run_extraction.py --anomaly
    python anomaly_model/scripts/run_anomaly_simulations.py
"""
__version__ = "0.4.0"
__phase__   = "Phase 4 — Anomaly Scoring + Reorganised Layout"
