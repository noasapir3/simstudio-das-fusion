"""
anomaly_model — SimStudio Anomaly Detection Research Module
============================================================
Read-only post-processing module. Never imports from simstudio.*

The implementation modules live under ``anomaly_model/scripts/``:
    scripts/feature_extractor.py    feature extraction (ScenarioLoader, extract_features)
    scripts/cusum_detector.py       temporal CUSUM detector
    scripts/make_features_excel.py  Excel export helper + FEATURE_CATALOG_RICH

For backwards compatibility the same modules are re-exported at the top level,
so both of these continue to work:
    from anomaly_model.feature_extractor import ScenarioLoader      # shim
    from anomaly_model.scripts.feature_extractor import ScenarioLoader  # canonical
"""
__version__ = "0.4.0"
__phase__ = "Phase 4 — Anomaly Scoring"
