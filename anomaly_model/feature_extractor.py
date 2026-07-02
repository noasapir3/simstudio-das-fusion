"""Compatibility shim.

The real implementation lives in ``anomaly_model/scripts/feature_extractor.py``.
This module re-exports it so that ``anomaly_model.feature_extractor`` keeps
working after the scripts were consolidated under ``scripts/``.
"""
import sys as _sys

from anomaly_model.scripts import feature_extractor as _impl

# Make `anomaly_model.feature_extractor` resolve to the real module object,
# so every attribute (ScenarioLoader, extract_features, FEATURE_CATALOG,
# events_to_feature_dataframe, _read_track_sheet_from_xlsx, ...) is available.
_sys.modules[__name__] = _impl
