"""Compatibility shim.

The real implementation lives in ``anomaly_model/scripts/make_features_excel.py``.
This module re-exports it so that ``anomaly_model.make_features_excel`` keeps
working after the scripts were consolidated under ``scripts/``.
"""
import sys as _sys

from anomaly_model.scripts import make_features_excel as _impl

_sys.modules[__name__] = _impl
