"""Compatibility shim.

The real implementation lives in ``anomaly_model/scripts/cusum_detector.py``.
This module re-exports it so that ``anomaly_model.cusum_detector`` keeps
working after the scripts were consolidated under ``scripts/``.
"""
import sys as _sys

from anomaly_model.scripts import cusum_detector as _impl

_sys.modules[__name__] = _impl
