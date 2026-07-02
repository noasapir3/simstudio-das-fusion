"""Shared utility functions for the SimStudio simulation engine.

This module provides pure mathematical helpers and sampling routines that are
used across sim_core, randomizer, and geometry.  Keeping them here avoids the
three-way duplication that previously existed and makes the behaviour easy to
find and adjust in one place.

Nothing in this module imports from other SimStudio modules.
"""

from __future__ import annotations

import math
import random
from typing import Tuple


# ---------------------------------------------------------------------------
# Basic math helpers
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    """Return x clamped to the closed interval [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def wrap_angle(a: float) -> float:
    """Wrap an angle in radians to the half-open interval (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# ---------------------------------------------------------------------------
# Probabilistic helpers
# ---------------------------------------------------------------------------

def trunc_gauss(
    mean: float,
    std: float,
    lo: float,
    hi: float,
    max_tries: int = 30,
) -> float:
    """Truncated Gaussian sampler with a clamp fallback.

    Draws from N(mean, std²) until a value in [lo, hi] is obtained.
    After *max_tries* failed attempts the mean is returned clamped to [lo, hi].

    Args:
        mean:      Distribution mean.
        std:       Standard deviation (clamped to ≥ 1e-9 internally).
        lo:        Lower bound (inclusive).
        hi:        Upper bound (inclusive).
        max_tries: Maximum rejection-sampling attempts before falling back.

    Returns:
        A float in [lo, hi].
    """
    mean = float(mean)
    std = max(1e-9, float(std))
    lo = float(lo)
    hi = float(hi)
    for _ in range(max_tries):
        x = random.gauss(mean, std)
        if lo <= x <= hi:
            return float(x)
    return float(clamp(mean, lo, hi))


# ---------------------------------------------------------------------------
# Vehicle profile sampling  (shared by sim_core and randomizer)
# ---------------------------------------------------------------------------

def sample_vehicle_profile(speed_limit_mps: float) -> Tuple[float, float]:
    """Sample a (weight_kg, speed_mps) pair for a realistic urban vehicle mix.

    Vehicle categories and their mixing proportions:

    +------------------------+--------+------------------+
    | Category               |  Share | Weight range (kg)|
    +========================+========+==================+
    | Small / compact car    |   55%  |  800 – 1 400     |
    | Sedan / mid-size       |   25%  | 1 200 – 1 800    |
    | SUV / crossover / van  |   12%  | 1 500 – 2 800    |
    | City bus (kerb)        |    5%  | 8 000 – 14 000   |
    | Heavy truck (empty)    |    3%  | 4 500 – 15 000   |
    +------------------------+--------+------------------+

    Speed is derived from the road speed limit and bounded to [20, 100] km/h
    per project requirements.

    Args:
        speed_limit_mps: Road speed limit in m/s.  Use 0 for unknown.

    Returns:
        (weight_kg, speed_mps) as a pair of floats.
    """
    from .constants import SPEED_MIN_KMH, SPEED_MAX_KMH  # local import avoids circularity

    limit_kmh = float(speed_limit_mps) * 3.6
    mean_kmh = clamp(0.95 * limit_kmh if limit_kmh > 0 else 55.0, SPEED_MIN_KMH, SPEED_MAX_KMH)
    v_kmh = trunc_gauss(mean_kmh, 13.0, SPEED_MIN_KMH, SPEED_MAX_KMH)
    v_mps = v_kmh / 3.6

    r = random.random()
    if r < 0.55:
        w = trunc_gauss(1050.0, 120.0, 800.0, 1400.0)
    elif r < 0.80:
        w = trunc_gauss(1400.0, 140.0, 1200.0, 1800.0)
    elif r < 0.92:
        w = trunc_gauss(1850.0, 220.0, 1500.0, 2800.0)
    elif r < 0.97:
        w = trunc_gauss(10000.0, 1500.0, 8000.0, 14000.0)
    else:
        w = trunc_gauss(8500.0, 2000.0, 4500.0, 15000.0)

    return float(w), float(v_mps)
