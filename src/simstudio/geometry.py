"""2-D polyline geometry helpers for SimStudio.

Provides distance, projection, tangent, and offset operations on piecewise-
linear curves (polylines).  All functions treat coordinates as plain Python
floats; no NumPy dependency.

Coordinate convention: (x, y) pairs in a right-handed 2-D plane, metres.
Arc-length *s* is measured from the first point of a polyline.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from .utils import clamp  # single definition; avoids duplication across modules

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Primitive distance / segment helpers
# ---------------------------------------------------------------------------

def dist(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_point_on_segment(p: Point, a: Point, b: Point) -> Tuple[Point, float]:
    """Return the nearest point on segment AB to point P and the parameter t ∈ [0, 1].

    When AB is degenerate (|AB| < ε), returns (A, 0.0).
    """
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = (bx - ax, by - ay)
    wx, wy = (px - ax, py - ay)
    vv = vx * vx + vy * vy
    if vv < 1e-9:
        return a, 0.0
    t = clamp((wx * vx + wy * vy) / vv, 0.0, 1.0)
    return (ax + t * vx, ay + t * vy), t


# ---------------------------------------------------------------------------
# Polyline length and arc-length queries
# ---------------------------------------------------------------------------

def polyline_length(poly: List[Point]) -> float:
    """Total arc length of a polyline.

    Returns 0.0 for empty or single-point inputs.
    """
    if len(poly) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(poly)):
        total += dist(poly[i - 1], poly[i])
    return total


def point_at_s(poly: List[Point], s: float) -> Point:
    """Return the world-space point at arc-length *s* along *poly*.

    Clamps to the endpoints for s < 0 or s > polyline_length(poly).
    Returns (0, 0) for an empty polyline.
    """
    if not poly:
        return (0.0, 0.0)
    if len(poly) == 1:
        return poly[0]
    rem = max(0.0, float(s))  # clamp negative arc-lengths to the start
    for i in range(1, len(poly)):
        a, b = poly[i - 1], poly[i]
        seg = dist(a, b)
        if seg < 1e-9:
            continue
        if rem <= seg:
            t = rem / seg
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        rem -= seg
    return poly[-1]


def polyline_nearest_s(poly: List[Point], p: Point) -> Tuple[float, Point, float]:
    """Project point *p* onto *poly* and return (arc_length_s, nearest_point, distance).

    Returns (0.0, first_point_or_origin, 1e9) for degenerate polylines (fewer
    than 2 points).  The returned distance is always a non-negative float.
    """
    if len(poly) < 2:
        fallback_pt: Point = poly[0] if poly else (0.0, 0.0)
        return 0.0, fallback_pt, 1e9

    best_d = 1e18
    best_s = 0.0
    best_pt: Point = poly[0]
    acc = 0.0
    for i in range(1, len(poly)):
        a, b = poly[i - 1], poly[i]
        q, t = nearest_point_on_segment(p, a, b)
        d = dist(p, q)
        if d < best_d:
            best_d = d
            best_pt = q
            best_s = acc + t * dist(a, b)
        acc += dist(a, b)
    return best_s, best_pt, best_d


# ---------------------------------------------------------------------------
# Tangent vectors
# ---------------------------------------------------------------------------

def pose_with_lateral(poly: List[Point], s: float, lateral_offset_m: float) -> Point:
    """Return the world-space position on *poly* at arc-length *s* displaced
    by *lateral_offset_m* along the left-hand normal of the tangent.

    Used by the simulator to place a vehicle whose lane-lateral position
    deviates from the centerline (Ornstein-Uhlenbeck drift, weaving,
    straddling, …) and by the GUI to render the same pose live — keep both
    consumers in sync by calling this helper.
    """
    x_center, y_center = point_at_s(poly, s)
    tx, ty = tangent_at_s(poly, s)
    # Left-hand normal of the unit tangent (consistent with _pose_with_lateral
    # in sim_core: nx = -sin(heading), ny = cos(heading) where tangent is
    # (cos(heading), sin(heading))).
    nx, ny = (-ty, tx)
    return (x_center + lateral_offset_m * nx, y_center + lateral_offset_m * ny)


def tangent_at_s(poly: List[Point], s: float) -> Point:
    """Return a unit tangent vector of *poly* at arc-length *s*.

    Falls back to (1, 0) when the polyline is too short to determine a direction.
    """
    if len(poly) < 2:
        return (1.0, 0.0)
    L = polyline_length(poly)
    s1 = clamp(float(s), 0.0, L)
    eps = 1.0
    s0 = clamp(s1 - eps, 0.0, L)
    s2 = clamp(s1 + eps, 0.0, L)
    p0 = point_at_s(poly, s0)
    p2 = point_at_s(poly, s2)
    vx, vy = (p2[0] - p0[0], p2[1] - p0[1])
    n = math.hypot(vx, vy)
    if n < 1e-9:
        # Degenerate interval: fall back to the first non-degenerate segment.
        for i in range(1, len(poly)):
            a, b = poly[i - 1], poly[i]
            vx, vy = (b[0] - a[0], b[1] - a[1])
            n = math.hypot(vx, vy)
            if n >= 1e-9:
                return (vx / n, vy / n)
        return (1.0, 0.0)
    return (vx / n, vy / n)


# ---------------------------------------------------------------------------
# Rotation and field-of-view helpers
# ---------------------------------------------------------------------------

def rotate(vx: float, vy: float, ang: float) -> Tuple[float, float]:
    """Rotate vector (vx, vy) by *ang* radians counter-clockwise."""
    c = math.cos(ang)
    s = math.sin(ang)
    return (c * vx - s * vy, s * vx + c * vy)


def fov_wedge(center: Point, heading_rad: float, fov_rad: float, rng: float) -> List[Point]:
    """Return a 3-point wedge polygon representing a camera field of view.

    The wedge spans *fov_rad* around *heading_rad* out to range *rng*.
    """
    vx, vy = math.cos(heading_rad), math.sin(heading_rad)
    lx, ly = rotate(vx, vy, +fov_rad / 2.0)
    rx, ry = rotate(vx, vy, -fov_rad / 2.0)
    return [
        center,
        (center[0] + lx * rng, center[1] + ly * rng),
        (center[0] + rx * rng, center[1] + ry * rng),
    ]


# ---------------------------------------------------------------------------
# Polyline offsetting  (for curved lane centerlines)
# ---------------------------------------------------------------------------

def _unit_normal(a: Point, b: Point) -> Point:
    """Left-perpendicular unit normal of the directed segment A→B."""
    dx, dy = (b[0] - a[0], b[1] - a[1])
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return (0.0, 0.0)
    return (-dy / L, dx / L)


def offset_polyline(poly: List[Point], offset_m: float) -> List[Point]:
    """Return *poly* shifted laterally by *offset_m* metres (mitered joins).

    A positive offset is to the **left** of the polyline direction.  This is
    used to construct individual lane centrelines from a segment centreline.

    For a single-point polyline the same point is returned unchanged (the
    shift direction is undefined).  For an empty polyline an empty list is
    returned.
    """
    if not poly:
        return []
    if len(poly) == 1:
        return [poly[0]]

    seg_n = [_unit_normal(poly[i], poly[i + 1]) for i in range(len(poly) - 1)]

    out: List[Point] = []
    for i, p in enumerate(poly):
        if i == 0:
            nx, ny = seg_n[0]
        elif i == len(poly) - 1:
            nx, ny = seg_n[-1]
        else:
            nx1, ny1 = seg_n[i - 1]
            nx2, ny2 = seg_n[i]
            nx, ny = (nx1 + nx2, ny1 + ny2)
            L = math.hypot(nx, ny)
            if L < 1e-9:
                nx, ny = seg_n[i]  # fall back to outgoing segment normal
            else:
                nx, ny = (nx / L, ny / L)
        out.append((p[0] + nx * offset_m, p[1] + ny * offset_m))
    return out
