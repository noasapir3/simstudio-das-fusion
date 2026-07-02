"""Tests for simstudio.geometry — polyline math primitives."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simstudio.geometry import (
    dist,
    nearest_point_on_segment,
    offset_polyline,
    point_at_s,
    polyline_length,
    polyline_nearest_s,
    tangent_at_s,
)


# ---------------------------------------------------------------------------
# dist
# ---------------------------------------------------------------------------

def test_dist_zero():
    assert dist((0.0, 0.0), (0.0, 0.0)) == 0.0


def test_dist_horizontal():
    assert abs(dist((0.0, 0.0), (3.0, 0.0)) - 3.0) < 1e-9


def test_dist_diagonal():
    assert abs(dist((0.0, 0.0), (3.0, 4.0)) - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# polyline_length
# ---------------------------------------------------------------------------

def test_polyline_length_empty():
    assert polyline_length([]) == 0.0


def test_polyline_length_single():
    assert polyline_length([(1.0, 2.0)]) == 0.0


def test_polyline_length_straight():
    poly = [(0.0, 0.0), (100.0, 0.0)]
    assert abs(polyline_length(poly) - 100.0) < 1e-9


def test_polyline_length_l_shape():
    poly = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
    assert abs(polyline_length(poly) - 7.0) < 1e-9


# ---------------------------------------------------------------------------
# point_at_s
# ---------------------------------------------------------------------------

def test_point_at_s_empty():
    assert point_at_s([], 10.0) == (0.0, 0.0)


def test_point_at_s_start():
    poly = [(0.0, 0.0), (100.0, 0.0)]
    pt = point_at_s(poly, 0.0)
    assert abs(pt[0]) < 1e-9 and abs(pt[1]) < 1e-9


def test_point_at_s_end():
    poly = [(0.0, 0.0), (100.0, 0.0)]
    pt = point_at_s(poly, 100.0)
    assert abs(pt[0] - 100.0) < 1e-9


def test_point_at_s_midpoint():
    poly = [(0.0, 0.0), (100.0, 0.0)]
    pt = point_at_s(poly, 50.0)
    assert abs(pt[0] - 50.0) < 1e-9 and abs(pt[1]) < 1e-9


def test_point_at_s_clamps_below():
    poly = [(10.0, 0.0), (20.0, 0.0)]
    pt = point_at_s(poly, -5.0)
    assert abs(pt[0] - 10.0) < 1e-9


def test_point_at_s_clamps_above():
    poly = [(0.0, 0.0), (50.0, 0.0)]
    pt = point_at_s(poly, 999.0)
    assert abs(pt[0] - 50.0) < 1e-9


# ---------------------------------------------------------------------------
# polyline_nearest_s
# ---------------------------------------------------------------------------

def test_polyline_nearest_s_empty():
    s, pt, d = polyline_nearest_s([], (5.0, 5.0))
    assert pt == (0.0, 0.0) and d == 1e9


def test_polyline_nearest_s_single_point():
    s, pt, d = polyline_nearest_s([(3.0, 4.0)], (0.0, 0.0))
    assert pt == (3.0, 4.0) and d == 1e9


def test_polyline_nearest_s_perpendicular():
    poly = [(0.0, 0.0), (100.0, 0.0)]
    s, pt, d = polyline_nearest_s(poly, (50.0, 10.0))
    assert abs(s - 50.0) < 1e-6
    assert abs(d - 10.0) < 1e-6


def test_polyline_nearest_s_past_end():
    poly = [(0.0, 0.0), (10.0, 0.0)]
    s, pt, d = polyline_nearest_s(poly, (20.0, 0.0))
    assert abs(pt[0] - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# offset_polyline
# ---------------------------------------------------------------------------

def test_offset_polyline_empty():
    assert offset_polyline([], 5.0) == []


def test_offset_polyline_single():
    result = offset_polyline([(1.0, 2.0)], 3.0)
    assert result == [(1.0, 2.0)]


def test_offset_polyline_horizontal_positive():
    poly = [(0.0, 0.0), (10.0, 0.0)]
    result = offset_polyline(poly, 2.0)
    # Positive offset = left = +y for a rightward polyline
    assert len(result) == 2
    assert abs(result[0][1] - 2.0) < 1e-9
    assert abs(result[1][1] - 2.0) < 1e-9


def test_offset_polyline_horizontal_negative():
    poly = [(0.0, 0.0), (10.0, 0.0)]
    result = offset_polyline(poly, -2.0)
    assert abs(result[0][1] - (-2.0)) < 1e-9


def test_offset_polyline_preserves_length():
    poly = [(0.0, 0.0), (10.0, 0.0)]
    result = offset_polyline(poly, 3.0)
    assert abs(polyline_length(result) - polyline_length(poly)) < 1e-6


# ---------------------------------------------------------------------------
# tangent_at_s
# ---------------------------------------------------------------------------

def test_tangent_at_s_degenerate():
    tx, ty = tangent_at_s([], 0.0)
    assert (tx, ty) == (1.0, 0.0)


def test_tangent_at_s_horizontal():
    poly = [(0.0, 0.0), (100.0, 0.0)]
    tx, ty = tangent_at_s(poly, 50.0)
    assert abs(tx - 1.0) < 1e-6 and abs(ty) < 1e-6


def test_tangent_at_s_vertical():
    poly = [(0.0, 0.0), (0.0, 100.0)]
    tx, ty = tangent_at_s(poly, 50.0)
    assert abs(tx) < 1e-6 and abs(ty - 1.0) < 1e-6


def test_tangent_is_unit_vector():
    poly = [(0.0, 0.0), (3.0, 4.0), (10.0, 10.0)]
    for s in [0.0, 2.5, 5.0, 8.0]:
        tx, ty = tangent_at_s(poly, s)
        assert abs(math.hypot(tx, ty) - 1.0) < 1e-6, f"Not unit at s={s}"


# ---------------------------------------------------------------------------
# nearest_point_on_segment
# ---------------------------------------------------------------------------

def test_nearest_point_degenerate_segment():
    pt, t = nearest_point_on_segment((3.0, 3.0), (1.0, 1.0), (1.0, 1.0))
    assert pt == (1.0, 1.0) and t == 0.0


def test_nearest_point_midpoint():
    pt, t = nearest_point_on_segment((5.0, 5.0), (0.0, 0.0), (10.0, 0.0))
    assert abs(pt[0] - 5.0) < 1e-9 and abs(pt[1]) < 1e-9
    assert abs(t - 0.5) < 1e-9


def test_nearest_point_clamps_to_start():
    pt, t = nearest_point_on_segment((-10.0, 0.0), (0.0, 0.0), (10.0, 0.0))
    assert abs(pt[0]) < 1e-9 and t == 0.0


def test_nearest_point_clamps_to_end():
    pt, t = nearest_point_on_segment((20.0, 0.0), (0.0, 0.0), (10.0, 0.0))
    assert abs(pt[0] - 10.0) < 1e-9 and t == 1.0


if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
