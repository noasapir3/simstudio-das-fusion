"""Phase 6 tests for simstudio.export_xlsx.

These tests focus on the Phase 6 changes:

* The Kalman sheet uses ``vehicle_id_oracle`` as its per-vehicle column
  name (propagated from the GUI Treeview through ``_tv_to_data``).
* The new ``_build_tracks`` / ``_build_diag`` builders render cleanly.
* ``export_workbook`` accepts the new kwargs as optional (``None`` by
  default) so pre-existing callers keep working.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# openpyxl is an optional dependency — skip this whole module when absent.
pytest.importorskip("openpyxl")

from openpyxl import load_workbook

from simstudio import export_xlsx
from simstudio.export_xlsx import (
    _tv_to_data,
    _build_tracks,
    _build_diag,
    export_workbook,
)


# ---------------------------------------------------------------------------
# Minimal "fake Treeview" that mimics the subset of the tkinter API used by
# ``_tv_to_data``.  Identical in shape to ``gui.app._FakeTv`` but local to the
# test so we do not have to import tkinter.
# ---------------------------------------------------------------------------

class _FakeTv:
    def __init__(self, columns, rows):
        self._columns = tuple(columns)
        self._rows = [tuple(r) for r in rows]

    # Treeview-compatible bits used by _tv_to_data / _write_table.
    def __getitem__(self, key):
        if key == "columns":
            return self._columns
        raise KeyError(key)

    def get_children(self):
        return list(range(len(self._rows)))

    def item(self, iid, what):
        assert what == "values"
        return self._rows[iid]


# ---------------------------------------------------------------------------
# _tv_to_data — Phase 6 rename propagates
# ---------------------------------------------------------------------------

def test_tv_to_data_preserves_vehicle_id_oracle_header():
    """Kalman tab headers (Phase 6 rename + Phase 7 gid) survive the
    Treeview → rows trip, and values align with the declared columns."""
    tv = _FakeTv(
        columns=(
            "t", "global_track_id", "vehicle_id_oracle", "sources",
            "x_hat", "y_hat", "vx_hat", "vy_hat", "ax_hat", "ay_hat",
            "sigma_pos_m", "sigma_vel_mps", "pos_err_m",
        ),
        rows=[
            ("0.10", "g0", "v0", "gps,cam", "1.0", "2.0", "0.5", "0.0",
             "0.0", "0.0", "0.50", "0.10", "0.05"),
        ],
    )
    headers, rows = _tv_to_data(tv)
    assert "vehicle_id_oracle" in headers
    assert "global_track_id" in headers
    assert "vehicle_id" not in headers
    # Phase 7: gid sits between t and vehicle_id_oracle.
    assert headers.index("global_track_id")   == 1
    assert headers.index("vehicle_id_oracle") == 2
    assert rows[0][1] == "g0"
    assert rows[0][2] == "v0"


# ---------------------------------------------------------------------------
# export_workbook — back-compat and optional kwargs
# ---------------------------------------------------------------------------

def _make_min_treeviews():
    """Return the 9 required treeviews with the bare minimum valid schemas."""
    tv_summary = _FakeTv(("key", "value"), [("events_total", 1)])
    tv_veh     = _FakeTv(
        ("t", "vehicle_id", "lane_id", "x", "y", "v",
         "heading_rad", "a_long_mps2", "ax_world_mps2", "ay_world_mps2"),
        [],
    )
    tv_gps     = _FakeTv(
        ("t", "sensor_id", "vehicle_id", "x", "y",
         "speed_mps", "a_long_mps2", "sigma_m", "confidence"),
        [],
    )
    tv_cam     = _FakeTv(
        ("t", "sensor_id", "vehicle_id", "x", "y",
         "speed_mps", "a_long_mps2", "sigma_m", "confidence"),
        [],
    )
    tv_das     = _FakeTv(
        ("t", "sensor_id", "vehicle_id", "x", "y",
         "fiber_position_m", "speed_mps", "fiber_angle_rad", "snr"),
        [],
    )
    tv_kalman  = _FakeTv(
        ("t", "global_track_id", "vehicle_id_oracle", "sources",
         "x_hat", "y_hat", "vx_hat", "vy_hat", "ax_hat", "ay_hat",
         "sigma_pos_m", "sigma_vel_mps", "pos_err_m"),
        [],
    )
    tv_rmse    = _FakeTv(
        ("source", "samples", "rmse_x_m", "rmse_y_m", "rmse_pos_m", "notes"),
        [],
    )
    tv_anom    = _FakeTv(("t", "type", "details"), [])
    return dict(
        tv_summary=tv_summary, tv_veh=tv_veh, tv_gps=tv_gps,
        tv_cam=tv_cam, tv_das=tv_das, tv_kalman=tv_kalman,
        tv_rmse=tv_rmse, tv_anom=tv_anom,
    )


def test_export_workbook_backcompat_without_new_kwargs(tmp_path):
    """Legacy signature — only 9 treeviews — must still produce a workbook.

    This guards against accidentally making the Phase 6 kwargs required.
    """
    snaps = _make_min_treeviews()
    out = tmp_path / "legacy.xlsx"
    export_workbook(out, **snaps)
    assert out.exists()

    wb = load_workbook(out)
    # Pre-existing sheets must still be present in the same relative order.
    assert wb.sheetnames[:9] == [
        "Dashboard", "Summary", "Vehicles", "GPS", "Cameras",
        "DAS", "Kalman", "RMSE", "Issues",
    ]
    # And the new Phase 6 sheets must NOT appear when kwargs are omitted.
    assert "Tracks" not in wb.sheetnames
    assert "Diagnostics" not in wb.sheetnames


def test_export_workbook_with_phase6_sheets(tmp_path):
    """When all three Phase 6 treeviews are supplied, new sheets appear."""
    snaps = _make_min_treeviews()

    snaps["tv_tracks"] = _FakeTv(
        ("t", "global_track_id", "vehicle_id_oracle", "source",
         "x", "y", "v", "segment_id", "lane_id", "tentative",
         "n_gps", "n_cam", "n_das", "n_state"),
        [("0.10", "g0", "v0", "gps", "1.0", "0.0", "10.0",
          "s0", "L0", "False", "1", "0", "0", "0")],
    )
    snaps["tv_diag_summary"] = _FakeTv(
        ("key", "value"),
        [("n_tracks_alive", 1), ("n_tracks_dead", 0)],
    )
    snaps["tv_diag"] = _FakeTv(
        ("global_track_id", "vehicle_id_oracle", "segment_id", "state",
         "t_born", "t_last", "duration_s",
         "n_gps", "n_cam", "n_das", "n_state",
         "n_segments", "segments_visited",
         "hypothesis_count", "id_switches", "n_updates_total"),
        [("g0", "v0", "s0", "confirmed",
          "0.10", "0.20", "0.10",
          "1", "0", "0", "0",
          "1", "s0",
          "1", "0", "1")],
    )

    out = tmp_path / "phase6.xlsx"
    export_workbook(out, **snaps)
    assert out.exists()

    wb = load_workbook(out)
    # Phase 6 sheets appended after Issues, preserving prior indices.
    assert "Tracks" in wb.sheetnames
    assert "Diagnostics" in wb.sheetnames
    assert wb.sheetnames.index("Tracks") > wb.sheetnames.index("Issues")
    assert wb.sheetnames.index("Diagnostics") > wb.sheetnames.index("Tracks")

    # The Kalman sheet header must use the renamed column and, since
    # Phase 7, also expose ``global_track_id`` alongside it.  We scan
    # the first few cells of row 3 (the sheet pre-fills columns 1..20
    # with the dark background, so iterating the whole row returns
    # extra blanks).
    ws_k = wb["Kalman"]
    kalman_headers = [ws_k.cell(3, c).value for c in range(1, 20)]
    assert "vehicle_id_oracle" in kalman_headers
    assert "global_track_id"   in kalman_headers
    assert "vehicle_id" not in kalman_headers

    # The Tracks sheet header must also carry the renamed oracle column.
    ws_t = wb["Tracks"]
    tracks_headers = [ws_t.cell(3, c).value for c in range(1, 20)]
    assert "vehicle_id_oracle" in tracks_headers
    assert "global_track_id" in tracks_headers

    # The Diagnostics sheet has two stacked tables; just confirm the
    # per-track header row shows up somewhere below the summary block.
    ws_d = wb["Diagnostics"]
    found_header_row = False
    for row in ws_d.iter_rows(values_only=True):
        if row and "global_track_id" in row and "vehicle_id_oracle" in row:
            found_header_row = True
            break
    assert found_header_row, "Diagnostics sheet missing per-track header"


def test_export_workbook_partial_phase6_kwargs_skips_diag(tmp_path):
    """Tracks-only (no diag pair) should still produce a valid workbook.

    This pins the current ``tv_diag is not None AND tv_diag_summary is not
    None`` gate — passing only one of them must NOT raise.
    """
    snaps = _make_min_treeviews()
    snaps["tv_tracks"] = _FakeTv(
        ("t", "global_track_id", "vehicle_id_oracle", "source",
         "x", "y", "v", "segment_id", "lane_id", "tentative",
         "n_gps", "n_cam", "n_das", "n_state"),
        [],
    )
    # tv_diag_summary intentionally omitted → Diagnostics sheet should be skipped.

    out = tmp_path / "tracks_only.xlsx"
    export_workbook(out, **snaps)
    assert out.exists()

    wb = load_workbook(out)
    assert "Tracks" in wb.sheetnames
    assert "Diagnostics" not in wb.sheetnames


# ---------------------------------------------------------------------------
# Direct helper calls — keeps the unit test close to the implementation.
# ---------------------------------------------------------------------------

def test_build_tracks_writes_schema_identical_headers():
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    cols = (
        "t", "global_track_id", "vehicle_id_oracle", "source",
        "x", "y", "v", "segment_id", "lane_id", "tentative",
        "n_gps", "n_cam", "n_das", "n_state",
    )
    tv = _FakeTv(cols, [])
    _build_tracks(wb, tv)
    ws = wb["Tracks"]
    # Only read the first len(cols) cells of row 3 — the sheet pre-fills
    # columns 1..20 with the background color so ws[3] yields extra blank
    # cells beyond the actual header width.
    headers = tuple(ws.cell(3, c + 1).value for c in range(len(cols)))
    assert headers == cols


def test_build_diag_writes_both_tables():
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)

    summary_tv = _FakeTv(
        ("key", "value"),
        [("n_tracks_alive", 2), ("n_tracks_dead", 0), ("id_switches", 0)],
    )
    diag_cols = (
        "global_track_id", "vehicle_id_oracle", "segment_id", "state",
        "t_born", "t_last", "duration_s",
        "n_gps", "n_cam", "n_das", "n_state",
        "n_segments", "segments_visited",
        "hypothesis_count", "id_switches", "n_updates_total",
    )
    diag_tv = _FakeTv(diag_cols, [])

    _build_diag(wb, summary_tv, diag_tv)
    ws = wb["Diagnostics"]

    # Summary header row is at row 3 (section title is row 1).
    assert ws.cell(3, 1).value == "key"
    assert ws.cell(3, 2).value == "value"

    # Per-track header row must appear somewhere below — locate by scan.
    found_row = None
    for r in range(4, 30):
        if ws.cell(r, 1).value == "global_track_id":
            found_row = r
            break
    assert found_row is not None, "Per-track header row not found"
    # Full column order must match the DIAGNOSTIC_COLUMNS contract.
    actual = tuple(ws.cell(found_row, c + 1).value for c in range(len(diag_cols)))
    assert actual == diag_cols
