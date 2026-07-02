"""Beautiful Excel export for SimStudio simulation results.

Produces a styled workbook with:
  • Dashboard   – key metrics + RMSE bar chart
  • DAS         – data table + space-time scatter chart (fiber position vs time)
  • Kalman      – data table + position-error timeline chart
  • GPS/Cameras – styled data tables
  • Vehicles/Summary/Issues – styled data tables
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import BarChart, LineChart, ScatterChart, Reference
    from openpyxl.chart.series import SeriesLabel
    from openpyxl.chart.label import DataLabelList
    from openpyxl.utils import get_column_letter
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

# ── Colour palette (matches SimStudio dark UI) ────────────────────────────────
_BG_DARK   = "0D1117"
_BG_MID    = "161B22"
_BG_CARD   = "1C2128"
_ACCENT1   = "58A6FF"   # blue
_ACCENT2   = "3FB950"   # green
_ACCENT3   = "F78166"   # red
_ACCENT4   = "D2A8FF"   # purple
_ACCENT5   = "FFA657"   # orange
_TEXT_PRI  = "E6EDF3"
_TEXT_SEC  = "8B949E"
_BORDER_C  = "30363D"

# Light-theme palette for data sheets (easier to read)
_HDR_FILL  = "1F4E79"   # dark blue header
_ROW_EVEN  = "EBF3FB"
_ROW_ODD   = "FFFFFF"
_HDR_TXT   = "FFFFFF"
_FONT      = "Arial"


def _solid(hex_color: str) -> "PatternFill":
    return PatternFill("solid", fgColor=hex_color)


def _thin_border() -> "Border":
    s = Side(style="thin", color=_BORDER_C)
    return Border(left=s, right=s, top=s, bottom=s)


def _hdr_font(sz: int = 10) -> "Font":
    return Font(name=_FONT, bold=True, color=_HDR_TXT, size=sz)


def _body_font(sz: int = 10, color: str = "000000") -> "Font":
    return Font(name=_FONT, size=sz, color=color)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _write_table(
    ws: Any,
    start_row: int,
    headers: List[str],
    rows: List[List[Any]],
    col_widths: Optional[List[int]] = None,
    freeze: bool = True,
) -> int:
    """Write a styled table starting at (start_row, 1).  Returns next free row."""
    # Header
    for ci, h in enumerate(headers, 1):
        c = ws.cell(start_row, ci, h)
        c.font  = _hdr_font()
        c.fill  = _solid(_HDR_FILL)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = _thin_border()
    if freeze:
        ws.freeze_panes = ws.cell(start_row + 1, 1)

    # Data
    for ri, row in enumerate(rows):
        bg = _ROW_EVEN if ri % 2 == 0 else _ROW_ODD
        for ci, val in enumerate(row, 1):
            c = ws.cell(start_row + 1 + ri, ci, val)
            c.font   = _body_font()
            c.fill   = _solid(bg)
            c.border = _thin_border()
            c.alignment = Alignment(horizontal="center", vertical="center")

    # Column widths
    if col_widths:
        for ci, w in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(ci)].width = w
    else:
        for ci, h in enumerate(headers, 1):
            best = len(str(h)) + 2
            for row in rows:
                if ci - 1 < len(row):
                    best = max(best, len(str(row[ci - 1])) + 2)
            ws.column_dimensions[get_column_letter(ci)].width = min(30, best)

    ws.row_dimensions[start_row].height = 22
    return start_row + 1 + len(rows)


def _section_title(ws: Any, row: int, text: str, color: str = _ACCENT1) -> None:
    ws.row_dimensions[row].height = 24
    c = ws.cell(row, 1, text)
    c.font = Font(name=_FONT, bold=True, size=13, color=color)
    c.fill = _solid(_BG_DARK)


# ── Data extraction from Treeview ─────────────────────────────────────────────

def _tv_to_data(tv: Any) -> Tuple[List[str], List[List[Any]]]:
    """Extract (headers, rows) from a tkinter Treeview."""
    headers = list(tv["columns"])
    rows: List[List[Any]] = []
    for iid in tv.get_children():
        vals = list(tv.item(iid, "values"))
        # Try to convert numeric strings
        converted = []
        for v in vals:
            try:
                converted.append(float(v) if "." in str(v) else int(v))
            except (ValueError, TypeError):
                converted.append(v)
        rows.append(converted)
    return headers, rows


# ── Individual sheet builders ─────────────────────────────────────────────────

def _build_generic(wb: Any, title: str, tv: Any, tab_color: str = "2E75B6") -> None:
    ws = wb.create_sheet(title=title)
    ws.sheet_properties.tabColor = tab_color
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows(min_row=1, max_row=3, min_col=1, max_col=20):
        for cell in row:
            cell.fill = _solid(_BG_DARK)
    _section_title(ws, 1, title)
    ws.row_dimensions[2].height = 6
    headers, rows = _tv_to_data(tv)
    _write_table(ws, 3, headers, rows)


def _build_rmse(wb: Any, tv_rmse: Any) -> None:
    ws = wb.create_sheet(title="RMSE")
    ws.sheet_properties.tabColor = _ACCENT2
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows(min_row=1, max_row=50, min_col=1, max_col=12):
        for cell in row:
            cell.fill = _solid(_BG_DARK)

    _section_title(ws, 1, "RMSE — Sensor Performance Comparison", _ACCENT2)
    ws.row_dimensions[2].height = 6

    headers, rows = _tv_to_data(tv_rmse)
    next_row = _write_table(ws, 3, headers, rows, freeze=False)

    # ── Bar chart: RMSE_pos per source ───────────────────────────────────────
    if rows:
        try:
            rmse_col = headers.index("rmse_pos_m") + 1
        except ValueError:
            rmse_col = len(headers)  # last col fallback
        src_col = headers.index("source") + 1 if "source" in headers else 1

        chart = BarChart()
        chart.type  = "col"
        chart.title = "Position RMSE by Sensor Source"
        chart.style = 10
        chart.y_axis.title = "RMSE (m)"
        chart.x_axis.title = "Sensor"
        chart.height = 12
        chart.width  = 18

        data_ref = Reference(ws, min_col=rmse_col, max_col=rmse_col,
                             min_row=3, max_row=3 + len(rows))
        cats_ref = Reference(ws, min_col=src_col,  max_col=src_col,
                             min_row=4, max_row=3 + len(rows))
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.series[0].graphicalProperties.solidFill     = _ACCENT1
        chart.series[0].graphicalProperties.ln.solidFill  = _ACCENT1
        chart.dataLabels = DataLabelList()
        chart.dataLabels.showVal = True

        ws.add_chart(chart, f"A{next_row + 2}")


def _build_das(wb: Any, tv_das: Any) -> None:
    ws = wb.create_sheet(title="DAS")
    ws.sheet_properties.tabColor = _ACCENT5
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows(min_row=1, max_row=4, min_col=1, max_col=20):
        for cell in row:
            cell.fill = _solid(_BG_DARK)

    _section_title(ws, 1, "DAS — Distributed Acoustic Sensing Measurements", _ACCENT5)
    ws.row_dimensions[2].height = 6

    headers, rows = _tv_to_data(tv_das)
    next_row = _write_table(ws, 3, headers, rows)

    if not rows:
        return

    # ── Space-time scatter chart: fiber_position_m (Y) vs t (X) ─────────────
    # Helper data goes in far-right columns (hidden from normal view)
    # so the chart appears right below the main data, not at row ~1600
    try:
        t_col   = headers.index("t") + 1
        pos_col = headers.index("fiber_position_m") + 1
    except ValueError:
        return

    # Write helper columns far to the right (columns 20, 21)
    hcol_t   = 20   # column T
    hcol_pos = 21   # column U
    ws.cell(3, hcol_t,   "t (s)")
    ws.cell(3, hcol_pos, "fiber_pos (m)")
    for ri, row in enumerate(rows):
        ws.cell(4 + ri, hcol_t,   row[t_col - 1]   if len(row) > t_col - 1   else "")
        ws.cell(4 + ri, hcol_pos, row[pos_col - 1] if len(row) > pos_col - 1 else "")
    helper_end = 4 + len(rows) - 1

    # Hide helper columns so they don't clutter the view
    ws.column_dimensions[get_column_letter(hcol_t)].hidden   = True
    ws.column_dimensions[get_column_letter(hcol_pos)].hidden = True

    chart = ScatterChart()
    chart.title  = "DAS Space-Time Map  (fiber position vs time)"
    chart.style  = 10
    chart.y_axis.title = "Fiber Position (m)"
    chart.x_axis.title = "Time (s)"
    chart.height = 14
    chart.width  = 22

    x_ref = Reference(ws, min_col=hcol_t,   max_col=hcol_t,
                      min_row=4, max_row=helper_end)
    y_ref = Reference(ws, min_col=hcol_pos, max_col=hcol_pos,
                      min_row=3, max_row=helper_end)
    from openpyxl.chart import Series
    series = Series(y_ref, x_ref, title="Vehicle detections")
    series.marker.symbol   = "circle"
    series.marker.size     = 3
    series.marker.graphicalProperties.solidFill    = _ACCENT5
    series.marker.graphicalProperties.ln.solidFill = _ACCENT5
    series.graphicalProperties.ln.noFill = True   # no connecting line
    chart.series.append(series)

    # Place chart right below the main data table
    ws.add_chart(chart, f"A{next_row + 2}")


def _build_kalman(wb: Any, tv_kalman: Any) -> None:
    ws = wb.create_sheet(title="Kalman")
    ws.sheet_properties.tabColor = _ACCENT4
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows(min_row=1, max_row=4, min_col=1, max_col=20):
        for cell in row:
            cell.fill = _solid(_BG_DARK)

    _section_title(ws, 1, "Kalman Filter — Fusion Estimates", _ACCENT4)
    ws.row_dimensions[2].height = 6

    headers, rows = _tv_to_data(tv_kalman)
    next_row = _write_table(ws, 3, headers, rows)

    if not rows:
        return

    # ── Line chart: pos_err_m and sigma_pos_m vs t ───────────────────────────
    try:
        t_col     = headers.index("t") + 1
        err_col   = headers.index("pos_err_m") + 1
        sigma_col = headers.index("sigma_pos_m") + 1
    except ValueError:
        return

    # Helper table for chart (t | pos_err_m | sigma_pos_m)
    hstart = next_row + 2
    ws.cell(hstart, 1, "t (s)")
    ws.cell(hstart, 2, "pos_err_m")
    ws.cell(hstart, 3, "sigma_pos_m")
    for ri, row in enumerate(rows):
        ws.cell(hstart + 1 + ri, 1, row[t_col - 1]     if len(row) > t_col - 1     else "")
        ws.cell(hstart + 1 + ri, 2, row[err_col - 1]   if len(row) > err_col - 1   else "")
        ws.cell(hstart + 1 + ri, 3, row[sigma_col - 1] if len(row) > sigma_col - 1 else "")
    hend = hstart + len(rows)

    chart = LineChart()
    chart.title  = "Kalman Filter — Position Error vs Self-Reported σ"
    chart.style  = 10
    chart.y_axis.title = "Position (m)"
    chart.x_axis.title = "Time (s)"
    chart.height = 13
    chart.width  = 22

    cats = Reference(ws, min_col=1, max_col=1, min_row=hstart + 1, max_row=hend)

    d_err   = Reference(ws, min_col=2, max_col=2, min_row=hstart, max_row=hend)
    d_sigma = Reference(ws, min_col=3, max_col=3, min_row=hstart, max_row=hend)

    chart.add_data(d_err,   titles_from_data=True)
    chart.add_data(d_sigma, titles_from_data=True)
    chart.set_categories(cats)

    chart.series[0].graphicalProperties.line.solidFill = _ACCENT3
    chart.series[0].graphicalProperties.line.width     = 15000
    chart.series[1].graphicalProperties.line.solidFill = _ACCENT1
    chart.series[1].graphicalProperties.line.width     = 10000
    chart.series[1].graphicalProperties.line.dashDot   = "dash"

    # Place chart below the helper table
    chart_row = hend + 2
    ws.add_chart(chart, f"A{chart_row}")


def _build_tracks(wb: Any, tv_tracks: Any) -> None:
    """Phase 6: Tracker events stream — one row per per-update tracker emit.

    Schema follows the GUI ``tv_tracks`` Treeview exactly (including the
    ``vehicle_id_oracle`` column renamed in Phase 6).  Kept intentionally
    simple — a styled data table with no chart — because the Tracks tab
    is a diagnostic firehose, not a summary metric.
    """
    ws = wb.create_sheet(title="Tracks")
    ws.sheet_properties.tabColor = _ACCENT4
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows(min_row=1, max_row=4, min_col=1, max_col=20):
        for cell in row:
            cell.fill = _solid(_BG_DARK)

    _section_title(ws, 1, "Tracker Events — per-Update Stream", _ACCENT4)
    ws.row_dimensions[2].height = 6

    headers, rows = _tv_to_data(tv_tracks)
    _write_table(ws, 3, headers, rows)


def _build_diag(wb: Any, tv_diag_summary: Any, tv_diag: Any) -> None:
    """Phase 6: Tracker diagnostics — manager summary + per-track table.

    Two stacked tables on one sheet, mirroring the GUI Diagnostics tab:
    a small ``(key, value)`` manager summary on top, then the per-track
    diagnostic table below (schema follows
    ``TrackManager.DIAGNOSTIC_COLUMNS``).
    """
    ws = wb.create_sheet(title="Diagnostics")
    ws.sheet_properties.tabColor = _ACCENT2
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows(min_row=1, max_row=4, min_col=1, max_col=20):
        for cell in row:
            cell.fill = _solid(_BG_DARK)

    _section_title(ws, 1, "Tracker Diagnostics — Manager Summary & Per-Track", _ACCENT2)
    ws.row_dimensions[2].height = 6

    # Top: manager summary (small key/value block).
    sum_headers, sum_rows = _tv_to_data(tv_diag_summary)
    next_row = _write_table(ws, 3, sum_headers, sum_rows, freeze=False)

    # Spacer row between the two tables.
    ws.row_dimensions[next_row].height = 10
    ws.cell(next_row, 1).fill = _solid(_BG_DARK)

    # Bottom: per-track table.
    diag_headers, diag_rows = _tv_to_data(tv_diag)
    _write_table(ws, next_row + 1, diag_headers, diag_rows)


def _build_dashboard(wb: Any, tv_rmse: Any, tv_summary: Any) -> None:
    """First sheet: key metrics table + RMSE bar chart."""
    ws = wb.create_sheet(title="Dashboard", index=0)
    ws.sheet_properties.tabColor = _ACCENT1
    ws.sheet_view.showGridLines  = False

    for row in ws.iter_rows(min_row=1, max_row=80, min_col=1, max_col=16):
        for cell in row:
            cell.fill = _solid(_BG_DARK)

    # Title
    ws.row_dimensions[1].height = 8
    ws.row_dimensions[2].height = 36
    ws.row_dimensions[3].height = 18
    ws.row_dimensions[4].height = 10

    ws.merge_cells("B2:I2")
    c = ws.cell(2, 2, "SimStudio — Fiber Optic Traffic Lab")
    c.font      = Font(name=_FONT, bold=True, size=20, color=_ACCENT1)
    c.fill      = _solid(_BG_DARK)
    c.alignment = Alignment(horizontal="left", vertical="center")

    ws.merge_cells("B3:I3")
    c = ws.cell(3, 2, "Simulation Export — Sensor Fusion Results")
    c.font      = Font(name=_FONT, size=11, color=_TEXT_SEC)
    c.fill      = _solid(_BG_DARK)
    c.alignment = Alignment(horizontal="left", vertical="center")

    # ── KPI cards from Summary ───────────────────────────────────────────────
    _, sum_rows = _tv_to_data(tv_summary)
    summary_map = {}
    for row in sum_rows:
        if len(row) >= 2:
            summary_map[str(row[0])] = row[1]

    kpi_data = [
        ("TOTAL EVENTS",    summary_map.get("events_total",     "—"), _ACCENT1),
        ("VEHICLE STATES",  summary_map.get("world.vehicle_state","—"), _ACCENT2),
        ("DAS READINGS",    summary_map.get("sensor.das",       "—"), _ACCENT5),
        ("CAMERA FRAMES",   summary_map.get("sensor.camera",    "—"), _ACCENT4),
    ]

    def _kpi_card(ws, row, col_s, col_e, label, value, color):
        ws.merge_cells(start_row=row,   start_column=col_s, end_row=row,   end_column=col_e)
        ws.merge_cells(start_row=row+1, start_column=col_s, end_row=row+1, end_column=col_e)
        ws.merge_cells(start_row=row+2, start_column=col_s, end_row=row+2, end_column=col_e)
        ws.row_dimensions[row].height   = 16
        ws.row_dimensions[row+1].height = 34
        ws.row_dimensions[row+2].height = 10
        for r in range(row, row+3):
            for cc in range(col_s, col_e+1):
                ws.cell(r, cc).fill = _solid(_BG_CARD)
        lc = ws.cell(row,   col_s, label)
        lc.font = Font(name=_FONT, size=8, color=_TEXT_SEC)
        lc.alignment = Alignment(horizontal="center", vertical="center")
        vc = ws.cell(row+1, col_s, value)
        vc.font = Font(name=_FONT, bold=True, size=22, color=color)
        vc.alignment = Alignment(horizontal="center", vertical="center")

    col_pairs = [(2,3),(4,5),(6,7),(8,9)]
    for (col_s, col_e), (label, val, color) in zip(col_pairs, kpi_data):
        _kpi_card(ws, 5, col_s, col_e, label, val, color)

    # ── RMSE table ───────────────────────────────────────────────────────────
    ws.row_dimensions[9].height = 8
    ws.row_dimensions[10].height = 20
    c = ws.cell(10, 2, "Sensor RMSE Comparison")
    c.font = Font(name=_FONT, bold=True, size=12, color=_ACCENT1)
    c.fill = _solid(_BG_DARK)
    ws.row_dimensions[11].height = 6

    rmse_headers, rmse_rows = _tv_to_data(tv_rmse)
    _write_table(ws, 12, rmse_headers, rmse_rows, freeze=False)

    # ── RMSE bar chart ───────────────────────────────────────────────────────
    if rmse_rows:
        try:
            rmse_col = rmse_headers.index("rmse_pos_m") + 1
            src_col  = rmse_headers.index("source") + 1
        except ValueError:
            rmse_col, src_col = len(rmse_headers), 1

        chart = BarChart()
        chart.type  = "col"
        chart.title = "Position RMSE by Sensor"
        chart.style = 10
        chart.y_axis.title = "RMSE (m)"
        chart.x_axis.title = "Sensor Source"
        chart.height = 12
        chart.width  = 14

        data_ref = Reference(ws, min_col=rmse_col, max_col=rmse_col,
                             min_row=12, max_row=12 + len(rmse_rows))
        cats_ref = Reference(ws, min_col=src_col,  max_col=src_col,
                             min_row=13, max_row=12 + len(rmse_rows))
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        bar_colors = [_ACCENT2, _ACCENT3, _ACCENT5, _ACCENT4]
        for i, s in enumerate(chart.series):
            s.graphicalProperties.solidFill    = bar_colors[i % len(bar_colors)]
            s.graphicalProperties.ln.solidFill = bar_colors[i % len(bar_colors)]
        chart.dataLabels = DataLabelList()
        chart.dataLabels.showVal = True

        ws.add_chart(chart, "K11")

    # Column widths for dashboard
    for col, w in [("A",2),("B",16),("C",16),("D",16),("E",16),
                   ("F",16),("G",16),("H",16),("I",16),("J",2)]:
        ws.column_dimensions[col].width = w


# ── Public entry point ────────────────────────────────────────────────────────

def export_workbook(
    out_path: "Path",
    tv_summary: Any,
    tv_veh: Any,
    tv_gps: Any,
    tv_cam: Any,
    tv_das: Any,
    tv_kalman: Any,
    tv_rmse: Any,
    tv_anom: Any,
    tv_tracks: Any = None,
    tv_diag: Any = None,
    tv_diag_summary: Any = None,
) -> None:
    """Build and save the styled Excel workbook to *out_path*.

    Phase 6: ``tv_tracks``, ``tv_diag`` and ``tv_diag_summary`` are optional
    kwargs.  When all three are provided, two additional sheets ("Tracks"
    and "Diagnostics") are appended after "Issues".  Older callers that
    pass only the original nine treeviews continue to work unchanged — the
    new sheets are simply omitted.
    """
    if not _AVAILABLE:
        raise ImportError("openpyxl is required for Excel export.")

    wb = Workbook()
    wb.remove(wb.active)   # remove default sheet

    # Build each sheet
    _build_dashboard(wb, tv_rmse, tv_summary)
    _build_generic(wb, "Summary",  tv_summary, tab_color="404040")
    _build_generic(wb, "Vehicles", tv_veh,     tab_color="7030A0")
    _build_generic(wb, "GPS",      tv_gps,     tab_color="2E75B6")
    _build_generic(wb, "Cameras",  tv_cam,     tab_color="538135")
    _build_das(wb, tv_das)
    _build_kalman(wb, tv_kalman)
    _build_rmse(wb, tv_rmse)
    _build_generic(wb, "Issues",   tv_anom,    tab_color="C00000")

    # Phase 6: append tracker sheets after Issues when treeviews are supplied.
    # The conditional keeps the sheet indices of all pre-existing sheets
    # stable, so downstream openpyxl readers keyed by name or position
    # continue to work.
    if tv_tracks is not None:
        _build_tracks(wb, tv_tracks)
    if tv_diag is not None and tv_diag_summary is not None:
        _build_diag(wb, tv_diag_summary, tv_diag)

    wb.save(out_path)
