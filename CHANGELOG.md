# Changelog

All notable user-visible changes to SimStudio are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
semantic versioning for tags.

## [0.4.1] — 2026-04-23

Surfaces the tracker's stable identity alongside the ground-truth
reference in the Kalman view.

### Added

- **`global_track_id` column in the Kalman tab and XLSX Kalman sheet.**
  The Kalman view now shows the tracker's identity (`global_track_id`)
  right before the ground-truth column (`vehicle_id_oracle`).  In the
  current (1:1) tracking phase the two values correlate directly; in
  later phases where the tracker may re-use or split identities across
  segments, this column is what lets you read the fusion output as
  "tracker's belief" rather than "oracle fact".
- Header tuple in `gui/app.py` and the row schema emitted by
  `build_kalman_rows_tracked` in `kalman.py` updated in lock-step.
- XLSX export propagates the new column automatically via
  `_tv_to_data(tv_kalman)` — no additional code in `export_xlsx.py`.

### Changed

- `build_kalman_rows_tracked` row width: 12 → 13 columns.  The new
  column is inserted at position 1 (i.e. column B), right after `t`.
  The legacy `build_kalman_rows` remains at 12 columns and byte-
  identical — any caller that needs the old schema can keep using it.
- Convergence tests in `tests/test_kalman.py` now strip the `gid`
  column from tracked rows before comparing with legacy, so bit-exact
  equivalence on Phase-1 scenes is still the contract.  A new
  assertion also checks that the tracker assigns distinct gids to
  distinct vehicles.

### Migration notes

- Downstream scripts that read the exported workbook's Kalman sheet by
  column index must shift columns C onward by one (Kalman data now
  starts at column D for `sources`).  Reading by header name is
  unaffected.
- GUI column order from left to right:
  `t, global_track_id, vehicle_id_oracle, sources, x_hat, …, pos_err_m`.

## [0.4.0] — 2026-04-20

This release completes the cross-segment / cross-sensor vehicle identity
tracking pipeline.  The tracker now sits in front of the Kalman filter
(`sensor → tracking → Kalman → GUI/export`), and the oracle vehicle id
emitted by the simulator is preserved in the UI and export strictly as
ground-truth for validation — it is no longer treated as an identity the
fused estimator has to agree with.

### Highlights

- **Tracker-driven Kalman pipeline.**  The GUI's Kalman tab is now
  populated from `build_kalman_rows_tracked`, which replays events
  through `TrackManager` before fusing.  Phase-1 behaviour (1:1
  vid ↔ global_track_id) is bit-exact with the previous output, so no
  regression in RMSE or convergence tests.
- **Oracle isolation in the UI.**  The Kalman tab's per-vehicle column
  is now labelled `vehicle_id_oracle` instead of `vehicle_id` to make
  it unambiguous that this value is ground truth from the simulator,
  not a tracker-inferred identity.
- **Tracker diagnostics are now exported.**  The Excel workbook gains
  two new sheets:
  - **Tracks** — the per-update tracker event stream (14 columns
    mirroring the GUI `Tracks` tab, including `vehicle_id_oracle`).
  - **Diagnostics** — the manager-level summary (top) plus the
    per-track diagnostic table (bottom), driven by
    `TrackManager.DIAGNOSTIC_COLUMNS` so the tracker owns the schema.
- **Back-compatible `export_workbook` signature.**  The three new
  kwargs (`tv_tracks`, `tv_diag`, `tv_diag_summary`) all default to
  `None`; callers that pass only the original nine treeviews continue
  to produce the same sheet set as before.

### Changed

- GUI: Kalman tab column header `vehicle_id` → `vehicle_id_oracle`.
  This rename propagates automatically to the Excel `Kalman` sheet
  via `_tv_to_data(tv_kalman)`, so any downstream script that reads
  the workbook by column name must be updated to expect
  `vehicle_id_oracle` in the Kalman sheet.

### Added

- `simstudio.kalman.build_kalman_rows_tracked(events, world=None,
  debug_das=False)` — the tracker-first row builder.  Re-groups the
  sensor measurements by `global_track_id`, runs the same
  `LegacyKalmanFilter` per track, and emits rows carrying the oracle
  vehicle id in the `vid` column.  Row order is deterministic and
  bit-identical to the legacy builder for Phase-1 scenes.
- `simstudio.export_xlsx._build_tracks` and
  `simstudio.export_xlsx._build_diag` — new XLSX sheet builders for
  the tracker data.  Colour-coded tabs (`Tracks` purple, `Diagnostics`
  green) match the rest of the palette.
- `tests/test_export_xlsx.py` — six new unit/smoke tests covering
  the rename, back-compat of the legacy signature, partial-kwarg
  behaviour, and schema correctness of the new builders.

### Migration notes

- Any external script reading the exported workbook's `Kalman` sheet
  by header name must now use `vehicle_id_oracle`.  The column
  position is unchanged (still index 1, i.e. column B).
- The XLSX sheet order for pre-existing sheets is unchanged
  (`Dashboard, Summary, Vehicles, GPS, Cameras, DAS, Kalman, RMSE,
  Issues, …`).  Phase 6 sheets are appended after `Issues`.
- No changes to `sim_core.py`, event payload contracts, the Kalman
  filter (`LegacyKalmanFilter`), or CSV fallback output.

## [0.3.0]

Prior release — see `README.md` for the feature set shipped before
the tracker-first pipeline.
