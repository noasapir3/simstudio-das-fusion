#!/usr/bin/env python3
"""Plot a DAS space–time heatmap (like in the paper) from SimStudio exports.

This script reads the exported `das.csv` file (created via the GUI: Export All…)
which contains rows with columns:
  t, sensor_id, fiber_distance_m, lateral_offset_m, vehicle_weight_kg, das_amplitude, snr

It then builds a 2D matrix:
  y-axis: sensor_id (sorted)
  x-axis: time (t, sorted)
  value : das_amplitude (default) or snr

Usage examples:
  python3 scripts/plot_das_map.py --export_root ~/Downloads --latest
  python3 scripts/plot_das_map.py --csv ~/Downloads/simstudio_export_2026-03-06_0212/das.csv

Outputs a PNG next to the csv by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _find_latest_export(export_root: Path) -> Path:
    # folders look like: simstudio_export_YYYY-MM-DD_HHMMSS
    candidates = [p for p in export_root.iterdir() if p.is_dir() and p.name.startswith("simstudio_export_")]
    if not candidates:
        raise FileNotFoundError(f"No simstudio_export_* folder found under: {export_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Path to das.csv")
    ap.add_argument("--export_root", type=str, default=None, help="Folder that contains simstudio_export_* folders")
    ap.add_argument("--latest", action="store_true", help="Use latest simstudio_export_* under --export_root")
    ap.add_argument("--value", choices=["das_amplitude", "snr"], default="das_amplitude")
    ap.add_argument("--out", type=str, default=None, help="Output PNG path (default: next to csv)")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    csv_path: Path
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
    else:
        if not args.export_root or not args.latest:
            raise SystemExit("Provide either --csv PATH, or (--export_root PATH --latest).")
        export_root = Path(args.export_root).expanduser().resolve()
        latest_dir = _find_latest_export(export_root)
        csv_path = latest_dir / "das.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"das.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Defensive: tolerate different column order/casing
    required = {"t", "sensor_id", args.value}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}. Found: {list(df.columns)}")

    # Ensure stable sorting
    df = df.sort_values(["sensor_id", "t"])  # type: ignore

    # Pivot into matrix
    mat = df.pivot_table(index="sensor_id", columns="t", values=args.value, aggfunc="mean")

    # Fill gaps with 0 (no event). If you prefer NaN, remove this line.
    mat = mat.fillna(0.0)

    out_path = Path(args.out).expanduser().resolve() if args.out else csv_path.with_suffix(".png")

    plt.figure()
    plt.imshow(mat.values, aspect="auto", origin="lower")
    plt.xlabel("time (t)")
    plt.ylabel("sensor_id")
    plt.title(f"DAS Map ({args.value})")
    cbar = plt.colorbar()
    cbar.set_label(args.value)

    # Optional: nicer y ticks for few sensors
    if mat.shape[0] <= 20:
        plt.yticks(range(mat.shape[0]), [str(i) for i in mat.index])

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
