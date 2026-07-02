#!/usr/bin/env python3
"""
scenario_scorer.py — score vehicle tracks against the trained anomaly model.

Reproduces the Z-Score-Max logic used by the SimStudio GUI "Anomaly Intel" tab
as a standalone CLI. A track is flagged anomalous when its maximum |z-score|
across all model features exceeds the trained threshold (stored in the model).

Needs only numpy + pandas (+ openpyxl for --export-dir). It loads the pure-numpy
model ``anomaly_model/outputs/model_live.pkl`` and never imports the simulator.

USAGE
-----
Score an already-extracted features.csv:
    python anomaly_model/scripts/scenario_scorer.py \
        --features anomaly_model/simulations/region_120/anomaly_collision/features.csv

Score straight from a SimStudio export folder (extracts features first):
    python anomaly_model/scripts/scenario_scorer.py \
        --export-dir <scenario>_export_..._tracking_audit \
        --scenario-json <scenario>.sim.json

Output: a printed table + a scores CSV (default: <input>_scores.csv) with
columns track_id, z_score_max, flagged, top_feature.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# anomaly_model/scripts/ -> repo root ; make `anomaly_model` importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = REPO_ROOT / "anomaly_model" / "outputs" / "model_live.pkl"


def load_model(model_path: Path) -> dict:
    with open(model_path, "rb") as fh:
        raw = pickle.load(fh)
    if int(raw.get("format_version", 1)) < 2:
        sys.exit(f"ERROR: {model_path.name} is an old (v1/sklearn) model. "
                 "Use model_live.pkl (format v2).")
    mean = np.asarray(raw["scaler_mean"], dtype=float)
    return {
        "feature_cols": list(raw["feature_cols"]),
        "mean": mean,
        "std": np.asarray(raw["scaler_std"], dtype=float),
        "median": np.asarray(raw.get("feature_medians", mean), dtype=float),
        "threshold": float(raw.get("zscore_threshold", 15.4)),
        "n_train": int(raw.get("n_train", 0)),
    }


def score_dataframe(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    fc = model["feature_cols"]
    mean, std, median = model["mean"], model["std"], model["median"]
    thr = model["threshold"]

    X = np.full((len(df), len(fc)), np.nan)
    for i, col in enumerate(fc):
        if col in df.columns:
            X[:, i] = pd.to_numeric(df[col], errors="coerce").values

    fill = np.where(np.isnan(median), 0.0, median)
    X = np.where(np.isnan(X), fill, X)
    std_safe = np.where(std < 1e-9, 1e-9, std)
    Xz = (X - mean) / std_safe

    z_abs = np.abs(Xz)
    z_max = z_abs.max(axis=1)
    top_idx = z_abs.argmax(axis=1)

    id_col = next((c for c in ("global_track_id", "track_id", "vehicle_id_oracle",
                               "scenario_id") if c in df.columns), None)
    ids = df[id_col].astype(str).values if id_col else [str(i) for i in range(len(df))]

    return pd.DataFrame({
        "track_id": ids,
        "z_score_max": np.round(z_max, 3),
        "flagged": z_max > thr,
        "top_feature": [fc[j] for j in top_idx],
    })


def features_from_export(export_dir: Path, scenario_json: Path | None) -> pd.DataFrame:
    from anomaly_model.feature_extractor import ScenarioLoader
    loader = ScenarioLoader(
        export_dir,
        scenario_id=export_dir.name,
        scenario_json=str(scenario_json) if scenario_json else None,
    )
    return loader.extract()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score SimStudio vehicle tracks against the trained anomaly model.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", type=Path, help="An already-extracted features.csv")
    src.add_argument("--export-dir", type=Path, help="A SimStudio export folder")
    ap.add_argument("--scenario-json", type=Path, default=None,
                    help="Scenario .sim.json (recommended with --export-dir)")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help=f"Model file (default: {DEFAULT_MODEL})")
    ap.add_argument("--out", type=Path, default=None, help="Output scores CSV")
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"ERROR: model not found: {args.model}")
    model = load_model(args.model)

    if args.features:
        if not args.features.exists():
            sys.exit(f"ERROR: features file not found: {args.features}")
        df = pd.read_csv(args.features)
        default_out = args.features.with_name(args.features.stem + "_scores.csv")
    else:
        if not args.export_dir.exists():
            sys.exit(f"ERROR: export dir not found: {args.export_dir}")
        df = features_from_export(args.export_dir, args.scenario_json)
        default_out = args.export_dir.parent / f"{args.export_dir.name}_scores.csv"

    scores = score_dataframe(df, model)
    out_path = args.out or default_out
    scores.to_csv(out_path, index=False)

    n_flag = int(scores["flagged"].sum())
    print(f"\nModel: Z-Score-Max  |  features: {len(model['feature_cols'])}  |  "
          f"threshold: {model['threshold']:.2f} sigma  |  "
          f"trained on {model['n_train']} vehicles")
    print(f"Scored {len(scores)} track(s) — {n_flag} flagged as anomalous:\n")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(scores.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
