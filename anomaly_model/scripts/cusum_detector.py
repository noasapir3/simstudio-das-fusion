"""
cusum_detector.py — Phase 3: Temporal anomaly detection via Page's CUSUM on NIS_x.

Theory
------
The Normalized Innovation Squared in the x-direction (NIS_x) is defined as

    NIS_x_t = ν_x_t² / S_xx_t

where ν_x is the pre-update x-innovation and S_xx is its marginal variance.
Under a consistent, well-tuned Kalman filter (H₀), NIS_x ~ χ²(1) with mean = 1.0.
When the filter becomes inconsistent — due to a sensor anomaly, unexpected
vehicle behaviour, or an upstream data fault — the mean of NIS_x rises above 1.0.

Page's one-sided CUSUM accumulates evidence of such a mean shift:

    S_0  = 0
    S_t  = max(0,  S_{t-1}  +  NIS_x_t  −  k)
    alarm when S_t > h

Reference parameter choices (tunable, see CUSUMDetector):
    k = 1.5   — "slack" / reference value; calibrated to detect a shift from
                 mean=1.0 (H₀) to mean≥2.0 (H₁) quickly while ignoring small
                 random excursions.  k = (μ₀ + μ₁) / 2 = (1 + 2) / 2.
    h = 5.0   — alarm threshold; gives ARL₀ ≈ 500 measurements under H₀,
                 i.e. a false alarm every ~17 s at 30 Hz (acceptably rare).

NaN handling
------------
Prediction-only timesteps (no measurement update) produce NIS_x = NaN.
The CUSUM carries S_{t-1} forward unchanged for those rows — neither
accumulating evidence nor resetting the statistic.

Gap reset
---------
After a measurement gap longer than `gap_reset_s` seconds (default 2.0 s),
the filter's covariance has grown large and past NIS_x values are no longer
comparable to the current measurement noise model.  S_t is reset to 0.

Usage
-----
    from anomaly_model.cusum_detector import CUSUMDetector, run_cusum, batch_cusum
    import pandas as pd

    # Single trajectory DataFrame (must contain columns 't' and 'NIS_x'):
    traj_df = pd.read_csv("outputs/audit/T000001_trajectory.csv")
    detector = CUSUMDetector(k=1.5, h=5.0, gap_reset_s=2.0)
    result   = detector.fit(traj_df)
    print(result.alarms)               # list[AlarmEvent]
    result.traj_df.to_csv("out.csv")   # trajectory with cusum_stat + cusum_alarm

    # Convenience wrapper for a single file path:
    result = run_cusum("outputs/audit/T000001_trajectory.csv")

    # Batch over many scenario output folders:
    summary_df = batch_cusum(["outputs/audit/folder_a", "outputs/audit/folder_b"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "AlarmEvent",
    "CUSUMResult",
    "CUSUMDetector",
    "run_cusum",
    "batch_cusum",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AlarmEvent:
    """A single contiguous alarm episode detected by the CUSUM."""

    track_id: str
    """Identifier of the Kalman track (e.g. 'T000001')."""

    onset_t: float
    """Simulation time [s] of the first alarm sample (S_t > h)."""

    offset_t: Optional[float]
    """Simulation time [s] of the last alarm sample, or None if the track ends
    while still in alarm."""

    duration_s: Optional[float]
    """Duration of the alarm episode [s], or None if open-ended."""

    peak_stat: float
    """Maximum CUSUM statistic S_t reached during this episode."""

    onset_index: int
    """Row index (in the trajectory DataFrame) of the alarm onset."""


@dataclass
class CUSUMResult:
    """Full output of a single CUSUM run."""

    track_id: str
    """Kalman track identifier."""

    traj_df: pd.DataFrame
    """Input trajectory DataFrame with two extra columns appended:
    - ``cusum_stat``  : float — the running CUSUM statistic S_t.
    - ``cusum_alarm`` : bool  — True on every sample where S_t > h.
    """

    alarms: List[AlarmEvent]
    """Detected alarm events (empty list if none)."""

    k: float
    """Reference / slack value used."""

    h: float
    """Alarm threshold used."""

    n_nis_valid: int
    """Number of timesteps with a valid (non-NaN) NIS_x value."""

    n_gap_resets: int
    """Number of times S_t was reset due to a measurement gap."""

    mean_NIS_x: Optional[float]
    """Mean NIS_x over all valid samples (H₀ expectation = 1.0)."""

    frac_NIS_exceedance: Optional[float]
    """Fraction of valid NIS_x samples exceeding 3.84 (χ²(1) 95th pct)."""


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """Page's one-sided CUSUM detector for Kalman NIS_x time-series.

    Parameters
    ----------
    k : float
        Reference (slack) value.  Samples that contribute S_t must exceed k.
        Default 1.5 targets detection of a mean shift from 1.0 to 2.0.
    h : float
        Alarm threshold.  An alarm is raised when S_t > h.
        Default 5.0 → ARL₀ ≈ 500 measurements under H₀.
    gap_reset_s : float
        If the interval between two consecutive *measurement* updates exceeds
        this value [s], S_t is reset to 0 before processing the later sample.
        Default 2.0 s.
    """

    def __init__(
        self,
        k: float = 1.5,
        h: float = 5.0,
        gap_reset_s: float = 2.0,
    ) -> None:
        self.k = float(k)
        self.h = float(h)
        self.gap_reset_s = float(gap_reset_s)

    # ------------------------------------------------------------------
    def fit(self, traj_df: pd.DataFrame, track_id: str = "unknown") -> CUSUMResult:
        """Run the CUSUM on a trajectory DataFrame and return a CUSUMResult.

        The DataFrame must contain at minimum:

        - ``t``     : simulation time in seconds (monotonically increasing).
        - ``NIS_x`` : per-step NIS_x value, NaN on prediction-only rows.

        Parameters
        ----------
        traj_df : pd.DataFrame
            Trajectory data (typically read from ``*_trajectory.csv``).
        track_id : str
            Human-readable identifier inserted into all output objects.

        Returns
        -------
        CUSUMResult
        """
        required = {"t", "NIS_x"}
        missing = required - set(traj_df.columns)
        if missing:
            raise ValueError(
                f"CUSUMDetector.fit(): trajectory DataFrame is missing columns: {missing}. "
                "Run the simulation with Phase 2 innovation export enabled."
            )

        df = traj_df.copy()
        n = len(df)

        t_arr   = df["t"].to_numpy(dtype=float)
        nis_arr = df["NIS_x"].to_numpy(dtype=float)  # NaN on predict-only rows

        cusum_stat  = np.empty(n, dtype=float)
        cusum_alarm = np.zeros(n, dtype=bool)

        S = 0.0
        last_meas_t: Optional[float] = None  # time of the last valid NIS_x sample
        n_nis_valid = 0
        n_gap_resets = 0
        nis_valid_vals: List[float] = []

        for i in range(n):
            nis_i = nis_arr[i]
            t_i   = t_arr[i]

            if math.isnan(nis_i):
                # Prediction-only: carry S forward unchanged.
                cusum_stat[i]  = S
                cusum_alarm[i] = S > self.h
                continue

            # Valid measurement update at time t_i.
            n_nis_valid += 1
            nis_valid_vals.append(nis_i)

            # Gap-reset check.
            if last_meas_t is not None:
                gap = t_i - last_meas_t
                if gap > self.gap_reset_s:
                    S = 0.0
                    n_gap_resets += 1
            last_meas_t = t_i

            # CUSUM update.
            S = max(0.0, S + nis_i - self.k)
            cusum_stat[i]  = S
            cusum_alarm[i] = S > self.h

        df["cusum_stat"]  = cusum_stat
        df["cusum_alarm"] = cusum_alarm

        # --- Aggregate NIS_x diagnostics ---
        if nis_valid_vals:
            mean_NIS = float(np.mean(nis_valid_vals))
            frac_exc = float(np.mean(np.array(nis_valid_vals) > 3.84))
        else:
            mean_NIS = None
            frac_exc = None

        # --- Extract alarm episodes ---
        alarms = _extract_alarm_events(
            t_arr=t_arr,
            alarm_arr=cusum_alarm,
            stat_arr=cusum_stat,
            track_id=track_id,
        )

        return CUSUMResult(
            track_id=track_id,
            traj_df=df,
            alarms=alarms,
            k=self.k,
            h=self.h,
            n_nis_valid=n_nis_valid,
            n_gap_resets=n_gap_resets,
            mean_NIS_x=mean_NIS,
            frac_NIS_exceedance=frac_exc,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_alarm_events(
    t_arr: np.ndarray,
    alarm_arr: np.ndarray,
    stat_arr: np.ndarray,
    track_id: str,
) -> List[AlarmEvent]:
    """Convert a boolean alarm mask into a list of contiguous AlarmEvent objects."""
    events: List[AlarmEvent] = []
    in_alarm = False
    onset_t = 0.0
    onset_idx = 0
    peak_stat = 0.0

    for i in range(len(t_arr)):
        a = bool(alarm_arr[i])
        s = float(stat_arr[i])

        if a and not in_alarm:
            # Rising edge — alarm starts.
            in_alarm  = True
            onset_t   = float(t_arr[i])
            onset_idx = i
            peak_stat = s

        elif a and in_alarm:
            # Still in alarm — update peak.
            if s > peak_stat:
                peak_stat = s

        elif not a and in_alarm:
            # Falling edge — alarm ends.
            in_alarm  = False
            off_t     = float(t_arr[i - 1])
            events.append(AlarmEvent(
                track_id   = track_id,
                onset_t    = onset_t,
                offset_t   = off_t,
                duration_s = off_t - onset_t,
                peak_stat  = peak_stat,
                onset_index= onset_idx,
            ))
            peak_stat = 0.0

    # Handle open alarm at end of track.
    if in_alarm:
        events.append(AlarmEvent(
            track_id   = track_id,
            onset_t    = onset_t,
            offset_t   = None,
            duration_s = None,
            peak_stat  = peak_stat,
            onset_index= onset_idx,
        ))

    return events


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def run_cusum(
    traj_path: Union[str, Path],
    track_id: Optional[str] = None,
    k: float = 1.5,
    h: float = 5.0,
    gap_reset_s: float = 2.0,
    save: bool = False,
) -> CUSUMResult:
    """Load a trajectory CSV and run the CUSUM detector on it.

    Parameters
    ----------
    traj_path : str | Path
        Path to a ``*_trajectory.csv`` file produced by ``audit.py``.
    track_id : str, optional
        If None, the track ID is inferred from the filename
        (e.g. ``T000001_trajectory.csv`` → ``T000001``).
    k, h, gap_reset_s : float
        Detector hyperparameters (see ``CUSUMDetector``).
    save : bool
        If True, write the augmented DataFrame back to disk alongside the
        source file as ``<track_id>_cusum.csv``.

    Returns
    -------
    CUSUMResult
    """
    traj_path = Path(traj_path)
    if not traj_path.exists():
        raise FileNotFoundError(f"run_cusum(): file not found — {traj_path}")

    if track_id is None:
        stem = traj_path.stem          # e.g. "T000001_trajectory"
        track_id = stem.split("_")[0]  # "T000001"

    df = pd.read_csv(traj_path)
    detector = CUSUMDetector(k=k, h=h, gap_reset_s=gap_reset_s)
    result = detector.fit(df, track_id=track_id)

    if save:
        out_path = traj_path.parent / f"{track_id}_cusum.csv"
        result.traj_df.to_csv(out_path, index=False)

    return result


def batch_cusum(
    folders: Sequence[Union[str, Path]],
    glob_pattern: str = "*_trajectory.csv",
    k: float = 1.5,
    h: float = 5.0,
    gap_reset_s: float = 2.0,
    save: bool = False,
) -> pd.DataFrame:
    """Run CUSUM on every trajectory CSV found under a list of folders.

    Returns a summary DataFrame with one row per (folder, track) containing:
    - ``folder``           : source folder path
    - ``track_id``         : Kalman track identifier
    - ``n_nis_valid``      : number of timesteps with valid NIS_x
    - ``n_gap_resets``     : number of S_t resets due to measurement gaps
    - ``mean_NIS_x``       : mean NIS_x (H₀ expectation = 1.0)
    - ``frac_NIS_exc``     : fraction of NIS_x > 3.84 (H₀ expectation = 0.05)
    - ``n_alarms``         : number of alarm episodes detected
    - ``total_alarm_s``    : total alarm duration [s] (None if any episode is open-ended)
    - ``first_alarm_t``    : time of first alarm onset [s], or NaN if no alarm

    Parameters
    ----------
    folders : sequence of str | Path
        Directories to search for trajectory CSV files.
    glob_pattern : str
        Glob pattern relative to each folder.
    k, h, gap_reset_s : float
        Detector hyperparameters.
    save : bool
        If True, write per-track ``*_cusum.csv`` files alongside inputs.

    Returns
    -------
    pd.DataFrame
    """
    records = []
    detector = CUSUMDetector(k=k, h=h, gap_reset_s=gap_reset_s)

    for folder in folders:
        folder = Path(folder)
        csv_files = sorted(folder.glob(glob_pattern))
        for csv_path in csv_files:
            stem     = csv_path.stem
            track_id = stem.split("_")[0]

            try:
                df = pd.read_csv(csv_path)
                result = detector.fit(df, track_id=track_id)
            except Exception as exc:  # noqa: BLE001
                records.append({
                    "folder"       : str(folder),
                    "track_id"     : track_id,
                    "error"        : str(exc),
                })
                continue

            if save:
                out = csv_path.parent / f"{track_id}_cusum.csv"
                result.traj_df.to_csv(out, index=False)

            # Alarm summary.
            alarms  = result.alarms
            n_al    = len(alarms)
            first_t = alarms[0].onset_t if alarms else float("nan")
            try:
                total_s = sum(a.duration_s for a in alarms)  # type: ignore[misc]
            except TypeError:
                total_s = None  # at least one open-ended episode

            records.append({
                "folder"        : str(folder),
                "track_id"      : track_id,
                "n_nis_valid"   : result.n_nis_valid,
                "n_gap_resets"  : result.n_gap_resets,
                "mean_NIS_x"    : result.mean_NIS_x,
                "frac_NIS_exc"  : result.frac_NIS_exceedance,
                "n_alarms"      : n_al,
                "total_alarm_s" : total_s,
                "first_alarm_t" : first_t,
                "error"         : None,
            })

    return pd.DataFrame(records)
