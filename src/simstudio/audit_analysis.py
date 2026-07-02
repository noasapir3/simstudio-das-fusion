"""Tracking-audit *analysis* layer (post-run, oracle-aware, read-only).

This module is purely additive on top of :mod:`simstudio.audit`.  It does
NOT touch the Kalman filter, the tracker, or any sensor model.  It only
consumes the ``AuditRow`` / ``TrajectoryRow`` tables produced by
:func:`simstudio.audit.build_audit_and_trajectory` and turns them into
*interpretations*: an executive summary, sensor-coverage / quality /
geometry diagnostics, Kalman-behavior diagnostics, segment-transition
consistency checks, fragmentation analysis, error-spike investigations,
and prioritised recommendations.

Design rules
------------
1.  Read-only.  Never mutates the audit / trajectory rows.
2.  Pure rule-based.  No ML, no LLM call.  Every interpretation is
    derived from the local evidence in the relevant time window.
3.  Honest about uncertainty.  Every interpretation tags its claims as
    *confirmed*, *likely*, *possible (hypothesis)*, or *next check*.
4.  Defensive.  Missing fields → that branch is skipped, never raised.

The two report writers in :mod:`simstudio.audit` import the high-level
helpers from here (``build_full_analysis``, ``Interpretation``,
``Section``) to render the new sections without duplicating logic.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class Interpretation:
    """A single intelligent explanation of an event / state.

    Attributes
    ----------
    label
        Short human label, e.g. ``"Temporary sensor coverage loss"``.
    severity
        ``"info" | "warning" | "critical"``.
    technical
        One-sentence factual description of what happened.
    meaning
        Plain-English explanation of what the technical fact probably
        means about the tracking system.
    why_matters
        One sentence on why a reader should care.
    evidence
        Bullet-list of *confirmed* facts that support the
        interpretation (extracted from the local window).
    hypotheses
        Bullet-list of *possible* causes — the report must not pretend
        these are confirmed.
    next_checks
        Concrete follow-up checks the user can run.
    """

    label: str
    severity: str
    technical: str
    meaning: str
    why_matters: str
    evidence: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    next_checks: List[str] = field(default_factory=list)


@dataclass
class Section:
    """One analysis section in the report (rendered by both writers)."""

    title: str
    summary: str                              # one-paragraph plain English
    bullets: List[str] = field(default_factory=list)
    table: Optional[Dict[str, Any]] = None    # {"headers": [...], "rows": [...]}
    interpretations: List[Interpretation] = field(default_factory=list)


@dataclass
class ExecutiveSummary:
    """Top-of-report quality scorecard."""

    overall_quality: str            # "good" | "medium" | "problematic"
    main_reason: str
    biggest_error_source: str
    most_suspicious_behavior: str
    best_sensor: str
    weakest_sensor: str
    weakest_time_window: str
    relied_on_prediction: bool
    too_few_sensor_measurements: bool
    suspicious_noisy_sensor: str    # name or ""
    fragmentation_suspected: bool
    plain_english_paragraph: str
    bullets: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_mean(xs: Iterable[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and math.isfinite(float(x))]
    return (sum(xs) / len(xs)) if xs else None


def _safe_rmse(xs: Iterable[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return (sum(x * x for x in xs) / len(xs)) ** 0.5 if xs else None


def _safe_max(xs: Iterable[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return max(xs) if xs else None


def _fmt(v: Optional[float], unit: str = "", nd: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}{unit}"


def _quality_from_rmse(rmse: Optional[float], pct_pred: float) -> Tuple[str, str]:
    """Return (label, why) — heuristic scorecard for a track / scenario."""
    if rmse is None:
        return "unknown", "no error measurements available"
    if rmse < 0.5 and pct_pred < 30.0:
        return "good", f"RMSE {rmse:.2f} m with only {pct_pred:.0f}% prediction-only rows"
    if rmse < 1.5 and pct_pred < 60.0:
        return "medium", (
            f"RMSE {rmse:.2f} m with {pct_pred:.0f}% prediction-only rows; "
            "the filter coasts noticeably but stays near truth"
        )
    return "problematic", (
        f"RMSE {rmse:.2f} m with {pct_pred:.0f}% prediction-only rows — "
        "the filter is mostly coasting and / or the error is large"
    )


# ---------------------------------------------------------------------------
# 1.  Sensor coverage analysis
# ---------------------------------------------------------------------------


def analyze_sensor_coverage(
    audit_rows: List[Any],
    traj_rows: List[Any],
    by_gid: Dict[str, List[Any]],
) -> Section:
    """Section 2 — does the scenario have *enough* sensor information?

    Computes per-track and global counts, prediction-only fraction,
    longest gap without a real sensor update, and time windows where
    only one (or no) sensor was active.  Then attaches plain-English
    conclusions.
    """
    headers = [
        "Track", "GPS", "Cam", "DAS", "Pred",
        "Rows", "%pred", "Longest meas-gap (s)",
    ]
    rows: List[List[str]] = []
    interpretations: List[Interpretation] = []

    n_pred_global = 0
    n_rows_global = 0
    cam_total = 0
    das_total = 0
    gps_total = 0
    longest_gap_global = 0.0

    for gid, gid_rows in sorted(by_gid.items()):
        gid_rows = sorted(gid_rows, key=lambda r: r.t)
        n_gps = sum(1 for r in gid_rows if r.gps_x is not None)
        n_cam = sum(1 for r in gid_rows if r.cam_x is not None)
        n_das = sum(1 for r in gid_rows if r.das_x is not None)
        n_pred = sum(1 for r in gid_rows if r.update_kind == "prediction_only")
        n_total = len(gid_rows)
        pct_pred = 100.0 * n_pred / max(1, n_total)

        gps_total += n_gps
        cam_total += n_cam
        das_total += n_das
        n_pred_global += n_pred
        n_rows_global += n_total

        meas_rows = [r for r in gid_rows if r.update_kind != "prediction_only"]
        meas_rows.sort(key=lambda r: r.t)
        longest_gap = 0.0
        for a, b in zip(meas_rows, meas_rows[1:]):
            longest_gap = max(longest_gap, b.t - a.t)
        longest_gap_global = max(longest_gap_global, longest_gap)

        rows.append([
            str(gid), str(n_gps), str(n_cam), str(n_das), str(n_pred),
            str(n_total), f"{pct_pred:.1f}", f"{longest_gap:.2f}",
        ])

        interpretations.append(_interpret_track_coverage(
            gid=gid,
            n_gps=n_gps, n_cam=n_cam, n_das=n_das,
            pct_pred=pct_pred, longest_gap=longest_gap,
            duration=(gid_rows[-1].t - gid_rows[0].t),
            gid_rows=gid_rows,
        ))

    pct_pred_global = 100.0 * n_pred_global / max(1, n_rows_global)
    sole_sensor_windows = _find_single_sensor_windows(traj_rows)
    no_sensor_windows = _find_no_sensor_windows(traj_rows)

    bullets: List[str] = []
    if pct_pred_global > 50:
        bullets.append(
            f"{pct_pred_global:.0f}% of all Kalman rows were prediction-only — "
            "the filter is coasting more than it is being corrected."
        )
    elif pct_pred_global < 15:
        bullets.append(
            f"Only {pct_pred_global:.0f}% prediction-only rows — sensor coverage "
            "is healthy across the scenario."
        )
    if longest_gap_global > 1.0:
        bullets.append(
            f"Longest measurement gap across all tracks: {longest_gap_global:.2f} s."
        )
    if cam_total < max(5, gps_total // 2) and das_total > 5 * max(1, cam_total):
        bullets.append(
            "Camera contributes very few measurements while DAS dominates — "
            "DAS is the de-facto primary sensor."
        )
    if sole_sensor_windows:
        bullets.append(
            f"{len(sole_sensor_windows)} time window(s) had only one sensor type active "
            "(see fragmentation / FOV sections for the likely cause)."
        )
    if no_sensor_windows:
        bullets.append(
            f"{len(no_sensor_windows)} time window(s) had no real sensor update "
            "of any kind — Kalman ran on prediction alone."
        )

    summary = (
        f"Across {len(by_gid)} track(s) the audit recorded "
        f"{gps_total} GPS, {cam_total} camera and {das_total} DAS measurements "
        f"plus {n_pred_global} prediction-only rows ({pct_pred_global:.1f}% of all rows). "
        + (bullets[0] if bullets else "Coverage looks balanced.")
    )

    return Section(
        title="Sensor Coverage Analysis",
        summary=summary,
        bullets=bullets,
        table={"headers": headers, "rows": rows},
        interpretations=interpretations,
    )


def _interpret_track_coverage(
    *,
    gid: str,
    n_gps: int,
    n_cam: int,
    n_das: int,
    pct_pred: float,
    longest_gap: float,
    duration: float,
    gid_rows: List[Any],
) -> Interpretation:
    """Plain-English read of one track's sensor coverage."""
    evidence = [
        f"GPS = {n_gps} measurements, Camera = {n_cam}, DAS = {n_das}.",
        f"Prediction-only rows = {pct_pred:.1f}% of the track.",
        f"Longest gap between real measurements = {longest_gap:.2f} s "
        f"(track duration = {duration:.2f} s).",
    ]
    hypotheses: List[str] = []
    next_checks: List[str] = []

    # Choose a label/severity from the worst issue.
    if pct_pred > 60:
        label = "Kalman coasting / prediction-driven track"
        severity = "warning"
        meaning = (
            f"Track {gid} spent most of its life on prediction-only updates. "
            "Whatever shape we see in the trajectory is largely the filter's "
            "constant-acceleration model, not direct sensor evidence."
        )
        why = (
            "When prediction dominates, even small model mismatch accumulates "
            "into position error, and the reported uncertainty (sigma) is the "
            "only honest signal of how much we should trust the estimate."
        )
        hypotheses += [
            "Sensor cadence may be too slow for the dynamics in this track.",
            "Camera / DAS coverage may end mid-track (see geometry section).",
        ]
        next_checks += [
            "Plot sensor availability vs time for this gid.",
            "Check whether sigma_pos_m matches the actual error during the gaps.",
        ]
    elif n_cam == 0 and n_das > 0:
        label = "DAS-only track (no camera support)"
        severity = "warning"
        meaning = (
            f"Track {gid} never received a camera measurement. The Kalman fuses "
            "only DAS (and possibly GPS), so spatial precision is bounded by "
            "the DAS sigma model."
        )
        why = (
            "Camera updates are usually the most precise; their absence shifts "
            "the burden to DAS and prediction, which typically grows error."
        )
        hypotheses += [
            "Vehicle was outside camera FOV for the whole track lifetime.",
            "Camera measurements were filtered before entering Kalman.",
        ]
        next_checks += [
            "Check camera FOV vs vehicle path for the relevant time window.",
        ]
    elif n_das == 0 and n_cam > 0:
        label = "Camera-only track (no DAS support)"
        severity = "info"
        meaning = (
            f"Track {gid} relied on camera (and GPS) without DAS continuity. "
            "Tracking is precise where the camera sees and degrades where it does not."
        )
        why = (
            "Without DAS, there is no fallback continuous sensor — gaps in camera "
            "coverage become straight prediction-only intervals."
        )
        hypotheses += [
            "Vehicle did not cross any DAS-instrumented segment.",
            "DAS SNR may have been too low for any measurement to be accepted.",
        ]
        next_checks += [
            "Compare DAS SNR distribution against the acceptance threshold.",
        ]
    elif longest_gap > 1.0:
        label = "Temporary sensor coverage loss"
        severity = "warning"
        meaning = (
            f"Track {gid} has a {longest_gap:.2f} s window with no real sensor "
            "update. During that window the Kalman extrapolates and uncertainty grows."
        )
        why = (
            "Long measurement gaps are the single most common cause of post-gap "
            "error spikes; if the next measurement disagrees with the prediction, "
            "the filter has to absorb the correction at once."
        )
        hypotheses += [
            "Sensor handoff between cameras / DAS sections was incomplete.",
            "A segment transition happened inside the gap.",
        ]
        next_checks += [
            "Plot error and sigma_pos_m around the gap.",
            "Confirm whether the post-gap measurement was accepted or skipped.",
        ]
    else:
        label = "Sensor coverage looks sufficient"
        severity = "info"
        meaning = (
            f"Track {gid} received a balanced mix of measurements with no large "
            "gap. The Kalman has enough evidence to stay close to truth."
        )
        why = "Healthy coverage is the baseline for trustworthy tracking."

    return Interpretation(
        label=label,
        severity=severity,
        technical=(
            f"GPS={n_gps}, Cam={n_cam}, DAS={n_das}, "
            f"prediction-only={pct_pred:.1f}%, longest gap={longest_gap:.2f} s."
        ),
        meaning=meaning,
        why_matters=why,
        evidence=evidence,
        hypotheses=hypotheses,
        next_checks=next_checks,
    )


def _find_single_sensor_windows(
    traj_rows: List[Any], min_duration_s: float = 0.5,
) -> List[Tuple[float, float, str]]:
    """Return time windows where only one sensor type was active."""
    by_gid: Dict[str, List[Any]] = {}
    for r in traj_rows:
        if r.update_kind == "prediction_only":
            continue
        by_gid.setdefault(r.global_track_id, []).append(r)
    out: List[Tuple[float, float, str]] = []
    for gid, rows in by_gid.items():
        rows = sorted(rows, key=lambda r: r.t)
        cur_start: Optional[float] = None
        cur_sensor: Optional[str] = None
        prev_t: Optional[float] = None
        for r in rows:
            sensors = []
            if r.gps_x is not None:
                sensors.append("GPS")
            if r.cam_x is not None:
                sensors.append("Camera")
            if r.das_x is not None:
                sensors.append("DAS")
            if len(sensors) == 1:
                if cur_sensor != sensors[0]:
                    if cur_start is not None and prev_t is not None and (prev_t - cur_start) >= min_duration_s:
                        out.append((cur_start, prev_t, cur_sensor or ""))
                    cur_start = r.t
                    cur_sensor = sensors[0]
                prev_t = r.t
            else:
                if cur_start is not None and prev_t is not None and (prev_t - cur_start) >= min_duration_s:
                    out.append((cur_start, prev_t, cur_sensor or ""))
                cur_start = None
                cur_sensor = None
                prev_t = r.t
        if cur_start is not None and prev_t is not None and (prev_t - cur_start) >= min_duration_s:
            out.append((cur_start, prev_t, cur_sensor or ""))
    return out


def _find_no_sensor_windows(
    traj_rows: List[Any], min_duration_s: float = 0.5,
) -> List[Tuple[float, float]]:
    """Return time windows fully filled by prediction-only rows."""
    by_gid: Dict[str, List[Any]] = {}
    for r in traj_rows:
        by_gid.setdefault(r.global_track_id, []).append(r)
    out: List[Tuple[float, float]] = []
    for gid, rows in by_gid.items():
        rows = sorted(rows, key=lambda r: r.t)
        cur_start: Optional[float] = None
        prev_t: Optional[float] = None
        for r in rows:
            if r.update_kind == "prediction_only":
                if cur_start is None:
                    cur_start = r.t
                prev_t = r.t
            else:
                if cur_start is not None and prev_t is not None and (prev_t - cur_start) >= min_duration_s:
                    out.append((cur_start, prev_t))
                cur_start = None
                prev_t = None
        if cur_start is not None and prev_t is not None and (prev_t - cur_start) >= min_duration_s:
            out.append((cur_start, prev_t))
    return out


# ---------------------------------------------------------------------------
# 3.  Sensor quality / noise suspicion
# ---------------------------------------------------------------------------


def analyze_sensor_quality(
    audit_rows: List[Any],
    traj_rows: List[Any],
) -> Section:
    """Section 3 — is one sensor too noisy / biased / overconfident?"""
    sensor_attrs = {
        "GPS":    ("gps_x", "gps_y"),
        "Camera": ("cam_x", "cam_y"),
        "DAS":    ("das_x", "das_y"),
    }
    headers = [
        "Sensor", "N", "Mean err (m)", "Max err (m)", "RMSE (m)",
        "Bias x (m)", "Bias y (m)", "Mean sigma", "Sigma vs err",
    ]
    rows: List[List[str]] = []
    interpretations: List[Interpretation] = []

    # Per-sensor reported sigma (from audit_rows).
    sigma_by_sensor: Dict[str, List[float]] = {"GPS": [], "Camera": [], "DAS": []}
    for a in audit_rows:
        if a.sensor_type in sigma_by_sensor and a.sigma_m and math.isfinite(float(a.sigma_m)):
            sigma_by_sensor[a.sensor_type].append(float(a.sigma_m))

    for sensor, (sx_attr, sy_attr) in sensor_attrs.items():
        errs: List[float] = []
        bx: List[float] = []
        by: List[float] = []
        for r in traj_rows:
            sx = getattr(r, sx_attr, None)
            sy = getattr(r, sy_attr, None)
            if sx is None or sy is None:
                continue
            if r.true_x is None or r.true_y is None:
                continue
            dx = float(sx) - float(r.true_x)
            dy = float(sy) - float(r.true_y)
            errs.append(math.hypot(dx, dy))
            bx.append(dx)
            by.append(dy)
        n = len(errs)
        mean_err = _safe_mean(errs)
        max_err = _safe_max(errs)
        rmse = _safe_rmse(errs)
        bias_x = _safe_mean(bx)
        bias_y = _safe_mean(by)
        mean_sig = _safe_mean(sigma_by_sensor[sensor])

        # Sigma vs actual error — overconfident if mean error >> mean sigma.
        if mean_sig is not None and mean_err is not None and mean_sig > 0:
            ratio = mean_err / mean_sig
            if ratio > 2.0:
                sigma_label = f"σ underestimated ×{ratio:.1f}"
            elif ratio < 0.5:
                sigma_label = f"σ overestimated ×{1.0 / max(ratio, 1e-3):.1f}"
            else:
                sigma_label = "σ ≈ actual"
        else:
            sigma_label = "n/a"

        rows.append([
            sensor, str(n), _fmt(mean_err), _fmt(max_err), _fmt(rmse),
            _fmt(bias_x), _fmt(bias_y), _fmt(mean_sig), sigma_label,
        ])

        if n == 0:
            continue

        interpretations.append(_interpret_sensor_quality(
            sensor=sensor,
            n=n,
            mean_err=mean_err,
            max_err=max_err,
            rmse=rmse,
            bias_x=bias_x,
            bias_y=bias_y,
            mean_sigma=mean_sig,
        ))

    summary = (
        "Per-sensor noise / bias is computed by comparing each accepted "
        "measurement against the oracle ground truth at the same timestep. "
        "‘σ vs err’ flags whether the reported sigma matches reality — a "
        "ratio far from 1.0 means the sensor model is mis-calibrated."
    )
    return Section(
        title="Sensor Quality / Noise Analysis",
        summary=summary,
        bullets=[],
        table={"headers": headers, "rows": rows},
        interpretations=interpretations,
    )


def _interpret_sensor_quality(
    *,
    sensor: str,
    n: int,
    mean_err: Optional[float],
    max_err: Optional[float],
    rmse: Optional[float],
    bias_x: Optional[float],
    bias_y: Optional[float],
    mean_sigma: Optional[float],
) -> Interpretation:
    evidence = [
        f"{n} {sensor} measurements compared to ground truth.",
        f"Mean error = {_fmt(mean_err)}, max error = {_fmt(max_err)}, RMSE = {_fmt(rmse)}.",
        f"Bias = (Δx={_fmt(bias_x)}, Δy={_fmt(bias_y)}).",
        f"Mean reported σ = {_fmt(mean_sigma)}.",
    ]
    hypotheses: List[str] = []
    next_checks: List[str] = []

    # Decide a label.
    bias_mag = math.hypot(bias_x or 0.0, bias_y or 0.0) if (bias_x is not None and bias_y is not None) else None
    overconfident = (
        mean_sigma is not None and mean_err is not None
        and mean_sigma > 0 and mean_err > 2.0 * mean_sigma
    )
    underconfident = (
        mean_sigma is not None and mean_err is not None
        and mean_err > 0 and mean_sigma > 2.0 * max(mean_err, 0.05)
    )

    if overconfident:
        label = "Sensor model may be overconfident"
        severity = "warning"
        meaning = (
            f"{sensor} reports σ ≈ {_fmt(mean_sigma)} but the actual mean error "
            f"is {_fmt(mean_err)}.  The Kalman therefore weights this sensor "
            "more heavily than the noise really justifies, which can pull the "
            "estimate towards bad measurements."
        )
        why = (
            "Overconfident sensor models cause exactly the kind of post-update "
            "jumps and slow biases that look like Kalman bugs but are really "
            "calibration issues."
        )
        hypotheses += [
            f"The {sensor} sigma model floor is too low for this scenario.",
            "Sensor noise is non-Gaussian — variance underestimates true tail.",
        ]
        next_checks += [
            f"Plot {sensor} residuals vs σ; expect ≈68% inside ±σ.",
            f"Try inflating {sensor} σ by ×{(mean_err / max(mean_sigma, 1e-3)):.1f} and re-run.",
        ]
    elif underconfident:
        label = "Sensor model may be too conservative"
        severity = "info"
        meaning = (
            f"{sensor} reports σ ≈ {_fmt(mean_sigma)} but the actual mean error "
            f"is only {_fmt(mean_err)}.  The Kalman is under-using this sensor."
        )
        why = (
            "Conservative sigmas keep the filter safe but waste good information "
            "and slow down convergence after dropouts."
        )
        next_checks += [
            f"Lower {sensor} σ floor and re-evaluate post-gap recovery time.",
        ]
    elif bias_mag is not None and bias_mag > 0.5 and (mean_err or 0.0) > 0 and bias_mag > 0.5 * (mean_err or 0.0):
        label = "Suspicious sensor bias"
        severity = "warning"
        meaning = (
            f"{sensor} measurements are systematically offset by "
            f"({_fmt(bias_x)}, {_fmt(bias_y)}) — that is most of the total error."
        )
        why = (
            "A biased sensor adds a constant offset to every Kalman update; "
            "it cannot be averaged out by more measurements."
        )
        hypotheses += [
            "Calibration / extrinsic alignment of this sensor may be off.",
            "Coordinate frame conversion may introduce a fixed offset.",
        ]
        next_checks += [
            f"Compare {sensor} bias across different segments — if constant, it's calibration.",
        ]
    elif sensor == "Camera" and mean_err is not None and mean_err > 1.0:
        label = "Camera measurements look noisy"
        severity = "info"
        meaning = (
            "Camera mean error is above 1 m, which is unusually high for a "
            "well-calibrated camera and may reflect FOV-edge effects."
        )
        why = "Camera updates are normally the most precise; if not, the filter loses its anchor."
        hypotheses += ["Vehicle frequently near the edge of the FOV (see camera-FOV section)."]
        next_checks += ["Plot camera error vs distance from FOV center."]
    elif sensor == "DAS" and rmse is not None and rmse > 1.5:
        label = "DAS spatial precision is limited"
        severity = "info"
        meaning = (
            f"DAS RMSE of {rmse:.2f} m is consistent with DAS being a continuous "
            "but lower-precision sensor — useful for continuity, not for absolute fixes."
        )
        why = (
            "Treating DAS like a high-precision sensor (low σ) leads to overconfidence; "
            "treating it as a continuity backstop usually gives the best fusion behaviour."
        )
    else:
        label = f"{sensor} behaves as expected"
        severity = "info"
        meaning = (
            f"{sensor} mean error and bias are consistent with the reported σ — "
            "no immediate noise / calibration concern."
        )
        why = "This sensor is likely a reliable contributor to the fusion."

    return Interpretation(
        label=label,
        severity=severity,
        technical=f"{sensor}: mean err {_fmt(mean_err)}, σ {_fmt(mean_sigma)}, bias ({_fmt(bias_x)}, {_fmt(bias_y)}).",
        meaning=meaning,
        why_matters=why,
        evidence=evidence,
        hypotheses=hypotheses,
        next_checks=next_checks,
    )


# ---------------------------------------------------------------------------
# 4.  Error spike investigation
# ---------------------------------------------------------------------------


def investigate_error_spikes(
    traj_rows: List[Any],
    audit_rows: List[Any],
    anomalies_by_gid: Dict[str, List[Dict[str, Any]]],
    by_gid: Dict[str, List[Any]],
    transitions_by_gid: Dict[str, List[Dict[str, Any]]],
) -> Section:
    """Section 4 — for each major spike, explain *why*."""
    interpretations: List[Interpretation] = []
    bullets: List[str] = []
    n_spikes_total = 0

    for gid, anomalies in anomalies_by_gid.items():
        gid_rows = sorted(by_gid.get(gid, []), key=lambda r: r.t)
        if not gid_rows:
            continue
        for a in anomalies:
            kind = a.get("kind", "")
            if kind not in ("error_spike", "high_rmse", "long_coasting",
                            "distance_jump", "post_transition_error_rise"):
                continue
            n_spikes_total += 1
            interp = _interpret_spike(
                gid=gid,
                anomaly=a,
                gid_rows=gid_rows,
                audit_rows=audit_rows,
                transitions=transitions_by_gid.get(gid, []),
            )
            interpretations.append(interp)

    if n_spikes_total == 0:
        bullets.append("No major error spikes were detected in any track.")
    else:
        bullets.append(
            f"Investigated {n_spikes_total} spike / RMSE / coasting event(s) — "
            "each carries an evidence-based explanation below."
        )

    return Section(
        title="Error Spike Investigation",
        summary=(
            "Every detected spike or jump is paired with the local sensor "
            "context (what was available before / during / after) and a "
            "rule-based explanation that distinguishes confirmed facts "
            "from likely causes."
        ),
        bullets=bullets,
        table=None,
        interpretations=interpretations,
    )


def _interpret_spike(
    *,
    gid: str,
    anomaly: Dict[str, Any],
    gid_rows: List[Any],
    audit_rows: List[Any],
    transitions: List[Dict[str, Any]],
) -> Interpretation:
    """Build an evidence-based narrative for one spike."""
    kind = str(anomaly.get("kind", ""))
    t_s = float(anomaly.get("t_start") or 0.0)
    t_e = float(anomaly.get("t_end") or t_s)
    ctx = 1.0
    bef = [r for r in gid_rows if t_s - ctx <= r.t < t_s]
    dur = [r for r in gid_rows if t_s <= r.t <= t_e]
    aft = [r for r in gid_rows if t_e < r.t <= t_e + ctx]

    def _sensors_present(rs: List[Any]) -> List[str]:
        out: set = set()
        for r in rs:
            if r.gps_x is not None:
                out.add("GPS")
            if r.cam_x is not None:
                out.add("Camera")
            if r.das_x is not None:
                out.add("DAS")
        return sorted(out)

    sens_b = _sensors_present(bef)
    sens_d = _sensors_present(dur)
    sens_a = _sensors_present(aft)

    me_b = _safe_mean([r.pos_err_m for r in bef])
    me_d = _safe_mean([r.pos_err_m for r in dur])
    me_a = _safe_mean([r.pos_err_m for r in aft])
    max_in_window = _safe_max([r.pos_err_m for r in dur])

    n_pred_d = sum(1 for r in dur if r.update_kind == "prediction_only")
    pct_pred_d = 100.0 * n_pred_d / max(1, len(dur))

    # Check segment transition proximity.
    near_transition = next(
        (tr for tr in transitions if abs(float(tr.get("t", 0.0)) - t_s) < 1.5),
        None,
    )

    # Check whether camera coverage ends at / near this window.
    cam_present_before = any(r.cam_x is not None for r in bef)
    cam_present_during = any(r.cam_x is not None for r in dur)
    cam_present_after = any(r.cam_x is not None for r in aft)
    cam_loss = cam_present_before and not cam_present_during

    # Check DAS SNR around the window if present in audit_rows.
    das_audit_in = [a for a in audit_rows
                    if a.sensor_type == "DAS" and a.global_track_id == gid
                    and t_s - ctx <= a.t <= t_e + ctx]
    snrs = [a.snr for a in das_audit_in if getattr(a, "snr", 0) > 0]
    low_snr = bool(snrs) and (_safe_mean(snrs) or 99) < 5.0

    # Recovered?
    recovered = (me_a is not None and me_d is not None and me_a < 0.7 * me_d)

    evidence = [
        f"Window: t = {t_s:.2f}–{t_e:.2f} s (duration {t_e - t_s:.2f} s).",
        f"Before sensors: {', '.join(sens_b) or 'none'}; "
        f"during: {', '.join(sens_d) or 'none'}; after: {', '.join(sens_a) or 'none'}.",
        f"Mean error before / during / after = "
        f"{_fmt(me_b)} / {_fmt(me_d)} / {_fmt(me_a)}.",
        f"Prediction-only rows during the window: {n_pred_d}/{len(dur)} "
        f"({pct_pred_d:.0f}%).",
    ]
    if max_in_window is not None:
        evidence.append(f"Peak error inside window: {max_in_window:.2f} m.")
    if near_transition:
        evidence.append(
            f"Segment transition nearby: {near_transition.get('from_segment','?')} → "
            f"{near_transition.get('to_segment','?')} at t={float(near_transition.get('t',0.0)):.2f} s."
        )
    if cam_loss:
        evidence.append("Camera was present before but disappeared inside the window.")
    if snrs:
        evidence.append(
            f"Local DAS SNR samples: mean={_safe_mean(snrs):.1f}, min={min(snrs):.1f}."
        )

    hypotheses: List[str] = []
    next_checks: List[str] = []

    # Choose label by strongest piece of evidence.
    if cam_loss and pct_pred_d > 30:
        label = "Possible camera FOV boundary issue"
        severity = "warning"
        technical = (
            f"Error rose from {_fmt(me_b)} to a peak of {_fmt(max_in_window)} when "
            "the camera stopped contributing measurements."
        )
        meaning = (
            "The filter lost its strongest spatial anchor (camera) and started "
            "relying on DAS / prediction.  The rise in error is the expected "
            "consequence of switching to a less-precise sensor mix."
        )
        why = (
            "If the vehicle keeps moving outside camera coverage, the error will "
            "grow until either DAS gets a lock or the camera sees it again."
        )
        hypotheses += [
            "Vehicle crossed the edge of the camera FOV.",
            "Camera sigma model may be too optimistic close to the edge.",
        ]
        next_checks += [
            "Overlay vehicle position on the camera FOV polygon.",
            "Add or extend a camera covering the post-transition area.",
        ]
    elif near_transition and pct_pred_d > 20:
        label = "Segment-transition coverage gap"
        severity = "warning"
        technical = (
            f"Error grew around a segment transition "
            f"({near_transition.get('from_segment','?')} → "
            f"{near_transition.get('to_segment','?')}) while {pct_pred_d:.0f}% of "
            "rows in the window were prediction-only."
        )
        meaning = (
            "Sensor coverage in the new segment is weaker (or starts later) "
            "than in the old one, so the filter coasts across the transition."
        )
        why = (
            "Transitions are sensitive moments — if the next sensor does not "
            "match the prediction, the post-transition correction is sharp."
        )
        hypotheses += [
            "Camera / DAS layout has a blind spot at the segment boundary.",
            "Track association may have dropped briefly across the transition.",
        ]
        next_checks += [
            "Plot per-sensor activity on top of the segment-transition vline.",
            "Inspect the audit CSV for skipped measurements right after the transition.",
        ]
    elif low_snr:
        label = "Possible DAS low-SNR interval"
        severity = "warning"
        technical = (
            f"DAS SNR is unusually low (mean {_safe_mean(snrs):.1f}) inside the "
            f"window while error rose to {_fmt(max_in_window)}."
        )
        meaning = (
            "Low SNR makes DAS measurements noisier and less informative; the "
            "Kalman either rejects them or accepts them with little weight, "
            "which lets uncertainty grow."
        )
        why = (
            "Low-SNR DAS often coincides with rail noise, distance from the "
            "interrogator, or environmental disturbances — useful to map."
        )
        hypotheses += [
            "Low SNR may stem from fiber distance or local noise sources.",
            "DAS sigma floor may not be inflated enough at low SNR.",
        ]
        next_checks += [
            "Plot DAS SNR vs time alongside the error trace.",
            "Review the σ-from-SNR curve in the DAS sensor model.",
        ]
    elif pct_pred_d > 70:
        label = "Kalman coasting / prediction drift"
        severity = "warning"
        technical = (
            f"{pct_pred_d:.0f}% of rows in the window were prediction-only; "
            f"error grew from {_fmt(me_b)} to peak {_fmt(max_in_window)}."
        )
        meaning = (
            "With almost no real measurements, the Kalman propagated its "
            "constant-acceleration model.  Any model mismatch becomes visible "
            "as a slow, smooth divergence from truth."
        )
        why = (
            "This is the classic dropout-driven drift; harmless if measurements "
            "return quickly, dangerous if they do not."
        )
        hypotheses += [
            "Sensor cadence too slow for the local dynamics.",
            "Vehicle entered an under-instrumented area.",
        ]
        next_checks += [
            "Confirm that sigma_pos_m grows monotonically inside the gap.",
            "Check whether the vehicle changed speed / heading sharply during the gap.",
        ]
    elif kind == "distance_jump":
        label = "Distance-estimate jump"
        severity = "warning"
        technical = (
            f"Estimated travelled distance jumped by "
            f"{float(anomaly.get('value', 0.0)):.1f} m at t={t_s:.2f} s."
        )
        meaning = (
            "A jump in cumulative distance usually means the Kalman position "
            "snapped to absorb a measurement that was far from the prediction."
        )
        why = (
            "Either the measurement was correct and the prediction had drifted, "
            "or the measurement was an outlier the filter accepted too easily."
        )
        hypotheses += [
            "Recovered from a long prediction gap by snapping to a fresh measurement.",
            "An outlier with overly-confident σ pulled the estimate away.",
        ]
        next_checks += [
            "Inspect the audit row for the measurement at t≈"
            f"{t_s:.2f}; check its sigma and source.",
        ]
    else:
        if recovered:
            label = "Normal recovery after dropout"
            severity = "info"
            technical = (
                f"Error rose to {_fmt(me_d)} mid-window and recovered to "
                f"{_fmt(me_a)} once measurements returned."
            )
            meaning = (
                "This is the textbook signature of a temporary sensor coverage "
                "loss followed by Kalman re-acquisition."
            )
            why = "Recovered errors are usually not bugs — they are diagnostics of coverage."
            next_checks += [
                "Confirm sigma_pos_m drops once measurements return.",
            ]
        else:
            label = "Suspicious unrecovered drift"
            severity = "warning"
            technical = (
                f"Error rose to {_fmt(me_d)} during the window and stayed at "
                f"{_fmt(me_a)} afterwards."
            )
            meaning = (
                "Either the post-window measurements are noisy / biased, or the "
                "Kalman locked onto a wrong estimate it cannot escape."
            )
            why = (
                "Persistent error is a sign that the system has not regained "
                "agreement with reality — worth a closer look."
            )
            hypotheses += [
                "Track switch / association error around this window.",
                "Sensor calibration drifted in the latest segment.",
            ]
            next_checks += [
                "Cross-check oracle vehicle_id against assigned global_track_id "
                "for the rows after the window.",
            ]

    return Interpretation(
        label=label,
        severity=severity,
        technical=technical,
        meaning=meaning,
        why_matters=why,
        evidence=evidence,
        hypotheses=hypotheses,
        next_checks=next_checks,
    )


# ---------------------------------------------------------------------------
# 5.  Track fragmentation analysis
# ---------------------------------------------------------------------------


def analyze_fragmentation(
    audit_rows: List[Any],
    traj_rows: List[Any],
    by_gid: Dict[str, List[Any]],
    skipped_gids: List[Tuple[str, str]],
) -> Section:
    """Section 5 — does one physical vehicle become multiple track IDs?"""
    # Map oracle vehicle_id → list of (gid, t_start, t_end, n_rows, in_skipped, reason)
    by_vid: Dict[str, List[Dict[str, Any]]] = {}
    skip_map = dict(skipped_gids)
    for gid, gid_rows in by_gid.items():
        rows = sorted(gid_rows, key=lambda r: r.t)
        if not rows:
            continue
        vid = (rows[0].vehicle_id_oracle or "").strip() or "(unassigned)"
        by_vid.setdefault(vid, []).append({
            "gid": gid,
            "t_start": rows[0].t,
            "t_end": rows[-1].t,
            "n_rows": len(rows),
            "duration": rows[-1].t - rows[0].t,
            "filtered": gid in skip_map,
            "filter_reason": skip_map.get(gid, ""),
        })

    headers = ["Vehicle", "# tracks", "Tracks", "Filtered tracks"]
    rows: List[List[str]] = []
    interpretations: List[Interpretation] = []
    fragmentation_count = 0

    for vid, items in sorted(by_vid.items()):
        items.sort(key=lambda d: d["t_start"])
        track_descs = [
            f"{it['gid']} [{it['t_start']:.2f}-{it['t_end']:.2f}s, "
            f"{it['n_rows']} rows{', filtered' if it['filtered'] else ''}]"
            for it in items
        ]
        n_filtered = sum(1 for it in items if it["filtered"])
        filtered_descs = [
            f"{it['gid']} ({it['filter_reason']})"
            for it in items if it["filtered"]
        ]
        rows.append([
            vid, str(len(items)),
            "; ".join(track_descs),
            "; ".join(filtered_descs) if filtered_descs else "—",
        ])

        if len(items) > 1:
            fragmentation_count += 1
            # Are there filtered tracks that may hide a transition?
            hides_transition = any(
                it["filtered"] and it["duration"] > 0.3
                for it in items
            )
            label = "Track fragmentation suspicion"
            severity = "warning" if hides_transition else "info"
            evidence = [
                f"Vehicle {vid} is associated with {len(items)} different track IDs.",
                f"Tracks: " + "; ".join(track_descs),
            ]
            if n_filtered:
                evidence.append(
                    f"{n_filtered} track(s) were filtered from the report — "
                    "they may still contain useful information."
                )
            meaning = (
                f"Vehicle {vid} appears to have been split into multiple tracker "
                f"identities.  Each identity is reported separately, which makes "
                "it harder to see the full picture of the vehicle's trajectory."
            )
            why = (
                "Fragmentation can mask segment transitions, hide error events, "
                "and inflate the sensor-coverage gaps reported per track."
                + (
                    "  Some of the fragments are filtered out of the report by "
                    "duration / row-count thresholds, so important moments may be invisible."
                    if hides_transition else ""
                )
            )
            hypotheses = [
                "Tracker association threshold may be too strict during sensor handoff.",
                "Sensor handoff between segments / cameras may break the gid.",
            ]
            next_checks = [
                "Lower the report filtering thresholds and re-render — confirm whether "
                "filtered fragments contain meaningful events.",
                "Inspect the moments between t_end of one fragment and t_start of the next.",
            ]
            interpretations.append(Interpretation(
                label=label, severity=severity,
                technical=f"Vehicle {vid} → {len(items)} tracks, {n_filtered} filtered.",
                meaning=meaning, why_matters=why,
                evidence=evidence, hypotheses=hypotheses, next_checks=next_checks,
            ))
        elif items[0]["filtered"]:
            interpretations.append(Interpretation(
                label="Filtered single track",
                severity="info",
                technical=f"Vehicle {vid} has only one track ({items[0]['gid']}) and it is below the report threshold.",
                meaning=(
                    "The vehicle is present in the data but its track was excluded from per-track reporting "
                    f"because: {items[0]['filter_reason']}."
                ),
                why_matters=(
                    "If this short track contains a real event (e.g. a brief sighting or a "
                    "segment crossing), the report would miss it entirely."
                ),
                evidence=[f"Track {items[0]['gid']} duration {items[0]['duration']:.2f} s, {items[0]['n_rows']} rows."],
                hypotheses=["Vehicle observed only briefly.", "Tracker lost association quickly."],
                next_checks=["Lower PNG_MIN_DURATION_S / PNG_MIN_ROWS thresholds and re-run."],
            ))

    summary = (
        f"{len(by_vid)} oracle vehicle(s) were observed across "
        f"{sum(len(v) for v in by_vid.values())} track ID(s).  "
        + (f"{fragmentation_count} vehicle(s) appear fragmented across multiple tracks."
           if fragmentation_count else
           "Each oracle vehicle maps to a single track — no fragmentation suspected.")
    )
    return Section(
        title="Track Fragmentation Analysis",
        summary=summary,
        bullets=[],
        table={"headers": headers, "rows": rows},
        interpretations=interpretations,
    )


# ---------------------------------------------------------------------------
# 6.  Segment transition analysis
# ---------------------------------------------------------------------------


def analyze_segment_transitions(
    traj_rows: List[Any],
    by_gid: Dict[str, List[Any]],
    transitions_by_gid: Dict[str, List[Dict[str, Any]]],
    events: Optional[List[Any]] = None,
    world: Any = None,
) -> Section:
    """Section 6 — true vs reported segment transitions per vehicle."""
    # Build oracle segment timeline per vehicle from world.vehicle_state events.
    oracle_segs_by_vid: Dict[str, List[Tuple[float, str]]] = {}
    if events is not None and world is not None:
        try:
            lane_to_seg = {
                str(lid): str(getattr(lane, "segment_id", "") or "")
                for lid, lane in getattr(world, "lanes", {}).items()
            }
        except Exception:
            lane_to_seg = {}
        for ev in events:
            if getattr(ev, "topic", "") != "world.vehicle_state":
                continue
            p = getattr(ev, "payload", None) or {}
            if not isinstance(p, dict):
                continue
            vid = str(p.get("vehicle_id", "") or "")
            t = float(p.get("t", 0.0) or 0.0)
            seg = lane_to_seg.get(str(p.get("lane_id", "") or ""), "")
            if vid and seg:
                lst = oracle_segs_by_vid.setdefault(vid, [])
                if not lst or lst[-1][1] != seg:
                    lst.append((t, seg))

    headers = ["Track", "Vehicle", "Reported transitions", "Oracle transitions", "Match"]
    rows: List[List[str]] = []
    interpretations: List[Interpretation] = []

    for gid, gid_rows in sorted(by_gid.items()):
        gid_rows = sorted(gid_rows, key=lambda r: r.t)
        if not gid_rows:
            continue
        vid = (gid_rows[0].vehicle_id_oracle or "").strip()
        reported = transitions_by_gid.get(gid, [])
        oracle = oracle_segs_by_vid.get(vid, [])
        # Restrict oracle to track time window.
        t0, t1 = gid_rows[0].t, gid_rows[-1].t
        oracle_in_track = [(t, s) for (t, s) in oracle if t0 - 0.1 <= t <= t1 + 0.1]
        oracle_transitions = max(0, len(oracle_in_track) - 1)
        rep_str = "; ".join(
            f"{tr.get('from_segment','?')}→{tr.get('to_segment','?')}@{float(tr.get('t',0.0)):.2f}s"
            for tr in reported
        ) or "—"
        ora_str = " → ".join(s for _, s in oracle_in_track) or "—"
        match = "yes" if len(reported) == oracle_transitions else "no"
        rows.append([gid, vid or "—", rep_str, ora_str, match])

        # Interpretation when there is a mismatch.
        if oracle_in_track and len(reported) != oracle_transitions:
            label = "Segment transition inconsistency"
            severity = "warning"
            evidence = [
                f"Reported transitions: {len(reported)} ({rep_str}).",
                f"Oracle transitions inside this track's time window: {oracle_transitions} "
                f"({ora_str}).",
                f"Track time window: t = {t0:.2f}–{t1:.2f} s.",
            ]
            if len(reported) < oracle_transitions:
                meaning = (
                    "The report missed at least one segment transition that the "
                    "oracle indicates actually happened.  This usually means the "
                    "estimated position did not match the new segment's geometry "
                    "in time, or the track started after the transition."
                )
                hypotheses = [
                    "Track started after the vehicle entered the new segment.",
                    "Estimated position remained inside the old segment due to drift.",
                    "Filtered fragments may contain the missing transition.",
                ]
            else:
                meaning = (
                    "The report shows more transitions than the oracle indicates — "
                    "the segment assignment is flickering between adjacent segments, "
                    "probably because the estimated position is on a boundary."
                )
                hypotheses = [
                    "Estimated position is oscillating across a segment boundary.",
                    "Lane-to-segment mapping has overlapping centroids near the boundary.",
                ]
            why = (
                "Segment transitions are how the report tells the story of the "
                "vehicle's journey; missing or extra ones distort that story."
            )
            next_checks = [
                "Compare reported vs oracle segment IDs on the trajectory plot.",
                "Decide whether segment_id should be assigned from oracle position "
                "instead of estimated position for reporting purposes.",
            ]
            interpretations.append(Interpretation(
                label=label, severity=severity,
                technical=f"Track {gid}: {len(reported)} reported vs {oracle_transitions} oracle transitions.",
                meaning=meaning, why_matters=why,
                evidence=evidence, hypotheses=hypotheses, next_checks=next_checks,
            ))

    if not interpretations:
        interpretations.append(Interpretation(
            label="Segment transition logic appears consistent",
            severity="info",
            technical="Reported transitions match the oracle for every meaningful track.",
            meaning="The segment-assignment pipeline produced the same story as the ground truth.",
            why_matters="Trustworthy transitions are necessary for any per-segment analysis.",
            evidence=["Per-track row in the table above shows match=yes for every row."],
        ))

    return Section(
        title="Segment Transition Analysis",
        summary=(
            "Compares the segment transitions the report would emit against the "
            "transitions implied by oracle lane_id over time.  Mismatches usually "
            "indicate a coverage / drift issue rather than a transition-logic bug."
        ),
        bullets=[],
        table={"headers": headers, "rows": rows},
        interpretations=interpretations,
    )


# ---------------------------------------------------------------------------
# 7.  Camera geometry / FOV analysis
# ---------------------------------------------------------------------------


def analyze_camera_fov(
    audit_rows: List[Any],
    traj_rows: List[Any],
    by_gid: Dict[str, List[Any]],
    transitions_by_gid: Dict[str, List[Dict[str, Any]]],
) -> Section:
    """Section 7 — does camera placement explain the errors?"""
    headers = ["Track", "Cam meas", "Cam window (s)", "Pre-tr / post-tr", "Mean σ", "Mean err"]
    rows: List[List[str]] = []
    interpretations: List[Interpretation] = []

    cam_audit_by_gid: Dict[str, List[Any]] = {}
    for a in audit_rows:
        if a.sensor_type == "Camera":
            cam_audit_by_gid.setdefault(a.global_track_id, []).append(a)

    for gid, gid_rows in sorted(by_gid.items()):
        gid_rows = sorted(gid_rows, key=lambda r: r.t)
        cam_rows = [r for r in gid_rows if r.cam_x is not None]
        if not cam_rows:
            rows.append([gid, "0", "—", "—", "—", "—"])
            interpretations.append(Interpretation(
                label="No camera coverage on this track",
                severity="info",
                technical=f"Track {gid} received no camera measurements.",
                meaning="The camera FOV did not include this vehicle at any time during the track.",
                why_matters="Without camera, the spatial precision of fusion is bounded by DAS / GPS.",
                evidence=[f"Camera count: 0 over track t=[{gid_rows[0].t:.2f}, {gid_rows[-1].t:.2f}] s."],
                hypotheses=["Vehicle path lies entirely outside the camera FOV.",
                            "All camera measurements were filtered before the audit."],
                next_checks=["Plot camera FOV polygon over the vehicle path.",
                             "Inspect the audit CSV for camera-row skip reasons."],
            ))
            continue
        t_start_cam = cam_rows[0].t
        t_end_cam = cam_rows[-1].t
        cam_window = t_end_cam - t_start_cam

        # Per-transition: count cam meas before / after.
        trs = transitions_by_gid.get(gid, [])
        if trs:
            tr_t = float(trs[0].get("t", 0.0))
            n_pre = sum(1 for r in cam_rows if r.t < tr_t)
            n_post = sum(1 for r in cam_rows if r.t >= tr_t)
            tr_str = f"{n_pre}/{n_post}"
        else:
            tr_str = "n/a"

        # Camera σ.
        sigmas = [a.sigma_m for a in cam_audit_by_gid.get(gid, []) if a.sigma_m]
        mean_sigma = _safe_mean(sigmas)

        # Camera mean error vs ground truth.
        cam_errs = [
            math.hypot(r.cam_x - r.true_x, r.cam_y - r.true_y)
            for r in cam_rows
            if r.true_x is not None and r.true_y is not None
        ]
        mean_err = _safe_mean(cam_errs)

        rows.append([
            gid, str(len(cam_rows)), f"{cam_window:.2f}",
            tr_str, _fmt(mean_sigma), _fmt(mean_err),
        ])

        # Interpretation per track.
        if trs and cam_rows[-1].t < float(trs[-1].get("t", 0.0)):
            interpretations.append(Interpretation(
                label="Camera coverage stops before track ends",
                severity="warning",
                technical=(
                    f"Camera last seen at t={cam_rows[-1].t:.2f}s but track continues "
                    f"to t={gid_rows[-1].t:.2f}s, including a transition at "
                    f"t={float(trs[-1].get('t',0.0)):.2f}s."
                ),
                meaning=(
                    "After the camera stops contributing, the filter relies on DAS "
                    "and prediction.  Any error growth in the post-camera window "
                    "should be read in that light."
                ),
                why_matters=(
                    "Adding a camera that covers the post-transition area would "
                    "directly address the most common source of post-transition error."
                ),
                evidence=[
                    f"Camera measurements: {len(cam_rows)}, last at t={cam_rows[-1].t:.2f}s.",
                    f"Track ends at t={gid_rows[-1].t:.2f}s.",
                ],
                hypotheses=["Camera FOV ends short of the segment boundary.",
                            "Vehicle exits FOV laterally just before the transition."],
                next_checks=["Plot camera FOV polygon and overlay vehicle path.",
                             "Consider extending camera range or adding a second camera."],
            ))
        elif trs and tr_str != "n/a":
            n_pre, n_post = (int(x) for x in tr_str.split("/"))
            if n_post < max(2, n_pre // 2):
                interpretations.append(Interpretation(
                    label="Camera contributes mostly before the transition",
                    severity="info",
                    technical=f"Camera meas pre/post first transition: {n_pre}/{n_post}.",
                    meaning=(
                        "The camera dominates the early segment and then largely "
                        "disappears; the post-transition tracking depends on DAS."
                    ),
                    why_matters=(
                        "Asymmetric camera coverage explains why error often rises "
                        "right after a segment transition even when the filter is healthy."
                    ),
                    evidence=[f"Camera-window length: {cam_window:.2f}s with the bulk before the transition."],
                    hypotheses=["FOV layout intentionally favors the early segment."],
                    next_checks=["Decide whether post-segment camera coverage is worth adding."],
                ))
        if mean_err is not None and mean_err > 1.0 and mean_sigma is not None and mean_err > 2.0 * mean_sigma:
            interpretations.append(Interpretation(
                label="Camera σ may underestimate FOV-edge error",
                severity="warning",
                technical=f"Mean camera error {mean_err:.2f} m vs reported σ {mean_sigma:.2f} m.",
                meaning=(
                    "The camera reports tighter uncertainty than its actual error.  "
                    "Inside the FOV the noise model is probably fine, but at the edges "
                    "or under angle it underestimates noise."
                ),
                why_matters="Overconfident camera σ pulls the Kalman towards bad measurements.",
                evidence=[
                    f"Mean camera err {mean_err:.2f} m, mean σ {mean_sigma:.2f} m.",
                ],
                hypotheses=["Edge-of-FOV pixels have larger projection error.",
                            "Camera noise model does not depend on viewing angle."],
                next_checks=[
                    "Plot camera error vs distance from FOV center.",
                    "Add a σ multiplier for measurements close to the FOV edge.",
                ],
            ))

    return Section(
        title="Camera Geometry / FOV Analysis",
        summary=(
            "Per-track camera measurement counts, time windows of camera "
            "presence, and a sanity check on the camera σ vs actual error.  "
            "Conclusions about FOV are heuristic — the audit data does not "
            "carry the camera polygon itself."
        ),
        bullets=[],
        table={"headers": headers, "rows": rows},
        interpretations=interpretations,
    )


# ---------------------------------------------------------------------------
# 8.  DAS quality analysis
# ---------------------------------------------------------------------------


def analyze_das_quality(
    audit_rows: List[Any],
    traj_rows: List[Any],
    by_gid: Dict[str, List[Any]],
) -> Section:
    """Section 8 — DAS-specific interpretation."""
    headers = ["Track", "DAS meas", "Mean SNR", "Min SNR", "Low-SNR rows", "Mean err"]
    rows: List[List[str]] = []
    interpretations: List[Interpretation] = []

    das_by_gid: Dict[str, List[Any]] = {}
    for a in audit_rows:
        if a.sensor_type == "DAS":
            das_by_gid.setdefault(a.global_track_id, []).append(a)

    global_low_snr_event = False
    global_correlations = []

    for gid, gid_rows in sorted(by_gid.items()):
        gid_rows = sorted(gid_rows, key=lambda r: r.t)
        das_audits = das_by_gid.get(gid, [])
        snrs = [a.snr for a in das_audits if getattr(a, "snr", 0.0) > 0]
        n_das = len(das_audits)
        if n_das == 0:
            rows.append([gid, "0", "—", "—", "—", "—"])
            continue
        mean_snr = _safe_mean(snrs)
        min_snr = min(snrs) if snrs else None
        low_snr_count = sum(1 for s in snrs if s < 5.0)

        # DAS-only error: estimate by Kalman pos err in rows where only DAS contributed.
        das_only_errs = [
            r.pos_err_m for r in gid_rows
            if r.das_x is not None and r.gps_x is None and r.cam_x is None
            and r.pos_err_m is not None
        ]
        mean_das_err = _safe_mean(das_only_errs)
        rows.append([
            gid, str(n_das), _fmt(mean_snr), _fmt(min_snr),
            str(low_snr_count), _fmt(mean_das_err),
        ])

        if low_snr_count > 0:
            global_low_snr_event = True

        # Correlation: rows where DAS SNR is low → higher error?
        if snrs and gid_rows:
            das_t = sorted(((a.t, a.snr) for a in das_audits if getattr(a, "snr", 0.0) > 0),
                           key=lambda x: x[0])
            if len(das_t) >= 5:
                snr_med = statistics.median(s for _, s in das_t)
                low = [t for t, s in das_t if s < snr_med]
                high = [t for t, s in das_t if s >= snr_med]
                err_low = [
                    r.pos_err_m for r in gid_rows
                    if r.pos_err_m is not None and any(abs(r.t - t) < 0.2 for t in low)
                ]
                err_high = [
                    r.pos_err_m for r in gid_rows
                    if r.pos_err_m is not None and any(abs(r.t - t) < 0.2 for t in high)
                ]
                me_l = _safe_mean(err_low)
                me_h = _safe_mean(err_high)
                if me_l is not None and me_h is not None and me_h > 0:
                    global_correlations.append((gid, me_l, me_h))

        # Per-track interpretation (only when interesting).
        if mean_das_err is not None and mean_das_err > 1.5:
            interpretations.append(Interpretation(
                label="DAS-driven error elevated",
                severity="info",
                technical=(
                    f"Track {gid}: DAS-only mean error {mean_das_err:.2f} m across "
                    f"{len(das_only_errs)} rows."
                ),
                meaning=(
                    "When only DAS is available, the Kalman estimate sits "
                    f"about {mean_das_err:.2f} m from truth on average.  "
                    "That is the realistic spatial precision of DAS for this track."
                ),
                why_matters="It bounds how good fusion can be in DAS-only intervals.",
                evidence=[f"{len(das_only_errs)} DAS-only rows analysed."],
                hypotheses=["DAS sigma floor may need to reflect this realistic error."],
                next_checks=["Compare DAS-only error against camera-active error within the same track."],
            ))

    bullets: List[str] = []
    if global_low_snr_event:
        bullets.append(
            "At least one track contains DAS measurements with SNR < 5 — these are "
            "likely down-weighted or rejected by the filter."
        )
    if global_correlations:
        worst = max(global_correlations, key=lambda t: t[2] / max(t[1], 1e-3))
        gid, me_l, me_h = worst
        if me_h > 1.3 * me_l:
            bullets.append(
                f"Track {gid}: error is ×{me_h / max(me_l, 1e-3):.1f} higher in low-SNR "
                "windows than in high-SNR windows — DAS SNR clearly correlates with quality."
            )
            interpretations.append(Interpretation(
                label="Low DAS SNR correlates with higher Kalman error",
                severity="warning",
                technical=f"Track {gid}: low-SNR mean err {me_l:.2f} m vs high-SNR {me_h:.2f} m.",
                meaning=(
                    "When DAS SNR drops, the position error tends to grow.  Either "
                    "low-SNR measurements are being trusted too much, or they are "
                    "being rejected and the resulting prediction-only intervals drift."
                ),
                why_matters="If real, the σ-from-SNR model should be inflated more aggressively at low SNR.",
                evidence=[
                    f"Mean error in low-SNR windows: {me_l:.2f} m",
                    f"Mean error in high-SNR windows: {me_h:.2f} m",
                ],
                hypotheses=["σ-from-SNR curve underestimates noise at the low-SNR end."],
                next_checks=["Plot DAS SNR and pos_err_m vs time on the same axis."],
            ))

    summary = (
        "DAS measurement counts, SNR distribution, and the Kalman error in "
        "DAS-only windows.  Where SNR data is available, correlations with "
        "tracking error are flagged as evidence of σ-from-SNR mis-calibration."
    )
    return Section(
        title="DAS Quality Analysis",
        summary=summary,
        bullets=bullets,
        table={"headers": headers, "rows": rows},
        interpretations=interpretations,
    )


# ---------------------------------------------------------------------------
# 9.  Kalman behavior analysis
# ---------------------------------------------------------------------------


def analyze_kalman_behavior(
    traj_rows: List[Any],
    by_gid: Dict[str, List[Any]],
) -> Section:
    """Section 9 — does the Kalman behave as expected?"""
    headers = ["Track", "% pred-only", "σ growth in gaps", "σ shrink after meas",
               "σ vs actual err"]
    rows: List[List[str]] = []
    interpretations: List[Interpretation] = []

    overconfident_count = 0
    healthy_count = 0
    n_tracks = 0

    for gid, gid_rows in sorted(by_gid.items()):
        rows_sorted = sorted(gid_rows, key=lambda r: r.t)
        if not rows_sorted:
            continue
        n_tracks += 1
        n_pred = sum(1 for r in rows_sorted if r.update_kind == "prediction_only")
        pct_pred = 100.0 * n_pred / max(1, len(rows_sorted))

        # σ growth during prediction-only segments.
        grows = 0
        gap_count = 0
        in_gap = False
        sigma_start: Optional[float] = None
        for r in rows_sorted:
            if r.update_kind == "prediction_only":
                if not in_gap:
                    in_gap = True
                    sigma_start = r.sigma_pos_m
                sigma_end = r.sigma_pos_m
            else:
                if in_gap and sigma_start is not None and r.sigma_pos_m is not None:
                    gap_count += 1
                    if r.sigma_pos_m < sigma_start * 0.95 or sigma_start < (sigma_end or sigma_start) * 0.95:
                        # measurement reduced σ → good shrink, growth was right
                        grows += 1
                in_gap = False
                sigma_start = None
        sigma_growth_str = f"{grows}/{gap_count}" if gap_count else "—"

        # σ shrink immediately after a measurement.
        shrink_count = 0
        compare_count = 0
        prev_pred_sigma: Optional[float] = None
        for r in rows_sorted:
            if r.update_kind == "prediction_only":
                prev_pred_sigma = r.sigma_pos_m
            else:
                if prev_pred_sigma is not None and r.sigma_pos_m is not None:
                    compare_count += 1
                    if r.sigma_pos_m < prev_pred_sigma:
                        shrink_count += 1
                prev_pred_sigma = None
        shrink_str = f"{shrink_count}/{compare_count}" if compare_count else "—"

        # σ vs actual error.
        pairs = [(r.sigma_pos_m, r.pos_err_m) for r in rows_sorted
                 if r.sigma_pos_m is not None and r.pos_err_m is not None]
        if pairs:
            mean_sig = sum(s for s, _ in pairs) / len(pairs)
            mean_err = sum(e for _, e in pairs) / len(pairs)
            if mean_sig > 0 and mean_err > 0:
                ratio = mean_err / mean_sig
                if ratio > 2.0:
                    sigma_judge = f"σ underestimates err ×{ratio:.1f}"
                    overconfident_count += 1
                elif ratio < 0.5:
                    sigma_judge = f"σ over-cautious ×{1.0 / ratio:.1f}"
                else:
                    sigma_judge = "σ ≈ err (healthy)"
                    healthy_count += 1
            else:
                sigma_judge = "n/a"
        else:
            sigma_judge = "n/a"

        rows.append([gid, f"{pct_pred:.1f}", sigma_growth_str, shrink_str, sigma_judge])

        if "underestimates" in sigma_judge:
            interpretations.append(Interpretation(
                label="Kalman may be overconfident",
                severity="warning",
                technical=f"Track {gid}: mean error {_fmt(mean_err)} but mean σ {_fmt(mean_sig)}.",
                meaning=(
                    "The filter believes it knows the position more accurately than "
                    "it actually does.  That under-confidence in the next sensor "
                    "update means good measurements are partially ignored."
                ),
                why_matters=(
                    "Overconfidence is the silent failure mode of Kalman tuning — "
                    "errors look small in σ but truth says otherwise."
                ),
                evidence=[
                    f"Mean σ_pos = {mean_sig:.2f} m",
                    f"Mean actual error = {mean_err:.2f} m",
                ],
                hypotheses=[
                    "Process noise (sigma_q) too low.",
                    "Sensor σ values too tight (see sensor-quality section).",
                ],
                next_checks=[
                    "Increase process noise and re-run — does σ track actual error?",
                    "Cross-check the sensor-quality section for σ overconfidence.",
                ],
            ))
        elif "over-cautious" in sigma_judge:
            interpretations.append(Interpretation(
                label="Kalman may be over-cautious",
                severity="info",
                technical=f"Track {gid}: σ_pos {_fmt(mean_sig)} dwarfs actual error {_fmt(mean_err)}.",
                meaning=(
                    "Reported uncertainty is much larger than the real error — the "
                    "filter under-trusts itself and slowly absorbs new measurements."
                ),
                why_matters="Slow convergence means longer recovery from dropouts than necessary.",
                evidence=[f"σ_pos {mean_sig:.2f} m vs err {mean_err:.2f} m."],
                next_checks=["Try lowering process noise and observe whether RMSE drops."],
            ))

    if healthy_count >= max(1, n_tracks - 1) and overconfident_count == 0:
        interpretations.append(Interpretation(
            label="Kalman behaves correctly",
            severity="info",
            technical="σ tracks actual error, σ grows during gaps, shrinks after measurements.",
            meaning=(
                "All meaningful tracks show the textbook Kalman pattern: uncertainty "
                "rises while coasting and falls when sensor data returns, and the "
                "reported uncertainty matches the actual error well."
            ),
            why_matters="No tuning intervention needed for the filter itself.",
            evidence=[f"{healthy_count}/{n_tracks} tracks classified healthy."],
        ))

    summary = (
        "For each track the Kalman is evaluated on: how often it received "
        "real updates (vs prediction), whether σ grows in gaps and shrinks "
        "after measurements, and whether the reported σ matches the actual "
        "error.  Strong mismatches indicate a tuning issue — not a bug."
    )
    return Section(
        title="Kalman Behavior Analysis",
        summary=summary,
        bullets=[],
        table={"headers": headers, "rows": rows},
        interpretations=interpretations,
    )


# ---------------------------------------------------------------------------
# 10.  Recommendations
# ---------------------------------------------------------------------------


def build_recommendations(sections: List[Section]) -> Section:
    """Aggregate next_checks from all interpretations into a prioritised list."""
    items: List[Tuple[int, str, str]] = []  # (priority, label, text)
    sev_priority = {"critical": 0, "warning": 1, "info": 2}
    for s in sections:
        for interp in s.interpretations:
            for check in interp.next_checks:
                items.append((sev_priority.get(interp.severity, 3), interp.label, check))
    # Stable de-duplication.
    seen: set = set()
    unique: List[Tuple[int, str, str]] = []
    for it in sorted(items, key=lambda t: t[0]):
        if it[2] in seen:
            continue
        seen.add(it[2])
        unique.append(it)

    bullets = [f"({lab}) {chk}" for _, lab, chk in unique]
    if not bullets:
        bullets = ["No specific follow-up checks were generated; the report did not detect any actionable issues."]

    return Section(
        title="Recommendations & Next Investigations",
        summary=(
            "Concrete checks aggregated from the diagnostic sections.  "
            "Items are ordered by severity (critical → warning → info) and "
            "de-duplicated.  Each is linked to the interpretation that suggested it."
        ),
        bullets=bullets,
    )


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------


def build_executive_summary(
    audit_rows: List[Any],
    traj_rows: List[Any],
    by_gid: Dict[str, List[Any]],
    stats_by_gid: Dict[str, Dict[str, Any]],
    anomalies_by_gid: Dict[str, List[Dict[str, Any]]],
    skipped_gids: List[Tuple[str, str]],
    coverage: List[Dict[str, Any]],
) -> ExecutiveSummary:
    """Top-of-report scorecard with a plain-English paragraph."""
    # Global error stats.
    all_errs = [r.pos_err_m for r in traj_rows if r.pos_err_m is not None]
    rmse = _safe_rmse(all_errs)
    n_total = len(traj_rows)
    n_pred = sum(1 for r in traj_rows if r.update_kind == "prediction_only")
    pct_pred = 100.0 * n_pred / max(1, n_total)

    quality, why_q = _quality_from_rmse(rmse, pct_pred)

    # Per-sensor error to pick best / worst.
    sens_err: Dict[str, Optional[float]] = {}
    for sensor, (sx, sy) in {
        "GPS":    ("gps_x", "gps_y"),
        "Camera": ("cam_x", "cam_y"),
        "DAS":    ("das_x", "das_y"),
    }.items():
        errs = []
        for r in traj_rows:
            ax, ay = getattr(r, sx, None), getattr(r, sy, None)
            if ax is not None and ay is not None and r.true_x is not None and r.true_y is not None:
                errs.append(math.hypot(float(ax) - r.true_x, float(ay) - r.true_y))
        sens_err[sensor] = _safe_mean(errs)

    valid_sens = [(s, e) for s, e in sens_err.items() if e is not None]
    best_sensor = min(valid_sens, key=lambda t: t[1])[0] if valid_sens else "n/a"
    weakest_sensor = max(valid_sens, key=lambda t: t[1])[0] if valid_sens else "n/a"

    # Worst time window: bin error into 1 s bins, take the worst.
    bin_w = 1.0
    bins: Dict[float, List[float]] = {}
    for r in traj_rows:
        if r.pos_err_m is None:
            continue
        b = math.floor(r.t / bin_w) * bin_w
        bins.setdefault(b, []).append(r.pos_err_m)
    worst_bin = max(bins.items(), key=lambda kv: _safe_mean(kv[1]) or 0.0, default=(None, []))
    if worst_bin[0] is not None and worst_bin[1]:
        weakest_window = (
            f"t = {worst_bin[0]:.1f}–{worst_bin[0] + bin_w:.1f} s "
            f"(mean err {_safe_mean(worst_bin[1]):.2f} m)"
        )
    else:
        weakest_window = "no measurable error windows"

    # Biggest error source heuristic.
    if pct_pred > 50:
        biggest = "long prediction-only intervals (Kalman coasting)"
    elif weakest_sensor in ("Camera", "DAS", "GPS") and (sens_err.get(weakest_sensor) or 0) > 1.5:
        biggest = f"{weakest_sensor} measurement noise / bias"
    elif rmse is not None and rmse > 1.0:
        biggest = "Kalman / sensor-fusion drift"
    else:
        biggest = "no single source dominates"

    # Most suspicious behavior.
    n_anom = sum(len(v) for v in anomalies_by_gid.values())
    if n_anom == 0:
        suspicious = "no anomalies were detected"
    else:
        kinds = [a.get("kind", "") for v in anomalies_by_gid.values() for a in v]
        most_kind = max(set(kinds), key=kinds.count)
        suspicious = f"recurring '{most_kind}' anomalies ({kinds.count(most_kind)} occurrences)"

    # Fragmentation suspicion.
    by_vid: Dict[str, set] = {}
    for r in traj_rows:
        vid = (r.vehicle_id_oracle or "").strip() or "(unassigned)"
        by_vid.setdefault(vid, set()).add(r.global_track_id)
    fragmentation_suspected = any(len(v) > 1 for v in by_vid.values())

    # Suspicious noisy sensor.
    noisy_sensor = ""
    if valid_sens:
        worst_sens, worst_err = max(valid_sens, key=lambda t: t[1])
        if worst_err > 1.0:
            noisy_sensor = worst_sens

    relied = pct_pred > 40
    too_few = (pct_pred > 60) or (rmse is not None and rmse > 2.0)

    paragraph_parts = [
        f"Overall tracking quality is **{quality}** — {why_q}.",
        f"The biggest error source is *{biggest}*.",
        f"Most suspicious behavior: {suspicious}.",
        f"Best-performing sensor: {best_sensor}; weakest: {weakest_sensor}.",
        f"Weakest time window: {weakest_window}.",
    ]
    if relied:
        paragraph_parts.append(
            f"The system relied heavily on prediction-only Kalman rows "
            f"({pct_pred:.0f}% of all rows)."
        )
    if too_few:
        paragraph_parts.append(
            "Sensor measurements may be too sparse for the dynamics in this scenario."
        )
    if noisy_sensor:
        paragraph_parts.append(
            f"{noisy_sensor} measurements appear unreliable — see the sensor-quality section."
        )
    if fragmentation_suspected:
        paragraph_parts.append(
            "At least one vehicle was associated with multiple track IDs — "
            "likely fragmentation (see fragmentation section)."
        )

    bullets = [
        f"Overall quality: **{quality}** — {why_q}",
        f"Biggest error source: {biggest}",
        f"Most suspicious behavior: {suspicious}",
        f"Best sensor: {best_sensor}",
        f"Weakest sensor: {weakest_sensor}",
        f"Weakest time window: {weakest_window}",
        f"Relied on prediction-only rows: {'yes' if relied else 'no'} "
        f"({pct_pred:.0f}% of all rows)",
        f"Too few sensor measurements: {'yes' if too_few else 'no'}",
        f"Suspicious noisy sensor: {noisy_sensor or 'none'}",
        f"Vehicle fragmentation suspected: {'yes' if fragmentation_suspected else 'no'}",
    ]

    return ExecutiveSummary(
        overall_quality=quality,
        main_reason=why_q,
        biggest_error_source=biggest,
        most_suspicious_behavior=suspicious,
        best_sensor=best_sensor,
        weakest_sensor=weakest_sensor,
        weakest_time_window=weakest_window,
        relied_on_prediction=relied,
        too_few_sensor_measurements=too_few,
        suspicious_noisy_sensor=noisy_sensor,
        fragmentation_suspected=fragmentation_suspected,
        plain_english_paragraph=" ".join(paragraph_parts),
        bullets=bullets,
    )


# ---------------------------------------------------------------------------
# Top-level entry point used by the report writers
# ---------------------------------------------------------------------------


def build_full_analysis(
    *,
    audit_rows: List[Any],
    traj_rows: List[Any],
    by_gid: Dict[str, List[Any]],
    stats_by_gid: Dict[str, Dict[str, Any]],
    anomalies_by_gid: Dict[str, List[Dict[str, Any]]],
    skipped_gids: List[Tuple[str, str]],
    coverage: List[Dict[str, Any]],
    transitions_by_gid: Dict[str, List[Dict[str, Any]]],
    events: Optional[List[Any]] = None,
    world: Any = None,
) -> Tuple[ExecutiveSummary, List[Section]]:
    """Run all 10 diagnostic sections and return them in display order."""
    exec_summary = build_executive_summary(
        audit_rows=audit_rows, traj_rows=traj_rows, by_gid=by_gid,
        stats_by_gid=stats_by_gid, anomalies_by_gid=anomalies_by_gid,
        skipped_gids=skipped_gids, coverage=coverage,
    )
    sections: List[Section] = [
        analyze_sensor_coverage(audit_rows, traj_rows, by_gid),
        analyze_sensor_quality(audit_rows, traj_rows),
        investigate_error_spikes(traj_rows, audit_rows, anomalies_by_gid, by_gid, transitions_by_gid),
        analyze_fragmentation(audit_rows, traj_rows, by_gid, skipped_gids),
        analyze_segment_transitions(traj_rows, by_gid, transitions_by_gid, events=events, world=world),
        analyze_camera_fov(audit_rows, traj_rows, by_gid, transitions_by_gid),
        analyze_das_quality(audit_rows, traj_rows, by_gid),
        analyze_kalman_behavior(traj_rows, by_gid),
    ]
    sections.append(build_recommendations(sections))
    return exec_summary, sections


__all__ = [
    "Interpretation",
    "Section",
    "ExecutiveSummary",
    "analyze_sensor_coverage",
    "analyze_sensor_quality",
    "investigate_error_spikes",
    "analyze_fragmentation",
    "analyze_segment_transitions",
    "analyze_camera_fov",
    "analyze_das_quality",
    "analyze_kalman_behavior",
    "build_executive_summary",
    "build_recommendations",
    "build_full_analysis",
]
