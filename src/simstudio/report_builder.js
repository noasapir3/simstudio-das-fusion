/**
 * report_builder.js  —  SimStudio Word report renderer
 *
 * Usage:  node report_builder.js <json_payload_path> <output_docx_path>
 *
 * Reads the JSON payload written by audit.py and produces a fully styled
 * .docx file using the docx-js library.  This is the ONLY renderer — there
 * is no parallel Python rendering path.
 */

'use strict';

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageBreak,
} = require('docx');
const fs = require('fs');
const path = require('path');

// ── CLI args ────────────────────────────────────────────────────────
if (process.argv.length < 4) {
  console.error('Usage: node report_builder.js <json_path> <out_path>');
  process.exit(1);
}
const jsonPath = process.argv[2];
const outPath  = process.argv[3];

const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

// ── Color palette ───────────────────────────────────────────────────
const C = {
  navy:       "1F3864",
  blue:       "2E75B6",
  lightBlue:  "D6E4F7",
  red:        "C00000",
  amber:      "C55A11",
  green:      "375623",
  lightGreen: "E2EFDA",
  lightRed:   "FFDAD6",
  lightAmber: "FFF3CD",
  grey:       "595959",
  lightGrey:  "F2F2F2",
  white:      "FFFFFF",
};

const CONTENT_W = 9360; // DXA (US Letter 8.5" − 2×1" margins)

// ── Border helpers ──────────────────────────────────────────────────
const border   = (color = "CCCCCC") => ({ style: BorderStyle.SINGLE, size: 1, color });
const allBorders = (color = "CCCCCC") => ({
  top: border(color), bottom: border(color),
  left: border(color), right: border(color),
});

// ── Paragraph helpers ────────────────────────────────────────────────
function hRule(color = C.blue) {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color, space: 1 } },
    spacing: { before: 0, after: 120 },
    children: [],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 28, bold: true, color: C.navy })],
  });
}

function h2(text, color = C.blue) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 60 },
    children: [new TextRun({ text, font: "Arial", size: 24, bold: true, color })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 160, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 22, bold: true, color: C.grey })],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 80 },
    children: [new TextRun({ text: String(text), font: "Arial", size: 22, color: C.grey, ...opts })],
  });
}

function bodyRuns(runs) {
  return new Paragraph({
    spacing: { before: 60, after: 80 },
    children: runs.map(r => new TextRun({ font: "Arial", size: 22, color: C.grey, ...r })),
  });
}

function bullet(text, color = C.grey) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text: String(text), font: "Arial", size: 22, color })],
  });
}

function space(before = 120) {
  return new Paragraph({ spacing: { before, after: 0 }, children: [] });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

// ── Cell helper ──────────────────────────────────────────────────────
function makeCell(text, opts = {}) {
  const {
    bold = false, color = C.grey, bg = C.white, align = AlignmentType.LEFT,
    width = null, italic = false,
  } = opts;
  const cellOpts = {
    borders: allBorders("CCCCCC"),
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), font: "Arial", size: 20, bold, italic, color })],
    })],
  };
  if (width) cellOpts.width = { size: width, type: WidthType.DXA };
  return new TableCell(cellOpts);
}

function hdrCell(text, width, align = AlignmentType.CENTER) {
  return makeCell(text, { bold: true, color: C.white, bg: C.blue, width, align });
}

function keyCell(text, width = null) {
  return makeCell(text, { bold: true, color: C.navy, bg: C.lightBlue, width });
}

function altRow(i, isAlt) {
  return isAlt ? C.lightGrey : C.white;
}

// ── Callout box (single-cell table) ─────────────────────────────────
function callout(lines, bgColor, borderColor) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({
      children: [new TableCell({
        borders: {
          top: border(borderColor), bottom: border(borderColor),
          left: { style: BorderStyle.SINGLE, size: 16, color: borderColor },
          right: border(borderColor),
        },
        shading: { fill: bgColor, type: ShadingType.CLEAR },
        margins: { top: 120, bottom: 120, left: 200, right: 200 },
        children: lines.map(l => body(l)),
      })]
    })]
  });
}

// ── Generic data table (header row + data rows) ──────────────────────
function dataTable(headers, rows, colWidths = null) {
  const nCols = headers.length;
  const defW = Math.floor(CONTENT_W / nCols);
  const widths = colWidths || headers.map(() => defW);

  const headerRow = new TableRow({
    children: headers.map((h, i) => hdrCell(h, widths[i])),
  });

  const dataRows = rows.map((rowVals, ri) => {
    const bg = ri % 2 === 1 ? C.lightGrey : C.white;
    return new TableRow({
      children: rowVals.map((v, ci) => makeCell(v, { width: widths[ci], bg })),
    });
  });

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headerRow, ...dataRows],
  });
}

// ── KV two-column table ──────────────────────────────────────────────
function kvTable(pairs) {
  const kw = Math.floor(CONTENT_W * 0.38);
  const vw = CONTENT_W - kw;
  const rows = pairs.map(([k, v], i) => {
    const bg = i % 2 === 1 ? C.lightGrey : C.white;
    return new TableRow({
      children: [
        keyCell(k, kw),
        makeCell(v, { width: vw, bg }),
      ],
    });
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [kw, vw],
    rows,
  });
}

// ── Severity colour mapping ──────────────────────────────────────────
function sevColor(sev) {
  if (sev === "critical") return C.red;
  if (sev === "warning")  return C.amber;
  return C.blue;
}

function sevMarker(sev) {
  if (sev === "critical") return "✖";
  if (sev === "warning")  return "⚠";
  return "ℹ";
}

function bulletColorFromText(text) {
  const low = text.toLowerCase();
  if (/critical|error|fail|✖/.test(low)) return C.red;
  if (/warning|caution|suspicious|⚠/.test(low)) return C.amber;
  if (/good|healthy|best|✓|ok/.test(low)) return C.green;
  return C.grey;
}

// ──────────────────────────────────────────────────────────────────────
// BUILD DOCUMENT
// ──────────────────────────────────────────────────────────────────────

const children = [];

const meta = data.meta || {};
const sim  = data.simulation || {};
const es   = data.exec_summary || {};
const enh  = data.enhanced || {};

// ── COVER ─────────────────────────────────────────────────────────────
children.push(space(1440));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "TRACKING AUDIT REPORT", font: "Arial", size: 52, bold: true, color: C.navy })],
}));
children.push(space(120));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Enhanced Diagnostic Analysis", font: "Arial", size: 28, color: C.blue, italic: true })],
}));
children.push(space(80));
children.push(hRule(C.blue));
children.push(space(80));
if (meta.sim_name) {
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: `Simulation: ${meta.sim_name}`, font: "Arial", size: 22, color: C.grey })],
  }));
}
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: `Generated: ${meta.generated}  |  ${sim.meaningful_tracks || 0} Track(s)  |  3 Sensor Types`,
    font: "Arial", size: 22, color: C.grey,
  })],
}));
children.push(pageBreak());

// ── 1. EXECUTIVE SUMMARY ─────────────────────────────────────────────
children.push(h1("1. Executive Summary"));
children.push(hRule());
children.push(body(es.paragraph || ""));
children.push(space(120));
children.push(body("Scorecard — quick reference; full evidence in the diagnostic sections at the end of this report.", { italic: true }));
children.push(space(80));

(es.bullets || []).forEach(line => {
  children.push(bullet(line, bulletColorFromText(line)));
});

children.push(space(120));
children.push(body(
  "Note: every claim is tagged as confirmed (from the data), likely (consistent with the data), " +
  "hypothesis (possible but unverified), or next check (a concrete follow-up).",
  { italic: true }
));
children.push(pageBreak());

// ── 2. SIMULATION SUMMARY ────────────────────────────────────────────
children.push(h1("2. Simulation Summary"));
children.push(hRule());
children.push(kvTable([
  ["Total tracks detected",        String(sim.total_tracks ?? "n/a")],
  ["Meaningful tracks (reported)", String(sim.meaningful_tracks ?? "n/a")],
  ["Filtered / skipped tracks",    String(sim.skipped_tracks ?? "n/a")],
  ["Trajectory rows",              String(sim.trajectory_rows ?? "n/a")],
  ["Prediction-only rows",         String(sim.pred_only_rows ?? "n/a")],
  ["Segment transitions",          String(sim.transitions ?? "n/a")],
  ["Anomalies detected",           String(sim.anomalies ?? "n/a")],
  ["Global RMSE",                  sim.global_rmse != null ? `${Number(sim.global_rmse).toFixed(2)} m` : "n/a"],
  ["Global max error",             sim.global_max_err != null ? `${Number(sim.global_max_err).toFixed(2)} m` : "n/a"],
  ["GPS measurements accepted",    String(sim.gps_accepted ?? 0)],
  ["Camera measurements accepted", String(sim.cam_accepted ?? 0)],
  ["DAS measurements accepted",    String(sim.das_accepted ?? 0)],
]));

// ── 3. PER-TRACK PAGES ───────────────────────────────────────────────
(data.tracks || []).forEach(track => {
  children.push(pageBreak());
  const label = track.vid ? `Track ${track.gid} — vehicle ${track.vid}` : `Track ${track.gid}`;
  children.push(h1(label));
  children.push(hRule());
  children.push(kvTable(track.kv || []));

  // Embedded trajectory figure
  if (track.figure_b64) {
    children.push(space(120));
    try {
      const imgBuf = Buffer.from(track.figure_b64, 'base64');
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({
          data: imgBuf,
          transformation: { width: 540, height: 280 },
        })],
      }));
    } catch (e) {
      children.push(body(`[Figure unavailable: ${e.message}]`, { italic: true }));
    }
  }

  // Anomaly narratives
  if (track.anomaly_narratives && track.anomaly_narratives.length > 0) {
    children.push(space(120));
    children.push(h2("Anomalies", C.red));
    track.anomaly_narratives.forEach(n => {
      children.push(bullet(n, C.red));
    });
  }
});

// ── 4. SEGMENT TRANSITIONS ──────────────────────────────────────────
if (data.transitions && data.transitions.length > 0) {
  children.push(pageBreak());
  children.push(h1("4. Segment Transitions"));
  children.push(hRule());
  children.push(body(`${data.transitions.length} transition(s) across all meaningful tracks.`));
  children.push(space(80));
  children.push(dataTable(
    ["Track", "Time (s)", "From segment", "To segment"],
    data.transitions.map(tr => [tr.gid, tr.t, tr.from_segment, tr.to_segment]),
    [1800, 1500, 3000, 3060],
  ));
}

// ── 5. FILTERED TRACKS ───────────────────────────────────────────────
if (data.skipped_gids && data.skipped_gids.length > 0) {
  children.push(pageBreak());
  children.push(h1("5. Filtered Tracks"));
  children.push(hRule());
  children.push(body(
    `${data.skipped_gids.length} track(s) excluded from this report ` +
    "(below duration, row-count, or distance thresholds)."
  ));
  children.push(space(80));
  children.push(dataTable(
    ["Track ID", "Skip reason"],
    data.skipped_gids.map(s => [s.gid, s.reason]),
    [2000, 7360],
  ));
}

// ── 6. DIAGNOSTIC ANALYSIS SECTIONS ─────────────────────────────────
if (data.analysis_sections && data.analysis_sections.length > 0) {
  children.push(pageBreak());
  children.push(h1("6. Diagnostic Analysis"));
  children.push(hRule());
  children.push(body(
    "The sections below interpret the tracking data — what worked, what looks suspicious, " +
    "what may explain the errors, and what to investigate next. " +
    "Each interpretation lists confirmed evidence, possible hypotheses, and concrete next checks " +
    "separately so the reader can tell facts from speculation.",
    { italic: true }
  ));

  data.analysis_sections.forEach(sec => {
    children.push(pageBreak());
    children.push(h1(sec.title || ""));
    children.push(hRule());

    if (sec.summary) {
      children.push(body(sec.summary));
    }

    if (sec.table && sec.table.headers && sec.table.rows && sec.table.rows.length > 0) {
      children.push(space(80));
      children.push(dataTable(sec.table.headers, sec.table.rows));
      children.push(space(80));
    }

    (sec.bullets || []).forEach(b => {
      children.push(bullet(b));
    });

    (sec.interpretations || []).forEach(interp => {
      const sev = interp.severity || "info";
      const col = sevColor(sev);
      const marker = sevMarker(sev);

      children.push(space(120));
      children.push(h2(`${marker}  ${interp.label || ""}`, col));

      if (interp.technical) {
        children.push(bodyRuns([
          { text: "What happened: ", bold: true },
          { text: interp.technical },
        ]));
      }
      if (interp.meaning) {
        children.push(bodyRuns([
          { text: "What it probably means: ", bold: true },
          { text: interp.meaning },
        ]));
      }
      if (interp.why_matters) {
        children.push(bodyRuns([
          { text: "Why it matters: ", bold: true },
          { text: interp.why_matters },
        ]));
      }
      if (interp.evidence && interp.evidence.length > 0) {
        children.push(bodyRuns([{ text: "Confirmed evidence:", bold: true }]));
        interp.evidence.forEach(e => children.push(bullet(e, C.grey)));
      }
      if (interp.hypotheses && interp.hypotheses.length > 0) {
        children.push(bodyRuns([{ text: "Possible explanations (hypotheses):", bold: true }]));
        interp.hypotheses.forEach(h => children.push(bullet(h, C.grey)));
      }
      if (interp.next_checks && interp.next_checks.length > 0) {
        children.push(bodyRuns([{ text: "Recommended next checks:", bold: true }]));
        interp.next_checks.forEach(n => children.push(bullet(n, C.blue)));
      }
    });
  });
}

// ── 7. ENHANCED DIAGNOSTICS ──────────────────────────────────────────

// 7.1 Sensor σ Calibration Check
if (enh.sigma_calibration) {
  const cal = enh.sigma_calibration;
  children.push(pageBreak());
  children.push(h1("7. Sensor σ Calibration Check"));
  children.push(hRule());
  children.push(body(
    "Compares each sensor's declared measurement noise (sigma_m) against the actual error " +
    "computed from ground truth. A well-calibrated sensor has mean actual error ≈ mean declared σ. " +
    "If actual error exceeds 1.2×σ, the Kalman filter is over-trusting that sensor " +
    "and the R matrix entry should be increased.",
    { italic: true }
  ));
  children.push(space(80));

  if (cal.headers && cal.rows) {
    children.push(dataTable(cal.headers, cal.rows));
    children.push(space(80));
  }

  (cal.flags || []).forEach(flag => {
    const col = flag.sev === "warning" ? C.red : flag.sev === "info" ? C.blue : C.green;
    children.push(bullet(flag.text, col));
  });
}

// 7.2 Per-Axis Error Breakdown
if (enh.axis_errors && enh.axis_errors.length > 0) {
  children.push(pageBreak());
  children.push(h1("8. Per-Axis Error Breakdown (X vs Y)"));
  children.push(hRule());
  children.push(body(
    "Scalar RMSE hides directional bias. A dominant Y error (lateral) usually points to " +
    "insufficient lateral process noise (Q_yy too small) or missing sensors that constrain " +
    "cross-track position. A dominant X error (longitudinal) suggests speed-model mismatch.",
    { italic: true }
  ));
  children.push(space(80));

  enh.axis_errors.forEach(ax => {
    const trackLabel = ax.vid ? `Track ${ax.gid} — ${ax.vid}` : `Track ${ax.gid}`;
    children.push(h2(trackLabel));
    if (ax.headers && ax.rows) {
      children.push(dataTable(ax.headers, ax.rows, [2400, 2000, 1600, 1960, 1400]));
      children.push(space(80));
    }
    if (ax.verdict) {
      const col = ax.verdict.startsWith("⚠") ? C.amber : C.green;
      children.push(bullet(ax.verdict, col));
    }
    children.push(space(80));
  });
}

// 7.3 Sensor Coverage Timeline
if (enh.coverage_timeline && enh.coverage_timeline.length > 0) {
  children.push(pageBreak());
  children.push(h1("9. Sensor Coverage Timeline"));
  children.push(hRule());
  children.push(body(
    "Each row represents one second of simulation time. " +
    "✓ = at least one accepted measurement from that sensor arrived; " +
    "— = that sensor was silent (filter coasted on prediction). " +
    "Seconds where ALL sensors are silent are marked as prediction-only — " +
    "these are the main source of position drift.",
    { italic: true }
  ));
  children.push(space(80));

  enh.coverage_timeline.forEach(tl => {
    const trackLabel = tl.vid ? `Track ${tl.gid} — ${tl.vid}` : `Track ${tl.gid}`;
    children.push(h2(trackLabel));
    if (tl.headers && tl.rows) {
      children.push(dataTable(tl.headers, tl.rows, [1500, 1500, 1800, 1500, 3060]));
      children.push(space(60));
    }
    const pct = tl.total_seconds > 0
      ? Math.round((100 * tl.dead_seconds) / tl.total_seconds)
      : 0;
    children.push(body(
      `Dead seconds (all sensors silent): ${tl.dead_seconds} / ${tl.total_seconds} (${pct}%)`,
      { italic: true }
    ));
    children.push(space(80));
  });
}

// 7.4 Cold-Start Latency
if (enh.cold_start) {
  children.push(pageBreak());
  children.push(h1("10. Cold-Start Latency"));
  children.push(hRule());
  children.push(body(
    "Time elapsed from the first trajectory row to the first moment the position error drops below 0.5 m. " +
    "A long cold-start indicates the filter is initialized far from the true position, " +
    "or the initial state covariance P₀ is too small to pull the estimate toward truth quickly. " +
    "Tuning the initial state (x₀, P₀) or adding a high-weight early measurement can shorten this window.",
    { italic: true }
  ));
  children.push(space(80));

  const cs = enh.cold_start;
  if (cs.headers && cs.rows) {
    children.push(dataTable(
      cs.headers, cs.rows,
      [1200, 1200, 1400, 2000, 1960, 1600],
    ));
  }
}

// 7.5 Speed vs Position Error
if (enh.speed_vs_error) {
  children.push(pageBreak());
  children.push(h1("11. Speed vs Position Error"));
  children.push(hRule());
  children.push(body(
    "Position error is bucketed by the estimated vehicle speed at each Kalman step. " +
    "If error grows with speed the process noise Q or the motion model is misspecified " +
    "for high-speed dynamics. If error is roughly flat across speeds, sensor dropout — " +
    "not model mismatch — is the dominant error driver.",
    { italic: true }
  ));
  children.push(space(80));

  const spd = enh.speed_vs_error;
  if (spd.headers && spd.rows) {
    children.push(dataTable(spd.headers, spd.rows, [2000, 1400, 2000, 2000, 1960]));
    children.push(space(80));
  }
  if (spd.verdict) {
    const col = spd.verdict.startsWith("⚠") ? C.amber : C.green;
    children.push(bullet(spd.verdict, col));
  }
}

// ──────────────────────────────────────────────────────────────────────
// RENDER TO FILE
// ──────────────────────────────────────────────────────────────────────

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0,
        format: "bullet",
        text: "•",
        alignment: AlignmentType.LEFT,
        style: {
          paragraph: { indent: { left: 360, hanging: 360 } },
          run: { font: "Arial" },
        },
      }],
    }],
  },
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 22, color: C.grey },
      },
    },
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 720, bottom: 720, left: 900, right: 900 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, buf);
  console.log(`Report written to ${outPath} (${buf.length} bytes)`);
}).catch(err => {
  console.error('Error generating report:', err);
  process.exit(1);
});
