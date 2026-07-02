"""
organize_north_tlv.py
=====================
Reads north_tlv_map.sim.json + north_tlv_metadata.json and produces:

  tiles_north_tlv/
    overview.png              — full map coloured by region, numbered
    001/
      map.sim.json            — SimStudio-ready tile (open this in SimStudio)
      region_info.json        — street names, speed limits, traversal estimate
    002/ ...

Tiling strategy
---------------
BFS growth from the longest uncovered segment. Each tile collects
MIN_SEGS–MAX_SEGS connected segments (default 10–15). Tiles smaller
than MIN_SEGS are discarded (isolated road stubs).

This gives geographically compact "blocks" similar to the Florentin split,
but sized for 20–60 s simulations.

Only arterial/distributor roads are tiled (primary / secondary / tertiary /
trunk / living_street). Residential streets are included in the map file
but not used as tile seeds, to avoid thousands of tiny residential patches.

Usage
-----
  cd maps/
  python ../scripts/organize_north_tlv.py
"""

import json
import math
import os
import colorsys
from collections import defaultdict, deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# =============================================================================
# CONFIGURATION  — adjust here if needed
# =============================================================================

# Match MAP_HALF in fetch_north_tlv_map.py — change "south"→"north" for second run
MAP_HALF   = "south"
INPUT_MAP  = f"north_tlv_{MAP_HALF}_map.sim.json"
INPUT_META = f"north_tlv_{MAP_HALF}_metadata.json"
OUTPUT_DIR = f"tiles_north_tlv_{MAP_HALF}"

# How many segments per tile
MIN_SEGS = 10
MAX_SEGS = 15

# Road types used as tile seeds (major roads only — keeps tile count reasonable)
# Residential segments are included in the tile if they are adjacent, but
# are not used as seeds.
SEED_HIGHWAY_TYPES = {
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    'trunk', 'trunk_link',
    'living_street',
}

# Minimum individual segment length to include (skip micro-stubs < 8 m)
MIN_SEG_M = 8.0

# Image
IMG_DPI        = 200
IMG_INCHES     = 20       # square canvas
FONT_SIZE      = 7        # region-number label size (pt)
COLOUR_SEED    = 7        # fixed seed → reproducible colours

# =============================================================================
# HELPERS
# =============================================================================

def seg_length(pts):
    t = 0.0
    for i in range(1, len(pts)):
        t += math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
    return t

def save_tile_image(folder, idx, tile_segs, streets_seen, avg_speed_kmh):
    """Per-tile overview PNG: each unique street gets its own colour."""

    if not tile_segs:
        return

    # ── 1. Assign a distinct colour to each unique street name ──────────────
    # Build label → [seg_ids] mapping
    label_to_sids = {}
    for sid in tile_segs:
        label = name_en_for(sid) or meta_segs.get(sid, {}).get('name_he') or 'Unknown'
        label_to_sids.setdefault(label, []).append(sid)

    unique_labels = list(label_to_sids.keys())
    n_streets = max(len(unique_labels), 1)

    # Use a perceptually-distinct colormap; wrap around if >10 streets
    PALETTE = [
        '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
        '#42d4f4','#f032e6','#bfef45','#fabed4','#469990',
        '#dcbeff','#9A6324','#fffac8','#800000','#aaffc3',
        '#808000','#ffd8b1','#000075','#a9a9a9','#000000',
    ]
    label_colour = {lbl: PALETTE[i % len(PALETTE)]
                    for i, lbl in enumerate(unique_labels)}

    # ── 2. Build per-street polyline lists ───────────────────────────────────
    label_lines = {lbl: [] for lbl in unique_labels}
    for sid in tile_segs:
        label = name_en_for(sid) or meta_segs.get(sid, {}).get('name_he') or 'Unknown'
        pts_plot = [[p[0], -p[1]] for p in all_segs[sid]['points']]
        label_lines[label].append(pts_plot)

    # ── 3. Axis limits ───────────────────────────────────────────────────────
    all_pts = [[p[0], -p[1]] for sid in tile_segs
               for p in all_segs[sid]['points']]
    all_x = [p[0] for p in all_pts]
    all_y = [p[1] for p in all_pts]
    span  = max(max(all_x)-min(all_x), max(all_y)-min(all_y), 1)
    pad   = span * 0.14 + 25
    x_min, x_max = min(all_x)-pad, max(all_x)+pad
    y_min, y_max = min(all_y)-pad, max(all_y)+pad

    width_m  = x_max - x_min
    height_m = y_max - y_min
    fig_w    = 9.0
    fig_h    = max(4.0, fig_w * height_m / max(width_m, 1))
    # Cap so the image doesn't become absurdly tall for a single diagonal road
    if fig_h > fig_w * 2.5:
        fig_h = fig_w * 2.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.set_facecolor('#f0f0f0')
    ax.axis('off')

    # ── 4. Draw each street in its colour ────────────────────────────────────
    for lbl, lines in label_lines.items():
        col = label_colour[lbl]
        lc  = LineCollection(lines, colors=col, linewidths=2.5, alpha=0.92,
                             zorder=2)
        ax.add_collection(lc)

    # ── 5. Street-name labels (same colour, readable size) ───────────────────
    # For each street: label on the longest segment, angled along the road
    label_placed = set()
    for sid in tile_segs:
        label = name_en_for(sid) or meta_segs.get(sid, {}).get('name_he') or 'Unknown'
        if label in label_placed:
            continue

        # Pick the longest segment for this label
        best_sid = max(label_to_sids[label],
                       key=lambda s: seg_length(all_segs[s]['points']))
        pts_plot = [[p[0], -p[1]] for p in all_segs[best_sid]['points']]
        slen     = seg_length([[p[0], p[1]] for p in pts_plot])
        if slen < 20:
            continue

        # Midpoint
        n      = len(pts_plot)
        mid    = n // 2
        mx     = (pts_plot[max(mid-1,0)][0] + pts_plot[mid][0]) / 2
        my     = (pts_plot[max(mid-1,0)][1] + pts_plot[mid][1]) / 2

        # Angle along road
        dx = pts_plot[mid][0] - pts_plot[max(mid-1,0)][0]
        dy = pts_plot[mid][1] - pts_plot[max(mid-1,0)][1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle >  90: angle -= 180
        if angle < -90: angle += 180

        col = label_colour[label]
        ax.text(mx, my, label,
                fontsize=8, color=col, fontweight='bold',
                rotation=angle, rotation_mode='anchor',
                ha='center', va='bottom', zorder=3,
                bbox=dict(facecolor='white', edgecolor=col,
                          alpha=0.85, pad=1.5,
                          linewidth=0.8, boxstyle='round,pad=0.3'))
        label_placed.add(label)

    # ── 6. Legend (colour swatch + name) in a compact box ───────────────────
    legend_handles = [
        plt.Line2D([0], [0], color=label_colour[lbl], linewidth=3, label=lbl)
        for lbl in unique_labels
    ]
    if legend_handles:
        ax.legend(handles=legend_handles,
                  loc='lower left', fontsize=7,
                  framealpha=0.90, edgecolor='#aaaaaa',
                  ncol=max(1, len(unique_labels)//8 + 1))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='datalim')

    sim_name = f"region_{idx:03d}.sim.json"
    ax.set_title(
        f"Region {idx:03d}  —  North Tel Aviv ({MAP_HALF})\n"
        f"File: {sim_name}  |  {len(tile_segs)} segments  |  avg speed {avg_speed_kmh:.0f} km/h",
        fontsize=10, fontweight='bold', pad=8, color='#111111'
    )

    plt.tight_layout(pad=0.5)
    out = os.path.join(folder, f"region_{idx:03d}_overview.png")
    plt.savefig(out, dpi=140, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

def centroid_of(seg_ids, all_segs):
    xs, ys = [], []
    for sid in seg_ids:
        for x, y in all_segs[sid]['points']:
            xs.append(x); ys.append(y)
    return float(np.mean(xs)), float(np.mean(ys))

# ---------------------------------------------------------------------------
# Hebrew → English helpers
# ---------------------------------------------------------------------------

# Word-level overrides for common Israeli street type prefixes/suffixes
_HE_WORDS = {
    'רחוב': 'St.', 'שדרות': 'Blvd.', 'שדרת': 'Blvd.', 'דרך': 'Rd.',
    'כיכר': 'Sq.', 'שביל': 'Path', 'סמטא': 'Alley', 'סמטת': 'Alley',
    'גן': 'Garden', 'גנים': 'Gardens', 'מעלה': 'Rise', 'ירידה': 'Descent',
    'קרן': 'Corner', 'מחלף': 'Interchange', 'גשר': 'Bridge',
    'פארק': 'Park', 'שכונת': 'Neighborhood',
}

# Character-level Hebrew→Latin transliteration (standard Israeli convention)
_HE_CHARS = {
    'א': '', 'ב': 'v', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'u',
    'ז': 'z', 'ח': 'ch', 'ט': 't', 'י': 'y', 'כ': 'k', 'ך': 'k',
    'ל': 'l', 'מ': 'm', 'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's',
    'ע': '', 'פ': 'p', 'ף': 'f', 'צ': 'tz', 'ץ': 'tz', 'ק': 'k',
    'ר': 'r', 'ש': 'sh', 'ת': 't',
}

def _transliterate_he(text: str) -> str:
    """Transliterate a Hebrew string to Latin characters."""
    if not text:
        return text
    words = text.split()
    result = []
    for w in words:
        if w in _HE_WORDS:
            result.append(_HE_WORDS[w])
        else:
            letters = ''.join(_HE_CHARS.get(c, c if c.isascii() else '') for c in w)
            # Capitalise first real letter
            letters = letters.strip()
            if letters:
                result.append(letters[0].upper() + letters[1:])
    out = ' '.join(result).strip()
    # Collapse double spaces
    import re as _re
    out = _re.sub(r'\s+', ' ', out)
    return out or text   # fallback: return original if transliteration is empty

def name_en_for(sid):
    """Return the best English name for a segment, transliterating if necessary."""
    entry = meta_segs.get(sid, {})
    en = entry.get('name_en')
    if en:
        return en
    he = entry.get('name_he') or entry.get('name')
    if he:
        return _transliterate_he(he)
    return None

# ---------------------------------------------------------------------------

def build_tile_world(world, seg_ids):
    segs  = {sid: world['segments'][sid] for sid in seg_ids}
    nids  = {s['n0'] for s in segs.values()} | {s['n1'] for s in segs.values()}
    nodes = {n: world['nodes'][n] for n in nids if n in world['nodes']}
    lanes = {}
    for sid, seg in segs.items():
        lid = f"{sid}_fwd_lane1"
        if lid in world.get('lanes', {}):
            lanes[lid] = world['lanes'][lid]
        else:
            lanes[lid] = {"id": lid, "segment_id": sid,
                          "offset_index": 0, "polyline": seg['points']}

    # Fix speed consistency + strip any non-SimStudio fields
    ALLOWED = {'id','n0','n1','lanes','lane_width','speed_limit_mps',
               'points','one_way','traffic_level'}
    enriched_segs = {}
    for sid, seg in segs.items():
        entry = meta_segs.get(sid, {})

        # Start from only the allowed SimStudio fields
        s = {k: v for k, v in seg.items() if k in ALLOWED}

        # Speed consistency: re-derive from metadata so sim == metadata
        try:
            meta_kmh = float(entry.get('maxspeed', 0))
            if meta_kmh > 0:
                s['speed_limit_mps'] = round(meta_kmh / 3.6, 3)
        except (ValueError, TypeError):
            pass   # keep original if metadata speed is missing / malformed

        enriched_segs[sid] = s

    return {"nodes": nodes, "segments": enriched_segs, "lanes": lanes,
            "vehicles": {}, "gps": {}, "cameras": {}, "das": {}}

# =============================================================================
# 1. LOAD
# =============================================================================

print("Loading map …")
with open(INPUT_MAP, encoding='utf-8') as f:
    world = json.load(f)
all_segs  = world['segments']
all_nodes = world['nodes']
print(f"  {len(all_segs):,} segments, {len(all_nodes):,} nodes")

print("Loading metadata …")
with open(INPUT_META, encoding='utf-8') as f:
    meta_raw = json.load(f)
meta_segs = meta_raw.get('segments', {})
origin    = meta_raw.get('origin', {})
bbox_raw  = meta_raw.get('bbox',   {})
lat0_map  = origin.get('lat0', 0.0)
lon0_map  = origin.get('lon0', 0.0)
print(f"  {len(meta_segs):,} metadata entries")

# Compute bbox limits in local coord space (used to clip image and filter segments)
_R_E = 6_378_137.0
_cl  = math.cos(math.radians(lat0_map))
_IMG_PAD = 80   # metres padding around bbox in the image

if bbox_raw:
    _x_clip_l = math.radians(bbox_raw['lon_w'] - lon0_map) * _R_E * _cl - _IMG_PAD
    _x_clip_r = math.radians(bbox_raw['lon_e'] - lon0_map) * _R_E * _cl + _IMG_PAD
    _y_clip_b = math.radians(bbox_raw['lat_s'] - lat0_map) * _R_E - _IMG_PAD  # negated later
    _y_clip_t = math.radians(bbox_raw['lat_n'] - lat0_map) * _R_E + _IMG_PAD
    print(f"  Bbox clip: x=[{_x_clip_l:.0f}, {_x_clip_r:.0f}] m  "
          f"y_lat=[{bbox_raw['lat_s']}, {bbox_raw['lat_n']}]")
else:
    _x_clip_l = _x_clip_r = _y_clip_b = _y_clip_t = None
    print("  Warning: no bbox in metadata — image will auto-scale")

# =============================================================================
# 2. IDENTIFY SEED-ELIGIBLE SEGMENTS
# =============================================================================

seg_len_cache = {}
for sid, seg in all_segs.items():
    pts = seg.get('points', [])
    seg_len_cache[sid] = seg_length(pts)

def highway_of(sid):
    return str(meta_segs.get(sid, {}).get('highway') or 'residential').lower()

def speed_mps_of(sid):
    return float(all_segs[sid].get('speed_limit_mps', 13.9))

def seg_in_bbox(sid):
    """True if ALL points of the segment fall within the bbox (+ small buffer)."""
    if _x_clip_l is None:
        return True
    buf = 200  # allow segments up to 200 m outside the bbox edge
    for x, y in all_segs[sid].get('points', []):
        # x: west = negative, east = positive — same sign as _x_clip_l/_x_clip_r
        if x < _x_clip_l - buf or x > _x_clip_r + buf:
            return False
        # y (local sim coords): south = positive, north = negative
        #   _y_clip_b = radians(lat_s - lat0)*R - pad  → negative  (lat_s < lat0)
        #   _y_clip_t = radians(lat_n - lat0)*R + pad  → positive  (lat_n > lat0)
        # y at south boundary ≈ -_y_clip_b (positive)
        # y at north boundary ≈ -_y_clip_t (negative)
        # valid y range (with extra buf): [-_y_clip_t - buf, -_y_clip_b + buf]
        if y < -_y_clip_t - buf or y > -_y_clip_b + buf:
            return False
    return True

# All segments long enough AND within bbox
all_valid = {sid for sid, seg in all_segs.items()
             if seg_len_cache[sid] >= MIN_SEG_M and seg_in_bbox(sid)}

# Segments eligible as BFS seeds (major roads only)
seed_eligible = {sid for sid in all_valid if highway_of(sid) in SEED_HIGHWAY_TYPES}

print(f"  Valid segments  : {len(all_valid):,}")
print(f"  Seed-eligible   : {len(seed_eligible):,}  (major roads)")

# =============================================================================
# 3. ADJACENCY  (node → segment list, for all valid segments)
# =============================================================================

node_to_segs = defaultdict(list)
for sid in all_valid:
    seg = all_segs[sid]
    node_to_segs[seg['n0']].append(sid)
    node_to_segs[seg['n1']].append(sid)

# =============================================================================
# 4. BFS TILING
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

uncovered_seeds = set(seed_eligible)   # seeds we haven't started from
used_any        = set()                # all segments assigned to any tile
tile_list       = []                   # final tiles: list of seg_id lists
skipped         = 0

print(f"\nTiling with {MIN_SEGS}–{MAX_SEGS} segments per tile …")

while uncovered_seeds:
    # Seed = longest uncovered seed segment
    seed = max(uncovered_seeds, key=lambda s: seg_len_cache[s])
    uncovered_seeds.discard(seed)

    if seed in used_any:
        continue

    # BFS — expand through ALL adjacent valid segments (not just seeds)
    tile_segs    = []
    visited      = set()
    queue        = deque([seed])

    while queue and len(tile_segs) < MAX_SEGS:
        sid = queue.popleft()
        if sid in visited or sid in used_any:
            continue
        visited.add(sid)
        tile_segs.append(sid)

        if len(tile_segs) >= MAX_SEGS:
            break

        for nid in (all_segs[sid]['n0'], all_segs[sid]['n1']):
            for nb in node_to_segs[nid]:
                if nb not in visited and nb not in used_any:
                    queue.append(nb)

    if len(tile_segs) < MIN_SEGS:
        # Too small — still mark as used so we don't revisit
        used_any |= visited
        skipped  += 1
        continue

    used_any |= set(tile_segs)
    # Also remove any seed-eligible segments consumed by this tile
    uncovered_seeds -= set(tile_segs)
    tile_list.append(tile_segs)

print(f"  {len(tile_list)} tiles  |  {skipped} small clusters skipped")

# =============================================================================
# 4b. SORT TILES GEOGRAPHICALLY (left→right, top→bottom)
#     so that region 1 is top-left and adjacent numbers are spatially close.
# =============================================================================

def _tile_centroid(seg_ids):
    xs, ys = [], []
    for sid in seg_ids:
        for x, y in all_segs[sid]['points']:
            xs.append(x); ys.append(y)
    return float(np.mean(xs)), float(np.mean(ys))

# Compute centroid for every tile
tile_centroids = [_tile_centroid(segs) for segs in tile_list]

# Sort into a grid: divide x into ~8 columns, sort each column top→bottom
# Use the x range to define column width
all_cx = [c[0] for c in tile_centroids]
all_cy = [c[1] for c in tile_centroids]
x_span = max(all_cx) - min(all_cx) if len(all_cx) > 1 else 1
NUM_COLS = 8
col_width = x_span / NUM_COLS

def _sort_key(pair):
    _, (cx, cy) = pair
    col = int((cx - min(all_cx)) / col_width)
    col = min(col, NUM_COLS - 1)
    # cy is negative for north (north-up display); sort north→south = ascending cy
    return (col, cy)

sorted_pairs = sorted(zip(tile_list, tile_centroids), key=_sort_key)
tile_list = [p[0] for p in sorted_pairs]

print(f"  Tiles re-sorted geographically (left→right, top→bottom)")

# =============================================================================
# 5. WRITE TILE FOLDERS
# =============================================================================

print("\nWriting folders …")
tile_meta_list = []

for idx, tile_segs in enumerate(tile_list, start=1):
    folder = os.path.join(OUTPUT_DIR, f"{idx:03d}")
    os.makedirs(folder, exist_ok=True)

    # region_NNN.sim.json  (named so the tile number is visible in SimStudio)
    sim_filename = f"region_{idx:03d}.sim.json"
    tile_world = build_tile_world(world, tile_segs)
    with open(os.path.join(folder, sim_filename), 'w', encoding='utf-8') as f:
        json.dump(tile_world, f, ensure_ascii=False, indent=2)

    # region_info.json
    tile_len  = sum(seg_len_cache[sid] for sid in tile_segs)
    avg_speed = sum(speed_mps_of(sid) for sid in tile_segs) / len(tile_segs)
    est_trav  = round(tile_len / avg_speed, 1)

    streets_seen  = {}
    speed_counts  = defaultdict(int)
    for sid in tile_segs:
        entry  = meta_segs.get(sid, {})
        way_id = entry.get('osm_way_id', 0)
        if way_id and way_id not in streets_seen:
            he   = entry.get('name_he') or entry.get('name')
            en   = name_en_for(sid)          # translated / transliterated
            streets_seen[way_id] = {
                "name_he":   he,
                "name_en":   en,
                "name":      en or he,       # display name: English preferred
                "highway":   entry.get('highway'),
                "speed_kmh": entry.get('maxspeed'),
                "oneway":    entry.get('oneway'),
            }
        kmh = entry.get('maxspeed')
        if kmh:
            speed_counts[str(kmh)] += 1

    cx, cy = centroid_of(tile_segs, all_segs)

    region_info = {
        "region_id":          idx,
        "num_segments":       len(tile_segs),
        "total_road_m":       round(tile_len, 1),
        "avg_speed_mps":      round(avg_speed, 3),
        "avg_speed_kmh":      round(avg_speed * 3.6, 1),
        "est_traversal_s":    est_trav,
        "centroid_x":         round(cx, 2),
        "centroid_y":         round(cy, 2),
        "speed_limit_counts": dict(sorted(speed_counts.items())),
        "streets":            list(streets_seen.values()),
        "segment_ids":        tile_segs,
    }
    with open(os.path.join(folder, "region_info.json"), 'w', encoding='utf-8') as f:
        json.dump(region_info, f, ensure_ascii=False, indent=2)

    # Per-tile overview PNG
    save_tile_image(folder, idx, tile_segs, streets_seen,
                    avg_speed_kmh=round(avg_speed * 3.6, 1))

    tile_meta_list.append({"id": idx, "seg_ids": tile_segs,
                            "cx": cx, "cy": cy,
                            "length_m": tile_len, "speed": avg_speed})

    if idx % 25 == 0:
        print(f"  … {idx} written")

print(f"  Done — {len(tile_list)} folders in {OUTPUT_DIR}/")

# =============================================================================
# 6. OVERVIEW IMAGE
# =============================================================================

print("\nGenerating overview.png …")

n = len(tile_meta_list)
rng = np.random.default_rng(COLOUR_SEED)

hues  = np.linspace(0, 1, n, endpoint=False)
rng.shuffle(hues)
sats  = rng.uniform(0.55, 0.85, n)
vals  = rng.uniform(0.60, 0.90, n)
colours = [colorsys.hsv_to_rgb(hues[i], sats[i], vals[i]) for i in range(n)]

# Axis limits: use bbox if available, else fall back to data percentiles.
# Negate y so north is at the top.
if _x_clip_l is not None:
    x_min, x_max = _x_clip_l, _x_clip_r
    # _y_clip values are in lat-space; convert to display (negated) space
    y_min = -_y_clip_t   # north in display space (smaller value before negation → larger after)
    y_max = -_y_clip_b   # south in display space
else:
    all_xs, all_ys_neg = [], []
    for seg in all_segs.values():
        for x, y in seg.get('points', []):
            all_xs.append(x); all_ys_neg.append(-y)
    import numpy as _np
    x_min = float(_np.percentile(all_xs, 2))  - 150
    x_max = float(_np.percentile(all_xs, 98)) + 150
    y_min = float(_np.percentile(all_ys_neg, 2))  - 150
    y_max = float(_np.percentile(all_ys_neg, 98)) + 150

width_m  = x_max - x_min
height_m = y_max - y_min
fig_w_in = IMG_INCHES
fig_h_in = max(4, fig_w_in * height_m / width_m)

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), facecolor='#eef0f0')
ax.set_facecolor('#eef0f0')
ax.axis('off')

# Draw each tile with negated y
for i, tm in enumerate(tile_meta_list):
    lines = [[[p[0], -p[1]] for p in all_segs[sid]['points']]
             for sid in tm['seg_ids']]
    lc = LineCollection(lines, colors=[colours[i]], linewidths=2.0, alpha=0.90)
    ax.add_collection(lc)

# Region number labels (negate cy too)
for i, tm in enumerate(tile_meta_list):
    ax.text(tm['cx'], -tm['cy'], str(tm['id']),
            fontsize=FONT_SIZE, color='black',
            ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.60, pad=0.8, boxstyle='round,pad=0.2'))

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.title(f"North Tel Aviv ({MAP_HALF}) — Simulation Regions  [{n} tiles]",
          fontsize=13, pad=10, color='#222222', fontweight='bold')
plt.tight_layout(pad=0.5)

out_img = os.path.join(OUTPUT_DIR, "overview.png")
plt.savefig(out_img, dpi=IMG_DPI, bbox_inches='tight',
            facecolor='#eef0f0', edgecolor='none')
plt.close()
print(f"  Saved: {out_img}")

# =============================================================================
# 7. SUMMARY
# =============================================================================

lengths    = [t['length_m'] for t in tile_meta_list]
speeds     = [t['speed']    for t in tile_meta_list]
traversals = [l/s for l, s in zip(lengths, speeds)]

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Regions created  : {len(tile_list)}
  Skipped (tiny)   : {skipped}

  Segments / tile  : {MIN_SEGS}–{MAX_SEGS}

  Road length / tile:
    min   {min(lengths):.0f} m
    mean  {sum(lengths)/len(lengths):.0f} m
    max   {max(lengths):.0f} m

  Estimated traversal (total road / avg speed):
    min   {min(traversals):.0f} s
    mean  {sum(traversals)/len(traversals):.0f} s
    max   {max(traversals):.0f} s

  Output : {OUTPUT_DIR}/
  Image  : {OUTPUT_DIR}/overview.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
