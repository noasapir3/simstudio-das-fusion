"""
tile_north_tlv.py
=================
Carves a large SimStudio map (north_tlv_map.sim.json) into simulation tiles,
each containing enough connected road for 30–60 second vehicle scenarios.

Algorithm
---------
1. Load the full map.
2. Filter out road types unsuitable for simulation (motorways, alleys, etc.).
3. Build a road-graph: nodes → which segments touch them.
4. BFS-grow tiles from uncovered seed segments until each tile has at least
   MIN_TILE_LENGTH_M metres of connected road, capped at MAX_TILE_LENGTH_M.
5. Write each tile as a .sim.json file (same format as the Florentin tiles).

Usage
-----
    cd maps/
    python ../scripts/tile_north_tlv.py

Output
------
    maps/tiles_north_tlv/tile_NNN_clean.sim.json   (one file per tile)
"""

import json
import math
import os
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_MAP  = "north_tlv_map.sim.json"
OUTPUT_DIR = "tiles_north_tlv"

# Tile road-length targets (metres of total segment polyline per tile)
# At 50 km/h (13.9 m/s): 30 s = 417 m, 60 s = 833 m
# At 30 km/h ( 8.3 m/s): 30 s = 250 m, 60 s = 500 m
# → target 500–900 m so both speed classes get usable scenarios
MIN_TILE_LENGTH_M = 500.0
MAX_TILE_LENGTH_M = 900.0

# Road types to keep (exclude motorways, service alleys, unclassified paths)
KEEP_HIGHWAY_TYPES = {
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    'residential',
    'living_street',
    'trunk', 'trunk_link',
}

# Minimum individual segment length to include (skip tiny stubs)
MIN_SEGMENT_LENGTH_M = 10.0

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _seg_length(pts):
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        total += math.hypot(dx, dy)
    return total


def _build_world_slice(world, seg_ids):
    """Extract a subset of the world dict containing only the given segment IDs."""
    seg_ids = set(seg_ids)

    # Gather segments
    segments = {sid: world['segments'][sid] for sid in seg_ids}

    # Gather nodes referenced by those segments
    node_ids = set()
    for seg in segments.values():
        node_ids.add(seg['n0'])
        node_ids.add(seg['n1'])
    nodes = {nid: world['nodes'][nid] for nid in node_ids if nid in world['nodes']}

    # Rebuild lanes (one lane per segment, same polyline)
    lanes = {}
    for sid, seg in segments.items():
        lid = f"{sid}_fwd_lane1"
        # Use existing lane if present, else reconstruct
        if lid in world.get('lanes', {}):
            lanes[lid] = world['lanes'][lid]
        else:
            lanes[lid] = {
                "id":           lid,
                "segment_id":   sid,
                "offset_index": 0,
                "polyline":     seg['points'],
            }

    return {
        "nodes":    nodes,
        "segments": segments,
        "lanes":    lanes,
        "vehicles": {},
        "gps":      {},
        "cameras":  {},
        "das":      {},
    }

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

print(f"Loading {INPUT_MAP} …")
with open(INPUT_MAP, encoding='utf-8') as f:
    world = json.load(f)

all_segments = world['segments']
all_nodes    = world['nodes']
print(f"  Full map: {len(all_segments)} segments, {len(all_nodes)} nodes")

# ---------------------------------------------------------------------------
# 2. FILTER
# ---------------------------------------------------------------------------

# We need the metadata file to know highway types.
# If it exists alongside the map, load it; otherwise skip the filter.
META_FILE = INPUT_MAP.replace('_map.sim.json', '_metadata.json')
highway_map = {}   # seg_id → highway type string

if os.path.exists(META_FILE):
    with open(META_FILE, encoding='utf-8') as f:
        meta = json.load(f)
    for sid, entry in meta.get('segments', {}).items():
        highway_map[sid] = str(entry.get('highway') or 'residential')
    print(f"  Metadata loaded: {len(highway_map)} entries")
else:
    print(f"  Warning: {META_FILE} not found — all segments kept (no highway filter)")

def _keep(seg_id, seg):
    pts = seg.get('points', [])
    if _seg_length(pts) < MIN_SEGMENT_LENGTH_M:
        return False
    if highway_map:
        hw = highway_map.get(seg_id, 'residential')
        if hw not in KEEP_HIGHWAY_TYPES:
            return False
    return True

filtered_ids = [sid for sid, seg in all_segments.items() if _keep(sid, seg)]
filtered_len_m = sum(_seg_length(all_segments[sid]['points']) for sid in filtered_ids)
print(f"  After filter: {len(filtered_ids)} segments, "
      f"{filtered_len_m/1000:.1f} km total road")

# ---------------------------------------------------------------------------
# 3. BUILD ADJACENCY (node → segment list)
# ---------------------------------------------------------------------------

node_to_segs = defaultdict(list)   # node_id → [seg_id, ...]
for sid in filtered_ids:
    seg = all_segments[sid]
    node_to_segs[seg['n0']].append(sid)
    node_to_segs[seg['n1']].append(sid)

seg_length_cache = {sid: _seg_length(all_segments[sid]['points']) for sid in filtered_ids}

# ---------------------------------------------------------------------------
# 4. BFS TILING
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

uncovered = set(filtered_ids)
tile_num  = 1
tiles_written = 0
skipped_short = 0

print(f"\nTiling …  (target {MIN_TILE_LENGTH_M}–{MAX_TILE_LENGTH_M} m per tile)")

while uncovered:
    # Pick a seed — prefer longer segments as anchors
    seed = max(uncovered, key=lambda s: seg_length_cache[s])

    tile_segs = []
    tile_len  = 0.0
    visited_segs  = set()
    visited_nodes = set()
    queue = deque([seed])

    while queue and tile_len < MAX_TILE_LENGTH_M:
        sid = queue.popleft()
        if sid in visited_segs or sid not in uncovered:
            continue
        visited_segs.add(sid)

        seg = all_segments[sid]
        slen = seg_length_cache[sid]
        tile_segs.append(sid)
        tile_len += slen

        if tile_len >= MAX_TILE_LENGTH_M:
            break

        # Expand from both endpoint nodes
        for nid in (seg['n0'], seg['n1']):
            if nid in visited_nodes:
                continue
            visited_nodes.add(nid)
            for neighbour_sid in node_to_segs[nid]:
                if neighbour_sid not in visited_segs and neighbour_sid in uncovered:
                    queue.append(neighbour_sid)

    # Mark all BFS-reached segments as covered (even if tile is too short)
    uncovered -= visited_segs

    # Only write tiles that meet the minimum length
    if tile_len < MIN_TILE_LENGTH_M:
        skipped_short += 1
        continue

    tile_world = _build_world_slice(world, tile_segs)
    fname = os.path.join(OUTPUT_DIR, f"tile_{tile_num:03d}_clean.sim.json")
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(tile_world, f, ensure_ascii=False, indent=2)

    tile_num  += 1
    tiles_written += 1

    if tiles_written % 50 == 0:
        print(f"  … {tiles_written} tiles written, {len(uncovered)} segments remaining")

# ---------------------------------------------------------------------------
# 5. SUMMARY
# ---------------------------------------------------------------------------

print(f"""
Done.
  Tiles written : {tiles_written}  →  {OUTPUT_DIR}/
  Skipped (too short) : {skipped_short} clusters
  Output format : tile_NNN_clean.sim.json

Next steps:
  1. Open a few tiles in SimStudio to verify the road network looks correct.
  2. For each tile, add DAS sensors, cameras, and vehicles in SimStudio.
  3. Run the anomaly-model scenario generator to produce normal + anomaly runs.
""")
