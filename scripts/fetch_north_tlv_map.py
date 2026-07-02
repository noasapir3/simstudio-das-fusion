"""
fetch_north_tlv_map.py
======================
Downloads the road network for any neighborhood from OpenStreetMap and
produces two output files:

  1. <area>_map.sim.json   — SimStudio map file (same format as
                              full_map_florentin_og_.json)
  2. <area>_metadata.json  — OSM metadata file (same format as
                              notbig_florentin_mata_data.json)

Usage
-----
    pip install osmnx
    python fetch_north_tlv_map.py

To use a different area, edit the LOCATION section at the top.

Coordinate system
-----------------
Local equirectangular projection (same formula as SimStudio's GUI import):
    x =  (lon - lon0) * R * cos(lat0_rad)
    y = -(lat - lat0) * R
Origin (0, 0) = centroid of the downloaded bounding box.
Units: metres.
"""

import json
import math
import re
from collections import Counter

import osmnx as ox

# Use a fresh temp cache so stale OSM data doesn't corrupt coordinates
import tempfile as _tmpfile
ox.settings.cache_folder      = _tmpfile.mkdtemp(prefix="osmnx_fresh_")
ox.settings.use_cache         = True    # still cache within this run
ox.settings.max_query_area_size = 1e12  # disable sub-query splitting (download as one query)

# ---------------------------------------------------------------------------
# CONFIGURATION — edit this section
# ---------------------------------------------------------------------------

# ── CHOOSE WHICH HALF TO DOWNLOAD ──────────────────────────────────────────
# The area is split at the Yarkon River (~lat 32.097) into two maps.
# Set MAP_HALF = "north" or "south" and re-run for each.

MAP_HALF = "south"   # ← change to "north" for the second run

if MAP_HALF == "south":
    # Old North (south + north), New North (south + Kikar Hamedina + north)
    #   South ~ Shaul HaMelech St.  (~32.073)
    #   North ~ Yarkon River        (~32.097)
    #   West  ~ Mediterranean coast (~34.769)
    #   East  ~ Ayalon Highway      (~34.812)
    BBOX = (32.073, 32.097, 34.769, 34.812)
    MAP_OUTPUT = "north_tlv_south_map.sim.json"
    META_OUTPUT= "north_tlv_south_metadata.json"
    IMG_OUTPUT = "north_tlv_south_overview.png"
else:
    # Bavli, Tzameret HaAyalon
    #   South ~ Yarkon River   (~32.097)
    #   North ~ north of Bavli (~32.109)
    #   West  ~ coast          (~34.769)
    #   East  ~ Ayalon         (~34.812)
    BBOX = (32.097, 32.109, 34.769, 34.812)
    MAP_OUTPUT = "north_tlv_north_map.sim.json"
    META_OUTPUT= "north_tlv_north_metadata.json"
    IMG_OUTPUT = "north_tlv_north_overview.png"

# SimStudio defaults
DEFAULT_LANE_WIDTH  = 3.6   # metres
DEFAULT_SPEED_MPS   = 13.9  # m/s (~50 km/h)

# Road types to include in the map.
KEEP_HIGHWAY_TYPES = {
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    'residential',
    'trunk', 'trunk_link',
    'living_street',
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

R_EARTH = 6_378_137.0  # WGS-84 equatorial radius, metres

def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def _latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float):
    """Equirectangular projection — identical to SimStudio's _latlon_to_xy_m."""
    x =  _deg2rad(lon - lon0) * R_EARTH * math.cos(_deg2rad(lat0))
    y = -_deg2rad(lat - lat0) * R_EARTH   # negative → north is up
    return (x, y)

def _scalar(val):
    """Return first element of a list, or the value itself; None on NaN."""
    if isinstance(val, list):
        return val[0] if val else None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

def _maxspeed_mps(val, highway: str) -> float:
    """
    Parse an OSM maxspeed tag and return m/s.
    Falls back to road-type heuristics for Israeli urban roads.
    """
    v = _scalar(val)
    if v is not None:
        s = str(v).strip().lower()
        # strip "km/h" / "kph" suffix; handle "50 km/h", "30", etc.
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if m:
            kmh = float(m.group(1))
            if 'mph' in s:
                kmh *= 1.60934
            return round(kmh / 3.6, 3)

    # No maxspeed tag — infer from highway classification (Israeli urban defaults)
    hw = str(highway or '').lower()
    if hw in ('living_street', 'residential', 'service'):
        return round(30 / 3.6, 3)   # 8.333 m/s
    return round(50 / 3.6, 3)       # 13.889 m/s

def _names(row_dict: dict):
    """Return (display_name, name_he, name_en) from OSM tag dict."""
    raw     = _scalar(row_dict.get('name'))
    name_he = _scalar(row_dict.get('name:he'))
    name_en = _scalar(row_dict.get('name:en'))

    # If the plain 'name' field looks Hebrew, assign it to name_he
    if raw and name_he is None:
        he_chars = sum(1 for c in str(raw) if 'א' <= c <= 'ת')
        if he_chars > 3:
            name_he = raw
        elif name_en is None:
            name_en = raw

    display = name_he or name_en or raw
    return display, name_he, name_en

def _polyline_length(pts):
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        total += math.hypot(dx, dy)
    return total

# ---------------------------------------------------------------------------
# 1. DOWNLOAD FROM OSM
# ---------------------------------------------------------------------------

lat_s, lat_n, lon_w, lon_e = BBOX
print(f"Downloading street network for bbox: "
      f"lat {lat_s}–{lat_n}, lon {lon_w}–{lon_e}")
print("(Requires internet — takes ~15–60 s for large areas)")
print()

# osmnx 2.x uses (west, south, east, north) = (lon_w, lat_s, lon_e, lat_n)
G = ox.graph_from_bbox(bbox=(lon_w, lat_s, lon_e, lat_n), network_type='drive')
# osmnx 2.0 simplifies automatically — no extra call needed

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
print(f"Downloaded: {len(nodes_gdf)} OSM nodes, {len(edges_gdf)} OSM edges")

# ---------------------------------------------------------------------------
# 2. ORIGIN — centroid of bounding box (same convention as og_ map)
# ---------------------------------------------------------------------------

lat0 = (lat_s + lat_n) / 2.0
lon0 = (lon_w + lon_e) / 2.0
print(f"Origin: lat={lat0:.6f}, lon={lon0:.6f}")

# ---------------------------------------------------------------------------
# 3. BUILD SimStudio WORLD DICT
# ---------------------------------------------------------------------------

# --- nodes ---
node_id_map: dict = {}   # osm_node_id → "nN"
sim_nodes:   dict = {}   # "nN" → {"x": ..., "y": ...}
node_counter = 1

for osm_nid, row in nodes_gdf.iterrows():
    lat = float(row.geometry.y)
    lon = float(row.geometry.x)
    x, y = _latlon_to_xy(lat, lon, lat0, lon0)
    nid = f"n{node_counter}"
    node_counter += 1
    node_id_map[osm_nid] = nid
    sim_nodes[nid] = {"x": x, "y": y}

# --- segments + lanes + metadata ---
sim_segments: dict = {}
sim_lanes:    dict = {}
meta_segments: dict = {}

seg_counter = 1

for (u, v, k), row in edges_gdf.iterrows():
    row_dict = row.to_dict()

    # Convert geometry to local x/y polyline
    geom = row.geometry
    if geom is None:
        continue
    coords = list(geom.coords)   # list of (lon, lat) for LineString
    if len(coords) < 2:
        continue

    pts = [_latlon_to_xy(lat, lon, lat0, lon0) for lon, lat in coords]

    if _polyline_length(pts) < 4.0:
        continue   # skip micro-stubs

    hw = str(_scalar(row_dict.get('highway')) or 'unclassified')

    # Skip road types that aren't useful for vehicle-tracking simulation
    if hw not in KEEP_HIGHWAY_TYPES:
        continue

    n0_osm = u
    n1_osm = v
    n0 = node_id_map.get(n0_osm)
    n1 = node_id_map.get(n1_osm)

    if n0 is None or n1 is None:
        continue
    oneway     = bool(row_dict.get('oneway') is True)
    speed_mps  = _maxspeed_mps(row_dict.get('maxspeed'), hw)
    lanes_osm  = str(_scalar(row_dict.get('lanes')) or '1')

    sid = f"seg{seg_counter}"
    seg_counter += 1

    # SimStudio segment
    sim_segments[sid] = {
        "id":              sid,
        "n0":              n0,
        "n1":              n1,
        "lanes":           1,
        "lane_width":      DEFAULT_LANE_WIDTH,
        "speed_limit_mps": speed_mps,
        "points":          [list(p) for p in pts],
        "one_way":         oneway,
        "traffic_level":   "none",
    }

    # SimStudio lane (single lane, no offset)
    lid = f"{sid}_fwd_lane1"
    sim_lanes[lid] = {
        "id":           lid,
        "segment_id":   sid,
        "offset_index": 0,
        "polyline":     [list(p) for p in pts],
    }

    # Metadata entry
    display_name, name_he, name_en = _names(row_dict)
    osm_way_id = int(_scalar(row_dict.get('osmid')) or 0)

    meta_segments[sid] = {
        "osm_way_id": osm_way_id,
        "name":       display_name,
        "name_he":    name_he,
        "name_en":    name_en,
        "highway":    hw,
        "lanes_osm":  lanes_osm,
        "maxspeed":   str(round(speed_mps * 3.6)),   # back to km/h string
        "oneway":     "yes" if oneway else "no",
        "junction":   _scalar(row_dict.get('junction')),
    }

print(f"Built: {len(sim_segments)} segments, {len(sim_nodes)} nodes")

# ---------------------------------------------------------------------------
# 4. ASSEMBLE & SAVE
# ---------------------------------------------------------------------------

world_dict = {
    "nodes":    sim_nodes,
    "segments": sim_segments,
    "lanes":    sim_lanes,
    "vehicles": {},
    "gps":      {},
    "cameras":  {},
    "das":      {},
}

meta_dict = {
    "segments": meta_segments,
    "origin": {
        "lat0": lat0,
        "lon0": lon0,
    },
    "bbox": {
        "lat_s": lat_s, "lat_n": lat_n,
        "lon_w": lon_w, "lon_e": lon_e,
    },
}

with open(MAP_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(world_dict, f, ensure_ascii=False, indent=2)
print(f"\nMap saved  → {MAP_OUTPUT}")

with open(META_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(meta_dict, f, ensure_ascii=False, indent=2)
print(f"Meta saved → {META_OUTPUT}")

# ---------------------------------------------------------------------------
# 5. SUMMARY
# ---------------------------------------------------------------------------

total_road_m = sum(
    _polyline_length([tuple(p) for p in seg["points"]])
    for seg in sim_segments.values()
)
print(f"\nTotal road network: {total_road_m/1000:.2f} km")

print("\nSpeed limit breakdown:")
speeds = Counter(s["maxspeed"] for s in meta_segments.values())
for spd, count in sorted(speeds.items()):
    print(f"  {spd} km/h : {count} segments")

print("\nHighway types:")
hw_types = Counter(s["highway"] for s in meta_segments.values())
for hw, count in sorted(hw_types.items(), key=lambda x: -x[1]):
    print(f"  {hw:30s}: {count}")

# ---------------------------------------------------------------------------
# 6. FULL-MAP OVERVIEW IMAGE
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    print(f"\nGenerating full-map image → {IMG_OUTPUT} …")

    if not sim_segments:
        raise ValueError("No segments found — check KEEP_HIGHWAY_TYPES filter")

    # In our coord system: y = -(lat - lat0) * R  → north is NEGATIVE y.
    # To show north at top we NEGATE y before plotting, making north POSITIVE.
    all_lines_plot = [[[p[0], -p[1]] for p in seg["points"]]
                      for seg in sim_segments.values()]

    # Axis limits in plot space (after negating y):
    #   x: west = negative, east = positive (unchanged)
    #   y: north = +R*(lat_n - lat0), south = +R*(lat_s - lat0)  [negated]
    _R   = 6_378_137.0
    _cl  = math.cos(math.radians(lat0))
    pad  = 120   # metres padding

    x_left   = math.radians(lon_w - lon0) * _R * _cl - pad
    x_right  = math.radians(lon_e - lon0) * _R * _cl + pad
    y_top    = math.radians(lat_n - lat0) * _R + pad   # north → positive after negation
    y_bottom = math.radians(lat_s - lat0) * _R - pad   # south → negative after negation

    # Figure size matches geographic aspect ratio
    width_m  = x_right - x_left
    height_m = y_top   - y_bottom
    fig_w_in = 14
    fig_h_in = max(5, fig_w_in * height_m / width_m)

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), facecolor='white')
    ax.set_facecolor('#eef0f0')
    ax.axis('off')

    lc = LineCollection(all_lines_plot, colors='#1a5276', linewidths=1.0, alpha=0.9)
    ax.add_collection(lc)

    # Lock view to bbox — clips any highway geometry that extends beyond it
    ax.set_xlim(x_left,  x_right)
    ax.set_ylim(y_bottom, y_top)    # normal (non-inverted), north at top ✓

    plt.title("North Tel Aviv — Full Road Network", fontsize=14,
              pad=12, color='#222222')
    plt.tight_layout(pad=0.5)
    plt.savefig(IMG_OUTPUT, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Image saved → {IMG_OUTPUT}")

except ImportError:
    print("(matplotlib not found — skipping image; run: pip install matplotlib)")

print("""
Done.
─────────────────────────────────────────────────────
Next steps:
  1. Run organize_north_tlv.py to create numbered regions + tiled image.
─────────────────────────────────────────────────────
""")
