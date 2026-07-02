"""
fetch_klausner_map.py
=====================
Downloads the road network for Klausner Street and surrounding streets
near Tel Aviv University from OpenStreetMap, and produces the two files
SimStudio needs to run on real-world data from this experiment:

  1. klausner_map.sim.json    — SimStudio map file (road network + lanes)
  2. klausner_metadata.json   — OSM metadata + origin for lat/lon conversion

Usage
-----
    pip install osmnx matplotlib
    python scripts/fetch_klausner_map.py

Outputs are saved to maps/klausner/ (created automatically).

Coordinate system
-----------------
Local equirectangular projection — identical to all other SimStudio maps:
    x =  (lon - lon0) * R_EARTH * cos(lat0_rad)   [metres, east = +x]
    y = -(lat - lat0) * R_EARTH                    [metres, north = -y]

Origin (0, 0) = centroid of the bounding box below.

To convert a real-world GPS position (lat, lon) into pipeline (x, y):
    import math, json
    meta  = json.load(open("maps/klausner/klausner_metadata.json"))
    lat0  = meta["origin"]["lat0"]
    lon0  = meta["origin"]["lon0"]
    R     = 6_378_137.0
    x     =  math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y     = -math.radians(lat - lat0) * R
"""

import json
import math
import os
import re
from collections import Counter

import osmnx as ox

# Use a fresh cache so stale OSM data doesn't corrupt coordinates
import tempfile as _tmpfile
ox.settings.cache_folder        = _tmpfile.mkdtemp(prefix="osmnx_klausner_")
ox.settings.use_cache           = True
ox.settings.max_query_area_size = 1e12   # download as one query, no sub-splitting

# ---------------------------------------------------------------------------
# BOUNDING BOX
# Centred on the experiment pin: lat=32.111343, lon=34.8074897
# Covers Klausner St., Einstein St., Chaim Levanon St., and the TAU campus
# boundary — roughly 900 m north-south × 1200 m east-west.
# ---------------------------------------------------------------------------

BBOX = (
    32.107,   # lat_s  (south boundary)
    32.116,   # lat_n  (north boundary)
    34.801,   # lon_w  (west boundary)
    34.815,   # lon_e  (east boundary)
)

OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "maps", "klausner")
MAP_OUTPUT = os.path.join(OUT_DIR, "klausner_map.sim.json")
META_OUTPUT= os.path.join(OUT_DIR, "klausner_metadata.json")
IMG_OUTPUT = os.path.join(OUT_DIR, "klausner_overview.png")

# SimStudio defaults
DEFAULT_LANE_WIDTH = 3.6    # metres
DEFAULT_SPEED_MPS  = 13.9   # m/s (~50 km/h)

# Road types to include.
# NOTE: 'unclassified' is added here because many named streets in the TAU
# area (including Klausner St. itself) are tagged as 'unclassified' in OSM.
# The original north-TLV script omitted this type but it is essential here.
KEEP_HIGHWAY_TYPES = {
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    'residential',
    'trunk', 'trunk_link',
    'living_street',
    'unclassified',   # many named streets in Israel are tagged this way
    'service',        # campus access roads and internal TAU roads
}

# ---------------------------------------------------------------------------
# HELPERS  (identical to fetch_north_tlv_map.py — do not change)
# ---------------------------------------------------------------------------

R_EARTH = 6_378_137.0   # WGS-84 equatorial radius, metres

def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def _latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float):
    """Equirectangular projection — identical to SimStudio's _latlon_to_xy_m."""
    x =  _deg2rad(lon - lon0) * R_EARTH * math.cos(_deg2rad(lat0))
    y = -_deg2rad(lat - lat0) * R_EARTH
    return (x, y)

def _scalar(val):
    if isinstance(val, list):
        return val[0] if val else None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

def _maxspeed_mps(val, highway: str) -> float:
    v = _scalar(val)
    if v is not None:
        s = str(v).strip().lower()
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if m:
            kmh = float(m.group(1))
            if 'mph' in s:
                kmh *= 1.60934
            return round(kmh / 3.6, 3)
    hw = str(highway or '').lower()
    if hw in ('living_street', 'residential', 'service'):
        return round(30 / 3.6, 3)
    return round(50 / 3.6, 3)

def _names(row_dict: dict):
    raw     = _scalar(row_dict.get('name'))
    name_he = _scalar(row_dict.get('name:he'))
    name_en = _scalar(row_dict.get('name:en'))
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

os.makedirs(OUT_DIR, exist_ok=True)

lat_s, lat_n, lon_w, lon_e = BBOX
print(f"Downloading Klausner St. road network")
print(f"  Bbox: lat {lat_s}–{lat_n}, lon {lon_w}–{lon_e}")
print(f"  ~900 m × 1200 m around lat=32.111343, lon=34.8074897")
print(f"  (Takes ~5–20 s — requires internet)")
print()

G = ox.graph_from_bbox(bbox=(lon_w, lat_s, lon_e, lat_n), network_type='all')
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
print(f"Downloaded: {len(nodes_gdf)} OSM nodes, {len(edges_gdf)} OSM edges")

# Show what highway types OSM returned — useful for debugging
from collections import Counter as _Counter
_all_hw = _Counter()
for _, _row in edges_gdf.iterrows():
    _v = _row.to_dict().get('highway')
    if isinstance(_v, list): _v = _v[0]
    _all_hw[str(_v)] += 1
print("All highway types in bbox:")
for _hw, _cnt in sorted(_all_hw.items(), key=lambda x: -x[1]):
    _kept = "✓ kept" if _hw in KEEP_HIGHWAY_TYPES else "✗ filtered"
    print(f"  {_hw:35s}: {_cnt:3d}  {_kept}")

# ---------------------------------------------------------------------------
# 2. ORIGIN — centroid of bounding box
# ---------------------------------------------------------------------------

lat0 = (lat_s + lat_n) / 2.0   # 32.1115
lon0 = (lon_w + lon_e) / 2.0   # 34.808
print(f"Origin (0,0): lat={lat0:.6f}, lon={lon0:.6f}")
print()

# Confirm the experiment pin location in pipeline coordinates:
pin_x, pin_y = _latlon_to_xy(32.111343, 34.8074897, lat0, lon0)
print(f"Experiment pin (Klausner St.): x={pin_x:.1f} m, y={pin_y:.1f} m")
print()

# ---------------------------------------------------------------------------
# 3. BUILD SimStudio WORLD DICT
# ---------------------------------------------------------------------------

node_id_map: dict = {}
sim_nodes:   dict = {}
node_counter = 1

for osm_nid, row in nodes_gdf.iterrows():
    lat = float(row.geometry.y)
    lon = float(row.geometry.x)
    x, y = _latlon_to_xy(lat, lon, lat0, lon0)
    nid = f"n{node_counter}"
    node_counter += 1
    node_id_map[osm_nid] = nid
    sim_nodes[nid] = {"x": x, "y": y}

sim_segments: dict = {}
sim_lanes:    dict = {}
meta_segments: dict = {}
seg_counter = 1

for (u, v, k), row in edges_gdf.iterrows():
    row_dict = row.to_dict()
    geom = row.geometry
    if geom is None:
        continue
    coords = list(geom.coords)
    if len(coords) < 2:
        continue

    pts = [_latlon_to_xy(lat, lon, lat0, lon0) for lon, lat in coords]
    if _polyline_length(pts) < 4.0:
        continue

    hw = str(_scalar(row_dict.get('highway')) or 'unclassified')
    if hw not in KEEP_HIGHWAY_TYPES:
        continue

    n0 = node_id_map.get(u)
    n1 = node_id_map.get(v)
    if n0 is None or n1 is None:
        continue

    oneway    = bool(row_dict.get('oneway') is True)
    speed_mps = _maxspeed_mps(row_dict.get('maxspeed'), hw)
    lanes_osm = str(_scalar(row_dict.get('lanes')) or '1')

    sid = f"seg{seg_counter}"
    seg_counter += 1

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

    lid = f"{sid}_fwd_lane1"
    sim_lanes[lid] = {
        "id":           lid,
        "segment_id":   sid,
        "offset_index": 0,
        "polyline":     [list(p) for p in pts],
    }

    display_name, name_he, name_en = _names(row_dict)
    osm_way_id = int(_scalar(row_dict.get('osmid')) or 0)

    meta_segments[sid] = {
        "osm_way_id": osm_way_id,
        "name":       display_name,
        "name_he":    name_he,
        "name_en":    name_en,
        "highway":    hw,
        "lanes_osm":  lanes_osm,
        "maxspeed":   str(round(speed_mps * 3.6)),
        "oneway":     "yes" if oneway else "no",
        "junction":   _scalar(row_dict.get('junction')),
    }

print(f"Built: {len(sim_segments)} road segments, {len(sim_nodes)} nodes")

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
    "experiment": {
        "street":      "Klausner Street",
        "city":        "Tel Aviv",
        "pin_lat":     32.111343,
        "pin_lon":     34.8074897,
        "pin_x_m":     round(pin_x, 2),
        "pin_y_m":     round(pin_y, 2),
        "notes":       "Camera + DAS experiment near Tel Aviv University. "
                       "Use pin_x_m / pin_y_m as reference point when "
                       "converting colleague's GPS coordinates to pipeline x,y.",
    },
}

with open(MAP_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(world_dict, f, ensure_ascii=False, indent=2)
print(f"Map saved  → {MAP_OUTPUT}")

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
# 6. OVERVIEW IMAGE
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    print(f"\nGenerating map image → {IMG_OUTPUT} …")

    all_lines_plot = [[[p[0], -p[1]] for p in seg["points"]]
                      for seg in sim_segments.values()]

    _R  = R_EARTH
    _cl = math.cos(math.radians(lat0))
    pad = 50  # metres

    x_left   = math.radians(lon_w - lon0) * _R * _cl - pad
    x_right  = math.radians(lon_e - lon0) * _R * _cl + pad
    y_top    = math.radians(lat_n - lat0) * _R + pad
    y_bottom = math.radians(lat_s - lat0) * _R - pad

    width_m  = x_right - x_left
    height_m = y_top - y_bottom
    fig_w_in = 12
    fig_h_in = max(4, fig_w_in * height_m / width_m)

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), facecolor='white')
    ax.set_facecolor('#eef0f0')
    ax.axis('off')

    lc = LineCollection(all_lines_plot, colors='#1a5276', linewidths=1.2, alpha=0.9)
    ax.add_collection(lc)

    # Mark the experiment pin
    ax.plot(pin_x, -pin_y, 'ro', markersize=10, zorder=5, label='Experiment pin')
    ax.annotate('Klausner St.\nexperiment',
                xy=(pin_x, -pin_y), xytext=(pin_x + 40, -pin_y - 60),
                fontsize=9, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.legend(loc='upper left', fontsize=9)

    plt.title("Klausner St. — Tel Aviv University area", fontsize=13, pad=10)
    plt.tight_layout(pad=0.5)
    plt.savefig(IMG_OUTPUT, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Image saved → {IMG_OUTPUT}")

except ImportError:
    print("(matplotlib not found — skipping image; run: pip install matplotlib)")

print(f"""
Done.
─────────────────────────────────────────────────────────────────
Output files:
  {MAP_OUTPUT}
  {META_OUTPUT}
  {IMG_OUTPUT}

To load in SimStudio, point the map loader at:
  maps/klausner/klausner_map.sim.json

To convert a GPS coordinate from your colleague to pipeline x,y:
  meta  = json.load(open("maps/klausner/klausner_metadata.json"))
  lat0, lon0 = meta["origin"]["lat0"], meta["origin"]["lon0"]
  x =  math.radians(lon - lon0) * 6_378_137 * math.cos(math.radians(lat0))
  y = -math.radians(lat - lat0) * 6_378_137
─────────────────────────────────────────────────────────────────
""")
