"""
fetch_osm_metadata.py
=====================
Downloads street data from OpenStreetMap for any neighborhood and converts it
to the metadata format used by SimStudio (same structure as
notbig_florentin_mata_data.json).

Usage
-----
    python fetch_osm_metadata.py

Requirements
------------
    pip install osmnx

To use a different neighborhood, edit the LOCATION / BBOX section below.
"""

import json
import math
import osmnx as ox

# ---------------------------------------------------------------------------
# 1.  CHOOSE YOUR LOCATION  (edit this section)
# ---------------------------------------------------------------------------

# Option A — by place name (easiest)
PLACE_NAME = "Florentin, Tel Aviv, Israel"

# Option B — by bounding box: (lat_south, lat_north, lon_west, lon_east)
# Uncomment and fill in if place-name search doesn't work for your area.
# BBOX = (32.054, 32.061, 34.765, 34.777)   # Florentin example

# Output file
OUTPUT_PATH = "notbig_florentin_mata_data.json"   # change name for other areas

# ---------------------------------------------------------------------------
# 2.  DOWNLOAD FROM OSM
# ---------------------------------------------------------------------------

print(f"Downloading street network for: {PLACE_NAME}")
print("(This requires an internet connection and takes ~10-30 seconds)")
print()

# network_type="drive" — only driveable roads (no footpaths, bike lanes, etc.)
try:
    G = ox.graph_from_place(PLACE_NAME, network_type="drive")
except Exception:
    # Fallback to bounding box if place name fails
    print("Place name lookup failed — trying bounding box...")
    G = ox.graph_from_bbox(*BBOX, network_type="drive")

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
print(f"Downloaded: {len(nodes_gdf)} nodes, {len(edges_gdf)} road segments")

# ---------------------------------------------------------------------------
# 3.  FIND THE ORIGIN (southwest corner → local coordinate system)
# ---------------------------------------------------------------------------

# The SimStudio coordinate system is centered at (lat0, lon0).
# We use the centroid of the bounding box as the origin, matching how
# notbig_florentin_mata_data.json was built.
lat0 = float(nodes_gdf.geometry.y.mean())
lon0 = float(nodes_gdf.geometry.x.mean())
print(f"Origin (center): lat={lat0:.6f}, lon={lon0:.6f}")

# ---------------------------------------------------------------------------
# 4.  BUILD THE METADATA DICTIONARY
# ---------------------------------------------------------------------------

# Helper: safely get a scalar from a column that may contain lists
def _scalar(val):
    if isinstance(val, list):
        return val[0] if val else None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

# Helper: normalise maxspeed to a clean string ("30" or "50") or None
def _maxspeed(val):
    v = _scalar(val)
    if v is None:
        return None
    s = str(v).strip().lower()
    # OSM values like "50", "50 km/h", "30 mph" etc.
    for token in s.split():
        try:
            kmh = float(token)
            # crude mph→kmh (OSM Israel uses km/h but handle edge cases)
            return str(int(round(kmh)))
        except ValueError:
            continue
    return None

# Helper: get both Hebrew and English name from the OSM name fields
def _names(row):
    raw = _scalar(row.get("name"))
    name_he = None
    name_en = None
    if raw:
        # OSM sometimes stores "Name - שם" or just one script
        # Try to split by script character ranges
        he_chars = sum(1 for c in str(raw) if "א" <= c <= "ת")
        if he_chars > 3:
            name_he = raw
        else:
            name_en = raw
    # OSM also has name:en and name:he tags
    if "name:en" in row and _scalar(row.get("name:en")):
        name_en = _scalar(row.get("name:en"))
    if "name:he" in row and _scalar(row.get("name:he")):
        name_he = _scalar(row.get("name:he"))
    return name_he or raw, name_he, name_en

segments = {}
for i, (idx, row) in enumerate(edges_gdf.iterrows(), start=1):
    seg_id = f"seg{i}"

    name, name_he, name_en = _names(row)

    maxspeed = _maxspeed(row.get("maxspeed"))

    # If maxspeed is missing, infer from highway type (Israeli urban defaults)
    if maxspeed is None:
        hw = str(_scalar(row.get("highway")) or "").lower()
        if hw in ("living_street", "residential"):
            maxspeed = "30"
        else:
            maxspeed = "50"

    segments[seg_id] = {
        "osm_way_id": int(_scalar(row.get("osmid")) or 0),
        "name":       name,
        "name_he":    name_he,
        "name_en":    name_en,
        "highway":    _scalar(row.get("highway")),
        "lanes_osm":  str(_scalar(row.get("lanes")) or "1"),
        "maxspeed":   maxspeed,
        "oneway":     "yes" if row.get("oneway") is True else "no",
        "junction":   _scalar(row.get("junction")),
    }

result = {
    "segments": segments,
    "origin": {
        "lat0": lat0,
        "lon0": lon0,
    }
}

# ---------------------------------------------------------------------------
# 5.  SAVE
# ---------------------------------------------------------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print()
print(f"Saved {len(segments)} segments to: {OUTPUT_PATH}")
print()
print("Speed limit breakdown:")
from collections import Counter
speeds = Counter(s["maxspeed"] for s in segments.values())
for spd, count in sorted(speeds.items()):
    print(f"  {spd} km/h : {count} segments")

print()
print("Highway types found:")
hwtypes = Counter(s["highway"] for s in segments.values())
for hw, count in sorted(hwtypes.items(), key=lambda x: -x[1]):
    print(f"  {hw:30s}: {count}")

print()
print("Done. To use a different neighborhood:")
print("  1. Change PLACE_NAME at the top of this script")
print("  2. Change OUTPUT_PATH to a new filename")
print("  3. Run again")
