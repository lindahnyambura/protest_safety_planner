#!/usr/bin/env python3
"""
plot_nairobi_highways_bbox.py

Fetch major highways from OSM via Overpass for a given bbox, plot them,
and highlight which ways cross the bbox or are very close to its edges.

Output: nairobi_bbox_highways.png
"""

import requests
import json
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import nearest_points
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
import math

# ---- CONFIG ----
# Your bbox (south, west, north, east)
south = -1.2870
west  = 36.8170
north = -1.2830
east  = 36.8200

# margin buffer to consider "near the edge" (in meters)
NEAR_EDGE_METERS = 50  # tweak as needed

# output file
OUT_PNG = "nairobi_bbox_highways.png"

# Overpass QL searching for major highway types
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
overpass_query = f"""
[out:json][timeout:25];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link"]({south},{west},{north},{east});
);
out geom;
"""

# ---- FUNCTIONS ----

def overpass_fetch(q):
    resp = requests.post(OVERPASS_ENDPOINT, data={"data": q})
    resp.raise_for_status()
    return resp.json()

def way_to_linestring(way):
    # 'geometry' is a list of {lat, lon}
    coords = [(pt['lon'], pt['lat']) for pt in way.get('geometry', [])]
    if len(coords) < 2:
        return None
    return LineString(coords)

def lonlat_to_utm_crs(lon, lat):
    # Return appropriate UTM zone CRS string for the given lon/lat
    zone = math.floor((lon + 180) / 6) + 1
    # For southern hemisphere use EPSG:327xx, for northern: 326xx
    if lat < 0:
        epsg = 32700 + zone
    else:
        epsg = 32600 + zone
    return CRS.from_epsg(epsg)

# ---- MAIN ----

print("Querying Overpass API for highways in bbox...")
data = overpass_fetch(overpass_query)

ways = [elem for elem in data.get("elements", []) if elem["type"] == "way"]
print(f"Found {len(ways)} way(s).")

geoms = []
rows = []
for w in ways:
    ls = way_to_linestring(w)
    if ls is None:
        continue
    name = w.get("tags", {}).get("name", w.get("id"))
    highway = w.get("tags", {}).get("highway", "")
    rows.append({
        "osmid": w.get("id"),
        "name": name,
        "highway": highway,
        "geometry": ls
    })

if not rows:
    raise SystemExit("No highway geometries found in bbox — try increasing bbox size.")

gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

# Define bbox polygon and boundary
bbox_poly = Polygon([
    (west, south),
    (east, south),
    (east, north),
    (west, north),
    (west, south)
])
bbox_gdf = gpd.GeoDataFrame([{"geometry": bbox_poly}], crs="EPSG:4326")

# Project to UTM (local) for accurate distance calculations and buffering:
center_lon = (west + east) / 2.0
center_lat = (south + north) / 2.0
utm_crs = lonlat_to_utm_crs(center_lon, center_lat)
print(f"Projecting geometries to CRS {utm_crs.to_authority()} for distance calculations...")

gdf_utm = gdf.to_crs(utm_crs)
bbox_utm = bbox_gdf.to_crs(utm_crs).iloc[0].geometry
bbox_boundary = bbox_utm.boundary

# compute min distance to bbox boundary and whether geometry intersects bbox
distances = []
crosses = []
near_edge_flag = []
for geom in gdf_utm.geometry:
    d = geom.distance(bbox_boundary)  # meters
    distances.append(d)
    crosses.append(geom.intersects(bbox_utm))
    near_edge_flag.append(d <= NEAR_EDGE_METERS)

gdf["dist_m_to_bbox_edge"] = distances
gdf["intersects_bbox"] = crosses
gdf["near_edge"] = near_edge_flag

# ---- PLOTTING ----

# Reproject to EPSG:4326 for consistent plotting
gdf_plot = gdf.to_crs("EPSG:4326")
bbox_plot = bbox_gdf.to_crs("EPSG:4326")

# Prepare plotting
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot baseline: bbox
bbox_plot.boundary.plot(ax=ax, linewidth=2, edgecolor="black", zorder=4, label="bbox")

# Plot all highways
gdf_plot.plot(ax=ax, column="highway", linewidth=2, legend=False, zorder=3)

# Highlight ways that cross bbox or are near edge (only if not empty)
cross_subset = gdf_plot[gdf_plot["intersects_bbox"]]
if not cross_subset.empty:
    cross_subset.plot(
        ax=ax, linewidth=3.5, edgecolor="red", zorder=6, label="crossing bbox"
    )

near_subset = gdf_plot[(~gdf_plot["intersects_bbox"]) & (gdf_plot["near_edge"])]
if not near_subset.empty:
    near_subset.plot(
        ax=ax, linewidth=2.5, edgecolor="orange", zorder=5,
        label=f"near edge (<= {NEAR_EDGE_METERS} m)"
    )


# Annotate names and distances
for idx, row in gdf_plot.iterrows():
    geom = row.geometry
    pt = geom.representative_point()
    try:
        ax.text(
            pt.x, pt.y,
            f"{row['name']}\n{int(row['dist_m_to_bbox_edge'])}m",
            fontsize=7, ha="center"
        )
    except Exception:
        pass

# Plot style
ax.set_title("Major highways around bounding box (Nairobi CBD)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xlim(west - 0.001, east + 0.001)
ax.set_ylim(south - 0.001, north + 0.001)

# legend
ax.legend(loc="lower left")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved map to {OUT_PNG}")

# ---- SUMMARY ----
print("\nSummary of returned ways:")
print(
    gdf[[
        "osmid", "name", "highway",
        "intersects_bbox", "near_edge", "dist_m_to_bbox_edge"
    ]].to_string(index=False)
)
