import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches
from shapely.geometry import box

# Load data
buildings = gpd.read_file("data/nairobi_buildings.geojson")
roads_drive = gpd.read_file("data/nairobi_roads_drive.geojson")
roads_all = gpd.read_file("data/nairobi_roads_all.geojson")

# Reproject to Web Mercator (for basemap)
target_crs = "EPSG:3857"
buildings = buildings.to_crs(target_crs)
roads_drive = roads_drive.to_crs(target_crs)
roads_all = roads_all.to_crs(target_crs)

# Define your "environment" bounding box
# 👇 Replace these coordinates with your actual area of interest in the same CRS
minx, miny, maxx, maxy = roads_all.total_bounds  # Example: entire dataset extent
# Example of smaller custom area:
# minx, miny, maxx, maxy = (4035000, -108500, 4040000, -104000)

bbox_geom = box(minx, miny, maxx, maxy)
bbox_gdf = gpd.GeoDataFrame(index=[0], crs=target_crs, geometry=[bbox_geom])

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

roads_all.plot(ax=ax, color="#c8d9f0", linewidth=0.6, label="All roads / paths")
roads_drive.plot(ax=ax, color="#004c91", linewidth=1.3, label="Drivable roads")
buildings.plot(ax=ax, color="#999999", alpha=0.7, edgecolor="none", label="Buildings")

# Add bounding box outline
bbox_gdf.boundary.plot(ax=ax, color="red", linewidth=2, label="Environment boundary")

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.6)

# Auto-zoom to bounding box
ax.set_xlim(minx - 200, maxx + 200)
ax.set_ylim(miny - 200, maxy + 200)

# North arrow
x, y, arrow_length = 0.05, 0.1, 0.08
ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=14,
            xycoords=ax.transAxes)

# Scale bar
scalebar = ScaleBar(1, location='lower left', box_alpha=0.7, pad=0.4)
ax.add_artist(scalebar)

# Title & legend
ax.set_title("Nairobi CBD — Environment Boundary & Road Network", fontsize=15, fontweight='bold', pad=15)
ax.axis("off")

handles = [
    mpatches.Patch(color="#999999", label="Buildings"),
    mpatches.Patch(color="#004c91", label="Drivable roads"),
    mpatches.Patch(color="#c8d9f0", label="All roads / paths"),
    mpatches.Patch(color="red", label="Environment boundary")
]
ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white", edgecolor="gray")

plt.tight_layout()
plt.show()
