"""
Download real Nairobi CBD data (buildings + roads)
Includes:
- Auto-sized exports with metadata-based filenames
- Optimized OSM downloads with fallbacks
- GeoJSON and NumPy outputs
"""

import os
import json
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, box
from rasterio import features
from rasterio.transform import from_bounds


def download_real_nairobi_cbd(
    bbox=None,
    sim_width_m=1000,
    sim_height_m=1000,
    grid_size=200,
    output_dir="data",
    place_name="Nairobi Central Business District, Nairobi, Kenya"
):
    """
    Download buildings and roads for Nairobi CBD from OpenStreetMap
    and export as GeoJSON and NumPy rasters with metadata.

    Args:
        bbox: dict with 'north','south','east','west' (optional)
        sim_width_m: width of simulated window (m)
        sim_height_m: height of simulated window (m)
        grid_size: raster resolution (number of cells)
        output_dir: folder to save results
        place_name: fallback OSM query area
    """

    # Default bounding box if none provided
    if bbox is None:
        bbox = {
            "north": -1.280,
            "south": -1.295,
            "east": 36.835,
            "west": 36.810,
        }

    os.makedirs(output_dir, exist_ok=True)

    print(f" Downloading Nairobi data in bbox: {bbox}")
    print(f" Sim area: {sim_width_m}×{sim_height_m} m, grid={grid_size}×{grid_size}")

    # Configure OSMnx settings
    ox.settings.timeout = 600
    ox.settings.memory = 1024 * 16
    ox.settings.max_query_area_size = 5e7

    # Polygon for bounding box
    bbox_polygon = Polygon([
        (bbox["west"], bbox["south"]),
        (bbox["east"], bbox["south"]),
        (bbox["east"], bbox["north"]),
        (bbox["west"], bbox["north"]),
    ])

    try:
        # === 1. BUILDINGS ===
        print(" Downloading buildings...")
        buildings_gdf = ox.features_from_polygon(bbox_polygon, tags={"building": True})
        if buildings_gdf is None or len(buildings_gdf) == 0:
            print(" Polygon query empty, trying named fallback area...")
            buildings_gdf = ox.features_from_place(place_name, tags={"building": True})
        if buildings_gdf is None:
            buildings_gdf = gpd.GeoDataFrame(columns=["geometry"])

        print(f" Buildings: {len(buildings_gdf)}")

        # === 2. ROADS ===
        print(" Downloading drivable roads...")
        try:
            roads_graph_drive = ox.graph_from_polygon(bbox_polygon, network_type="drive")
            roads_edges_drive = ox.graph_to_gdfs(roads_graph_drive, nodes=False, edges=True)
            if isinstance(roads_edges_drive, tuple):
                _, roads_edges_drive = roads_edges_drive
        except Exception as e_drive:
            print(f" Drivable road download failed: {e_drive}")
            roads_edges_drive = gpd.GeoDataFrame(columns=["geometry"])

        print(" Downloading all OSM roads (including paths)...")
        try:
            roads_graph_all = ox.graph_from_polygon(bbox_polygon, network_type="all")
            roads_edges_all = ox.graph_to_gdfs(roads_graph_all, nodes=False, edges=True)
            if isinstance(roads_edges_all, tuple):
                _, roads_edges_all = roads_edges_all
        except Exception as e_all:
            print(f" All roads download failed: {e_all}")
            roads_edges_all = gpd.GeoDataFrame(columns=["geometry"])

        print(f" Roads (drive): {len(roads_edges_drive)}, all types: {len(roads_edges_all)}")

        # === 3. CRS ===
        target_crs = "EPSG:32737"  # UTM Zone 37S
        for gdf in (buildings_gdf, roads_edges_drive, roads_edges_all):
            if len(gdf) > 0:
                gdf.to_crs(target_crs, inplace=True)

        # === 4. Save GeoJSONs ===
        print(" Saving GeoJSON files...")
        buildings_gdf.to_file(f"{output_dir}/nairobi_buildings.geojson", driver="GeoJSON")
        roads_edges_drive.to_file(f"{output_dir}/nairobi_roads_drive.geojson", driver="GeoJSON")
        roads_edges_all.to_file(f"{output_dir}/nairobi_roads_all.geojson", driver="GeoJSON")

        # === 5. Simulation window and rasterization ===
        if len(buildings_gdf) > 0:
            bounds = buildings_gdf.total_bounds
        elif len(roads_edges_drive) > 0:
            bounds = roads_edges_drive.total_bounds
        else:
            bounds = [294000, 9850000, 294500, 9850500]

        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        sim_bounds = [
            cx - sim_width_m / 2,
            cy - sim_height_m / 2,
            cx + sim_width_m / 2,
            cy + sim_height_m / 2,
        ]
        sim_poly = box(*sim_bounds)
        transform = from_bounds(*sim_bounds, grid_size, grid_size)

        # Rasterize
        def rasterize_shapes(gdf):
            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and geom.is_valid and geom.intersects(sim_poly)]
            return features.rasterize(
                shapes,
                out_shape=(grid_size, grid_size),
                transform=transform,
                fill=0,
                dtype=np.uint8
            ) if shapes else np.zeros((grid_size, grid_size), dtype=np.uint8)

        building_grid = rasterize_shapes(buildings_gdf)
        road_grid = rasterize_shapes(roads_edges_drive)
        building_mask = building_grid.astype(bool)
        road_mask = road_grid.astype(bool)
        obstacle_mask = np.logical_and(building_mask, np.logical_not(road_mask))

        # Add boundary walls
        obstacle_mask[0, :] = obstacle_mask[-1, :] = True
        obstacle_mask[:, 0] = obstacle_mask[:, -1] = True

        # === 6. Dynamic Filenames ===
        tag = f"{grid_size}x{grid_size}"
        base = f"real_nairobi_cbd_{tag}"
        np.save(f"{output_dir}/{base}.npy", obstacle_mask)
        np.save(f"{output_dir}/{base}_roads.npy", road_mask)

        metadata = {
            "bbox": bbox,
            "bounds_m": sim_bounds,
            "crs": target_crs,
            "width": grid_size,
            "height": grid_size,
            "cell_size_m": sim_width_m / grid_size,
            "buildings_count": int(len(buildings_gdf)),
            "road_segments_drive": int(len(roads_edges_drive)),
            "road_segments_all": int(len(roads_edges_all)),
            "obstacle_cells": int(obstacle_mask.sum()),
            "coverage_percent": 100 * float(obstacle_mask.sum()) / float(obstacle_mask.size),
            "output_files": {
                "obstacle": f"{base}.npy",
                "roads": f"{base}_roads.npy",
                "metadata": f"{base}_metadata.json",
            },
            "note": "Buildings and roads from OSM - Nairobi CBD (drive + all variants)",
        }

        meta_path = f"{output_dir}/{base}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f" Saved raster & metadata → {meta_path}")
        print(f"   Obstacles: {metadata['obstacle_cells']} ({metadata['coverage_percent']:.1f}%)")
        print(f"   Buildings: {metadata['buildings_count']} | Roads: {metadata['road_segments_drive']}")

        return obstacle_mask

    except Exception as e:
        print(f" Download failed: {e}")
        print("Creating synthetic fallback...")
        return _create_backup_nairobi_grid(output_dir=output_dir)


def _create_backup_nairobi_grid(output_dir="data"):
    """Synthetic backup grid if OSM download fails."""
    os.makedirs(output_dir, exist_ok=True)
    grid_size = 200
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    grid[20:25, :] = False
    grid[:, 25:30] = False
    buildings = [(10, 10, 15, 12), (35, 40, 10, 8), (45, 15, 8, 10)]
    for y, x, h, w in buildings:
        grid[y:y+h, x:x+w] = True
    road_mask = np.zeros((grid_size, grid_size), dtype=bool)
    road_mask[20:25, :] = True
    road_mask[:, 25:30] = True
    obstacle_mask = np.logical_and(grid, np.logical_not(road_mask))

    base = f"{output_dir}/real_nairobi_cbd_backup_{grid_size}x{grid_size}"
    np.save(f"{base}.npy", obstacle_mask)
    np.save(f"{base}_roads.npy", road_mask)
    metadata = {
        "width": grid_size,
        "height": grid_size,
        "cell_size_m": 5.0,
        "obstacle_cells": int(obstacle_mask.sum()),
        "coverage_percent": 100 * float(obstacle_mask.sum()) / float(obstacle_mask.size),
        "note": "Synthetic backup grid - Nairobi CBD layout",
    }
    with open(f"{base}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f" Created synthetic backup grid {grid_size}×{grid_size}")
    return obstacle_mask


if __name__ == "__main__":
    download_real_nairobi_cbd()
