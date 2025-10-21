"""
Download real Nairobi CBD data (buildings + roads)
Includes:
- Optimized OSM downloads
- Fallback synthetic grid if download fails
- Separate exports for drivable and all road networks
- GeoJSON and NumPy outputs with metadata
"""

import os
import json
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, box
from rasterio import features
from rasterio.transform import from_bounds


def download_real_nairobi_cbd():
    """
    Download buildings and roads for Nairobi CBD from OpenStreetMap,
    export as GeoJSON and NumPy arrays, and provide metadata.

    If OSM download fails, a synthetic backup grid is generated instead.
    """

    # Define a small bounding box around the central business district
    nairobi_bbox = {
        'north': -1.2830,
        'south': -1.2870,
        'east': 36.8200,
        'west': 36.8170
    }

    print("Downloading real Nairobi CBD (buildings + roads, optimized)...")
    print(f"Bounding box: {nairobi_bbox}")

    # Configure OSMnx settings for stability
    ox.settings.timeout = 300
    ox.settings.memory = 1024 * 8
    ox.settings.max_query_area_size = 50000

    try:
        # Create polygon geometry for the bounding box
        bbox_polygon = Polygon([
            (nairobi_bbox['west'], nairobi_bbox['south']),
            (nairobi_bbox['east'], nairobi_bbox['south']),
            (nairobi_bbox['east'], nairobi_bbox['north']),
            (nairobi_bbox['west'], nairobi_bbox['north']),
        ])

        # 1. Buildings
        
        print("Downloading buildings...")
        buildings_gdf = ox.features_from_polygon(bbox_polygon, tags={'building': True})

        # Fallback if polygon query returns nothing
        if buildings_gdf is None or len(buildings_gdf) == 0:
            print("No buildings found with polygon query — trying named area fallback...")
            buildings_gdf = ox.features_from_place(
                "Nairobi Central Business District, Nairobi, Kenya",
                tags={'building': True}
            )

        if buildings_gdf is None:
            buildings_gdf = gpd.GeoDataFrame(columns=['geometry'])

        print(f"Buildings downloaded: {len(buildings_gdf)}")

        # 2. Roads: drivable and all types

        print("Downloading drivable roads...")
        try:
            roads_graph_drive = ox.graph_from_polygon(bbox_polygon, network_type='drive')
            roads_edges_drive = ox.graph_to_gdfs(roads_graph_drive, nodes=False, edges=True)
            if isinstance(roads_edges_drive, tuple):
                _, roads_edges_drive = roads_edges_drive
        except Exception as e_drive:
            print(f"Warning: drivable roads download failed: {e_drive}")
            roads_edges_drive = gpd.GeoDataFrame(columns=['geometry'])
        print(f"Drivable road segments: {len(roads_edges_drive)}")

        print("Downloading all OSM roads (including pedestrian paths)...")
        try:
            roads_graph_all = ox.graph_from_polygon(bbox_polygon, network_type='all')
            roads_edges_all = ox.graph_to_gdfs(roads_graph_all, nodes=False, edges=True)
            if isinstance(roads_edges_all, tuple):
                _, roads_edges_all = roads_edges_all
        except Exception as e_all:
            print(f"Warning: all roads download failed: {e_all}")
            roads_edges_all = gpd.GeoDataFrame(columns=['geometry'])
        print(f"All road/path segments: {len(roads_edges_all)}")

        # 3. Coordinate reference system conversion

        target_crs = 'EPSG:32737'  # UTM Zone 37S for Nairobi
        if len(buildings_gdf) > 0:
            buildings_gdf = buildings_gdf.to_crs(target_crs)
        if len(roads_edges_drive) > 0:
            roads_edges_drive = roads_edges_drive.to_crs(target_crs)
        if len(roads_edges_all) > 0:
            roads_edges_all = roads_edges_all.to_crs(target_crs)

        # 4. Save GeoJSON outputs
    
        os.makedirs("data", exist_ok=True)
        paths = {
            "buildings": "data/nairobi_buildings.geojson",
            "roads_drive": "data/nairobi_roads_drive.geojson",
            "roads_all": "data/nairobi_roads_all.geojson"
        }

        try:
            buildings_gdf.to_file(paths["buildings"], driver="GeoJSON")
            print(f"Saved buildings GeoJSON to {paths['buildings']} ({len(buildings_gdf)} features)")
        except Exception as e:
            print(f"Warning: Could not save buildings GeoJSON: {e}")

        try:
            roads_edges_drive.to_file(paths["roads_drive"], driver="GeoJSON")
            print(f"Saved drivable roads GeoJSON to {paths['roads_drive']} ({len(roads_edges_drive)} features)")
        except Exception as e:
            print(f"Warning: Could not save drivable roads GeoJSON: {e}")

        try:
            roads_edges_all.to_file(paths["roads_all"], driver="GeoJSON")
            print(f"Saved all roads GeoJSON to {paths['roads_all']} ({len(roads_edges_all)} features)")
        except Exception as e:
            print(f"Warning: Could not save all roads GeoJSON: {e}")

        # 5. Compute simulation area and rasterization
    
        if len(buildings_gdf) > 0:
            bounds = buildings_gdf.total_bounds
        elif len(roads_edges_drive) > 0:
            bounds = roads_edges_drive.total_bounds
        else:
            bounds = [294000, 9850000, 294500, 9850500]  # approximate fallback in UTM

        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

        sim_width_m = 500
        sim_height_m = 500
        sim_bounds = [
            center_x - sim_width_m / 2,
            center_y - sim_height_m / 2,
            center_x + sim_width_m / 2,
            center_y + sim_height_m / 2
        ]

        grid_size = 100
        transform = from_bounds(*sim_bounds, grid_size, grid_size)
        sim_poly = box(*sim_bounds)

        # Rasterize buildings
        building_shapes = [
            (geom, 1)
            for geom in buildings_gdf.geometry
            if geom is not None and geom.is_valid and geom.intersects(sim_poly)
        ]
        building_grid = features.rasterize(
            building_shapes,
            out_shape=(grid_size, grid_size),
            transform=transform,
            fill=0,
            dtype=np.uint8
        ) if building_shapes else np.zeros((grid_size, grid_size), dtype=np.uint8)

        # Rasterize drivable roads
        road_shapes_drive = [
            (geom, 1)
            for geom in roads_edges_drive.geometry
            if geom is not None and geom.is_valid and geom.intersects(sim_poly)
        ]
        road_grid = features.rasterize(
            road_shapes_drive,
            out_shape=(grid_size, grid_size),
            transform=transform,
            fill=0,
            dtype=np.uint8
        ) if road_shapes_drive else np.zeros((grid_size, grid_size), dtype=np.uint8)

        # Combine to create obstacle and road masks
        building_mask = building_grid.astype(bool)
        road_mask = road_grid.astype(bool)
        obstacle_mask = np.logical_and(building_mask, np.logical_not(road_mask))

        # Add boundary walls
        obstacle_mask[0, :] = obstacle_mask[-1, :] = True
        obstacle_mask[:, 0] = obstacle_mask[:, -1] = True

        # 6. Save NumPy arrays and metadata
    
        np.save("data/real_nairobi_cbd_100x100.npy", obstacle_mask)
        np.save("data/real_nairobi_cbd_roads_100x100.npy", road_mask)

        metadata = {
            'bounds': sim_bounds,
            'crs': target_crs,
            'width': grid_size,
            'height': grid_size,
            'cell_size_m': 5.0,
            'buildings_count': int(len(buildings_gdf)),
            'road_segments_drive': int(len(roads_edges_drive)),
            'road_segments_all': int(len(roads_edges_all)),
            'obstacle_cells': int(obstacle_mask.sum()),
            'coverage_percent': 100 * float(obstacle_mask.sum()) / float(obstacle_mask.size),
            'note': 'Buildings and roads from OSM - Nairobi CBD (drive + all variants)'
        }

        with open("data/real_nairobi_cbd_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Saved real Nairobi CBD raster and metadata.")
        print(f"Grid: {grid_size}x{grid_size}")
        print(f"Obstacles: {metadata['obstacle_cells']} ({metadata['coverage_percent']:.1f}% area)")
        print(f"Buildings: {metadata['buildings_count']}")
        print(f"Roads (drive): {metadata['road_segments_drive']}")
        print(f"Roads (all types): {metadata['road_segments_all']}")

        return obstacle_mask

    except Exception as e:
        print(f"Download failed: {e}")
        print("Creating synthetic backup grid...")
        return _create_backup_nairobi_grid()


def _create_backup_nairobi_grid():
    """
    Create a synthetic backup grid that approximates Nairobi CBD layout.
    Used when OSM downloads fail.
    """
    grid_size = 100
    grid = np.zeros((grid_size, grid_size), dtype=bool)

    # Add border walls
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True

    # Create two clear corridors representing major roads
    grid[20:25, :] = False     # horizontal corridor
    grid[:, 25:30] = False     # vertical corridor

    # Add rectangular building blocks
    buildings = [
        (10, 10, 15, 12),  # KICC area
        (35, 40, 10, 8),   # Central area
        (45, 15, 8, 10)    # Southern area
    ]
    for y, x, h, w in buildings:
        grid[y:y+h, x:x+w] = True

    # Define a simple road mask
    road_mask = np.zeros((grid_size, grid_size), dtype=bool)
    road_mask[20:25, :] = True
    road_mask[:, 25:30] = True
    road_mask[50, 40:60] = True

    obstacle_mask = np.logical_and(grid, np.logical_not(road_mask))

    # Save backup data
    os.makedirs("data", exist_ok=True)
    np.save("data/real_nairobi_cbd_100x100.npy", obstacle_mask)
    np.save("data/real_nairobi_cbd_roads_100x100.npy", road_mask)

    # Save metadata for the fallback
    metadata = {
        'width': grid_size,
        'height': grid_size,
        'cell_size_m': 5.0,
        'buildings_count': len(buildings),
        'road_segments': int(road_mask.sum()),
        'obstacle_cells': int(obstacle_mask.sum()),
        'coverage_percent': 100 * float(obstacle_mask.sum()) / float(obstacle_mask.size),
        'note': 'Synthetic backup - simulated Nairobi CBD layout with roads'
    }

    with open("data/real_nairobi_cbd_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Created synthetic backup Nairobi CBD with metadata.")
    return obstacle_mask


if __name__ == "__main__":
    download_real_nairobi_cbd()
