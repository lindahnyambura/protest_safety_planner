"""
map_loader.py - Load and process Nairobi CBD map from OpenStreetMap

Features:
- Download building footprints and streets from OSM
- Project to UTM Zone 37S (metric coordinates)
- Rasterize buildings to obstacle grid
- Cache results locally for fast testing
"""

import numpy as np
import osmnx as ox
import geopandas as gpd
from pathlib import Path
import pickle
from typing import Dict, Tuple, Optional
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box
import warnings

# Suppress OSMnx warnings
warnings.filterwarnings('ignore', category=UserWarning)
ox.settings.use_cache = True
ox.settings.log_console = False


class NairobiCBDMap:
    """
    Manages Nairobi CBD map data from OpenStreetMap.
    
    Handles:
    - OSM data download and caching
    - Coordinate projection to UTM
    - Building rasterization to obstacle mask
    - Metadata for CV integration
    """
    
    # UTM Zone 37S for Nairobi (EPSG:32737)
    NAIROBI_UTM_CRS = "EPSG:32737"
    
    # Default Nairobi CBD bounding box (lat/lon)
    DEFAULT_BBOX = {
        'north': -1.2800,
        'south': -1.2920,
        'east': 36.8250,
        'west': 36.8150
    }
    
    def __init__(self, config: Dict, cache_dir: str = "data/osm_cache"):
        """
        Initialize map loader.
        
        Args:
            config: Configuration dict with 'grid' and 'osm' sections
            cache_dir: Directory for caching OSM data
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract configuration
        self.osm_config = config.get('osm', {})
        self.grid_config = config['grid']
        
        # Get bounding box
        if 'bbox' in self.osm_config:
            self.bbox = self.osm_config['bbox']
        else:
            self.bbox = self.DEFAULT_BBOX
        
        # Grid parameters
        self.width = self.grid_config['width']
        self.height = self.grid_config['height']
        self.cell_size = self.grid_config['cell_size_m']
        
        # CRS
        self.crs = self.osm_config.get('crs', self.NAIROBI_UTM_CRS)
        
        # Downloaded data
        self.buildings_gdf = None
        self.streets_graph = None
        self.bounds_utm = None
        
    def load_or_download(self, force_download: bool = False) -> Dict:
        """
        Load cached OSM data or download if not available.
        
        Args:
            force_download: If True, re-download even if cached
            
        Returns:
            Dictionary with obstacle_mask, metadata, and geodataframes
        """
        cache_file = self._get_cache_filename()
        
        # Try loading from cache
        if not force_download and cache_file.exists():
            print(f"Loading cached OSM data from {cache_file}")
            return self._load_from_cache(cache_file)
        
        # Download fresh data
        print(f"Downloading OSM data for Nairobi CBD...")
        print(f"  Bounding box: {self.bbox}")
        
        try:
            self._download_osm_data()
            result = self._process_osm_data()
            self._save_to_cache(cache_file, result)
            return result
        
        except Exception as e:
            print(f"Error downloading OSM data: {e}")
            print("Falling back to synthetic obstacles")
            return None
    
    def _get_cache_filename(self) -> Path:
        """Generate cache filename based on config parameters."""
        bbox_str = f"{self.bbox['north']:.4f}_{self.bbox['south']:.4f}_" \
                   f"{self.bbox['east']:.4f}_{self.bbox['west']:.4f}"
        grid_str = f"{self.width}x{self.height}_{self.cell_size}m"
        return self.cache_dir / f"nairobi_cbd_{bbox_str}_{grid_str}.pkl"
    
    def _download_osm_data(self):
        """Download buildings and streets from OSM."""
        # Download buildings
        print("  Downloading buildings...")
        self.buildings_gdf = ox.features_from_bbox(
            bbox=(self.bbox['north'], self.bbox['south'], 
                  self.bbox['east'], self.bbox['west']),
            tags={'building': True}
        )
        
        # Project to UTM
        self.buildings_gdf = self.buildings_gdf.to_crs(self.crs)
        print(f"    Found {len(self.buildings_gdf)} buildings")
        
        # Download streets (optional, for visualization)
        if self.osm_config.get('features', {}).get('streets', True):
            print("  Downloading streets...")
            try:
                self.streets_graph = ox.graph_from_bbox(
                    bbox=(self.bbox['north'], self.bbox['south'],
                          self.bbox['east'], self.bbox['west']),
                    network_type='all'
                )
                self.streets_graph = ox.project_graph(
                    self.streets_graph,
                    to_crs=self.crs
                )
                print(f"    Found {len(self.streets_graph.nodes)} street nodes")
            except Exception as e:
                print(f"    Warning: Could not download streets: {e}")
                self.streets_graph = None
    
    def _process_osm_data(self) -> Dict:
        """Process downloaded data into obstacle mask and metadata."""
        print("  Processing OSM data...")
        
        # Get UTM bounds from buildings
        minx, miny, maxx, maxy = self.buildings_gdf.total_bounds
        self.bounds_utm = {
            'minx': minx,
            'miny': miny,
            'maxx': maxx,
            'maxy': maxy,
            'width_m': maxx - minx,
            'height_m': maxy - miny
        }
        
        print(f"    UTM bounds: {self.bounds_utm['width_m']:.0f}m × "
              f"{self.bounds_utm['height_m']:.0f}m")
        
        # Adjust grid to fit actual coverage
        actual_width_m = self.width * self.cell_size
        actual_height_m = self.height * self.cell_size
        
        # Use requested grid size, center on data
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        
        grid_minx = center_x - actual_width_m / 2
        grid_miny = center_y - actual_height_m / 2
        grid_maxx = center_x + actual_width_m / 2
        grid_maxy = center_y + actual_height_m / 2
        
        # Rasterize buildings
        obstacle_mask = self._rasterize_buildings(
            grid_minx, grid_miny, grid_maxx, grid_maxy
        )
        
        # Create metadata
        metadata = {
            'crs': self.crs,
            'bbox_latlon': self.bbox,
            'bounds_utm': self.bounds_utm,
            'grid_bounds_utm': {
                'minx': grid_minx,
                'miny': grid_miny,
                'maxx': grid_maxx,
                'maxy': grid_maxy
            },
            'width': self.width,
            'height': self.height,
            'cell_size_m': self.cell_size,
            'origin': 'top_left',
            'coordinate_system': 'utm'
        }
        
        result = {
            'obstacle_mask': obstacle_mask,
            'metadata': metadata,
            'buildings_gdf': self.buildings_gdf,
            'streets_graph': self.streets_graph
        }
        
        print(f"  Generated {self.width}×{self.height} obstacle grid")
        print(f"    {obstacle_mask.sum()} cells blocked "
              f"({100*obstacle_mask.sum()/obstacle_mask.size:.1f}%)")
        
        return result
    
    def _rasterize_buildings(self, minx: float, miny: float,
                            maxx: float, maxy: float) -> np.ndarray:
        """
        Rasterize building polygons to boolean grid.
        
        Args:
            minx, miny, maxx, maxy: Grid bounds in UTM coordinates
            
        Returns:
            obstacle_mask: Boolean array (True = building/impassable)
        """
        # Create rasterio transform
        transform = from_bounds(
            minx, miny, maxx, maxy,
            self.width, self.height
        )
        
        # Filter buildings within grid bounds
        grid_box = box(minx, miny, maxx, maxy)
        buildings_in_grid = self.buildings_gdf[
            self.buildings_gdf.geometry.intersects(grid_box)
        ]
        
        if len(buildings_in_grid) == 0:
            print("    Warning: No buildings found in grid bounds")
            return np.zeros((self.height, self.width), dtype=bool)
        
        # Create shapes for rasterization
        shapes = [
            (geom, 1) for geom in buildings_in_grid.geometry
            if geom is not None and geom.is_valid
        ]
        
        # Rasterize
        obstacle_grid = features.rasterize(
            shapes,
            out_shape=(self.height, self.width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        # Convert to boolean
        obstacle_mask = obstacle_grid.astype(bool)
        
        # Add border walls
        obstacle_mask[0, :] = True
        obstacle_mask[-1, :] = True
        obstacle_mask[:, 0] = True
        obstacle_mask[:, -1] = True
        
        return obstacle_mask
    
    def _save_to_cache(self, cache_file: Path, result: Dict):
        """Save processed data to cache."""
        print(f"  Saving to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    def _load_from_cache(self, cache_file: Path) -> Dict:
        """Load processed data from cache."""
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
        
        print(f"  ✓ Loaded {result['metadata']['width']}×"
              f"{result['metadata']['height']} grid from cache")
        return result
    
    @staticmethod
    def latlon_to_utm(lat: float, lon: float, crs: str) -> Tuple[float, float]:
        """
        Convert lat/lon to UTM coordinates.
        
        Args:
            lat, lon: Geographic coordinates
            crs: Target CRS (e.g., 'EPSG:32737')
            
        Returns:
            (easting, northing) in meters
        """
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        return transformer.transform(lon, lat)
    
    @staticmethod
    def utm_to_latlon(easting: float, northing: float, 
                     crs: str) -> Tuple[float, float]:
        """
        Convert UTM to lat/lon coordinates.
        
        Args:
            easting, northing: UTM coordinates in meters
            crs: Source CRS (e.g., 'EPSG:32737')
            
        Returns:
            (lat, lon) in degrees
        """
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        return lat, lon


def load_nairobi_cbd_map(config: Dict, 
                        force_download: bool = False) -> Optional[Dict]:
    """
    Convenience function to load Nairobi CBD map.
    
    Args:
        config: Configuration dictionary
        force_download: If True, re-download OSM data
        
    Returns:
        Dictionary with obstacle_mask and metadata, or None if failed
    """
    loader = NairobiCBDMap(config)
    return loader.load_or_download(force_download=force_download)