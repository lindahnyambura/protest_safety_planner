"""
main.py - FastAPI backend integrating RiskAwareRoutePlanner
"""

import uvicorn
from fastapi import FastAPI, Query
import time
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pathlib import Path
import yaml
import numpy as np
import networkx as nx
import json
import pyproj
import asyncio
import requests
from typing import Optional
from typing import Dict, List, Tuple
from affine import Affine
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scipy.spatial import KDTree
from pydantic import BaseModel


from PIL import Image
import io
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.ndimage import gaussian_filter

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import planner
from src.planner.route_planner import RiskAwareRoutePlanner

# Import report aggregator
from src.report_adapter.report_aggregator import (
    ReportAggregator, 
    update_edge_harm_with_aggregation
)

# Import fusion
from src.fusion.simple_fusion import SimpleFusionEngine, apply_fusion_to_graph


# Global state
aggregator = None
baseline_p_sim = {}  # Store original simulation probs
fusion_engine = None

# 0. Coordinate conversion
UTM_CRS = "EPSG:32737"  # UTM Zone 37S
WGS84_CRS = "EPSG:4326"
transformer = pyproj.Transformer.from_crs(UTM_CRS, WGS84_CRS, always_xy=True)

def utm_to_latlng(x_utm: float, y_utm: float) -> tuple[float, float]:
    """Convert UTM coordinates to lat/lng for Mapbox"""
    lng, lat = transformer.transform(x_utm, y_utm)
    return lat, lng

def convert_route_geometry(geometry_utm: list) -> list:
    """Convert route geometry from UTM to [lng, lat] for GeoJSON"""
    return [[lng, lat] for x, y in geometry_utm 
            for lng, lat in [transformer.transform(x, y)]]


# Cache for KDTree (build once on startup)
NODE_KDTREE = None
NODE_IDS = None

def build_node_kdtree(graph: nx.Graph):
    """Build KDTree for fast nearest node lookup"""
    global NODE_KDTREE, NODE_IDS
    
    coords = []
    node_ids = []
    
    for node_id, data in graph.nodes(data=True):
        if 'x' in data and 'y' in data:
            coords.append([float(data['x']), float(data['y'])])
            node_ids.append(node_id)
    
    NODE_KDTREE = KDTree(coords)
    NODE_IDS = np.array(node_ids)
    print(f"[Backend] Built KDTree with {len(node_ids)} nodes")

# 1. Initialize FastAPI app
app = FastAPI(
    title="Protest Safety Planner API",
    description="Backend for risk-aware route computation and comparison",
    version="1.0.0"
)

# Add CORS middleware after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load configuration and planner on startup
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "planner_config.yaml"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "rollouts_test" / "test_run"
# Global cache for street names
STREET_NAME_CACHE = {}
CACHE_FILE = BASE_DIR / "backend" / "street_names_cache.json"

planner = None
planner_config = None


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    # Get the default
    openapi_schema = get_openapi(
        title="Protest Safety Planner API",
        version="1.0.0",
        description="Backend for risk-aware route computation and comparison",
        routes=app.routes,
    )
    
    # Simplify the schema to avoid large objects in docs
    openapi_schema["components"]["schemas"] = {}
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


class ReportSubmission(BaseModel):
    type: str
    lat: float
    lng: float
    confidence: float
    notes: Optional[str] = None
    timestamp: Optional[str] = None

@app.post("/report")
async def submit_report(report: ReportSubmission):
    """
    Submit an anonymous hazard report
    Updates the risk map in real-time
    """
    try:
        import time
        # 1. Validate report type
        valid_types = ['safe', 'crowd', 'police', 'tear_gas', 'water_cannon']
        if report.type not in valid_types:
            return JSONResponse(
                {"error": f"Invalid type. Must be one of: {', '.join(valid_types)}"},
                status_code=400
            )
        
        # Validate confidence range
        if not 0 <= report.confidence <= 1:
            return JSONResponse(
                {"error": "Confidence must be between 0 and 1"},
                status_code=400
            )
        
        # Validate coordinates (rough Nairobi bounds)
        if not (-1.35 <= report.lat <= -1.20 and 36.70 <= report.lng <= 36.95):
            return JSONResponse(
                {"error": "Location outside Nairobi area"},
                status_code=400
            )
        
        print(f"[Backend] Report received: {report.type} at ({report.lat}, {report.lng})")
        
        # 2. Find nearest node using KDTree
        if NODE_KDTREE is None or NODE_IDS is None:
            return JSONResponse(
                {"error": "Graph index not initialized"},
                status_code=503
            )
        
        # Convert lat/lng to UTM for nearest node search
        transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32737", always_xy=True)
        x_utm, y_utm = transformer_to_utm.transform(report.lng, report.lat)
        
        # Find nearest node
        distance, index = NODE_KDTREE.query([x_utm, y_utm])
        nearest_node_id = str(NODE_IDS[index])
        
        print(f"[Backend] Snapped to node {nearest_node_id} (distance: {distance:.1f}m)")
        
        # 3. Create report record
        timestamp = time.time()
        report_record = {
            "type": report.type,
            "node_id": nearest_node_id,
            "confidence": report.confidence,
            "timestamp": timestamp,
            "expires_at": timestamp + REPORT_EXPIRY_SECONDS,
            "notes": report.notes
        }
        
        # Store report
        if nearest_node_id not in RECENT_REPORTS:
            RECENT_REPORTS[nearest_node_id] = []
        RECENT_REPORTS[nearest_node_id].append(report_record)
        
        # 4. UPDATED: Apply Fusion (replaces simple update_edge_harm)
        fusion_stats = apply_fusion_to_graph(
            planner.osm_graph,
            RECENT_REPORTS,
            aggregator,
            baseline_p_sim,
            fusion_engine
        )
        
        # 5. Gather and return updated statistics
        stats = aggregator.get_report_statistics(RECENT_REPORTS)
        
        print(f"[Backend] ✓ Fusion applied:")
        print(f"  - Active reports: {stats['total_active_reports']}")
        print(f"  - Edges updated: {fusion_stats['edges_updated']}")
        print(f"  - Mean change: {fusion_stats['mean_absolute_change']:.4f}")

        return {
            "status": "ok",
            "report_id": f"r_{int(timestamp)}",
            "node_id": nearest_node_id,
            "snapped_distance_m": float(distance),
            "expires_in_seconds": REPORT_EXPIRY_SECONDS,
            "graph_update": {
                "total_active_reports": stats.get("total_active_reports", 0),
                "nodes_affected": stats.get("nodes_affected", 0),
                "edges_updated": fusion_stats.get("edges_updated", 0),
                "mean_absolute_change": round(fusion_stats.get("mean_absolute_change", 0), 4)
            }
        }
        
    except Exception as e:
        import traceback
        print(f"[Backend] Report submission error: {e}")
        print(f"[Backend] Full traceback:")
        print(traceback.format_exc())  # <-- This will show you WHICH file is missing 'time'
        return JSONResponse(
            {"error": f"Internal server error: {str(e)}"},
            status_code=500
        )


@app.get("/reports/aggregated")
async def get_aggregated_report_data():
    """
    Return aggregated report probabilities with uncertainty bounds.
    Shows how reports are being statistically combined.
    """
    current_time = time.time()
    
    aggregated_data = []
    
    for node_id in RECENT_REPORTS.keys():
        p_report, ci_lower, ci_upper = aggregator.aggregate_node_reports(
            node_id, RECENT_REPORTS, current_time
        )
        
        if abs(p_report) > 0.001:  # Only return significant adjustments
            if node_id in NODE_TO_COORDS:
                lat, lng = NODE_TO_COORDS[node_id]
                
                # Count active reports at this node
                num_active = len([
                    r for r in RECENT_REPORTS[node_id]
                    if current_time - r['timestamp'] < aggregator.time_window
                ])
                
                aggregated_data.append({
                    "node_id": node_id,
                    "lat": lat,
                    "lng": lng,
                    "p_report": round(p_report, 3),
                    "ci_lower": round(ci_lower, 3),
                    "ci_upper": round(ci_upper, 3),
                    "num_reports": num_active,
                    "interpretation": (
                        "increased_risk" if p_report > 0.01 else
                        "decreased_risk" if p_report < -0.01 else
                        "neutral"
                    )
                })
    
    return {
        "aggregated_reports": aggregated_data,
        "aggregation_method": "weighted_average_with_ci",
        "time_window_seconds": aggregator.time_window,
        "timestamp": int(current_time * 1000)
    }


@app.on_event("startup")
async def start_report_expiry_task():
    """Enhanced expiry task that re-applies fusion after cleanup"""
    
    async def expire_old_reports():
        while True:
            await asyncio.sleep(60)  # Check every minute
            current_time = time.time()
            
            # Remove expired reports
            for node_id in list(RECENT_REPORTS.keys()):
                RECENT_REPORTS[node_id] = [
                    r for r in RECENT_REPORTS[node_id]
                    if r['expires_at'] > current_time
                ]
                
                if not RECENT_REPORTS[node_id]:
                    del RECENT_REPORTS[node_id]
            
            # Re-apply fusion with remaining reports
            if aggregator and fusion_engine and baseline_p_sim:
                fusion_stats = apply_fusion_to_graph(
                    planner.osm_graph,
                    RECENT_REPORTS,
                    aggregator,
                    baseline_p_sim,
                    fusion_engine
                )
                
                stats = aggregator.get_report_statistics(RECENT_REPORTS)
                print(f"[Backend] Periodic update: {stats['total_active_reports']} active reports")
    
    asyncio.create_task(expire_old_reports())  


# Global node mapping cache
NODE_TO_COORDS: Dict[str, Tuple[float, float]] = {}  # node_id -> (lat, lng)
LANDMARK_TO_NODE: Dict[str, str] = {}  # landmark_name -> node_id

def build_node_coordinate_cache(graph: nx.Graph):
    """
    Build a mapping of node IDs to lat/lng coordinates
    Uses proper UTM to WGS84 transformation
    """
    global NODE_TO_COORDS
    
    transformer = pyproj.Transformer.from_crs("EPSG:32737", "EPSG:4326", always_xy=True)
    
    for node_id, data in graph.nodes(data=True):
        x_utm = float(data.get('x', 0))
        y_utm = float(data.get('y', 0))
        
        lng, lat = transformer.transform(x_utm, y_utm)
        NODE_TO_COORDS[node_id] = (lat, lng)
    
    print(f"[Backend] Cached coordinates for {len(NODE_TO_COORDS)} nodes")

def load_landmark_mappings():
    """
    Load landmark to node mappings from your script output
    Run your script first and save the output
    """
    global LANDMARK_TO_NODE
    
    # Correct coordinates from Google Maps (lat, lng format)
    landmarks_with_coords = {
        "Bus Station": (-1.288275, 36.828192),
        "National Archives": (-1.2848354, 36.8214961),
        "Uhuru Park": (-1.2900825, 36.8174183),
        "KICC": (-1.2882881, 36.820189),
        "Koja": (-1.2818321, 36.8206052),
        "Times Tower": (-1.2901877, 36.8214664),
        "Railway Station": (-1.2908054, 36.8250816),
        "Jamia Mosque": (-1.2832261, 36.8179915),
        "GPO": (-1.2860694, 36.8162181),
        "Afya Center": (-1.2878776, 36.8271596),
        "Odeon": (-1.282821, 36.824996),
        "Kencom": (-1.285640, 36.824984),
        "City Market": (-1.2836408, 36.8168827),
    }
    
    # Find nearest nodes for each landmark
    import pyproj
    transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32737", always_xy=True)
    
    for name, (lat, lng) in landmarks_with_coords.items():
        try:
            # Convert to UTM
            x_utm, y_utm = transformer_to_utm.transform(lng, lat)
            
            # Find nearest node using KDTree
            if NODE_KDTREE is not None and NODE_IDS is not None:
                distance, index = NODE_KDTREE.query([x_utm, y_utm])
                nearest_node_id = str(NODE_IDS[index])
                
                LANDMARK_TO_NODE[name.lower()] = nearest_node_id
                print(f"[Backend] Mapped '{name}' to node {nearest_node_id} (distance: {distance:.1f}m)")
        except Exception as e:
            print(f"[Backend] Failed to map landmark '{name}': {e}")
    
    print(f"[Backend] Loaded {len(LANDMARK_TO_NODE)} landmark mappings")

def load_street_name_cache():
    """Load cached street names on startup"""
    global STREET_NAME_CACHE
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            STREET_NAME_CACHE = json.load(f)
        print(f"[Backend] Loaded {len(STREET_NAME_CACHE)} cached street names")
    else:
        STREET_NAME_CACHE = {}
        print("[Backend] No street name cache found, will build on demand")

def save_street_name_cache():
    """Save cache to disk"""
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(STREET_NAME_CACHE, f, indent=2)

def is_placeholder_name(name: str) -> bool:
    """Check if name is a placeholder like 'unnamed_1115308925'"""
    if not name:
        return True
    name_lower = name.lower().strip()
    return (
        name_lower == '' or
        name_lower.startswith('unnamed') or
        name_lower == 'road' or
        name_lower == 'street' or
        name_lower == 'none'
    )

def query_nominatim_for_street(lat: float, lng: float) -> Optional[str]:
    """
    Query OSM Nominatim API for street name at coordinates
    Returns None if failed or rate limited
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lng,
            'format': 'json',
            'zoom': 18,
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'ProtestSafetyPlanner/1.0 (Academic Research)'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            
            # Try multiple address fields in priority order
            for field in ['road', 'pedestrian', 'footway', 'path', 'cycleway']:
                street = address.get(field)
                if street and not is_placeholder_name(street):
                    return street
            
            # Try suburb/neighbourhood as fallback
            suburb = address.get('suburb') or address.get('neighbourhood')
            if suburb:
                return f"{suburb} Area"
        
        return None
        
    except Exception as e:
        print(f"Nominatim query failed: {e}")
        return None
    
def get_coordinate_based_name(lat: float, lng: float) -> str:
    """
    Generate street name based on known Nairobi CBD landmarks
    This is the ultimate fallback
    """
    # Nairobi CBD grid with major streets
    # Based on typical CBD layout
    
    # North-South division (latitude)
    if -1.284 <= lat <= -1.282:
        area_ns = "Upper CBD"
    elif -1.286 <= lat <= -1.284:
        area_ns = "Central CBD"
    elif -1.288 <= lat <= -1.286:
        area_ns = "Mid CBD"
    elif -1.290 <= lat <= -1.288:
        area_ns = "Lower CBD"
    else:
        area_ns = "CBD"
    
    # East-West division (longitude)
    if 36.819 <= lng <= 36.822:
        area_ew = "West"
    elif 36.822 <= lng <= 36.826:
        area_ew = "Central"
    elif 36.826 <= lng <= 36.830:
        area_ew = "East"
    elif 36.830 <= lng <= 36.835:
        area_ew = "Far East"
    else:
        area_ew = ""
    
    # Known major streets by approximate coordinates
    major_streets = {
        (-1.287, 36.825): "Kenyatta Avenue",
        (-1.286, 36.824): "Moi Avenue", 
        (-1.285, 36.823): "Tom Mboya Street",
        (-1.288, 36.822): "Haile Selassie Avenue",
        (-1.287, 36.826): "University Way",
        (-1.284, 36.825): "Uhuru Highway",
    }
    
    # Find closest major street
    min_dist = float('inf')
    closest_street = None
    for (st_lat, st_lng), name in major_streets.items():
        dist = ((lat - st_lat)**2 + (lng - st_lng)**2)**0.5
        if dist < min_dist and dist < 0.003:  # Within ~300m
            min_dist = dist
            closest_street = name
    
    if closest_street:
        return closest_street
    
    # Final fallback: area-based naming
    return f"{area_ns} {area_ew}".strip()

def grid_to_lat_lng(i: int, j: int) -> tuple[float, float]:
    """
    Convert grid indices to latitude/longitude using affine transform from metadata.
    
    Args:
        i: row index (0 = top)
        j: column index (0 = left)
        
    Returns:
        (lat, lng) in WGS84 (EPSG:4326)
    """
    try:
        # Load metadata
        metadata_path = DATA_DIR / "real_nairobi_cbd_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get affine transform parameters
        affine_params = metadata.get("affine_transform")
        if not affine_params:
            raise ValueError("No affine_transform in metadata")
        
        # Create affine transform
        affine = Affine(*affine_params[:6])
        
        # Convert grid cell center to UTM coordinates
        # Note: i is row (0 = top), j is column (0 = left)
        x_utm, y_utm = affine * (j + 0.5, i + 0.5)
        
        # Convert UTM to lat/lng (WGS84)
        import pyproj
        
        # Define UTM 37S (your TARGET_CRS) and WGS84
        utm_crs = "EPSG:32737"  # UTM zone 37S, WGS84
        wgs84_crs = "EPSG:4326"  # Standard lat/lng
        
        transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
        lng, lat = transformer.transform(x_utm, y_utm)
        
        return lat, lng
        
    except Exception as e:
        print(f"Coordinate conversion error at ({i}, {j}): {e}")
        # Fallback: use bounds for approximate conversion
        return _grid_to_lat_lng_fallback(i, j)

def get_street_name_from_edge(graph: nx.Graph, node1: str, node2: str) -> str:
    """
    Foolproof street name extraction with multiple fallbacks:
    1. Check cache
    2. Extract from OSM edge data
    3. Query Nominatim API (with rate limiting)
    4. Use coordinate-based naming
    
    This WILL NOT FAIL.
    """
    cache_key = f"{node1}-{node2}"
    
    try:
        # STEP 1: Check cache first
        if cache_key in STREET_NAME_CACHE:
            cached_name = STREET_NAME_CACHE[cache_key]
            if not is_placeholder_name(cached_name):
                return cached_name
        
        # STEP 2: Try to get from OSM edge data
        if not graph.has_edge(node1, node2):
            return get_coordinate_based_name_for_node(graph, node1)
        
        # Handle MultiGraph
        if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
            edges = graph[node1][node2]
            edge_data = edges[0] if isinstance(edges, dict) else list(edges.values())[0]
        else:
            edge_data = graph[node1][node2]
        
        # Check if OSM has a real name
        osm_name = edge_data.get('name', '')
        if osm_name and not is_placeholder_name(osm_name):
            STREET_NAME_CACHE[cache_key] = osm_name
            return osm_name
        
        # STEP 3: Calculate coordinates for API query
        node1_data = graph.nodes[node1]
        node2_data = graph.nodes[node2]
        
        x1, y1 = float(node1_data['x']), float(node1_data['y'])
        x2, y2 = float(node2_data['x']), float(node2_data['y'])
        
        # Use midpoint of edge
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        
        lat, lng = utm_to_latlng(x_mid, y_mid)
        
        # STEP 4: Try Nominatim (with rate limiting)
        nominatim_name = query_nominatim_for_street(lat, lng)
        if nominatim_name and not is_placeholder_name(nominatim_name):
            STREET_NAME_CACHE[cache_key] = nominatim_name
            save_street_name_cache()  # Save after each successful query
            time.sleep(1.1)  # Respect Nominatim rate limit
            return nominatim_name
        
        # STEP 5: Use coordinate-based naming (CANNOT FAIL)
        fallback_name = get_coordinate_based_name(lat, lng)
        STREET_NAME_CACHE[cache_key] = fallback_name
        
        return fallback_name
        
    except Exception as e:
        print(f"Street name extraction error for {node1}-{node2}: {e}")
        # ABSOLUTE FALLBACK: return something based on node position
        try:
            return get_coordinate_based_name_for_node(graph, node1)
        except:
            return "CBD Road"

def get_coordinate_based_name_for_node(graph: nx.Graph, node: str) -> str:
    """Emergency fallback using single node"""
    try:
        node_data = graph.nodes[node]
        x, y = float(node_data['x']), float(node_data['y'])
        lat, lng = utm_to_latlng(x, y)
        return get_coordinate_based_name(lat, lng)
    except:
        return "CBD Road"

def generate_turn_by_turn_directions(
    graph: nx.Graph,
    path: list,
    geometry: list
) -> list[dict]:
    """
    Generate human-readable turn-by-turn directions with street names
    Uses foolproof street name extraction
    """
    directions = []
    current_street = None
    
    for i, node in enumerate(path):
        if i == 0:
            # Start instruction
            if len(path) > 1:
                current_street = get_street_name_from_edge(graph, path[0], path[1])
            else:
                current_street = "your location"
            
            directions.append({
                "step": i,
                "instruction": f"Start on {current_street}",
                "node": node,
                "distance_m": 0.0,
                "street_name": current_street
            })
            
        elif i == len(path) - 1:
            # End instruction
            directions.append({
                "step": i,
                "instruction": "Arrive at destination",
                "node": node,
                "distance_m": 0.0,
                "street_name": None
            })
            
        else:
            # Intermediate steps
            prev_node = path[i-1]
            next_node = path[i+1]
            
            # Get street name for upcoming segment
            next_street = get_street_name_from_edge(graph, node, next_node)
            
            # Calculate distance to next waypoint
            if i < len(geometry) - 1:
                x1, y1 = geometry[i]
                x2, y2 = geometry[i+1]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            else:
                distance = 0.0
            
            # Determine instruction based on street change
            if current_street and next_street != current_street:
                instruction = f"Turn onto {next_street}"
            else:
                instruction = f"Continue on {next_street}"
            
            directions.append({
                "step": i,
                "instruction": instruction,
                "node": node,
                "distance_m": round(distance, 1),
                "street_name": next_street
            })
            
            current_street = next_street
    
    return directions

def _grid_to_lat_lng_fallback(i: int, j: int) -> tuple[float, float]:
    """Fallback conversion using bounds from metadata"""
    try:
        metadata_path = DATA_DIR / "real_nairobi_cbd_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get grid dimensions
        height, width = metadata.get("grid_size", [200, 200])
        
        # Get UTM bounds
        xmin, ymin, xmax, ymax = metadata.get("bounds", [257178.29204338358, 9857069.74174053, 258178.29204338358, 9858069.74174053])
        
        # Convert current cell to UTM
        cell_width = (xmax - xmin) / width
        cell_height = (ymax - ymin) / height
        
        x_utm = xmin + (j + 0.5) * cell_width
        y_utm = ymax - (i + 0.5) * cell_height  # Note: y decreases as i increases
        
        # UTM to lat/lng approximate conversion for Nairobi area
        lat = -1.295 + (y_utm - ymin) / (ymax - ymin) * ( -1.280 + 1.295 )  # more dynamic mapping
        lng = 36.810 + (x_utm - xmin) / (xmax - xmin) * ( 36.835 - 36.810 )  # Account for longitude compression
        
        return lat, lng
        
    except Exception as e:
        print(f"Fallback conversion also failed: {e}")
        # Final fallback - centered on Nairobi CBD
        return -1.2875, 36.8225


def generate_risk_heatmap_png(p_sim: np.ndarray) -> io.BytesIO:
    """
    Convert risk probability grid to a smooth PNG heatmap overlay.
    - Low risk: fully transparent
    - Moderate → High risk: green → yellow → red
    - Smooth circular gradients for realism
    """

    # --- Clean and normalize input ---
    p_sim = np.nan_to_num(p_sim, nan=0.0, posinf=1.0, neginf=0.0)
    p_sim = np.clip(p_sim, 0, 1)

    # --- Smooth spatially (makes gradients circular & neat) ---
    smooth_p = gaussian_filter(p_sim, sigma=2)

    # --- Colormap: vivid green → yellow → red ---
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "green_yellow_red",
        ["#3CB371", "#FFD700", "#DC143C"]
    )

    norm = plt.Normalize(vmin=0, vmax=smooth_p.max())
    rgba = cmap(norm(smooth_p))

    # --- Transparency: fade out low risk completely ---
    # 0 risk = 0 alpha (transparent), high risk = up to 0.85 opacity
    alpha = np.clip((smooth_p / smooth_p.max()) ** 1.5 * 0.85, 0, 0.85)
    rgba[:, :, 3] = alpha

    # --- Convert to image ---
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode='RGBA')

    # --- Save to buffer ---
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    return buffer

@app.on_event("startup")
def load_planner():
    global planner, planner_config, aggregator, fusion_engine, baseline_p_sim

    print("[Backend] Starting initialization...")
    
    # 1. Load planner config
    with open(CONFIG_PATH, "r") as f:
        planner_config = yaml.safe_load(f)

    # 2. Load OSM graph and fix coordinate types
    graph_path = DATA_DIR / "nairobi_walk.graphml"
    osm_graph = nx.read_graphml(graph_path)
    
    # Convert node coordinates from strings to floats
    for node, data in osm_graph.nodes(data=True):
        if 'x' in data:
            data['x'] = float(data['x'])
        if 'y' in data:
            data['y'] = float(data['y'])
    
    print(f"[Backend] Loaded OSM graph with {len(osm_graph.nodes)} nodes.")

    # 3. Load harm probabilities (simulation results) and cell-to-node mapping
    p_sim_path = ARTIFACTS_DIR / "p_sim.npy"
    p_sim = np.load(p_sim_path)
    print(f"[Backend] Loaded harm probability grid: {p_sim.shape}")

    # Optional mapping
    cell_to_node_path = DATA_DIR / "cell_to_node.npy"
    cell_to_node = np.load(cell_to_node_path, allow_pickle=True)

    # 4. Initialize RiskAwareRoutePlanner
    planner = RiskAwareRoutePlanner(
        osm_graph=osm_graph,
        p_sim=p_sim,
        config=planner_config,
        cell_to_node=cell_to_node
    )
    # 5. Build caches and supporting indices
    # Build coordinate cache
    build_node_coordinate_cache(planner.osm_graph)
    # Load landmark mappings
    load_landmark_mappings()
    # Build KDTree for nearest node lookup
    build_node_kdtree(planner.osm_graph)
    # Load street name cache
    load_street_name_cache()

    # 6. Initialize the report aggregator
    aggregator = ReportAggregator(
        time_window=300,      # 5 minutes
        spatial_radius=1,     # Affect immediate neighbors
        confidence_level=0.95
    )
    print("[Backend] Report Aggregator initialized.")

    # Initialize Fusion Engine
    fusion_engine = SimpleFusionEngine(
        simulation_weight=0.7,    # 70% trust simulation
        report_weight=0.3,        # 30% trust reports
        use_uncertainty_weighting=True
    )
    print("[Backend] ✓ Fusion Engine initialized")

    # 7. Store baseline harm probabilities (for later reset)
    for u, v in planner.osm_graph.edges():
        if isinstance(planner.osm_graph, nx.MultiDiGraph):
            for key in planner.osm_graph[u][v]:
                edge_data = planner.osm_graph[u][v][key]
                baseline_p_sim[(u, v, key)] = edge_data.get('p_harm', 0.0)
        else:
            edge_data = planner.osm_graph[u][v]
            baseline_p_sim[(u, v)] = edge_data.get('p_harm', 0.0)
    print(f"[Backend] Stored {len(baseline_p_sim)} baseline edge probabilities")

    print("[Backend] Planner initialized successfully.")
    print("[Backend] Fusion pipeline ready")


# 3. API Endpoints

@app.get("/fusion/stats")
async def get_fusion_statistics():
    """
    Return statistics about the current fusion state.
    Useful for dashboard/debugging.
    """
    stats = aggregator.get_report_statistics(RECENT_REPORTS)
    
    # Count how many edges differ from baseline
    edges_modified = 0
    total_edges = 0
    max_delta = 0.0
    
    for edge_key, baseline_p in baseline_p_sim.items():
        total_edges += 1
        
        # Get current p_harm
        if len(edge_key) == 3:  # MultiDiGraph
            u, v, key = edge_key
            if planner.osm_graph.has_edge(u, v):
                current_p = planner.osm_graph[u][v][key].get('p_harm', 0.0)
        else:  # DiGraph
            u, v = edge_key
            if planner.osm_graph.has_edge(u, v):
                current_p = planner.osm_graph[u][v].get('p_harm', 0.0)
        
        delta = abs(current_p - baseline_p)
        if delta > 0.001:
            edges_modified += 1
            max_delta = max(max_delta, delta)
    
    return {
        "fusion_active": True,
        "simulation_weight": fusion_engine.sim_weight,
        "report_weight": fusion_engine.report_weight,
        "reports": {
            "total_active": stats['total_active_reports'],
            "nodes_affected": stats['nodes_affected'],
            "by_type": stats['by_type']
        },
        "graph_state": {
            "total_edges": total_edges,
            "edges_modified": edges_modified,
            "percent_modified": round(100 * edges_modified / total_edges, 2),
            "max_probability_change": round(max_delta, 4)
        },
        "interpretation": (
            f"{stats['total_active_reports']} reports influencing "
            f"{edges_modified} edges ({100*edges_modified/total_edges:.1f}%)"
        )
    }

@app.get("/nearest-node")
def get_nearest_node(lat: float = Query(...), lng: float = Query(...)):
    """Find the nearest OSM node to given lat/lng coordinates"""
    if planner is None or NODE_KDTREE is None:
        return JSONResponse({"error": "Planner not initialized"}, status_code=503)
    
    try:
        # Convert lat/lng to UTM
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32737", always_xy=True)
        x_utm, y_utm = transformer.transform(lng, lat)
        
        # Find nearest node
        distance, index = NODE_KDTREE.query([x_utm, y_utm])
        nearest_node_id = NODE_IDS[index]
        
        # Get node data
        node_data = planner.osm_graph.nodes[nearest_node_id]
        
        return {
            "node_id": nearest_node_id,
            "distance_m": float(distance),
            "coordinates": {
                "lat": lat,
                "lng": lng,
                "x_utm": float(node_data['x']),
                "y_utm": float(node_data['y'])
            }
        }
        
    except Exception as e:
        print(f"Nearest node error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/debug/harm-at-node")
def debug_harm_at_node(node: str = Query(...)):
    """Check harm probability at a specific node"""
    if planner is None:
        return {"error": "Planner not initialized"}
    
    if node not in planner.osm_graph.nodes:
        return {"error": f"Node {node} not found"}
    
    # Get all edges from this node
    edges_harm = []
    for neighbor in planner.osm_graph.neighbors(node):
        edge_data = planner.osm_graph[node][neighbor]
        if isinstance(planner.osm_graph, nx.MultiDiGraph):
            edge_data = edge_data[0]
        
        harm = edge_data.get('p_harm', 0.0)
        edges_harm.append({
            "to_node": neighbor,
            "p_harm": float(harm)
        })
    
    return {
        "node_id": node,
        "edges": edges_harm,
        "total_edges": len(edges_harm),
        "max_harm": max([e['p_harm'] for e in edges_harm]) if edges_harm else 0.0
    }

@app.get("/landmarks")
def get_landmarks(limit: int = Query(default=13, ge=1, le=13)):
    """
    Return landmarks with their CORRECT Google Maps coordinates
    
    Args:
        limit: Number of landmarks to return (1-13). Defaults to all 13.
               Frontend can request fewer for cleaner UI.
    """
    
    # Correct coordinates from Google Maps (ordered by popularity/importance)
    landmarks_data = [
        {"name": "KICC", "lat": -1.2882881, "lng": 36.820189},
        {"name": "Railway Station", "lat": -1.2908054, "lng": 36.8250816},
        {"name": "Uhuru Park", "lat": -1.2900825, "lng": 36.8174183},
        {"name": "National Archives", "lat": -1.2848354, "lng": 36.8214961},
        {"name": "Afya Center", "lat": -1.2878776, "lng": 36.8271596},
        {"name": "GPO", "lat": -1.2860694, "lng": 36.8162181},
        {"name": "Jamia Mosque", "lat": -1.2832261, "lng": 36.8179915},
        {"name": "Kencom", "lat": -1.285640, "lng": 36.824984},
        {"name": "Bus Station", "lat": -1.288275, "lng": 36.828192},
        {"name": "Times Tower", "lat": -1.2901877, "lng": 36.8214664},
        {"name": "Odeon", "lat": -1.282821, "lng": 36.824996},
        {"name": "City Market", "lat": -1.2836408, "lng": 36.8168827},
        {"name": "Koja", "lat": -1.2818321, "lng": 36.8206052},
    ]
    
    # Limit the results
    limited_landmarks = landmarks_data[:limit]
    
    # Enrich with node IDs
    landmarks = []
    for landmark in limited_landmarks:
        node_id = LANDMARK_TO_NODE.get(landmark["name"].lower())
        landmarks.append({
            "name": landmark["name"],
            "node_id": node_id,
            "coordinates": {"lat": landmark["lat"], "lng": landmark["lng"]}
        })
    
    return {
        "landmarks": landmarks,
        "total_available": len(landmarks_data),
        "returned": len(landmarks)
    }

@app.get("/nearest-landmark")
def get_nearest_landmark(lat: float = Query(...), lng: float = Query(...)):
    """
    Find the nearest landmark to given coordinates
    Uses Haversine distance
    """
    def haversine(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000  # Earth radius in meters
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))
    
    nearest_landmark = None
    min_distance = float('inf')
    
    for name, node_id in LANDMARK_TO_NODE.items():
        if node_id in NODE_TO_COORDS:
            node_lat, node_lng = NODE_TO_COORDS[node_id]
            distance = haversine(lat, lng, node_lat, node_lng)
            
            if distance < min_distance:
                min_distance = distance
                nearest_landmark = {
                    "name": name.title(),
                    "node_id": node_id,
                    "distance_m": round(distance),
                    "coordinates": {"lat": node_lat, "lng": node_lng}
                }
    
    return nearest_landmark if nearest_landmark else {"error": "No landmarks found"}


@app.get("/riskmap-image")
async def get_risk_heatmap_image():
    """
    Serve risk heatmap as PNG overlay image
    Returns: PNG image with transparent low-risk areas
    """
    try:
        p_sim_path = ARTIFACTS_DIR / "p_sim.npy"
        p_sim = np.load(p_sim_path)
        
        png_buffer = generate_risk_heatmap_png(p_sim)
        
        return StreamingResponse(
            png_buffer,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=60",  # Cache for 1 minute
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        print(f"Risk heatmap image error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/riskmap-bounds")
async def get_risk_heatmap_bounds():
    """
    Return geographic bounds for risk heatmap overlay
    Format: [west, south, east, north] in WGS84
    """
    try:
        metadata_path = DATA_DIR / "real_nairobi_cbd_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get UTM bounds
        bounds_utm = metadata.get("bounds_m")
        if not bounds_utm:
            # Fallback to your config values
            return {
                "bounds": [36.81, -1.295, 36.835, -1.28],  # [west, south, east, north]
                "source": "config"
            }
        
        xmin, ymin, xmax, ymax = bounds_utm
        
        # Convert corners to lat/lng
        sw_lng, sw_lat = transformer.transform(xmin, ymin)
        ne_lng, ne_lat = transformer.transform(xmax, ymax)
        
        return {
            "bounds": [sw_lng, sw_lat, ne_lng, ne_lat],
            "source": "metadata"
        }
        
    except Exception as e:
        print(f"Bounds error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/riskmap")
async def get_risk_map():
    """Serve risk data for map overlay - optimized for 200x200 grid"""
    try:
        # Load your Monte Carlo output
        p_sim_path = ARTIFACTS_DIR / "p_sim.npy"
        p_sim = np.load(p_sim_path)
        
        # Load metadata for bounds checking
        metadata_path = DATA_DIR / "real_nairobi_cbd_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        height, width = p_sim.shape
        risk_features = []
        
        # Sample strategy: take every 2nd cell to reduce data by 75%
        stride = 2
        risk_threshold = 0.02  # Only show meaningful risks
        
        for i in range(0, height, stride):
            for j in range(0, width, stride):
                risk_value = float(p_sim[i, j])
                if risk_value > risk_threshold:
                    try:
                        lat, lng = grid_to_lat_lng(i, j)
                        
                        # Basic bounds checking for Nairobi CBD approximate area
                        if (-1.29 < lat < -1.27) and (36.80 < lng < 36.83):
                            feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [lng, lat]
                                },
                                "properties": {
                                    "risk": risk_value,
                                    "cell_id": f"{i}_{j}",
                                    "intensity": "high" if risk_value > 0.3 else "medium" if risk_value > 0.1 else "low"
                                }
                            }
                            risk_features.append(feature)
                    except Exception as e:
                        # Skip this cell if conversion fails
                        continue
        
        print(f"[Backend] Generated risk map with {len(risk_features)} features")
        return {
            "type": "FeatureCollection",
            "features": risk_features
        }
        
    except Exception as e:
        print(f"Risk map error: {e}")
        return JSONResponse({"error": f"Failed to load risk map: {str(e)}"}, status_code=500)

@app.get("/geocode")
async def geocode_destination(q: str = Query(..., description="Search query")):
    """
    Geocode a destination name within Nairobi CBD bounds.
    Returns nearest node if found, or suggestions if ambiguous.
    
    Args:
        q: Search query (e.g., "Tom Mboya Street", "GPO", "Kencom")
    
    Returns:
        {
            "found": bool,
            "node_id": str | None,
            "coordinates": {"lat": float, "lng": float},
            "distance_m": float,
            "matched_name": str,
            "suggestions": List[str]  # If multiple matches
        }
    """
    # Define strict Nairobi CBD bbox
    bbox = {
        "north": -1.280,
        "south": -1.295,
        "east": 36.835,
        "west": 36.810,
    }
    
    try:
        # STEP 1: Check if query matches a known landmark
        query_lower = q.lower().strip()
        
        for name, node_id in LANDMARK_TO_NODE.items():
            if query_lower in name or name in query_lower:
                if node_id in NODE_TO_COORDS:
                    lat, lng = NODE_TO_COORDS[node_id]
                    return {
                        "found": True,
                        "node_id": node_id,
                        "coordinates": {"lat": lat, "lng": lng},
                        "distance_m": 0.0,
                        "matched_name": name.title(),
                        "type": "landmark"
                    }
        
        # STEP 2: Query Nominatim with strict bbox
        bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': q,
            'format': 'json',
            'limit': 5,
            'bounded': 1,
            'viewbox': bbox_str,
            'countrycodes': 'ke',
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'ProtestSafetyPlanner/1.0 (Academic Research)'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code != 200:
            return {
                "found": False,
                "error": f"Geocoding service returned {response.status_code}",
                "suggestions": []
            }
        
        results = response.json()
        
        if not results:
            return {
                "found": False,
                "error": "No locations found matching your query",
                "suggestions": ["Try a landmark like 'KICC' or 'Railway Station'"]
            }
        
        # STEP 3: Filter results within bbox
        valid_results = []
        for result in results:
            lat = float(result['lat'])
            lng = float(result['lon'])
            
            if (bbox['south'] <= lat <= bbox['north'] and 
                bbox['west'] <= lng <= bbox['east']):
                valid_results.append({
                    "lat": lat,
                    "lng": lng,
                    "display_name": result.get('display_name', ''),
                    "type": result.get('type', ''),
                    "class": result.get('class', '')
                })
        
        if not valid_results:
            return {
                "found": False,
                "error": "Location found but outside Nairobi CBD area",
                "suggestions": ["Search within Nairobi CBD bounds"]
            }
        
        # STEP 4: Find nearest node for best result
        best_result = valid_results[0]
        lat, lng = best_result['lat'], best_result['lng']
        
        # Convert to UTM
        import pyproj
        transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32737", always_xy=True)
        x_utm, y_utm = transformer_to_utm.transform(lng, lat)
        
        # Find nearest node using KDTree
        if NODE_KDTREE is None or NODE_IDS is None:
            return {
                "found": False,
                "error": "Graph index not available"
            }
        
        distance, index = NODE_KDTREE.query([x_utm, y_utm])
        nearest_node_id = str(NODE_IDS[index])
        
        # STEP 5: Return result with alternatives
        suggestions = [
            result['display_name'].split(',')[0] 
            for result in valid_results[1:4]
        ]
        
        return {
            "found": True,
            "node_id": nearest_node_id,
            "coordinates": {"lat": lat, "lng": lng},
            "distance_m": float(distance),
            "matched_name": best_result['display_name'].split(',')[0],
            "type": "geocoded",
            "suggestions": suggestions
        }
        
    except Exception as e:
        import traceback
        print(f"[Backend] Geocode error: {e}")
        print(traceback.format_exc())
        
        return {
            "found": False,
            "error": f"Geocoding failed: {str(e)}",
            "suggestions": []
        }


@app.get("/search-destinations")
async def search_destinations(q: str = Query(..., min_length=2)):
    """
    Autocomplete search for destinations within Nairobi CBD.
    Returns matching landmarks and street names.
    
    Args:
        q: Search query (minimum 2 characters)
    
    Returns:
        {
            "results": [
                {
                    "name": str,
                    "type": "landmark" | "street" | "place",
                    "node_id": str,
                    "coordinates": {"lat": float, "lng": float}
                }
            ]
        }
    """
    query_lower = q.lower().strip()
    results = []
    
    # Search landmarks
    for name, node_id in LANDMARK_TO_NODE.items():
        if query_lower in name and node_id in NODE_TO_COORDS:
            lat, lng = NODE_TO_COORDS[node_id]
            results.append({
                "name": name.title(),
                "type": "landmark",
                "node_id": node_id,
                "coordinates": {"lat": lat, "lng": lng}
            })
    
    # Search street names from cache
    matching_streets = set()
    for cache_key, street_name in STREET_NAME_CACHE.items():
        if query_lower in street_name.lower() and not is_placeholder_name(street_name):
            matching_streets.add(street_name)
    
    for street_name in list(matching_streets)[:5]:  # Limit to 5 streets
        results.append({
            "name": street_name,
            "type": "street",
            "node_id": None,  # Would need geocoding
            "coordinates": None
        })
    
    return {"results": results[:10]}  # Return top 10 results

@app.get("/riskmap-raster")
async def get_risk_map_raster():
    """Return risk data as a raster grid for heatmap visualization"""
    try:
        # Load your Monte Carlo output
        p_sim_path = ARTIFACTS_DIR / "p_sim.npy"
        p_sim = np.load(p_sim_path)
        
        # Load metadata
        metadata_path = DATA_DIR / "real_nairobi_cbd_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get bounds and convert corners to lat/lng
        bounds_utm = metadata.get("bounds")
        if bounds_utm:
            xmin, ymin, xmax, ymax = bounds_utm
            import pyproj
            transformer = pyproj.Transformer.from_crs("EPSG:32737", "EPSG:4326", always_xy=True)
            
            # Convert bounds corners
            sw_lng, sw_lat = transformer.transform(xmin, ymin)
            ne_lng, ne_lat = transformer.transform(xmax, ymax)
            
            bounds_latlng = [sw_lng, sw_lat, ne_lng, ne_lat]
        else:
            bounds_latlng = [36.810, -1.295, 36.835, -1.280]  # Fallback
        
        return {
            "type": "raster",
            "bounds": bounds_latlng,
            "data": p_sim.tolist(),  # 2D array of risk values
            "risk_range": [float(p_sim.min()), float(p_sim.max())]
        }
        
    except Exception as e:
        print(f"Raster risk map error: {e}")
        return JSONResponse({"error": f"Failed to load raster risk map: {str(e)}"}, status_code=500)

@app.post("/admin/build-street-cache")
async def build_street_name_cache_endpoint(max_edges: int = 100):
    """
    Admin endpoint to pre-build street name cache
    Call this once after deployment
    """
    if planner is None:
        return {"error": "Planner not initialized"}
    
    edges_processed = 0
    edges_found = 0
    
    # Process most common edges in your routes
    for u, v in list(planner.osm_graph.edges())[:max_edges]:
        name = get_street_name_from_edge(planner.osm_graph, str(u), str(v))
        edges_processed += 1
        
        if not is_placeholder_name(name):
            edges_found += 1
    
    save_street_name_cache()
    
    return {
        "edges_processed": edges_processed,
        "names_found": edges_found,
        "cache_size": len(STREET_NAME_CACHE)
    }



@app.get("/reports/active")
async def get_active_reports():
    """
    Return all currently active (non-expired) reports with accurate locations
    """
    import time
    
    current_time = time.time()
    active_reports = []
    
    for node_id, reports_list in RECENT_REPORTS.items():
        for report in reports_list:
            if report['expires_at'] > current_time:
                # Get node coordinates
                if node_id in NODE_TO_COORDS:
                    lat, lng = NODE_TO_COORDS[node_id]
                    
                    # Try to get accurate street name from graph edges
                    location_name = None
                    
                    # Check edges from this node for street names
                    if planner and planner.osm_graph:
                        neighbors = list(planner.osm_graph.neighbors(node_id))
                        
                        if neighbors:
                            # Get street name from first edge
                            try:
                                location_name = get_street_name_from_edge(
                                    planner.osm_graph, 
                                    node_id, 
                                    neighbors[0]
                                )
                            except Exception as e:
                                print(f"[Backend] Street name lookup failed for {node_id}: {e}")
                    
                    # Fallback to coordinate-based name if no street name found
                    if not location_name or is_placeholder_name(location_name):
                        location_name = get_coordinate_based_name(lat, lng)
                    
                    active_reports.append({
                        "id": f"{node_id}_{int(report['timestamp'])}",
                        "type": report['type'],
                        "lat": lat,
                        "lng": lng,
                        "confidence": report['confidence'],
                        "timestamp": int(report['timestamp'] * 1000),  # ms
                        "expires_at": int(report['expires_at'] * 1000),  # ms
                        "node_id": node_id,
                        "location_name": location_name
                    })
    
    # Sort by timestamp (newest first)
    active_reports.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {
        "reports": active_reports,
        "count": len(active_reports),
        "timestamp": int(current_time * 1000)
    }


@app.get("/reports/by-type")
async def get_reports_by_type(report_type: str = Query(...)):
    """
    Get active reports filtered by type
    """
    import time
    
    current_time = time.time()
    filtered_reports = []
    
    for node_id, reports_list in RECENT_REPORTS.items():
        for report in reports_list:
            if report['expires_at'] > current_time and report['type'] == report_type:
                if node_id in NODE_TO_COORDS:
                    lat, lng = NODE_TO_COORDS[node_id]
                    location_name = get_coordinate_based_name(lat, lng)
                    
                    filtered_reports.append({
                        "id": f"{node_id}_{int(report['timestamp'])}",
                        "type": report['type'],
                        "lat": lat,
                        "lng": lng,
                        "confidence": report['confidence'],
                        "timestamp": int(report['timestamp'] * 1000),
                        "expires_at": int(report['expires_at'] * 1000),
                        "node_id": node_id,
                        "location_name": location_name
                    })
    
    filtered_reports.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {
        "reports": filtered_reports,
        "type": report_type,
        "count": len(filtered_reports)
    }


@app.get("/reports/stats")
async def get_report_statistics():
    """
    Get statistics about current reports
    """
    import time
    from collections import Counter
    
    current_time = time.time()
    
    # Count active reports by type
    type_counts = Counter()
    total_active = 0
    
    for node_id, reports_list in RECENT_REPORTS.items():
        for report in reports_list:
            if report['expires_at'] > current_time:
                type_counts[report['type']] += 1
                total_active += 1
    
    return {
        "total_active": total_active,
        "by_type": dict(type_counts),
        "total_nodes_affected": len([
            node_id for node_id, reports in RECENT_REPORTS.items()
            if any(r['expires_at'] > current_time for r in reports)
        ])
    }


@app.delete("/reports/expired")
async def cleanup_expired_reports():
    """
    Admin endpoint to manually cleanup expired reports
    """
    import time
    
    current_time = time.time()
    removed_count = 0
    
    for node_id in list(RECENT_REPORTS.keys()):
        original_count = len(RECENT_REPORTS[node_id])
        RECENT_REPORTS[node_id] = [
            r for r in RECENT_REPORTS[node_id]
            if r['expires_at'] > current_time
        ]
        
        removed_count += original_count - len(RECENT_REPORTS[node_id])
        
        # Remove node if no reports left
        if not RECENT_REPORTS[node_id]:
            del RECENT_REPORTS[node_id]
    
    return {
        "status": "cleaned",
        "removed_count": removed_count,
        "remaining_nodes": len(RECENT_REPORTS)
    }


@app.get("/reports/heatmap")
async def get_report_heatmap():
    """
    Generate a heatmap of report density for visualization
    """
    import time
    from collections import defaultdict
    
    current_time = time.time()
    
    # Aggregate reports by grid cell
    grid_reports = defaultdict(lambda: {"count": 0, "max_confidence": 0.0, "types": []})
    
    for node_id, reports_list in RECENT_REPORTS.items():
        for report in reports_list:
            if report['expires_at'] > current_time and node_id in NODE_TO_COORDS:
                lat, lng = NODE_TO_COORDS[node_id]
                
                # Round to grid (~50m resolution)
                grid_key = f"{round(lat, 4)}_{round(lng, 4)}"
                
                grid_reports[grid_key]["count"] += 1
                grid_reports[grid_key]["max_confidence"] = max(
                    grid_reports[grid_key]["max_confidence"],
                    report['confidence']
                )
                if report['type'] not in grid_reports[grid_key]["types"]:
                    grid_reports[grid_key]["types"].append(report['type'])
                
                if "lat" not in grid_reports[grid_key]:
                    grid_reports[grid_key]["lat"] = lat
                    grid_reports[grid_key]["lng"] = lng
    
    # Convert to GeoJSON
    features = []
    for grid_key, data in grid_reports.items():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [data["lng"], data["lat"]]
            },
            "properties": {
                "report_count": data["count"],
                "max_confidence": data["max_confidence"],
                "types": data["types"],
                "intensity": "high" if data["count"] > 5 else "medium" if data["count"] > 2 else "low"
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

# endpoints to verify reports are working

@app.get("/debug/reports-impact")
async def debug_reports_impact(node_id: str = Query(...)):
    """
    Debug endpoint to verify that reports are affecting edge harm probabilities
    
    Usage: 
    1. Submit a report at a location
    2. Call this endpoint with the node_id
    3. Check that edge p_harm values have been updated
    """
    if planner is None:
        return {"error": "Planner not initialized"}
    
    if node_id not in planner.osm_graph.nodes:
        return {"error": f"Node {node_id} not found"}
    
    # Get all edges from this node
    edges_info = []
    for neighbor in planner.osm_graph.neighbors(node_id):
        if isinstance(planner.osm_graph, nx.MultiDiGraph):
            for key in planner.osm_graph[node_id][neighbor]:
                edge_data = planner.osm_graph[node_id][neighbor][key]
                edges_info.append({
                    "to_node": neighbor,
                    "edge_key": key,
                    "p_harm": float(edge_data.get('p_harm', 0.0)),
                    "length": float(edge_data.get('length', 0.0)),
                    "cost_at_lambda_10": float(
                        1.0 * edge_data.get('length', 0.0) + 
                        10.0 * (-np.log(1 - edge_data.get('p_harm', 0.01)))
                    )
                })
        else:
            edge_data = planner.osm_graph[node_id][neighbor]
            edges_info.append({
                "to_node": neighbor,
                "p_harm": float(edge_data.get('p_harm', 0.0)),
                "length": float(edge_data.get('length', 0.0)),
                "cost_at_lambda_10": float(
                    1.0 * edge_data.get('length', 0.0) + 
                    10.0 * (-np.log(1 - edge_data.get('p_harm', 0.01)))
                )
            })
    
    # Check if there are recent reports at this node
    node_reports = RECENT_REPORTS.get(node_id, [])
    active_reports = [
        r for r in node_reports 
        if r['expires_at'] > time.time()
    ]
    
    # Calculate statistics
    harm_values = [e['p_harm'] for e in edges_info]
    
    return {
        "node_id": node_id,
        "has_active_reports": len(active_reports) > 0,
        "active_reports": active_reports,
        "num_edges": len(edges_info),
        "edges": edges_info,
        "statistics": {
            "min_harm": float(min(harm_values)) if harm_values else 0.0,
            "max_harm": float(max(harm_values)) if harm_values else 0.0,
            "mean_harm": float(sum(harm_values) / len(harm_values)) if harm_values else 0.0
        },
        "interpretation": {
            "baseline_p_harm": "Original simulation values (before reports)",
            "current_p_harm": "Updated values (after reports)",
            "effect": "Higher p_harm → Higher cost → Route avoids this area"
        }
    }


@app.get("/debug/compare-before-after-report")
async def compare_routes_before_after_report(
    start: str = Query(...),
    goal: str = Query(...),
    report_node: str = Query(...),
    report_type: str = Query("tear_gas")
):
    """
    Simulate a report and compare routes before/after
    
    This demonstrates that reports DO affect routing!
    
    Usage:
    GET /debug/compare-before-after-report?start=NODE1&goal=NODE2&report_node=NODE3&report_type=tear_gas
    """
    if planner is None:
        return {"error": "Planner not initialized"}
    
    # Step 1: Compute route BEFORE simulated report
    original_lambda = planner.config.get('lambda_risk', 10.0)
    planner.config['lambda_risk'] = 10.0
    
    from src.planner.cost_functions import get_cost_function
    cost_type = planner.config.get('cost_function', 'log_odds')
    planner.cost_fn = get_cost_function(cost_type, planner.config)
    planner.optimizer.cost_fn = planner.cost_fn
    
    route_before = planner.plan_route(start, goal, 'astar')
    
    # Step 2: Simulate a report at report_node
    print(f"\n[Debug] Simulating {report_type} report at node {report_node}")
    update_edge_harm_from_reports(report_node, report_type, confidence=0.9)
    
    # Step 3: Compute route AFTER simulated report
    route_after = planner.plan_route(start, goal, 'astar')
    
    # Restore config
    planner.config['lambda_risk'] = original_lambda
    
    return {
        "experiment": "Report Impact Analysis",
        "setup": {
            "start_node": start,
            "goal_node": goal,
            "report_at_node": report_node,
            "report_type": report_type,
            "confidence": 0.9
        },
        "before_report": {
            "path": route_before.get('path', []),
            "num_nodes": len(route_before.get('path', [])),
            "distance_m": route_before.get('metadata', {}).get('total_distance_m', 0),
            "max_edge_risk": route_before.get('metadata', {}).get('max_edge_risk', 0),
            "mean_edge_risk": route_before.get('metadata', {}).get('mean_edge_risk', 0),
            "passes_through_report_node": report_node in route_before.get('path', [])
        },
        "after_report": {
            "path": route_after.get('path', []),
            "num_nodes": len(route_after.get('path', [])),
            "distance_m": route_after.get('metadata', {}).get('total_distance_m', 0),
            "max_edge_risk": route_after.get('metadata', {}).get('max_edge_risk', 0),
            "mean_edge_risk": route_after.get('metadata', {}).get('mean_edge_risk', 0),
            "passes_through_report_node": report_node in route_after.get('path', [])
        },
        "analysis": {
            "route_changed": route_before.get('path') != route_after.get('path'),
            "now_avoids_report": (
                report_node in route_before.get('path', []) and
                report_node not in route_after.get('path', [])
            ),
            "distance_change_m": (
                route_after.get('metadata', {}).get('total_distance_m', 0) -
                route_before.get('metadata', {}).get('total_distance_m', 0)
            ),
            "risk_change": (
                route_after.get('metadata', {}).get('max_edge_risk', 0) -
                route_before.get('metadata', {}).get('max_edge_risk', 0)
            )
        },
        "conclusion": "If route_changed=true and now_avoids_report=true, reports ARE affecting routing!"
    }

# temporary route to test coordinates
@app.get("/test-coords")
async def test_coordinates():
    """Test coordinate conversion for a few points in a 200x200 grid"""
    test_points = [
        (0, 0),    # Top-left
        (100, 100),  # Center
        (199, 199),  # Bottom-right
    ]
    
    results = []
    for i, j in test_points:
        lat, lng = grid_to_lat_lng(i, j)
        results.append({
            "grid": (i, j),
            "latlng": (lat, lng),
            "coordinates": [lng, lat]
        })
    
    return results

# Global report storage (in-memory for demo)
RECENT_REPORTS: Dict[str, List[dict]] = {}  # cell_id -> list of reports
REPORT_EXPIRY_SECONDS = 600  # 10 minutes


def update_edge_harm_from_reports(node_id: str, report_type: str, confidence: float):
    """
    Update edge harm probabilities based on new report
    """
    if planner is None:
        return
    
    # Map report types to harm multipliers
    harm_multipliers = {
        'safe': 0.5,       # Reduces harm
        'crowd': 1.2,      # Slight increase
        'police': 1.5,     # Moderate increase
        'tear_gas': 2.0,   # High increase
        'water_cannon': 2.0
    }
    
    multiplier = harm_multipliers.get(report_type, 1.0)
    
    # Update all edges connected to this node
    for neighbor in planner.osm_graph.neighbors(node_id):
        if isinstance(planner.osm_graph, nx.MultiDiGraph):
            for key in planner.osm_graph[node_id][neighbor]:
                edge_data = planner.osm_graph[node_id][neighbor][key]
                current_harm = edge_data.get('p_harm', 0.0)
                # Weighted update based on confidence
                new_harm = min(1.0, current_harm + (multiplier - 1.0) * confidence)
                edge_data['p_harm'] = new_harm
        else:
            edge_data = planner.osm_graph[node_id][neighbor]
            current_harm = edge_data.get('p_harm', 0.0)
            new_harm = min(1.0, current_harm + (multiplier - 1.0) * confidence)
            edge_data['p_harm'] = new_harm
    
    print(f"[Backend] Updated harm around node {node_id} with multiplier {multiplier}")


@app.post("/user/delete-data")
async def delete_user_data(session_id: str = Query(...)):
    """
    Delete all data associated with a user session
    """
    # In production, you'd have session-based storage
    # For now, just clear all recent reports
    global RECENT_REPORTS
    RECENT_REPORTS.clear()
    
    return {"status": "deleted", "message": "All data cleared"}


@app.get("/alerts")
async def get_alerts():
    """Get recent alerts/events"""
    # Mock data matching your frontend
    return [
        {"id": "1", "type": "Tear gas reported", "location": "Tom Mboya St", "time": "2 min ago", "severity": "high"},
        {"id": "2", "type": "Heavy crowd", "location": "University Way", "time": "8 min ago", "severity": "medium"},
        {"id": "3", "type": "Police presence", "location": "Kenyatta Ave", "time": "15 min ago", "severity": "medium"}
    ]

@app.get("/health")
def health_check():
    return {"status": "ok", "planner_loaded": planner is not None}


@app.get("/config")
def get_config():
    """Return current planner configuration."""
    return planner_config


# backend/main.py - FIXED /route endpoint with dynamic risk weighting

@app.get("/route")
def get_route(
    start: str = Query(..., description="Start node ID"),
    goal: str = Query(..., description="Goal node ID"),
    algorithm: str = Query("astar", description="astar or dijkstra"),
    lambda_risk: float = Query(10.0, description="Risk weight (1.0=distance, 20.0=safety)")
):
    """
    Compute a risk-aware route with dynamic risk weighting.
    
    Algorithm selection:
    - astar: Fast heuristic search (recommended for balanced/shortest routes)
    - dijkstra: Optimal search (recommended for safest routes)
    
    Lambda_risk interpretation:
    - λ = 1.0:  Shortest path (minimal risk penalty)
    - λ = 10.0: Balanced (default, moderate risk penalty)
    - λ = 20.0: Safest path (maximum risk aversion)
    
    Cost formula: Cost(edge) = λ_distance × distance + λ_risk × (-log(1 - p_harm))
    """
    if planner is None:
        return JSONResponse({"error": "Planner not initialized"}, status_code=503)

    # Validate nodes exist in graph
    if start not in planner.osm_graph.nodes:
        return JSONResponse(
            {
                "error": f"Start node '{start}' not found in road network",
                "suggestion": "Use /nearest-node endpoint to find valid nodes"
            },
            status_code=400
        )
    
    if goal not in planner.osm_graph.nodes:
        return JSONResponse(
            {
                "error": f"Goal node '{goal}' not found in road network",
                "suggestion": "Use /nearest-node endpoint to find valid nodes"
            },
            status_code=400
        )

    # Validate algorithm choice
    if algorithm not in ['astar', 'dijkstra']:
        return JSONResponse(
            {"error": f"Invalid algorithm '{algorithm}'. Must be 'astar' or 'dijkstra'"},
            status_code=400
        )

    # Validate lambda_risk range
    if lambda_risk < 0.1 or lambda_risk > 100.0:
        return JSONResponse(
            {"error": f"lambda_risk must be in range [0.1, 100.0], got {lambda_risk}"},
            status_code=400
        )

    print(f"\n{'='*60}")
    print(f"[Backend] NEW ROUTE REQUEST")
    print(f"  Start node: {start}")
    print(f"  Goal node:  {goal}")
    print(f"  Algorithm:  {algorithm}")
    print(f"  λ_risk:     {lambda_risk}")
    print(f"{'='*60}\n")

    # CRITICAL: Temporarily override planner config with user preferences
    original_lambda = planner.config.get('lambda_risk', 10.0)
    planner.config['lambda_risk'] = lambda_risk
    
    # Recreate cost function with new lambda_risk
    from src.planner.cost_functions import get_cost_function
    cost_type = planner.config.get('cost_function', 'log_odds')
    planner.cost_fn = get_cost_function(cost_type, planner.config)
    
    # Update optimizer's cost function reference
    planner.optimizer.cost_fn = planner.cost_fn
    
    try:
        # Compute route using selected algorithm
        result = planner.plan_route(start, goal, algorithm)
        
        # Check for routing errors
        if "error" in result:
            print(f"[Backend] Route planning failed: {result['error']}")
            return JSONResponse(result, status_code=404)
        
        # Convert UTM geometry to lat/lng for Mapbox
        if "geometry" in result and result["geometry"]:
            result["geometry_latlng"] = convert_route_geometry(result["geometry"])
            print(f"[Backend] ✓ Converted {len(result['geometry'])} waypoints to lat/lng")
        
        # Generate turn-by-turn directions with street names
        if "path" in result and "geometry" in result:
            result["directions"] = generate_turn_by_turn_directions(
                planner.osm_graph,
                result["path"],
                result["geometry"]
            )
            
            # Add lat/lng coordinates to each direction step
            for step in result["directions"]:
                node_id = step["node"]
                if node_id in planner.osm_graph.nodes:
                    node_data = planner.osm_graph.nodes[node_id]
                    x_utm = float(node_data.get('x', 0))
                    y_utm = float(node_data.get('y', 0))
                    lat, lng = utm_to_latlng(x_utm, y_utm)
                    step["lat"] = lat
                    step["lng"] = lng
            
            print(f"[Backend] ✓ Generated {len(result['directions'])} turn-by-turn directions")
        
        # NEW: Add Fusion Metadata
        stats = aggregator.get_report_statistics(RECENT_REPORTS)
        
        # Check how many edges on this route were affected by reports
        route_path = result.get('path', [])
        affected_edges_on_route = 0
        
        for i in range(len(route_path) - 1):
            u, v = route_path[i], route_path[i+1]
            
            if len(baseline_p_sim) > 0:
                # Check if this edge differs from baseline
                if isinstance(planner.osm_graph, nx.MultiDiGraph):
                    for key in planner.osm_graph[u][v]:
                        baseline = baseline_p_sim.get((u, v, key), 0.0)
                        current = planner.osm_graph[u][v][key].get('p_harm', 0.0)
                        if abs(current - baseline) > 0.001:
                            affected_edges_on_route += 1
                            break
                else:
                    baseline = baseline_p_sim.get((u, v), 0.0)
                    current = planner.osm_graph[u][v].get('p_harm', 0.0)
                    if abs(current - baseline) > 0.001:
                        affected_edges_on_route += 1
        
        result["fusion_metadata"] = {
            "reports": {
                "total_active": stats['total_active_reports'],
                "nodes_affected": stats['nodes_affected'],
                "by_type": stats['by_type']
            },
            "route_analysis": {
                "total_edges": len(route_path) - 1,
                "edges_influenced_by_reports": affected_edges_on_route,
                "percent_influenced": round(
                    100 * affected_edges_on_route / max(len(route_path) - 1, 1), 
                    1
                )
            },
            "fusion_config": {
                "simulation_weight": fusion_engine.sim_weight,
                "report_weight": fusion_engine.report_weight,
                "method": "uncertainty_weighted_bayesian"
            },
            "interpretation": (
                f"This route was influenced by {stats['total_active_reports']} "
                f"active reports affecting {affected_edges_on_route} edges"
            )
        }
        
        # Add routing params
        result["routing_params"] = {
            "algorithm": algorithm,
            "lambda_risk": lambda_risk,
            "lambda_distance": planner.config.get('lambda_distance', 1.0),
            "cost_function": cost_type
        }
        
        print(f"\n[Backend] ✓ ROUTE COMPUTED")
        print(f"  Distance: {result.get('metadata', {}).get('total_distance_m', 0):.1f}m")
        print(f"  Safety: {result.get('safety_score', 0)*100:.1f}%")
        print(f"  Reports influencing route: {affected_edges_on_route} edges")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"\n[Backend] ROUTE ERROR: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            {"error": f"Route computation failed: {str(e)}"},
            status_code=500
        )
    
    finally:
        # Restore original config
        planner.config['lambda_risk'] = original_lambda
        planner.cost_fn = get_cost_function(cost_type, planner.config)
        planner.optimizer.cost_fn = planner.cost_fn


@app.get("/route/validate")
def validate_route_params(
    start: str = Query(...),
    goal: str = Query(...),
    lambda_risk: float = Query(10.0)
):
    """
    Validate route parameters before computing.
    Useful for frontend to pre-check inputs.
    """
    if planner is None:
        return {"valid": False, "error": "Planner not initialized"}
    
    errors = []
    warnings = []
    
    # Check start node
    if start not in planner.osm_graph.nodes:
        errors.append(f"Start node '{start}' not in road network")
    
    # Check goal node
    if goal not in planner.osm_graph.nodes:
        errors.append(f"Goal node '{goal}' not in road network")
    
    # Check lambda_risk range
    if lambda_risk < 0.1 or lambda_risk > 100.0:
        errors.append(f"lambda_risk must be in [0.1, 100.0], got {lambda_risk}")
    
    # Check if path exists
    if not errors:
        try:
            # Quick networkx check (ignores risk)
            path = nx.shortest_path(planner.osm_graph, start, goal)
            
            # Estimate distance
            total_distance = sum(
                planner.osm_graph[path[i]][path[i+1]].get('length', 0)
                for i in range(len(path)-1)
            )
            
            # Estimate risk
            edge_risks = [
                planner.osm_graph[path[i]][path[i+1]].get('p_harm', 0)
                for i in range(len(path)-1)
            ]
            mean_risk = sum(edge_risks) / len(edge_risks) if edge_risks else 0
            max_risk = max(edge_risks) if edge_risks else 0
            
            # Generate warnings
            if max_risk > 0.5:
                warnings.append(f"High risk detected on route (max: {max_risk:.2f})")
            
            if total_distance > 5000:
                warnings.append(f"Long route ({total_distance/1000:.1f} km)")
            
            return {
                "valid": True,
                "warnings": warnings,
                "estimate": {
                    "hops": len(path),
                    "distance_m": round(total_distance, 1),
                    "mean_risk": round(mean_risk, 3),
                    "max_risk": round(max_risk, 3)
                }
            }
            
        except nx.NetworkXNoPath:
            errors.append("No path exists between nodes")
        except Exception as e:
            errors.append(f"Path validation failed: {str(e)}")
    
    return {"valid": False, "errors": errors}


@app.get("/route/compare")
def compare_route_options(
    start: str = Query(...),
    goal: str = Query(...)
):
    """
    Compare all three route types side-by-side:
    1. Safest (λ=20, Dijkstra)
    2. Balanced (λ=10, A*)
    3. Shortest (λ=1, A*)
    
    Returns summary metrics without full geometry.
    Useful for showing users their options.
    """
    if planner is None:
        return JSONResponse({"error": "Planner not initialized"}, status_code=503)
    
    route_types = {
        "safest": {"lambda_risk": 20.0, "algorithm": "dijkstra"},
        "balanced": {"lambda_risk": 10.0, "algorithm": "astar"},
        "shortest": {"lambda_risk": 1.0, "algorithm": "astar"}
    }
    
    comparisons = {}
    original_lambda = planner.config.get('lambda_risk')
    
    from src.planner.cost_functions import get_cost_function
    cost_type = planner.config.get('cost_function', 'log_odds')
    
    try:
        for route_type, params in route_types.items():
            # Set config for this route type
            planner.config['lambda_risk'] = params['lambda_risk']
            planner.cost_fn = get_cost_function(cost_type, planner.config)
            planner.optimizer.cost_fn = planner.cost_fn
            
            result = planner.plan_route(start, goal, params['algorithm'])
            
            if "error" not in result:
                comparisons[route_type] = {
                    "distance_m": round(result['metadata']['total_distance_m'], 1),
                    "distance_km": round(result['metadata']['total_distance_m'] / 1000, 2),
                    "eta_min": round(result['metadata']['estimated_time_s'] / 60),
                    "safety_score": round(result['safety_score'] * 100, 1),
                    "num_turns": result['metadata']['num_turns'],
                    "max_edge_risk": round(result['metadata']['max_edge_risk'], 3),
                    "mean_edge_risk": round(result['metadata']['mean_edge_risk'], 3),
                    "nodes_explored": result['metadata'].get('nodes_explored', 0)
                }
            else:
                comparisons[route_type] = {"error": result['error']}
        
        # Determine recommendation
        if "safest" in comparisons and "shortest" in comparisons:
            safest_dist = comparisons["safest"].get("distance_m", float('inf'))
            shortest_dist = comparisons["shortest"].get("distance_m", float('inf'))
            
            # Recommend safest if distance penalty < 50%
            if safest_dist < shortest_dist * 1.5:
                recommendation = "safest"
            else:
                recommendation = "balanced"
        else:
            recommendation = "balanced"
        
        return {
            "routes": comparisons,
            "recommendation": recommendation
        }
        
    finally:
        # Restore original config
        planner.config['lambda_risk'] = original_lambda
        planner.cost_fn = get_cost_function(cost_type, planner.config)
        planner.optimizer.cost_fn = planner.cost_fn


@app.get("/debug/edge-data")
def debug_edge_data(node1: str = Query(...), node2: str = Query(...)):
    """Debug endpoint to see what's in an OSM edge"""
    if planner is None:
        return {"error": "Planner not initialized"}
    
    if not planner.osm_graph.has_edge(node1, node2):
        return {"error": f"No edge between {node1} and {node2}"}
    
    edge_data = dict(planner.osm_graph[node1][node2])
    
    # Also check if there are multiple edges (MultiGraph)
    all_edges = []
    if isinstance(planner.osm_graph, nx.MultiGraph):
        for key in planner.osm_graph[node1][node2]:
            all_edges.append({
                "key": key,
                "data": dict(planner.osm_graph[node1][node2][key])
            })
    
    return {
        "edge_data": edge_data,
        "all_edges": all_edges if all_edges else None,
        "graph_type": type(planner.osm_graph).__name__
    }

@app.get("/compare")
def compare_routes(
    start: str = Query(..., description="Start node ID"),
    goal: str = Query(..., description="Goal node ID")
):
    """Compare baseline (shortest) vs risk-aware route."""
    if planner is None:
        return JSONResponse({"error": "Planner not initialized"}, status_code=503)

    result = planner.compare_routes(start, goal)
    return result


# 4. Run server
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
