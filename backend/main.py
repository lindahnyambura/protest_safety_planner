"""
main.py - FastAPI backend integrating RiskAwareRoutePlanner
"""

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pathlib import Path
import yaml
import numpy as np
import networkx as nx
import json
import pyproj
import time
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
    import time
    
    try:
        # Validate report type
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
        
        # Find nearest node using KDTree
        if NODE_KDTREE is None or NODE_IDS is None:
            return JSONResponse(
                {"error": "Graph index not initialized"},
                status_code=503
            )
        
        # Convert lat/lng to UTM for nearest node search
        import pyproj
        transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32737", always_xy=True)
        x_utm, y_utm = transformer_to_utm.transform(report.lng, report.lat)
        
        # Find nearest node
        distance, index = NODE_KDTREE.query([x_utm, y_utm])
        nearest_node_id = str(NODE_IDS[index])
        
        print(f"[Backend] Snapped to node {nearest_node_id} (distance: {distance:.1f}m)")
        
        # Create report record
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
        
        # Update edge harm probabilities
        update_edge_harm_from_reports(nearest_node_id, report.type, report.confidence)
        
        print(f"[Backend] Report stored. Total active reports: {sum(len(r) for r in RECENT_REPORTS.values())}")
        
        return {
            "status": "ok",
            "report_id": f"r_{int(timestamp)}",
            "node_id": nearest_node_id,
            "snapped_distance_m": float(distance),
            "expires_in_seconds": REPORT_EXPIRY_SECONDS
        }
        
    except Exception as e:
        import traceback
        print(f"[Backend] Report submission error: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            {"error": f"Internal server error: {str(e)}"},
            status_code=500
        )


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
    
    # Results from your script - update these with actual output
    LANDMARK_TO_NODE = {
        # "uhuru park": "12343642875",
        # "jamia mosque": "6580961457",
        # "city market": "9859577513",
        "kencom": "12343534285",
        "bus station": "10873342299",
        "odeon": "12361156623",
        # "afya center": "10873342295",
        "national archives": "12414258058",
        # "kicc": "13134429074",
        "gpo": "12361445752",
        # "teleposta towers": "5555073936",
        # "times tower": "10701041875",
    }
    
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
    global planner, planner_config

    # Load config
    with open(CONFIG_PATH, "r") as f:
        planner_config = yaml.safe_load(f)

    # Load OSM graph
    graph_path = DATA_DIR / "nairobi_walk.graphml"
    osm_graph = nx.read_graphml(graph_path)
    
    # FIX: Convert node coordinates from strings to floats
    for node, data in osm_graph.nodes(data=True):
        if 'x' in data:
            data['x'] = float(data['x'])
        if 'y' in data:
            data['y'] = float(data['y'])
    
    print(f"[Backend] Loaded OSM graph with {len(osm_graph.nodes)} nodes.")

    # Load harm probabilities
    p_sim_path = ARTIFACTS_DIR / "p_sim.npy"
    p_sim = np.load(p_sim_path)
    print(f"[Backend] Loaded harm probability grid: {p_sim.shape}")

    # Optional mapping
    cell_to_node_path = DATA_DIR / "cell_to_node.npy"
    cell_to_node = np.load(cell_to_node_path, allow_pickle=True)

    # Initialize planner
    planner = RiskAwareRoutePlanner(
        osm_graph=osm_graph,
        p_sim=p_sim,
        config=planner_config,
        cell_to_node=cell_to_node
    )

    # Build coordinate cache
    build_node_coordinate_cache(planner.osm_graph)
    
    # Load landmark mappings
    load_landmark_mappings()

    # Build KDTree for nearest node lookup
    build_node_kdtree(planner.osm_graph)

    # Load street name cache
    load_street_name_cache()

    print("[Backend] Planner initialized successfully.")

@app.on_event("startup")
async def start_report_expiry_task():
    import asyncio
    
    async def expire_old_reports():
        while True:
            await asyncio.sleep(60)  # Check every minute
            current_time = time.time()
            
            for node_id in list(RECENT_REPORTS.keys()):
                RECENT_REPORTS[node_id] = [
                    r for r in RECENT_REPORTS[node_id]
                    if r['expires_at'] > current_time
                ]
                
                if not RECENT_REPORTS[node_id]:
                    del RECENT_REPORTS[node_id]
    
    asyncio.create_task(expire_old_reports())

# 3. API Endpoints

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
def get_landmarks():
    """Return all known landmarks with their coordinates"""
    landmarks = []
    for name, node_id in LANDMARK_TO_NODE.items():
        if node_id in NODE_TO_COORDS:
            lat, lng = NODE_TO_COORDS[node_id]
            landmarks.append({
                "name": name.title(),
                "node_id": node_id,
                "coordinates": {"lat": lat, "lng": lng}
            })
    return {"landmarks": landmarks}

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
    Return all currently active (non-expired) reports
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
                    
                    # Try to get location name
                    location_name = get_coordinate_based_name(lat, lng)
                    
                    active_reports.append({
                        "id": f"{node_id}_{int(report['timestamp'])}",
                        "type": report['type'],
                        "lat": lat,
                        "lng": lng,
                        "confidence": report['confidence'],
                        "timestamp": int(report['timestamp'] * 1000),  # Convert to ms
                        "expires_at": int(report['expires_at'] * 1000),  # Convert to ms
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

# Add this temporary route to test coordinates
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


@app.get("/route")
def get_route(
    start: str = Query(..., description="Start node ID"),
    goal: str = Query(..., description="Goal node ID"),
    algorithm: str = Query("astar", description="astar or dijkstra"),
    lambda_risk: float = Query(10.0, description="Risk weight (1.0=distance, 10.0=safety)")
):
    """Compute a risk-aware route with street names"""
    if planner is None:
        return JSONResponse({"error": "Planner not initialized"}, status_code=503)

    # Temporarily override planner config
    original_lambda = planner.config.get('lambda_risk', 10.0)
    planner.config['lambda_risk'] = lambda_risk
    
    result = planner.plan_route(start, goal, algorithm)

    # Restore original config
    planner.config['lambda_risk'] = original_lambda
    
    # Convert UTM geometry to lat/lng
    if "geometry" in result:
        result["geometry_latlng"] = convert_route_geometry(result["geometry"])
    
    # Generate better directions with street names
    if "path" in result and "geometry" in result:
        result["directions"] = generate_turn_by_turn_directions(
            planner.osm_graph,
            result["path"],
            result["geometry"]
        )
        
        # Add lat/lng to each direction step
        for step in result["directions"]:
            node_id = step["node"]
            if node_id in planner.osm_graph.nodes:
                x_utm = planner.osm_graph.nodes[node_id].get('x')
                y_utm = planner.osm_graph.nodes[node_id].get('y')
                if x_utm and y_utm:
                    lat, lng = utm_to_latlng(float(x_utm), float(y_utm))
                    step["lat"] = lat
                    step["lng"] = lng
    
    return result


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
