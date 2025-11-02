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
from affine import Affine
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import your planner
from src.planner.route_planner import RiskAwareRoutePlanner



# 1. Initialize FastAPI app
app = FastAPI(
    title="Protest Safety Planner API",
    description="Backend for risk-aware route computation and comparison",
    version="1.0.0"
)

# Add CORS middleware after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load configuration and planner on startup
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "planner_config.yaml"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "rollouts" / "production_run"

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

def _grid_to_lat_lng_fallback(i: int, j: int) -> tuple[float, float]:
    """Fallback conversion using bounds from metadata"""
    try:
        metadata_path = DATA_DIR / "real_nairobi_cbd_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get grid dimensions
        height, width = metadata.get("grid_size", [100, 100])
        
        # Get UTM bounds
        xmin, ymin, xmax, ymax = metadata.get("bounds", [257024.35880730546, 9857598.638626937, 257486.7266486904, 9858116.289497813])
        
        # Convert current cell to UTM
        cell_width = (xmax - xmin) / width
        cell_height = (ymax - ymin) / height
        
        x_utm = xmin + (j + 0.5) * cell_width
        y_utm = ymax - (i + 0.5) * cell_height  # Note: y decreases as i increases
        
        # Approximate conversion (rough but works for demo)
        # UTM to lat/lng approximate conversion for Nairobi area
        lat = -1.2833 + (y_utm - 9857857.0) / 111000  # Rough conversion
        lng = 36.8172 + (x_utm - 257255.0) / (111000 * 0.8)  # Account for longitude compression
        
        return lat, lng
        
    except Exception as e:
        print(f"Fallback conversion also failed: {e}")
        # Final fallback - centered on Nairobi CBD
        return -1.2833, 36.8172

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

    print("[Backend] Planner initialized successfully.")


# 3. API Endpoints

@app.get("/riskmap")
async def get_risk_map():
    """Serve risk data for map overlay - optimized for 100x100 grid"""
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
            bounds_latlng = [36.805, -1.292, 36.830, -1.275]  # Fallback
        
        return {
            "type": "raster",
            "bounds": bounds_latlng,
            "data": p_sim.tolist(),  # 2D array of risk values
            "risk_range": [float(p_sim.min()), float(p_sim.max())]
        }
        
    except Exception as e:
        print(f"Raster risk map error: {e}")
        return JSONResponse({"error": f"Failed to load raster risk map: {str(e)}"}, status_code=500)

# Add this temporary route to test coordinates
@app.get("/test-coords")
async def test_coordinates():
    """Test coordinate conversion for a few points"""
    test_points = [
        (0, 0),    # Top-left
        (50, 50),  # Center
        (99, 99),  # Bottom-right
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

@app.post("/report")
async def submit_report(report_data: dict):
    """Mock report submission for demo"""
    print(f"Received report: {report_data}")
    # In real implementation, this would go to your report adapter
    return {"status": "received", "id": "mock_report_123"}

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
    algorithm: str = Query("astar", description="Routing algorithm (astar or dijkstra)")
):
    """Compute a risk-aware route."""
    if planner is None:
        return JSONResponse({"error": "Planner not initialized"}, status_code=503)

    result = planner.plan_route(start, goal, algorithm)
    return result


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
