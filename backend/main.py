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

# Import your planner
from src.planner.route_planner import RiskAwareRoutePlanner


# -------------------------------------------------------------------
# 1. Initialize FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Protest Safety Planner API",
    description="Backend for risk-aware route computation and comparison",
    version="1.0.0"
)


# -------------------------------------------------------------------
# 2. Load configuration and planner on startup
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# 3. API Endpoints
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# 4. Run server
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
