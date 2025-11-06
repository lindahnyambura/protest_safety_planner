"""
route_planner.py - FR6: Risk-Aware Route Planner

Computes real-road routes that minimize expected harm while maintaining 
reasonable travel distance and practicality.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from collections import deque
import json
from shapely.geometry import mapping, LineString

from .cost_functions import get_cost_function, CostFunction
from .route_optimizer import RouteOptimizer


class RiskAwareRoutePlanner:
    """
    Risk-aware route planner for protest navigation.
    
    Integrates Monte Carlo harm probabilities with OSM road network to
    produce safe, practical routes with uncertainty quantification.
    """
    
    def __init__(self, 
                 osm_graph: nx.Graph,
                 p_sim: np.ndarray,
                 config: Dict,
                 cell_to_node: Optional[np.ndarray] = None,
                 street_names: Optional[Dict] = None):
        """
        Initialize route planner.
        
        Args:
            osm_graph: NetworkX graph with node coords and edge lengths
            p_sim: Harm probability grid from Monte Carlo (height, width)
            config: Planner configuration
            cell_to_node: Optional grid-to-node mapping array
            street_names: Optional dict mapping node_id → street name
        """
        self.graph = osm_graph
        self.p_sim = p_sim
        self.config = config
        self.cell_to_node = cell_to_node
        self.street_names = street_names or {}
        
        # Grid dimensions
        self.height, self.width = p_sim.shape
        
        # Initialize cost function
        cost_type = config.get('cost_function', 'log_odds')
        self.cost_fn = get_cost_function(cost_type, config)
        
        # Initialize optimizer
        self.optimizer = RouteOptimizer(osm_graph)
        
        # Map harm probabilities to graph edges
        self._map_risk_to_edges()
        
        # Route cache (stores last N routes)
        self.cache_size = config.get('cache_size', 10)
        self.route_cache = deque(maxlen=self.cache_size)
        
        print(f"[Planner] Initialized with {len(self.graph.nodes)} nodes, "
              f"{len(self.graph.edges)} edges")
        print(f"[Planner] Cost function: {cost_type}")
        print(f"[Planner] Mean p(harm): {p_sim.mean():.4f}, "
              f"Max p(harm): {p_sim.max():.4f}")
    
    def _map_risk_to_edges(self):
        """
        Map grid-based harm probabilities to graph edges.
        
        Strategy:
        1. For each edge (u, v), sample p_sim along the edge
        2. Take max or mean of sampled probabilities
        3. Store as edge attribute 'p_harm'
        """
        # Get node positions
        node_positions = {
            node: (data['x'], data['y']) 
            for node, data in self.graph.nodes(data=True)
        }
        
        # Process each edge
        for u, v, data in self.graph.edges(data=True):
            # Get endpoint coordinates (UTM)
            x1, y1 = node_positions[u]
            x2, y2 = node_positions[v]
            
            # Sample p_sim along edge (use 5 samples)
            p_harms = []
            for t in np.linspace(0, 1, 5):
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                
                # Convert UTM to grid cell
                cell_x, cell_y = self._utm_to_cell(x, y)
                
                # Bounds check
                if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                    p_harms.append(self.p_sim[cell_y, cell_x])
            
            # Take max (conservative) or mean
            if p_harms:
                strategy = self.config.get('edge_risk_strategy', 'max')
                if strategy == 'max':
                    p_harm = np.max(p_harms)
                else:
                    p_harm = np.mean(p_harms)
            else:
                p_harm = 0.0  # Edge outside grid
            
            # Store on edge
            data['p_harm'] = float(p_harm)
    
    def _utm_to_cell(self, x_utm: float, y_utm: float) -> Tuple[int, int]:
        """
        Convert UTM coordinates to grid cell indices.
        
        Assumes affine transformation is available in environment.
        For now, use simple linear interpolation from node bounds.
        
        Args:
            x_utm, y_utm: UTM coordinates
            
        Returns:
            cell_x, cell_y: Grid cell indices
        """
        # Get graph bounds
        xs = [data['x'] for _, data in self.graph.nodes(data=True)]
        ys = [data['y'] for _, data in self.graph.nodes(data=True)]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Normalize to [0, 1]
        x_norm = (x_utm - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        y_norm = (y_utm - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        
        # Scale to grid
        cell_x = int(x_norm * self.width)
        cell_y = int((1.0 - y_norm) * self.height)  # Flip Y (image coords)
        
        return cell_x, cell_y
    
    def plan_route(self, 
                   start: str, 
                   goal: str, 
                   algorithm: str = 'astar') -> Dict:
        """
        Compute risk-aware route from start to goal.
        
        Args:
            start: Start node ID (string)
            goal: Goal node ID (string)
            algorithm: 'dijkstra' or 'astar' (default: 'astar')
            
        Returns:
            route_dict: Dictionary with:
                - path: List of node IDs
                - geometry: List of (x, y) coordinates
                - directions: List of textual instructions
                - safety_score: P(safe arrival)
                - metadata: Path statistics
        """
        # Validate inputs
        if start not in self.graph:
            return {'error': f'Start node {start} not in graph'}
        if goal not in self.graph:
            return {'error': f'Goal node {goal} not in graph'}
        
        # Check cache
        cache_key = (start, goal, algorithm)
        for cached in self.route_cache:
            if cached['cache_key'] == cache_key:
                print(f"[Planner] Cache hit for {start} → {goal}")
                return cached['route']
        
        # Run path finding
        if algorithm == 'dijkstra':
            path, cost, metadata = self.optimizer.dijkstra(start, goal, self.cost_fn)
        elif algorithm == 'astar':
            path, cost, metadata = self.optimizer.astar(start, goal, self.cost_fn)
        else:
            return {'error': f'Unknown algorithm: {algorithm}'}
        
        if not path:
            return {'error': 'No path found', 'metadata': metadata}
        
        # Generate geometry
        geometry = self._path_to_geometry(path)
        
        # Generate directions
        directions = self._generate_directions(path)
        
        # Assemble result
        route_dict = {
            'path': path,
            'geometry': geometry,
            'directions': directions,
            'safety_score': metadata.get('p_safe', 0.0),
            'metadata': metadata,
            'algorithm': algorithm
        }
        
        # Cache result
        self.route_cache.append({
            'cache_key': cache_key,
            'route': route_dict
        })
        
        return route_dict
    
    def _path_to_geometry(self, path: List[str]) -> List[Tuple[float, float]]:
        """Convert path to list of (x, y) UTM coordinates."""
        geometry = []
        for node in path:
            data = self.graph.nodes[node]
            geometry.append((data['x'], data['y']))
        return geometry
    
    def _generate_directions(self, path: List[str]) -> List[Dict]:
        """
        Generate turn-by-turn directions.
        
        Args:
            path: List of node IDs
            
        Returns:
            directions: List of instruction dicts
        """
        if len(path) < 2:
            return []
        
        directions = []
        
        # Start instruction
        start_street = self.street_names.get(path[0], "your location")
        directions.append({
            'step': 0,
            'instruction': f"Start at {start_street}",
            'node': path[0],
            'distance_m': 0.0
        })
        
        # Middle segments
        for i in range(1, len(path) - 1):
            u, v = path[i-1], path[i]
            edge_data = self.graph[u][v]
            
            street = self.street_names.get(v, "unnamed road")
            distance = edge_data.get('length', 0.0)
            
            # Simple direction (TODO: compute actual bearing)
            directions.append({
                'step': i,
                'instruction': f"Continue on {street}",
                'node': v,
                'distance_m': float(distance)
            })
        
        # End instruction
        goal_street = self.street_names.get(path[-1], "destination")
        u, v = path[-2], path[-1]
        final_distance = self.graph[u][v].get('length', 0.0)
        
        directions.append({
            'step': len(path) - 1,
            'instruction': f"Arrive at {goal_street}",
            'node': path[-1],
            'distance_m': float(final_distance)
        })
        
        return directions
    
    def compare_routes(self, start: str, goal: str) -> Dict:
        """
        Compare baseline shortest path vs risk-aware route.

        Args:
            start, goal: Node IDs

        Returns:
            comparison: Dict with both routes and metrics
        """
        # Baseline: shortest path (ignore risk)
        baseline_config = self.config.copy()
        baseline_config['lambda_risk'] = 0.0  # Zero risk weight
        baseline_cost_fn = get_cost_function('linear', baseline_config)

        baseline_path, _, baseline_meta = self.optimizer.dijkstra(start, goal, baseline_cost_fn)
        risk_path, _, risk_meta = self.optimizer.astar(start, goal, self.cost_fn)

        # --- Validate route outputs ---
        if not baseline_path or not risk_path:
            return {
                'error': 'Failed to compute one or both routes',
                'baseline': baseline_meta,
                'risk_aware': risk_meta,
            }

        # --- Extract and guard against invalid metadata ---
        dist_base = baseline_meta.get('total_distance_m', 0.0)
        dist_risk = risk_meta.get('total_distance_m', 0.0)
        safe_base = baseline_meta.get('p_safe', 0.0)
        safe_risk = risk_meta.get('p_safe', 0.0)

        # Avoid division by zero
        if dist_base <= 1e-3:
            dist_increase_pct = None
        else:
            dist_increase_pct = 100.0 * (dist_risk / dist_base - 1.0)

        safety_improvement = safe_risk - safe_base

        return {
            'baseline': {
                'path': baseline_path,
                'geometry': self._path_to_geometry(baseline_path),
                'metadata': baseline_meta,
            },
            'risk_aware': {
                'path': risk_path,
                'geometry': self._path_to_geometry(risk_path),
                'metadata': risk_meta,
            },
            'comparison': {
                'distance_increase_pct': dist_increase_pct,
                'safety_improvement': safety_improvement,
                'baseline_safety': safe_base,
                'risk_aware_safety': safe_risk,
            },
        }

    def export_debug_visualization(self, route_dict, output_path: str):
        """
        Export a GeoJSON for debugging a computed route.
        Includes route geometry and node coordinates.
        """
        try:
            path = route_dict.get("path", [])
            geometry = route_dict.get("geometry")

            # Build GeoJSON line
            features = []
            if geometry:
                line = LineString(geometry)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(line),
                    "properties": {
                        "type": "route",
                        "n_nodes": len(path),
                        "safety": route_dict.get("safety_score"),
                        "total_distance_m": route_dict["metadata"].get("total_distance_m"),
                    },
                })

            # Add nodes as points
            for node in path:
                node_data = self.graph.nodes[node]
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [node_data["x"], node_data["y"]],
                    },
                    "properties": {"node": node},
                })

            geojson = {"type": "FeatureCollection", "features": features}
            with open(output_path, "w") as f:
                json.dump(geojson, f, indent=2)
            print(f"[Planner] Exported debug GeoJSON → {output_path}")
        except Exception as e:
            print(f"[Planner] Failed to export debug visualization: {e}")

    @property
    def osm_graph(self) -> nx.Graph:
        """
        Backward-compatible alias for self.graph.
        Used by API routes that reference planner.osm_graph.
        """
        return self.graph
    def save_results(self, route: Dict, output_path: str):
        """Save route to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(route, f, indent=2)
        
        print(f"[Planner] Saved route to {output_path}")