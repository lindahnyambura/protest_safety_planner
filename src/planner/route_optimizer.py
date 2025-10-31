"""
route_optimizer.py - Path finding algorithms for risk-aware routing

Implements Dijkstra's algorithm and A* search with risk-aware cost functions.
"""

import numpy as np
import networkx as nx
import heapq
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class PathMetadata:
    """Metadata about a computed path."""
    total_distance_m: float
    total_cost: float
    p_safe: float
    edge_risks: List[float]
    num_turns: int
    estimated_time_s: float
    nodes_explored: int


class RouteOptimizer:
    """
    Path finding algorithms for risk-aware routing.
    
    Supports both Dijkstra's algorithm (optimal) and A* (faster with heuristic).
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize optimizer.
        
        Args:
            graph: NetworkX graph with node coordinates and edge attributes
        """
        self.graph = graph
        
        # Precompute node positions for heuristic
        self.node_positions = {
            node: (data['x'], data['y']) 
            for node, data in graph.nodes(data=True)
        }
        
        # Average walking speed for time estimates (m/s)
        self.walking_speed = 1.4  # ~5 km/h
    
    def dijkstra(self, 
                 start: str, 
                 goal: str, 
                 cost_fn: Callable) -> Tuple[List[str], float, Dict]:
        """
        Dijkstra's algorithm with risk-aware costs.
        
        Args:
            start: Start node ID
            goal: Goal node ID
            cost_fn: CostFunction instance with compute_cost method
            
        Returns:
            path: List of node IDs (empty if no path)
            total_cost: Cumulative cost
            metadata: PathMetadata dictionary
        """
        # Priority queue: (cost, node, path_so_far)
        heap = [(0.0, start, [start])]
        visited = set()
        nodes_explored = 0
        
        # Best costs to each node
        best_cost = {start: 0.0}
        
        while heap:
            current_cost, current_node, path = heapq.heappop(heap)
            
            # Goal check
            if current_node == goal:
                metadata = self._compute_metadata(path, cost_fn)
                metadata['nodes_explored'] = nodes_explored
                return path, current_cost, metadata
            
            # Skip if already visited
            if current_node in visited:
                continue
            
            visited.add(current_node)
            nodes_explored += 1
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                # Get edge data
                edge_data = self.graph[current_node][neighbor]
                distance = edge_data.get('length', 0.0)
                p_harm = edge_data.get('p_harm', 0.0)
                
                # Compute edge cost
                edge_cost = cost_fn.compute_cost(distance, p_harm)
                new_cost = current_cost + edge_cost
                
                # Update if better
                if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                    best_cost[neighbor] = new_cost
                    new_path = path + [neighbor]
                    heapq.heappush(heap, (new_cost, neighbor, new_path))
        
        # No path found
        return [], float('inf'), {
            'error': 'No path found',
            'nodes_explored': nodes_explored
        }
    
    def astar(self, 
              start: str, 
              goal: str, 
              cost_fn: Callable) -> Tuple[List[str], float, Dict]:
        """
        A* search with Euclidean distance heuristic.
        
        Args:
            start: Start node ID
            goal: Goal node ID
            cost_fn: CostFunction instance
            
        Returns:
            path: List of node IDs
            total_cost: Cumulative cost
            metadata: PathMetadata dictionary
        """
        # Heuristic: Euclidean distance to goal
        goal_x, goal_y = self.node_positions[goal]
        
        def heuristic(node: str) -> float:
            """Admissible heuristic (straight-line distance)."""
            x, y = self.node_positions[node]
            return np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        
        # Priority queue: (f_score, g_score, node, path)
        # f_score = g_score + heuristic
        h_start = heuristic(start)
        heap = [(h_start, 0.0, start, [start])]
        
        visited = set()
        best_g = {start: 0.0}
        nodes_explored = 0
        
        while heap:
            f_score, g_score, current_node, path = heapq.heappop(heap)
            
            # Goal check
            if current_node == goal:
                metadata = self._compute_metadata(path, cost_fn)
                metadata['nodes_explored'] = nodes_explored
                return path, g_score, metadata
            
            # Skip if visited
            if current_node in visited:
                continue
            
            visited.add(current_node)
            nodes_explored += 1
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                # Edge cost
                edge_data = self.graph[current_node][neighbor]
                distance = edge_data.get('length', 0.0)
                p_harm = edge_data.get('p_harm', 0.0)
                edge_cost = cost_fn.compute_cost(distance, p_harm)
                
                new_g = g_score + edge_cost
                
                # Update if better
                if neighbor not in best_g or new_g < best_g[neighbor]:
                    best_g[neighbor] = new_g
                    h = heuristic(neighbor)
                    new_f = new_g + h
                    new_path = path + [neighbor]
                    heapq.heappush(heap, (new_f, new_g, neighbor, new_path))
        
        # No path found
        return [], float('inf'), {
            'error': 'No path found',
            'nodes_explored': nodes_explored
        }
    
    def _compute_metadata(self, path: List[str], cost_fn: Callable) -> Dict:
        """
        Compute comprehensive path metadata.
        
        Args:
            path: List of node IDs
            cost_fn: CostFunction instance
            
        Returns:
            metadata: Dictionary with path statistics
        """
        if len(path) < 2:
            return {
                'total_distance_m': 0.0,
                'total_cost': 0.0,
                'p_safe': 1.0,
                'edge_risks': [],
                'num_turns': 0,
                'estimated_time_s': 0.0
            }
        
        # Collect edge statistics
        distances = []
        p_harms = []
        costs = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.graph[u][v]

            # --- Compute distance (fallback if missing) ---
            if 'length' in edge_data and edge_data['length'] > 0:
                distance = edge_data['length']
            else:
                # Fallback: Euclidean distance between node coordinates
                u_data = self.graph.nodes[u]
                v_data = self.graph.nodes[v]
                dx = u_data['x'] - v_data['x']
                dy = u_data['y'] - v_data['y']
                distance = float((dx**2 + dy**2) ** 0.5)

            # --- Risk & cost ---
            p_harm = edge_data.get('p_harm', 0.0)
            cost = cost_fn.compute_cost(distance, p_harm)

            distances.append(distance)
            p_harms.append(p_harm)
            costs.append(cost)

        
        # Aggregate metrics
        total_distance = sum(distances)
        total_cost = sum(costs)
        
        # Route safety (independence assumption)
        p_safe = cost_fn.compute_route_safety(np.array(p_harms))
        
        # Count turns (bearing change > 30°)
        num_turns = self._count_turns(path)
        
        # Estimated time
        estimated_time_s = total_distance / self.walking_speed
        
        return {
            'total_distance_m': float(total_distance),
            'total_cost': float(total_cost),
            'p_safe': float(p_safe),
            'edge_risks': [float(p) for p in p_harms],
            'num_turns': num_turns,
            'estimated_time_s': float(estimated_time_s),
            'mean_edge_risk': float(np.mean(p_harms)) if p_harms else 0.0,
            'max_edge_risk': float(np.max(p_harms)) if p_harms else 0.0
        }
    
    def _count_turns(self, path: List[str], threshold_deg: float = 30.0) -> int:
        """
        Count number of significant turns in path.
        
        Args:
            path: List of node IDs
            threshold_deg: Minimum bearing change to count as turn
            
        Returns:
            num_turns: Count of turns
        """
        if len(path) < 3:
            return 0
        
        num_turns = 0
        
        for i in range(1, len(path) - 1):
            # Get three consecutive points
            prev_node = path[i-1]
            curr_node = path[i]
            next_node = path[i+1]
            
            # Get positions
            x1, y1 = self.node_positions[prev_node]
            x2, y2 = self.node_positions[curr_node]
            x3, y3 = self.node_positions[next_node]
            
            # Compute bearings
            bearing1 = np.arctan2(y2 - y1, x2 - x1)
            bearing2 = np.arctan2(y3 - y2, x3 - x2)
            
            # Bearing change
            delta = np.abs(np.degrees(bearing2 - bearing1))
            
            # Normalize to [0, 180]
            if delta > 180:
                delta = 360 - delta
            
            if delta > threshold_deg:
                num_turns += 1
        
        return num_turns
    
    def compute_bearing(self, node1: str, node2: str) -> float:
        """
        Compute bearing from node1 to node2 (degrees, 0=North).
        
        Args:
            node1, node2: Node IDs
            
        Returns:
            bearing: Bearing in degrees [0, 360)
        """
        x1, y1 = self.node_positions[node1]
        x2, y2 = self.node_positions[node2]
        
        # Compute angle (0=East, counterclockwise)
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        
        # Convert to bearing (0=North, clockwise)
        bearing = (90 - np.degrees(angle_rad)) % 360
        
        return bearing
    
    def bearing_to_direction(self, bearing: float) -> str:
        """
        Convert bearing to cardinal direction.
        
        Args:
            bearing: Bearing in degrees [0, 360)
            
        Returns:
            direction: One of ['north', 'northeast', 'east', 'southeast',
                              'south', 'southwest', 'west', 'northwest']
        """
        directions = [
            'north', 'northeast', 'east', 'southeast',
            'south', 'southwest', 'west', 'northwest'
        ]
        
        # Map to 8 directions
        idx = int((bearing + 22.5) / 45.0) % 8
        return directions[idx]