# src/env/mixins/graph_movement_mixin.py
"""
GraphMovementMixin
-------------------
Provides graph-constrained movement utilities for agents navigating the
real-world OSM network. This mixin assumes:
 - self.G : networkx.MultiDiGraph of the current walkable graph
 - self.cell_to_node : 2D np.ndarray mapping each cell (i,j) to nearest node id
 - self.goal_node : the OSM node id of the agent's current goal
 - self.current_node : the agent’s current node
 - self.metadata : dict containing affine transform + grid metadata

Implements:
    _score_neighbors(node)  -> numeric cost or utility
    _utility_goal(node)     -> estimated goal utility
    _at_goal(node)          -> goal reached check
"""

import numpy as np
import networkx as nx
from shapely.geometry import Point


class GraphMovementMixin:
    """
    Adds graph-based navigation scoring for agents.
    Intended to be mixed into your Agent base class.
    """

    # === Core Graph Accessors ===

    def _get_neighbors(self, node):
        """Return neighbor node IDs (outgoing edges)."""
        if self.G is None or node not in self.G:
            return []
        return list(self.G.neighbors(node))

    # === Scoring Functions ===

    def _score_neighbors(self, node):
        """
        Compute movement desirability for neighboring nodes.
        Returns dict[node_id] = -cost or +utility score.
        """
        scores = {}
        if self.G is None or node not in self.G:
            return scores

        for nbr in self._get_neighbors(node):
            try:
                # Use edge length as base movement cost
                data = min(
                    (d for d in self.G.get_edge_data(node, nbr).values()),
                    key=lambda e: e.get("length", 1.0),
                )
                dist = data.get("length", 1.0)

                # Add congestion, hazard, or closure penalties (later extension)
                hazard_cost = 0.0
                if hasattr(self, "hazard_field"):
                    hazard_cost = self.hazard_field.get(nbr, 0.0)

                # Basic score: smaller distance = better
                scores[nbr] = -dist - hazard_cost

            except Exception as e:
                print(f"[WARN] Failed scoring neighbor {nbr}: {e}")
        return scores

    def _utility_goal(self, node):
        """
        Compute heuristic utility relative to goal node.
        By default uses shortest-path distance if available, else straight-line.
        """
        if getattr(self, "goal_node", None) is None or self.G is None:
            return 0.0
        try:
            path_len = nx.shortest_path_length(
                self.G, node, self.goal_node, weight="length"
            )
            return -path_len  # smaller distance → higher utility
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # fallback: straight-line Euclidean
            try:
                ndata = self.G.nodes[node]
                gdata = self.G.nodes[self.goal_node]
                dx = ndata["x"] - gdata["x"]
                dy = ndata["y"] - gdata["y"]
                return -np.hypot(dx, dy)
            except KeyError:
                return -1e6

    def _at_goal(self, node, tolerance_m=5.0):
        """
        Check whether the agent has reached its goal node.
        Uses Euclidean tolerance in projected CRS (meters).
        """
        if getattr(self, "goal_node", None) is None or self.G is None:
            return False
        try:
            ndata = self.G.nodes[node]
            gdata = self.G.nodes[self.goal_node]
            dx = ndata["x"] - gdata["x"]
            dy = ndata["y"] - gdata["y"]
            return (dx * dx + dy * dy) ** 0.5 <= tolerance_m
        except KeyError:
            return False

    # === Helpers ===

    def nearest_node_from_cell(self, i, j):
        """Return nearest graph node for cell indices (i,j)."""
        if not hasattr(self, "cell_to_node"):
            raise RuntimeError("cell_to_node map not loaded in agent/env.")
        node_id = self.cell_to_node[i, j]
        if node_id == -1:
            return None
        return str(node_id)
