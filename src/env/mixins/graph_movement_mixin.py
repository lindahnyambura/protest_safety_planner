"""
GraphMovementMixin - FIXED VERSION
-----------------------------------
Corrects action return types and adds proper occupancy-aware routing.

Key fixes:
1. Always return INTEGER action codes (0-8), never node IDs
2. Check edge existence before scoring neighbors
3. Account for node capacity in scoring
4. Proper UTM↔grid coordinate handling
"""

import numpy as np
import networkx as nx
import pandas as pd


class GraphMovementMixin:
    """Graph-based, risk-aware movement for OSM agents."""

    def graph_decide_action(self, env):
        """
        Decide next node to move to, return as INTEGER action.
        
        Returns:
            int: Index of chosen neighbor in G.neighbors() list, or 0 for STAY
        """
        G = getattr(env, "osm_graph", None)
        if G is None or self.current_node not in G:
            return 0  # STAY action (not node ID!)

        # Update behavioral state before routing
        if hasattr(self, "update_behavioral_state"):
            self.update_behavioral_state(env)

        # Sync goal_node with goal (if goal changed via update_goal)
        self._sync_goal_node(env, G)

        # State-dependent logic
        if hasattr(self, "behavioral_state"):
            if self.behavioral_state == "FLEEING":
                chosen_node = self._flee_on_graph(env, G)
                return self._node_to_action(G, chosen_node)
            elif self.behavioral_state == "PANIC":
                self._temp_hazard_multiplier = 3.0

        # Check if at goal
        if self._at_goal_node(G):
            return 0  # STAY action

        # Evaluate all neighboring nodes
        neighbors = list(G.neighbors(self.current_node))
        if not neighbors:
            return 0  # No valid moves

        neighbor_scores = {}
        for nbr in neighbors:
            # Skip if edge doesn't exist (one-way streets)
            if not G.has_edge(self.current_node, nbr):
                continue
            
            score, _ = self._compute_neighbor_score(env, G, nbr)
            neighbor_scores[nbr] = score

        if not neighbor_scores:
            return 0

        # Select best neighbor
        best_node = max(neighbor_scores, key=neighbor_scores.get)

        # Reset temporary hazard multiplier
        if hasattr(self, "_temp_hazard_multiplier"):
            delattr(self, "_temp_hazard_multiplier")

        # Convert node choice to action index
        return self._node_to_action(G, best_node)

    def _node_to_action(self, G, chosen_node):
        """
        Convert a chosen neighbor node into an action index.
        
        Maps the chosen node to its position in the neighbors list.
        This preserves compatibility with the action dict in env.step().
        
        Returns:
            int: Action index (1-N for move, 0 for stay)
        """
        if chosen_node == self.current_node:
            return 0  # STAY
        
        neighbors = list(G.neighbors(self.current_node))
        try:
            # Return index+1 (since 0=STAY in action space)
            return neighbors.index(chosen_node) + 1
        except ValueError:
            return 0  # Fallback: STAY
    
    def _sync_goal_node(self, env, G):
        """
        Sync goal_node with goal (grid coordinates → graph node).
        Only updates if goal has changed significantly.
        """
        if not hasattr(self, 'goal') or not hasattr(env, 'cell_to_node'):
            return
        
        gx, gy = map(int, np.clip(self.goal, [0, 0], [env.width - 1, env.height - 1]))
        
        try:
            potential_goal_node = env.cell_to_node[gy, gx]
            if potential_goal_node not in (-1, None, "None", "nan") and not pd.isna(potential_goal_node):
                new_goal_node = str(potential_goal_node)
                if new_goal_node in G and new_goal_node != getattr(self, 'goal_node', None):
                    self.goal_node = new_goal_node
        except Exception:
            pass  # Keep existing goal_node if conversion fails
    
    def _flee_on_graph(self, env, G):
        """
        Flee behavior: move away from hazard hotspots.
        
        Returns:
            str: Node ID to flee to (not action index!)
        """
        if self.current_node not in G:
            return self.current_node

        node_data = G.nodes[self.current_node]
        cx, cy = node_data.get("x", 0), node_data.get("y", 0)
        panic_threshold = getattr(self, "panic_threshold", 0.5)

        # Identify hazard hotspots
        hotspots = []
        for node_id, ndata in G.nodes(data=True):
            try:
                cell = env._node_to_cell(str(node_id))
                hazard = env.hazard_field.concentration[cell[1], cell[0]]
                if hazard > panic_threshold:
                    hotspots.append((node_id, ndata.get("x", 0), ndata.get("y", 0), hazard))
            except Exception:
                continue

        if not hotspots:
            return self.current_node

        # Find nearest hotspot
        nearest_hotspot = min(hotspots, key=lambda h: (h[1] - cx) ** 2 + (h[2] - cy) ** 2)
        hx, hy = nearest_hotspot[1], nearest_hotspot[2]

        # Find neighbor that maximizes distance from hotspot
        best_node = self.current_node
        max_distance = 0.0

        for nbr in G.neighbors(self.current_node):
            if not G.has_edge(self.current_node, nbr):
                continue
            
            nbr_data = G.nodes[nbr]
            nx, ny = nbr_data.get("x", 0), nbr_data.get("y", 0)
            dist_from_hotspot = np.hypot(nx - hx, ny - hy)

            # Skip if neighbor is itself hazardous
            try:
                nbr_cell = env._node_to_cell(str(nbr))
                nbr_hazard = env.hazard_field.concentration[nbr_cell[1], nbr_cell[0]]
                if nbr_hazard > panic_threshold * 0.5:
                    continue
            except Exception:
                pass

            if dist_from_hotspot > max_distance:
                max_distance = dist_from_hotspot
                best_node = nbr

        return best_node

    def _compute_neighbor_score(self, env, G, nbr):
        """
        Score a neighbor node with capacity-aware penalties.
        
        Returns:
            (score, components): Total score and breakdown dict
        """
        components = {}

        # Distance cost (edge length)
        try:
            edge_data = min(
                (d for d in G.get_edge_data(self.current_node, nbr).values()),
                key=lambda e: e.get("length", 1.0),
            )
            dist = edge_data.get("length", 1.0)
        except Exception:
            dist = 1.0
        components["distance"] = dist

        # Goal utility (shortest path heuristic)
        goal_utility = self._goal_heuristic(G, nbr)
        components["goal_utility"] = goal_utility

        # Hazard penalty (state-aware)
        hazard_penalty = self._compute_hazard_penalty(env, nbr)
        components["hazard_penalty"] = hazard_penalty

        # Capacity-aware congestion penalty
        node_capacity = G.nodes[nbr].get('capacity', 6)
        current_occ = env.node_occupancy.get(str(nbr), 0)  # Ensure string key
        
        if current_occ >= node_capacity:
            # Node full - make extremely unattractive
            occ_penalty = 1000.0  # Effectively blocks movement
        else:
            utilization = current_occ / node_capacity
            if utilization > 0.8:
                occ_penalty = 50.0
            elif utilization > 0.6:
                occ_penalty = 25.0
            else:
                occ_penalty = utilization * 15.0
        
        components["occ_penalty"] = occ_penalty

        # Dispersion bonus (for agents with low clustering affinity)
        dispersion_bonus = 0.0
        if hasattr(self, "clustering_affinity"):
            affinity = getattr(self, "clustering_affinity", 0.5)
            if occ_penalty > 0:
                dispersion_bonus = -(1.0 - affinity) * occ_penalty * 0.5
        components["dispersion_bonus"] = dispersion_bonus

        # Combine components
        risk_tolerance = getattr(self, "risk_tolerance", 0.5)
        total_score = (
            goal_utility
            - dist * 0.1  # Small distance penalty
            - hazard_penalty * (1.0 - risk_tolerance)
            - occ_penalty
            + dispersion_bonus
        )

        return total_score, components

    def _compute_hazard_penalty(self, env, node_id):
        """Compute hazard penalty for a node with behavioral state scaling."""
        if not hasattr(env, "hazard_field") or env.hazard_field is None:
            return 0.0
        
        try:
            cell = env._node_to_cell(str(node_id))
            if 0 <= cell[1] < env.height and 0 <= cell[0] < env.width:
                concentration = env.hazard_field.concentration[cell[1], cell[0]]
                
                # Base multiplier
                base_multiplier = 5.0
                
                # Modify by behavioral state
                state = getattr(self, "behavioral_state", "CALM")
                if state == "PANIC":
                    base_multiplier = 15.0
                elif state == "FLEEING":
                    base_multiplier = 30.0
                elif state == "ALERT":
                    base_multiplier = 10.0
                
                # Apply temporary panic scaling
                if hasattr(self, "_temp_hazard_multiplier"):
                    base_multiplier *= self._temp_hazard_multiplier
                
                return concentration * base_multiplier
        except Exception:
            pass
        
        return 0.0

    def _goal_heuristic(self, G, node):
        """Estimate utility of being at node relative to goal."""
        if getattr(self, "goal_node", None) is None or self.goal_node not in G:
            return 0.0

        try:
            # Use shortest path length as heuristic
            path_len = nx.shortest_path_length(G, node, self.goal_node, weight="length")
            return -path_len  # Negative because we minimize distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback: Euclidean distance
            try:
                ndata = G.nodes[node]
                gdata = G.nodes[self.goal_node]
                dx = ndata["x"] - gdata["x"]
                dy = ndata["y"] - gdata["y"]
                return -np.hypot(dx, dy)
            except Exception:
                return -1e6  # Very bad if unreachable

    def _at_goal_node(self, G, tolerance_m=10.0):
        """
        Check if agent is sufficiently close to goal node.
        
        Uses graph distance (via shortest path), not Euclidean distance,
        to avoid coordinate system mismatches.
        """
        if not hasattr(self, 'goal_node') or self.goal_node not in G:
            return False
        
        if self.current_node == self.goal_node:
            return True
        
        try:
            # Check if within tolerance (in meters)
            dist = nx.shortest_path_length(G, self.current_node, self.goal_node, weight="length")
            return dist <= tolerance_m
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    def nearest_node_from_cell(self, env, i, j):
        """Return nearest graph node for grid cell (i, j)."""
        if not hasattr(env, "cell_to_node") or env.cell_to_node is None:
            raise RuntimeError("cell_to_node map not loaded in environment.")
        
        node_id = env.cell_to_node[j, i]
        if node_id in (-1, None, "None", "nan"):
            return None
        return str(node_id)