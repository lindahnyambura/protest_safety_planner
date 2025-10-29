"""
GraphMovementMixin
-------------------
Graph-constrained navigation logic for agents operating within
real-world OpenStreetMap (OSM) networks.

Agents use this mixin to make movement decisions on a graph,
balancing path distance, hazard exposure, and congestion.

Expected attributes:
 - self.current_node : current graph node id
 - self.goal_node    : target goal node id
 - env.osm_graph     : networkx.Graph (walkable road network)
 - env.hazard_field  : np.ndarray (optional hazard concentration map)
 - env.node_occupancy: dict[node_id -> agent count]
 - env.n_cell_max    : occupancy limit per node
"""

import numpy as np
import networkx as nx


class GraphMovementMixin:
    """
    Adds graph-based, risk-aware movement decision-making for agents.

    Agents using this mixin must track their current and goal nodes
    within an environment that defines an OSM graph and occupancy map.
    """

    def graph_decide_action(self, env):
        """
        Graph-based movement with behavioral state awareness.

        Agents now make routing decisions that depend on their
        current behavioral state (CALM, ALERT, PANIC, FLEEING).

        Behaviors:
        - CALM / ALERT: normal goal-directed navigation
        - PANIC: hypersensitive to hazard proximity (triples hazard penalty)
        - FLEEING: overrides normal routing, moves away from hazard zones
        """
    
        G = getattr(env, "osm_graph", None)
        if G is None or self.current_node not in G:
            return self.current_node

        # Update behavioral state before routing
        if hasattr(self, "update_behavioral_state"):
            self.update_behavioral_state(env)

        # State-dependent logic
        if hasattr(self, "behavioral_state"):
            if self.behavioral_state == "FLEEING":
                # Override goal entirely: flee hazard hotspots
                return self._flee_on_graph(env, G)
            elif self.behavioral_state == "PANIC":
                # Increase hazard penalty temporarily
                self._temp_hazard_multiplier = 3.0

        # Standard goal-directed movement
        if self._at_goal_node(G):
            return self.current_node

        # Evaluate all neighboring nodes
        neighbor_scores = {}
        breakdowns = {}
        for nbr in G.neighbors(self.current_node):
            score, components = self._compute_neighbor_score(env, G, nbr)
            neighbor_scores[nbr] = score
            breakdowns[nbr] = components

        if not neighbor_scores:
            return self.current_node

        # Select best neighbor based on combined score
        best_node = max(neighbor_scores, key=neighbor_scores.get)

        # Reset temporary hazard multiplier (used in PANIC mode)
        if hasattr(self, "_temp_hazard_multiplier"):
            delattr(self, "_temp_hazard_multiplier")

        # Optional: Debug visualization or logs
        if getattr(self, "debug_routing", False):
            self._print_diagnostic(env, G, neighbor_scores, breakdowns, best_node)

        return best_node
    
    def _flee_on_graph(self, env, G):
        """
        Flee behavior: move away from hazard sources on graph.

        Strategy:
        1. Identify nearest hazard hotspot (above panic threshold)
        2. Choose neighbor node that maximizes distance *from* the hotspot
        3. Skip neighbors with significant hazard exposure
        4. Fallback to normal routing if no viable escape path
        """
        if self.current_node not in G:
            return self.current_node

        node_data = G.nodes[self.current_node]
        cx, cy = node_data.get("x", 0), node_data.get("y", 0)

        panic_threshold = getattr(self, "panic_threshold", 0.5)

        # Step 1: Identify hazard hotspots
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
            # No visible hazard → maintain current position
            return self.current_node

        # Step 2: Find nearest hotspot
        nearest_hotspot = min(hotspots, key=lambda h: (h[1] - cx) ** 2 + (h[2] - cy) ** 2)
        hx, hy = nearest_hotspot[1], nearest_hotspot[2]

        # Step 3: Evaluate neighbor escape options
        best_node = self.current_node
        max_distance = 0.0

        for nbr in G.neighbors(self.current_node):
            nbr_data = G.nodes[nbr]
            nx, ny = nbr_data.get("x", 0), nbr_data.get("y", 0)

            # Distance from hazard hotspot
            dist_from_hotspot = np.hypot(nx - hx, ny - hy)

            # Skip if the neighbor is itself hazardous
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

        # Fallback if no safe move found
        if best_node == self.current_node and hasattr(self, "_compute_neighbor_score"):
            # Try normal scoring if fleeing fails
            try:
                neighbor_scores = {
                    nbr: self._compute_neighbor_score(env, G, nbr)[0]
                    for nbr in G.neighbors(self.current_node)
                }
                if neighbor_scores:
                    best_node = max(neighbor_scores, key=neighbor_scores.get)
            except Exception:
                pass

        return best_node

    def _compute_neighbor_score(self, env, G, nbr):
        """
        Compute desirability of moving to a neighbor node with
        behavioral state awareness.

        Components:
            - Distance cost (shorter is better)
            - Goal utility (proximity to goal)
            - Hazard penalty (scaled by behavioral state + tolerance)
            - Congestion penalty (crowd avoidance)
            - Dispersion bonus (group cohesion or independence)

        Returns:
            (total_score: float, components: dict[str, float])
        """
        components = {}

        # Distance / movement cost
        try:
            edge_data = min(
                (d for d in G.get_edge_data(self.current_node, nbr).values()),
                key=lambda e: e.get("length", 1.0),
            )
            dist = edge_data.get("length", 1.0)
        except Exception:
            dist = 1.0
        components["distance"] = dist

        # Goal utility
        goal_utility = self._goal_heuristic(G, nbr)
        components["goal_utility"] = goal_utility

        # Hazard penalty (state-aware)
        hazard_penalty = 0.0
        if hasattr(env, "hazard_field") and env.hazard_field is not None:
            try:
                cell = env._node_to_cell(nbr)
                if 0 <= cell[1] < env.height and 0 <= cell[0] < env.width:
                    concentration = env.hazard_field.concentration[cell[1], cell[0]]

                    # Base hazard weighting
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
                        base_multiplier *= getattr(self, "_temp_hazard_multiplier")

                    hazard_penalty = concentration * base_multiplier
            except Exception:
                pass
        components["hazard_penalty"] = hazard_penalty

        # Congestion penalty
        occ_penalty = 0.0
        if nbr in getattr(env, "node_occupancy", {}):
            occ = env.node_occupancy[nbr]
            nmax = getattr(env, "n_cell_max", 10)
            if occ >= nmax:
                occ_penalty = 20.0  # hard block
            else:
                occ_penalty = (occ / nmax) * 5.0
        components["occ_penalty"] = occ_penalty

        # Dispersion bonus
        dispersion_bonus = 0.0
        if hasattr(self, "clustering_affinity"):
            affinity = getattr(self, "clustering_affinity", 0.5)
            # Agents with low affinity prefer open, less crowded paths
            if occ_penalty > 0:
                dispersion_bonus = -(1.0 - affinity) * occ_penalty * 0.5
        components["dispersion_bonus"] = dispersion_bonus

        # Combine components into final score
        total_score = (
            goal_utility
            - dist
            - hazard_penalty * (1.0 - getattr(self, "risk_tolerance", 0.5))
            - occ_penalty
            + dispersion_bonus
        )

        components["total_score"] = total_score
        return total_score, components

    def _goal_heuristic(self, G, node):
        """
        Estimate the utility of being at `node` relative to the goal.

        Uses weighted shortest-path distance if available,
        otherwise falls back to Euclidean distance.
        """
        if getattr(self, "goal_node", None) is None or self.goal_node not in G:
            return 0.0

        try:
            path_len = nx.shortest_path_length(G, node, self.goal_node, weight="length")
            return -path_len
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback: approximate with Euclidean distance
            try:
                ndata = G.nodes[node]
                gdata = G.nodes[self.goal_node]
                dx, dy = ndata["x"] - gdata["x"], ndata["y"] - gdata["y"]
                return -np.hypot(dx, dy)
            except Exception:
                return -1e6

    def _at_goal_node(self, G, tolerance_m=5.0):
        """
        Check whether the agent is sufficiently close to its goal node.

        Args:
            G: NetworkX graph.
            tolerance_m (float): Maximum distance in meters to count as "at goal".
        """
        try:
            ndata = G.nodes[self.current_node]
            gdata = G.nodes[self.goal_node]
            dx, dy = ndata["x"] - gdata["x"], ndata["y"] - gdata["y"]
            return (dx * dx + dy * dy) ** 0.5 <= tolerance_m
        except KeyError:
            return False

    def nearest_node_from_cell(self, env, i, j):
        """
        Return the nearest graph node for a given grid cell (i, j).

        Uses the precomputed `cell_to_node` lookup if available.
        """
        if not hasattr(env, "cell_to_node") or env.cell_to_node is None:
            raise RuntimeError("cell_to_node map not loaded in environment.")
        node_id = env.cell_to_node[j, i]
        if node_id in (-1, None, "None", "nan"):
            return None
        return str(node_id)

    def _print_diagnostic(self, env, G, neighbor_scores, breakdowns, best_node):
        """
        Print a detailed routing breakdown for debugging.

        Shows each neighbor node’s distance, goal utility,
        hazard penalty, congestion penalty, and final score.
        """
        print(f"\n[Routing Diagnostic] Agent {self.id} ({self.agent_type})")
        curr_xy = (G.nodes[self.current_node]["x"], G.nodes[self.current_node]["y"])
        goal_xy = (G.nodes[self.goal_node]["x"], G.nodes[self.goal_node]["y"])
        print(f"  Current node: {self.current_node} at {curr_xy}")
        print(f"  Goal node:    {self.goal_node} at {goal_xy}")

        sorted_nodes = sorted(neighbor_scores.items(), key=lambda kv: kv[1], reverse=True)
        print("  Neighbor scoring breakdown:")
        for nbr, score in sorted_nodes:
            c = breakdowns[nbr]
            print(
                f"   → Node {nbr}: score={score:+.2f} | "
                f"dist={c['distance']:.1f} | "
                f"goal_util={c['goal_utility']:+.1f} | "
                f"hazard={c['hazard_penalty']:.1f} | "
                f"occ_penalty={c['occ_penalty']:.1f}"
            )

        print(f"  Chosen next node: {best_node}\n")
