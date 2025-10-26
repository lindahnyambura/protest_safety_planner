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
        Decide which graph node to move toward next.

        Balances multiple factors:
        - Distance (shorter edges preferred)
        - Goal proximity (heuristic utility)
        - Hazard avoidance (based on env.hazard_field)
        - Congestion avoidance (based on env.node_occupancy)

        Returns:
            next_node_id (str or int): Chosen neighbor node to move to,
                                       or current node if staying put.
        """
        G = getattr(env, "osm_graph", None)
        if G is None or self.current_node not in G:
            return self.current_node

        # If agent is already close to its goal, do not move
        if self._at_goal_node(G):
            return self.current_node

        # Evaluate all neighbors
        neighbor_scores = {}
        breakdowns = {}
        for nbr in G.neighbors(self.current_node):
            score, components = self._compute_neighbor_score(env, G, nbr)
            neighbor_scores[nbr] = score
            breakdowns[nbr] = components

        if not neighbor_scores:
            return self.current_node

        # Choose best neighbor by score
        best_node = max(neighbor_scores, key=neighbor_scores.get)

        # Optional debug diagnostics
        if getattr(self, "debug_routing", False):
            self._print_diagnostic(env, G, neighbor_scores, breakdowns, best_node)

        return best_node

    def _compute_neighbor_score(self, env, G, nbr):
        """
        Compute a desirability score for a neighbor node.

        Combines several components:
        - Distance cost (shorter edges are better)
        - Goal utility (closeness to goal)
        - Hazard penalty (discourages moving into dangerous zones)
        - Congestion penalty (discourages crowded nodes)

        Returns:
            tuple:
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

        # Goal-directed utility
        goal_utility = self._goal_heuristic(G, nbr)
        components["goal_utility"] = goal_utility

        # Hazard penalty (optional)
        hazard_penalty = 0.0
        if hasattr(env, "hazard_field") and env.hazard_field is not None:
            cell = env._node_to_cell(nbr)
            if 0 <= cell[1] < env.height and 0 <= cell[0] < env.width:
                hazard_penalty = env.hazard_field.concentration[cell[1], cell[0]] * 5.0
        components["hazard_penalty"] = hazard_penalty

        # Congestion penalty based on occupancy
        occ_penalty = 0.0
        if nbr in env.node_occupancy:
            occ = env.node_occupancy[nbr]
            if occ >= env.n_cell_max:
                occ_penalty = 10.0
            else:
                occ_penalty = (occ / env.n_cell_max) * 5.0
        components["occ_penalty"] = occ_penalty

        # Combine all components into a total score
        total_score = (
            goal_utility
            - dist
            - hazard_penalty * (1.0 - self.risk_tolerance)
            - occ_penalty
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
