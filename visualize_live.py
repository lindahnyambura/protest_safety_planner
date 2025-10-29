#!/usr/bin/env python3
"""
visualize_live.py — Real-time protest simulation visualization (Graph + Grid aware)
-------------------------------------------------------------------------------
Supports both synthetic grid-based and real OSM graph-based environments.

Features:
- Real-time matplotlib animation
- Graph overlays (streets, buildings, names)
- Hazard, crowd density, and agent states
- Optional path visualization for debug_routing agents
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import numpy as np
import sys
from pathlib import Path
import networkx as nx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.protest_env import ProtestEnv, load_config
from src.env.agent import AgentState


# ------------------------------------------------------------
def _draw_osm_features(ax, env):
    """
    Draw roads, buildings and (sparse) street labels projected into the
    environment's grid coordinate system so they align with the hazard mask
    and agent positions.

    This function expects the environment to provide:
      - env.osm_graph : networkx graph with node attributes 'x','y'
      - env._node_to_cell(node_id) -> (x_cell, y_cell)
      - env.buildings_gdf (optional) with shapely geometries and an env.affine to convert coords
    """
    G = getattr(env, "osm_graph", None)

    # --- Draw buildings first (so they appear under hazard & roads) ---
    buildings = getattr(env, "buildings_gdf", None)
    affine = getattr(env, "affine", None)
    if buildings is not None and affine is not None:
        try:
            for geom in buildings.geometry:
                if geom is None:
                    continue
                # Handle MultiPolygon by iterating parts
                geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
                for part in geoms:
                    try:
                        xs = []
                        ys = []
                        # exterior may not exist for degenerate geometries; guard it
                        exterior = getattr(part, "exterior", None)
                        if exterior is None:
                            continue
                        for x_coord, y_coord in exterior.coords:
                            # project spatial coordinate -> grid cell using affine
                            gx, gy = affine * (x_coord, y_coord)
                            xs.append(gx)
                            ys.append(gy)
                        if xs and ys:
                            ax.fill(xs, ys, color="#b0b0b0", alpha=0.30, linewidth=0.2, zorder=0)
                    except Exception:
                        # skip malformed polygon part
                        continue
        except Exception:
            # if geopandas/shapely missing or geometry malformed, skip buildings
            pass

    # --- Draw graph edges projected into grid coords ---
    if G is not None:
        for (u, v, k, data) in G.edges(keys=True, data=True):
            try:
                # Map node u->cell and v->cell; _node_to_cell should handle node ids
                cell_u = env._node_to_cell(u)
                cell_v = env._node_to_cell(v)
                if cell_u is None or cell_v is None:
                    continue
                x1, y1 = float(cell_u[0]), float(cell_u[1])
                x2, y2 = float(cell_v[0]), float(cell_v[1])
                ax.plot([x1, x2], [y1, y2], color="gray", linewidth=0.6, alpha=0.75, zorder=2)
            except Exception:
                # sometimes node projection fails; skip edge
                continue

        # Optional: plot small node markers (low alpha)
        try:
            node_coords = []
            for n, d in G.nodes(data=True):
                try:
                    c = env._node_to_cell(n)
                    if c is None:
                        continue
                    node_coords.append((float(c[0]), float(c[1])))
                except Exception:
                    continue
            if node_coords:
                xs, ys = zip(*node_coords)
                ax.scatter(xs, ys, s=4, color="black", alpha=0.25, zorder=3)
        except Exception:
            pass

        # --- Sparse street labeling (projected midpoint of edge) ---
        for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
            if i % 20 != 0:
                continue
            name = data.get("name")
            if not name:
                continue
            # if name is list/tuple, take first string element
            if isinstance(name, (list, tuple)):
                name = next((n for n in name if isinstance(n, str)), None)
            if not isinstance(name, str):
                continue
            if "unnamed" in name.lower():
                continue
            # project midpoint to cell coords
            try:
                cu = env._node_to_cell(u)
                cv = env._node_to_cell(v)
                if cu is None or cv is None:
                    continue
                midx, midy = (cu[0] + cv[0]) / 2.0, (cu[1] + cv[1]) / 2.0
                ax.text(midx, midy, name, fontsize=6, color="dimgray",
                        alpha=0.6, ha="center", va="center", zorder=5)
            except Exception:
                continue

def _draw_agent_paths(ax, env):
    """
    Draw faint path lines (node sequence) from current_node -> goal_node
    for protesters that have `debug_routing=True`. Paths are projected into
    grid coordinates using env._node_to_cell.
    """
    if getattr(env, "osm_graph", None) is None:
        return

    G = env.osm_graph
    for agent in env.protesters:
        if not getattr(agent, "debug_routing", False):
            continue
        if not hasattr(agent, "current_node") or not hasattr(agent, "goal_node"):
            continue
        try:
            path = nx.shortest_path(G, agent.current_node, agent.goal_node, weight="length")
            coords = []
            for n in path:
                try:
                    c = env._node_to_cell(n)
                    if c is None:
                        raise RuntimeError("node->cell returned None")
                    coords.append((float(c[0]), float(c[1])))
                except Exception:
                    # abort this path if any node fails to project
                    coords = []
                    break
            if len(coords) >= 2:
                xs, ys = zip(*coords)
                ax.plot(xs, ys, color="deepskyblue", linewidth=0.8, alpha=0.35, zorder=1)
        except Exception:
            # fallback: skip problematic agent/path
            continue

# ------------------------------------------------------------
def run_live_episode(config_path='configs/default_scenario.yaml', max_steps=100, seed=42):
    """Run episode with live visualization and diagnostics."""
    config = load_config(config_path)
    env = ProtestEnv(config)
    obs, info = env.reset(seed=seed)

    graph_mode = getattr(env, "osm_graph", None) is not None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Live Protest Simulation', fontsize=16, fontweight='bold')
    plt.style.use('seaborn-v0_8-whitegrid')
    for ax in axes:
        ax.set_facecolor('#f7f7f7')

    exits = config['agents']['protesters']['goals']['exit_points']
    harm_timeline, max_conc_timeline, incap_history, mean_harm_history = [], [], [], []

    # --------------------------------------------------------
    def update(frame):
        nonlocal harm_timeline, max_conc_timeline, incap_history, mean_harm_history
        obs, _, terminated, truncated, info = env.step(None)

        harm_timeline.append(info['harm_grid'].sum())
        max_conc_timeline.append(env.hazard_field.concentration.max())
        incap_history.append(info['agent_states']['n_incapacitated'])
        mean_harm_history.append(info['agent_states']['mean_harm'])

        for ax in axes:
            ax.clear()

        # ====== PANEL 1: HAZARD FIELD ======
        hazard_display = env.hazard_field.concentration.copy()
        im0 = axes[0].imshow(
            hazard_display, cmap='Reds', vmin=0, vmax=20,
            origin='upper', alpha=0.8, interpolation='bilinear',
            extent=[0, env.width, env.height, 0],  # ensures correct scaling
            aspect='equal'                         # keeps it square
        )
        axes[0].set_xlim(0, env.width)
        axes[0].set_ylim(env.height, 0)


        if graph_mode:
            _draw_osm_features(axes[0], env)
        else:
            obs_mask = env.obstacle_mask.astype(float)
            obs_mask[obs_mask == 0] = np.nan
            axes[0].imshow(obs_mask, cmap='gray', alpha=0.3, origin='upper')

        for exit_pos in exits:
            rect = mpatches.Rectangle((exit_pos[0] - 0.5, exit_pos[1] - 0.5), 1, 1,
                                      color='limegreen', alpha=0.25, zorder=5)
            axes[0].add_patch(rect)

        axes[0].set_title(f'Hazard Field (Step {frame+1})', fontsize=12, fontweight='bold')
        axes[0].set_xticks([]); axes[0].set_yticks([])

        # ====== PANEL 2: AGENTS ======
        if graph_mode:
            _draw_osm_features(axes[1], env)
            _draw_agent_paths(axes[1], env)
        else:
            axes[1].imshow(env.obstacle_mask, cmap='gray', alpha=0.15, origin='upper')

        for exit_pos in exits:
            rect = mpatches.Rectangle((exit_pos[0]-1.5, exit_pos[1]-1.5), 3, 3,
                                      facecolor='limegreen', alpha=0.2,
                                      edgecolor='green', linestyle='--',
                                      linewidth=1.0, zorder=1)
            axes[1].add_patch(rect)

        marker_size = 20
        for agent in env.protesters:
            if agent.state == AgentState.MOVING: color = 'deepskyblue'
            elif agent.state == AgentState.STUNNED: color = 'gold'
            elif agent.state == AgentState.INCAPACITATED: color = 'dimgray'
            elif agent.state == AgentState.SAFE: color = 'limegreen'
            else: color = 'lightgray'

            # For graph mode → project via _node_to_cell
            pos = agent.pos
            if graph_mode and hasattr(agent, "current_node"):
                pos = env._node_to_cell(agent.current_node)

            axes[1].scatter(pos[0], pos[1], s=marker_size, c=color,
                            alpha=0.85, edgecolors='none', zorder=5)

        for agent in env.police_agents:
            axes[1].scatter(agent.pos[0], agent.pos[1], s=marker_size,
                            c='crimson', alpha=0.9, edgecolors='black',
                            linewidths=0.4, zorder=6)

        axes[1].set_title(f'Agents (Step {frame+1})', fontsize=12, fontweight='bold')
        axes[1].set_xlim(0, env.width); axes[1].set_ylim(env.height, 0)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        # ====== PANEL 3: CROWD DENSITY ======
        occ = env.occupancy_count
        im3 = axes[2].imshow(
            occ, cmap='viridis', vmin=0, vmax=max(5, occ.max()),
            origin='upper', interpolation='nearest'
        )
        if not hasattr(update, 'colorbar'):
            update.colorbar = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            update.colorbar.set_label('Agents per cell')
        update.colorbar.update_normal(im3)

        axes[2].set_title(f'Crowd Density (max: {occ.max():.0f})',
                          fontsize=12, fontweight='bold')
        axes[2].set_xticks([]); axes[2].set_yticks([])

        if terminated or truncated or frame + 1 >= max_steps:
            ani.event_source.stop()
            print(f"Simulation ended at step {env.step_count}.")

    ani = FuncAnimation(fig, update, frames=max_steps, interval=120, repeat=False)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Live protest simulation visualization')
    parser.add_argument('--config', type=str, default='configs/default_scenario.yaml')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_live_episode(config_path=args.config, max_steps=args.steps, seed=args.seed)
