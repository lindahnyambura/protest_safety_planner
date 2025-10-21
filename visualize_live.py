#!/usr/bin/env python3
"""
visualize_live.py - Real-time episode visualization with diagnostics

Features:
- Live matplotlib animation of environment state
- Agent state color coding (moving/stunned/incapacitated)
- Hazard overlay with transparency
- Post-episode diagnostic plots
- Event tracking and summary
"""

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env import ProtestEnv, load_config
from src.env.agent import AgentState  # CRITICAL: Import for state checks


def run_live_episode(config_path='configs/default_scenario.yaml', max_steps=100, seed=42):
    """
    Run episode with live visualization and post-episode diagnostics.
    
    Args:
        config_path: Path to YAML config
        max_steps: Maximum simulation steps
        seed: Random seed for reproducibility
    """
    # Load config with UTF-8 encoding
    config = load_config(config_path)
    env = ProtestEnv(config)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Live Protest Simulation', fontsize=16, fontweight='bold')

    plt.style.use('seaborn-v0_8-whitegrid')
    for ax in axes:
        ax.set_facecolor('#f7f7f7')

    
    # Initialize environment
    obs, info = env.reset(seed=seed)
    
    # Tracking variables
    harm_timeline = []
    hazard_history = []
    max_conc_timeline = []
    incap_history = []
    mean_harm_history = []
    
    # Exit locations for visualization
    exits = config['agents']['protesters']['goals']['exit_points']

    def update(frame):
        """Animation update function."""
        nonlocal harm_timeline, hazard_history, max_conc_timeline
        nonlocal incap_history, mean_harm_history

        obs, _, terminated, truncated, info = env.step(None)

        # Track metrics
        harm_timeline.append(info['harm_grid'].sum())
        hazard_history.append(env.hazard_field.concentration.copy())
        max_conc_timeline.append(env.hazard_field.concentration.max())
        incap_history.append(info['agent_states']['n_incapacitated'])
        mean_harm_history.append(info['agent_states']['mean_harm'])

        for ax in axes:
            ax.clear()

        # ===== PANEL 1: HAZARD FIELD =====
        hazard_display = env.hazard_field.concentration.copy()
        im1 = axes[0].imshow(
            hazard_display,
            cmap='Reds',
            vmin=0,
            vmax=20,
            origin='upper',
            alpha=0.8,
            interpolation='bilinear'
        )

        # Overlay obstacles (light gray)
        obstacles_display = env.obstacle_mask.astype(float)
        obstacles_display[obstacles_display == 0] = np.nan
        axes[0].imshow(obstacles_display, cmap='gray', alpha=0.3, origin='upper')

        # Overlay OSM roads if available
        if hasattr(env, "roads_all") and env.roads_all is not None:
            roads_all = env.roads_all

            # Separate main vs local for color
            if "road_type" in roads_all.columns:
                main_roads = roads_all[roads_all["road_type"] == "main"]
                local_roads = roads_all[roads_all["road_type"] == "local"]
            else:
                # Fallback: all roads same style
                main_roads, local_roads = roads_all, None

            # Plot roads on Panel 1
            if local_roads is not None and not local_roads.empty:
                local_roads.plot(ax=axes[0], color="gray", linewidth=0.4, alpha=0.7, label="Local roads", zorder=3)
            if main_roads is not None and not main_roads.empty:
                main_roads.plot(ax=axes[0], color="red", linewidth=1.0, alpha=0.9, label="Main roads", zorder=4)


        # Green translucent overlays for exits
        for exit_pos in exits:
            rect = mpatches.Rectangle(
                (exit_pos[0] - 0.5, exit_pos[1] - 0.5),
                1, 1,
                color='limegreen',
                alpha=0.25,
                zorder=2
            )
            axes[0].add_patch(rect)

        axes[0].set_title(
            f'Hazard Field (Step {frame+1})\nMax: {hazard_display.max():.1f} mg/m³',
            fontsize=12, fontweight='bold'
        )
        axes[0].set_xlabel('X (cells)')
        axes[0].set_ylabel('Y (cells)')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # ===== PANEL 2: AGENT VISUALIZATION =====
        axes[1].imshow(env.obstacle_mask, cmap='gray', alpha=0.15, origin='upper')

        # Green translucent exit areas
        for exit_pos in exits:
            rect = mpatches.Rectangle(
                (exit_pos[0] - 1.5, exit_pos[1] - 1.5),
                3, 3,
                facecolor='limegreen',
                alpha=0.2,
                edgecolor='green',
                linestyle='--',
                linewidth=1.0,
                zorder=1
            )
            axes[1].add_patch(rect)
            axes[1].text(
                exit_pos[0], exit_pos[1], 'EXIT',
                color='green', fontsize=8, fontweight='bold',
                ha='center', va='center', alpha=0.8, zorder=3
            )

        # --- Equal-sized agents ---
        marker_size = 20  # consistent for all
        for agent in env.protesters:
            if agent.state == AgentState.MOVING:
                color = 'deepskyblue'
            elif agent.state == AgentState.STUNNED:
                color = 'gold'
            elif agent.state == AgentState.INCAPACITATED:
                color = 'dimgray'
            elif agent.state == AgentState.SAFE:
                color = 'limegreen'
            else:
                color = 'lightgray'

            axes[1].scatter(
                agent.pos[0], agent.pos[1],
                s=marker_size, c=color,
                alpha=0.85, edgecolors='none', zorder=2
            )

        # Police (same size, just outlined)
        for agent in env.police_agents:
            axes[1].scatter(
                agent.pos[0], agent.pos[1],
                s=marker_size, c='crimson',
                alpha=0.9, edgecolors='black',
                linewidths=0.4, zorder=3
            )

        axes[1].set_xlim(0, env.width)
        axes[1].set_ylim(env.height, 0)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title(
            f'Agents (Step {frame+1})\n'
            f'Moving: {info["agent_states"]["n_moving"]}, '
            f'Incap: {info["agent_states"]["n_incapacitated"]}',
            fontsize=12, fontweight='bold'
        )

        # --- Legend ---
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='deepskyblue',
                        markersize=8, label='Moving'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                        markersize=8, label='Stunned'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='dimgray',
                        markersize=8, label='Incapacitated'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                        markersize=8, label='Safe'),
            plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='crimson',
                        markersize=8, label='Police'),
            plt.Line2D([0], [0], marker='s', color='green', markerfacecolor='limegreen',
                        markersize=8, label='Exit Zone')
        ]

        axes[1].legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=6,
            fontsize=9,
            frameon=False
        )

        # ===== PANEL 3: CROWD DENSITY =====
        occ = env.occupancy_count
        im3 = axes[2].imshow(
            occ,
            cmap='viridis',
            vmin=0,
            vmax=max(5, occ.max()),
            origin='upper',
            interpolation='nearest'
        )

        axes[2].imshow(obstacles_display, cmap='gray', alpha=0.25, origin='upper')

        for exit_pos in exits:
            rect = mpatches.Rectangle(
                (exit_pos[0] - 0.5, exit_pos[1] - 0.5),
                1, 1,
                color='limegreen',
                alpha=0.25,
                zorder=2
            )
            axes[2].add_patch(rect)

        axes[2].set_title(
            f'Crowd Density (max: {occ.max():.0f})',
            fontsize=12, fontweight='bold'
        )
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        # Fixed colorbar that updates properly
        if not hasattr(update, 'colorbar'):
            update.colorbar = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            update.colorbar.set_label('Agents per cell')

        update.colorbar.update_normal(im3)

        # Stop animation if done
        if terminated or truncated or frame + 1 >= max_steps:
            ani.event_source.stop()
            print_episode_summary(env, info, harm_timeline, hazard_history,
                                max_conc_timeline, incap_history, mean_harm_history)
            plot_diagnostics(max_conc_timeline, incap_history, mean_harm_history)



    ani = FuncAnimation(fig, update, frames=max_steps, interval=100, repeat=False)
    plt.tight_layout()
    plt.show()

def print_episode_summary(env, info, harm_timeline, hazard_history, 
                         max_conc_timeline, incap_history, mean_harm_history):
    """Print comprehensive episode summary."""
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    print(f"Final step: {env.step_count}")
    print(f"Total harm cells: {info['harm_grid'].sum()}")
    print(f"Mean agent harm: {info['agent_states']['mean_harm']:.3f}")
    print(f"Agents incapacitated: {info['agent_states']['n_incapacitated']}")
    
    # Incapacitation rate
    incap_rate = info['agent_states']['n_incapacitated'] / len(env.protesters)
    print(f"Incapacitation rate: {incap_rate:.1%}")

    # Relocation diagnostics
    relocations = [e for e in env.events_log if e['event_type'] == 'relocation_due_to_overcrowding']
    print(f"\nCrowd Control Diagnostics:")
    print(f"  Total relocations due to overcrowding: {len(relocations)}")
    
    # Event analysis
    gas_deployments = [e for e in env.events_log if e['event_type'] == 'gas_deployment']
    water_deployments = [e for e in env.events_log if e['event_type'] == 'water_cannon']
    shootings = [e for e in env.events_log if e['event_type'] == 'shooting']
    
    print(f"\nGas Deployment Analysis:")
    print(f"  Total deployments: {len(gas_deployments)}")
    print(f"  Water cannons: {len(water_deployments)}")
    print(f"  Shootings: {len(shootings)}")
    print(f"  Max hazard concentration (final): {env.hazard_field.concentration.max():.2f}")

    # Timeline statistics
    print(f"\nHarm Timeline:")
    print(f"  Peak harm cells: {max(harm_timeline)}")
    print(f"  Average harm cells: {np.mean(harm_timeline):.2f}")

    # Diagnostic check
    max_conc_ever = max(max_conc_timeline)
    steps_with_gas = sum(1 for c in max_conc_timeline if c > 1.0)
    print(f"\nDiagnostic Check:")
    print(f"  Max concentration ever: {max_conc_ever:.2f}")
    print(f"  Steps with visible gas (>1.0): {steps_with_gas}")
    
    # Exit reach analysis
    if hasattr(env, "protesters"):
        reached_exits = {}
        for a in env.protesters:
            if getattr(a, "state", None) == AgentState.SAFE:
                g = tuple(a.goal)
                reached_exits[g] = reached_exits.get(g, 0) + 1
        print(f"\nAgents who reached each exit:")
        for g, count in reached_exits.items():
            print(f"  Exit {g}: {count}")

    print("="*60)


def plot_diagnostics(max_conc_timeline, incap_history, mean_harm_history):
    """Generate post-episode diagnostic plots."""
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    fig.suptitle('Episode Diagnostics', fontsize=16, fontweight='bold')

    # Plot 1: Hazard concentration timeline
    axs[0].plot(max_conc_timeline, linewidth=2.2, label="Max concentration")
    axs[0].fill_between(range(len(max_conc_timeline)), 0, max_conc_timeline, alpha=0.25)
    axs[0].set_title("Hazard Concentration Over Time", fontsize=13, fontweight='bold')
    axs[0].set_ylabel("Max Conc. (mg/m³)")
    axs[0].legend(frameon=False)
    axs[0].grid(alpha=0.4)


    # Plot 2: Incapacitated agents
    axs[1].plot(incap_history, linewidth=2.2, label="Incapacitated agents")
    axs[1].fill_between(range(len(incap_history)), 0, incap_history, alpha=0.25)
    axs[1].set_title("Incapacitated Agents Over Time", fontsize=13, fontweight='bold')
    axs[1].set_ylabel("Count")
    axs[1].legend(frameon=False)
    axs[1].grid(alpha=0.4)

    # Plot 3: Mean harm accumulation
    axs[2].plot(mean_harm_history, linewidth=2.2, label="Mean harm per agent", color='darkorange')
    axs[2].fill_between(range(len(mean_harm_history)), 0, mean_harm_history, alpha=0.25, color='orange')
    axs[2].set_title("Mean Harm Per Agent Over Time", fontsize=13, fontweight='bold')
    axs[2].set_xlabel("Simulation Step")
    axs[2].set_ylabel("Mean Harm")
    axs[2].legend(frameon=False)
    axs[2].grid(alpha=0.4)

    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Live protest simulation visualization')
    parser.add_argument('--config', type=str, default='configs/default_scenario.yaml',
                       help='Path to config file')
    parser.add_argument('--steps', type=int, default=100,
                       help='Maximum steps to simulate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    run_live_episode(config_path=args.config, max_steps=args.steps, seed=args.seed)