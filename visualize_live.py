import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import numpy as np

import sys
from pathlib import Path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env import ProtestEnv, load_config


def run_live_episode(env, max_steps=500):
    """Watch episode in real-time with matplotlib animation + final summary."""
    config = load_config('configs/default_scenario.yaml')
    env = ProtestEnv(config)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    obs, info = env.reset(seed=42)
    harm_timeline = []
    hazard_history = []   # store hazard concentration grids
    max_conc_timeline = []  # per-step max hazard concentration
    incap_history = []      # per-step number of incapacitated agents
    mean_harm_history = []  # per-step mean agent harm

    def update(frame):
        nonlocal harm_timeline, hazard_history, max_conc_timeline
        nonlocal incap_history, mean_harm_history

        obs, _, term, trunc, info = env.step(None)
        harm_timeline.append(info['harm_grid'].sum())
        hazard_history.append(env.hazard_field.concentration.copy())
        max_conc_timeline.append(env.hazard_field.concentration.max())
        incap_history.append(info['agent_states']['n_incapacitated'])
        mean_harm_history.append(info['agent_states']['mean_harm'])

        # Clear and redraw
        for ax in axes:
            ax.clear()
        
        
        # Panel 1: Hazards (with annotation + higher vmax)
        
        axes[0].imshow(env.hazard_field.concentration, cmap='Reds', vmin=0, vmax=15)
        axes[0].set_title(f'Hazards (Step {frame+1}, Max: {env.hazard_field.concentration.max():.2f})')

        # Add text annotation: recent hazard events (gas, water, shooting)
        recent_events = [e for e in env.events_log[-10:] if e['event_type'] in 
                         ['gas_deployment', 'water_cannon', 'shooting']]
        axes[0].text(
            5, 5, f"Recent events: {len(recent_events)}",
            color='white', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7)
        )
        
        # Panel 2: Occupancy
        axes[2].imshow(env.occupancy_count, cmap='Blues')
        axes[2].set_title('Crowd Density')
        
        # Panel 3: Agents + obstacles
        for agent in env.agents:
            color = 'red' if agent.agent_type == 'police' else 'cyan'
            axes[1].plot(agent.pos[0], agent.pos[1], 'o', color=color, markersize=4)
        axes[1].imshow(env.obstacle_mask, cmap='gray', alpha=0.3)
        axes[1].set_title(f'Agents ({info["agent_states"]["n_moving"]} moving)')

        # Stop if environment terminates or max steps reached
        if term or trunc or frame + 1 >= max_steps:
            ani.event_source.stop()

            # Print summary after stopping
            print("\n" + "="*50)
            print("EPISODE SUMMARY")
            print("="*50)
            print(f"  Final step: {env.step_count}")
            print(f"  Total harm cells: {info['harm_grid'].sum()}")
            print(f"  Mean agent harm: {info['agent_states']['mean_harm']:.3f}")
            print(f"  Agents incapacitated: {info['agent_states']['n_incapacitated']}")

            # Hazard deployment analysis
            gas_deployments = [e for e in env.events_log if e['event_type'] == 'gas_deployment']
            water_deployments = [e for e in env.events_log if e['event_type'] == 'water_cannon']
            shootings = [e for e in env.events_log if e['event_type'] == 'shooting']
            print(f"\n  Hazard Deployment Analysis:")
            print(f"    Gas deployments: {len(gas_deployments)}")
            print(f"    Water cannons: {len(water_deployments)}")
            print(f"    Shootings: {len(shootings)}")
            print(f"    Max hazard concentration (final): {env.hazard_field.concentration.max():.2f}")

            # Harm timeline statistics
            print(f"\n  Harm Timeline:")
            print(f"    Peak harm cells: {max(harm_timeline)}")
            print(f"    Average harm cells: {sum(harm_timeline)/len(harm_timeline):.2f}")

            # Diagnostic check
            max_conc_ever = np.max([h.max() for h in hazard_history])
            steps_with_gas = sum(1 for h in hazard_history if h.max() > 1.0)
            print("\nDiagnostic Check:")
            print(f"  Max concentration ever: {max_conc_ever:.2f}")
            print(f"  Steps with visible gas (>1.0): {steps_with_gas}")

            # Generate diagnostic plots

            fig2, axs = plt.subplots(3, 1, figsize=(8, 10))

            # Plot 1: Hazard spike timeline
            axs[0].plot(max_conc_timeline, label="Max concentration per step")
            axs[0].set_title("Hazard Spike Timeline")
            axs[0].set_xlabel("Step")
            axs[0].set_ylabel("Max Concentration")
            axs[0].legend()

            # Plot 2: Incapacitated agents over time
            axs[1].plot(incap_history, color="red", label="Incapacitated agents")
            axs[1].set_title("Agents Incapacitated Over Time")
            axs[1].set_xlabel("Step")
            axs[1].set_ylabel("Count")
            axs[1].legend()

            # Plot 3: Mean harm over time
            axs[2].plot(mean_harm_history, color="orange", label="Mean harm per agent")
            axs[2].set_title("Mean Harm Per Agent Over Time")
            axs[2].set_xlabel("Step")
            axs[2].set_ylabel("Mean Harm")
            axs[2].legend()

            plt.tight_layout()
            plt.show()

        return axes
    
    ani = FuncAnimation(fig, update, frames=max_steps, interval=50, repeat=False)
    plt.show()


# Usage:
if __name__ == "__main__":
    config = load_config('configs/default_scenario.yaml')
    env = ProtestEnv(config)
    run_live_episode(env, max_steps=500)
