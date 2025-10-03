"""
visualization.py - Comprehensive visualization for FR1 & FR2 validation

Creates publication-quality figures for:
- Environment state (hazards, occupancy, obstacles)
- Monte Carlo results (p_sim heatmaps, confidence intervals)
- Agent trajectories and harm events
- Summary statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional, List
from pathlib import Path


class ProtestVisualizer:
    """
    Visualization suite for protest simulation.
    
    Generates publication-quality figures for validation and demonstration.
    """
    
    def __init__(self, figsize_scale: float = 1.0, dpi: int = 150):
        """
        Initialize visualizer.
        
        Args:
            figsize_scale: Scale factor for figure sizes
            dpi: Resolution for saved figures
        """
        self.figsize_scale = figsize_scale
        self.dpi = dpi
        
        # Color schemes
        self.cmap_hazard = 'Reds'
        self.cmap_occupancy = 'Blues'
        self.cmap_harm = 'YlOrRd'
        
    def plot_environment_state(self, env, save_path: Optional[str] = None):
        """
        Plot current environment state (3-panel figure).
        
        Args:
            env: ProtestEnv instance
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15 * self.figsize_scale, 5 * self.figsize_scale))
        
        # Panel 1: Hazard concentration
        im1 = axes[0].imshow(env.hazard_field.concentration, 
                            cmap=self.cmap_hazard, 
                            origin='upper',
                            vmin=0, vmax=10)
        axes[0].set_title('Hazard Concentration', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (cells)')
        axes[0].set_ylabel('Y (cells)')
        plt.colorbar(im1, ax=axes[0], label='Concentration')
        
        # Overlay obstacle mask
        obstacles = env.obstacle_mask.astype(float)
        obstacles[obstacles == 0] = np.nan
        axes[0].imshow(obstacles, cmap='gray', alpha=0.3, origin='upper')
        
        # Panel 2: Occupancy
        im2 = axes[1].imshow(env.occupancy_count, 
                            cmap=self.cmap_occupancy, 
                            origin='upper',
                            vmin=0, vmax=env.n_cell_max)
        axes[1].set_title('Agent Occupancy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (cells)')
        axes[1].set_ylabel('Y (cells)')
        plt.colorbar(im2, ax=axes[1], label='Agents per cell')
        
        # Overlay obstacles
        axes[1].imshow(obstacles, cmap='gray', alpha=0.3, origin='upper')
        
        # Panel 3: Combined view with agent positions
        combined = np.zeros((*env.hazard_field.concentration.shape, 3))
        
        # Red channel: hazards
        hazard_norm = np.clip(env.hazard_field.concentration / 10, 0, 1)
        combined[:, :, 0] = hazard_norm
        
        # Blue channel: occupancy
        occupancy_norm = np.clip(env.occupancy_count / env.n_cell_max, 0, 1)
        combined[:, :, 2] = occupancy_norm
        
        axes[2].imshow(combined, origin='upper')
        
        # Plot individual agents
        for agent in env.agents:
            color = 'red' if agent.agent_type == 'police' else 'cyan'
            marker = 's' if agent.agent_type == 'police' else 'o'
            axes[2].plot(agent.pos[0], agent.pos[1], marker, 
                        color=color, markersize=3, alpha=0.7)
        
        axes[2].set_title('Combined View', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X (cells)')
        axes[2].set_ylabel('Y (cells)')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='red', alpha=0.5, label='Hazards'),
            mpatches.Patch(color='blue', alpha=0.5, label='Crowd density'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                      markersize=8, label='Protesters'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=8, label='Police')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.suptitle(f'Environment State (Step {env.step_count})', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f" Saved environment state to {save_path}")
        
        return fig
    
    def plot_monte_carlo_results(self, results: Dict, 
                                 save_path: Optional[str] = None):
        """
        Plot Monte Carlo results (4-panel figure).
        
        Args:
            results: Output from MonteCarloAggregator.run_monte_carlo()
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12 * self.figsize_scale, 
                                                11 * self.figsize_scale))
        
        p_sim = results['p_sim']
        ci_lower = results['p_sim_ci_lower']
        ci_upper = results['p_sim_ci_upper']
        
        # Panel 1: p_sim heatmap
        im1 = axes[0, 0].imshow(p_sim, cmap=self.cmap_harm, 
                               origin='upper', vmin=0, vmax=1)
        axes[0, 0].set_title('Empirical Harm Probability (p_sim)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X (cells)')
        axes[0, 0].set_ylabel('Y (cells)')
        plt.colorbar(im1, ax=axes[0, 0], label='P(harm)')
        
        # Panel 2: Confidence interval width
        ci_width = ci_upper - ci_lower
        im2 = axes[0, 1].imshow(ci_width, cmap='viridis', origin='upper')
        axes[0, 1].set_title('95% CI Width', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('X (cells)')
        axes[0, 1].set_ylabel('Y (cells)')
        plt.colorbar(im2, ax=axes[0, 1], label='CI width')
        
        # Panel 3: Histogram of p_sim values
        axes[1, 0].hist(p_sim.flatten(), bins=50, color='coral', 
                       edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Harm Probability', fontsize=11)
        axes[1, 0].set_ylabel('Frequency (cells)', fontsize=11)
        axes[1, 0].set_title('Distribution of Harm Probabilities', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axvline(p_sim.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {p_sim.mean():.4f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 4: Summary statistics
        axes[1, 1].axis('off')
        
        summary = results['summary']
        stats_text = f"""
        Monte Carlo Summary
        {'='*40}
        
        Rollouts: {results['n_rollouts']}
        Runtime: {results['runtime_seconds']:.1f}s
        
        Harm Probabilities:
          Mean: {p_sim.mean():.4f}
          Median: {np.median(p_sim):.4f}
          Max: {p_sim.max():.4f}
          Cells with p>0.1: {(p_sim > 0.1).sum()} ({100*(p_sim > 0.1).sum()/p_sim.size:.1f}%)
          Cells with p>0.5: {(p_sim > 0.5).sum()} ({100*(p_sim > 0.5).sum()/p_sim.size:.1f}%)
        
        Episode Statistics:
          Mean length: {summary['mean_episode_length']:.1f} steps
          Std dev: {summary['std_episode_length']:.1f} steps
          Range: [{summary['min_episode_length']}, {summary['max_episode_length']}]
        
        Agent Harm:
          Mean harmed: {summary['mean_agents_harmed']:.1f} agents
          Std dev: {summary['std_agents_harmed']:.1f}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('Monte Carlo Aggregation Results (FR2)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f" Saved Monte Carlo results to {save_path}")
        
        return fig
    
    def plot_agent_profiles(self, env, save_path: Optional[str] = None):
        """
        Plot agent heterogeneity (profile distribution).
        
        Args:
            env: ProtestEnv instance
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15 * self.figsize_scale, 
                                                4 * self.figsize_scale))
        
        # Count agent types
        profile_counts = {}
        speeds = {'cautious': [], 'average': [], 'bold': []}
        risk_tolerances = {'cautious': [], 'average': [], 'bold': []}
        
        for agent in env.protesters:
            profile = agent.profile_name
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
            speeds[profile].append(agent.speed)
            risk_tolerances[profile].append(agent.risk_tolerance)
        
        # Panel 1: Profile distribution
        profiles = list(profile_counts.keys())
        counts = list(profile_counts.values())
        colors = ['lightblue', 'steelblue', 'darkblue']
        
        axes[0].bar(profiles, counts, color=colors, edgecolor='black', alpha=0.7)
        axes[0].set_ylabel('Number of agents', fontsize=11)
        axes[0].set_title('Agent Profile Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, (profile, count) in enumerate(zip(profiles, counts)):
            axes[0].text(i, count + 2, f'{count}\n({100*count/len(env.protesters):.1f}%)', 
                        ha='center', fontsize=10)
        
        # Panel 2: Speed distribution
        positions = range(len(profiles))
        bp1 = axes[1].boxplot([speeds[p] for p in profiles], 
                              positions=positions,
                              labels=profiles,
                              patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        axes[1].set_ylabel('Speed (cells/step)', fontsize=11)
        axes[1].set_title('Speed Distribution by Profile', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Panel 3: Risk tolerance distribution
        bp2 = axes[2].boxplot([risk_tolerances[p] for p in profiles], 
                              positions=positions,
                              labels=profiles,
                              patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        axes[2].set_ylabel('Risk tolerance', fontsize=11)
        axes[2].set_title('Risk Tolerance by Profile', fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Agent Heterogeneity Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f" Saved agent profiles to {save_path}")
        
        return fig
    
    def plot_time_series(self, hazard_history: List[float],
                         incapacitated_timeline: List[int],
                         mean_harm_timeline: List[float],
                         save_path: Optional[str] = None):
        """
        Plot key time series diagnostics:
        - Hazard concentration spikes
        - Number of incapacitated agents
        - Mean harm per agent

        Args:
            hazard_history: Max hazard concentration per step
            incapacitated_timeline: Number of incapacitated agents per step
            mean_harm_timeline: Mean harm per agent per step
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(10 * self.figsize_scale, 10 * self.figsize_scale),
                                sharex=True)

        # 1. Hazard spikes
        axes[0].plot(hazard_history, label="Max hazard concentration")
        axes[0].set_ylabel("Hazard level")
        axes[0].set_title("Hazard concentration spikes over time", fontsize=12, fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 2. Incapacitated agents
        axes[1].plot(incapacitated_timeline, color="red", label="Incapacitated agents")
        axes[1].set_ylabel("Number of agents")
        axes[1].set_title("Incapacitated agents over time", fontsize=12, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # 3. Mean harm
        axes[2].plot(mean_harm_timeline, color="orange", label="Mean harm per agent")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Mean harm")
        axes[2].set_title("Mean harm per agent over time", fontsize=12, fontweight="bold")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.suptitle("Simulation Time Series Diagnostics", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f" Saved time series plots to {save_path}")

        return fig

    def plot_osm_map(self, env, save_path: Optional[str] = None):
        """
        Plot OSM map with building footprints.
        
        Args:
            env: ProtestEnv instance (must have OSM data loaded)
            save_path: Path to save figure
        """
        if env.osm_metadata is None:
            print("Warning: No OSM data loaded in environment")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14 * self.figsize_scale, 
                                                6 * self.figsize_scale))
        
        # Panel 1: Building footprints
        if env.buildings_gdf is not None:
            env.buildings_gdf.plot(ax=axes[0], color='gray', edgecolor='black', alpha=0.6)
            axes[0].set_title('OSM Building Footprints (UTM)', 
                             fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Easting (m)')
            axes[0].set_ylabel('Northing (m)')
            axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Rasterized obstacle grid
        axes[1].imshow(env.obstacle_mask, cmap='gray', origin='upper')
        axes[1].set_title('Rasterized Obstacle Grid', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X (cells)')
        axes[1].set_ylabel('Y (cells)')
        
        # Add grid info
        metadata = env.osm_metadata
        info_text = f"Grid: {metadata['width']}×{metadata['height']}\n" \
                   f"Cell size: {metadata['cell_size_m']}m\n" \
                   f"Coverage: {metadata['cell_size_m']*metadata['width']:.0f}m × " \
                   f"{metadata['cell_size_m']*metadata['height']:.0f}m\n" \
                   f"CRS: {metadata['crs']}"
        axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Nairobi CBD Map (OpenStreetMap)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f" Saved OSM map to {save_path}")
        
        return fig
    
    def create_validation_report(self, env, results: Dict, 
                                output_dir: str = "artifacts/validation"):
        """
        Create comprehensive validation report with all figures.
        
        Args:
            env: ProtestEnv instance
            results: Monte Carlo results
            output_dir: Directory to save figures
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating validation report...")
        print(f"Output directory: {output_dir}")
        
        # 1. Environment state
        self.plot_environment_state(env, save_path=output_dir / "01_environment_state.png")
        plt.close()
        
        # 2. Monte Carlo results
        self.plot_monte_carlo_results(results, save_path=output_dir / "02_monte_carlo_results.png")
        plt.close()
        
        # 3. Agent profiles
        self.plot_agent_profiles(env, save_path=output_dir / "03_agent_profiles.png")
        plt.close()
        
        # 4. OSM map (if available)
        if env.osm_metadata is not None:
            self.plot_osm_map(env, save_path=output_dir / "04_osm_map.png")
            plt.close()
        
        print(f"\n Validation report complete!")
        print(f"  Generated {3 + (1 if env.osm_metadata else 0)} figures")
        print(f"  Location: {output_dir}/")

        # 5. Time series diagnostics (if available)
        if 'hazard_history' in results:
            self.plot_time_series(
                hazard_history=results['hazard_history'],
                incapacitated_timeline=results['incapacitated_timeline'],
                mean_harm_timeline=results['mean_harm_timeline'],
                save_path=output_dir / "05_time_series.png"
            )
            plt.close()


def quick_visualize_environment(env, title: str = "Environment State"):
    """
    Quick visualization for debugging (single figure).
    
    Args:
        env: ProtestEnv instance
        title: Figure title
    """
    visualizer = ProtestVisualizer()
    fig = visualizer.plot_environment_state(env)
    plt.show()
    return fig


def quick_visualize_monte_carlo(results: Dict):
    """
    Quick visualization of Monte Carlo results.
    
    Args:
        results: Output from MonteCarloAggregator
    """
    visualizer = ProtestVisualizer()
    fig = visualizer.plot_monte_carlo_results(results)
    plt.show()
    return fig