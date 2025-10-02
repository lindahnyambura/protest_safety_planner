#!/usr/bin/env python3
"""
run_complete_demo.py - Complete demonstration of FR1 + FR2

Demonstrates:
1. OSM Nairobi CBD map loading
2. Heterogeneous agent spawning
3. Single environment rollout
4. Monte Carlo aggregation (FR2)
5. Comprehensive visualization

Run this to validate complete implementation.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env import ProtestEnv, load_config
from src.monte_carlo.aggregator import MonteCarloAggregator
from src.utils.visualization import ProtestVisualizer

print("="*70)
print("PROTEST SAFETY PLANNER - Complete Demo")
print("FR1: Stylized Digital Twin + FR2: Monte Carlo Aggregator")
print("="*70)

# ============================================================================
# Part 1: Load Configuration
# ============================================================================
print("\n[1/5] Loading configuration...")
config_path = Path(__file__).parent / 'configs' / 'default_scenario.yaml'
config = load_config(str(config_path))

print(f"  Grid: {config['grid']['width']}x{config['grid']['height']}")
print(f"  Cell size: {config['grid']['cell_size_m']}m")
print(f"  Obstacle source: {config['grid']['obstacle_source']}")
print(f"  Protesters: {config['agents']['protesters']['count']}")
print(f"  Police: {config['agents']['police']['count']}")

# ============================================================================
# Part 2: Initialize Environment with OSM Map
# ============================================================================
print("\n[2/5] Initializing environment...")
env = ProtestEnv(config)
obs, info = env.reset(seed=42)

print(f"  Environment initialized")
print(f"  Agents spawned: {info['n_agents']}")

# Check if OSM loaded
if env.osm_metadata is not None:
    print(f"  OSM map loaded")
    print(f"    CRS: {env.osm_metadata['crs']}")
    print(f"    Coverage: {env.osm_metadata['cell_size_m']*env.width:.0f}m × "
          f"{env.osm_metadata['cell_size_m']*env.height:.0f}m")
else:
    print(f"  Using synthetic obstacles")

# Check agent heterogeneity
profile_counts = {}
for agent in env.protesters:
    profile_counts[agent.profile_name] = profile_counts.get(agent.profile_name, 0) + 1

print(f"  Agent profiles:")
for profile, count in profile_counts.items():
    print(f"    {profile}: {count} ({100*count/len(env.protesters):.1f}%)")

# ============================================================================
# Part 3: Run Single Episode Demonstration
# ============================================================================
print("\n[3/5] Running single episode demonstration...")
print("  Simulating 500 steps...")

for i in range(500):
    obs, reward, terminated, truncated, info = env.step(actions=None)
    
    if (i + 1) % 100 == 0:
        print(f"    Step {i+1}: {info['agent_states']['n_moving']} agents moving, "
              f"{info['agent_states']['n_incapacitated']} incapacitated")
    
    if terminated or truncated:
        print(f"  Episode terminated at step {i+1}: {info.get('termination_reason', 'truncated')}")
        break

print(f"   Episode complete")
print(f"    Final harm grid sum: {info['harm_grid'].sum()} cells affected")
print(f"    Mean agent harm: {info['agent_states']['mean_harm']:.2f}")

# ============================================================================
# Part 4: Monte Carlo Aggregation (FR2)
# ============================================================================
print("\n[4/5] Running Monte Carlo aggregation (FR2)...")

# Use development settings for faster demo (or production for full run)
use_dev_settings = input("  Use development settings (faster)? [Y/n]: ").strip().lower() != 'n'

if use_dev_settings:
    config['monte_carlo']['n_rollouts'] = config['monte_carlo'].get('n_rollouts_dev', 50)
    config['monte_carlo']['bootstrap_samples'] = config['monte_carlo'].get('bootstrap_samples_dev', 200)
    print(f"  Using development settings: {config['monte_carlo']['n_rollouts']} rollouts")
else:
    print(f"  Using production settings: {config['monte_carlo']['n_rollouts']} rollouts")

# Create aggregator
aggregator = MonteCarloAggregator(
    env_class=ProtestEnv,
    config=config,
    output_dir="artifacts/rollouts"
)

# Run Monte Carlo
results = aggregator.run_monte_carlo(base_seed=42, verbose=True)

# Save results
aggregator.save_results(results, run_id="demo_run")

# ============================================================================
# Part 5: Generate Visualizations
# ============================================================================
print("\n[5/5] Generating visualizations...")

visualizer = ProtestVisualizer(figsize_scale=1.0, dpi=150)

# Create validation report
visualizer.create_validation_report(
    env=env,
    results=results,
    output_dir="artifacts/validation"
)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)

print("\nFR1 Validation (Stylized Digital Twin):")
print(f"   Environment initialized with {info['n_agents']} agents")
print(f"   Grid: {env.width}x{env.height} cells, {env.cell_size}m resolution")
print(f"   Heterogeneous agents: {len(profile_counts)} profile types")
if env.osm_metadata:
    print(f"   Real OSM map: Nairobi CBD")
else:
    print(f"   Synthetic obstacles")
print(f"   Episode simulation: {env.step_count} steps")

print("\nFR2 Validation (Monte Carlo Aggregator):")
print(f"   Rollouts completed: {results['n_rollouts']}")
print(f"   Runtime: {results['runtime_seconds']:.1f}s")
print(f"   Mean harm probability: {results['p_sim'].mean():.4f}")
print(f"   High-risk cells (p>0.1): {(results['p_sim'] > 0.1).sum()}")
print(f"   Bootstrap confidence intervals computed")

print("\nOutputs:")
print(f"  Monte Carlo results: artifacts/rollouts/demo_run/")
print(f"  Validation figures: artifacts/validation/")
print(f"    - 01_environment_state.png")
print(f"    - 02_monte_carlo_results.png")
print(f"    - 03_agent_profiles.png")
if env.osm_metadata:
    print(f"    - 04_osm_map.png")

print("\nNext steps:")
print("  1. Review generated figures in artifacts/validation/")
print("  2. Check Monte Carlo outputs in artifacts/rollouts/demo_run/")
print("  3. Run tests: pytest tests/ -v")
print("  4. Ready for supervisor demo!")

print("\n" + "="*70)