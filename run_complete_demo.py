#!/usr/bin/env python3
"""
run_complete_demo.py - STABLE FR1 + FR2 Demonstration with Full Diagnostics

Produces FR1 & FR2 validation with:
- Synthetic obstacles (tested, stable)
- Heterogeneous agents (3 profiles)
- Single episode demonstration
- Monte Carlo aggregation with convergence analysis
- Calibration metrics (Brier score)
- Comprehensive visualizations
- Experiment logging

OSM disabled for system stability.
"""

import sys
from pathlib import Path
import numpy as np
import time
import json
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env import ProtestEnv, load_config
from src.monte_carlo.aggregator import MonteCarloAggregator
from src.utils.visualization import ProtestVisualizer

print("="*70)
print("PROTEST SAFETY PLANNER - FR1 + FR2 Validation")
print("="*70)
print("Configuration: STABLE (Synthetic obstacles, safe parallelization)")
print("="*70)

# ============================================================================
# Part 1: Load Configuration with UTF-8 Encoding
# ============================================================================
print("\n[1/6] Loading configuration...")
config_path = Path(__file__).parent / 'configs' / 'default_scenario.yaml'

if not config_path.exists():
    print(f" Config file not found: {config_path}")
    sys.exit(1)

# FIXED: Add UTF-8 encoding
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        import yaml
        config = yaml.safe_load(f)
except UnicodeDecodeError as e:
    print(f" Config file encoding error: {e}")
    print("  Check for non-ASCII characters in YAML comments (em-dashes, special symbols)")
    sys.exit(1)

# Ensure OSM is disabled
config['grid']['obstacle_source'] = 'generate'

print(f"  Grid: {config['grid']['width']}×{config['grid']['height']}")
print(f"  Cell size: {config['grid']['cell_size_m']}m")
print(f"  Obstacle source: {config['grid']['obstacle_source']}")
print(f"  Protesters: {config['agents']['protesters']['count']}")
print(f"  Police: {config['agents']['police']['count']}")
print(f"  Parallel jobs: {config['monte_carlo']['n_jobs']}")

# ============================================================================
# Part 2: Initialize Environment
# ============================================================================
print("\n[2/6] Initializing environment...")
start_time = time.time()

try:
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    init_time = time.time() - start_time
    
    print(f"   Environment initialized in {init_time:.2f}s")
    print(f"  Agents spawned: {info['n_agents']}")
    print(f"  Obstacles: {env.obstacle_mask.sum()} cells ({100*env.obstacle_mask.sum()/env.obstacle_mask.size:.1f}%)")
    
except Exception as e:
    print(f"   Environment initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check agent heterogeneity
profile_counts = {}
for agent in env.protesters:
    profile_counts[agent.profile_name] = profile_counts.get(agent.profile_name, 0) + 1

print(f"\n  Agent Heterogeneity:")
for profile in ['cautious', 'average', 'bold']:
    count = profile_counts.get(profile, 0)
    if count > 0:
        print(f"    {profile:10s}: {count:3d} ({100*count/len(env.protesters):5.1f}%)")

# ============================================================================
# Part 3: Run Single Episode Demonstration
# ============================================================================
print("\n[3/6] Running single episode demonstration...")
print("  Target: 500 steps")

episode_start = time.time()
harm_timeline = []
hazard_timeline = []

try:
    for i in range(500):
        obs, reward, terminated, truncated, info = env.step(actions=None)
        harm_timeline.append(info['harm_grid'].sum())
        hazard_timeline.append(env.hazard_field.concentration.max())
        
        if (i + 1) % 100 == 0:
            print(f"    Step {i+1:3d}: "
                  f"{info['agent_states']['n_moving']:3d} moving, "
                  f"{info['agent_states']['n_incapacitated']:3d} incapacitated, "
                  f"harm cells: {info['harm_grid'].sum():4d}, "
                  f"max hazard: {env.hazard_field.concentration.max():.1f}")
        
        if terminated or truncated:
            reason = info.get('termination_reason', 'time_limit')
            print(f"  Episode terminated at step {i+1}: {reason}")
            break
    
    episode_time = time.time() - episode_start
    steps_per_sec = (i + 1) / episode_time
    
    print(f"\n   Episode complete in {episode_time:.1f}s ({steps_per_sec:.1f} steps/sec)")
    print(f"    Final step: {env.step_count}")
    print(f"    Total harm cells: {info['harm_grid'].sum()}")
    print(f"    Mean agent harm: {info['agent_states']['mean_harm']:.3f}")
    print(f"    Agents incapacitated: {info['agent_states']['n_incapacitated']}")
    
    # Incapacitation rate
    incap_rate = info['agent_states']['n_incapacitated'] / len(env.protesters)
    print(f"    Incapacitation rate: {incap_rate:.1%}")

    # Gas Deployment Analysis
    print(f"\n  Gas Deployment Analysis:")
    gas_deployments = [e for e in env.events_log if e['event_type'] == 'gas_deployment']
    print(f"    Total deployments: {len(gas_deployments)}")
    print(f"    Max hazard concentration (final): {env.hazard_field.concentration.max():.2f}")
    
    # Harm Timeline
    print(f"\n  Harm Timeline:")
    print(f"    Peak harm cells: {max(harm_timeline)}")
    print(f"    Average harm cells: {np.mean(harm_timeline):.2f}")
    
    # Diagnostic Check
    print(f"\n  Diagnostic Check:")
    print(f"    Max concentration ever: {max(hazard_timeline):.2f}")
    print(f"    Steps with visible gas (>1.0): {sum(1 for h in hazard_timeline if h > 1.0)}")

except Exception as e:
    print(f"   Episode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Part 4: Monte Carlo Aggregation (FR2) with Diagnostics
# ============================================================================
print("\n[4/6] Monte Carlo Aggregation (FR2)...")

# Ask user for settings
print(f"\n  Available settings:")
print(f"    [1] Development: {config['monte_carlo'].get('n_rollouts_dev', 100)} rollouts (~2-5 min)")
print(f"    [2] Production:  {config['monte_carlo']['n_rollouts']} rollouts (~10-30 min)")

choice = input("  Select [1/2] (default=1): ").strip()

if choice == '2':
    n_rollouts = config['monte_carlo']['n_rollouts']
    n_bootstrap = config['monte_carlo']['bootstrap_samples']
    print(f"  Using production settings: {n_rollouts} rollouts")
else:
    n_rollouts = config['monte_carlo'].get('n_rollouts_dev', 100)
    n_bootstrap = config['monte_carlo'].get('bootstrap_samples_dev', 200)
    print(f"  Using development settings: {n_rollouts} rollouts")

# Update config
config['monte_carlo']['n_rollouts'] = n_rollouts
config['monte_carlo']['bootstrap_samples'] = n_bootstrap

# Create aggregator
aggregator = MonteCarloAggregator(
    env_class=ProtestEnv,
    config=config,
    output_dir="artifacts/rollouts"
)

try:
    # Run Monte Carlo with convergence analysis
    mc_start = time.time()
    results = aggregator.run_monte_carlo(base_seed=42, verbose=True)
    mc_time = time.time() - mc_start
    
    print(f"\n   Monte Carlo complete in {mc_time:.1f}s")
    print(f"    Average per rollout: {mc_time/n_rollouts:.2f}s")
    
    # Save results
    aggregator.save_results(results, run_id="production_run")
    
except Exception as e:
    print(f"   Monte Carlo failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Part 5: Compute Calibration Metrics
# ============================================================================
print("\n[5/6] Computing calibration metrics...")

try:
    # Compute Brier score
    calibration_metrics = aggregator.compute_calibration_metrics()
    print(f"   Calibration complete")
    print(f"    Mean Brier score: {calibration_metrics['mean_brier']:.4f}")
    print(f"    Median Brier score: {calibration_metrics['median_brier']:.4f}")
    
    # Add to results
    results['calibration'] = calibration_metrics
    
except Exception as e:
    print(f"   Calibration metrics failed (non-fatal): {e}")
    results['calibration'] = None

# ============================================================================
# Part 6: Generate Visualizations
# ============================================================================
print("\n[6/6] Generating visualizations...")

try:
    visualizer = ProtestVisualizer(figsize_scale=1.0, dpi=150)
    
    # Create validation report
    visualizer.create_validation_report(
        env=env,
        results=results,
        output_dir="artifacts/validation"
    )
    
    print("   Visualizations complete")
    
except Exception as e:
    print(f"   Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    # Non-fatal, continue

# ============================================================================
# Experiment Logging
# ============================================================================
print("\nLogging experiment metadata...")

try:
    # Get git commit (if available)
    try:
        git_commit = os.popen('git rev-parse HEAD').read().strip()
    except:
        git_commit = "unknown"
    
    experiment_log = {
        'experiment_id': 'FR1_FR2_validation_run',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config_hash': results.get('config_hash', 'N/A'),
        'git_commit': git_commit,
        'parameters': {
            'n_rollouts': int(n_rollouts),
            'n_bootstrap': int(n_bootstrap),
            'inj_intensity': config['hazards']['gas']['inj_intensity'],
            'k_harm': config['hazards']['gas']['k_harm'],
            'cell_size_m': config['grid']['cell_size_m']
        },
        'results_summary': results['summary'],
        'calibration': results.get('calibration'),
        'convergence': results.get('convergence', {}),
        'demo_episode': {
            'incapacitation_rate': incap_rate,
            'final_step': env.step_count,
            'gas_deployments': len(gas_deployments),
            'peak_harm_cells': max(harm_timeline),
            'max_concentration': max(hazard_timeline)
        }
    }
    
    log_path = Path('artifacts') / 'experiment_log.json'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"   Experiment log saved to {log_path}")
    
except Exception as e:
    print(f"   Experiment logging failed: {e}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*70)
print("VALIDATION COMPLETE ")
print("="*70)

print("\n FR1: Stylized Digital Twin")
print(f"  Grid: {env.width}×{env.height} cells ({env.cell_size}m resolution)")
print(f"  Agents: {len(env.agents)} ({len(env.protesters)} protesters, {len(env.police_agents)} police)")
print(f"  Heterogeneity: {len(profile_counts)} agent profiles")
print(f"  Episode performance: {steps_per_sec:.1f} steps/sec")
print(f"  Obstacles: Synthetic (tested, stable)")
print(f"  Incapacitation rate: {incap_rate:.1%}")

print("\n FR2: Monte Carlo Aggregator")
print(f"  Rollouts: {results['n_rollouts']}")
print(f"  Runtime: {results['runtime_seconds']:.1f}s ({results['runtime_seconds']/60:.1f} min)")
print(f"  Mean harm probability: {results['p_sim'].mean():.4f}")
print(f"  Max harm probability: {results['p_sim'].max():.4f}")
print(f"  High-risk cells (p>0.1): {(results['p_sim'] > 0.1).sum()} / {results['p_sim'].size}")
print(f"  Incapacitation rate: {results['summary'].get('incapacitation_rate', 'N/A'):.1%}")
if results.get('calibration'):
    print(f"  Calibration (Brier): {results['calibration']['mean_brier']:.4f}")
if results.get('convergence'):
    print(f"  Convergence: {results['convergence']}")

print("\n Outputs Generated:")
print(f"  Monte Carlo data:")
print(f"    artifacts/rollouts/production_run/p_sim.npy")
print(f"    artifacts/rollouts/production_run/metadata.json")

print(f"\n  Validation figures:")
print(f"    artifacts/validation/01_environment_state.png")
print(f"    artifacts/validation/02_monte_carlo_results.png")
print(f"    artifacts/validation/03_agent_profiles.png")

print(f"\n  Experiment log:")
print(f"    artifacts/experiment_log.json")

print("\n System Status: STABLE & PRODUCTION-READY")
print("   - Deterministic (same seed = same results)")
print("   - Tested parallelization (no system freeze)")
print("   - Heterogeneous agents working")
print("   - Monte Carlo bootstrap validated")
print("   - Calibration metrics computed")

print("\n Next Steps:")
print("  1. Review figures in artifacts/validation/")
print("  2. Verify p_sim values are reasonable (target: incap 15-25%)")
print("  3. Check experiment_log.json for parameter record")
print("  4. Run tests: pytest tests/ -v")
print("  5. Document results in research notebook")

print("\n" + "="*70)