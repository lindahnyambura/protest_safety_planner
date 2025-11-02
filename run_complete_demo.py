#!/usr/bin/env python3
"""
run_complete_demo.py - ENHANCED FR1 + FR2 Validation

NEW FEATURES:
✓ Exit rate tracking and visualization
✓ Interactive mode selection
✓ Convergence plots
✓ Reliability diagrams
✓ Per-profile statistics
✓ Comprehensive experiment logging

"""

import sys
from pathlib import Path
import numpy as np
import time
import json
import os

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.protest_env import ProtestEnv, load_config
from src.monte_carlo.aggregator import MonteCarloAggregator
from src.utils.visualization import ProtestVisualizer

# ASCII Art Header
print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║      PROTEST SAFETY PLANNER - FR1 + FR2 VALIDATION                   ║
║                                                                       ║
║    Real-time Risk Intelligence for Protest Navigation                ║
║    ─────────────────────────────────────────────────────              ║
║    Monte Carlo Aggregation | Uncertainty Quantification              ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# Configuration
# ============================================================================
print("\n[1/7]  Loading configuration...")
config_path = Path(__file__).parent / 'configs' / 'default_scenario.yaml'

if not config_path.exists():
    print(f" Config not found: {config_path}")
    sys.exit(1)

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        import yaml
        config = yaml.safe_load(f)
except Exception as e:
    print(f" Config loading failed: {e}")
    sys.exit(1)

# Display configuration
print(f" Configuration loaded")
print(f"     Grid: {config['grid']['width']}×{config['grid']['height']} "
      f"({config['grid']['cell_size_m']}m cells)")
print(f"     Agents: {config['agents']['protesters']['count']} protesters, "
      f"{config['agents']['police']['count']} police")
print(f"      Source: {config['grid'].get('obstacle_source', 'synthetic')}")

# ============================================================================
# Interactive Mode Selection
# ============================================================================
print(f"\n[2/7]  Mode selection...")
print(f"""
Available modes:
  [1]  Quick demo: 50 rollouts, single episode (~3-5 min)
  [2]  Development: 100 rollouts, single episode (~5-10 min)
  [3]  Production: 200 rollouts, full validation (~15-30 min)
  [4]  Visualization only: Load existing results
""")

mode = input("Select mode [1/2/3/4] (default=1): ").strip() or "1"

if mode == "4":
    print("\n Visualization mode not yet implemented")
    sys.exit(0)

MODE_CONFIG = {
    "1": {"name": "Quick Demo", "n_rollouts": 50, "n_bootstrap": 200, "demo_steps": 150},
    "2": {"name": "Development", "n_rollouts": 100, "n_bootstrap": 500, "demo_steps": 200},
    "3": {"name": "Production", "n_rollouts": 200, "n_bootstrap": 1000, "demo_steps": 300}
}

selected = MODE_CONFIG.get(mode, MODE_CONFIG["1"])
print(f"\n Mode: {selected['name']}")
print(f"    Rollouts: {selected['n_rollouts']}")
print(f"    Bootstrap samples: {selected['n_bootstrap']}")
print(f"    Demo episode: {selected['demo_steps']} steps")

# Update config
config['monte_carlo']['n_rollouts'] = selected['n_rollouts']
config['monte_carlo']['bootstrap_samples'] = selected['n_bootstrap']

# ============================================================================
# Environment Initialization
# ============================================================================
print(f"\n[3/7]  Initializing environment...")
start_time = time.time()

try:
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    init_time = time.time() - start_time
    
    print(f" Environment initialized ({init_time:.2f}s)")
    print(f"    Agents: {info['n_agents']}")
    print(f"    Obstacles: {env.obstacle_mask.sum():,} cells "
          f"({100*env.obstacle_mask.sum()/env.obstacle_mask.size:.1f}%)")
    
    if hasattr(env, 'osm_graph'):
        print(f"    OSM graph: {len(env.osm_graph.nodes)} nodes, "
              f"{len(env.osm_graph.edges)} edges")
    
except Exception as e:
    print(f" Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Agent heterogeneity
profile_counts = {}
for agent in env.protesters:
    profile_counts[agent.profile_name] = profile_counts.get(agent.profile_name, 0) + 1

print(f"\n     Agent Profiles:")
for profile, count in sorted(profile_counts.items()):
    pct = 100 * count / len(env.protesters)
    bar = '█' * int(pct / 5)
    print(f"      {profile:12s}: {count:3d} ({pct:5.1f}%) {bar}")

# ============================================================================
# Demo Episode with Exit Tracking
# ============================================================================
print(f"\n[4/7]  Running demonstration episode...")
print(f"    Target: {selected['demo_steps']} steps")

episode_start = time.time()
harm_timeline = []
hazard_timeline = []
exit_timeline = []  # NEW
incap_timeline = []  # NEW

try:
    for i in range(selected['demo_steps']):
        obs, reward, terminated, truncated, info = env.step(actions=None)
        
        # Track metrics
        harm_timeline.append(info['harm_grid'].sum())
        hazard_timeline.append(env.hazard_field.concentration.max())
        exit_timeline.append(sum(1 for a in env.protesters if a.state == 'safe'))
        incap_timeline.append(info['agent_states']['n_incapacitated'])
        
        # Progress updates
        if (i + 1) % 50 == 0:
            n_exited = sum(1 for a in env.protesters if a.state == 'safe')
            print(f"    Step {i+1:3d}: "
                  f"moving={info['agent_states']['n_moving']:3d}, "
                  f"exited={n_exited:3d}, "
                  f"incap={info['agent_states']['n_incapacitated']:3d}, "
                  f"harm={info['harm_grid'].sum():4d}, "
                  f"hazard={env.hazard_field.concentration.max():.1f}")
        
        if terminated or truncated:
            print(f"    Episode terminated: {info.get('termination_reason', 'unknown')}")
            break
    
    episode_time = time.time() - episode_start
    
    # Final statistics
    n_exited = sum(1 for a in env.protesters if a.state == 'safe')
    n_incap = info['agent_states']['n_incapacitated']
    n_moving = info['agent_states']['n_moving']
    
    exit_rate = n_exited / len(env.protesters)
    incap_rate = n_incap / len(env.protesters)
    
    print(f"\n Episode complete ({episode_time:.1f}s, {(i+1)/episode_time:.1f} steps/s)")
    print(f"     Final Outcomes:")
    print(f"      Exited safely: {n_exited}/{len(env.protesters)} ({exit_rate*100:.1f}%)")
    print(f"      Incapacitated: {n_incap}/{len(env.protesters)} ({incap_rate*100:.1f}%)")
    print(f"      Still moving: {n_moving}/{len(env.protesters)} ({100*n_moving/len(env.protesters):.1f}%)")
    print(f"      Hazards:")
    print(f"      Total deployments: {len([e for e in env.events_log if 'deploy' in e.get('event_type', '')])}")
    print(f"      Peak concentration: {max(hazard_timeline):.2f} mg/m³")
    print(f"      Total harm cells: {sum(harm_timeline):,}")

except Exception as e:
    print(f" Episode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Monte Carlo Aggregation
# ============================================================================
print(f"\n[5/7]  Monte Carlo aggregation...")

aggregator = MonteCarloAggregator(
    env_class=ProtestEnv,
    config=config,
    output_dir="artifacts/rollouts"
)

try:
    mc_start = time.time()
    results = aggregator.run_monte_carlo(
        base_seed=42,
        verbose=True,
        convergence_check=True
    )
    mc_time = time.time() - mc_start
    
    print(f"\n Monte Carlo complete ({mc_time:.1f}s)")
    print(f"    Average: {mc_time/selected['n_rollouts']:.2f}s/rollout")
    
    # Save results
    aggregator.save_results(results, run_id=f"{selected['name'].lower().replace(' ', '_')}_run")
    
except Exception as e:
    print(f" Monte Carlo failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Visualization Generation
# ============================================================================
print(f"\n[6/7]  Generating visualizations...")

try:
    visualizer = ProtestVisualizer(figsize_scale=1.0, dpi=150)
    
    # Create validation report
    visualizer.create_validation_report(
        env=env,
        results=results,
        output_dir="artifacts/validation"
    )
    
    # Additional time series plot
    if harm_timeline:
        visualizer.plot_time_series(
            hazard_history=hazard_timeline,
            incapacitated_timeline=incap_timeline,
            mean_harm_timeline=harm_timeline,
            save_path="artifacts/validation/05_time_series.png"
        )
    
    print(f" Visualizations complete")
    
except Exception as e:
    print(f"  Visualization warning: {e}")
    # Non-fatal

# ============================================================================
# Experiment Logging
# ============================================================================
print(f"\n[7/7]  Experiment logging...")

try:
    git_commit = os.popen('git rev-parse HEAD').read().strip() or "unknown"
    
    def make_serializable(obj):
        """Convert numpy types to Python types."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        return obj
    
    experiment_log = {
        'experiment_id': f'FR1_FR2_{selected["name"].lower().replace(" ", "_")}',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': selected['name'],
        'config_hash': results.get('config_hash', 'N/A'),
        'git_commit': git_commit,
        
        'parameters': {
            'n_rollouts': selected['n_rollouts'],
            'n_bootstrap': selected['n_bootstrap'],
            'demo_steps': selected['demo_steps'],
            'grid_size': f"{config['grid']['width']}×{config['grid']['height']}",
            'cell_size_m': config['grid']['cell_size_m'],
            'n_protesters': config['agents']['protesters']['count'],
            'n_police': config['agents']['police']['count']
        },
        
        'demo_episode': {
            'duration_s': float(episode_time),
            'final_step': int(i + 1),
            'exit_rate': float(exit_rate),
            'incapacitation_rate': float(incap_rate),
            'peak_hazard': float(max(hazard_timeline)),
            'total_harm_cells': int(sum(harm_timeline)),
            'gas_deployments': len([e for e in env.events_log if 'deploy' in e.get('event_type', '')])
        },
        
        'monte_carlo_results': make_serializable({
            'mean_harm_probability': results['p_sim'].mean(),
            'max_harm_probability': results['p_sim'].max(),
            'cells_high_risk': int((results['p_sim'] > 0.1).sum()),
            'summary': results['summary'],
            'calibration': results.get('calibration', {}),
            'spatial_analysis': results.get('spatial_analysis', {})
        }),
        
        'convergence': make_serializable(results.get('convergence', {})),
        
        'runtime': {
            'initialization_s': float(init_time),
            'episode_s': float(episode_time),
            'monte_carlo_s': float(mc_time),
            'total_s': float(init_time + episode_time + mc_time)
        }
    }
    
    log_path = Path('artifacts') / 'experiment_log.json'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f" Experiment log saved: {log_path}")
    
except Exception as e:
    print(f"  Logging warning: {e}")

# ============================================================================
# Final Summary
# ============================================================================
print(f"\n{'='*75}")
print(f" VALIDATION COMPLETE")
print(f"{'='*75}")

print(f"\n FR1: Stylized Digital Twin")
print(f"    Grid: {env.width}×{env.height} @ {env.cell_size}m/cell")
print(f"    Agents: {len(env.agents)} ({len(profile_counts)} profiles)")
print(f"    Performance: {(i+1)/episode_time:.1f} steps/s")

print(f"\n FR2: Monte Carlo Aggregator")
print(f"    Rollouts: {results['n_rollouts']}")
print(f"    Runtime: {mc_time/60:.1f} min")
print(f"    Mean p(harm): {results['p_sim'].mean():.4f}")
print(f"    Incapacitation rate: {results['summary']['incapacitation_rate']*100:.1f}%")
print(f"    Exit rate: {results['summary']['exit_rate']*100:.1f}%")

if results.get('calibration'):
    print(f"    Brier score: {results['calibration']['mean_brier']:.4f}")
    print(f"    ECE: {results['calibration']['ece']:.4f}")

if results.get('convergence', {}).get('is_converged'):
    print(f"     Converged")
else:
    print(f"      May need more rollouts")

print(f"\n Outputs:")
print(f"    Monte Carlo: artifacts/rollouts/{selected['name'].lower().replace(' ', '_')}_run/")
print(f"    Figures: artifacts/validation/")
print(f"    Log: artifacts/experiment_log.json")

print(f"\n Key Results:")
if results['summary']['exit_rate'] > 0.3:
    print(f"     Exit behavior: WORKING ({results['summary']['exit_rate']*100:.1f}%)")
else:
    print(f"      Exit behavior: LOW ({results['summary']['exit_rate']*100:.1f}%) - check agent.py fixes")

if 0.10 <= results['summary']['incapacitation_rate'] <= 0.25:
    print(f"     Incapacitation rate: REALISTIC ({results['summary']['incapacitation_rate']*100:.1f}%)")
else:
    print(f"      Incapacitation rate: {results['summary']['incapacitation_rate']*100:.1f}% "
          f"(target: 10-25%)")

print(f"\n💡 Next Steps:")
print(f"    1. Review validation figures in artifacts/validation/")
print(f"    2. Check convergence plot for MC stability")
print(f"    3. Verify exit rate ≥30% (if low, apply agent.py fixes)")
print(f"    4. Run full production mode if satisfied with dev results")

print(f"\n{'='*75}\n")