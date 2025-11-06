#!/usr/bin/env python3
"""
test_monte_carlo_compat.py - Quick test before full Monte Carlo run

Tests:
1. Single rollout completes successfully
2. Harm grids are properly accumulated
3. Aggregator can process results
4. Visualization can handle results

Ran BEFORE run_complete_demo.py to catch issues early.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.protest_env import ProtestEnv, load_config
from src.monte_carlo.aggregator import MonteCarloAggregator

print("="*60)
print("MONTE CARLO COMPATIBILITY TEST")
print("="*60)

# Load config
config = load_config('configs/default_scenario.yaml')

# Check mode
obstacle_source = config['grid'].get('obstacle_source', 'generate')
using_osm = (obstacle_source == 'nairobi')

print(f"\nConfiguration:")
print(f"  Obstacle source: {obstacle_source}")
print(f"  Mode: {'OSM graph' if using_osm else 'grid-based'}")
print(f"  Grid: {config['grid']['width']}×{config['grid']['height']}")


# TEST 1: SINGLE ENVIRONMENT ROLLOUT

print(f"\n[Test 1/4] Single environment rollout...")

try:
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    # Verify mode
    has_osm = hasattr(env, 'osm_graph') and env.osm_graph is not None
    print(f"  ✓ Environment created (mode: {'OSM' if has_osm else 'grid'})")
    
    # Run 10 steps
    harm_total = 0
    for step in range(10):
        obs, reward, term, trunc, info = env.step()
        harm_total += info['harm_grid'].sum()
    
    print(f"  ✓ Rollout completed (10 steps)")
    print(f"  ✓ Total harm events: {harm_total}")
    
    if harm_total == 0:
        print(f"  Warning: No harm events (may be OK if gas didn't reach agents)")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 2: AGGREGATOR INITIALIZATION

print(f"\n[Test 2/4] Aggregator initialization...")

try:
    # Set to minimal rollouts for testing
    config['monte_carlo']['n_rollouts'] = 4
    config['monte_carlo']['bootstrap_samples'] = 100
    config['monte_carlo']['n_jobs'] = 4  # Set to 1 for testing
    
    aggregator = MonteCarloAggregator(
        env_class=ProtestEnv,
        config=config,
        output_dir="artifacts/rollouts_test"
    )
    
    print(f"  ✓ Aggregator created")
    print(f"  ✓ Will run {aggregator.n_rollouts} rollouts")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 3: MINI MONTE CARLO (4 rollouts)

print(f"\n[Test 3/4] Mini Monte Carlo (4 rollouts)...")

try:
    results = aggregator.run_monte_carlo(base_seed=42, verbose=True)
    
    print(f"  ✓ Monte Carlo completed")
    print(f"  ✓ p_sim shape: {results['p_sim'].shape}")
    print(f"  ✓ Mean p(harm): {results['p_sim'].mean():.4f}")
    print(f"  ✓ Max p(harm): {results['p_sim'].max():.4f}")
    print(f"  ✓ High-risk cells (p>0.1): {(results['p_sim'] > 0.1).sum()}")
    
    # Verify results structure
    assert 'p_sim' in results
    assert 'p_sim_ci_lower' in results
    assert 'p_sim_ci_upper' in results
    assert 'summary' in results
    
    print(f"  ✓ Results structure valid")
    
    # Save results
    aggregator.save_results(results, run_id="test_run")
    print(f"  ✓ Results saved to artifacts/rollouts_test/")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 4: VISUALIZATION COMPATIBILITY

print(f"\n[Test 4/4] Visualization compatibility...")

try:
    from src.utils.visualization import ProtestVisualizer
    
    visualizer = ProtestVisualizer()
    
    # Test environment state plot
    fig1 = visualizer.plot_environment_state(env, save_path="artifacts/rollouts_test/test_env_state.png")
    print(f"  ✓ Environment state plot created")
    
    # Test Monte Carlo results plot
    fig2 = visualizer.plot_monte_carlo_results(results, save_path="artifacts/rollouts_test/test_monte_carlo.png")
    print(f"  ✓ Monte Carlo results plot created")
    
    # Test agent profiles plot
    fig3 = visualizer.plot_agent_profiles(env, save_path="artifacts/rollouts_test/test_agent_profiles.png")
    print(f"  ✓ Agent profiles plot created")
    
    # Test OSM map plot (if using OSM)
    if has_osm:
        fig4 = visualizer.plot_osm_map(env, save_path="artifacts/rollouts_test/test_osm_map.png")
        if fig4 is not None:
            print(f"  ✓ OSM map plot created")
        else:
            print(f"  OSM map plot skipped (no data)")
    
    print(f"  ✓ All visualizations compatible")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# SUMMARY

print(f"\n" + "="*60)
print(" ALL TESTS PASSED")
print("="*60)

print(f"\nSystem Status:")
print(f"  ✓ Environment: {'OSM graph mode' if has_osm else 'Grid mode'}")
print(f"  ✓ Monte Carlo: Working")
print(f"  ✓ Aggregator: Working")
print(f"  ✓ Visualization: Compatible")

print(f"\nNext steps:")
print(f"  1. Run: python run_complete_demo.py")
print(f"  2. Select [1] for dev (50 rollouts) or [2] for production (200 rollouts)")
print(f"  3. Results will be in artifacts/validation/")

print(f"\nExpected runtime:")
print(f"  Dev (50 rollouts):  5-10 minutes")
print(f"  Production (200):   20-40 minutes")

sys.exit(0)