#!/usr/bin/env python3
"""
validate_monte_carlo.py - Validate Monte Carlo compatibility with graph mode

UPDATED: Extended test duration to capture harm events (gas takes time to reach agents)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.protest_env import ProtestEnv, load_config
from src.utils.logging_config import logger, LogLevel

logger.set_level(LogLevel.MINIMAL)


def validate_single_rollout():
    """Test a single rollout for Monte Carlo compatibility."""
    print("\n" + "="*60)
    print("MONTE CARLO VALIDATION - Single Rollout Test")
    print("="*60)
    
    config = load_config('configs/default_scenario.yaml')
    env = ProtestEnv(config)
    
    # Test rollout
    obs, info = env.reset(seed=42)
    
    # Validation checks
    checks = {
        'Grid shape matches': (
            env.hazard_field.concentration.shape == (env.height, env.width)
        ),
        'Graph loaded': env.osm_graph is not None,
        'Cell→node mapping exists': env.cell_to_node is not None,
        'Node occupancy initialized': len(env.node_occupancy) > 0,
        'All agents have positions': all(
            hasattr(a, 'pos') or hasattr(a, 'current_node') for a in env.agents
        ),
    }
    
    print("\n✓ Initial State Checks:")
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    # ============================================================
    # CRITICAL FIX: Run 30 steps instead of 10
    # Gas takes ~20 steps to reach agents at harmful concentrations
    # ============================================================
    n_steps = 30  # Changed from 10
    harm_grids = []
    
    print(f"\n✓ Running {n_steps} steps...")
    
    for step in range(n_steps):
        obs, reward, terminated, truncated, info = env.step()
        harm_grids.append(info['harm_grid'])
        
        # Log progress every 10 steps
        if (step + 1) % 10 == 0:
            harm_cells = info['harm_grid'].sum()
            peak_hazard = env.hazard_field.concentration.max()
            print(f"    Step {step+1}: harm_cells={harm_cells}, peak_hazard={peak_hazard:.1f}")
        
        if step == 0:
            # Validate harm grid shape
            assert info['harm_grid'].shape == (env.height, env.width), \
                f"Harm grid shape mismatch: {info['harm_grid'].shape}"
    
    harm_grids = np.array(harm_grids)
    
    # Validation checks after steps
    runtime_checks = {
        'Harm grids are boolean': harm_grids.dtype == bool,
        'Harm grids have correct shape': harm_grids.shape == (n_steps, env.height, env.width),
        'Some harm occurred': harm_grids.sum() > 0,
        'Agents moved': any(
            a.pos != (0, 0) for a in env.agents
        ),
        'Node occupancy updated': len(env.node_occupancy) > 0,
    }
    
    print(f"\n✓ Runtime Checks ({n_steps} steps):")
    for check, passed in runtime_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    # Harm statistics
    print(f"\n📊 Harm Statistics:")
    print(f"  Total harm events: {harm_grids.sum()}")
    print(f"  Cells with harm: {(harm_grids.sum(axis=0) > 0).sum()}")
    print(f"  Max harm per cell: {harm_grids.sum(axis=0).max()}")
    print(f"  Steps with harm: {(harm_grids.sum(axis=(1,2)) > 0).sum()}/{n_steps}")
    
    all_passed = all(checks.values()) and all(runtime_checks.values())
    
    if all_passed:
        print("\n✅ VALIDATION PASSED - Ready for Monte Carlo")
        return True
    else:
        print("\n❌ VALIDATION FAILED - Fix issues before Monte Carlo")
        return False


def validate_determinism():
    """Test that rollouts are deterministic (same seed = same result)."""
    print("\n" + "="*60)
    print("DETERMINISM TEST - Same Seed = Same Result")
    print("="*60)
    
    config = load_config('configs/default_scenario.yaml')
    
    def run_short_rollout(seed):
        env = ProtestEnv(config)
        obs, info = env.reset(seed=seed)
        
        harm_total = 0
        # EXTENDED: 10 steps instead of 5 to capture harm
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step()
            harm_total += info['harm_grid'].sum()
        
        return harm_total
    
    # Run same seed twice
    result1 = run_short_rollout(42)
    result2 = run_short_rollout(42)
    
    print(f"\n  Rollout 1 (seed=42): {result1} harm events")
    print(f"  Rollout 2 (seed=42): {result2} harm events")
    
    if result1 == result2:
        print("\n✅ DETERMINISM PASSED - Same seed produces same result")
        return True
    else:
        print("\n❌ DETERMINISM FAILED - Results differ for same seed!")
        print("   This will break Monte Carlo reproducibility.")
        return False


def validate_independence():
    """Test that different seeds produce different results."""
    print("\n" + "="*60)
    print("INDEPENDENCE TEST - Different Seeds = Different Results")
    print("="*60)
    
    config = load_config('configs/default_scenario.yaml')
    
    results = []
    # EXTENDED: 15 steps to ensure enough time for harm
    n_steps = 15
    
    for seed in [42, 43, 44]:
        env = ProtestEnv(config)
        obs, info = env.reset(seed=seed)
        
        harm_grid = np.zeros((env.height, env.width), dtype=bool)
        for _ in range(n_steps):
            obs, reward, terminated, truncated, info = env.step()
            harm_grid |= info['harm_grid']
        
        results.append(harm_grid)
    
    # Check that results differ
    same_01 = np.array_equal(results[0], results[1])
    same_12 = np.array_equal(results[1], results[2])
    
    # Also check harm counts for more detail
    harm_counts = [r.sum() for r in results]
    
    print(f"\n  Seed 42: {harm_counts[0]} harm cells")
    print(f"  Seed 43: {harm_counts[1]} harm cells")
    print(f"  Seed 44: {harm_counts[2]} harm cells")
    
    print(f"\n  Seed 42 vs 43: {'SAME' if same_01 else 'DIFFERENT'}")
    print(f"  Seed 43 vs 44: {'SAME' if same_12 else 'DIFFERENT'}")
    
    if not same_01 and not same_12:
        print("\n✅ INDEPENDENCE PASSED - Different seeds produce different results")
        return True
    else:
        print("\n❌ INDEPENDENCE FAILED - Seeds not properly isolated")
        return False


if __name__ == "__main__":
    test1 = validate_single_rollout()
    test2 = validate_determinism()
    test3 = validate_independence()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"  Single Rollout:  {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Determinism:     {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"  Independence:    {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ALL TESTS PASSED - Monte Carlo ready!")
        print("\nNext steps:")
        print("  1. Run: python run_complete_demo.py")
        print("  2. Select option [1] for development (50 rollouts)")
        print("  3. Check artifacts/validation/ for results")
        sys.exit(0)
    else:
        print("\n⚠️ SOME TESTS FAILED - Review errors above")
        sys.exit(1)