#!/usr/bin/env python3
"""
quick_harm_test.py - Minimal test to verify harm calculation works

Run this FIRST before full validation.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.protest_env import ProtestEnv, load_config

print("="*60)
print("QUICK HARM TEST - Verify harm calculation")
print("="*60)

config = load_config('configs/default_scenario.yaml')
env = ProtestEnv(config)
obs, info = env.reset(seed=42)

print(f"\nInitial state:")
print(f"  Agents: {len(env.agents)}")
print(f"  Gas intensity: {config['hazards']['gas']['inj_intensity']}")
print(f"  k_harm: {env.hazard_field.k_harm}")

print(f"\nRunning 30 steps...")

total_harm_cells = 0
total_harm_events = 0
first_harm_step = None

for step in range(30):
    obs, reward, terminated, truncated, info = env.step()
    
    harm_cells = info['harm_grid'].sum()
    total_harm_cells += harm_cells
    
    if harm_cells > 0:
        total_harm_events += 1
        if first_harm_step is None:
            first_harm_step = step
    
    # Log every 5 steps
    if step % 5 == 0 or harm_cells > 0:
        peak_hazard = env.hazard_field.concentration.max()
        n_moving = sum(1 for a in env.agents if a.state == 'moving')
        print(f"  Step {step:2d}: hazard={peak_hazard:6.1f}, "
              f"harm_cells={harm_cells:3d}, moving={n_moving:3d}")

print(f"\n" + "="*60)
print("RESULTS:")
print("="*60)

print(f"\nHarm Statistics:")
print(f"  First harm at step: {first_harm_step if first_harm_step else 'NEVER'}")
print(f"  Total harm events: {total_harm_events}")
print(f"  Total harm cells: {total_harm_cells}")
print(f"  Harm rate: {total_harm_events/30:.1%}")

# Detailed agent check
agents_with_harm = sum(1 for a in env.agents if a.harm_events > 0)
agents_with_cumulative = sum(1 for a in env.agents if a.cumulative_harm > 0)
print(f"\nAgent Harm:")
print(f"  Agents with harm events: {agents_with_harm}")
print(f"  Agents with cumulative harm: {agents_with_cumulative}")

if agents_with_cumulative > 0:
    max_harm = max(a.cumulative_harm for a in env.agents)
    mean_harm = np.mean([a.cumulative_harm for a in env.agents if a.cumulative_harm > 0])
    print(f"  Max cumulative harm: {max_harm:.3f}")
    print(f"  Mean cumulative harm (harmed agents): {mean_harm:.3f}")

# Gas deployment check
gas_deployments = [e for e in env.events_log if e.get('event_type') == 'gas_deployment']
print(f"\nGas Deployments: {len(gas_deployments)}")

print(f"\n" + "="*60)
if total_harm_events > 0:
    print("✅ TEST PASSED - Harm calculation working!")
    print(f"   {total_harm_events} harm events recorded")
    sys.exit(0)
else:
    print("❌ TEST FAILED - No harm events!")
    print("\nDiagnostics:")
    
    # Check if gas was deployed
    if len(gas_deployments) == 0:
        print("  ✗ No gas deployments (check police deploy_prob)")
    else:
        print(f"  ✓ {len(gas_deployments)} gas deployments")
    
    # Check peak concentration
    peak = max(env.hazard_field.concentration.flatten())
    if peak < 1.0:
        print(f"  ✗ Peak concentration too low: {peak:.2f} (need >1.0)")
    else:
        print(f"  ✓ Peak concentration: {peak:.2f}")
    
    # Check if agents near gas
    print("\nAgent-Gas Proximity Check:")
    gas_cells = np.argwhere(env.hazard_field.concentration > 1.0)
    if len(gas_cells) > 0:
        for agent in env.agents[:5]:
            if hasattr(agent, 'current_node'):
                x, y = env._node_to_cell(agent.current_node)
            else:
                x, y = agent.pos
            
            # Find nearest gas cell
            distances = [np.hypot(x - gx, y - gy) for gy, gx in gas_cells]
            min_dist = min(distances) if distances else float('inf')
            
            print(f"    Agent {agent.id} at ({x},{y}): nearest gas = {min_dist:.1f} cells")
    
    sys.exit(1)