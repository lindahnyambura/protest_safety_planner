#!/usr/bin/env python3
"""
diagnose_harm.py - Debug why no harm is occurring
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.protest_env import ProtestEnv, load_config

config = load_config('configs/default_scenario.yaml')
env = ProtestEnv(config)
obs, info = env.reset(seed=42)

print("\n" + "="*60)
print("HARM DIAGNOSIS")
print("="*60)

# Run until gas is deployed
for step in range(30):
    obs, reward, term, trunc, info = env.step()
    
    # Check hazard field
    hazard_peak = env.hazard_field.concentration.max()
    hazard_cells = (env.hazard_field.concentration > 0).sum()
    
    # Check agent positions
    sample_agent = env.protesters[0] if env.protesters else None
    
    if step in [5, 10, 15, 20]:
        print(f"\n[Step {step}] Diagnostics:")
        print(f"  Peak hazard: {hazard_peak:.2f}")
        print(f"  Hazard cells: {hazard_cells}")
        print(f"  Agents: {len(env.agents)}")
        
        if sample_agent:
            if hasattr(sample_agent, 'current_node'):
                # Graph mode
                node_id = sample_agent.current_node
                node_data = env.osm_graph.nodes[node_id]
                node_x_utm, node_y_utm = node_data['x'], node_data['y']
                
                # Convert to grid
                try:
                    grid_x, grid_y = env._node_to_cell(node_id)
                    hazard_at_agent = env.hazard_field.concentration[grid_y, grid_x]
                    
                    print(f"  Sample Agent {sample_agent.id}:")
                    print(f"    Node: {node_id}")
                    print(f"    Node UTM: ({node_x_utm:.1f}, {node_y_utm:.1f})")
                    print(f"    Grid cell: ({grid_x}, {grid_y})")
                    print(f"    Hazard at position: {hazard_at_agent:.4f}")
                    print(f"    Cumulative harm: {sample_agent.cumulative_harm:.4f}")
                except Exception as e:
                    print(f"  ERROR converting node to cell: {e}")
        
        # Find where hazards are deployed
        if hazard_cells > 0:
            hazard_yx = np.argwhere(env.hazard_field.concentration > 0)
            print(f"\n  Hazard locations (first 5):")
            for i, (y, x) in enumerate(hazard_yx[:5]):
                conc = env.hazard_field.concentration[y, x]
                print(f"    Cell ({x},{y}): {conc:.2f}")
        
        # Check if any agents are near hazards
        if hazard_cells > 0 and sample_agent:
            agents_near_hazard = 0
            for agent in env.agents[:10]:  # Check first 10 agents
                if hasattr(agent, 'current_node'):
                    try:
                        gx, gy = env._node_to_cell(agent.current_node)
                        if env.hazard_field.concentration[gy, gx] > 0:
                            agents_near_hazard += 1
                    except:
                        pass
            
            print(f"\n  Agents in hazardous cells: {agents_near_hazard}/10 (sample)")
        
        # Check harm grid
        harm_cells = info['harm_grid'].sum()
        if harm_cells > 0:
            print(f"\n  ✓ HARM DETECTED: {harm_cells} cells")
            break
        else:
            print(f"\n  ✗ No harm occurred this step")

print("\n" + "="*60)
print("KEY DIAGNOSTICS:")
print("="*60)

# Final check
if env.hazard_field.concentration.max() > 0:
    print("✓ Hazards are being deployed")
else:
    print("✗ No hazards deployed")

if any(hasattr(a, 'current_node') for a in env.agents):
    print("✓ Agents using graph mode")
else:
    print("✗ Agents using grid mode")

if env.cell_to_node is not None:
    print("✓ Cell→node mapping loaded")
else:
    print("✗ Cell→node mapping missing")

# Test coordinate conversion
try:
    test_node = list(env.osm_graph.nodes())[0]
    test_x, test_y = env._node_to_cell(test_node)
    print(f"✓ Node→cell conversion works (test: node {test_node} → cell ({test_x},{test_y}))")
except Exception as e:
    print(f"✗ Node→cell conversion FAILED: {e}")

# Check if _update_agent_harm is being called
print("\n" + "="*60)
print("CHECKING _update_agent_harm LOGIC:")
print("="*60)

# Manually call _update_agent_harm to see what happens
print("\nManually testing harm calculation...")
sample_agent = env.protesters[0]

if hasattr(sample_agent, 'current_node'):
    node_id = sample_agent.current_node
    try:
        x, y = env._node_to_cell(node_id)
        print(f"Agent {sample_agent.id} at node {node_id} → grid ({x},{y})")
        
        concentration = env.hazard_field.concentration[y, x]
        print(f"Hazard concentration: {concentration:.4f}")
        
        if concentration > 0:
            print("✓ Agent is in hazardous cell!")
            
            # Test harm calculation
            k_harm = env.hazard_field.k_harm
            delta_t = env.delta_t
            p_harm = 1 - np.exp(-k_harm * concentration * delta_t)
            print(f"Harm probability: {p_harm:.4f}")
            
            if p_harm > 0.01:
                print("✓ Non-trivial harm probability")
            else:
                print("✗ Harm probability too low")
        else:
            print("✗ Agent NOT in hazardous cell")
            
            # Find nearest hazard
            if env.hazard_field.concentration.max() > 0:
                hazard_yx = np.argwhere(env.hazard_field.concentration > 0)
                distances = [np.hypot(x - hx, y - hy) for hy, hx in hazard_yx]
                min_dist = min(distances) if distances else float('inf')
                print(f"Nearest hazard: {min_dist:.1f} cells away")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()