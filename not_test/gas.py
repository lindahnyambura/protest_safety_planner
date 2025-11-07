"""
test_gas_deployment.py - Validate gas deployment and diffusion

Run after fixing agent.py and hazards.py
"""

import numpy as np
import yaml
from src.env.protest_env import ProtestEnv

def test_gas_deployment():
    """Test gas deployment in real Nairobi environment."""
    
    # Load config
    with open('configs/default_scenario.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    print("="*60)
    print("GAS DEPLOYMENT VALIDATION TEST")
    print("="*60)
    
    # Test 1: Verify obstacle mask coverage
    obstacle_coverage = 100 * env.obstacle_mask.sum() / env.obstacle_mask.size
    print(f"\nObstacle Coverage: {obstacle_coverage:.1f}%")
    
    # Test 2: Find valid deployment locations
    valid_cells = ~env.obstacle_mask
    n_valid = valid_cells.sum()
    print(f"Valid deployment cells: {n_valid}/{env.width * env.height}")
    
    # Test 3: Deploy gas at center (should relocate if obstacle)
    test_x, test_y = env.width // 2, env.height // 2
    
    print(f"\n[TEST] Attempting deployment at center ({test_x},{test_y})")
    print(f"  Is obstacle? {env.obstacle_mask[test_y, test_x]}")
    
    # Manually trigger deployment
    if env.police_agents:
        police = env.police_agents[0]
        police.pos = (test_x, test_y)
        police._attempt_gas_deployment(env)
    
    # Verify sources were added
    print(f"\n[RESULT] Active gas sources: {len(env.hazard_field.active_sources)}")
    
    if env.hazard_field.active_sources:
        for i, src in enumerate(env.hazard_field.active_sources):
            sx, sy = src['x'], src['y']
            is_obstacle = env.obstacle_mask[sy, sx]
            print(f"  Source {i}: ({sx},{sy}) - On obstacle? {is_obstacle}")
            
            if is_obstacle:
                print(f"    [ERROR] Source deployed on obstacle!")
                return False
    
    # Test 4: Run simulation for 10 steps
    print("\n[TEST] Running 10 simulation steps...")
    
    for step in range(10):
        env.hazards.update(env.delta_t)
        
        peak_conc = env.hazard_field.concentration.max()
        cells_with_gas = (env.hazard_field.concentration > 0).sum()
        
        if step % 3 == 0:
            print(f"  Step {step}: Peak={peak_conc:.2f}, Cells with gas={cells_with_gas}")
    
    # Test 5: Verify gas didn't penetrate obstacles
    gas_in_obstacles = (env.hazard_field.concentration[env.obstacle_mask] > 0).sum()
    
    print(f"\n[FINAL CHECK] Gas in obstacle cells: {gas_in_obstacles}")
    
    if gas_in_obstacles > 0:
        print("  [ERROR] Gas penetrated obstacles!")
        return False
    
    # Test 6: Verify diffusion occurred
    if peak_conc > 0 and cells_with_gas > 1:
        print("  [SUCCESS] Gas diffused properly")
        return True
    else:
        print("  [WARNING] Gas did not diffuse (check diffusion coefficient)")
        return False

if __name__ == "__main__":
    success = test_gas_deployment()
    
    if success:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ TESTS FAILED - Review error messages above")
        print("="*60)