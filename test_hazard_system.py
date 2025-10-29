# test_hazard_system.py
"""
Test 4: Verify hazards work with both movement systems
"""
from src.env.protest_env import ProtestEnv
import yaml

def test_hazard_system():
    print("🧪 TEST 4: Hazard System Integration")
    
    with open('configs/default_scenario.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['grid']['obstacle_source'] = 'nairobi'
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    # Check initial hazard state
    hazard_grid = obs['hazard_concentration']
    print(f"✅ Hazard grid: {hazard_grid.shape}, max concentration: {hazard_grid.max():.3f}")
    
    # Run a few steps to see hazard propagation
    print("🔄 Testing hazard propagation...")
    
    for step in range(5):
        obs, reward, terminated, truncated, info = env.step()
        hazard_grid = obs['hazard_concentration']
        
        # Check if agents are being affected
        harmed_agents = sum(agent.cumulative_harm > 0 for agent in env.agents)
        print(f"   Step {step}: {harmed_agents} agents with harm, max hazard: {hazard_grid.max():.3f}")
    
    print("✅ Hazard test completed")
    return env

test_hazard_system()