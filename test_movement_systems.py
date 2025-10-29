# test_movement_systems.py
"""
Test 3: Verify both grid and graph movement systems
"""
from src.env.protest_env import ProtestEnv
import yaml

def test_movement_systems():
    print("🧪 TEST 3: Movement Systems")
    
    with open('configs/default_scenario.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['grid']['obstacle_source'] = 'nairobi'
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    # Take a few steps and observe movement
    print("🔄 Running 10 simulation steps...")
    
    for step in range(10):
        obs, reward, terminated, truncated, info = env.step()
        
        # Count movement types
        grid_movers = 0
        graph_movers = 0
        
        for agent in env.agents:
            if hasattr(agent, 'current_node'):
                graph_movers += 1
                if step == 5:  # Sample some debug info
                    street = env.street_names.get(str(agent.current_node), f"node_{agent.current_node}")
                    print(f"   Agent {agent.id} at {street}")
            else:
                grid_movers += 1
        
        if step in [0, 5, 9]:
            print(f"   Step {step}: {graph_movers} graph agents, {grid_movers} grid agents")
    
    print("✅ Movement test completed")
    return env

test_movement_systems()