# test_basic_functionality.py
"""
Test 1: Basic environment setup and agent spawning
"""

from src.env.protest_env import ProtestEnv
import yaml

def test_basic_functionality():
    print("🧪 TEST 1: Basic Environment Functionality")
    
    # Load your config
    with open('configs/default_scenario.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = ProtestEnv(config)
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"✅ Reset successful: {len(env.agents)} agents spawned")
    print(f"✅ Observation keys: {list(obs.keys())}")
    print(f"✅ Info: {info}")
    
    # Test step
    obs, reward, terminated, truncated, info = env.step()
    print(f"✅ Step successful: step={env.step_count}, terminated={terminated}")
    
    return env

test_basic_functionality()