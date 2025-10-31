# test_gas_deployment.py
from src.env.protest_env import ProtestEnv
import yaml

with open('configs/default_scenario.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['agents']['police']['deploy_prob'] = 1.0  # Force deployment
config['grid']['obstacle_source'] = 'generate'  # Use synthetic to avoid encoding error

env = ProtestEnv(config)
obs, info = env.reset(seed=42)

print("Running 10 steps with guaranteed gas deployment...")
for i in range(10):
    obs, reward, terminated, truncated, info = env.step()
    print(f"Step {i}: Max hazard = {obs['hazard_concentration'].max():.2f}")

print("\nIf you see [HAZARD] messages above, gas deployment is working!")
