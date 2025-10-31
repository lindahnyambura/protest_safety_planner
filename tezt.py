from src.env.protest_env import ProtestEnv, load_config

config = load_config('configs/default_scenario.yaml')

# Test 1: Harm occurs
env = ProtestEnv(config)
obs, info = env.reset(seed=42)

for step in range(20):
    obs, reward, term, trunc, info = env.step()
    if info['harm_grid'].sum() > 0:
        print(f"✅ Harm occurred at step {step}: {info['harm_grid'].sum()} cells")
        break
else:
    print("❌ No harm after 20 steps!")

# Test 2: Different seeds produce different results
def count_harm(seed):
    env = ProtestEnv(config)
    obs, info = env.reset(seed=seed)
    total = 0
    for _ in range(10):
        obs, r, t, tr, info = env.step()
        total += info['harm_grid'].sum()
    return total

h42 = count_harm(42)
h43 = count_harm(43)
h44 = count_harm(44)

print(f"\nSeed 42: {h42} harm events")
print(f"Seed 43: {h43} harm events")
print(f"Seed 44: {h44} harm events")

if h42 != h43 or h43 != h44:
    print("✅ Seeds properly isolated!")
else:
    print("❌ Seeds still producing same results")