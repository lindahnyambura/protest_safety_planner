# FILE: tests/test_phase1_phase2.py

"""
Validation tests for Phase 1 & 2 implementation.
"""

import numpy as np
import yaml
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]  # ../ (project root)
sys.path.insert(0, str(project_root))

from src.env.protest_env import ProtestEnv, load_config


def test_spawn_on_roads():
    """Validate that no agents spawn on obstacles."""
    config = load_config("configs/default_scenario.yaml")
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    obstacles_spawned = 0
    for agent in env.protesters:
        x, y = agent.pos
        if env.obstacle_mask[y, x]:
            obstacles_spawned += 1
            print(f"[FAIL] Agent {agent.id} spawned on obstacle at {agent.pos}")
    
    assert obstacles_spawned == 0, f"{obstacles_spawned} agents spawned on obstacles"
    print("[PASS] test_spawn_on_roads: All agents spawned on valid roads")


def test_exit_nodes_identified():
    """Validate exit node identification."""
    config = load_config("configs/default_scenario.yaml")
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    assert hasattr(env, 'exit_nodes'), "Exit nodes not identified"
    assert 'primary' in env.exit_nodes, "Primary exits missing"
    
    primary_exits = env.exit_nodes['primary']
    assert len(primary_exits) > 0, "No Uhuru Highway exits found"
    
    print(f"[PASS] test_exit_nodes_identified: Found {len(primary_exits)} primary exits")
    for exit_info in primary_exits[:3]:
        print(f"  - {exit_info['street_name']} at grid {exit_info['grid_pos']}")


def test_street_names_in_logs():
    """Validate human-readable debug logs."""
    config = load_config("configs/default_scenario.yaml")
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    assert hasattr(env, 'street_names'), "Street name lookup not built"
    assert len(env.street_names) > 0, "Street names dictionary empty"
    
    # Sample some nodes
    sample_nodes = list(env.street_names.keys())[:5]
    print(f"[PASS] test_street_names_in_logs: {len(env.street_names)} nodes mapped")
    for node_id in sample_nodes:
        print(f"  Node {node_id}: {env.street_names[node_id]}")


def test_no_immediate_exit_flocking():
    """Validate agents don't immediately flock to exits."""
    config = load_config("configs/default_scenario.yaml")
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    # Track initial and step-50 positions
    initial_positions = [agent.pos for agent in env.protesters]
    
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step()
        if terminated or truncated:
            break
    
    step50_positions = [agent.pos for agent in env.protesters]
    
    # Count agents that moved toward exits
    exit_positions = [e['grid_pos'] for e in env.exit_nodes.get('primary', [])]
    if not exit_positions:
        print("[SKIP] No exits to test against")
        return
    
    agents_moving_toward_exit = 0
    for i, agent in enumerate(env.protesters):
        if agent.state != 'moving':
            continue
        
        initial_pos = initial_positions[i]
        current_pos = step50_positions[i]
        
        # Distance to nearest exit
        initial_dist = min(np.hypot(initial_pos[0] - ex[0], initial_pos[1] - ex[1]) 
                          for ex in exit_positions)
        current_dist = min(np.hypot(current_pos[0] - ex[0], current_pos[1] - ex[1]) 
                          for ex in exit_positions)
        
        if current_dist < initial_dist * 0.8:  # Moved significantly closer
            agents_moving_toward_exit += 1
    
    percent_exiting = 100 * agents_moving_toward_exit / len(env.protesters)
    print(f"[INFO] {percent_exiting:.1f}% of agents moved toward exits in first 50 steps")
    
    # Should be less than 40% (most should be dispersing/seeking safety)
    assert percent_exiting < 40, f"Too many agents ({percent_exiting}%) immediately moved toward exits"
    print(f"[PASS] test_no_immediate_exit_flocking: Only {percent_exiting:.1f}% moved toward exits")


def test_panic_state_transitions():
    """Validate behavioral state machine."""
    config = load_config("configs/default_scenario.yaml")
    
    # Modify config to force hazard deployment
    config['agents']['police']['deploy_prob'] = 0.5  # 50% chance
    config['agents']['police']['deploy_cooldown'] = 20  # Faster cooldown
    
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    # Track state transitions for sample agents
    sample_agents = env.protesters[:5]
    state_history = {agent.id: [] for agent in sample_agents}
    
    for step in range(10):
        for agent in sample_agents:
            if hasattr(agent, 'behavioral_state'):
                state_history[agent.id].append(agent.behavioral_state)
        
        obs, reward, terminated, truncated, info = env.step()
        if terminated or truncated:
            break
    
    # Check that at least one agent transitioned states
    transitions_observed = 0
    for agent_id, states in state_history.items():
        unique_states = set(states)
        if len(unique_states) > 1:
            transitions_observed += 1
            print(f"[INFO] Agent {agent_id} states: {' → '.join(states[:10])}")
    
    assert transitions_observed > 0, "No behavioral state transitions observed"
    print(f"[PASS] test_panic_state_transitions: {transitions_observed}/5 agents showed state changes")


def test_agent_heterogeneity():
    """Validate different agent types behave differently."""
    config = load_config("configs/default_scenario.yaml")
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    # Group agents by profile
    profiles = {}
    for agent in env.protesters:
        profile = getattr(agent, 'profile_name', 'unknown')
        if profile not in profiles:
            profiles[profile] = []
        profiles[profile].append(agent)
    
    print(f"[INFO] Agent distribution:")
    for profile, agents in profiles.items():
        count = len(agents)
        avg_speed = np.mean([a.speed for a in agents])
        avg_risk_tol = np.mean([a.risk_tolerance for a in agents])
        print(f"  {profile}: {count} agents (speed={avg_speed:.2f}, risk_tol={avg_risk_tol:.2f})")
    
    # Validate expected profiles exist
    expected_profiles = {'cautious', 'average', 'bold', 'vulnerable'}
    found_profiles = set(profiles.keys())
    
    assert found_profiles == expected_profiles, f"Profile mismatch: {found_profiles} vs {expected_profiles}"
    print(f"[PASS] test_agent_heterogeneity: All 4 profiles present")


def run_all_tests():
    """Run all validation tests."""
    tests = [
        test_spawn_on_roads,
        test_exit_nodes_identified,
        test_street_names_in_logs,
        test_no_immediate_exit_flocking,
        test_panic_state_transitions,
        test_agent_heterogeneity
    ]
    
    print("=" * 60)
    print("PHASE 1 & 2 VALIDATION TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    test_panic_state_transitions()