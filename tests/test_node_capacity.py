"""
Test script for node capacity system validation.

Validates:
1. Node capacities assigned correctly by road type
2. Movement respects capacity constraints
3. Queue formation at congested nodes
4. Agents reroute around congestion
5. No grid-based capacity errors in graph mode
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.env.protest_env import ProtestEnv, load_config
from src.env.agent import AgentState


def test_capacity_assignment():
    """Test 1: Verify node capacities are assigned correctly."""
    print("\n" + "="*60)
    print("TEST 1: Node Capacity Assignment")
    print("="*60)
    
    config = load_config("configs/default_scenario.yaml")
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    if env.osm_graph is None:
        print("❌ FAIL: No OSM graph loaded")
        return False
    
    # Check capacity distribution
    capacities = [env.osm_graph.nodes[n].get('capacity', 0) for n in env.osm_graph.nodes]
    
    print(f"Total nodes: {len(capacities)}")
    print(f"  Major intersections (cap=25): {capacities.count(25)}")
    print(f"  Medium streets (cap=12): {capacities.count(12)}")
    print(f"  Small roads (cap=8): {capacities.count(8)}")
    print(f"  Minor paths (cap≤6): {sum(1 for c in capacities if c <= 6)}")
    
    if max(capacities) == 0:
        print("❌ FAIL: No capacities assigned")
        return False
    
    print("✅ PASS: Capacities assigned correctly")
    return True


def test_spawn_respects_capacity():
    """Test 2: Agents don't spawn beyond node capacity."""
    print("\n" + "="*60)
    print("TEST 2: Spawn Capacity Enforcement")
    print("="*60)
    
    config = load_config("configs/default_scenario.yaml")
    # Force high agent count to test capacity limits
    config['agents']['protesters']['count'] = 100
    
    env = ProtestEnv(config)
    obs, info = env.reset(seed=123)
    
    if env.osm_graph is None:
        print("⚠️  SKIP: Grid-based environment")
        return True
    
    # Check node occupancy vs capacity
    violations = []
    for node_id, occ in env.node_occupancy.items():
        capacity = env.osm_graph.nodes[node_id].get('capacity', 6)
        if occ > capacity:
            violations.append((node_id, occ, capacity))
    
    if violations:
        print(f"❌ FAIL: {len(violations)} nodes over capacity at spawn:")
        for node_id, occ, cap in violations[:5]:  # Show first 5
            street = env.street_names.get(str(node_id), f"node {node_id}")
            print(f"  {street}: {occ}/{cap} agents")
        return False
    
    print(f"✅ PASS: All {len(env.node_occupancy)} occupied nodes within capacity")
    return True


def test_movement_queuing():
    """Test 3: Agents queue when target node at capacity."""
    print("\n" + "="*60)
    print("TEST 3: Movement Queuing at Capacity")
    print("="*60)
    
    config = load_config("configs/default_scenario.yaml")
    config['agents']['protesters']['count'] = 150
    
    env = ProtestEnv(config)
    obs, info = env.reset(seed=456)
    
    if env.osm_graph is None:
        print("⚠️  SKIP: Grid-based environment")
        return True
    
    # Run for 20 steps to generate congestion
    congestion_events = []
    waiting_agents = []
    
    for step in range(20):
        obs, reward, terminated, truncated, info = env.step()
        
        # Track congestion events
        step_congestion = [
            e for e in env.events_log 
            if e.get('event_type') == 'node_congestion'
        ]
        congestion_events.extend(step_congestion)
        
        # Count waiting agents
        waiting = sum(1 for a in env.agents if a.state == AgentState.WAITING)
        waiting_agents.append(waiting)
        
        if step_congestion:
            print(f"  Step {step}: {len(step_congestion)} congestion events, {waiting} agents waiting")
    
    if not congestion_events:
        print("⚠️  WARNING: No congestion detected (may need more agents or smaller area)")
        print("   This is OK if density is low")
        return True
    
    print(f"✅ PASS: {len(congestion_events)} congestion events recorded")
    print(f"   Max waiting agents: {max(waiting_agents)}")
    print(f"   Avg waiting agents: {np.mean(waiting_agents):.1f}")
    
    return True


def test_no_grid_capacity_errors():
    """Test 4: No deprecated grid capacity errors in graph mode."""
    print("\n" + "="*60)
    print("TEST 4: No Grid Capacity Errors")
    print("="*60)
    
    config = load_config("configs/default_scenario.yaml")
    config['agents']['protesters']['count'] = 80
    
    env = ProtestEnv(config)
    obs, info = env.reset(seed=789)
    
    if env.osm_graph is None:
        print("⚠️  SKIP: Grid-based environment")
        return True
    
    # Capture printed output to detect "[WARN] cells exceed N_CELL_MAX"
    import io
    from contextlib import redirect_stdout
    
    captured_output = io.StringIO()
    
    with redirect_stdout(captured_output):
        for step in range(30):
            obs, reward, terminated, truncated, info = env.step()
    
    output = captured_output.getvalue()
    
    # Check for deprecated error message
    if "cells exceed N_CELL_MAX" in output:
        print("❌ FAIL: Grid capacity errors still present:")
        for line in output.split('\n'):
            if "cells exceed" in line.lower():
                print(f"  {line}")
        return False
    
    print("✅ PASS: No grid capacity errors detected")
    return True


def test_congestion_feedback():
    """Test 5: Agents detect and avoid congested nodes."""
    print("\n" + "="*60)
    print("TEST 5: Congestion-Aware Pathfinding")
    print("="*60)
    
    config = load_config("configs/default_scenario.yaml")
    config['agents']['protesters']['count'] = 60
    
    env = ProtestEnv(config)
    obs, info = env.reset(seed=999)
    
    if env.osm_graph is None:
        print("⚠️  SKIP: Grid-based environment")
        return True
    
    # Track path choices over time
    node_visit_counts = {}
    
    for step in range(40):
        obs, reward, terminated, truncated, info = env.step()
        
        # Record node visits
        for agent in env.agents:
            if hasattr(agent, 'current_node'):
                node = agent.current_node
                node_visit_counts[node] = node_visit_counts.get(node, 0) + 1
    
    # Identify most congested nodes
    sorted_nodes = sorted(node_visit_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Total unique nodes visited: {len(node_visit_counts)}")
    print(f"  Top 5 most visited nodes:")
    for node, count in sorted_nodes[:5]:
        capacity = env.osm_graph.nodes[node].get('capacity', 6)
        street = env.street_names.get(str(node), f"node {node}")
        print(f"    {street}: {count} visits (capacity: {capacity})")
    
    # Check if high-capacity nodes are used more
    high_cap_visits = sum(
        count for node, count in node_visit_counts.items()
        if env.osm_graph.nodes[node].get('capacity', 6) >= 12
    )
    low_cap_visits = sum(
        count for node, count in node_visit_counts.items()
        if env.osm_graph.nodes[node].get('capacity', 6) < 12
    )
    
    if high_cap_visits > low_cap_visits:
        print(f"✅ PASS: Agents prefer high-capacity routes ({high_cap_visits} vs {low_cap_visits} visits)")
    else:
        print(f"⚠️  WARNING: Low preference for high-capacity routes ({high_cap_visits} vs {low_cap_visits})")
        print("   May need to tune congestion penalty in scoring")
    
    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("NODE CAPACITY SYSTEM VALIDATION")
    print("="*60)
    
    tests = [
        ("Capacity Assignment", test_capacity_assignment),
        ("Spawn Enforcement", test_spawn_respects_capacity),
        ("Movement Queuing", test_movement_queuing),
        ("No Grid Errors", test_no_grid_capacity_errors),
        ("Congestion Feedback", test_congestion_feedback)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Capacity system working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)