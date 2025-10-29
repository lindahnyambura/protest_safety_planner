# test_graph_system.py
"""
Test 2: Verify OSM graph integration
"""
from src.env.protest_env import ProtestEnv
import yaml

def test_graph_system():
    print("🧪 TEST 2: Graph System Verification")
    
    with open('configs/default_scenario.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force Nairobi mode
    config['grid']['obstacle_source'] = 'nairobi'
    
    env = ProtestEnv(config)
    obs, info = env.reset(seed=42)
    
    # Check graph components
    print(f"✅ OSM Graph: {'Present' if env.osm_graph else 'Missing'}")
    if env.osm_graph:
        print(f"   - Nodes: {len(env.osm_graph.nodes)}")
        print(f"   - Edges: {len(env.osm_graph.edges)}")
    
    print(f"✅ Cell-to-node mapping: {'Present' if env.cell_to_node is not None else 'Missing'}")
    if env.cell_to_node is not None:
        valid_mappings = (env.cell_to_node != -1).sum()
        print(f"   - Valid mappings: {valid_mappings}/{env.cell_to_node.size}")
    
    print(f"✅ Street names: {'Present' if hasattr(env, 'street_names') else 'Missing'}")
    if hasattr(env, 'street_names'):
        print(f"   - Mapped streets: {len(env.street_names)}")
        # Show sample street names
        sample = list(env.street_names.items())[:3]
        for node, street in sample:
            print(f"     {node} → {street}")
    
    # Check if agents have graph attributes
    graph_agents = [a for a in env.agents if hasattr(a, 'current_node')]
    print(f"✅ Graph-enabled agents: {len(graph_agents)}/{len(env.agents)}")
    
    return env

test_graph_system()