def test_diagnostics():
    from src.env.real_nairobi_loader import load_real_nairobi_cbd_map, identify_exit_nodes
    import yaml
    
    config = yaml.safe_load(open("configs/default_scenario.yaml"))
    
    print("🔍 Running diagnostics...")
    
    # Test 1: Graph loading
    result = load_real_nairobi_cbd_map(config)
    if result and "graph" in result:
        G = result["graph"]
        print(f"✓ Graph loaded: {len(G.nodes)} nodes")
        print(f"✓ Node types: {set(type(n).__name__ for n in list(G.nodes)[:5])}")
        
        # Test 2: Exit node detection
        exits = identify_exit_nodes(G, result["metadata"], result.get("affine"), (100, 100))
        print(f"✓ Exit nodes: {len(exits['primary'])} primary, {len(exits['secondary'])} secondary")
        
        # Test 3: Check for specific Uhuru nodes
        target_nodes = ['4730073', '4730074', '4741915', '4741916']
        found = [n for n in target_nodes if n in G.nodes]
        print(f"✓ Specific nodes found: {found}")
        
    else:
        print("✗ Graph loading failed")

test_diagnostics()