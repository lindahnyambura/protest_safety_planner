# test_osm_integration.py
"""
Diagnostic script to test Nairobi OSM graph integration
Run this to identify exactly where the system is breaking
"""


import numpy as np
import networkx as nx
import sys  
import sys
from pathlib import Path

# Add src/ to the module search path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from env.real_nairobi_loader import NairobiLoader, load_real_nairobi_cbd_map



def test_graph_loading():
    """Test if we can load the existing graph file"""
    print("🔍 TEST 1: Loading existing graph file...")
    
    graph_path = Path("data/nairobi_walk.graphml")
    if graph_path.exists():
        try:
            G = nx.read_graphml(graph_path)
            print(f"✅ SUCCESS: Loaded graph with {len(G.nodes)} nodes, {len(G.edges)} edges")
            
            # Check node attributes
            sample_node = list(G.nodes(data=True))[0]
            print(f"📊 Sample node: {sample_node}")
            
            # Check edge attributes  
            sample_edge = list(G.edges(data=True))[0]
            print(f"📊 Sample edge: {sample_edge}")
            
            return G
        except Exception as e:
            print(f"❌ FAILED: {e}")
    else:
        print("❌ Graph file doesn't exist")
    return None

def test_cell_to_node_mapping():
    """Test the cell-to-node mapping"""
    print("\n🔍 TEST 2: Cell-to-node mapping...")
    
    mapping_path = Path("data/cell_to_node.npy")
    if mapping_path.exists():
        try:
            cell_to_node = np.load(mapping_path)
            print(f"✅ SUCCESS: Loaded mapping shape {cell_to_node.shape}")
            
            # Count valid mappings
            valid_cells = np.sum(cell_to_node != -1)
            total_cells = cell_to_node.size
            print(f"📊 {valid_cells}/{total_cells} cells mapped ({100*valid_cells/total_cells:.1f}%)")
            
            # Show sample mappings
            unique_nodes = np.unique(cell_to_node[cell_to_node != -1])
            print(f"📊 {len(unique_nodes)} unique OSM nodes referenced")
            
            return cell_to_node
        except Exception as e:
            print(f"❌ FAILED: {e}")
    else:
        print("❌ Mapping file doesn't exist")
    return None

def test_obstacle_grid():
    """Test the obstacle grid"""
    print("\n🔍 TEST 3: Obstacle grid...")
    
    grid_path = Path("data/real_nairobi_cbd_200x200.npy")
    if grid_path.exists():
        try:
            obstacle_mask = np.load(grid_path)
            print(f"✅ SUCCESS: Loaded grid shape {obstacle_mask.shape}")
            print(f"📊 {obstacle_mask.sum()} obstacle cells ({100*obstacle_mask.sum()/obstacle_mask.size:.1f}%)")
            return obstacle_mask
        except Exception as e:
            print(f"❌ FAILED: {e}")
    else:
        print("❌ Grid file doesn't exist")
    return None

def test_metadata():
    """Test metadata file"""
    print("\n🔍 TEST 4: Metadata...")
    
    meta_path = Path("data/real_nairobi_cbd_metadata.json")
    if meta_path.exists():
        import json
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print("✅ SUCCESS: Loaded metadata")
            print(f"📊 Bounds: {metadata.get('bounds', 'MISSING')}")
            print(f"📊 CRS: {metadata.get('crs', 'MISSING')}")
            return metadata
        except Exception as e:
            print(f"❌ FAILED: {e}")
    else:
        print("❌ Metadata file doesn't exist")
    return None

def test_full_loader():
    """Test the complete loader pipeline"""
    print("\n🔍 TEST 5: Full loader pipeline...")
    
    try:
        # Mock config matching your structure
        config = {
            "grid": {
                "width": 100,
                "height": 100,
                "cell_size_m": 5.0,
                "obstacle_source": "nairobi"
            }
        }
        
        result = load_real_nairobi_cbd_map(config)
        if result and result.get('is_real_osm'):
            print("✅ SUCCESS: Full pipeline completed")
            print(f"📊 Graph: {'Present' if result.get('graph') else 'Missing'}")
            print(f"📊 Roads: {'Present' if result.get('roads_all') else 'Missing'}")
        else:
            print("❌ FAILED: Pipeline returned None or missing real_osm flag")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")

def main():
    print("🚀 STARTING NAIROBI OSM INTEGRATION DIAGNOSTICS")
    print("=" * 50)
    
    # Run all tests
    G = test_graph_loading()
    cell_to_node = test_cell_to_node_mapping() 
    obstacle_mask = test_obstacle_grid()
    metadata = test_metadata()
    test_full_loader()
    
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY:")
    
    # Provide actionable next steps based on test results
    if G is None:
        print("❌ CRITICAL: Graph not loaded - need to rebuild OSM graph")
        print("   → Run: python rebuild_osm_graph.py")
    elif cell_to_node is None:
        print("❌ CRITICAL: Cell-to-node mapping missing")
        print("   → Need to run precompute_cell_lookup()")
    else:
        print("✅ Graph system is ready for agent movement")
        
    if obstacle_mask is not None and metadata is not None:
        print("✅ Grid system is ready")
    else:
        print("❌ Grid files missing or corrupted")

if __name__ == "__main__":
    main()