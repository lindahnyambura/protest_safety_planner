# fixed_nairobi_loader.py
"""
Fixed version with reliable cell-to-node mapping and simplified graph
"""

import numpy as np
import networkx as nx
import json
from pathlib import Path
from typing import Dict, List, Tuple
from affine import Affine

class NairobiLoader:
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.simple_graph_path = Path("data/nairobi_simple.graphml")
        self.cell_to_node_path = Path("data/cell_to_node_fixed.npy")
        self.obstacle_path = Path("data/real_nairobi_cbd_100x100.npy")
        self.metadata_path = Path("data/real_nairobi_cbd_metadata.json")
        
    def load_simple_graph(self) -> nx.MultiDiGraph:
        """Load the working simplified graph"""
        if not self.simple_graph_path.exists():
            raise FileNotFoundError(f"Simple graph not found at {self.simple_graph_path}")
        
        G = nx.read_graphml(self.simple_graph_path)
        print(f"✅ Loaded simplified graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G
    
    def load_obstacle_grid(self) -> np.ndarray:
        """Load obstacle grid"""
        return np.load(self.obstacle_path)
    
    def load_metadata(self) -> Dict:
        """Load metadata"""
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def create_cell_to_node_mapping(self, G: nx.MultiDiGraph) -> np.ndarray:
        """
        Create a reliable cell-to-node mapping with proper data types
        """
        print("🔄 Creating reliable cell-to-node mapping...")
        
        # Load metadata for coordinate transforms
        metadata = self.load_metadata()
        obstacle_mask = self.load_obstacle_grid()
        
        affine_vals = metadata.get("affine_transform")
        if not affine_vals:
            raise ValueError("Missing affine_transform in metadata")
        
        affine = Affine(*affine_vals[:6])
        h, w = obstacle_mask.shape
        
        # Convert node positions to numpy for fast distance calculation
        node_data = []
        for node_id, data in G.nodes(data=True):
            node_data.append({
                'id': int(node_id),  # Use integer IDs for simplicity
                'x': float(data['x']),
                'y': float(data['y'])
            })
        
        # Create mapping array (use -1 for obstacles/no mapping)
        cell_to_node = np.full((h, w), -1, dtype=np.int32)
        
        # Map each cell to nearest node
        for i in range(h):
            for j in range(w):
                if obstacle_mask[i, j]:
                    continue  # Skip obstacles
                
                # Convert grid cell to world coordinates
                world_x, world_y = affine * (j + 0.5, i + 0.5)
                
                # Find nearest node
                min_dist = float('inf')
                nearest_node_id = -1
                
                for node in node_data:
                    dist = ((world_x - node['x'])**2 + (world_y - node['y'])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node_id = node['id']
                
                if min_dist < 50:  # Only map if reasonably close to a road (50m threshold)
                    cell_to_node[i, j] = nearest_node_id
        
        # Save with proper format
        np.save(self.cell_to_node_path, cell_to_node)
        
        valid_cells = np.sum(cell_to_node != -1)
        print(f"✅ Created mapping: {valid_cells}/{h*w} cells mapped to roads")
        
        return cell_to_node
    
    def load_cell_to_node_mapping(self) -> np.ndarray:
        """Load the fixed cell-to-node mapping"""
        if not self.cell_to_node_path.exists():
            raise FileNotFoundError(f"Run create_cell_to_node_mapping() first")
        
        # Load with allow_pickle=True for compatibility
        return np.load(self.cell_to_node_path, allow_pickle=False)
    
    def get_street_names(self, G: nx.MultiDiGraph) -> Dict[int, str]:
        """Create node_id -> street_name mapping (robust to bad edge data)"""
        street_names = {}

        for node_id in G.nodes():
            best_street = f"Node_{node_id}"  # Default fallback

            for neighbor in G.neighbors(node_id):
                edge_data = G.get_edge_data(node_id, neighbor)
                if not edge_data:
                    continue

                # Get the first edge record safely
                data = list(edge_data.values())[0]
                

                # Some edge_data entries might be floats (NaN) or malformed
                if not isinstance(data, dict):
                    continue

                street_name = data.get('name', '')
                if isinstance(street_name, float):  # skip NaNs or bad strings
                    continue

                if street_name and street_name.lower() not in ['unnamed', '']:
                    best_street = street_name
                    break

            street_names[int(node_id)] = best_street

        print(f"✅ Extracted street names for {len(street_names)} nodes")
        return street_names

    
    def load_all(self) -> Dict:
        """Load everything needed for graph-based simulation"""
        G = self.load_simple_graph()
        obstacle_mask = self.load_obstacle_grid()
        metadata = self.load_metadata()
        
        # Create or load cell-to-node mapping
        try:
            cell_to_node = self.load_cell_to_node_mapping()
        except FileNotFoundError:
            print("🔄 Cell-to-node mapping not found, creating...")
            cell_to_node = self.create_cell_to_node_mapping(G)
        
        street_names = self.get_street_names(G)
        
        return {
            'graph': G,
            'obstacle_mask': obstacle_mask,
            'cell_to_node': cell_to_node,
            'street_names': street_names,
            'metadata': metadata,
            'is_real_osm': True
        }

# Test the fixed loader
def test_loader():
    loader = NairobiLoader()
    result = loader.load_all()
    
    print(f"📊 Graph nodes: {len(result['graph'].nodes)}")
    print(f"📊 Graph edges: {len(result['graph'].edges)}")
    print(f"📊 Cell-to-node shape: {result['cell_to_node'].shape}")
    print(f"📊 Street names: {len(result['street_names'])}")
    
    # Show sample street names
    sample_nodes = list(result['street_names'].items())[:5]
    print("📍 Sample street mappings:")
    for node_id, street in sample_nodes:
        print(f"   Node {node_id} → {street}")
    
    return result

if __name__ == "__main__":
    test_loader()