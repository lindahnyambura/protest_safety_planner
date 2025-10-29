import networkx as nx
import geopandas as gpd
from pathlib import Path
import math
import ast

def build_simple_osm_graph():
    """Build a simple, working OSM graph from your roads data"""
    
    roads_path = Path("data/nairobi_roads_all.geojson")
    if not roads_path.exists():
        print(" Roads file not found")
        return None
    
    print(" Building simplified OSM graph...")
    
    roads_gdf = gpd.read_file(roads_path)
    
    G = nx.MultiDiGraph()
    node_counter = 0
    node_coords_to_id = {}
    
    for idx, road in roads_gdf.iterrows():
        geometry = road.geometry
        if geometry is None or geometry.geom_type != 'LineString':
            continue
        
        coords = list(geometry.coords)
        node_ids = []
        
        for coord in coords:
            if coord not in node_coords_to_id:
                node_coords_to_id[coord] = node_counter
                G.add_node(node_counter, x=float(coord[0]), y=float(coord[1]))
                node_counter += 1
            node_ids.append(node_coords_to_id[coord])
        
        for i in range(len(node_ids) - 1):
            u, v = node_ids[i], node_ids[i + 1]
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            
            # --- Clean attributes ---
            highway = road.get('highway') or 'unclassified'
            name = road.get('name') or f'road_{idx}'
            osmid = road.get('osmid', idx)
            
            # Handle osmid being a list, string, NaN, or None
            try:
                if isinstance(osmid, str) and osmid.startswith('['):
                    # Convert stringified list like "[123, 456]" -> first element
                    parsed = ast.literal_eval(osmid)
                    osmid = parsed[0] if isinstance(parsed, (list, tuple)) and parsed else idx
                elif isinstance(osmid, (list, tuple)) and osmid:
                    osmid = osmid[0]
                elif isinstance(osmid, float) and math.isnan(osmid):
                    osmid = idx
            except Exception:
                osmid = idx
            
            # Safely extract name and highway
            road_name = road.get('name')
            if not road_name or str(road_name).lower() in ['nan', 'none', '']:
                road_name = road.get('highway', f'road_{idx}')

            # Normalize to string for GraphML safety
            road_name = str(road_name)
            
            # Always stringify attributes to be GraphML-safe
            G.add_edge(u, v,
                       length=length,
                       #highway=road.get('highway', 'unclassified'),
                       highway=str(road.get('highway', 'unclassified')),
                       #name=road.get('name') or road.get('highway') or f'road_{idx}',
                       name=road_name,
                       #osmid=road.get('osmid', idx))
                       osmid=str(road.get('osmid', idx)))
    
    output_path = Path("data/nairobi_simple.graphml")
    nx.write_graphml(G, output_path)
    
    print(f" Built simplified graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    print(f" Saved to: {output_path}")
    
    return G

if __name__ == "__main__":
    build_simple_osm_graph()
