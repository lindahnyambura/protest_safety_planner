# real_nairobi_loader.py
import numpy as np
import json
import geopandas as gpd
import osmnx as ox
import networkx as nx
from pathlib import Path
from typing import Dict, Optional
from affine import Affine
import pandas as pd
from typing import List, Tuple
from scipy.spatial.distance import cdist


TARGET_CRS = "EPSG:32737"  # UTM Zone 37S – Nairobi


class RealNairobiLoader:
    """Loads, verifies, and aligns real Nairobi CBD grid + vector + OSM graph."""

    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.grid_path = Path("data/real_nairobi_cbd_200x200.npy")
        self.metadata_path = Path("data/real_nairobi_cbd_metadata.json")
        self.paths = {
            "buildings": Path("data/nairobi_buildings.geojson"),
            "roads_drive": Path("data/nairobi_roads_drive.geojson"),
            "roads_all": Path("data/nairobi_roads_all.geojson"),
        }
        self.graph = Path("data/nairobi_walk.graphml")

    # Step 1: Coordinate transforms & metadata
    def _ensure_crs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Ensure data is in TARGET_CRS (UTM 37S)."""
        if gdf.crs is None:
            print("[WARN] Missing CRS; assuming EPSG:4326 (lat/lon)")
            gdf.set_crs("EPSG:4326", inplace=True)
        if gdf.crs.to_string() != TARGET_CRS:
            gdf = gdf.to_crs(TARGET_CRS)
        return gdf

    def load_vector_data(self) -> Dict[str, gpd.GeoDataFrame]:
        """Load and reproject building and road GeoJSONs."""
        data = {}
        for key, path in self.paths.items():
            if not path.exists():
                print(f"[WARN] Missing {key} file: {path}")
                continue
            gdf = gpd.read_file(path)
            gdf = self._ensure_crs(gdf)
            data[key] = gdf
        return data

    def _compute_affine(self, bounds, grid_size):
        """Compute rasterio-style affine transform (row 0 = top)."""
        xmin, ymin, xmax, ymax = bounds
        width, height = grid_size
        cell_w = (xmax - xmin) / width
        cell_h = (ymax - ymin) / height
        return Affine(cell_w, 0, xmin, 0, -cell_h, ymax)

    def _update_metadata(self, gdfs: Dict[str, gpd.GeoDataFrame]):
        """Validate and persist metadata file."""
        if self.metadata_path.exists():
            meta = json.load(open(self.metadata_path))
        else:
            meta = {}

        # compute unified bounds
        all_bounds = np.array([g.total_bounds for g in gdfs.values()])
        xmin, ymin, xmax, ymax = (
            all_bounds[:, 0].min(),
            all_bounds[:, 1].min(),
            all_bounds[:, 2].max(),
            all_bounds[:, 3].max(),
        )
        meta.update(
            {
                "crs": TARGET_CRS,
                "bounds": [xmin, ymin, xmax, ymax],
                "grid_size": [self.grid_size, self.grid_size],
                "cell_size_m": (xmax - xmin) / self.grid_size,
                "row_origin": "top",
                "affine_transform": list(
                    self._compute_affine([xmin, ymin, xmax, ymax], (self.grid_size, self.grid_size))
                ),
            }
        )

        with open(self.metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Metadata updated: {self.metadata_path}")

        return meta

    # Step 2: Build canonical OSM graph
    def build_canonical_graph(self, edges_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
        """
        Build MultiDiGraph from road edges, harmonized for routing.
        
        Returns:
            MultiDiGraph
        Args:
            edges_gdf: GeoDataFrame

        """

        print("[INFO] Building canonical OSM graph...")

        # Ensure CRS
        edges_gdf = self._ensure_crs(edges_gdf)

        # Validate minimal columns
        if "geometry" not in edges_gdf.columns:
            raise ValueError("roads_all.geojson missing 'geometry' column")

        # Build nodes GeoDataFrame
        if "u" not in edges_gdf.columns or "v" not in edges_gdf.columns:
            print("[WARN] roads_all.geojson missing u/v. Extracting node coords from geometry.")

            def extract_uv(row):
                line = row.geometry
                start, end = line.coords[0], line.coords[-1]
                return tuple(start), tuple(end)  # start, end

            uv_coords = edges_gdf.apply(extract_uv, axis=1, result_type="expand")
            edges_gdf["u_coord"] = uv_coords[0]
            edges_gdf["v_coord"] = uv_coords[1]

            all_nodes = list(set(edges_gdf["u_coord"]) | set(edges_gdf["v_coord"]))
            # Additional safety: convert any lists to tuples
            all_nodes = [tuple(n) if isinstance(n, list) else n for n in all_nodes]
            
            # ensure all node IDs are strings and hashable
            node_ids = {}
            for i, coord in enumerate(all_nodes):
                # Safety: handle both tuple and already-string IDs
                if isinstance(coord, (tuple, list)):
                    coord = tuple(coord)  # Ensure tuple
                node_ids[coord] = f"node_{i}"
            
            edges_gdf["u"] = edges_gdf["u_coord"].map(node_ids)
            edges_gdf["v"] = edges_gdf["v_coord"].map(node_ids)

            # Build node records with explicit x, y extraction
            node_records = []
            for coord, nid in node_ids.items():
                if isinstance(coord, (tuple, list)) and len(coord) >= 2:
                    node_records.append({
                        "osmid": nid,
                        "x": float(coord[0]),  # Explicit float conversion
                        "y": float(coord[1])
                    })

            nodes_gdf = gpd.GeoDataFrame(
                node_records,
                geometry=gpd.points_from_xy([r["x"] for r in node_records],
                                            [r["y"] for r in node_records]),
                crs=TARGET_CRS,
            )
        else:
            # Ensure u/v are string-typed for consistency
            edges_gdf["u"] = edges_gdf["u"].astype(str)
            edges_gdf["v"] = edges_gdf["v"].astype(str)

            # Reconstruct node geometries from edges
            u_coords = edges_gdf.geometry.apply(lambda g: g.coords[0])
            v_coords = edges_gdf.geometry.apply(lambda g: g.coords[-1])

            node_df = pd.DataFrame({
                "osmid": list(edges_gdf["u"]) + list(edges_gdf["v"]),
                "x": [c[0] for c in u_coords] + [c[0] for c in v_coords],
                "y": [c[1] for c in u_coords] + [c[1] for c in v_coords],
            }).drop_duplicates(subset=["osmid"])

            nodes_gdf = gpd.GeoDataFrame(
                node_df,
                geometry=gpd.points_from_xy(node_df.x, node_df.y),
                crs=TARGET_CRS,
            ).set_index("osmid")

        # NEW: Sanitize highway and name columns
        print("[INFO] Sanitizing edge attributes...")

        if 'highway' in edges_gdf.columns:
            def sanitize_highway(val):
                if isinstance(val, list):
                    return val[0] if len(val) > 0 else 'unclassified'
                elif pd.isna(val):
                    return 'unclassified'
                return str(val)

            edges_gdf['highway'] = edges_gdf['highway'].apply(sanitize_highway)
        else:
            edges_gdf['highway'] = 'unclassified'
        
    
        if 'name' in edges_gdf.columns:
            def sanitize_name(val):
                if isinstance(val, list):
                    return val[0] if len(val) > 0 else None
                elif pd.isna(val):
                    return None
                return str(val) if val else None
            edges_gdf['name'] = edges_gdf['name'].apply(sanitize_name)
        else:
            edges_gdf['name'] = None
    
        edges_gdf["name"] = edges_gdf["name"].fillna(
            edges_gdf.apply(lambda r: f"unnamed_{r.get('osmid', 'na')}", axis=1)
        )

        # Compute length if missing
        if "length_m" not in edges_gdf.columns:
            edges_gdf["length_m"] = edges_gdf.geometry.length

        # Reset index for consistency
        edges_gdf = edges_gdf.reset_index(drop=True)

        # Ensure valid u/v columns
        if "u" not in edges_gdf.columns or "v" not in edges_gdf.columns:
            raise ValueError("edges_gdf must have 'u' and 'v' columns before building the graph.")
        
        edges_gdf["u"] = edges_gdf["u"].astype(str)
        edges_gdf["v"] = edges_gdf["v"].astype(str)

        # Create or sanitize 'key' column
        if "key" not in edges_gdf.columns:
            edges_gdf["key"] = 0
        else:
            edges_gdf["key"] = edges_gdf["key"].fillna(0).astype(int)

        # Set MultiIndex (u,v,key)
        try:
            edges_gdf.set_index(["u", "v", "key"], inplace=True)
        except Exception as e:
            raise ValueError(f"Failed to set MultiIndex (u,v,key): {e}")

        # Drop duplicate edges
        edges_gdf = edges_gdf[~edges_gdf.index.duplicated(keep="first")]

        # Ensure unique node index
        if "osmid" not in nodes_gdf.columns:
            nodes_gdf = nodes_gdf.reset_index(names="osmid")
        nodes_gdf["osmid"] = nodes_gdf["osmid"].astype(str)
        nodes_gdf = nodes_gdf.drop_duplicates(subset=["osmid"])
        nodes_gdf = nodes_gdf.set_index("osmid")

        # Validate MultiIndex integrity
        if not isinstance(edges_gdf.index, pd.MultiIndex):
            raise ValueError("edges_gdf is not a MultiIndex — cannot build OSMnx graph.")

        if tuple(edges_gdf.index.names) != ("u", "v", "key"):
            edges_gdf.index.set_names(("u", "v", "key"), inplace=True)

        print("[DEBUG] edges_gdf.index.names:", edges_gdf.index.names)
        print("[DEBUG] edges_gdf index sample:", edges_gdf.index[:3])

        # Patch OSMnx validator (optional bypass for custom graphs)
        import osmnx.convert as convert
        old_validate = convert._validate_node_edge_gdfs

        def _patched_validate_node_edge_gdfs(gdf_nodes, gdf_edges):
            if isinstance(gdf_edges.index, pd.MultiIndex) and list(gdf_edges.index.names) == ["u", "v", "key"]:
                return  # bypass strict tuple check
            return old_validate(gdf_nodes, gdf_edges)

        convert._validate_node_edge_gdfs = _patched_validate_node_edge_gdfs

        # Build and simplify the graph
        G = ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs={"crs": TARGET_CRS})
        G = ox.simplify_graph(G)

        ox.save_graphml(G, self.graph)
        print(f"[INFO] Canonical graph saved: {self.graph}")
        print(f"[INFO] Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

        return G

    # Public API
    def load_real_nairobi(self) -> np.ndarray:
        """Load grid obstacle mask from NumPy file."""
        if not self.grid_path.exists():
            raise FileNotFoundError(f"Missing {self.grid_path}")
        return np.load(self.grid_path)

    # Step 3: Precompute cell → nearest graph node lookup
    def precompute_cell_lookup(self, G: nx.MultiDiGraph, metadata: dict, mask: np.ndarray) -> np.ndarray:
        """
        Precompute a lookup from each grid cell (i, j) → nearest OSM graph node ID.

        Args:
            G: networkx.MultiDiGraph already projected to UTM (TARGET_CRS)
            metadata: dict containing 'affine_transform', 'grid_size', etc.
            mask: 2D obstacle mask (1=blocked, 0=free)

        Returns:
            np.ndarray of shape (H, W) with nearest node ID per cell (-1 if obstacle)
        """
        from shapely.geometry import Point
        from shapely.strtree import STRtree

        print("[INFO] Precomputing cell → nearest node lookup...")

        # --- Setup affine transform ---
        affine_vals = metadata.get("affine_transform")
        if not affine_vals:
            raise ValueError("Metadata missing 'affine_transform'")
        affine = Affine(*affine_vals[:6])

        h, w = mask.shape

        # --- Build spatial index of graph nodes ---
        nodes = list(G.nodes(data=True))
        node_points = [Point(d["x"], d["y"]) for _, d in nodes]
        node_ids = [n for n, _ in nodes]
        tree = STRtree(node_points)

        nearest_node_map = np.full((h, w), -1, dtype=object)
        valid_count = 0

        # --- Iterate over grid cells ---
        for i in range(h):
            if i % 10 == 0:
                print(f"  [DEBUG] processed {i}/{h} rows...")
            for j in range(w):
                if mask[i, j] == 1:
                    continue
                x, y = affine * (j + 0.5, i + 0.5)
                pt = Point(x, y)
                nearest_result = tree.nearest(pt)

                # Handle both Shapely return modes (geometry or index)
                if isinstance(nearest_result, (int, np.integer)):
                    idx = int(nearest_result)
                else:
                    # fall back to geometry match
                    try:
                        idx = list(tree.geometries).index(nearest_result)
                    except ValueError:
                        # fallback to manual distance min
                        distances = [pt.distance(g) for g in tree.geometries]
                        idx = int(np.argmin(distances))

                nearest_node_map[i, j] = node_ids[idx]
                valid_count += 1

        np.save("data/cell_to_node.npy", nearest_node_map)
        print(f"[INFO] Saved data/cell_to_node.npy ({valid_count} valid cells mapped)")
        print(f"[INFO] Map shape: {nearest_node_map.shape}")

        return nearest_node_map

    def load_all(self, build_graph=True) -> Dict:
        """Full loading pipeline (steps 1+2)."""
        gdfs = self.load_vector_data()
        meta = self._update_metadata(gdfs)
        obstacle_mask = self.load_real_nairobi()

        G = None
        if build_graph and "roads_all" in gdfs:
            G = self.build_canonical_graph(gdfs["roads_all"])
            node_map = self.precompute_cell_lookup(G, meta, obstacle_mask)
            meta["cell_to_node_path"] = "data/cell_to_node.npy"
        return {
            "obstacle_mask": obstacle_mask,
            "metadata": meta,
            "buildings_gdf": gdfs.get("buildings"),
            "roads_all": gdfs.get("roads_all"),
            "roads_drive": gdfs.get("roads_drive"),
            "graph": G,
            "is_real_osm": True,
        }

#  NEW FUNCTION: Generate spawn mask 
def generate_spawn_mask(obstacle_mask: np.ndarray,
                       osm_graph: nx.Graph,
                       cell_to_node: np.ndarray) -> np.ndarray:
    """
    Generate spawn mask: 1 = valid spawn point (road), 0 = invalid (building/no road)
    """
    spawn_mask = np.zeros_like(obstacle_mask, dtype=bool)

    for i in range(obstacle_mask.shape[0]):
        for j in range(obstacle_mask.shape[1]):
            # Rule 1: Not an obstacle
            if obstacle_mask[i, j]:
                continue

            # Rule 2: Has valid road node
            node_id = cell_to_node[i, j]
            if node_id in (-1, None) or pd.isna(node_id):
                continue

            # Rule 3: Node exists in graph
            if str(node_id) not in osm_graph:
                continue

            # Rule 4: Node connects to walkable edge
            valid_highway = False
            for neighbor in osm_graph.neighbors(str(node_id)):
                edge_data = osm_graph.get_edge_data(str(node_id), neighbor)
                if edge_data:
                    for _, data in edge_data.items():
                        highway = data.get("highway", "")
                        if highway not in ["motorway", "motorway_link"]:
                            valid_highway = True
                            break
                if valid_highway:
                    break

            if valid_highway:
                spawn_mask[i, j] = True

    coverage = 100 * spawn_mask.sum() / spawn_mask.size
    print(f"[INFO] Spawn mask generated: {spawn_mask.sum()} valid cells ({coverage:.1f}%)")
    return spawn_mask

def suggest_spawn_centers(osm_graph: nx.Graph,
                         spawn_mask: np.ndarray,
                         cell_to_node: np.ndarray,
                         n_clusters: int = 3) -> List[Tuple[int, int]]:
    """
    Suggest realistic spawn centers based on:
    1. High road connectivity (major intersections)
    2. Valid spawn locations
    3. Spatial distribution (not too clustered)
    """

    # Find highly connected nodes (degree > 3)
    high_connectivity_nodes = [
        (node_id, data)
        for node_id, data in osm_graph.nodes(data=True)
        if osm_graph.degree(node_id) > 3
    ]

    if not high_connectivity_nodes:
        # Fallback: any valid spawn cells
        valid_cells = np.argwhere(spawn_mask)
        if len(valid_cells) == 0:
            print("[WARN] No valid spawn cells found.")
            return []
        indices = np.random.choice(len(valid_cells), min(n_clusters, len(valid_cells)), replace=False)
        return [(int(x), int(y)) for y, x in valid_cells[indices]]

    # Convert to grid coordinates
    candidates = []
    for node_id, node_data in high_connectivity_nodes:
        matches = np.argwhere(cell_to_node == node_id)
        if len(matches) > 0:
            y, x = matches[0]
            if spawn_mask[y, x]:
                candidates.append((x, y))

    if len(candidates) < n_clusters:
        print(f"[WARN] Only {len(candidates)} high-connectivity spawn points found")
        return candidates

    # Select n_clusters centers with max separation
    candidates = np.array(candidates)
    selected = [candidates[0]]

    while len(selected) < n_clusters:
        distances = cdist(candidates, selected, metric='euclidean')
        min_distances = distances.min(axis=1)
        farthest_idx = min_distances.argmax()
        selected.append(candidates[farthest_idx])

    print(f"[INFO] Suggested {len(selected)} spawn centers at: {selected}")
    return [(int(x), int(y)) for x, y in selected]

def assign_node_capacities(graph, default_cap=6):
    """
    Assign realistic capacity to each node based on connected road types.
    
    Rationale:
    - Intersections on major roads (trunk/primary) = high capacity (20-30 people)
    - Secondary roads = medium (10-15)
    - Footpaths/alleys = low (5-8)
    
    Args:
        graph: NetworkX graph with OSM data
        default_cap: Fallback capacity for unclassified nodes
    
    Returns:
        graph: Modified graph with 'capacity' attribute on each node
    """
    for node in graph.nodes():
        # Get all edges connected to this node
        edges = graph.edges(node, data=True)
        road_types = [e[2].get('highway', 'unclassified') for e in edges]
        
        # Determine capacity from most significant road type
        if any(rt in ['trunk', 'primary', 'motorway'] for rt in road_types):
            capacity = 25  # Major intersection (e.g., Moi Ave/Haile Selassie)
        elif any(rt in ['secondary', 'tertiary'] for rt in road_types):
            capacity = 12  # Medium street
        elif any(rt in ['residential', 'service'] for rt in road_types):
            capacity = 8   # Small road
        else:
            capacity = default_cap  # Footpath/alley/unclassified
        
        graph.nodes[node]['capacity'] = capacity
    
    print(f"[INFO] Assigned capacities: "
          f"{sum(1 for n in graph.nodes if graph.nodes[n]['capacity'] == 25)} major, "
          f"{sum(1 for n in graph.nodes if graph.nodes[n]['capacity'] == 12)} medium, "
          f"{sum(1 for n in graph.nodes if graph.nodes[n]['capacity'] <= 8)} minor nodes")
    
    return graph

def identify_exit_nodes(osm_graph: nx.Graph,
                        metadata: Dict,
                        affine: Affine,
                        grid_shape: Tuple[int, int]) -> Dict[str, List[Dict]]:
    """
    Identify graph nodes near the map boundary that serve as exits.

    Prioritizes:
        1. Uhuru Highway (western edge)
        2. Major roads at northern/southern edges

    Returns:
        Dict with keys:
            'primary': List of dict(node_id, street_name, grid_pos, lat, lon)
            'secondary': same for other edges
    """
    exits = {'primary': [], 'secondary': []}
    EDGE_TOLERANCE_M = 100.0  # Increased tolerance

    # Defensive check for metadata integrity
    if "bounds" not in metadata:
        print("[ERROR] identify_exit_nodes: metadata missing 'bounds' key")
        return exits

    try:
        bounds = metadata["bounds"]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            print(f"[ERROR] Invalid bounds format: {bounds}")
            return exits
        xmin, ymin, xmax, ymax = bounds
    except Exception as e:
        print(f"[ERROR] Failed to parse bounds: {e}")
        return exits
    
    print(f"[DEBUG] Searching for exits within {EDGE_TOLERANCE_M}m of bounds")
    print(f"[DEBUG] Bounds: x=[{xmin:.1f}, {xmax:.1f}], y=[{ymin:.1f}, {ymax:.1f}]")
    
    candidate_nodes = []  # Track all boundary nodes for debugging

    # Iterate over all nodes and find boundary exits
    for node_id, node_data in osm_graph.nodes(data=True):
        x, y = node_data.get('x'), node_data.get('y')
        if x is None or y is None:
            continue

        # Distance to map edges
        dist_to_west = abs(x - xmin)
        dist_to_east = abs(x - xmax)
        dist_to_south = abs(y - ymin)
        dist_to_north = abs(y - ymax)

        min_dist = min(dist_to_west, dist_to_east, dist_to_north, dist_to_south)
        at_boundary = min(dist_to_west, dist_to_east, dist_to_north, dist_to_south) < EDGE_TOLERANCE_M
        
        if not at_boundary:
            continue
        
        # NEW: Collect ALL boundary nodes first (relaxed criteria)
        candidate_nodes.append({
            'node_id': str(node_id),
            'x': x,
            'y': y,
            'dist_to_boundary': min_dist,
            'closest_edge': 'west' if min_dist == dist_to_west else 
                           'east' if min_dist == dist_to_east else
                           'north' if min_dist == dist_to_north else 'south'
        })

    print(f"[DEBUG] Found {len(candidate_nodes)} nodes within {EDGE_TOLERANCE_M}m of boundary")

    # Now process candidates and find major roads
    for candidate in candidate_nodes:
        node_id = candidate['node_id']
        x, y = candidate['x'], candidate['y']

        # Check if connected to a major road
        street_name = None
        highway_type = None
        for neighbor in osm_graph.neighbors(node_id):
            edge_data = osm_graph.get_edge_data(node_id, neighbor)
            if edge_data:
                for _, data in edge_data.items():
                    if not isinstance(data, dict):
                        continue
                    hw = data.get('highway', '')
                    name = data.get('name', '')
                    
                    # Accept ANY road type at boundary (not just major)
                    if hw and hw != 'footway' and hw != 'path':
                        highway_type = hw
                        street_name = name if name else f"{hw.title()} Road"
                        break
                if highway_type:
                    break

        if not highway_type:
            continue

        # Convert to grid position
        try:
            col, row = ~affine * (x, y)
            grid_pos = (int(col), int(row))
        except Exception:
            continue

        entry = {
            'node_id': str(node_id),
            'street_name': street_name,
            'grid_pos': grid_pos,
            'x': x,
            'y': y,
            'highway_type': highway_type
        }

        # Categorize - prioritize western edge for Uhuru Highway
        if candidate['closest_edge'] == 'west':
            exits['primary'].append(entry)
        elif candidate['closest_edge'] in ['north', 'south']:
            exits['secondary'].append(entry)
        elif candidate['closest_edge'] == 'east':
            exits['secondary'].append(entry)

    print(f"[INFO] Identified {len(exits['primary'])} primary exits (western boundary)")
    print(f"[INFO] Identified {len(exits['secondary'])} secondary exits")
    
    # Debug: show what we found
    if exits['primary']:
        print("[DEBUG] Primary exits:")
        for e in exits['primary'][:3]:
            # Use ASCII-safe formatting
            print(f"  {e['street_name']} at grid {e['grid_pos']}")

    if exits['secondary']:
        print("[DEBUG] Secondary exits:")
        for e in exits['secondary'][:3]:
            print(f"  {e['street_name']} at grid {e['grid_pos']}")

    # If no exits found, create grid-edge fallbacks
    if not exits['primary'] and not exits['secondary']:
        print("[WARN] No boundary exits found — creating fallback grid exits")
        # Create fallback exits at grid edges
        width, height = grid_shape
        exits['primary'] = [
            {'node_id': 'fallback_west', 'street_name': 'Western Edge', 
             'grid_pos': (5, height//2), 'x': xmin, 'y': (ymin+ymax)/2},
            {'node_id': 'fallback_north', 'street_name': 'Northern Edge', 
             'grid_pos': (width//2, 5), 'x': (xmin+xmax)/2, 'y': ymax}
        ]

    return exits

def build_street_name_lookup(osm_graph: nx.Graph) -> Dict[str, str]:
    """
    Build node_id -> street_name mapping.
    Handles malformed edge data and list-type highway attributes.
    """
    HIGHWAY_PRIORITY = {
        'motorway': 0,
        'trunk': 1,
        'primary': 2,
        'secondary': 3,
        'tertiary': 4,
        'residential': 5,
        'unclassified': 6
    }

    node_to_street = {}
    skipped_edges = 0

    for node_id in list(osm_graph.nodes()):
        # Ensure node_id is hashable
        try:
            hash(node_id)
        except TypeError:
            node_id = str(node_id)

        best_name = None
        best_priority = 999

        # Try to iterate neighbors safely
        try:
            neighbors = list(osm_graph.neighbors(node_id))
        except Exception:
            skipped_edges += 1
            continue

        for neighbor in neighbors:
            try:
                edge_data = osm_graph.get_edge_data(node_id, neighbor)
            except Exception:
                skipped_edges += 1
                continue

            if edge_data is None:
                continue

            # Handle MultiGraph edge data (dict of dicts)
            if isinstance(edge_data, dict):
                items_to_check = edge_data.values() if edge_data else []
            else:
                items_to_check = [edge_data]

            for data in items_to_check:
                if not isinstance(data, dict):
                    continue
                
                # CRITICAL FIX: Handle highway as list or string
                highway_raw = data.get('highway', 'unclassified')
                
                # Convert list to single value (take first/most important)
                if isinstance(highway_raw, list):
                    if len(highway_raw) > 0:
                        # Take the highest priority highway type from the list
                        valid_types = [h for h in highway_raw if h in HIGHWAY_PRIORITY]
                        if valid_types:
                            highway = min(valid_types, key=lambda h: HIGHWAY_PRIORITY[h])
                        else:
                            highway = highway_raw[0]  # Fallback to first element
                    else:
                        highway = 'unclassified'
                elif isinstance(highway_raw, str):
                    highway = highway_raw
                else:
                    highway = 'unclassified'
                
                name = data.get('name', '')
                
                # Handle name as list too (some OSM data has multiple names)
                if isinstance(name, list):
                    name = name[0] if len(name) > 0 else ''
                
                priority = HIGHWAY_PRIORITY.get(highway, 999)

                if name and priority < best_priority:
                    best_name = name
                    best_priority = priority

        node_to_street[str(node_id)] = best_name or f"Node {node_id}"

    print(f"[INFO] Street name lookup built ({len(node_to_street)} nodes, skipped {skipped_edges} problematic edges).")
    return node_to_street

# Convenience entrypoint for external code
def load_real_nairobi_cbd_map(config: Dict) -> Optional[Dict]:
    """
    Load the real Nairobi CBD map and ensure the graph uses hashable node IDs.
    Returns a dict containing obstacle mask, metadata, buildings_gdf, graph, etc.
    """
    try:
        loader = RealNairobiLoader(grid_size=config["grid"]["width"])
        result = loader.load_all(build_graph=True)

        if result and "graph" in result and result["graph"] is not None:
            G = result["graph"]

            # Sanitize node IDs: ensure all are hashable strings
            bad_nodes = [n for n in G.nodes if isinstance(n, (list, dict, set, tuple))]
            if bad_nodes:
                print(f"[WARN] Found {len(bad_nodes)} unhashable node IDs; relabeling…")
                mapping = {}
                for n in G.nodes:
                    # Convert any unhashable or non-string ID to a safe string
                    try:
                        hash(n)
                        mapping[n] = n  # already hashable
                    except TypeError:
                        mapping[n] = str(n)
                nx.relabel_nodes(G, mapping, copy=False)
                print("[INFO] Node relabeling complete. All IDs are now strings.")

            result["graph"] = G

        return result

    except Exception as e:
        print(f"[ERROR] Failed to load real Nairobi map: {e}")
        return None
