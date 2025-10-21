# real_nairobi_loader.py
import numpy as np
import json
import geopandas as gpd
import osmnx as ox
import networkx as nx
from pathlib import Path
from typing import Dict, Optional
from affine import Affine



TARGET_CRS = "EPSG:32737"  # UTM Zone 37S – Nairobi


class RealNairobiLoader:
    """Loads, verifies, and aligns real Nairobi CBD grid + vector + OSM graph."""

    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.grid_path = Path("data/real_nairobi_cbd_100x100.npy")
        self.metadata_path = Path("data/real_nairobi_cbd_metadata.json")
        self.paths = {
            "buildings": Path("data/nairobi_buildings.geojson"),
            "roads_drive": Path("data/nairobi_roads_drive.geojson"),
            "roads_all": Path("data/nairobi_roads_all.geojson"),
        }
        self.graph_path = Path("data/nairobi_walk.graphml")

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
        """Build MultiDiGraph from road edges, harmonized for routing."""
        import pandas as pd
        print("[INFO] Building canonical OSM graph...")

        edges_gdf = self._ensure_crs(edges_gdf)

        # Validate minimal columns
        if "geometry" not in edges_gdf.columns:
            raise ValueError("roads_all.geojson missing 'geometry' column")

        # --- Build nodes GeoDataFrame (robust) ---
        if "u" not in edges_gdf.columns or "v" not in edges_gdf.columns:
            print("[WARN] roads_all.geojson missing u/v. Extracting node coords from geometry.")

            def extract_uv(row):
                line = row.geometry
                start, end = line.coords[0], line.coords[-1]
                return start, end

            uv_coords = edges_gdf.apply(extract_uv, axis=1, result_type="expand")
            edges_gdf["u_coord"] = uv_coords[0]
            edges_gdf["v_coord"] = uv_coords[1]

            all_nodes = list(set(edges_gdf["u_coord"]) | set(edges_gdf["v_coord"]))
            node_ids = {coord: str(i) for i, coord in enumerate(all_nodes)}
            edges_gdf["u"] = edges_gdf["u_coord"].map(node_ids)
            edges_gdf["v"] = edges_gdf["v_coord"].map(node_ids)

            node_records = [
                {"osmid": nid, "x": x, "y": y} for (x, y), nid in node_ids.items()
            ]
            nodes_gdf = gpd.GeoDataFrame(
                node_records,
                geometry=gpd.points_from_xy([r["x"] for r in node_records],
                                            [r["y"] for r in node_records]),
                crs=TARGET_CRS,
            )
        else:
            # Reconstruct node geometries from edges directly
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


        # Fallback for missing names
        if "name" not in edges_gdf.columns:
            edges_gdf["name"] = None
        edges_gdf["name"] = edges_gdf["name"].fillna(
            edges_gdf.apply(lambda r: f"unnamed_{r.get('osmid', 'na')}", axis=1)
        )

        # Compute length if missing
        if "length_m" not in edges_gdf.columns:
            edges_gdf["length_m"] = edges_gdf.geometry.length

        # --- Ensure proper MultiIndex for OSMnx ---
        edges_gdf = edges_gdf.reset_index(drop=True)

        # Make sure u/v exist and are numeric or string
        if "u" not in edges_gdf.columns or "v" not in edges_gdf.columns:
            raise ValueError("edges_gdf must have 'u' and 'v' columns before building the graph.")
        edges_gdf["u"] = edges_gdf["u"].astype(str)
        edges_gdf["v"] = edges_gdf["v"].astype(str)

        # Create or sanitize 'key' column
        if "key" not in edges_gdf.columns:
            edges_gdf["key"] = 0
        else:
            edges_gdf["key"] = edges_gdf["key"].fillna(0).astype(int)

        # Now enforce MultiIndex
        try:
            edges_gdf.set_index(["u", "v", "key"], inplace=True)
        except Exception as e:
            raise ValueError(f"Failed to set MultiIndex (u,v,key): {e}")

        # Drop any duplicate edges by (u,v,key)
        edges_gdf = edges_gdf[~edges_gdf.index.duplicated(keep="first")]

        # --- Ensure unique node index ---
        if "osmid" not in nodes_gdf.columns:
            # If osmid is already the index, preserve it
            nodes_gdf = nodes_gdf.reset_index(names="osmid")

        nodes_gdf["osmid"] = nodes_gdf["osmid"].astype(str)
        nodes_gdf = nodes_gdf.drop_duplicates(subset=["osmid"])
        nodes_gdf = nodes_gdf.set_index("osmid")

        # --- Final sanity check (force exact tuple names for OSMnx) ---
        import pandas as pd

        if not isinstance(edges_gdf.index, pd.MultiIndex):
            raise ValueError("edges_gdf is not a MultiIndex — cannot build OSMnx graph.")

        # If names are stored as list, patch them as a tuple explicitly
        if tuple(edges_gdf.index.names) != ("u", "v", "key"):
            edges_gdf.index.set_names(("u", "v", "key"), inplace=True)

        print("[DEBUG] edges_gdf.index.names (patched):", edges_gdf.index.names)
        print("[DEBUG] edges_gdf.index type:", type(edges_gdf.index))
        print("[DEBUG] edges_gdf index sample:", edges_gdf.index[:3])

        # Small hack to override the internal OSMnx validator expecting tuple
        import osmnx.convert as convert
        old_validate = convert._validate_node_edge_gdfs

        def _patched_validate_node_edge_gdfs(gdf_nodes, gdf_edges):
            if isinstance(gdf_edges.index, pd.MultiIndex) and list(gdf_edges.index.names) == ["u", "v", "key"]:
                return  # bypass strict tuple check
            return old_validate(gdf_nodes, gdf_edges)

        convert._validate_node_edge_gdfs = _patched_validate_node_edge_gdfs

        # --- Build OSMnx graph safely ---
        G = ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs={"crs": TARGET_CRS})
        G = ox.simplify_graph(G)

        ox.save_graphml(G, self.graph_path)
        print(f"[INFO] Canonical graph saved: {self.graph_path}")
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


# Convenience entrypoint for external code
def load_real_nairobi_cbd_map(config: Dict) -> Optional[Dict]:
    try:
        loader = RealNairobiLoader(grid_size=config["grid"]["width"])
        return loader.load_all(build_graph=True)
    except Exception as e:
        print(f"[ERROR] Failed to load real Nairobi map: {e}")
        return None
