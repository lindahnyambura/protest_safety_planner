import networkx as nx
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
from pyproj import Transformer

# === Load data ===
G = nx.read_graphml("data/nairobi_walk.graphml")
cell_to_node = np.load("data/cell_to_node.npy", allow_pickle=True)

# === Set up CRS transformer (UTM → WGS84 lat/lon) ===
# Try EPSG:32737 (WGS 84 / UTM zone 37S), the most likely CRS for Nairobi OSM
transformer = Transformer.from_crs("EPSG:32737", "EPSG:4326", always_xy=True)

# === Known landmarks near your bbox ===
landmarks = {
    "bus station": (-1.2869, 36.8275),
    "national archives": (-1.2845, 36.8253),
    "uhuru park": (-1.2885, 36.8128),
    "kicc": (-1.2888, 36.8234),
    "teleposta towers": (-1.2837, 36.8200),
    "times tower": (-1.2882, 36.8231),
    "railways": (-1.2917, 36.8286),
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# === Map each node to nearest landmark ===
node_info = []
for node_id, data in G.nodes(data=True):
    x = float(data.get("x", data.get("lon", 0)))
    y = float(data.get("y", data.get("lat", 0)))

    # Convert to lat/lon if data are in meters
    lon, lat = transformer.transform(x, y)

    # Find closest known landmark
    closest_landmark, min_dist = None, float("inf")
    for name, (lt, ln) in landmarks.items():
        d = haversine(lat, lon, lt, ln)
        if d < min_dist:
            closest_landmark, min_dist = name, d

    node_info.append((node_id, lat, lon, closest_landmark, round(min_dist)))

print(f"Graph has {len(G.nodes)} nodes.")
print("Sample of 10 random nodes:\n")

for node_id, lat, lon, landmark, dist in random.sample(node_info, min(10, len(node_info))):
    print(f"{node_id:>10} | ({lat:.6f}, {lon:.6f}) | nearest: {landmark:20s} ({dist} m)")

mapping = {landmark: node_id for node_id, _, _, landmark, dist in node_info if dist < 150}
print("\nSuggested frontend NODE_MAPPINGS:")
for name, nid in mapping.items():
    print(f'  "{name}": "{nid}",')
