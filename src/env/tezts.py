from real_nairobi_loader import RealNairobiLoader
import numpy as np
import json

# 1. Initialize loader
loader = RealNairobiLoader(grid_size=100)

# 2. Run the full pipeline (build graph + precompute lookup)
result = loader.load_all(build_graph=True)
loader.precompute_cell_lookup(result["graph"], result["metadata"], result["obstacle_mask"])

# 3. If graph was created, compute cell→node mapping
if result["graph"] is not None:
    node_map = loader.precompute_cell_lookup(
        result["graph"], result["metadata"], result["obstacle_mask"]
    )

    print("\n=== TEST SUMMARY ===")
    print("Node map shape:", node_map.shape)
    print("Sample (top-left 5x5 region):")
    print(node_map[:5, :5])

    # Confirm file saved
    arr = np.load("data/cell_to_node.npy", allow_pickle=True)
    print("Loaded from file, shape:", arr.shape)
    print("Unique non-obstacle entries:", np.unique(arr[arr != -1])[:10])

else:
    print("[ERROR] Graph not created!")
