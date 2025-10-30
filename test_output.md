## Agent Goal Hierarchy (priority-based):
1. IMMEDIATE SURVIVAL: If hazard within danger_radius → flee opposite direction
2. SAFETY SEEKING: If no immediate threat → move toward lower-risk zones
3. DISPERSAL: If crowd density > threshold → seek less congested nodes
4. EVENTUAL EXIT: Only after lingering in "safe zones" → move toward major highway nodes

**Implementation Strategy:**
- Replace single goal with **dynamic weighted objective function**:
```
  objective = w_flee * flee_urgency 
            + w_safety * (1 - node_risk) 
            + w_disperse * (1 / crowd_density)
            + w_exit * exit_proximity  # LOW weight unless prolonged safety
```
- Exits should be **major highway nodes** identified from OSM data (Uhuru Hwy, Mombasa Rd intersections within bbox)
- Agents only prioritize exits after:
  - Being in simulation for T_min steps (e.g., 300 steps = ~5min)
  - AND current node risk < threshold (e.g., 0.2)

---

## Spawing Logic
**Proposed Logic:**
1. **Pre-compute valid spawn zones:**
   - Load `nairobi_buildings.geojson`
   - Create `spawn_mask.npy` where:
     - `1` = road node (from `nairobi_roads_*.geojson`)
     - `0` = building/obstacle
   - Store in `data/spawn_mask_100x100.npy`

2. **Cluster-based spawning on valid nodes:**
```
   For each cluster center:
     1. Find nearest valid road node
     2. Sample agents within radius on road network
     3. Use Dijkstra/BFS to ensure connected nodes only
```

3. **Alternative: Realistic congregation points:**
   - Identify landmarks from OSM (e.g., "Uhuru Park", "Parliament Road", "City Hall")
   - Spawn agents near these nodes (protests typically gather at symbolic locations)

---

### **Issue 5: Overcrowding on Graph Structure**

**Problem:** Previous cell capacity (`N_CELL_MAX = 6`) doesn't translate to nodes/edges.

**New Formulation:**
```
Node Capacity: Based on real road width
- Major highway node: cap = 20 agents
- Secondary road node: cap = 10 agents
- Alley/footpath node: cap = 5 agents

Edge Capacity: Flow constraint
- While moving on edge (u, v), max simultaneous agents = edge_capacity
- If edge full, agent queues at node u
def assign_edge_capacity(graph):
       for u, v, data in graph.edges(data=True):
           road_type = data.get('highway', 'unclassified')
           if road_type in ['primary', 'trunk']:
               data['capacity'] = 20
           elif road_type in ['secondary', 'tertiary']:
               data['capacity'] = 10
           else:
               data['capacity'] = 5
```

2. **Queue management at nodes:**
```
   If next_edge capacity reached:
       agent.wait_at_current_node()
       reconsider_path()  # May find alternate route
```

3. **Congestion feedback:**
   - High queue lengths → increase node risk score
   - Agents detect congestion → influence route choice

---

### **Issue 6: Monte Carlo Harm Calculation on Graphs**

**Current Uncertainty:** How to compute per-cell harm when movement is node-based?

**Proposed Approach:**
```
Two-level spatial representation:
1. Graph-level (nodes/edges): For agent pathfinding
2. Grid-level (100×100 cells): For visualization & Monte Carlo aggregation

Mapping Strategy:
- Each cell (i, j) maps to nearest graph node(s)
- Harm field computed as:
  
  P_harm(cell) = max over {nodes in cell}(
      base_harm(node) + hazard_contribution(node)
  )
  
  where:
    base_harm(node) = density(node) / capacity(node)
    hazard_contribution(node) = Σ decay(distance(hazard_k, node))