"""
protest_env.py - Core ProtestEnv Gymnasium environment

Implementation: Stylized digital twin for protest scenarios.
Grid-based ABM with:
- Discrete 2D grid (100x100 default)
- Heterogeneous agents (protesters, police)
- Hazard fields (gas diffusion, water cannon, shooting, )
- Deterministic seeding for Monte Carlo
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import yaml
from pathlib import Path
from affine import Affine
import pandas as pd
import json

from .agent import Agent, PoliceAgent, AgentState, GraphAgent
from .mixins.graph_movement_mixin import GraphMovementMixin
from .hazards import HazardField
from .hazard_manager import HazardManager


@dataclass
class GridMetadata:
    """Shared metadata for coordinate system alignment with CV module"""
    width: int
    height: int
    cell_size_m: float
    origin: str  # 'top_left'
    coordinate_system: str  # 'image'
    
    def to_dict(self) -> Dict:
        return {
            'width': self.width,
            'height': self.height,
            'cell_size_m': self.cell_size_m,
            'origin': self.origin,
            'coordinate_system': self.coordinate_system
        }


class ProtestEnv(gym.Env):
    """
    Stylized digital twin for protest scenarios.
    
    Grid-based ABM with:
    - Discrete 2D grid (100x100 default)
    - Heterogeneous agents (protesters, police)
    - Hazard fields (gas diffusion)
    - Deterministic seeding for Monte Carlo
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, config: Dict):
        """
        Initialize environment from configuration.
        
        Args:
            config: Dictionary with keys: grid, time, agents, hazards, simulation
        """
        super().__init__()
        
        self.config = config
        
        # Grid setup
        grid_cfg = config['grid']
        self.width = grid_cfg['width']
        self.height = grid_cfg['height']
        self.cell_size = grid_cfg['cell_size_m']
        
        # Grid metadata
        self.grid_metadata = GridMetadata(
            width=self.width,
            height=self.height,
            cell_size_m=self.cell_size,
            origin='top_left',
            coordinate_system='image'
        )
        
        # Time setup
        time_cfg = config['time']
        self.delta_t = time_cfg['delta_t']
        self.max_steps = time_cfg['max_steps']
        
        # Simulation parameters
        sim_cfg = config.get('simulation', {})
        self.base_seed = sim_cfg.get('base_seed', 123456)
        self.n_cell_max = sim_cfg.get('n_cell_max', 6)
        
        # State arrays (initialized in reset)
        self.occupancy_count = None
        self.obstacle_mask = None
        self.hazard_field = None

        # OSM and Graph-related state (if OSM loaded)
        self.osm_metadata = None
        self.buildings_gdf = None
        self.osm_graph = None
        self.affine = None
        self.cell_to_node = None
        self.node_to_xy = {}  # node_id -> (x, y) for visualization
        self.node_occupancy = {}  # node_id -> agent count
        self.osm_bounds = None  # optional: for visualization extents

        # Agent management
        self.agents: List[Agent] = []
        self.protesters: List[Agent] = []
        self.police_agents: List[PoliceAgent] = []
        
        self.exit_points = config['agents']['protesters']['goals'].get('exit_points', [])
        
        # If exit_points is the string "auto" or empty, set temporary defaults
        if not self.exit_points or (isinstance(self.exit_points, str) and self.exit_points == "auto"):
            mid_x = self.width // 2
            mid_y = self.height // 2
            self.exit_points = [(5, mid_y), (self.width-5, mid_y), 
                                (mid_x, 5), (mid_x, self.height-5)]

        # Simulation state
        self.step_count = 0
        self.rng = None  # Set in reset()
        self.events_log = []
        
        # Action/Observation spaces
        # For Monte Carlo aggregator, we use global observation
        self.observation_space = spaces.Dict({
            'hazard_concentration': spaces.Box(
                low=0, high=100, shape=(self.height, self.width), dtype=np.float32
            ),
            'occupancy_count': spaces.Box(
                low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
            ),
            'obstacle_mask': spaces.Box(
                low=0, high=1, shape=(self.height, self.width), dtype=np.uint8
            ),
            'time': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        # Action space (discrete moves for each agent - handled internally)
        # For evaluation, external controller can override
        self.action_space = spaces.Discrete(9)  # 0=STAY, 1-8=directions
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: RNG seed for deterministic rollouts
            options: Additional reset options
            
        Returns:
            observation: Initial observation dict
            info: Additional information
        """
        # Set seed for deterministic behavior
        if seed is None:
            seed = self.base_seed
        self.rng = np.random.default_rng(seed)
        
        # Reset step counter
        self.step_count = 0
        self.events_log = []
        
        # Initialize grid arrays (locked data types)
        self.occupancy_count = np.zeros((self.height, self.width), dtype=np.uint8)
        self.obstacle_mask = self._load_or_generate_obstacles()
        
        # Initialize hazard manager (gas + instant hazards)
        hazard_cfg = self.config.get('hazards', {})
        # Pass top-level hazards config and delta_t for gas init
        hm_cfg = hazard_cfg.copy()
        hm_cfg['delta_t'] = self.delta_t
        self.hazards = HazardManager(
            height=self.height, 
            width=self.width, 
            config=self.config, 
            rng=self.rng,
            cell_size_m=self.cell_size,
            obstacle_mask=self.obstacle_mask)
        
        # Backward compatibility: existing code expects self.hazard_field
        self.hazard_field = self.hazards.gas

        # NEW: Identify exit nodes (FIXED)
        if self.osm_graph is not None and self.osm_metadata is not None:
            try:
                from .real_nairobi_loader import identify_exit_nodes
            
                # CRITICAL FIX: Ensure metadata has correct structure
                if isinstance(self.osm_metadata, dict) and "bounds" in self.osm_metadata:
                    self.exit_nodes = identify_exit_nodes(
                        self.osm_graph,
                        self.osm_metadata,  # Now guaranteed to have "bounds"
                        self.affine,
                        (self.width, self.height)
                    )
                
                    # Update config for agent access
                    if "agents" in self.config and "protesters" in self.config["agents"]:
                        goals_cfg = self.config["agents"]["protesters"].setdefault("goals", {})
                        goals_cfg["exit_nodes"] = self.exit_nodes
                    
                        # CRITICAL: If exit_points is "auto", populate it with real coordinates
                        if goals_cfg.get("exit_points") == "auto":
                            primary = self.exit_nodes.get('primary', [])
                            secondary = self.exit_nodes.get('secondary', [])
                            all_exits = primary + secondary
                        
                            if all_exits:
                                # Extract ONLY grid_pos as simple tuples
                                exit_points_list = []
                                for e in all_exits:
                                    gp = e['grid_pos']
                                    exit_points_list.append((int(gp[0]), int(gp[1])))

                                goals_cfg["exit_points"] = exit_points_list
                                # CRITICAL: Also set self.exit_points
                                self.exit_points = exit_points_list
                                print(f"[INFO] Populated {len(exit_points_list)} exit points from graph")
                                
                            else:
                                print("[WARN] No exit nodes found; using grid edges")
                                self.exit_points = [(5, self.height//2), 
                                                   (self.width-5, self.height//2)]
                                goals_cfg["exit_points"] = self.exit_points

                else:
                    print("[WARN] Metadata missing 'bounds'; cannot identify exit nodes")
                    self.exit_nodes = {'primary': [], 'secondary': []}
                    self.exit_points = [(5, self.height//2), (self.width-5, self.height//2)]

                
            except Exception as e:
                print(f"[WARN] Failed to identify exit nodes: {e}")
                import traceback
                traceback.print_exc()
                self.exit_nodes = {'primary': [], 'secondary': []}
                self.exit_points = [(5, self.height//2), (self.width-5, self.height//2)]
        else:
            # Synthetic obstacles - use grid edges
            print("[WARN] No OSM data; using grid-edge fallback exits")
            mid_y = self.height // 2
            mid_x = self.width // 2
            self.exit_nodes = {
                'primary': [
                    {'grid_pos': (0, mid_y)},
                    {'grid_pos': (self.width - 1, mid_y)}
                ],
                'secondary': [
                    {'grid_pos': (mid_x, 0)},
                    {'grid_pos': (mid_x, self.height - 1)}
                ]
            }
            # Also set self.exit_points as simple tuples
            self.exit_points = [
                (0, mid_y),
                (self.width - 1, mid_y),
                (mid_x, 0),
                (mid_x, self.height - 1)
            ]


            # Propagate to agent goals config
            if "agents" in self.config and "protesters" in self.config["agents"]:
                goals_cfg = self.config["agents"]["protesters"].setdefault("goals", {})
                goals_cfg["exit_points"] = self.exit_points
                
                # sanitize all exit points (floats only)
                if isinstance(goals_cfg.get("exit_points"), list):
                    clean_exit_points = []
                    for ex in goals_cfg["exit_points"]:
                        if isinstance(ex, (list, tuple)):
                            try:
                               clean_exit_points.append((float(ex[0]), float(ex[1])))
                            except Exception:
                                continue
                        elif isinstance(ex, dict) and "grid_pos" in ex:
                            gx, gy = ex["grid_pos"]
                            clean_exit_points.append((float(gx), float(gy)))
                    goals_cfg["exit_points"] = clean_exit_points

                # safety net: ensure non-empty
                if not goals_cfg.get("exit_points"):
                    print("[WARN] No exit points set; injecting defaults")
                    goals_cfg["exit_points"] = [
                        (0.0, float(mid_y)),
                        (float(self.width - 1), float(mid_y)),
                        (float(mid_x), 0.0),
                        (float(mid_x), float(self.height - 1))
                    ]
        
        # After setting self.exit_points, add:
        print(f"\n[DEBUG] reset() completed exit_points setup:")
        print(f"  self.exit_points type: {type(self.exit_points)}")
        print(f"  self.exit_points length: {len(self.exit_points) if hasattr(self.exit_points, '__len__') else 'N/A'}")
        if isinstance(self.exit_points, (list, tuple)) and len(self.exit_points) > 0:
            print(f"  First 3 exit points: {self.exit_points[:3]}")
            for idx, ep in enumerate(self.exit_points[:3]):
                print(f"    [{idx}]: {ep} (type: {type(ep)})")
        print()


        # Spawn agents
        self._spawn_agents()

        # DEBUG 00: AGENT SPAWN ON OBSTACLE CELL
        self.check_spawned_on_obstacle()
        
        # Update occupancy grid
        self._update_occupancy_grid_simple()
        
        # Return observation and info
        obs = self._get_observation()
        info = {
            'step': self.step_count,
            'seed': seed,
            'n_agents': len(self.agents),
            'grid_metadata': self.grid_metadata.to_dict()
        }
        
        return obs, info
    
    def check_spawned_on_obstacle(self):
        """Check if any agents spawned on obstacle cells."""
        for agent in self.agents:
            x, y = agent.pos
            if self.obstacle_mask[y, x]:
                print(f"[DEBUG] Agent {agent.id} spawned on obstacle at {agent.pos} ({agent.agent_type})")
    
    def step(self, actions: Optional[Dict[int, int]] = None) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one simulation timestep.
    
        Args:
            actions: Optional dict mapping agent_id -> action (0-8)
                    If None, agents use internal decision logic
    
        Returns:
            observation: Updated observation
            reward: Reward signal (not used for Monte Carlo)
            terminated: Episode ended (all agents safe/incapacitated)
            truncated: Time limit reached
            info: Additional information including harm indicators
        """
        # 1. Compute agent actions (internal policy if not provided)
        if actions is None:
            actions = {}
            
            # DEBUG: Check exit_points state
            print(f"\n[DEBUG] Step {self.step_count}: Checking exit_points")
            print(f"  hasattr(self, 'exit_points'): {hasattr(self, 'exit_points')}")
            if hasattr(self, 'exit_points'):
                print(f"  type(self.exit_points): {type(self.exit_points)}")
                print(f"  len(self.exit_points): {len(self.exit_points) if isinstance(self.exit_points, (list, tuple)) else 'N/A'}")
                if isinstance(self.exit_points, (list, tuple)) and len(self.exit_points) > 0:
                    print(f"  First exit point: {self.exit_points[0]} (type: {type(self.exit_points[0])})")

            for agent in self.agents:
                # only process protesters with update_goal
                if hasattr(agent, 'update_goal'):  # Protesters have this, police don't
                    print(f"[DEBUG] Processing agent {agent.id} (type: {agent.agent_type})")
                    
                    # FIX: Only call _get_safe_exit_points if exit_points is broken
                    # Convert exit_points to simple list of tuples
                    if hasattr(self, 'exit_points') and isinstance(self.exit_points, (list, tuple)):
                        # Ensure all exit points are (x, y) tuples
                        safe_exit_points = []
                        for idx, ep in enumerate(self.exit_points):
                            print(f"  [DEBUG] Exit point {idx}: {ep} (type: {type(ep)})")
                            if isinstance(ep, (list, tuple)) and len(ep) >= 2:
                                safe_exit_points.append((int(ep[0]), int(ep[1])))
                            elif isinstance(ep, dict) and 'grid_pos' in ep:
                                gp = ep['grid_pos']
                                safe_exit_points.append((int(gp[0]), int(gp[1])))

                        print(f"  [DEBUG] Sanitized to {len(safe_exit_points)} exit points")

                        if not safe_exit_points:
                            print(f"  [WARN] Sanitization produced empty list - using fallback")
                            # Fallback if sanitization failed
                            safe_exit_points = [(5, 5), (self.width-5, 5), 
                                                (5, self.height-5), (self.width-5, self.height-5)]
                    else:
                        print(f"  [WARN] exit_points not valid - using fallback")
                        # No exit_points configured - use grid edges
                        safe_exit_points = [(5, 5), (self.width-5, 5), 
                                            (5, self.height-5), (self.width-5, self.height-5)]

                    print(f"  [DEBUG] Calling agent.update_goal with {len(safe_exit_points)} exits")
                    print(f"  [DEBUG] First safe exit: {safe_exit_points[0]}")

                    try:
                        # Now call update_goal with the correct signature
                        agent.update_goal(self, safe_exit_points)
                    except Exception as e:
                        print(f"  [ERROR] agent.update_goal failed: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                actions[agent.id] = agent.decide_action(self)

                # Only print for first agent to avoid spam
                if agent.id == 0:
                    print(f"[DEBUG] Breaking after first agent to avoid spam\n")
                    break  # Remove this break after debugging

            # --- Goal distribution diagnostics (added block) ---
            if self.step_count in {50, 100, 200}:
                goal_counts = {}
                for a in self.protesters:
                    g = tuple(a.goal)
                    goal_counts[g] = goal_counts.get(g, 0) + 1
                print(f"\n[Diagnostics] Step {self.step_count} Goal Distribution:")
                for g, count in goal_counts.items():
                    print(f"  Exit {g}: {count} agents")
    
        # 2. Execute movement with conflict resolution
        if any(hasattr(a, "current_node") for a in self.agents):
            self._execute_graph_movement(actions)
        else:
            self._execute_movement(actions)
    
        # 3. Update hazard field (diffusion, decay, sources)
        # Update all hazards (gas diffusion + instant hazard bookkeeping)
        self.hazards.update(self.delta_t)
        # keep the old alias
        self.hazard_field = self.hazards.gas
    
        # 3.5. NEW: Check stun recovery
        if hasattr(self.hazards, 'check_stun_recovery'):
            recovered = self.hazards.check_stun_recovery(self)
            if recovered:
                print(f"[INFO] {len(recovered)} agents recovered from stun")

        # 3.7. NEW FIX: Despawn protesters reaching exit points
        exited = []
        for agent in list(self.protesters):  # Copy since we might modify list
            if agent.state == AgentState.MOVING and agent.pos in [tuple(ep) for ep in self.exit_points]:
                agent.state = AgentState.SAFE
                exited.append(agent)
                # Remove from active grid
                self.agents.remove(agent)
                self.protesters.remove(agent)
                self.events_log.append({
                    'timestep': self.step_count,
                    'event_type': 'agent_exited',
                    'agent_id': agent.id,
                    'position': agent.pos
                })

        if exited:
            print(f"[INFO] {len(exited)} protesters exited the grid at this step.")

        # Recompute occupancy after despawning
        self._update_occupancy_grid_simple()

        # 4. Update agent exposures and harm
        harm_grid = self._update_agent_harm()
    
        # 5. Check termination conditions
        terminated, termination_reason = self._check_termination()
        truncated = self.step_count >= self.max_steps
    
        # 6. Increment step counter
        self.step_count += 1
    
        # 7. Construct observation and info
        obs = self._get_observation()
        reward = 0.0  # Not used for Monte Carlo evaluation
    
        info = {
            'step': self.step_count,
            'harm_grid': harm_grid,  # Binary indicators for I_i
            'exposure_grid': self._compute_exposure_grid(),
            'events': self.events_log[-10:],  # Last 10 events
            'termination_reason': termination_reason if terminated else None,
            'agent_states': self._get_agent_states_summary()
        }
    
        return obs, reward, terminated, truncated, info
    

    def _load_or_generate_obstacles(self) -> np.ndarray:
        """
        Load obstacle mask from real Nairobi CBD (if available) or generate synthetic.
        Returns:
            obstacle_mask: Boolean array (True = impassable)
        """
        obstacle_source = self.config['grid'].get('obstacle_source', 'generate')

        if obstacle_source == 'nairobi':
            try:
                from .real_nairobi_loader import load_real_nairobi_cbd_map, generate_spawn_mask
                result = load_real_nairobi_cbd_map(self.config)

                if result and result.get('is_real_osm'):
                    mask = result['obstacle_mask']
                    meta = result.get('metadata', {})
                    self.osm_metadata = meta
                    self.buildings_gdf = result.get('buildings_gdf')
                    self.osm_graph = result.get('graph')
                    print("[DEBUG] Graph node types:", {type(n) for n in self.osm_graph.nodes()})
                    self.affine = result.get('affine') or Affine(*meta.get("affine_transform", (1, 0, 0, 0, -1, 0)))

                    # Flip vertically if raster origin mismatch
                    if meta.get("row_origin", "top") == "top":
                        mask = np.flipud(mask)

                    # Optional: warn if obstacle coverage too high
                    coverage = 100 * mask.sum() / mask.size
                    if coverage > 70:
                        print(f"[WARN] High obstacle coverage ({coverage:.1f}%) – agent spawning may fail.")

                    print(f" Using REAL Nairobi CBD map ({coverage:.1f}% coverage).")

                    # Load cell→node mapping
                    cell_to_node_path = self.osm_metadata.get("cell_to_node_path", "data/cell_to_node.npy")
                    if Path(cell_to_node_path).exists():
                        self.cell_to_node = np.load(cell_to_node_path, allow_pickle=True)
                        print(f"[INFO] Loaded cell→node lookup ({self.cell_to_node.shape}).")
                    else:
                        print("[WARN] Missing cell→node lookup; graph movement disabled.")
                        self.cell_to_node = None

                    # Build node→xy for visualization
                    if self.osm_graph:
                        self.node_to_xy = {nid: (d["x"], d["y"]) for nid, d in self.osm_graph.nodes(data=True)}

                        # NEW: Generate and save spawn mask
                        self.spawn_mask = generate_spawn_mask(mask, self.osm_graph, self.cell_to_node)
                        np.save("data/spawn_mask_100x100.npy", self.spawn_mask)
                        print(f"[INFO] Spawn mask saved to data/spawn_mask_100x100.npy")
                    
                        # Build street name lookup
                        from .real_nairobi_loader import build_street_name_lookup
                        self. street_names= build_street_name_lookup(self.osm_graph)

                        # Save to JSON for reproducibility
                        with open("data/street_names.json", "w") as f:
                            json.dump(self.street_names, f, indent=2)
                        print(f"[INFO] Street name lookup saved ({len(self.street_names)} nodes)")

                    return mask

            except UnicodeEncodeError as e:
                print(f"[ERROR] Unicode encoding error (Windows console issue): {e}")
                print("[INFO] Falling back to synthetic obstacles")
                # Force fallback
                result = None
            
            except Exception as e:
                print(f"[ERROR] Failed to load real Nairobi CBD map: {e}")
                import traceback
                traceback.print_exc()
                # CRITICAL: Clear partial state to force clean fallback
                self.osm_metadata = None
                self.cell_to_node = None
                self.osm_graph = None
                self.spawn_mask = None

        # --- Fallback: Synthetic obstacles ---
        print(" Generating synthetic obstacles...")
        mask = np.zeros((self.height, self.width), dtype=bool)

        # Add border walls
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        # Add internal obstacles (buildings) - scaled for grid size
        scale_factor = self.width / 100  # Scale from original 100×100

        # Building 1 (northwest)
        mask[int(20*scale_factor):int(30*scale_factor),
            int(20*scale_factor):int(35*scale_factor)] = True

        # Building 2 (southeast)
        mask[int(60*scale_factor):int(75*scale_factor),
            int(50*scale_factor):int(70*scale_factor)] = True

        # Building 3 (northeast)
        mask[int(40*scale_factor):int(50*scale_factor),
            int(70*scale_factor):int(80*scale_factor)] = True

        # Additional building 4 (southwest)
        mask[int(65*scale_factor):int(78*scale_factor),
            int(15*scale_factor):int(28*scale_factor)] = True

        # Central plaza
        mask[int(45*scale_factor):int(55*scale_factor),
            int(45*scale_factor):int(55*scale_factor)] = True

        print(f"   Generated {mask.sum()} obstacle cells ({100*mask.sum()/mask.size:.1f}%) [synthetic]")
        # CRITICAL: Generate spawn mask for synthetic obstacles too
        self.spawn_mask = ~mask  # Simple inversion for synthetic
        print(f"[INFO] Synthetic spawn mask generated: {self.spawn_mask.sum()} valid cells")
        
        return mask

    def _spawn_agents(self):
        """Spawn protesters and police according to config."""
        self.agents = []
        self.protesters = []
        self.police_agents = []
        
        # Spawn protesters with heterogeneity
        protester_cfg = self.config['agents']['protesters']
        n_protesters = protester_cfg['count']
        spawn_cfg = protester_cfg['spawn']
        
        positions = self._generate_spawn_positions(
            n_agents=n_protesters,
            spawn_type=spawn_cfg['type'],
            spawn_params=spawn_cfg
        )
        
        # Check if heterogeneous types are configured
        if 'types' in protester_cfg:
            # Heterogeneous protesters
            agent_profiles = self._assign_agent_profiles(
                n_protesters,
                protester_cfg['types']
            )
        else:
            # Homogeneous protesters (fallback)
            agent_profiles = ['average'] * n_protesters
        
        # Create protesters with assigned profiles
        base_speed = protester_cfg.get('speed_m_s', 1.2)

        # Check if graph movement is enabled
        use_graph = self.osm_graph is not None and self.cell_to_node is not None

        if use_graph:
            from .agent import GraphAgent
        else:
            from .agent import Agent
        
        # create protesters
        for i, (pos, profile) in enumerate(zip(positions, agent_profiles)):
            # Clamp positions to grid
            x, y = map(int, np.clip(pos, [0, 0], [self.width - 1, self.height - 1]))
            goal_pos = self._assign_goal((x, y), protester_cfg.get('goals', {}))

            if use_graph:
                # Map to graph nodes
                node_id = self.cell_to_node[y, x]
                if node_id in (-1, None, "None", "nan") or pd.isna(node_id):
                    node_id = min(
                        self.osm_graph.nodes,
                        key=lambda n: (
                            (self.osm_graph.nodes[n]["x"] - x) ** 2 +
                            (self.osm_graph.nodes[n]["y"] - y) ** 2
                        )
                    )

                # Ensure goal_pos is valid
                # Defensive fallback if invalid goal returned
                valid_goal = (
                    isinstance(goal_pos, (tuple, list))
                    and len(goal_pos) == 2
                    and all(isinstance(v, (int, float, np.integer, np.floating)) for v in goal_pos)
                )

                if not valid_goal:
                    print(f"[WARN] Invalid goal for agent {i}: {goal_pos}. Assigning random exit.")
                    if hasattr(self, "exit_points") and len(self.exit_points) > 0:
                        # Use deterministic RNG for reproducibility
                        idx = int(self.rng.integers(0, len(self.exit_points)))
                        goal_pos = tuple(self.exit_points[idx])
                    else:
                        valid_cells = np.argwhere(getattr(self, "spawn_mask", np.ones((self.height, self.width), dtype=bool)))
                        if len(valid_cells) > 0:
                            yx = valid_cells[self.rng.integers(0, len(valid_cells))]
                            goal_pos = (int(yx[1]), int(yx[0]))
                        else:
                            goal_pos = (x, y)

                gx, gy = map(int, np.clip(goal_pos, [0, 0], [self.width - 1, self.height - 1]))

                # Now map to nearest graph nodes
                goal_node = None
                if hasattr(self, "cell_to_node") and self.cell_to_node is not None:
                    try:
                        goal_node = self.cell_to_node[gy, gx]
                    except Exception:
                        print(f"[WARN] Goal ({gx},{gy}) out of range; fallback to nearest node.")
                        goal_node = -1

                if goal_node in (-1, None, "None", "nan") or (isinstance(goal_node, float) and np.isnan(goal_node)):
                    goal_node = min(
                        self.osm_graph.nodes,
                        key=lambda n: (
                            (self.osm_graph.nodes[n]["x"] - gx) ** 2 +
                            (self.osm_graph.nodes[n]["y"] - gy) ** 2
                        )
                    )

                agent = GraphAgent(
                    agent_id=i,
                    agent_type='protester',
                    pos=(x, y),
                    goal=goal_pos,
                    speed=base_speed,
                    risk_tolerance=0.3,
                    rng=self.rng,
                    profile_name=profile
                )
                agent.current_node = node_id
                agent.goal_node = goal_node
            
            else:
                agent = Agent(
                    agent_id=i,
                    agent_type='protester',
                    pos=(x, y),
                    goal=goal_pos,
                    speed=base_speed,
                    risk_tolerance=0.3,
                    rng=self.rng,
                    profile_name=profile
                )
            self.agents.append(agent)
            self.protesters.append(agent)

        # Spawn police
        police_cfg = self.config['agents']['police']
        n_police = police_cfg['count']
        police_spawn_cfg = police_cfg['spawn']
        
        if police_spawn_cfg['type'] == 'fixed':
            police_positions_raw = police_spawn_cfg['positions']

            # NEW: Validate and relocate police spawns if on obstacles
            police_positions = []
            for pos in police_positions_raw:
                x, y = map(int, np.clip(pos, [0, 0], [self.width - 1, self.height - 1]))

                # Check if valid spawn location
                if self._validate_spawn_position(x, y):
                    police_positions.append((x, y))
                else:
                    # Find nearest valid position
                    print(f"[WARN] Police spawn {(x, y)} invalid; relocating...")
                    found_valid = False
                    for radius in range(1, 20):  # Search up to 20 cells away
                        for dx in range(-radius, radius + 1):
                            for dy in range(-radius, radius + 1):
                                if abs(dx) != radius and abs(dy) != radius:
                                    continue  # Only check perimeter
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    if self._validate_spawn_position(nx, ny):
                                        police_positions.append((nx, ny))
                                        print(f"         Relocated to {(nx, ny)}")
                                        found_valid = True
                                        break
                            if found_valid:
                                break
                        if found_valid:
                            break
                    if not found_valid:
                        # Last resort: use spawn mask
                        if hasattr(self, 'spawn_mask') and self.spawn_mask.any():
                            valid_cells = np.argwhere(self.spawn_mask)
                            idx = self.rng.integers(0, len(valid_cells))
                            ny, nx = valid_cells[idx]
                            police_positions.append((int(nx), int(ny)))
                            print(f"         Relocated to random valid cell {(nx, ny)}")

        else:
            police_positions = self._generate_spawn_positions(
                n_agents=n_police,
                spawn_type=police_spawn_cfg['type'],
                spawn_params=police_spawn_cfg
            )
        
        # create police agents
        for j, pos in enumerate(police_positions):
            x, y = map(int, np.clip(pos, [0, 0], [self.width - 1, self.height - 1]))
            
            if use_graph:
                # map police to graph nodes
                node_id = self.cell_to_node[y, x]
                if node_id in (-1, None, "None", "nan") or pd.isna(node_id):
                    node_id = min(
                        self.osm_graph.nodes,
                        key=lambda n: (
                            (self.osm_graph.nodes[n]["x"] - x) ** 2 +
                            (self.osm_graph.nodes[n]["y"] - y) ** 2
                        )
                    )
                # Create base police agent
                from .agent import PoliceAgent
                from .mixins.graph_movement_mixin import GraphMovementMixin 

                agent = PoliceAgent(
                    agent_id=len(self.agents),
                    pos=(x, y),
                    speed=police_cfg['speed_m_s'],
                    deploy_prob=police_cfg.get('deploy_prob', 0.01),
                    deploy_cooldown_max=police_cfg.get('deploy_cooldown', 50),
                    config=self.config,
                    rng=self.rng
                )
                agent.current_node = node_id
                agent.graph_decide_action = GraphMovementMixin.graph_decide_action.__get__(agent)
                agent._compute_neighbor_score = GraphMovementMixin._compute_neighbor_score.__get__(agent)

            else:
                from .agent import PoliceAgent 
                agent = PoliceAgent(
                    agent_id=len(self.agents),
                    pos=(x, y),
                    speed=police_cfg['speed_m_s'],
                    deploy_prob=police_cfg.get('deploy_prob', 0.01),
                    deploy_cooldown_max=police_cfg.get('deploy_cooldown', 50),
                    config=self.config,
                    rng=self.rng
                )
            self.agents.append(agent)
            self.police_agents.append(agent)
        
        
    def _assign_agent_profiles(self, n_agents: int, 
                               types_config: Dict) -> List[str]:
        """
        Assign agent profiles based on configured ratios.
        
        Args:
            n_agents: Total number of agents
            types_config: Dict with type names and ratios
            
        Returns:
            List of profile names for each agent
        """

        profiles = []

        # Extract types and ratios
        type_names = []
        ratios = []
        for type_name, type_cfg in types_config.items():
            type_names.append(type_name)
            ratios.append(type_cfg['ratio'])
        
        # Normalize ratios
        total_ratio = sum(ratios)
        ratios = [r / total_ratio for r in ratios]

        # Assign agents to types
        for i in range(n_agents):
            # Use RNG to sample from distribution
            profile = self.rng.choice(type_names, p=ratios)
            profiles.append(profile)
        
        return profiles
    
    def _validate_spawn_position(self, x: int, y: int) -> bool:
        """
        Validate if a position is safe for spawning.
        Returns True if position is valid, False otherwise.
        """
        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
    
        # Check spawn mask if available
        if hasattr(self, 'spawn_mask') and self.spawn_mask is not None:
            if not self.spawn_mask[y, x]:
                return False
        
        # Check obstacles
        if self.obstacle_mask[y, x]:
            return False
    
        # Check graph connectivity (if using graph)
        if hasattr(self, 'cell_to_node') and self.cell_to_node is not None:
            node_id = self.cell_to_node[y, x]
            if node_id in (-1, None, "None", "nan") or pd.isna(node_id):
                return False
            if str(node_id) not in self.osm_graph:
                return False
    
        return True
    
    def _generate_spawn_positions(self, n_agents: int, spawn_type: str, 
                                  spawn_params: Dict) -> List[Tuple[int, int]]:
        """
        Generate spawn positions for agents.

        Args:
            n_agents: Number of agents to spawn
            spawn_type: Type of spawn position generation
            spawn_params: Parameters for spawn position generation

        Returns:
            List of spawn positions for agents
        """
        positions = []

        # --- Normalize spawn_params and handle "auto" centers ---
        centers_param = spawn_params.get('centers', None)
        radius = spawn_params.get('radius', 8)

        if isinstance(centers_param, str) and centers_param.lower() == "auto":
            print("[INFO] Auto-generating cluster centers")
            # Use spawn_mask if available
            if hasattr(self, "spawn_mask") and self.spawn_mask is not None:
                valid_cells = np.argwhere(self.spawn_mask)
            else:
                valid_cells = np.argwhere(~self.obstacle_mask)

            if len(valid_cells) == 0:
                centers = [(self.width // 2, self.height // 2)]
            else:
                n_clusters = min(3, len(valid_cells))
                # Use suggest_spawn_centers if available
                if self.osm_graph is not None and hasattr(self, 'cell_to_node'):
                    from .real_nairobi_loader import suggest_spawn_centers
                    centers = suggest_spawn_centers(
                        self.osm_graph, 
                        self.spawn_mask if hasattr(self, 'spawn_mask') else ~self.obstacle_mask,
                        self.cell_to_node,
                        n_clusters=n_clusters
                    )
                else:
                    # Simple random selection
                    idxs = self.rng.choice(len(valid_cells), size=n_clusters, replace=False)
                    centers = [(int(valid_cells[i][1]), int(valid_cells[i][0])) for i in idxs]
        
            print(f"[INFO] Generated {len(centers)} spawn centers: {centers}")
        else:
            centers = centers_param or []
    
        # Normalize centers to list of (x, y) tuples
        normalized_centers = []
        for c in centers:
            if isinstance(c, (list, tuple)) and len(c) == 2:
                normalized_centers.append((int(c[0]), int(c[1])))
            elif isinstance(c, dict) and 'x' in c and 'y' in c:
                normalized_centers.append((int(c['x']), int(c['y'])))
        centers = normalized_centers

        if not centers:
            print("[WARN] No valid centers; using random spawn")
            # Fallback to random spawning
            if hasattr(self, "spawn_mask"):
                valid_cells = np.argwhere(self.spawn_mask)
                if len(valid_cells) > 0:
                    picks = self.rng.choice(len(valid_cells), size=min(n_agents, len(valid_cells)), replace=True)
                    return [(int(valid_cells[i][1]), int(valid_cells[i][0])) for i in picks]
        
            # Last resort: brute force
            while len(positions) < n_agents:
                x = int(self.rng.integers(1, self.width - 1))
                y = int(self.rng.integers(1, self.height - 1))
                if not self.obstacle_mask[y, x]:
                    positions.append((x, y))
            return positions

        # Cluster-based spawning
        if spawn_type == 'clusters':
            agents_per_cluster = max(1, n_agents // len(centers))
        
            for center in centers:
                cx, cy = center
                spawned = 0
                attempts = 0
            
                while spawned < agents_per_cluster and attempts < 1000:
                    angle = self.rng.uniform(0, 2 * np.pi)
                    r = self.rng.uniform(0, radius)
                    x = int(cx + r * np.cos(angle))
                    y = int(cy + r * np.sin(angle))
                
                    # Bounds check
                    if not (1 <= x < self.width - 1 and 1 <= y < self.height - 1):
                        attempts += 1
                        continue
                
                    # Check spawn validity
                    valid = False
                    if hasattr(self, 'spawn_mask') and self.spawn_mask is not None:
                        valid = self.spawn_mask[y, x]
                    else:
                        valid = not self.obstacle_mask[y, x]
                
                    if valid:
                        positions.append((x, y))
                        spawned += 1
                
                    attempts += 1
        
            # Fill remainder
            if hasattr(self, 'spawn_mask') and self.spawn_mask is not None:
                valid_cells = np.argwhere(self.spawn_mask)
                while len(positions) < n_agents and len(valid_cells) > 0:
                    idx = self.rng.integers(0, len(valid_cells))
                    y, x = valid_cells[idx]
                    positions.append((int(x), int(y)))
            else:
                while len(positions) < n_agents:
                    x = int(self.rng.integers(1, self.width - 1))
                    y = int(self.rng.integers(1, self.height - 1))
                    if not self.obstacle_mask[y, x]:
                        positions.append((x, y))
    
        return positions[:n_agents]
    
    def _assign_goal(self, pos: Tuple[int, int], goals_cfg: Dict) -> Tuple[int, int]:
        """
        Assign goal position to agent.

        Args:
            pos: Agent's current position
            goals_cfg: Goal configuration

        Returns:
            (x, y) goal position
        """
        strategy = goals_cfg.get('strategy', 'nearest_exit')
    
        # Handle "auto" or missing exit_points
        exit_points_raw = goals_cfg.get('exit_points', [[50, 95]])
    
        # If exit_points is the string "auto", resolve it to actual coordinates
        if isinstance(exit_points_raw, str) and exit_points_raw.lower() == "auto":
            # Use exit_nodes if they were identified
            if hasattr(self, 'exit_nodes') and self.exit_nodes:
                primary = self.exit_nodes.get('primary', [])
                secondary = self.exit_nodes.get('secondary', [])
                all_exits = primary + secondary
            
                if all_exits:
                    # Extract grid_pos from exit node dicts
                    exit_points_raw = [e['grid_pos'] for e in all_exits]
                else:
                    # Fallback to grid edges
                    print("[WARN] No exit nodes found; using grid edge fallback")
                    exit_points_raw = [
                        [5, 50],           # Western edge (Uhuru Highway proxy)
                        [self.width - 5, 50],  # Eastern edge
                        [50, 5],           # Southern edge
                        [50, self.height - 5]  # Northern edge
                    ]
            else:
                # Fallback if exit_nodes not set
                exit_points_raw = [[5, 50], [95, 50]]
    
        # SANITIZE exit_points (handle various input formats)
        clean_exit_points = []
        for ep in exit_points_raw:
            try:
                # Handle different input types
                if isinstance(ep, (list, tuple)) and len(ep) == 2:
                    x, y = int(ep[0]), int(ep[1])
                    clean_exit_points.append((x, y))
                elif isinstance(ep, dict) and 'x' in ep and 'y' in ep:
                    clean_exit_points.append((int(ep['x']), int(ep['y'])))
                else:
                    print(f"[WARN] Skipping invalid exit point format: {type(ep).__name__}")
            except (ValueError, TypeError, IndexError) as e:
                print(f"[WARN] Skipping invalid exit point {ep}: {e}")
    
        if len(clean_exit_points) == 0:
            print("[ERROR] No valid exit points after sanitization; using center fallback")
            clean_exit_points = [(self.width // 2, self.height // 2)]
    
        exit_points = clean_exit_points

        # Strategy: nearest exit
        if strategy == 'nearest_exit' or strategy == 'dynamic_weighted':
            distances = [np.hypot(pos[0] - ex[0], pos[1] - ex[1]) for ex in exit_points]
            nearest_idx = np.argmin(distances)
            return tuple(exit_points[nearest_idx])
    
        # Default fallback
        return tuple(exit_points[0])
    
    def _execute_movement(self, actions: Dict[int, int]):
        """
        Execute agent movement with conflict resolution, priority queuing,
        and occupancy control.

        Args:
            actions: Dict mapping agent_id -> action (0-8)
        """
        # 8-neighbor offsets: 0=STAY, 1=N, 2=S, 3=E, 4=W, 5=NE, 6=NW, 7=SE, 8=SW
        MOVE_OFFSETS = [
            (0, 0),    # 0: STAY
            (0, -1),   # 1: NORTH
            (0, 1),    # 2: SOUTH
            (1, 0),    # 3: EAST
            (-1, 0),   # 4: WEST
            (1, -1),   # 5: NORTHEAST
            (-1, -1),  # 6: NORTHWEST
            (1, 1),    # 7: SOUTHEAST
            (-1, 1)    # 8: SOUTHWEST
        ]

        move_requests = {}  # target_cell -> list of (agent, original_pos)
        PRIORITY = {'police': 0, 'medic': 1, 'protester': 2, 'bystander': 3}

        # === Step 1: Collect movement requests ===
        for agent in self.agents:
            # Skip waiting agents (they're delayed from previous congestion)
            if getattr(agent, "state", None) == AgentState.WAITING:
                # Optionally decay waiting timer if congestion clears
                agent.wait_timer = max(0, getattr(agent, "wait_timer", 0) - 1)
                # If timer is now 0, resume normal movement next step
                if agent.wait_timer == 0:
                    agent.state = AgentState.MOVING
                continue

            agent.move_accum += agent.speed

            if agent.move_accum >= 1.0 and agent.state == AgentState.MOVING:
                action = actions.get(agent.id, 0)
                dx, dy = MOVE_OFFSETS[action]
                target_x = agent.pos[0] + dx
                target_y = agent.pos[1] + dy

                # Validate bounds and obstacle mask
                if (0 <= target_x < self.width and
                    0 <= target_y < self.height and
                    not self.obstacle_mask[target_y, target_x]):

                    target_cell = (target_x, target_y)
                    move_requests.setdefault(target_cell, []).append((agent, agent.pos))

                agent.move_accum -= 1.0  # consume fractional accumulator

        # === Step 2: Resolve conflicts (per target cell) ===
        for target_cell, requests in move_requests.items():
            if len(requests) <= self.n_cell_max:
                # All can move freely
                for agent, _ in requests:
                    agent.pos = target_cell
            else:
                # Over-capacity: enforce priority queuing
                sorted_requests = sorted(
                    requests,
                    key=lambda r: (
                        PRIORITY.get(r[0].agent_type, 99),
                        -getattr(r[0], "wait_timer", 0),  # agents waiting longer get precedence
                        self.rng.random()
                    )
                )

                winners = sorted_requests[:self.n_cell_max]
                losers = sorted_requests[self.n_cell_max:]

                for agent, _ in winners:
                    agent.pos = target_cell
                    agent.state = AgentState.MOVING
                    agent.wait_timer = 0  # reset waiting time if they moved

                for agent, original_pos in losers:
                    agent.pos = original_pos
                    agent.state = AgentState.WAITING
                    agent.wait_timer = getattr(agent, "wait_timer", 0) + 1  # increment waiting time

        # === Step 3: Update occupancy ===
        self._update_occupancy_grid_simple()

        # === Step 4: Post-movement congestion diagnostics (non-relocating) ===
        overcrowded_cells = np.argwhere(self.occupancy_count > self.n_cell_max)
        for (y, x) in overcrowded_cells:
            agents_here = [a for a in self.agents if a.pos == (x, y)]
            if len(agents_here) > self.n_cell_max:
                # No teleportation — just mark excess as waiting
                sorted_agents = sorted(
                    agents_here,
                    key=lambda a: (
                        PRIORITY.get(a.agent_type, 99),
                        -getattr(a, "wait_timer", 0),
                        self.rng.random()
                    )
                )
                survivors = sorted_agents[:self.n_cell_max]
                displaced = sorted_agents[self.n_cell_max:]

                for a in survivors:
                    a.state = AgentState.MOVING
                    a.wait_timer = 0
                for a in displaced:
                    a.state = AgentState.WAITING
                    a.wait_timer = getattr(a, "wait_timer", 0) + 1

                print(f"[WARN] Cell ({x},{y}) overcrowded; {len(displaced)} agents waiting instead of relocating.")

        # === Step 5: Final occupancy update ===
        self._update_occupancy_grid_simple()

    def _execute_graph_movement(self, actions: Dict[int, Any]):
        """Execute agent movement on the OSM graph with occupancy control."""
        if self.osm_graph is None:
            print("[WARN] Graph movement called without graph.")
            return

        move_requests = {}
        PRIORITY = {'police': 0, 'medic': 1, 'protester': 2}

        # Step 1: Gather requests
        for agent in self.agents:
            if not hasattr(agent, "current_node"):
                continue  # skip grid agents

            next_node = actions.get(agent.id, agent.current_node)
            
            # Normalize node IDs to string for consistent comparison
            if isinstance(next_node, (bytes, np.generic)):
                next_node = next_node.item() if hasattr(next_node, "item") else next_node
            next_node = str(next_node)
            agent.current_node = str(agent.current_node)

            # Skip if invalid or same node
            if next_node == agent.current_node:
                continue
            if next_node not in self.osm_graph:
                continue  # invalid

            move_requests.setdefault(next_node, []).append(agent)

        # Step 2: Resolve congestion
        for node_id, agents_here in move_requests.items():
            if len(agents_here) <= self.n_cell_max:
                for a in agents_here:
                    a.current_node = node_id
                    try:
                        a.pos = self._node_to_cell(node_id)
                    except Exception:
                        print(f"[WARN] Failed to map node {node_id} to cell for agent {a.id}")
            else:
                # Over-capacity: enforce priority queuing
                sorted_agents = sorted(
                    agents_here,
                    key=lambda a: (PRIORITY.get(a.agent_type, 99), self.rng.random())
                )
                # Allow top N_CELL_MAX to move
                for a in sorted_agents[:self.n_cell_max]:
                    a.current_node = node_id
                    try:
                        a.pos = self._node_to_cell(node_id)
                    except Exception:
                        print(f"[WARN] Failed to map node {node_id} to cell for agent {a.id}")
                # Others wait
                for a in sorted_agents[self.n_cell_max:]:
                    a.state = AgentState.WAITING
                
                # info log; Use street name lookup
                street_name = self.street_names.get(str(node_id), f"node {node_id}") if hasattr(self, 'street_names') else f"node {node_id}"
                print(f"[INFO] {len(sorted_agents) - self.n_cell_max} agents waiting at {street_name} (node {node_id})")

    # Final occupancy update
        # Final occupancy update
        self._update_node_occupancy()

    def _node_to_cell(self, node_id: str) -> Tuple[int, int]:
        """Convert node coordinates to grid cell indices for raster overlays."""
        if self.affine is None:
            return (0, 0)
        x, y = self.node_to_xy[node_id]
        col, row = ~self.affine * (x, y)
        return int(col), int(row)

    def _update_node_occupancy(self):
        self.node_occupancy.clear()
        for a in self.agents:
            if hasattr(a, "current_node"):
                n = a.current_node
                self.node_occupancy[n] = self.node_occupancy.get(n, 0) + 1


    def _update_occupancy_grid_simple(self):
        """
        Update the occupancy count grid from current agent positions.

        This version enforces safety, avoids out-of-bounds increments,
        and logs overcrowding only once per cell per step to prevent spam.
        """
        # Reset occupancy map
        self.occupancy_count.fill(0)

        # Accumulate per-cell occupancy safely
        for agent in self.agents:
            x, y = agent.pos
            if 0 <= x < self.width and 0 <= y < self.height:
                self.occupancy_count[y, x] += 1
            else:
                # Safety guard: out-of-bound positions (should never happen)
                print(f"[ERROR] Agent {agent.id} at invalid position {agent.pos}")
                # Optionally re-locate the agent safely
                fx, fy = self._find_free_cell()
                agent.pos = (fx, fy)
                self.occupancy_count[fy, fx] += 1

        # === Diagnostic Section ===
        max_occupancy = self.occupancy_count.max()
        if max_occupancy > self.n_cell_max:
            overcrowded = np.argwhere(self.occupancy_count > self.n_cell_max)
            msg_lines = [
                f"[WARN] {len(overcrowded)} cells exceed N_CELL_MAX={self.n_cell_max}"
            ]
            for y, x in overcrowded[:10]:  # Limit to first 10 to avoid spam
                msg_lines.append(f"   Cell ({x},{y}) has {self.occupancy_count[y, x]} agents")
            print("\n".join(msg_lines))

            # Optional: record in event log for later analysis
            for y, x in overcrowded:
                self.events_log.append({
                    'timestep': self.step_count,
                    'event_type': 'overcrowding_warning',
                    'cell': (x, y),
                    'occupancy': int(self.occupancy_count[y, x])
                })

    def _find_free_cell(self):
        """Helper to find a random free cell (not obstacle, below N_CELL_MAX)."""
        while True:
            x = self.rng.integers(1, self.width - 1)
            y = self.rng.integers(1, self.height - 1)
            if not self.obstacle_mask[y, x] and self.occupancy_count[y, x] < self.n_cell_max:
                return x, y
    
    def _update_agent_harm(self) -> np.ndarray:
        """
        Update agent harm status and return binary harm grid.
        
        Returns:
            harm_grid: Boolean array (True if any harm occurred in cell)
        """
        harm_grid = np.zeros((self.height, self.width), dtype=bool)
        
        for agent in self.agents:
            x, y = agent.pos
            concentration = self.hazard_field.concentration[y, x]
            
            # Update agent harm (dual-purpose model)
            harm_occurred = agent.update_harm(
                concentration=concentration,
                k_harm=self.hazard_field.k_harm,
                delta_t=self.delta_t,
                rng=self.rng
            )
            
            if harm_occurred:
                harm_grid[y, x] = True
            
            # Check incapacitation
            if agent.cumulative_harm >= self.config['hazards']['gas'].get('H_crit', 5.0):
                agent.state = AgentState.INCAPACITATED
        
        return harm_grid
    
    def _compute_exposure_grid(self) -> np.ndarray:
        """Compute cumulative exposure time in each cell."""
        exposure_grid = np.zeros((self.height, self.width), dtype=np.float32)
        for agent in self.agents:
            x, y = agent.pos
            exposure_grid[y, x] += agent.cumulative_harm
        return exposure_grid
    
    def _check_termination(self) -> Tuple[bool, Optional[str]]:
        """
        Check if episode should terminate.
        
        Returns:
            (is_terminated, reason)
        """
        # Count agent states (use AgentState enum)
        active = sum(a.state == AgentState.MOVING for a in self.protesters)
        safe = sum(a.state == AgentState.SAFE for a in self.protesters)
        incapacitated = sum(a.state == AgentState.INCAPACITATED for a in self.protesters)
    
        # All protesters either safe or incapacitated
        if active == 0:
            return True, "all_protesters_done"
    
        # Mass casualty (>80% incapacitated)
        if len(self.protesters) > 0:
            incap_rate = incapacitated / len(self.protesters)
            if incap_rate > 0.8:
                return True, "mass_casualty"
    
        return False, None
    
    def _get_observation(self) -> Dict:
        """Construct observation dict."""
        return {
            'hazard_concentration': self.hazard_field.concentration.copy(),
            'occupancy_count': self.occupancy_count.copy(),
            'obstacle_mask': self.obstacle_mask.astype(np.uint8),
            'time': np.array([self.step_count * self.delta_t], dtype=np.float32)
        }
    
    def _get_agent_states_summary(self) -> Dict:
        """Get summary of agent states for logging."""
        return {
            'n_moving': sum(a.state == 'moving' for a in self.agents),
            'n_incapacitated': sum(a.state == 'incapacitated' for a in self.agents),
            'n_safe': sum(a.state == 'safe' for a in self.agents),
            'mean_harm': np.mean([a.cumulative_harm for a in self.agents])
        }
    
    def render(self, mode='human'):
        """Basic rendering (defer fancy viz)."""
        if mode == 'human':
            print(f"Step {self.step_count}: {len(self.agents)} agents, "
                  f"{sum(a.state == 'moving' for a in self.agents)} moving")
    
    def close(self):
        """Cleanup resources."""
        pass


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config