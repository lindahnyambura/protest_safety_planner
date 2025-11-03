"""
protest_env.py - Core ProtestEnv Gymnasium environment

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
import networkx as nx

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
    
    def __init__(self, config: Dict, verbose: bool = False):
        """
        Initialize environment from configuration.
        
        Args:
            config: Dictionary with keys: grid, time, agents, hazards, simulation
        """
        super().__init__()
        
        self.verbose = verbose
        
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
        from src.utils.logging_config import logger
    
        # Set seed
        if seed is None:
            seed = self.base_seed
        self.rng = np.random.default_rng(seed)
    
        # Reset counters
        self.step_count = 0
        self.events_log = []
    
        logger.info(f"\n{'='*60}")
        logger.info(f"[RESET] Initializing environment (seed={seed})")
        logger.info(f"{'='*60}")
    
        # Initialize grids
        self.occupancy_count = np.zeros((self.height, self.width), dtype=np.uint8)
        self.obstacle_mask = self._load_or_generate_obstacles()
    
        # Load OSM graph
        if self.osm_graph is not None:
            from .real_nairobi_loader import assign_node_capacities
            self.osm_graph = assign_node_capacities(self.osm_graph)
            logger.info(f"[INFO] OSM graph: {len(self.osm_graph.nodes)} nodes, {len(self.osm_graph.edges)} edges")
        
            # Validate graph connectivity
            self.validate_and_repair_graph()
    
        # Initialize hazard manager
        self.hazards = HazardManager(
            height=self.height, 
            width=self.width, 
            config=self.config, 
            rng=self.rng,
            cell_size_m=self.cell_size,
            obstacle_mask=self.obstacle_mask
        )
        self.hazard_field = self.hazards.gas

        # Identify exit nodes
        self._setup_exit_points()
    
        # Spawn agents
        self._spawn_agents()
    
        # Validate spawns
        self.check_spawned_on_obstacle()
        self._update_occupancy_grid_simple()
    
        # Initialize node occupancy (graph mode)
        if self.osm_graph is not None:
            self.node_occupancy = {}
            for agent in self.agents:
                if hasattr(agent, 'current_node'):
                    node_id = str(agent.current_node)
                    self.node_occupancy[node_id] = self.node_occupancy.get(node_id, 0) + 1
        
            occupied_nodes = sum(1 for occ in self.node_occupancy.values() if occ > 0)
            max_occ = max(self.node_occupancy.values()) if self.node_occupancy else 0
        
            logger.info(f"[INFO] Initial occupancy: {occupied_nodes} nodes, max {max_occ} agents/node")
        
            # Warn about overcrowding
            overcrowded = [(nid, occ) for nid, occ in self.node_occupancy.items() 
                            if occ > self.osm_graph.nodes[nid].get('capacity', 6)]
        
            if overcrowded:
                logger.minimal(f"[WARN] {len(overcrowded)} nodes overcrowded at spawn")
                for nid, occ in overcrowded[:3]:
                    cap = self.osm_graph.nodes[nid].get('capacity', 6)
                    street = self.street_names.get(nid, nid) if hasattr(self, 'street_names') else nid
                    logger.debug(f"  {street}: {occ}/{cap}")
    
        logger.info(f"[RESET] Complete: {len(self.agents)} agents spawned\n")
    
        # Return observation
        obs = self._get_observation()
        info = {
            'step': self.step_count,
            'seed': seed,
            'n_agents': len(self.agents),
            'grid_metadata': self.grid_metadata.to_dict()
        }
    
        return obs, info

    def _setup_exit_points(self):
        """Setup exit points for protesters (extracted for clarity)."""
        from src.utils.logging_config import logger
    
        if self.osm_graph is not None and self.osm_metadata is not None:
            try:
                from .real_nairobi_loader import identify_exit_nodes
            
                if isinstance(self.osm_metadata, dict) and "bounds" in self.osm_metadata:
                    self.exit_nodes = identify_exit_nodes(
                        self.osm_graph,
                        self.osm_metadata,
                        self.affine,
                        (self.width, self.height)
                    )
                
                    # Update config
                    if "agents" in self.config and "protesters" in self.config["agents"]:
                        goals_cfg = self.config["agents"]["protesters"].setdefault("goals", {})
                        goals_cfg["exit_nodes"] = self.exit_nodes
                    
                        if goals_cfg.get("exit_points") == "auto":
                            primary = self.exit_nodes.get('primary', [])
                            secondary = self.exit_nodes.get('secondary', [])
                            all_exits = primary + secondary
                        
                            if all_exits:
                                exit_points_list = [
                                    (int(e['grid_pos'][0]), int(e['grid_pos'][1]))
                                    for e in all_exits
                                ]
                                goals_cfg["exit_points"] = exit_points_list
                                self.exit_points = exit_points_list
                                logger.info(f"[INFO] Identified {len(exit_points_list)} exit points")
                            else:
                                logger.minimal("[WARN] No exit nodes found; using grid edges")
                                self.exit_points = [(5, self.height//2), (self.width-5, self.height//2)]
                                goals_cfg["exit_points"] = self.exit_points
                else:
                    logger.minimal("[WARN] Metadata missing 'bounds'")
                    self.exit_nodes = {'primary': [], 'secondary': []}
                    self.exit_points = [(5, self.height//2), (self.width-5, self.height//2)]
                
            except Exception as e:
                logger.minimal(f"[WARN] Failed to identify exit nodes: {e}")
                self.exit_nodes = {'primary': [], 'secondary': []}
                self.exit_points = [(5, self.height//2), (self.width-5, self.height//2)]
        else:
            # Synthetic mode
            logger.debug("[INFO] Using synthetic exit points")
            mid_y, mid_x = self.height // 2, self.width // 2
            self.exit_points = [(5, mid_y), (self.width-1, mid_y), (mid_x, 5), (mid_x, self.height-1)]
        
            # Update config
            if "agents" in self.config and "protesters" in self.config["agents"]:
                goals_cfg = self.config["agents"]["protesters"].setdefault("goals", {})
                goals_cfg["exit_points"] = self.exit_points

    def check_spawned_on_obstacle(self):
        """Validate agent spawn locations (silent unless errors found)."""
        from src.utils.logging_config import logger
    
        errors = []
        for agent in self.agents:
            if hasattr(agent, 'current_node'):
                continue  # Graph agents validated separately
        
            x, y = agent.pos
            if not (0 <= x < self.width and 0 <= y < self.height):
                errors.append(f"Agent {agent.id} out of bounds: {agent.pos}")
            elif self.obstacle_mask[y, x]:
                errors.append(f"Agent {agent.id} on obstacle: {agent.pos}")
    
        if errors:
            logger.minimal(f"[WARN] {len(errors)} spawn validation errors:")
            for err in errors[:3]:  # Show first 3
                logger.debug(f"  {err}")
    
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
        # Import logger (add at top of file: from src.utils.logging_config import logger)
        from src.utils.logging_config import logger
        logger.set_step(self.step_count)
        
        # 1. Compute agent actions (internal policy if not provided)
        if actions is None:
            actions = {}
            
            # Sanitize exit points once (not per agent)
            if hasattr(self, 'exit_points') and isinstance(self.exit_points, (list, tuple)):
                safe_exit_points = []
                for ep in self.exit_points:
                    if isinstance(ep, (list, tuple)) and len(ep) >= 2:
                        safe_exit_points.append((int(ep[0]), int(ep[1])))
                    elif isinstance(ep, dict) and 'grid_pos' in ep:
                        gp = ep['grid_pos']
                        safe_exit_points.append((int(gp[0]), int(gp[1])))

                if not safe_exit_points:
                    logger.minimal("[WARN] No valid exit points; using fallback")
                    safe_exit_points = [(5, 5), (self.width-5, 5), 
                                        (5, self.height-5), (self.width-5, self.height-5)]
            else:
                safe_exit_points = [(5, 5), (self.width-5, 5), 
                                    (5, self.height-5), (self.width-5, self.height-5)]
            
            # Compute actions for all agents
            for agent in self.agents:
                if hasattr(agent, 'update_goal'):
                    try:
                        agent.update_goal(self, safe_exit_points)
                    except Exception as e:
                        logger.minimal(f"[ERROR] agent.update_goal failed for agent {agent.id}: {e}")
                        raise
                
                actions[agent.id] = agent.decide_action(self)

        # 2. Execute movement
        if any(hasattr(a, "current_node") for a in self.agents):
            self._execute_graph_movement(actions)
        else:
            self._execute_movement(actions)

        # 3. Update hazards
        self.hazards.update(self.delta_t)
        self.hazard_field = self.hazards.gas

        # 4. Check stun recovery
        if hasattr(self.hazards, 'check_stun_recovery'):
            recovered = self.hazards.check_stun_recovery(self)
            if recovered:
                logger.verbose(f"[INFO] {len(recovered)} agents recovered from stun")

        # 5. Despawn protesters reaching exits
        exited = []
        for agent in list(self.protesters):
            if agent.state == AgentState.MOVING and agent.pos in [tuple(ep) for ep in self.exit_points]:
                agent.state = AgentState.SAFE
                exited.append(agent)
                self.agents.remove(agent)
                self.protesters.remove(agent)
                self.events_log.append({
                    'timestep': self.step_count,
                    'event_type': 'agent_exited',
                    'agent_id': agent.id,
                    'position': agent.pos
                })

        if exited:
            logger.info(f"[INFO] {len(exited)} protesters exited")
        
        # Recompute occupancy
        self._update_occupancy_grid_simple()

        # 6. Update agent harm
        harm_grid = self._update_agent_harm()

        # 7. Check termination
        terminated, termination_reason = self._check_termination()
        truncated = self.step_count >= self.max_steps

        # 8. Increment step counter
        self.step_count += 1

        # 9. Construct observation and info
        obs = self._get_observation()
        reward = 0.0

        info = {
            'step': self.step_count,
            'harm_grid': harm_grid,
            'exposure_grid': self._compute_exposure_grid(),
            'events': self.events_log[-10:],
            'termination_reason': termination_reason if terminated else None,
            'agent_states': self._get_agent_states_summary()
        }

        # Congestion metrics
        congestion_events = [
            e for e in self.events_log
            if e.get('event_type') == 'node_congestion'
        ]
        info['congestion_events'] = len(congestion_events)
        info['avg_queue_length'] = float(np.mean([
            e.get('queued', 0) for e in congestion_events
        ])) if congestion_events else 0.0

        # CONSOLIDATED PERIODIC LOGGING
        if self.step_count % 20 == 0:
            moving = sum(1 for a in self.protesters if a.state == AgentState.MOVING)
            waiting = sum(1 for a in self.protesters if a.state == AgentState.WAITING)
            safe = sum(1 for a in self.protesters if a.state == AgentState.SAFE)
            incapacitated = sum(1 for a in self.protesters if a.state == AgentState.INCAPACITATED)
        
            logger.info(f"\n[Step {self.step_count}] Agent States:")
            logger.info(f"  Moving: {moving}, Waiting: {waiting}, Safe: {safe}, Incap: {incapacitated}")
        
            if getattr(self, "osm_graph", None):
                unique_nodes = len({
                    getattr(a, 'current_node', None)
                    for a in self.agents
                    if getattr(a, 'current_node', None) is not None
                })
                logger.info(f"  Occupied nodes: {unique_nodes}/{len(self.osm_graph.nodes)}")
            
            peak_hazard = float(np.max(getattr(self.hazard_field, "concentration", [0])))
            logger.info(f"  Peak hazard: {peak_hazard:.2f}")
        
            if hasattr(self, "police_agents") and self.police_agents:
                deployments = len([e for e in self.events_log 
                                if e.get('event_type') in ['gas_deployment', 'water_cannon']])
                logger.info(f"  Hazard deployments: {deployments}")
            
        # Verbose diagnostics (every 20 steps)
        if self.step_count % 20 == 0:
            logger.verbose(f"\n{'='*60}")
            logger.verbose(f"[DIAGNOSTIC] Step {self.step_count}")
        
            # Action distribution
            action_counts = {}
            for action in actions.values():
                action_counts[action] = action_counts.get(action, 0) + 1
        
            logger.verbose("\nAction Distribution:")
            for action, count in sorted(action_counts.items())[:5]:
                logger.verbose(f"  Action {action}: {count} agents")
        
            # Node occupancy (top 3)
            if hasattr(self, 'node_occupancy'):
                occupied = [(nid, occ) for nid, occ in self.node_occupancy.items() if occ > 0]
                occupied.sort(key=lambda x: x[1], reverse=True)
            
                logger.verbose("\nTop Occupied Nodes:")
                for nid, occ in occupied[:3]:
                    cap = self.osm_graph.nodes[nid].get('capacity', 6)
                    street = self.street_names.get(str(nid), nid) if hasattr(self, 'street_names') else nid
                    logger.verbose(f"  {street}: {occ}/{cap}")
        
            logger.verbose(f"{'='*60}\n")
        
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
                
                # CRITICAL FIX: Read-only mode for parallel execution
                # Check if we're in a parallel worker by looking for joblib markers

                import os
                is_parallel_worker = 'JOBLIB_MULTIPROCESSING' in os.environ or \
                                    hasattr(os, '_is_joblib_worker')
                
                # If parallel worker, only READ existing data (never write)
                if is_parallel_worker:
                    result = load_real_nairobi_cbd_map(self.config, read_only=True)
                else:
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

                    # CRITICAL FIX: Safe cell_to_node loading
                    cell_to_node_path = self.osm_metadata.get("cell_to_node_path", "data/cell_to_node.npy")
                    
                    if Path(cell_to_node_path).exists():
                        try:
                            # Add retry logic for corrupted reads
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    self.cell_to_node = np.load(cell_to_node_path, allow_pickle=True)
                                    print(f"[INFO] Loaded cell→node lookup ({self.cell_to_node.shape}).")
                                    break
                                except (EOFError, ValueError) as e:
                                    if attempt < max_retries - 1:
                                        import time
                                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                                        continue
                                    else:
                                        raise Exception(f"Failed to load cell_to_node after {max_retries} attempts: {e}")
                        except Exception as e:
                            print(f"[ERROR] cell_to_node loading failed: {e}")
                            print("[WARN] Falling back to synthetic mode")
                            return self._generate_synthetic_obstacles()
                    else:
                        print("[WARN] Missing cell→node lookup; graph movement disabled.")
                        self.cell_to_node = None

                    # Build node_to_xy for visualization
                    if self.osm_graph:
                        self.node_to_xy = {
                            str(nid): (d["x"], d["y"]) 
                            for nid, d in self.osm_graph.nodes(data=True)
                        }
                    
                        # === CRITICAL FIX: Load spawn mask (don't regenerate) ===
                        spawn_mask_path = Path("data/spawn_mask_200x200.npy")
                        if spawn_mask_path.exists() and not is_parallel_worker:
                            try:
                                self.spawn_mask = np.load(spawn_mask_path)
                                print(f"[INFO] Loaded spawn mask: {self.spawn_mask.sum()} valid cells")
                            except Exception as e:
                                print(f"[WARN] Failed to load spawn mask, regenerating: {e}")
                                self.spawn_mask = generate_spawn_mask(mask, self.osm_graph, self.cell_to_node)
                                np.save(spawn_mask_path, self.spawn_mask)
                        elif spawn_mask_path.exists() and is_parallel_worker:
                            # Parallel workers: read-only
                            self.spawn_mask = np.load(spawn_mask_path)
                        else:
                            # First time: generate and save
                            self.spawn_mask = generate_spawn_mask(mask, self.osm_graph, self.cell_to_node)
                            if not is_parallel_worker:
                                np.save(spawn_mask_path, self.spawn_mask)
                    
                        # Build street name lookup (cached)
                        street_names_path = Path("data/street_names.json")
                        if street_names_path.exists():
                            import json
                            with open(street_names_path, 'r') as f:
                                self.street_names = json.load(f)
                            print(f"[INFO] Loaded street names ({len(self.street_names)} nodes)")
                        elif not is_parallel_worker:
                            from .real_nairobi_loader import build_street_name_lookup
                            self.street_names = build_street_name_lookup(self.osm_graph)
                            with open(street_names_path, 'w') as f:
                                json.dump(self.street_names, f, indent=2)
                            print(f"[INFO] Built and saved street names ({len(self.street_names)} nodes)")
                        else:
                            # Parallel worker without cached data: use empty dict
                            self.street_names = {}

                    return mask

            except Exception as e:
                print(f"[ERROR] Failed to load real Nairobi CBD map: {e}")
                import traceback
                traceback.print_exc()
                # CRITICAL: Clear partial state
                self.osm_metadata = None
                self.cell_to_node = None
                self.osm_graph = None
                self.spawn_mask = None

        # Fallback: Synthetic obstacles
        return self._generate_synthetic_obstacles()


    def _generate_synthetic_obstacles(self) -> np.ndarray:
        """Helper method for synthetic obstacle generation."""
        print(" Generating synthetic obstacles...")
        mask = np.zeros((self.height, self.width), dtype=bool)

        # Add border walls
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        # Add internal obstacles (scaled for grid size)
        scale_factor = self.width / 100

        # Building 1 (northwest)
        mask[int(20*scale_factor):int(30*scale_factor),
            int(20*scale_factor):int(35*scale_factor)] = True

        # Building 2 (southeast)
        mask[int(60*scale_factor):int(75*scale_factor),
            int(50*scale_factor):int(70*scale_factor)] = True

        # Building 3 (northeast)
        mask[int(40*scale_factor):int(50*scale_factor),
            int(70*scale_factor):int(80*scale_factor)] = True

        # Building 4 (southwest)
        mask[int(65*scale_factor):int(78*scale_factor),
            int(15*scale_factor):int(28*scale_factor)] = True

        # Central plaza
        mask[int(45*scale_factor):int(55*scale_factor),
            int(45*scale_factor):int(55*scale_factor)] = True

        print(f"   Generated {mask.sum()} obstacle cells ({100*mask.sum()/mask.size:.1f}%) [synthetic]")
    
        # Generate synthetic spawn mask
        self.spawn_mask = ~mask
        print(f"[INFO] Synthetic spawn mask generated: {self.spawn_mask.sum()} valid cells")
    
        return mask

    def _validate_and_relocate_spawn(self, x: int, y: int, agent_type: str = 'protester') -> Tuple[int, int]:
        """
        Guaranteed valid spawn location with multi-strategy fallback.
    
        Strategy cascade:
        1. Check immediate location
        2. Radial search (up to 30 cells for 200×200 grid)
        3. Spawn mask sampling (guaranteed valid)
        4. Emergency: nearest road node (graph-based)
    
        Literature: Helbing (2000) - avoid initial overlaps for stability
        """
        # Strategy 1: Check original position
        if self._validate_spawn_position(x, y):
            return (x, y)
    
        # Strategy 2: Radial search with adaptive sampling
        max_radius = min(30, self.width // 6)  # Scale with grid size
    
        for radius in range(1, max_radius + 1):
            # Sample perimeter points (not all points - optimization)
            n_samples = min(8 * radius, 100)  # Cap at 100 checks per radius
            angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        
            for angle in angles:
                test_x = int(x + radius * np.cos(angle))
                test_y = int(y + radius * np.sin(angle))
            
                if not (0 <= test_x < self.width and 0 <= test_y < self.height):
                    continue
            
                if self._validate_spawn_position(test_x, test_y):
                    return (test_x, test_y)
    
        # Strategy 3: Spawn mask random sampling
        if hasattr(self, 'spawn_mask') and self.spawn_mask is not None:
            valid_cells = np.argwhere(self.spawn_mask)
            if len(valid_cells) > 0:
                # Weight by distance to original point (prefer nearby)
                distances = np.hypot(valid_cells[:, 1] - x, valid_cells[:, 0] - y)
                weights = np.exp(-distances / 10.0)  # Exponential decay
                weights /= weights.sum()
            
                idx = self.rng.choice(len(valid_cells), p=weights)
                ny, nx = valid_cells[idx]
                return (int(nx), int(ny))
    
        # Strategy 4: Emergency - use graph nodes directly
        if self.osm_graph is not None:
            # Find nearest uncrowded node
            available_nodes = [
                n for n in self.osm_graph.nodes()
                if self.node_occupancy.get(str(n), 0) < 
                   self.osm_graph.nodes[n].get('capacity', 6)
            ]
        
            if available_nodes:
                # Convert to grid coordinates
                nearest_node = min(
                    available_nodes,
                    key=lambda n: (
                        (self.osm_graph.nodes[n]['x'] - x)**2 + 
                        (self.osm_graph.nodes[n]['y'] - y)**2
                    )
                )
                return self._node_to_cell(str(nearest_node))
    
        # FATAL: Should never reach here with proper spawn_mask
        raise RuntimeError(
            f"Cannot find valid spawn for {agent_type} near ({x},{y}). "
            f"Check spawn_mask coverage and OSM graph connectivity."
        )

    
    def _spawn_agents(self):
        """Spawn protesters and police according to config."""
        self.agents = []
        self.protesters = []
        self.police_agents = []
    
        # Spawn protesters with heterogeneity
        protester_cfg = self.config['agents']['protesters']
        n_protesters = protester_cfg['count']
        spawn_cfg = protester_cfg['spawn']
    
        # Generate CANDIDATE positions (may need validation)
        positions = self._generate_spawn_positions(
            n_agents=n_protesters,
            spawn_type=spawn_cfg['type'],
            spawn_params=spawn_cfg
        )
    
        # Check if heterogeneous types are configured
        if 'types' in protester_cfg:
            agent_profiles = self._assign_agent_profiles(
                n_protesters,
                protester_cfg['types']
            )
        else:
            agent_profiles = ['average'] * n_protesters
    
        # Create protesters with assigned profiles
        base_speed = protester_cfg.get('speed_m_s', 1.2)
        use_graph = self.osm_graph is not None and self.cell_to_node is not None

        if use_graph:
            from .agent import GraphAgent
        else:
            from .agent import Agent
    
        # MAIN CHANGE: Use validation for EVERY protester spawn
        for i, (pos, profile) in enumerate(zip(positions, agent_profiles)):
            # STEP 1: Validate/relocate spawn position
            x_raw, y_raw = map(int, np.clip(pos, [0, 0], [self.width - 1, self.height - 1]))
        
            try:
                x, y = self._validate_and_relocate_spawn(x_raw, y_raw, agent_type='protester')
            except RuntimeError as e:
                # CRITICAL: Log failure and skip this agent (better than crashing)
                print(f"[FATAL] Cannot spawn protester {i}: {e}")
                continue
        
            # STEP 2: Assign goal
            goal_pos = self._assign_goal((x, y), protester_cfg.get('goals', {}))

            if use_graph:
                # GRAPH MODE
                # Map validated position to graph node
                node_id = self.cell_to_node[y, x]
            
                # Fallback if cell_to_node mapping invalid
                if node_id in (-1, None, "None", "nan") or pd.isna(node_id):
                    node_id = min(
                        self.osm_graph.nodes,
                        key=lambda n: (
                            (self.osm_graph.nodes[n]["x"] - x) ** 2 +
                            (self.osm_graph.nodes[n]["y"] - y) ** 2
                        )
                    )
            
                # Validate node connectivity (should not be needed with proper spawn mask)
                neighbors = list(self.osm_graph.neighbors(node_id))
                if len(neighbors) == 0:
                    print(f"[ERROR] Agent {i} at validated position ({x},{y}) mapped to isolated node {node_id}")
                    # Emergency: find nearest connected node
                    valid_nodes = [
                        n for n in self.osm_graph.nodes
                        if len(list(self.osm_graph.neighbors(n))) > 0
                    ]
                    if valid_nodes:
                        node_id = min(
                            valid_nodes,
                            key=lambda n: (
                                (self.osm_graph.nodes[n]["x"] - x) ** 2 +
                                (self.osm_graph.nodes[n]["y"] - y) ** 2
                            )
                        )
                        # Update x,y to match new node
                        node_data = self.osm_graph.nodes[node_id]
                        if self.affine is not None:
                            col, row = ~self.affine * (node_data["x"], node_data["y"])
                            x = int(np.clip(col, 0, self.width - 1))
                            y = int(np.clip(row, 0, self.height - 1))
                    else:
                        print(f"[FATAL] No connected nodes available for agent {i}")
                        continue
            
                # Check capacity (with improved handling)
                node_occ = self.node_occupancy.get(str(node_id), 0)
                node_cap = self.osm_graph.nodes[node_id].get("capacity", 6)
            
                if node_occ >= node_cap:
                    # Find nearest node with capacity
                    available_nodes = [
                        n for n in self.osm_graph.nodes
                        if (self.node_occupancy.get(str(n), 0) < 
                            self.osm_graph.nodes[n].get("capacity", 6) and
                            len(list(self.osm_graph.neighbors(n))) > 0)  # Must be connected
                    ]
                
                    if available_nodes:
                        node_id = min(
                            available_nodes,
                            key=lambda n: (
                                (self.osm_graph.nodes[n]["x"] - x) ** 2 + 
                                (self.osm_graph.nodes[n]["y"] - y) ** 2
                            )
                        )
                        # Update position to match new node
                        node_data = self.osm_graph.nodes[node_id]
                        if self.affine is not None:
                            col, row = ~self.affine * (node_data["x"], node_data["y"])
                            x = int(np.clip(col, 0, self.width - 1))
                            y = int(np.clip(row, 0, self.height - 1))
                    else:
                        # LAST RESORT: Force spawn with warning
                        print(f"[WARN] Agent {i} forced spawn at over-capacity node {node_id} "
                              f"({node_occ}/{node_cap})")
            
                # Map goal to node
                gx, gy = map(int, np.clip(goal_pos, [0, 0], [self.width - 1, self.height - 1]))
                goal_node = self.cell_to_node[gy, gx]
                if goal_node in (-1, None, "None", "nan") or pd.isna(goal_node):
                    goal_node = min(
                        self.osm_graph.nodes,
                        key=lambda n: (
                            (self.osm_graph.nodes[n]["x"] - gx) ** 2 +
                            (self.osm_graph.nodes[n]["y"] - gy) ** 2
                        )
                    )
            
                # Create graph agent
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
                agent.current_node = str(node_id)
                agent.goal_node = str(goal_node)
            
                # Increment occupancy
                self.node_occupancy[str(node_id)] = self.node_occupancy.get(str(node_id), 0) + 1

            else:
                # GRID MODE (simpler)
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

        # POLICE SPAWN (also use validation)
        police_cfg = self.config['agents']['police']
        n_police = police_cfg['count']
        police_spawn_cfg = police_cfg['spawn']
    
        if police_spawn_cfg['type'] == 'fixed':
            police_positions_raw = police_spawn_cfg['positions']
            police_positions = []
        
            for pos in police_positions_raw:
                x_raw, y_raw = map(int, np.clip(pos, [0, 0], [self.width - 1, self.height - 1]))
            
                try:
                    x, y = self._validate_and_relocate_spawn(x_raw, y_raw, agent_type='police')
                    police_positions.append((x, y))
                except RuntimeError as e:
                    print(f"[ERROR] Cannot spawn police at {pos}: {e}")
                    # Skip this police agent
                    continue
        else:
            # Generate positions and validate each
            raw_positions = self._generate_spawn_positions(
                n_agents=n_police,
                spawn_type=police_spawn_cfg['type'],
                spawn_params=police_spawn_cfg
            )
            police_positions = []
            for pos in raw_positions:
                x_raw, y_raw = map(int, np.clip(pos, [0, 0], [self.width - 1, self.height - 1]))
                try:
                    x, y = self._validate_and_relocate_spawn(x_raw, y_raw, agent_type='police')
                    police_positions.append((x, y))
                except RuntimeError:
                    continue
    
        # Create police agents (existing logic, now with validated positions)
        for j, pos in enumerate(police_positions):
            x, y = pos  # Already validated
        
            if use_graph:
                # Map to graph node
                node_id = self.cell_to_node[y, x]
                if node_id in (-1, None, "None", "nan") or pd.isna(node_id):
                    node_id = min(
                        self.osm_graph.nodes,
                        key=lambda n: (
                            (self.osm_graph.nodes[n]["x"] - x) ** 2 +
                            (self.osm_graph.nodes[n]["y"] - y) ** 2
                        )
                    )
            
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
                agent.current_node = str(node_id)
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
    
        # Summary log
        print(f"[SPAWN] Created {len(self.protesters)} protesters, {len(self.police_agents)} police")
        if len(self.agents) < n_protesters + n_police:
            print(f"[WARN] Spawned fewer agents than requested "
                  f"({len(self.agents)} vs {n_protesters + n_police})")
        
        
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

        # Normalize spawn_params and handle "auto" centers
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

        # Step 1: Collect movement requests
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

        # Step 2: Resolve conflicts (per target cell)
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

        # Step 3: Update occupancy
        self._update_occupancy_grid_simple()

        # Step 4: Post-movement congestion diagnostics (non-relocating)
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

        # Step 5: Final occupancy update
        self._update_occupancy_grid_simple()
    
    
    def validate_and_repair_graph(self):
        """Validate OSM graph connectivity and repair common issues."""
        from src.utils.logging_config import logger
    
        if self.osm_graph is None:
            return
    
        logger.verbose("\n[GRAPH VALIDATION] Analyzing connectivity...")
    
        # Remove isolated nodes
        isolated_nodes = [
            node for node in self.osm_graph.nodes()
            if len(list(self.osm_graph.neighbors(node))) == 0
        ]
    
        if isolated_nodes:
            logger.minimal(f"[WARN] Removing {len(isolated_nodes)} isolated nodes")
            self.osm_graph.remove_nodes_from(isolated_nodes)
    
        # Check connectivity
        if self.osm_graph.is_directed():
            components = list(nx.weakly_connected_components(self.osm_graph))
        else:
            components = list(nx.connected_components(self.osm_graph))
    
        if len(components) > 1:
            largest = max(components, key=len)
            logger.minimal(f"[WARN] Keeping largest component ({len(largest)}/{len(self.osm_graph.nodes)} nodes)")
            nodes_to_remove = set(self.osm_graph.nodes()) - largest
            self.osm_graph.remove_nodes_from(nodes_to_remove)
    
        # Convert to undirected
        if self.osm_graph.is_directed():
            logger.verbose("[INFO] Converting to undirected graph")
            self.osm_graph = self.osm_graph.to_undirected()
    
        # Rebuild mappings
        self.node_to_xy = {
            str(nid): (d["x"], d["y"]) 
            for nid, d in self.osm_graph.nodes(data=True)
        }
    
        # Statistics
        neighbor_counts = [len(list(self.osm_graph.neighbors(n))) for n in self.osm_graph.nodes()]
        min_neighbors = min(neighbor_counts) if neighbor_counts else 0
        avg_neighbors = sum(neighbor_counts) / len(neighbor_counts) if neighbor_counts else 0
    
        logger.verbose(f"[GRAPH] {len(self.osm_graph.nodes)} nodes, {len(self.osm_graph.edges)} edges")
        logger.verbose(f"[GRAPH] Neighbors: min={min_neighbors}, avg={avg_neighbors:.1f}")
    
        if min_neighbors == 0:
            logger.minimal("[ERROR] Graph still has isolated nodes!")
    
    
    def _execute_graph_movement(self, actions: Dict[int, Any]):
        """Execute graph movement (cleaned logging)."""
        from src.utils.logging_config import logger
    
        if self.osm_graph is None:
            return

        # Build neighbor lists
        agent_neighbors = {}
        for agent in self.agents:
            if not hasattr(agent, "current_node"):
                continue
            agent.current_node = str(agent.current_node)
            if agent.current_node in self.osm_graph:
                agent_neighbors[agent.id] = list(self.osm_graph.neighbors(agent.current_node))

        # Decode actions to target nodes
        move_requests = {}
        vacancies = {}
    
        for agent in self.agents:
            if not hasattr(agent, "current_node"):
                continue
        
            current_node = str(agent.current_node)
            action = actions.get(agent.id, 0)
        
            if action == 0 or action is None:
                continue
        
            neighbors = agent_neighbors.get(agent.id, [])
            if not neighbors:
                continue
        
            neighbor_idx = int(action) - 1
            if 0 <= neighbor_idx < len(neighbors):
                target_node = str(neighbors[neighbor_idx])
            
                if self.osm_graph.has_edge(current_node, target_node):
                    move_requests.setdefault(target_node, []).append(agent)
                    vacancies[current_node] = vacancies.get(current_node, 0) + 1

        # Resolve conflicts
        PRIORITY = {'police': 0, 'medic': 1, 'protester': 2}
        congestion_count = 0
    
        for target_node, agents_here in move_requests.items():
            node_capacity = self.osm_graph.nodes[target_node].get('capacity', 6)
            current_occupancy = self.node_occupancy.get(target_node, 0)
            departures = vacancies.get(target_node, 0)
            available_slots = max(0, node_capacity - (current_occupancy - departures))
        
            sorted_agents = sorted(
                agents_here,
                key=lambda a: (PRIORITY.get(a.agent_type, 99), self.rng.random())
            )
        
            # Move agents
            for agent in sorted_agents[:available_slots]:
                old_node = agent.current_node
                agent.current_node = target_node
                agent.pos = self._node_to_cell(target_node)
                agent.state = AgentState.MOVING
            
                self.node_occupancy[old_node] = max(0, self.node_occupancy.get(old_node, 0) - 1)
                self.node_occupancy[target_node] = self.node_occupancy.get(target_node, 0) + 1
        
            # Queue remaining
            queued = len(sorted_agents) - available_slots
            if queued > 0:
                congestion_count += 1
                for agent in sorted_agents[available_slots:]:
                    agent.state = AgentState.WAITING
                    agent.wait_timer = getattr(agent, 'wait_timer', 0) + 1
            
                # Log only if significant congestion
                if hasattr(self, 'street_names') and queued >= 3:
                    street_name = self.street_names.get(target_node, f"node {target_node}")
                    logger.verbose(f"[CONGESTION] {queued} agents queued at {street_name}")
            
                self.events_log.append({
                    'timestep': self.step_count,
                    'event_type': 'node_congestion',
                    'node_id': target_node,
                    'street_name': self.street_names.get(target_node, target_node) if hasattr(self, 'street_names') else target_node,
                    'capacity': node_capacity,
                    'queued': queued
                })
    
        # Summary logging
        if congestion_count > 0:
            logger.verbose(f"[INFO] {congestion_count} congestion events this step")


    def _node_to_cell(self, node_id: str) -> Tuple[int, int]:
        """Convert node coordinates to grid cell indices for raster overlays."""
        if self.affine is None:
            print(f"[WARN] No affine transform available for node {node_id}")
            return (0, 0)
    
        # Ensure node_id is string
        node_id = str(node_id)
    
        if node_id not in self.node_to_xy:
            print(f"[WARN] Node {node_id} not in node_to_xy mapping")
            return (0, 0)
    
        x_utm, y_utm = self.node_to_xy[node_id]
    
        # Transform UTM to grid cell (col, row)
        col, row = ~self.affine * (x_utm, y_utm)
    
        # Clamp to grid bounds
        x = int(np.clip(col, 0, self.width - 1))
        y = int(np.clip(row, 0, self.height - 1))
    
        return (x, y)

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
        # Guard: Skip in graph-based mode
        if getattr(self, "osm_graph", None) is not None:
            # Graph mode: occupancy is tracked via node_occupancy, not grid
            return
        
        # Grid mode
        if not hasattr(self, "occupancy_count"):
            raise AttributeError("Environment missing occupancy_count grid for grid-based tracking.")
        
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

        # Diagnostic Section
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
        FIXED harm detection with spatial interpolation.
    
        Issues resolved:
        1. Agents moving between cells miss exposure
        2. Concentration threshold too conservative
        3. No integration of exposure over path
    
        Literature: Bearing et al. (2018) - sub-cell position tracking
        """
        harm_grid = np.zeros((self.height, self.width), dtype=bool)
    
        # ADAPTIVE threshold based on grid resolution
        # Finer grid = lower threshold (agent spends less time per cell)
        BASE_THRESHOLD = 0.1  # mg/m³
        CONC_THRESHOLD = BASE_THRESHOLD * (100 / self.width)  # Scale with resolution
    
        agents_checked = 0
        agents_exposed = 0
        harm_events_this_step = 0
    
        for agent in self.agents:
            agents_checked += 1
        
            # Get current grid position (fresh for graph agents)
            if hasattr(agent, 'current_node') and self.osm_graph is not None:
                try:
                    x, y = self._node_to_cell(agent.current_node)
                    agent.pos = (x, y)  # Keep synchronized
                except Exception as e:
                    if self.step_count == 0:
                        print(f"[ERROR] _node_to_cell failed for agent {agent.id}: {e}")
                    continue
            else:
                x, y = agent.pos
        
            # Bounds check
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
        
            # MULTI-POINT SAMPLING (9-point stencil for better coverage)
            # Literature: Moussaïd et al. (2011) - pedestrian body occupies ~0.5m radius
            sample_points = [
                (x, y),      # Center
                (x-1, y),    # Left
                (x+1, y),    # Right
                (x, y-1),    # Up
                (x, y+1),    # Down
                (x-1, y-1),  # Diagonals
                (x+1, y-1),
                (x-1, y+1),
                (x+1, y+1)
            ]
        
            max_concentration = 0.0
            for px, py in sample_points:
                if 0 <= px < self.width and 0 <= py < self.height:
                    c = self.hazard_field.concentration[py, px]
                    max_concentration = max(max_concentration, c)
        
            concentration = max_concentration
        
            # Skip if below threshold
            if concentration < CONC_THRESHOLD:
                continue
        
            agents_exposed += 1
        
            # Calculate harm probability (unchanged)
            p_harm = 1.0 - np.exp(-self.hazard_field.k_harm * concentration * self.delta_t)
        
            # Update harm with ENVIRONMENT RNG (critical for reproducibility)
            harm_occurred = agent.update_harm(
                concentration=concentration,
                k_harm=self.hazard_field.k_harm,
                delta_t=self.delta_t,
                rng=self.rng  # Use env RNG, not agent RNG
            )
        
            if harm_occurred:
                harm_grid[y, x] = True
                harm_events_this_step += 1
            
                # Log to events (only first occurrence per agent)
                if not hasattr(agent, '_first_harm_logged'):
                    agent._first_harm_logged = True
                    self.events_log.append({
                        'timestep': self.step_count,
                        'event_type': 'first_harm',
                        'agent_id': agent.id,
                        'position': (x, y),
                        'concentration': float(concentration),
                        'cumulative_harm': float(agent.cumulative_harm)
                    })
        
            # Check incapacitation
            H_crit = self.config['hazards']['gas'].get('H_crit', 5.0)
            if agent.cumulative_harm >= H_crit:
                agent.state = AgentState.INCAPACITATED
                harm_grid[y, x] = True
            
                if not hasattr(agent, '_incap_logged'):
                    agent._incap_logged = True
                    print(f"[INCAP] Agent {agent.id} incapacitated at step {self.step_count} "
                        f"(H={agent.cumulative_harm:.2f})")
                
                    self.events_log.append({
                        'timestep': self.step_count,
                        'event_type': 'incapacitation',
                        'agent_id': agent.id,
                        'position': (x, y),
                        'cumulative_harm': float(agent.cumulative_harm)
                    })
    
        # Summary logging (adaptive frequency)
        log_interval = 5 if self.step_count <= 20 else 20
        if self.step_count % log_interval == 0 and agents_exposed > 0:
            print(f"[HARM] Step {self.step_count}: {agents_exposed}/{agents_checked} agents exposed, "
                f"{harm_events_this_step} harm events")
    
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
        IMPROVED termination with graduated urgency system.
    
        New mechanism: Agents gradually prioritize exits as time passes,
        preventing indefinite wandering.
    
        Literature: Lovreglio et al. (2016) - time-dependent risk perception
        """
        active = sum(a.state == AgentState.MOVING for a in self.protesters)
        waiting = sum(a.state == AgentState.WAITING for a in self.protesters)
        safe = sum(a.state == AgentState.SAFE for a in self.protesters)
        incapacitated = sum(a.state == AgentState.INCAPACITATED for a in self.protesters)
    
        total_protesters = len(self.protesters)
    
        # GRADUATED URGENCY: Increase exit-seeking over time
        if self.step_count > 100:  # After 100 steps (~1.5 minutes)
            urgency_multiplier = 1.0 + (self.step_count - 100) / 200.0
            urgency_multiplier = min(urgency_multiplier, 3.0)  # Cap at 3×
        
            # Broadcast urgency to all agents
            for agent in self.protesters:
                if hasattr(agent, 'goal_weights') and agent.state == AgentState.MOVING:
                    # Boost exit weight
                    agent.goal_weights['exit'] = min(
                        agent.goal_weights.get('exit', 0.2) * urgency_multiplier,
                        0.8  # Never completely override safety concerns
                    )
    
        # Termination conditions (unchanged)
        if active + waiting == 0:
            return True, "all_protesters_done"
    
        if safe / total_protesters > 0.9:
            return True, "mass_exodus"
    
        if incapacitated / total_protesters > 0.8:
            return True, "mass_casualty"
    
        # Enhanced stalemate detection
        if (active + waiting) / total_protesters < 0.1:
            if not hasattr(self, '_stalemate_counter'):
                self._stalemate_counter = 0
                self._last_active_count = active + waiting
        
            if active + waiting == self._last_active_count:
                self._stalemate_counter += 1
            else:
                self._stalemate_counter = 0
                self._last_active_count = active + waiting
        
            # INCREASED threshold for larger environment
            if self._stalemate_counter > 80:  # Was 50, now 80 for 200×200
                return True, "stalemate"
    
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