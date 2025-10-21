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

from .agent import Agent, PoliceAgent, AgentState
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
        
        # Grid metadata (for CV integration)
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

        # OSM data (if loaded)
        self.osm_metadata = None
        self.buildings_gdf = None
        self.streets_graph = None
        
        # Agent management
        self.agents: List[Agent] = []
        self.protesters: List[Agent] = []
        self.police_agents: List[PoliceAgent] = []
        self.exit_points = config['agents']['protesters']['goals']['exit_points']
        
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
        # Set seed for deterministic behavior (CRITICAL)
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
            for agent in self.agents:
                # CRITICAL: Let agents update goals dynamically
                if hasattr(agent, 'update_goal'):  # Protesters have this, police don't
                    agent.update_goal(self, self.exit_points)

                actions[agent.id] = agent.decide_action(self)

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
                from .real_nairobi_loader import load_real_nairobi_cbd_map
                result = load_real_nairobi_cbd_map(self.config)

                if result and result.get('is_real_osm'):
                    mask = result['obstacle_mask']

                    # Flip vertically if raster origin mismatch (rasterio uses bottom-left)
                    if self.grid_metadata.origin == 'top_left':
                        mask = np.flipud(mask)

                    # Optional: warn if obstacle coverage too high
                    coverage = 100 * mask.sum() / mask.size
                    if coverage > 70:
                        print(f"[WARN] High obstacle coverage ({coverage:.1f}%) – agent spawning may fail.")

                    print(f" Using REAL Nairobi CBD map ({coverage:.1f}% coverage).")
                    self.osm_metadata = result.get('metadata', {})
                    self.buildings_gdf = result.get('buildings_gdf')
                    self.streets_graph = result.get('streets_graph')
                    return mask

            except Exception as e:
                print(f"[ERROR] Failed to load real Nairobi CBD map: {e}")

        # Fallback: Synthetic obstacles
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

        # Homogeneous protesters (all identical parameters)
        for i, (pos, profile) in enumerate(zip(positions, agent_profiles)):
            agent = Agent(
                agent_id=i,
                agent_type='protester',
                pos=pos,
                goal=self._assign_goal(pos, protester_cfg.get('goals', {})),
                speed=base_speed,  # Will be modified by profile
                risk_tolerance=0.3,  # Will be modified by profile
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
            police_positions = police_spawn_cfg['positions']
        else:
            police_positions = self._generate_spawn_positions(
                n_agents=n_police,
                spawn_type=police_spawn_cfg['type'],
                spawn_params=police_spawn_cfg
            )
        
        for i, pos in enumerate(police_positions):
            agent = PoliceAgent(
                agent_id=len(self.agents),
                pos=tuple(pos),
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
    
    def _generate_spawn_positions(self, n_agents: int, spawn_type: str, 
                                  spawn_params: Dict) -> List[Tuple[int, int]]:
        """
        Generate agent spawn positions.
        
        Args:
            n_agents: Number of agents to spawn
            spawn_type: 'clusters', 'uniform', 'fixed'
            spawn_params: Type-specific parameters
            
        Returns:
            List of (x, y) positions
        """
        positions = []
        
        if spawn_type == 'clusters':
            centers = spawn_params['centers']
            radius = spawn_params['radius']
            agents_per_cluster = n_agents // len(centers)
        
            for center in centers:
                cx, cy = center
                attempts = 0
                spawned_in_cluster = 0

                while spawned_in_cluster < agents_per_cluster and attempts < 1000:
                    angle = self.rng.uniform(0, 2 * np.pi)
                    r = self.rng.uniform(0, radius)
                    x = int(cx + r * np.cos(angle))
                    y = int(cy + r * np.sin(angle))
                
                    # Check bounds AND obstacles
                    if (1 <= x < self.width - 1 and 
                        1 <= y < self.height - 1 and
                        not self.obstacle_mask[y, x]):
                        positions.append((x, y))
                        spawned_in_cluster += 1

                    attempts += 1

                if attempts >= 1000:
                    print(f"[WARN] Cluster {center}: only spawned {spawned_in_cluster}/{agents_per_cluster}")
        
            # Handle remainder + failed spawns
            while len(positions) < n_agents:
                x = self.rng.integers(1, self.width - 1)
                y = self.rng.integers(1, self.height - 1)
                if not self.obstacle_mask[y, x]:
                    positions.append((x, y))  
        
        elif spawn_type == 'fixed':
            
            positions = []
            for pos in spawn_params['positions']:
                x, y = tuple(pos)
                # If it's an obstacle, re-roll somewhere random
                if self.obstacle_mask[y, x]:
                   print(f"[WARN] Fixed spawn {pos} is on obstacle, relocating.")
                   while True:
                       rx = self.rng.integers(1, self.width - 1)
                       ry = self.rng.integers(1, self.height - 1)
                       if not self.obstacle_mask[ry, rx]:
                           positions.append((rx, ry))
                           break
                else:
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
        exit_points = goals_cfg.get('exit_points', [[50, 95]])
        
        if strategy == 'nearest_exit':
            # Find nearest exit
            distances = [np.hypot(pos[0] - ex[0], pos[1] - ex[1]) for ex in exit_points]
            nearest_idx = np.argmin(distances)
            return tuple(exit_points[nearest_idx])
        
        return tuple(exit_points[0])  # Default
    

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