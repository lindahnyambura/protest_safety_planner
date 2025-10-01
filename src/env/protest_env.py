"""
protest_env.py - Core ProtestEnv Gymnasium environment

implementation: Grid initialization, basic agent movement, deterministic seeding
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
        
        # Agent management
        self.agents: List[Agent] = []
        self.protesters: List[Agent] = []
        self.police_agents: List[PoliceAgent] = []
        
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
        
        # Initialize hazard field
        hazard_cfg = self.config.get('hazards', {}).get('gas', {})
        self.hazard_field = HazardField(
            height=self.height,
            width=self.width,
            diffusion_coeff=hazard_cfg.get('diffusion_coeff', 0.2),
            decay_rate=hazard_cfg.get('decay_rate', 0.05),
            k_harm=hazard_cfg.get('k_harm', 0.1386),
            delta_t=self.delta_t
        )
        
        
        # Spawn agents
        self._spawn_agents()

        # DEBUG 00: AGENT SPAWN ON OBSTACLE CELL
        self.check_spawned_on_obstacle()
        
        # Update occupancy grid
        self._update_occupancy_grid()
        
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
                actions[agent.id] = agent.decide_action(self)
        
        # 2. Execute movement with conflict resolution
        self._execute_movement(actions)
        
        # 3. Update hazard field (diffusion, decay, sources)
        self.hazard_field.update(self.delta_t)
        
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
        Load obstacle mask from file or generate simple obstacles.
        
        Returns:
            obstacle_mask: Boolean array (True = impassable)
        """
        obstacle_cfg = self.config['grid'].get('obstacle_raster', 'generate')
        
        if obstacle_cfg == 'generate':
            # Generate simple rectangular obstacles for Day 1
            mask = np.zeros((self.height, self.width), dtype=bool)
            
            # Add border walls
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            
            # Add a few internal obstacles (buildings)
            mask[20:30, 20:35] = True
            mask[60:75, 50:70] = True
            mask[40:50, 70:80] = True
            
            return mask
        else:
            # TODO Day 3: Load from PNG
            raise NotImplementedError("PNG obstacle loading deferred to Day 3")
    
    def _spawn_agents(self):
        """Spawn protesters and police according to config."""
        self.agents = []
        self.protesters = []
        self.police_agents = []
        
        # Spawn protesters
        protester_cfg = self.config['agents']['protesters']
        n_protesters = protester_cfg['count']
        spawn_cfg = protester_cfg['spawn']
        
        positions = self._generate_spawn_positions(
            n_agents=n_protesters,
            spawn_type=spawn_cfg['type'],
            spawn_params=spawn_cfg
        )
        
        # Homogeneous protesters (all identical parameters)
        for i, pos in enumerate(positions):
            agent = Agent(
                agent_id=i,
                agent_type='protester',
                pos=pos,
                goal=self._assign_goal(pos, protester_cfg.get('goals', {})),
                speed=protester_cfg['speed_m_s'],
                risk_tolerance=protester_cfg.get('risk_tolerance_mean', 0.3),
                rng=self.rng
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
                for _ in range(agents_per_cluster):
                    # Sample from circular cluster
                    angle = self.rng.uniform(0, 2 * np.pi)
                    r = self.rng.uniform(0, radius)
                    x = int(cx + r * np.cos(angle))
                    y = int(cy + r * np.sin(angle))
                    
                    # Ensure within bounds and not on obstacle
                    x = np.clip(x, 1, self.width - 2)
                    y = np.clip(y, 1, self.height - 2)
                    
                    if not self.obstacle_mask[y, x]:
                        positions.append((x, y))
            
            # Handle remainder agents
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
        Execute agent movement with conflict resolution.
        
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
        
        # Track movement requests
        move_requests = {}  # target_cell -> list of agents
        
        # # Clear occupancy grid
        # self.occupancy_count.fill(0)
        
        # Process each agent
        for agent in self.agents:
            # Fractional speed accumulation
            agent.move_accum += agent.speed
            
            if agent.move_accum >= 1.0 and agent.state == 'moving':
                action = actions.get(agent.id, 0)
                dx, dy = MOVE_OFFSETS[action]
                target_x = agent.pos[0] + dx
                target_y = agent.pos[1] + dy
                
                # Boundary and obstacle check
                if (0 <= target_x < self.width and 
                    0 <= target_y < self.height and 
                    not self.obstacle_mask[target_y, target_x]):
                    
                    target_cell = (target_x, target_y)
                    # if target_cell not in move_requests:
                    #     move_requests[target_cell] = []
                    # move_requests[target_cell].append(agent)
                    move_requests.setdefault(target_cell, []).append(agent)
                
                agent.move_accum -= 1.0
        
        # Resolve conflicts (N_CELL_MAX limit)
        PRIORITY = {'police': 0, 'medic': 1, 'protester': 2, 'bystander': 3}
        
        for target_cell, candidates in move_requests.items():
            if len(candidates) <= self.n_cell_max:
                # All can move
                for agent in candidates:
                    agent.pos = target_cell
            else:
                # Conflict resolution: priority + RNG tie-break
                sorted_candidates = sorted(
                    candidates,
                    key=lambda a: (PRIORITY[a.agent_type], self.rng.random())
                )

                # DEBUG 02: AGENT OVERFLOW ; Enforce N_CELL_MAX strictly
                winners = sorted_candidates[:self.n_cell_max]
                losers = sorted_candidates[self.n_cell_max:]

                for agent in winners:
                    agent.pos = target_cell
                for agent in losers:
                    # Ensure losers remain in valid cells
                    x, y = agent.pos
                    if self.obstacle_mask[y, x]:
                        # Relocate to a random valid free cell
                        new_x, new_y = self._find_free_cell()
                        agent.pos = (new_x, new_y)    
        
        # Update occupancy grid
        self._update_occupancy_grid()
    
    def _update_occupancy_grid(self):
        """Update occupancy count grid from agent positions."""
        self.occupancy_count.fill(0)
        for agent in self.agents:
            x, y = agent.pos
            # Cap occupancy to N_CELL_MAX
            if self.occupancy_count[y, x] < self.n_cell_max:
                self.occupancy_count[y, x] += 1
            else:
                # Too many in one cell, relocate this agent
                new_x, new_y = self._find_free_cell()
                agent.pos = (new_x, new_y)
                self.occupancy_count[new_y, new_x] += 1

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
                agent.state = 'incapacitated'
        
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
        # All protesters reached goals or incapacitated
        active_protesters = [
            a for a in self.protesters 
            if a.state == 'moving'
        ]
        
        if len(active_protesters) == 0:
            return True, "all_protesters_done"
        
        # Mass casualty (>80% incapacitated)
        incap_rate = sum(
            a.state == 'incapacitated' for a in self.protesters
        ) / len(self.protesters)
        
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
        """Basic rendering (defer fancy viz to Day 3)."""
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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config