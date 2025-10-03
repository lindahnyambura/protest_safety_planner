"""
agent.py - Agent classes for protesters and police

Version 1: Basic agent movement with fractional speed, homogeneous protesters
Version 2: Add agent heterogeneity (3 types)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class AgentState:
    """Agent state enum values"""
    MOVING = 'moving'
    STUNNED = 'stunned'
    INCAPACITATED = 'incapacitated'
    ARRESTED = 'arrested'
    SAFE = 'safe'


class Agent:
    """
    Base agent class for protesters, medics, bystanders.
    
    Supports heterogeneous agent types with different risk tolerances and speeds.
    """
    
    # Type priorities for conflict resolution
    PRIORITY = {'police': 0, 'medic': 1, 'protester': 2, 'bystander': 3}
    
    # Agent type profiles (heterogeneity)
    AGENT_PROFILES = {
        'cautious': {
            'speed_multiplier': 0.83,      # 1.0 m/s (slower)
            'risk_tolerance': 0.1,         # Very cautious
            'w_hazard_multiplier': 1.5     # Extra weight on hazard avoidance
        },
        'average': {
            'speed_multiplier': 1.0,       # 1.2 m/s (baseline)
            'risk_tolerance': 0.3,         # Moderate caution
            'w_hazard_multiplier': 1.0     # Standard weight
        },
        'bold': {
            'speed_multiplier': 1.17,      # 1.4 m/s (faster)
            'risk_tolerance': 0.6,         # Risk-taking
            'w_hazard_multiplier': 0.5     # Less concerned about hazards
        }
    }
    
    # 8-neighbor movement offsets
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
    
    def __init__(self, 
                 agent_id: int,
                 agent_type: str,
                 pos: Tuple[int, int],
                 goal: Tuple[int, int],
                 speed: float,
                 risk_tolerance: float,
                 rng: np.random.Generator,
                 profile_name: str = 'average'):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier
            agent_type: 'protester', 'police', 'medic', 'bystander'
            pos: Initial (x, y) cell position
            goal: Target (x, y) cell position
            speed: Base cells per step (will be modified by profile)
            risk_tolerance: Base risk tolerance (will be modified by profile)
            rng: Random number generator
            profile_name: Agent profile ('cautious', 'average', 'bold')
        """
        # Apply profile modifications
        profile = self.AGENT_PROFILES.get(profile_name, self.AGENT_PROFILES['average'])
        
        # Core attributes (locked data types)
        self.id: int = agent_id
        self.agent_type: str = agent_type
        self.profile_name: str = profile_name
        self.pos: Tuple[int, int] = pos
        self.goal: Tuple[int, int] = goal
        
        # Apply profile speed multiplier
        self.speed: np.float32 = np.float32(speed * profile['speed_multiplier'])
        
        # Use profile risk tolerance
        self.risk_tolerance: np.float32 = np.float32(profile['risk_tolerance'])
        
        # State
        self.state: str = AgentState.MOVING
        
        # Harm tracking (dual-purpose model)
        self.cumulative_harm: np.float32 = np.float32(0.0)
        self.harm_events: int = 0
        
        # Movement mechanics
        self.move_accum: np.float32 = np.float32(0.0)
        self.last_move_direction: Optional[int] = None
        
        # RNG for stochastic decisions
        self.rng = rng
        
        # Scoring weights (locked parameters, modified by profile)
        self.w_goal: float = 1.0
        self.w_hazard: float = 5.0 * (1.0 - self.risk_tolerance) * profile['w_hazard_multiplier']
        self.w_occupancy: float = 0.5
        self.w_inertia: float = 0.2
        self.beta: float = 5.0  # Boltzmann temperature model)
        self.cumulative_harm: np.float32 = np.float32(0.0)
        self.harm_events: int = 0
        
        # Movement mechanics
        self.move_accum: np.float32 = np.float32(0.0)
        self.last_move_direction: Optional[int] = None
        
        # RNG for stochastic decisions
        self.rng = rng
        
        # Scoring weights (locked parameters, modified by profile)
        self.w_goal: float = 1.0
        self.w_hazard: float = 5.0 * (1.0 - self.risk_tolerance) * profile['w_hazard_multiplier']
        self.w_occupancy: float = 0.5
        self.w_inertia: float = 0.2
        self.beta: float = 5.0  # Boltzmann temperature model)
        self.cumulative_harm: np.float32 = np.float32(0.0)
        self.harm_events: int = 0
        
        # Movement mechanics
        self.move_accum: np.float32 = np.float32(0.0)  # Fractional speed accumulator
        self.last_move_direction: Optional[int] = None  # For inertia
        
        # RNG for stochastic decisions
        self.rng = rng
        
        # Scoring weights (locked parameters)
        self.w_goal: float = 1.0
        self.w_hazard: float = 5.0 * (1.0 - risk_tolerance)
        self.w_occupancy: float = 0.5
        self.w_inertia: float = 0.2
        self.beta: float = 5.0  # Boltzmann temperature
    
    def decide_action(self, env) -> int:
        """
        Decide next action based on scoring function.
        
        Args:
            env: ProtestEnv instance
            
        Returns:
            action: Integer 0-8 (direction to move)
        """
        if self.state != AgentState.MOVING:
            return 0  # STAY if not moving
        
        # Check if reached goal
        if self._at_goal():
            self.state = AgentState.SAFE
            return 0
        
        # Score all possible moves
        scores = self._score_neighbors(env)
        
        # Stochastic action selection (Boltzmann softmax)
        action = self._select_action_stochastic(scores)
        
        return action
    
    def _at_goal(self) -> bool:
        """Check if agent reached goal."""
        return (abs(self.pos[0] - self.goal[0]) <= 1 and 
                abs(self.pos[1] - self.goal[1]) <= 1)
    
    def _score_neighbors(self, env) -> np.ndarray:
        """
        Score all 9 possible moves (including STAY).
        
        Args:
            env: ProtestEnv instance
            
        Returns:
            scores: Array of shape (9,) with utility scores
        """
        scores = np.zeros(9, dtype=np.float32)
        
        for action_idx, (dx, dy) in enumerate(self.MOVE_OFFSETS):
            nx = self.pos[0] + dx
            ny = self.pos[1] + dy
            
            # Boundary check
            if not (0 <= nx < env.width and 0 <= ny < env.height):
                scores[action_idx] = -np.inf
                continue
            
            # Obstacle check
            if env.obstacle_mask[ny, nx]:
                scores[action_idx] = -np.inf
                continue
            
            # Compute utility components
            U_goal = self._utility_goal(nx, ny)
            U_hazard = self._utility_hazard(nx, ny, env)
            U_occupancy = self._utility_occupancy(nx, ny, env)
            U_inertia = self._utility_inertia(action_idx)
            
            # Total score
            scores[action_idx] = (
                self.w_goal * U_goal +
                self.w_hazard * U_hazard +
                self.w_occupancy * U_occupancy +
                self.w_inertia * U_inertia
            )
        
        return scores
    
    def _utility_goal(self, x: int, y: int) -> float:
        """Utility for progress toward goal."""
        current_dist = np.hypot(self.pos[0] - self.goal[0], 
                               self.pos[1] - self.goal[1])
        new_dist = np.hypot(x - self.goal[0], y - self.goal[1])
        return -(new_dist - current_dist)  # Negative of distance increase
    
    def _utility_hazard(self, x: int, y: int, env) -> float:
        """Utility for hazard avoidance."""
        concentration = env.hazard_field.concentration[y, x]
        # Convert to harm probability estimate
        p_harm_est = 1 - np.exp(-env.hazard_field.k_harm * concentration * env.delta_t)
        return -p_harm_est  # Negative utility for risk
    
    def _utility_occupancy(self, x: int, y: int, env) -> float:
        """Utility for crowd avoidance."""
        occupancy = env.occupancy_count[y, x]
        return -float(occupancy)
    
    def _utility_inertia(self, action_idx: int) -> float:
        """Utility for continuing in same direction."""
        if self.last_move_direction is None:
            return 0.0
        return 1.0 if action_idx == self.last_move_direction else 0.0
    
    def _select_action_stochastic(self, scores: np.ndarray) -> int:
        """
        Select action using Boltzmann softmax.
        
        Args:
            scores: Utility scores for each action
            
        Returns:
            Selected action index
        """
        # Handle -inf scores (invalid moves)
        valid_mask = np.isfinite(scores)
        if not valid_mask.any():
            return 0  # STAY if no valid moves
        
        # Boltzmann probabilities
        valid_scores = scores[valid_mask]
        exp_scores = np.exp(self.beta * (valid_scores - np.max(valid_scores)))
        probs = exp_scores / exp_scores.sum()
        
        # Sample action
        valid_actions = np.where(valid_mask)[0]
        action = self.rng.choice(valid_actions, p=probs)
        
        self.last_move_direction = action
        return int(action)
    
    def update_harm(self, concentration: float, k_harm: float, 
                   delta_t: float, rng: np.random.Generator) -> bool:
        """
        Update agent harm status (dual-purpose model).
        
        1. Cumulative damage (deterministic) → incapacitation
        2. Binary harm event (stochastic) → I_i grid indicator
        
        Args:
            concentration: Hazard concentration at agent's location
            k_harm: Harm rate parameter
            delta_t: Timestep duration
            rng: Random number generator
            
        Returns:
            harm_occurred: True if harm event sampled this step
        """
        # 1. Accumulate deterministic damage
        alpha_h = 0.1  # Damage rate multiplier
        self.cumulative_harm += alpha_h * concentration * delta_t
        
        # 2. Sample stochastic harm event
        p_harm = 1 - np.exp(-k_harm * concentration * delta_t)
        p_harm = np.clip(p_harm, 1e-6, 0.999999)  # Numerical stability
        
        harm_occurred = rng.random() < p_harm
        if harm_occurred:
            self.harm_events += 1
        
        return harm_occurred


class PoliceAgent(Agent):
    """
    Police agent with gas deployment capability.
    
    Day 1: Rule-based behavior (move toward crowd, deploy gas)
    """
    
    def __init__(self,
                 agent_id: int,
                 pos: Tuple[int, int],
                 speed: float,
                 deploy_prob: float,
                 deploy_cooldown_max: int,
                 config: Dict,
                 rng: np.random.Generator):
        """
        Initialize police agent.
        
        Args:
            agent_id: Unique identifier
            pos: Initial position
            speed: Movement speed (cells/step)
            deploy_prob: Probability of deploying gas per step
            deploy_cooldown_max: Steps before can deploy again
            config: Environment configuration
            rng: Random number generator
        """
        # Initialize as agent with high risk tolerance
        super().__init__(
            agent_id=agent_id,
            agent_type='police',
            pos=pos,
            goal=pos,  # Police don't have fixed goals
            speed=speed,
            risk_tolerance=0.7,  # Less risk-averse
            rng=rng
        )
        
        # Police-specific attributes
        self.deploy_prob = deploy_prob
        self.deploy_cooldown = 0
        self.deploy_cooldown_max = deploy_cooldown_max
        self.config = config
    
    def decide_action(self, env) -> int:
        """
        Police decision logic (fixed + intercept behavior):
        1. Compute crowd centroid
        2. Find nearest exit and intercept point (midpoint)
        3. Position with small offset so multiple police spread along a line
        4. Deploy gas/water/shooting via env.hazards with cooldowns & probs
        """
        # Ensure we have x,y defined early (fixes NameError)
        x, y = self.pos

        # 1. Get moving protesters positions
        protester_positions = [a.pos for a in env.protesters if a.state == AgentState.MOVING]
        if not protester_positions:
            return 0  # STAY if no active protesters
        
        # 2. Crowd centroid (as float)
        centroid = np.mean(protester_positions, axis=0)  # array([x, y])

        # 3. Find nearest exit from config
        exits = env.config['agents']['protesters'].get('goals', {}).get('exit_points', [])
        if not exits:
            # fallback to center if no exits configured
            nearest_exit = np.array([env.width // 2, env.height // 2], dtype=float)
        else:
            dists = [np.hypot(centroid[0] - ex[0], centroid[1] - ex[1]) for ex in exits]
            nearest_exit = np.array(exits[int(np.argmin(dists))], dtype=float)

        # 4. Intercept point: halfway between centroid and chosen exit
        intercept = ((centroid + nearest_exit) / 2.0).astype(int)

        # 5. Compute a small perpendicular offset to spread police along a short line
        #    Use police id to make offset deterministic and reproducible
        dir_vec = nearest_exit - centroid
        perp = np.array([-dir_vec[1], dir_vec[0]])
        perp_norm = perp / (np.linalg.norm(perp) + 1e-8)

        # offset magnitude (cells)
        offset_magnitude = (self.id % 7) - 3  # values [-3..3] -> spreads police deterministically
        offset = (perp_norm * offset_magnitude).astype(int)

        target = intercept + offset
        target_x = int(np.clip(target[0], 0, env.width - 1))
        target_y = int(np.clip(target[1], 0, env.height - 1))

        # set goal toward intercept-target
        self.goal = (target_x, target_y)

        # 6. Score moves toward goal (use base class scoring)
        scores = self._score_neighbors(env)
        action = self._select_action_stochastic(scores)

        # 7. Deploy gas (respect cooldown)
        if not hasattr(self, 'deploy_cooldown'):
            self.deploy_cooldown = 0
        if self.deploy_cooldown <= 0:
            # try gas
            self._attempt_gas_deployment(env)
        else:
            self.deploy_cooldown -= 1

        # 8. Water cannon: separate cooldown + probability, via config hazards.water_cannon
        wc_cfg = env.config.get('hazards', {}).get('water_cannon', {})
        if wc_cfg.get('enabled', False):
            if not hasattr(self, 'wc_cooldown'):
                self.wc_cooldown = 0
            if self.wc_cooldown <= 0:
                # require some local density to use water cannon
                local_density = env.occupancy_count[y, x]
                wc_prob = wc_cfg.get('prob', 0.01)
                if local_density >= wc_cfg.get('min_density', 6) and self.rng.random() < wc_prob:
                    # push direction: away from crowd centroid (police aim between crowd and exit)
                    dir_x = int(np.sign(centroid[0] - x))
                    dir_y = int(np.sign(centroid[1] - y))
                    # If dir is zero (coincident), default push away from centroid->exit axis:
                    if dir_x == 0 and dir_y == 0:
                        dir_x = int(np.sign(nearest_exit[0] - x))
                        dir_y = int(np.sign(nearest_exit[1] - y))
                    env.hazards.deploy_water_cannon(
                        env=env,
                        x=x,
                        y=y,
                        direction=(dir_x, dir_y),
                        strength=wc_cfg.get('strength', 3),
                        radius=wc_cfg.get('radius', 6),
                        stun_prob=wc_cfg.get('stun_prob', 0.1),
                        agent_id=self.id
                    )
                    self.wc_cooldown = wc_cfg.get('cooldown', 30)
            else:
                self.wc_cooldown = max(0, self.wc_cooldown - 1)

        # 9. Shooting: very rare, via hazards.shooting_event
        shoot_cfg = env.config.get('hazards', {}).get('shooting', {})
        if shoot_cfg.get('enabled', False):
            if not hasattr(self, 'shoot_cooldown'):
                self.shoot_cooldown = 0
            p_shoot = shoot_cfg.get('prob_per_step', 0.002)
            if self.shoot_cooldown <= 0 and self.rng.random() < p_shoot:
                # choose nearest protester (if any)
                if env.protesters:
                    # only consider moving protesters if possible
                    candidates = [p for p in env.protesters if p.state == AgentState.MOVING]
                    if not candidates:
                        candidates = env.protesters
                    target = min(candidates, key=lambda a: (a.pos[0] - x) ** 2 + (a.pos[1] - y) ** 2)
                    env.hazards.shooting_event(env=env, shooter_agent=self, targets=[target], fatal=shoot_cfg.get('fatal', False))
                    # set cooldown to avoid rapid repeats
                    self.shoot_cooldown = shoot_cfg.get('cooldown', 100)
            else:
                self.shoot_cooldown = max(0, getattr(self, 'shoot_cooldown', 0) - 1)

        return int(action)

    
    def _attempt_gas_deployment(self, env):
        """
        Attempt to deploy gas at current location using HazardManager.
        Conditions:
        - Local density >= threshold
        - Random check with deploy_prob
        """
        
        x, y = self.pos
        # Count nearby protesters (3-cell radius)
        nearby_protesters = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.width and 0 <= ny < env.height:
                    nearby_protesters += sum(1 for a in env.agents 
                                            if a.pos == (nx, ny) and a.agent_type == 'protester')
    
        threshold = 10  # deploy threshold (configurable)
        if nearby_protesters >= threshold and self.rng.random() < self.deploy_prob:
            inj_intensity = self.config['hazards']['gas'].get('inj_intensity', 5.0)
            # Use HazardManager
            env.hazards.deploy_gas(env=env, x=x, y=y, intensity=inj_intensity, agent_id=self.id)
            self.deploy_cooldown = self.deploy_cooldown_max
            # note: HazardManager logs the event; no duplicate logging here