"""
agent.py - Agent classes for protesters and police

Version 1: Basic agent movement with fractional speed, homogeneous protesters
Version 2: Add agent heterogeneity (3 types)
"""

import numpy as np
import math
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
    WAITING = 'waiting'


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
            'speed_multiplier': 0.83,      # 1.0 m/s (typical cautious walking speed)
            'risk_tolerance': 0.1,         # Flees at low hazard concentration
            'w_hazard_multiplier': 1.5,     # 50% more sensitive to hazards
            'lookahead_multiplier': 1.5,  # More concerned about future hazards'
            'profile_multiplier': 1.5
        },
        'average': {
            'speed_multiplier': 1.0,       # 1.2 m/s (normal walking speed)
            'risk_tolerance': 0.3,         # Moderate caution
            'w_hazard_multiplier': 1.0,     # Standard weight
            'lookahead_multiplier': 1.0,  # Standard lookahead
            'profile_multiplier': 1.0
        },
        'bold': {
            'speed_multiplier': 1.17,      # 1.4 m/s (fast walking/jogging speed)
            'risk_tolerance': 0.6,         # Risk-taking
            'w_hazard_multiplier': 0.5 ,    # Less concerned about hazards
            'lookahead_multiplier': 0.5,  # Less concerned about future hazards
            'profile_multiplier': 0.5
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
                 profile_name: str = 'average',
                 wait_timer: int = 0):
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
        
        # Apply profile multiplier
        self.profile_multiplier: float = profile.get('profile_multiplier', 1.0)
        
        # Apply profile lookahead multiplier
        self.lookahead_multiplier: float = profile.get('lookahead_multiplier', 1.0)

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
        # w_goal: preference for goal proximity
        # w_hazard: penalty for entering hazardous zones (scaled by risk_tolerance and profile)
        # w_occupancy: penalty for crowded cells
        # w_inertia: bias toward continuing previous direction
        # beta: Boltzmann temperature for stochastic action choice
        self.w_goal: float = 1.0
        self.w_hazard: float = 15.0 * (1.0 - self.risk_tolerance) * profile['w_hazard_multiplier'] # Now average agent: w_hazard = 15 × 0.7 × 1.0 = 10.5 (much stronger)
        self.w_occupancy: float = 0.5
        self.w_inertia: float = 0.2
        self.beta: float = 5.0  # Boltzmann temperature model - controls randomness in movement choice
        # self.cumulative_harm: np.float32 = np.float32(0.0)
        # self.harm_events: int = 0

        # Wait timer
        self.wait_timer: int = wait_timer
        
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
        
        # Update goal dynamically
        exits = env.config['agents']['protesters']['goals']['exit_points']
        self.update_goal(env, exits)

        # Score all possible moves
        scores = self._score_neighbors(env)
        
        # Stochastic action selection (Boltzmann softmax)
        action = self._select_action_stochastic(scores)
        
        return action
    
    def update_goal(self, env, exits: list):
        """
        Dynamically select a (probabilistic) safe reachable exit.
        - Uses superlinear congestion cost.
        - Uses softmax sampling over exit utilities to encourage dispersion.
        - Hysteresis: only resample every `goal_update_period` steps unless
          current path hazard exceeds `risk_tolerance_immediate`.
        """
        
        # === Parameters (tweakable) ===
        goal_update_period = getattr(self, "goal_update_period", 20)  # default every 20 steps
        beta_exit = getattr(self, "beta_exit", 0.2)                  # softmax inverse-temperature
        congestion_exp = getattr(self, "congestion_exp", 2.0)       # superlinear exponent
        congestion_thresh = getattr(self, "congestion_thresh", None) # optional thresholded behavior
        usage_penalty_k = getattr(self, "usage_penalty_k", 2.0)     # scale exit usage penalty
        immediate_reassess_hazard_factor = getattr(self, "risk_tolerance_immediate", 1.5)
        hysteresis_factor = getattr(self, "goal_hysteresis_factor", 1.05)
        
        
        # initialize counters if needed
        if not hasattr(self, '_goal_update_counter'):
            self._goal_update_counter = 0
        self._goal_update_counter += 1

        # If not time yet, only re-evaluate if current path hazard is large
        do_resample = False
        if self._goal_update_counter >= goal_update_period:
            do_resample = True
            self._goal_update_counter = 0
        else:
            # quick check: if current path hazard to current goal is very large, force resample
            try:
                curr_path_hazard = self._estimate_path_hazard(env, tuple(self.goal))
            except Exception:
                curr_path_hazard = 0.0
            if curr_path_hazard > immediate_reassess_hazard_factor * getattr(self, "risk_tolerance", 0.3):
                do_resample = True
                # reset counter so we don't immediately re-sample again next step
                self._goal_update_counter = 0

        if not do_resample:
            return  # keep current goal
        
        # Precompute exit usage counts (how many agents currently targeting each exit)
        exit_usage = [self._count_agents_targeting(env, exit_pos) for exit_pos in exits]
        total_agents = max(1, len(getattr(env, "agents", [])))
        
        # Score all exits
        exit_scores = []
        for idx, exit_pos in enumerate(exits):
            # Distance cost
            dist = math.hypot(self.pos[0] - exit_pos[0], self.pos[1] - exit_pos[1])
            norm_dist = dist / max(env.width, env.height)   # normalize to [0, 1]

            # Path hazard
            path_hazard = self._estimate_path_hazard(env, exit_pos)

            # Congestion: local agent count near exit
            congestion = self._count_agents_near(env, exit_pos, radius=5)
            cost_congestion = 0.5 * (congestion ** congestion_exp)

            # Usage penalty (relative to total agents) - prevents runaway usage
            usage_penalty = usage_penalty_k * (exit_usage[idx] / float(total_agents))

            # Combined utility (higher = better)
            # We invert cost terms to negative, combine with weights
            # Adjust scaling factors so utilities are in reasonable range
            utility = (
                - (norm_dist * 10.0)       # distance cost
                - (5.0 * path_hazard)      # hazard cost
                - cost_congestion          # congestion cost
                - (20.0 * usage_penalty)   # usage penalty
            )
            exit_scores.append(utility)
        
        # Convert to probabilities via stabilized softmax
        scores = np.array(exit_scores, dtype=np.float64)
        # numerical stabilization
        max_s = np.max(scores)
        exp_scores = np.exp(beta_exit * (scores - max_s))
        probs = exp_scores / (np.sum(exp_scores) + 1e-12)

        # Sample exit according to probs (stochastic choice)
        chosen_idx = self.rng.choice(len(exits), p=probs)
        chosen_exit = tuple(exits[chosen_idx])

        # --- DEBUG PRINT: now probs and chosen_exit exist ---
        if getattr(self, "id", -1) < 5:
            print(f"[Agent {self.id}] exit_scores={np.round(exit_scores,3)} probs={np.round(probs,3)} chosen={chosen_exit}")
        
        # Hysteresis: only switch if chosen exit gives a sufficient improvement over current
        # (avoid flip-flopping if probabilities similar). We measure expected utility difference.
        current_goal = tuple(self.goal) if hasattr(self, "goal") else None
        if current_goal is None:
            self.goal = chosen_exit
            return
        
        # compute current_goal_score for comparison
        try:
            curr_idx = exits.index(list(current_goal))
        except ValueError:
            curr_idx = None

        if curr_idx is not None:
            current_score = scores[curr_idx]
        else:
            # fallback revision of utility for current goal
            d0 = math.hypot(self.pos[0] - current_goal[0], self.pos[1] - current_goal[1])
            norm_d0 = d0 / max(env.width, env.height)
            ph0 = self._estimate_path_hazard(env, current_goal)
            cong0 = self._count_agents_near(env, current_goal, radius=5)
            cc0 = 0.5 * (cong0 ** congestion_exp)
            up0 = self._count_agents_targeting(env, current_goal) / float(total_agents)
            usage0 = usage_penalty_k * up0
            current_score = (
                - (norm_d0 * 10.0)
                - (5.0 * ph0)
                - cc0
                - (20.0 * usage0)
            )
        
        # require at least a fractional improvement to switch (hysteresis factor)
        if scores[chosen_idx] > hysteresis_factor * current_score or \
            self._estimate_path_hazard(env, current_goal) > getattr(self, "risk_tolerance", 0.3) * 2.0:
                self.goal = chosen_exit
        # else: keep current goal

    
    def _count_agents_targeting(self, env, exit_pos, radius: int = 2) -> int:
        """
        Count how many agents currently have `exit_pos` as their goal (within small tolerance).
        Used to estimate intentional congestion.
        """
        cnt = 0
        for a in getattr(env, "agents", []):
            try:
                if hasattr(a, "goal") and tuple(a.goal) == tuple(exit_pos):
                    cnt += 1
            except Exception:
                continue
        return cnt

    def _estimate_path_hazard(self, env, target: Tuple[int, int]) -> float:
        """
        Estimate hazard along straight-line path to target.

        Simple heuristic: sample 10 points along line to target.
        """

        x0, y0 = self.pos
        x1, y1 = target
    
        hazard_sum = 0.0
        n_samples = 10

        for i in range(n_samples):
            t = i / (n_samples - 1)
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
        
            if 0 <= x < env.width and 0 <= y < env.height:
                concentration = env.hazard_field.concentration[y, x]
                hazard_sum += concentration
        
        return hazard_sum / n_samples
    
    def _count_agents_near(self, env, pos: Tuple[int, int], radius: int) -> int:
        """
        Count agents within radius of position.
        """
        count = 0
        for agent in env.agents:
            if agent.state == AgentState.MOVING:
                dist = np.hypot(agent.pos[0] - pos[0], agent.pos[1] - pos[1])
                if dist <= radius:
                    count += 1
        return count

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
        """
        Multi-step lookahead hazard utility.
    
        Converts concentration → probability at each surveyed cell,
        then combines with discounting (Lovreglio et al. 2016).

        Returns negative cost (will be multiplied by w_hazard).
        """
        # Immediate hazard
        c0 = env.hazard_field.concentration[y, x]
        p0 = 1 - np.exp(-env.hazard_field.k_harm * c0 * env.delta_t)
        p0 = np.clip(p0, 0, 0.999999)  # Numerical stability
    
        # 1-step lookahead (8 neighbors)
        sum_p1 = 0.0
        count1 = 0
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.width and 0 <= ny < env.height:
                c1 = env.hazard_field.concentration[ny, nx]
                p1 = 1 - np.exp(-env.hazard_field.k_harm * c1 * env.delta_t)
                sum_p1 += p1
                count1 += 1
        p1_avg = sum_p1 / count1 if count1 > 0 else 0.0
    
        # 2-step lookahead (radius=2, ~24 cells)
        sum_p2 = 0.0
        count2 = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) <= 1 and abs(dy) <= 1:
                    continue  # Skip cells already in 1-step
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.width and 0 <= ny < env.height:
                    c2 = env.hazard_field.concentration[ny, nx]
                    p2 = 1 - np.exp(-env.hazard_field.k_harm * c2 * env.delta_t)
                    sum_p2 += p2
                    count2 += 1
        p2_avg = sum_p2 / count2 if count2 > 0 else 0.0
    
        # Combine with discounting (profile-specific)
        alpha1 = 0.5 * self.profile_multiplier  # Cautious: higher, Bold: lower
        alpha2 = 0.25 * self.profile_multiplier
    
        utility = -(p0 + alpha1 * p1_avg + alpha2 * p2_avg)
        return utility
    
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

from .mixins.graph_movement_mixin import GraphMovementMixin

class GraphAgent(Agent, GraphMovementMixin):
    """Agent subclass using graph-constrained movement."""
    def decide_action(self, env):
        return self.graph_decide_action(env)


class PoliceAgent(Agent):
    """
    Police agent with gas deployment capability.
    
    Version 1: Rule-based behavior (move toward crowd, deploy gas)
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
            risk_tolerance=0.9,  # Less risk-averse
            rng=rng,
            profile_name='bold'  # Fast and less concerned about hazards
        )

        # OVERRIDE PARENT WEIGHTS
        self.w_hazard = 0.5        # Ignore gas (don't flee own deployment)
        self.w_goal = 5.0          # Strongly prioritize tactical position
        self.w_occupancy = 0.1     # Less concerned about crowding
        
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
                nearby_density = 0
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < env.width and 0 <= ny < env.height:
                            nearby_density += env.occupancy_count[ny, nx]
                wc_prob = wc_cfg.get('prob', 0.01)
                if nearby_density >= 15 and self.rng.random() < wc_prob:
                    env.hazards.deploy_water_cannon(
                        env=env,
                        x=x, y=y,
                        direction=(0, 1),
                        strength=wc_cfg.get('strength_m', 5.0),
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
            p_shoot = shoot_cfg.get('prob_per_step', 0.005)
            if self.shoot_cooldown <= 0 and self.rng.random() < p_shoot:
                candidates = [p for p in env.protesters if p.state == AgentState.MOVING]
                if not candidates:
                    candidates = env.protesters
                if candidates:
                    target = min(candidates, key=lambda a: (a.pos[0] - x)**2 + (a.pos[1] - y)**2)
                    env.hazards.shooting_event(env=env, shooter_agent=self, targets=[target])
                    self.shoot_cooldown = shoot_cfg.get('cooldown', 100)
            else:
                self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        return int(action)

    
    def _attempt_gas_deployment(self, env):
        """Deploy gas AHEAD toward crowd, not at police position."""
        x, y = self.pos
    
        # Get crowd centroid
        protester_positions = [a.pos for a in env.protesters if a.state == AgentState.MOVING]
        if not protester_positions:
            return
    
        centroid = np.mean(protester_positions, axis=0)
    
        # Deploy at point between police and crowd (not on obstacle)
        deploy_x = int((x + centroid[0]) / 2)
        deploy_y = int((y + centroid[1]) / 2)
    
        # Ensure deployment location is valid
        if env.obstacle_mask[deploy_y, deploy_x]:
            # Find nearest non-obstacle cell
            for offset in range(1, 10):
                for dx, dy in [(-offset,0), (offset,0), (0,-offset), (0,offset)]:
                    test_x, test_y = deploy_x + dx, deploy_y + dy
                    if (0 <= test_x < env.width and 0 <= test_y < env.height and 
                        not env.obstacle_mask[test_y, test_x]):
                        deploy_x, deploy_y = test_x, test_y
                        break
                        
        print(f"Police at {(x,y)}, deploying at {(deploy_x, deploy_y)}, obstacle={env.obstacle_mask[deploy_y, deploy_x]}")
        # Deploy gas
        inj_intensity = self.config['hazards']['gas'].get('inj_intensity', 12.0)
        env.hazards.deploy_gas(env=env, x=deploy_x, y=deploy_y, 
                            intensity=inj_intensity, agent_id=self.id)
        self.deploy_cooldown = self.deploy_cooldown_max