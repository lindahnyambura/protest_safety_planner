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
        },
        'vulnerable': {
            'speed_multiplier': 0.58,      # 0.7 m/s
            'risk_tolerance': 0.05,
            'w_hazard_multiplier': 2.0,    # Highly sensitive
            'lookahead_multiplier': 2.0,
            'profile_multiplier': 2.0,
            'panic_threshold': 0.2,
            'clustering_affinity': 0.9
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
            profile_name: Agent profile ('cautious', 'average', 'bold, 'vulnerable')
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
        
        # Panic threshold
        self.panic_threshold: float = 0.5  # default baseline; can be tuned per agent type

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

        # Behavioral and cognitive dynamics
        self.behavioral_state: str = 'CALM'   # CALM, ALERT, PANIC, FLEEING
        self.time_in_safe_zone: int = 0       # Steps spent in safe area
        self.T_MIN_SAFE: int = 300            # 5 minutes (steps) before exit-seeking

        # Dynamic multi-objective weights (will evolve via state transitions)
        self.goal_weights: Dict[str, float] = {
            'flee': 0.10,
            'safety': 0.40,
            'disperse': 0.30,
            'exit': 0.20
        }

    def update_behavioral_state(self, env):
        """
        Update behavioral state based on hazard exposure.
        
        State transitions:
        CALM → ALERT (hazard detected nearby)
        ALERT → PANIC (high concentration at position)
        PANIC → FLEEING (sustained exposure)
        FLEEING → CALM (reached safe zone)
        """
        x, y = self.pos
        local_hazard = env.hazard_field.concentration[y, x]
        
        # Check nearby hazard (3x3 neighborhood)
        nearby_hazard = 0.0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.width and 0 <= ny < env.height:
                    nearby_hazard = max(nearby_hazard, env.hazard_field.concentration[ny, nx])
        
        # State transitions
        if self.behavioral_state == 'CALM':
            if nearby_hazard > self.panic_threshold * 0.5:
                self.behavioral_state = 'ALERT'
                self._update_goal_weights('ALERT')
        
        elif self.behavioral_state == 'ALERT':
            if local_hazard > self.panic_threshold:
                self.behavioral_state = 'PANIC'
                self._update_goal_weights('PANIC')
            elif nearby_hazard < self.panic_threshold * 0.3:
                self.behavioral_state = 'CALM'
                self._update_goal_weights('CALM')
        
        elif self.behavioral_state == 'PANIC':
            if local_hazard > self.panic_threshold * 1.5:
                self.behavioral_state = 'FLEEING'
                self._update_goal_weights('FLEEING')
            elif local_hazard < self.panic_threshold * 0.5:
                self.behavioral_state = 'ALERT'
                self._update_goal_weights('ALERT')
        
        elif self.behavioral_state == 'FLEEING':
            if local_hazard < self.panic_threshold * 0.2 and nearby_hazard < self.panic_threshold * 0.3:
                self.behavioral_state = 'CALM'
                self._update_goal_weights('CALM')
                self.time_in_safe_zone += 1
            else:
                self.time_in_safe_zone = 0

        # Log state changes
        if hasattr(self, '_prev_behavioral_state'):
            if self._prev_behavioral_state != self.behavioral_state:
                if hasattr(env, '_log_event'):
                    current_node = getattr(self, 'current_node', None)
                    env._log_event('agent_state_change',
                                agent_id=self.id,
                                old_state=self._prev_behavioral_state,
                                new_state=self.behavioral_state,
                                current_node_id=current_node,
                                grid_pos=self.pos)
    
        self._prev_behavioral_state = self.behavioral_state
    def _update_goal_weights(self, state: str):
        """Update goal weights based on behavioral state."""
        if state == 'CALM':
            self.goal_weights = {
                'flee': 0.05,
                'safety': 0.35,
                'disperse': 0.35,
                'exit': 0.25 if self.time_in_safe_zone > self.T_MIN_SAFE else 0.05
            }
        elif state == 'ALERT':
            self.goal_weights = {
                'flee': 0.4,
                'safety': 0.4,
                'disperse': 0.15,
                'exit': 0.05
            }
        elif state == 'PANIC':
            self.goal_weights = {
                'flee': 0.7,
                'safety': 0.2,
                'disperse': 0.05,
                'exit': 0.05
            }
        elif state == 'FLEEING':
            self.goal_weights = {
                'flee': 0.85,
                'safety': 0.10,
                'disperse': 0.00,
                'exit': 0.05
            }

    def _select_dynamic_goal(self, env):
        """
        Select goal based on weighted behavioral objectives.

        Objectives:
        1. Flee: Move away from nearest hazard source
        2. Safety: Move toward low-risk zones
        3. Disperse: Avoid crowded areas
        4. Exit: Move toward nearest exit (only after T_MIN_SAFE in safe zone)
        """
        x, y = self.pos

        # Exit activation logic
        should_exit = self._should_prioritize_exit(env)
        if should_exit:
            # Override weights to prioritize exit
            self.goal_weights["exit"] = 0.7
            self.goal_weights["flee"] = 0.1
            self.goal_weights["safety"] = 0.15
            self.goal_weights["disperse"] = 0.05

        # Objective 1: Flee vector (away from hazards)
        flee_vector = np.zeros(2, dtype=float)
        if hasattr(env, "hazard_field") and hasattr(env.hazard_field, "concentration"):
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < env.width and 0 <= ny < env.height:
                        hazard = env.hazard_field.concentration[ny, nx]
                        if hazard > self.panic_threshold * 0.3:
                            # Repulsion inversely proportional to distance
                            dist = max(np.hypot(dx, dy), 1e-3)
                            flee_vector[0] -= dx / dist * hazard
                            flee_vector[1] -= dy / dist * hazard

        # Objective 2: Safety vector (toward low-risk areas)
        safety_vector = self._find_safest_direction(env)

        # Objective 3: Dispersion vector (away from crowds)
        disperse_vector = np.zeros(2, dtype=float)
        for agent in env.agents:
            if agent.id != self.id and agent.state == AgentState.MOVING:
                dx = self.pos[0] - agent.pos[0]
                dy = self.pos[1] - agent.pos[1]
                dist = max(np.hypot(dx, dy), 1e-3)
                if dist < 5:  # Only consider nearby agents
                    disperse_vector[0] += dx / (dist ** 2)
                    disperse_vector[1] += dy / (dist ** 2)

        # Objective 4: Exit vector (toward nearest exit)
        exit_vector = np.zeros(2, dtype=float)
        if hasattr(env, "exit_nodes") and self.time_in_safe_zone > self.T_MIN_SAFE:
            exits = env.exit_nodes.get("primary", []) + env.exit_nodes.get("secondary", [])
            if exits:
                nearest_exit = min(
                    exits, key=lambda e: np.hypot(e["grid_pos"][0] - x, e["grid_pos"][1] - y)
                )
                exit_vector[0] = nearest_exit["grid_pos"][0] - x
                exit_vector[1] = nearest_exit["grid_pos"][1] - y

        # Combine weighted vectors safely
        combined = (
            self.goal_weights.get("flee", 0.0) * flee_vector
            + self.goal_weights.get("safety", 0.0) * safety_vector
            + self.goal_weights.get("disperse", 0.0) * disperse_vector
            + self.goal_weights.get("exit", 0.0) * exit_vector
        )

        # Normalize and set goal
        magnitude = np.linalg.norm(combined)
        if magnitude > 1e-3:
            direction = combined / magnitude
            goal_x = int(np.clip(x + direction[0] * 20, 0, env.width - 1))
            goal_y = int(np.clip(y + direction[1] * 20, 0, env.height - 1))
            self.goal = (goal_x, goal_y)

    def _should_prioritize_exit(self, env) -> bool:
        """
        Determine if agent should prioritize exiting based on:
        1. Time spent in safe zone (T_MIN_SAFE threshold)
        2. Cumulative harm budget exceeded
        3. Mass exodus observed (social influence)
        """
        # Sufficient time spent in a safe zone
        if getattr(self, "time_in_safe_zone", 0.0) > getattr(self, "T_MIN_SAFE", 30.0):
            return True

        # Personal harm budget exceeded
        # H_crit = (
        #     self.config.get("hazards", {})
        #     .get("gas", {})
        #     .get("H_crit", 5.0)
        # )
        H_crit = getattr(self, "H_crit", 5.0)
        H_BUDGET = 0.5 * H_crit
        if getattr(self, "cumulative_harm", 0.0) > H_BUDGET:
            return True

        # Social influence: mass exodus around agent
        if not hasattr(env, "exit_nodes"):
            return False

        exits = env.exit_nodes.get("primary", []) + env.exit_nodes.get("secondary", [])
        if not exits:
            return False

        exit_positions = [e["grid_pos"] for e in exits]
        nearby_agents = 0
        exiting_agents = 0

        for agent in env.agents:
            # Skip self or inactive agents
            if agent.id == self.id or agent.state != AgentState.MOVING:
                continue

            dist = np.hypot(agent.pos[0] - self.pos[0], agent.pos[1] - self.pos[1])
            if dist < 10:  # Within 10-cell neighborhood
                nearby_agents += 1

                # Check if this agent’s goal is near an exit
                if hasattr(agent, "goal"):
                    for exit_pos in exit_positions:
                        if np.hypot(agent.goal[0] - exit_pos[0], agent.goal[1] - exit_pos[1]) < 5:
                            exiting_agents += 1
                            break

        # Require at least 5 neighbors to trigger social behavior
        if nearby_agents >= 5 and (exiting_agents / nearby_agents) > 0.3:
            return True

        return False

    def _find_safest_direction(self, env) -> np.ndarray:
        """
        Find direction toward safer areas using local hazard gradient descent.
        """
        x, y = self.pos
        gradient = np.zeros(2, dtype=float)

        if not hasattr(env, "hazard_field") or not hasattr(env.hazard_field, "concentration"):
            return gradient

        current_hazard = env.hazard_field.concentration[y, x]

        # Sample 8 neighboring directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.width and 0 <= ny < env.height:
                neighbor_hazard = env.hazard_field.concentration[ny, nx]
                hazard_diff = current_hazard - neighbor_hazard
                gradient[0] += dx * hazard_diff
                gradient[1] += dy * hazard_diff

        return gradient
 
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
        
        # Update behavioral state
        self.update_behavioral_state(env)

        # Check if reached goal
        if self._at_goal():
            self.state = AgentState.SAFE
            return 0
        
        # Dynamic goal selection
        self._select_dynamic_goal(env)

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
        - Ensures goal_node is kept in sync with goal for OSM graph mode.
        """
        
        # Parameters (tweakable)
        goal_update_period = getattr(self, "goal_update_period", 20)  # default every 20 steps
        beta_exit = getattr(self, "beta_exit", 0.2)                  # softmax inverse-temperature
        congestion_exp = getattr(self, "congestion_exp", 2.0)       # superlinear exponent
        congestion_thresh = getattr(self, "congestion_thresh", None) # optional thresholded behavior
        usage_penalty_k = getattr(self, "usage_penalty_k", 2.0)     # scale exit usage penalty
        immediate_reassess_hazard_factor = getattr(self, "risk_tolerance_immediate", 1.5)
        hysteresis_factor = getattr(self, "goal_hysteresis_factor", 1.05)
        
        
        # Step1: initialize counters if needed
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
        
        # Step 2: Precompute exit usage counts (how many agents currently targeting each exit)
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
        
        # Step 3: Convert to probabilities via stabilized softmax
        scores = np.array(exit_scores, dtype=np.float64)
        # numerical stabilization
        max_s = np.max(scores)
        exp_scores = np.exp(beta_exit * (scores - max_s))
        probs = exp_scores / (np.sum(exp_scores) + 1e-12)

        # Sample exit according to probs (stochastic choice)
        chosen_idx = self.rng.choice(len(exits), p=probs)
        chosen_exit = tuple(exits[chosen_idx])
        
        # Step 4: Hysteresis: only switch if chosen exit gives a sufficient improvement over current
        # (avoid flip-flopping if probabilities similar). We measure expected utility difference.
        current_goal = tuple(self.goal) if hasattr(self, "goal") else None
        if current_goal is None:
            self.goal = chosen_exit
        else:
            # compute current_goal_score for comparison
            try:
                curr_idx = exits.index(list(current_goal))
                current_score = scores[curr_idx]
            except ValueError:
                current_score = -np.inf

            should_switch = (
                scores[chosen_idx] > hysteresis_factor * current_score
                or self._estimate_path_hazard(env, current_goal)
                    > getattr(self, "risk_tolerance", 0.3) * 2.0
            )

            if should_switch:
                self.goal = chosen_exit
        
        # --- Step 5: sync to graph goal if applicable ---
        # Keep grid goal for visualization; update goal_node only if in graph mode
        if hasattr(self, "goal_node") and hasattr(env, "cell_to_node"):
            gx, gy = map(int, np.clip(chosen_exit, [0, 0], [env.width - 1, env.height - 1]))
            try:
                goal_node_id = env.cell_to_node[gy, gx]
                if goal_node_id not in (-1, None, "None", "nan"):
                    self.goal_node = str(goal_node_id)
            except Exception as e:
                print(f"[WARN] Agent {self.id} failed to sync goal_node: {e}")
    
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
        pos_x, pos_y = self.pos 

        # 1. Get moving protesters positions
        protester_positions = [a.pos for a in env.protesters if a.state == AgentState.MOVING]
        if not protester_positions:
            return 0  # STAY if no active protesters
        
        # 2. Crowd centroid (as float)
        centroid = np.mean(protester_positions, axis=0)  # array([x, y])

        # 3. Find nearest exit from config
        exits = env.config['agents']['protesters'].get('goals', {}).get('exit_points', [])
        # sanitize exit coordinates
        clean_exits = []
        for ex in exits:
            if isinstance(ex, (list, tuple)) and len(ex) >= 2:
                try:
                    clean_exits.append((float(ex[0]), float(ex[1])))
                except (ValueError, TypeError):
                    continue
            
        if not clean_exits:
            nearest_exit = np.array([env.width // 2, env.height // 2], dtype=float)
        else:
            dists = [np.hypot(centroid[0] - ex[0], centroid[1] - ex[1]) for ex in clean_exits]
            nearest_exit = np.array(clean_exits[int(np.argmin(dists))], dtype=float)
        
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
            # Check if protesters nearby
            nearby_protesters = sum(
                1 for p in env.protesters 
                if p.state == AgentState.MOVING and 
                np.hypot(p.pos[0] - pos_x, p.pos[1] - pos_y) < 15
            )
            # Deploy if 5+ protesters within range (instead of random prob)
            if nearby_protesters >= 2:
                self._attempt_gas_deployment(env)
                self.deploy_cooldown = self.deploy_cooldown_max
                print(f"[DEBUG] Police {self.id} deployed gas at step {env.step_count}")
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
                        nx, ny = pos_x + dx, pos_y + dy 
                        if 0 <= nx < env.width and 0 <= ny < env.height:
                            nearby_density += env.occupancy_count[ny, nx]
                
                wc_prob = wc_cfg.get('prob', 0.01)
                if nearby_density >= 15 and self.rng.random() < wc_prob:
                    env.hazards.deploy_water_cannon(
                        env=env,
                        x=pos_x, y=pos_y,
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
                if candidates:
                    target = min(candidates, key=lambda a: (a.pos[0] - pos_x)**2 + (a.pos[1] - pos_y)**2)
                    env.hazards.shooting_event(env=env, shooter_agent=self, targets=[target])
                    self.shoot_cooldown = shoot_cfg.get('cooldown', 100)
            else:
                self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        return int(action)

    
    def _attempt_gas_deployment(self, env):
        """Deploy gas AHEAD toward crowd, not at police position."""
        pos_x, pos_y = self.pos

        # Get crowd centroid
        protester_positions = [a.pos for a in env.protesters if a.state == AgentState.MOVING]
        if not protester_positions:
            return
    
        centroid = np.mean(protester_positions, axis=0)
    
        # Deploy at point between police and crowd (not on obstacle)
        deploy_x = int((pos_x + centroid[0]) / 2)
        deploy_y = int((pos_y + centroid[1]) / 2)
    
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
                        
        # Deploy gas
        inj_intensity = self.config['hazards']['gas'].get('inj_intensity', 12.0)
        env.hazards.deploy_gas(env=env, x=deploy_x, y=deploy_y, 
                            intensity=inj_intensity, agent_id=self.id)
        self.deploy_cooldown = self.deploy_cooldown_max

        print(f"[HAZARD] Step {env.step_count}: Police {self.id} deployed tear gas at ({deploy_x}, {deploy_y}), intensity={inj_intensity:.1f}")
        
        # NEW: Log with street name
        if hasattr(env, "_log_event"):
            deploy_node = None
            if hasattr(env, "cell_to_node"):
                try:
                    deploy_node = env.cell_to_node[deploy_y, deploy_x]
                except Exception:
                    pass  # fail-safe if grid-node mapping incomplete
            env._log_event(
                "hazard_deployed",
                agent_id=self.id,
                node_id=deploy_node,
                grid_pos=(deploy_x, deploy_y),
                hazard_type="tear_gas",
                intensity=inj_intensity,
            )