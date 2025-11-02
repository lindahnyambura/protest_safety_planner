"""
agent.py - Agent classes for protesters and police

Version 1: Basic agent movement with fractional speed, homogeneous protesters
Version 2: Add agent heterogeneity (4 types)
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
            'w_hazard_multiplier': 3.0,     # sensitive to hazards
            'lookahead_multiplier': 2.0,  # More concerned about future hazards'
            'profile_multiplier': 2.0,
            'panic_threshold': 0.3,
            'goal_update_period': 8,       # NEW: Faster reassessment
            'T_MIN_SAFE': 60,  # NEW: Shorter wait (was 300, now 60s = 1 min)
            'exit_urgency_multiplier': 2.0  # NEW: Boost exit seeking
        },
        'average': {
            'speed_multiplier': 1.0,       # 1.2 m/s (normal walking speed)
            'risk_tolerance': 0.3,         # Moderate caution
            'w_hazard_multiplier': 2.0,     # Standard weight
            'lookahead_multiplier': 1.5,  # Standard lookahead
            'profile_multiplier': 1.5,
            'panic_threshold': 0.5,
            'goal_update_period': 12,       # NEW: Slower reassessment   
            'T_MIN_SAFE': 80,  # NEW: Was 300
            'exit_urgency_multiplier': 1.5
        },
        'bold': {
            'speed_multiplier': 1.17,      # 1.4 m/s (fast walking/jogging speed)
            'risk_tolerance': 0.6,         # Risk-taking
            'w_hazard_multiplier': 1.0 ,    # Less concerned about hazards
            'lookahead_multiplier': 1.0,  # Less concerned about future hazards
            'profile_multiplier': 1.0,
            'panic_threshold': 0.7,
            'goal_update_period': 15,       # NEW: Faster reassessment
            'T_MIN_SAFE': 100,  # NEW: Bold agents stay longer
            'exit_urgency_multiplier': 1.0
        },
        'vulnerable': {
            'speed_multiplier': 0.58,      # 0.7 m/s
            'risk_tolerance': 0.05,
            'w_hazard_multiplier': 5.0,    # Highly sensitive
            'lookahead_multiplier': 3.0,
            'profile_multiplier': 3.0,
            'panic_threshold': 0.2,
            'clustering_affinity': 0.9,
            'goal_update_period': 6,       # NEW: Faster reassessment
            'T_MIN_SAFE': 40,  # NEW: Vulnerable flee quickly
            'exit_urgency_multiplier': 3.0
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

        # Apply profile-specific parameters
        self.T_MIN_SAFE = profile.get('T_MIN_SAFE', 80)
        self.exit_urgency_multiplier = profile.get('exit_urgency_multiplier', 1.0)
        self.goal_update_period = profile.get('goal_update_period', 12)
        
        # State
        self.state: str = AgentState.MOVING
        
        # Harm tracking (dual-purpose model)
        self.cumulative_harm: np.float32 = np.float32(0.0)
        self.harm_events: int = 0
        
        # Movement mechanics
        self.move_accum: np.float32 = np.float32(0.0)
        self.last_move_direction: Optional[int] = None
        
        # RNG for stochastic decisions
        agent_seed = rng.integers(0, 2**31)  # Draw unique seed from parent RNG
        self.rng = np.random.default_rng(agent_seed)  # Create independent RNG
        #print(f"[DEBUG] Agent {agent_id} RNG created: seed drawn from parent")
        
        # Scoring weights (locked parameters, modified by profile)
        # w_goal: preference for goal proximity
        # w_hazard: penalty for entering hazardous zones (scaled by risk_tolerance and profile)
        # w_occupancy: penalty for crowded cells
        # w_inertia: bias toward continuing previous direction
        # beta: Boltzmann temperature for stochastic action choice
        self.w_goal: float = 1.0
        self.w_hazard = 20.0 * (1.0 - self.risk_tolerance) * profile['w_hazard_multiplier'] # Now average agent: w_hazard = 20 × 0.7 × 2.0 = 28.0 (much stronger)
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
        
        # Check nearby hazard (5X5 neighborhood)
        nearby_hazard = 0.0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.width and 0 <= ny < env.height:
                    nearby_hazard = max(nearby_hazard, env.hazard_field.concentration[ny, nx])
        
        # NEW: Safe zone detection (even while moving)
        SAFE_ZONE_THRESHOLD = self.panic_threshold * 0.15  # Very low hazard
        if nearby_hazard < SAFE_ZONE_THRESHOLD:
            self.time_in_safe_zone += 1
        else:
            self.time_in_safe_zone = max(0, self.time_in_safe_zone - 2)  # Decay faster

        # State transitions
        if self.behavioral_state == 'CALM':
            if nearby_hazard > self.panic_threshold * 0.4:
                self.behavioral_state = 'ALERT'
                self._update_goal_weights('ALERT')
        
        elif self.behavioral_state == 'ALERT':
            if local_hazard > self.panic_threshold:
                self.behavioral_state = 'PANIC'
                self._update_goal_weights('PANIC')
            elif nearby_hazard < self.panic_threshold * 0.2:
                self.behavioral_state = 'CALM'
                self._update_goal_weights('CALM')
        
        elif self.behavioral_state == 'PANIC':
            if local_hazard > self.panic_threshold * 1.5:
                self.behavioral_state = 'FLEEING'
                self._update_goal_weights('FLEEING')
            elif local_hazard < self.panic_threshold * 0.4:
                self.behavioral_state = 'ALERT'
                self._update_goal_weights('ALERT')
        
        elif self.behavioral_state == 'FLEEING':
            if local_hazard < self.panic_threshold * 0.1 and nearby_hazard < self.panic_threshold * 0.2:
                self.behavioral_state = 'CALM'
                self._update_goal_weights('CALM')
            #     self.time_in_safe_zone += 1
            # else:
            #     self.time_in_safe_zone = 0

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
                    pass
    
        self._prev_behavioral_state = self.behavioral_state
    def _update_goal_weights(self, state: str):
        """Update goal weights based on behavioral state."""
        if state == 'CALM':
            base_weights = {
                'flee': 0.05,
                'safety': 0.35,
                'disperse': 0.35,
                'exit': 0.15
            }
        elif state == 'ALERT':
            base_weights = {
                'flee': 0.4,
                'safety': 0.3,
                'disperse': 0.15,
                'exit': 0.15
            }
        elif state == 'PANIC':
            base_weights = {
                'flee': 0.6,
                'safety': 0.15,
                'disperse': 0.05,
                'exit': 0.15
            }
        elif state == 'FLEEING':
            base_weights = {
                'flee': 0.70,
                'safety': 0.05,
                'disperse': 0.00,
                'exit': 0.15
            }

        else:
            base_weights = {
                'flee': 0.1,
                'safety': 0.3,
                'disperse': 0.3,
                'exit': 0.3
            }

        # NEW: Time-based urgency boost
        # After T_MIN_SAFE, gradually increase exit weight
        T_MIN_SAFE = getattr(self, 'T_MIN_SAFE', 80)
        safe_time = getattr(self, 'time_in_safe_zone', 0)
    
        if safe_time > T_MIN_SAFE:
            # Exponential urgency increase
            urgency_factor = min(3.0, 1.0 + (safe_time - T_MIN_SAFE) / 50.0)
            exit_urgency_mult = getattr(self, 'exit_urgency_multiplier', 1.0)
        
            # Redistribute weights: boost exit, reduce others
            total_non_exit = sum(v for k, v in base_weights.items() if k != 'exit')
            scale_down = 1.0 - (base_weights['exit'] * urgency_factor * exit_urgency_mult)
            scale_down = max(0.1, scale_down)  # Ensure we don't zero out other behaviors
        
            for key in ['flee', 'safety', 'disperse']:
                base_weights[key] *= scale_down / total_non_exit
        
            base_weights['exit'] *= urgency_factor * exit_urgency_mult
            base_weights['exit'] = min(0.85, base_weights['exit'])  # Cap at 85%
    
        # Normalize to sum to 1.0
        total = sum(base_weights.values())
        self.goal_weights = {k: v/total for k, v in base_weights.items()}

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
        ENHANCED: Multiple exit triggers with global simulation time.
    
        New: After 150 steps (~2.5 minutes), agents start seeking exits
        even if not in safe zone.
        """
        # Trigger 1: Time in safe zone (personal)
        T_MIN_SAFE = getattr(self, 'T_MIN_SAFE', 80)
        T_MIN_SAFE_SCALED = T_MIN_SAFE * (100 / env.width)  # Scale with grid
    
        if getattr(self, 'time_in_safe_zone', 0) > T_MIN_SAFE_SCALED:
            return True
    
        # Trigger 2: Global simulation time (NEW - CRITICAL)
        # After 150 steps, protest is "winding down" - agents leave
        if env.step_count > 150:
            # Probability increases with time
            exit_prob = (env.step_count - 150) / 100.0  # 0.0 at 150, 0.5 at 200
            exit_prob = min(0.8, exit_prob)
        
            if env.rng.random() < exit_prob:
                return True
    
        # Trigger 3: Harm budget (unchanged)
        H_crit = env.config.get('hazards', {}).get('gas', {}).get('H_crit', 5.0)
        H_BUDGET = 0.5 * H_crit
        if getattr(self, 'cumulative_harm', 0.0) > H_BUDGET:
            return True
    
        # Trigger 4: Social influence (unchanged)
        if not hasattr(env, 'exit_nodes'):
            return False
    
        exits = env.exit_nodes.get('primary', []) + env.exit_nodes.get('secondary', [])
        if not exits:
            return False
    
        exit_positions = [e['grid_pos'] for e in exits]
        nearby_agents = 0
        exiting_agents = 0
    
        for agent in env.agents:
            if agent.id == self.id or agent.state != AgentState.MOVING:
                continue
        
            dist = np.hypot(agent.pos[0] - self.pos[0], agent.pos[1] - self.pos[1])
            if dist < 15:
                nearby_agents += 1
            
                if hasattr(agent, 'goal'):
                    for exit_pos in exit_positions:
                        if np.hypot(agent.goal[0] - exit_pos[0], 
                                agent.goal[1] - exit_pos[1]) < 8:
                            exiting_agents += 1
                            break
    
        if nearby_agents >= 5 and (exiting_agents / nearby_agents) > 0.3:
            return True
    
        # Trigger 5: Hazard saturation (unchanged)
        x, y = self.pos
        hazard_count = 0
        total_checked = 0
    
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.width and 0 <= ny < env.height:
                    total_checked += 1
                    if env.hazard_field.concentration[ny, nx] > self.panic_threshold * 0.5:
                        hazard_count += 1
    
        hazard_saturation = hazard_count / total_checked if total_checked > 0 else 0.0
        if hazard_saturation > 0.4:
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
        action = self._select_action_stochastic(scores, env)
        
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
        goal_update_period = getattr(self, "goal_update_period", 12)  # default every 12 steps
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

            immediate_threshold = getattr(self, 'risk_tolerance', 0.3) * 2.0
            if curr_path_hazard > immediate_threshold:
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
        chosen_idx = env.rng.choice(len(exits), p=probs)
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
    
    def _select_action_stochastic(self, scores: np.ndarray, env) -> int:
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
        action = env.rng.choice(valid_actions, p=probs)
        
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

        # 1. Get moving protesters with proper position handling
        protester_positions = []
        for a in env.protesters:
            if a.state != AgentState.MOVING:
                continue

            # Get actual grid position (critical for graph mode)
            if hasattr(a, 'current_node') and env.osm_graph is not None:
                try:
                    px, py = env._node_to_cell(a.current_node)
                    protester_positions.append((px, py))
                except:
                    continue
            else:
                protester_positions.append(a.pos)

        if not protester_positions:
            return 0  # STAY if no active protesters
        
        # 2. Crowd centroid (as float)
        centroid = np.mean(protester_positions, axis=0)  # array([x, y])

        # 3. Find nearest exit and compute intercept
        exits = env.config['agents']['protesters'].get('goals', {}).get('exit_points', [])
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
        
        # 4. Intercept point with offset
        intercept = ((centroid + nearest_exit) / 2.0).astype(int)
        dir_vec = nearest_exit - centroid
        perp = np.array([-dir_vec[1], dir_vec[0]])
        perp_norm = perp / (np.linalg.norm(perp) + 1e-8)
        offset_magnitude = (self.id % 7) - 3
        offset = (perp_norm * offset_magnitude).astype(int)
        target = intercept + offset
        target_x = int(np.clip(target[0], 0, env.width - 1))
        target_y = int(np.clip(target[1], 0, env.height - 1))

        # set goal toward intercept-target
        self.goal = (target_x, target_y)

        # 5. Score moves toward goal
        scores = self._score_neighbors(env)
        action = self._select_action_stochastic(scores, env)

        # 6. SMART GAS DEPLOYMENT
        if not hasattr(self, 'deploy_cooldown'):
            self.deploy_cooldown = 0
    
        if self.deploy_cooldown <= 0:
            # Count protesters within tactical range (< 20 cells)
            nearby_protesters = []
            for px, py in protester_positions:
                dist = np.hypot(px - pos_x, py - pos_y)
                if dist < 20:  # Within tactical range
                    nearby_protesters.append((px, py, dist))
        
            n_nearby = len(nearby_protesters)
        
            # ADAPTIVE deployment probability based on proximity and density
            if n_nearby >= 10:
                # High density: deploy more frequently
                effective_prob = self.deploy_prob * 2.0
            elif n_nearby >= 5:
                # Medium density: normal rate
                effective_prob = self.deploy_prob
            elif n_nearby >= 2:
                # Low density: reduced rate
                effective_prob = self.deploy_prob * 0.3
            else:
                # No nearby targets: don't deploy
                effective_prob = 0.0
        
            # Deploy if probability check passes
            if effective_prob > 0 and env.rng.random() < effective_prob:
                self._attempt_gas_deployment(env)
                self.deploy_cooldown = self.deploy_cooldown_max
            
                # DEBUG logging (first 20 steps only)
                if env.step_count <= 20:
                    print(f"[DEBUG] Police {self.id} deployed gas at step {env.step_count} "
                        f"({n_nearby} protesters within 20 cells)")
        else:
            self.deploy_cooldown -= 1

        # 7. Water cannon: separate cooldown + probability, via config hazards.water_cannon
        wc_cfg = env.config.get('hazards', {}).get('water_cannon', {})
        if wc_cfg.get('enabled', False):
            if not hasattr(self, 'wc_cooldown'):
                self.wc_cooldown = 0

            if self.wc_cooldown <= 0:
                # Count nearby protesters (within tactical range)
                nearby_protesters = []
                for px, py in protester_positions:
                    dist = np.hypot(px - pos_x, py - pos_y)
                    if dist < 15:  # Within water cannon effective range
                        nearby_protesters.append((px, py, dist))
            
                n_nearby = len(nearby_protesters)
                
                # ADAPTIVE probability based on proximity
                if n_nearby >= 8:
                    wc_prob = wc_cfg.get('prob', 0.02) * 2.0  # 4% if many nearby
                elif n_nearby >= 4:
                    wc_prob = wc_cfg.get('prob', 0.02) * 1.5  # 3% if some nearby
                elif n_nearby >= 2:
                    wc_prob = wc_cfg.get('prob', 0.02)        # 2% if few nearby
                else:
                    wc_prob = 0.0  # Don't waste on empty areas
            
                # Deploy if roll succeeds
                if wc_prob > 0 and env.rng.random() < wc_prob:
                    # Find nearest protester to target
                    if nearby_protesters:
                        target = min(nearby_protesters, key=lambda p: p[2])
                        target_x, target_y, _ = target
                    
                        # Direction from police to target
                        direction = np.array([target_x - pos_x, target_y - pos_y])
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction = direction / norm
                            direction = (int(np.sign(direction[0])), int(np.sign(direction[1])))
                        else:
                            direction = (0, 1)  # Default: forward
                    
                        env.hazards.deploy_water_cannon(
                            env=env,
                            x=pos_x, y=pos_y,
                            direction=direction,
                            strength=wc_cfg.get('strength_m', 5.0),
                            radius=wc_cfg.get('radius', 6),
                            stun_prob=wc_cfg.get('stun_prob', 0.15),
                            agent_id=self.id
                        )
                        self.wc_cooldown = wc_cfg.get('cooldown', 30)
                    
                        # DEBUG logging
                        from src.utils.logging_config import logger
                        logger.debug(f"[WATER] Police {self.id} deployed water cannon "
                                   f"at ({pos_x},{pos_y}), {n_nearby} targets")
            else:
                self.wc_cooldown = max(0, self.wc_cooldown - 1)

        # 8. Shooting: very rare, via hazards.shooting_event
        shoot_cfg = env.config.get('hazards', {}).get('shooting', {})
        if shoot_cfg.get('enabled', False):
            if not hasattr(self, 'shoot_cooldown'):
                self.shoot_cooldown = 0
        
            if self.shoot_cooldown <= 0:
                # Count nearby moving protesters
                candidates = []
                for a in env.protesters:
                    if a.state != AgentState.MOVING:
                        continue
                
                    # Get position
                    if hasattr(a, 'current_node') and env.osm_graph:
                        try:
                            ax, ay = env._node_to_cell(a.current_node)
                        except:
                            continue
                    else:
                        ax, ay = a.pos
                
                    dist = np.hypot(ax - pos_x, ay - pos_y)
                    if dist < 20:  # Within shooting range
                        candidates.append((a, dist))
            
                # Only shoot if close targets exist
                if candidates:
                    p_shoot = shoot_cfg.get('prob_per_step', 0.0005)
                
                    # INCREASED probability if many nearby (crowd control)
                    n_candidates = len(candidates)
                    if n_candidates >= 10:
                        p_shoot *= 3.0  # 0.15% if dense crowd
                    elif n_candidates >= 5:
                        p_shoot *= 2.0  # 0.1% if some crowd
                
                    if env.rng.random() < p_shoot:
                        # Target nearest protester
                        target_agent, target_dist = min(candidates, key=lambda x: x[1])
                    
                        env.hazards.shooting_event(
                            env=env,
                            shooter_agent=self,
                            targets=[target_agent]
                        )
                        self.shoot_cooldown = shoot_cfg.get('cooldown', 100)
                    
                        # DEBUG logging
                        from src.utils.logging_config import logger
                        outcome = "INCAP" if target_agent.state == AgentState.INCAPACITATED else "STUNNED"
                        logger.debug(f"[SHOOT] Police {self.id} fired at Agent {target_agent.id} "
                                   f"(dist={target_dist:.1f}) → {outcome}")
            else:
                self.shoot_cooldown = max(0, self.shoot_cooldown - 1)

        return int(action)

    
    def _attempt_gas_deployment(self, env):
        """
        STRATEGIC gas deployment with formation tactics.
    
        Literature: King & Waddington (2004) - Police tactical formations
    
        Strategy:
        1. Deploy in line perpendicular to crowd flow
        2. Create gas "curtain" to halt advance
        3. Space deployments 15-20m apart (3-4 cells)
        """
        pos_x, pos_y = self.pos
    
        # Get crowd flow direction
        protester_positions = [a.pos for a in env.protesters if a.state == AgentState.MOVING]
        if not protester_positions:
            return
    
        centroid = np.mean(protester_positions, axis=0)
    
        # Crowd movement vector
        crowd_dir = centroid - np.array([pos_x, pos_y])
        crowd_norm = np.linalg.norm(crowd_dir)
        if crowd_norm < 1e-6:
            return
        crowd_dir = crowd_dir / crowd_norm
    
        # Perpendicular vector (for line formation)
        perp_dir = np.array([-crowd_dir[1], crowd_dir[0]])
    
        # Deploy along perpendicular line (offset by police ID)
        offset = ((self.id % 5) - 2) * 4  # -8, -4, 0, 4, 8 cells
    
        deploy_x = int(np.clip(pos_x + offset * perp_dir[0], 0, env.width - 1))
        deploy_y = int(np.clip(pos_y + offset * perp_dir[1], 0, env.height - 1))
    
        # CRITICAL: Validate deployment location (with improved fallback)
        if not (0 <= deploy_x < env.width and 0 <= deploy_y < env.height):
            return
    
        if env.obstacle_mask[deploy_y, deploy_x]:
            # Find nearest valid cell (within 10-cell radius)
            found_valid = False
            for radius in range(1, 11):
                for angle in np.linspace(0, 2*np.pi, 8*radius, endpoint=False):
                    test_x = int(deploy_x + radius * np.cos(angle))
                    test_y = int(deploy_y + radius * np.sin(angle))
                
                    if (0 <= test_x < env.width and 0 <= test_y < env.height and
                        not env.obstacle_mask[test_y, test_x]):
                        deploy_x, deploy_y = test_x, test_y
                        found_valid = True
                        break
                if found_valid:
                    break
        
            if not found_valid:
                return  # Abort deployment
    
        # Deploy with configured intensity
        inj_intensity = self.config['hazards']['gas'].get('inj_intensity', 50.0)
        env.hazards.deploy_gas(
            env=env,
            x=deploy_x,
            y=deploy_y,
            intensity=inj_intensity,
            agent_id=self.id
        )
        self.deploy_cooldown = self.deploy_cooldown_max
    
        # Conditional logging
        from src.utils.logging_config import logger
        logger.debug(f"[GAS] Police {self.id} deployed at ({deploy_x},{deploy_y}), "
                    f"intensity={inj_intensity:.1f}, formation_offset={offset}")