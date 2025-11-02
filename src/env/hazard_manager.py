"""
hazard_manager.py - HazardManager class to manage multiple hazard types

Manage multiple hazard types:
  - gas: diffusive HazardField (existing)
  - water_cannon: instant directional push + stun
  - shooting: rare, direct incapacitation events
Logging of hazard events centralized here.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from .hazards import HazardField
from .agent import AgentState

class HazardManager:
    """
    Manage multiple hazard types:
      - gas: diffusive HazardField (existing)
      - water_cannon: instant directional push + stun
      - shooting: rare, direct incapacitation events
    Logging of hazard events centralized here.
    """

    def __init__(self, height: int, width: int, config: Dict[str, Any], rng: np.random.Generator,
                 cell_size_m: float, obstacle_mask: np.ndarray):
        """
        Initialize HazardManager.
    
        Args:
            height, width: Grid dimensions
            config: Full config dict with 'hazards' section
            rng: Random number generator
            cell_size_m: Physical size of grid cells (meters)
            obstacle_mask: Boolean array marking impassable cells
        """

        self.rng = rng
        self.config = config.get('hazards', {})  # Extract hazards section
        self.events: List[Dict[str, Any]] = []
        self.stunned_agents = {}  # {agent_id: recovery_timestep}

        # Gas configuration
        gas_cfg = self.config.get('gas', {})
        time_cfg = config.get('time', {})
        delta_t = time_cfg.get('delta_t', 1.0)

        # Create gas field with all required parameters
        self.gas = HazardField(
            height=height,
            width=width,
            diffusion_coeff=gas_cfg.get('diffusion_coeff', 0.3),
            decay_rate=gas_cfg.get('decay_rate', 0.01),
            k_harm=gas_cfg.get('k_harm', 0.0083),
            delta_t=self.config.get('delta_t', 1.0),
            cell_size_m=cell_size_m,
            wind_direction=tuple(gas_cfg.get('wind_direction', [0, 0])),
            wind_speed_m_s=gas_cfg.get('wind_speed_m_s', 0.0),
            obstacle_mask=obstacle_mask  # to be set by env if needed
        )


    def update(self, delta_t: float):
        """Update diffusive hazards (gas) each timestep and recover stunned agents."""
        # Update gas field
        self.gas.update(delta_t)

    def stun_agent(self, agent, env, stun_duration: int = 10):
        """
        Stun an agent for a duration.
        
        Args:
            agent: Agent to stun
            env: Environment (for timestep)
            stun_duration: Steps until recovery (default 10s)
        """
        agent.state = AgentState.STUNNED
        recovery_time = env.step_count + stun_duration
        self.stunned_agents[agent.id] = recovery_time

    def check_stun_recovery(self, env):
        """
        Check if any stunned agents should recover.
        Call this every step from env.
        """
        current_time = env.step_count
        recovered = []
        
        for agent_id, recovery_time in list(self.stunned_agents.items()):
            if current_time >= recovery_time:
                # Find agent and recover
                agent = next((a for a in env.agents if a.id == agent_id), None)
                if agent and agent.state == AgentState.STUNNED:
                    agent.state = AgentState.MOVING
                    recovered.append(agent_id)
                del self.stunned_agents[agent_id]
        
        return recovered
    

    # Gas deployment (wrapper)
    def deploy_gas(self, env, x: int, y: int, intensity: float, duration_steps: int = 30, agent_id: Optional[int] = None):
        """
        Deploy tear gas canister with sustained emission.
    
        Args:
            env: Environment instance
            x, y: Deployment location
            intensity: Emission rate (mg/m³/s)
            duration_steps: How long canister emits (default 30s)
            agent_id: Police officer who deployed
        """
        
        duration = self.config.get('gas', {}).get('emission_duration', 30)
    
        # Add source with metadata
        self.gas.active_sources.append({
            'x': int(x),
            'y': int(y),
            'intensity': float(intensity),
            'duration': int(duration),
            'initial_duration': int(duration),  # NEW: Track for emission profile
            'deployed_at_step': env.step_count  # NEW: For analytics
        })
    
        # Log event
        ev = {
            'timestep': env.step_count,
            'event_type': 'gas_deployment',
            'agent_id': agent_id,
            'location': (int(x), int(y)),
            'intensity': float(intensity),
            'duration_steps': int(duration)
        }
        self.events.append(ev)
        if hasattr(env, 'events_log'):
            env.events_log.append(ev)


    # Water cannon (instant)
    def deploy_water_cannon(self, env, x: int, y: int, direction: Tuple[int, int],
                             strength: int = 3, radius: int = 5, stun_prob: float = 0.1,
                             agent_id: Optional[int] = None):
        """
        Water cannon: directional push + stun within radius.
    
        Args:
            direction: (dx, dy) push direction (-1, 0, 1)
            strength: Push distance in CELLS
            radius: Effect radius in CELLS (Euclidean distance)
        """
       
        affected = []

        for agent in list(env.protesters):
            ax, ay = agent.pos
        
            # Use Euclidean distance
            dist = np.hypot(ax - x, ay - y)
            if dist > radius:
                continue  # Out of range
        
            # Compute push displacement
            push_x = int(np.clip(ax + direction[0] * strength, 0, env.width - 1))
            push_y = int(np.clip(ay + direction[1] * strength, 0, env.height - 1))
        
            # Only move if target cell is valid
            if not env.obstacle_mask[push_y, push_x]:
                agent.pos = (push_x, push_y)

            else:
                # Hit obstacle: agent is stunned instead
                agent.state = AgentState.STUNNED
        
            # Stun probability
            if self.rng.random() < stun_prob:
                self.stun_agent(agent, env, stun_duration=10)
            
        
            affected.append(agent.id)
        
        # Log event
        ev = {
            'timestep': env.step_count,
            'event_type': 'water_cannon',
            'agent_id': agent_id,
            'origin': (int(x), int(y)),
            'direction': (int(direction[0]), int(direction[1])),
            'strength': int(strength),
            'radius': int(radius),
            'affected_count': len(affected),
            'affected_ids': affected
        }
        self.events.append(ev)
        if hasattr(env, 'events_log'):
            env.events_log.append(ev)
        return affected


    # Shooting event (rare)
    def shooting_event(self, env, shooter_agent, targets: List = None):
        """
        Shooting event: probabilistic incapacitation.
    
        Args:
            env: Environment instance
            shooter_agent: Police officer who shot
            targets: List of target agents (if None, auto-select)
        """
        # Get lethality from config
        shoot_cfg = self.config.get('shooting', {})
        lethality = shoot_cfg.get('lethality', 0.3)

        # Auto-select targets if not provided
        if targets is None:
            if not env.protesters:
                return []
            # Find nearest moving protester
            candidates = [a for a in env.protesters if a.state == AgentState.MOVING]
            if not candidates:
                candidates = env.protesters
            nearest = min(candidates, key=lambda a: 
                        (a.pos[0] - shooter_agent.pos[0])**2 + 
                        (a.pos[1] - shooter_agent.pos[1])**2)
            targets = [nearest]

        affected = []
        for target in targets:
            # Probabilistic incapacitation
            if self.rng.random() < lethality:
                target.state = AgentState.INCAPACITATED
            else:
                self.stun_agent(target, env, stun_duration=20)
        
            affected.append(target.id)

            # Log each shooting
            ev = {
                'timestep': env.step_count,
                'event_type': 'shooting',
                'shooter_id': shooter_agent.id,
                'target_id': target.id,
                'location': tuple(target.pos),
                'incapacitated': (target.state == AgentState.INCAPACITATED),
                'lethality_param': float(lethality)  # NEW: Log probability used
            }
            self.events.append(ev)
            if hasattr(env, 'events_log'):
                env.events_log.append(ev)
    
        return affected