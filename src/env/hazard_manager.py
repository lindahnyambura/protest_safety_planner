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

    def __init__(self, height: int, width: int, config: Dict[str, Any], rng: np.random.Generator):
        self.rng = rng
        self.config = config or {}
        gas_cfg = self.config.get('gas', {})
        # Keep existing HazardField API for gas
        self.gas = HazardField(
            height=height,
            width=width,
            diffusion_coeff=gas_cfg.get('diffusion_coeff', 0.2),
            decay_rate=gas_cfg.get('decay_rate', 0.05),
            k_harm=gas_cfg.get('k_harm', 0.1386),
            delta_t=self.config.get('delta_t', 1.0)
        )

        # expose convenience for backward compatibility
        # e.g. env.hazard_field -> env.hazards.gas
        self.events: List[Dict[str, Any]] = []

    def update(self, delta_t: float):
        """Update diffusive hazards (gas) each timestep."""
        self.gas.update(delta_t)

    # Gas deployment (wrapper)
    def deploy_gas(self, env, x: int, y: int, intensity: float, agent_id: Optional[int] = None):
        """Add gas source and log event."""
        self.gas.add_source(x, y, intensity)
        ev = {
            'timestep': env.step_count,
            'event_type': 'gas_deployment',
            'agent_id': agent_id,
            'location': (int(x), int(y)),
            'intensity': float(intensity)
        }
        self.events.append(ev)
        # also add to env events_log for compatibility if available
        if hasattr(env, 'events_log'):
            env.events_log.append(ev)


    # Water cannon (instant)
    def deploy_water_cannon(self, env, x: int, y: int, direction: Tuple[int, int],
                             strength: int = 3, radius: int = 5, stun_prob: float = 0.1,
                             agent_id: Optional[int] = None):
        """
        Directional instantaneous effect: push agents and optionally stun.
        direction: (dx, dy) integer direction vector (should be normalized to -1,0,1)
        """
        affected = []
        for agent in list(env.protesters):  # copy to allow modifications
            ax, ay = agent.pos
            if abs(ax - x) <= radius and abs(ay - y) <= radius:
                # compute push displacement
                push_x = int(np.clip(ax + int(direction[0]) * strength, 0, env.width - 1))
                push_y = int(np.clip(ay + int(direction[1]) * strength, 0, env.height - 1))
                # only move if target is not obstacle and within bounds
                if not env.obstacle_mask[push_y, push_x]:
                    agent.pos = (push_x, push_y)
                # stun chance
                if self.rng.random() < stun_prob:
                    agent.state = AgentState.STUNNED
                affected.append(agent.id)

        ev = {
            'timestep': env.step_count,
            'event_type': 'water_cannon',
            'agent_id': agent_id,
            'origin': (int(x), int(y)),
            'direction': (int(direction[0]), int(direction[1])),
            'strength': int(strength),
            'radius': int(radius),
            'affected': affected
        }
        self.events.append(ev)
        if hasattr(env, 'events_log'):
            env.events_log.append(ev)
        return affected


    # Shooting event (rare)
    def shooting_event(self, env, shooter_agent, targets: List = None, fatal: bool = True):
        """
        Apply shooting effect: directly change target state (incapacitated or arrested).
        `targets`: list of Agent objects (if None, choose nearest within some limit)
        """
        if targets is None:
            # choose nearest protester if any
            if not env.protesters:
                return None
            # find nearest moving protester
            targets = [min(env.protesters, key=lambda a: (a.pos[0]-shooter_agent.pos[0])**2 + (a.pos[1]-shooter_agent.pos[1])**2)]

        affected = []
        for t in targets:
            if fatal:
                t.state = AgentState.INCAPACITATED
            else:
                # non-fatal: high chance incapacitate else stunned
                if self.rng.random() < 0.7:
                    t.state = AgentState.INCAPACITATED
                else:
                    t.state = AgentState.STUNNED
            affected.append(t.id)

            ev = {
                'timestep': env.step_count,
                'event_type': 'shooting',
                'shooter_id': shooter_agent.id,
                'target_id': t.id,
                'location': t.pos,
                'fatal': bool(fatal)
            }
            self.events.append(ev)
            if hasattr(env, 'events_log'):
                env.events_log.append(ev)
        return affected
