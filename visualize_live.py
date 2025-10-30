#!/usr/bin/env python3
"""
visualize_live.py — Enhanced real-time protest simulation visualization
------------------------------------------------------------------------
Features:
- Clear agent movement with trails
- Real-time statistics overlay
- Hazard deployment tracking
- Detailed terminal summary
- Support for both grid and OSM graph modes
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict, deque
import time
import matplotlib
matplotlib.use("TkAgg")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.protest_env import ProtestEnv, load_config
from src.env.agent import AgentState
from src.utils.logging_config import logger, LogLevel

# Set visualization logging level
logger.set_level(LogLevel.NORMAL)  # Quiet during live viz


class SimulationTracker:
    """Track simulation statistics for terminal summary."""
    
    def __init__(self):
        """
        Initialize simulation tracker.

        Tracks:
        - Agent trajectories (position over time)
        - Hazard deployment events (time, location, type)
        - Congestion events (time, location, type)
        - Agent state transitions (time, agent_id, state)
        - Harm timeline (time, total harm cells)
        - Max hazard timeline (time, max hazard concentration)
        - Step count (total simulation steps)
        """
        self.start_time = time.time()
        self.agent_trajectories = defaultdict(list)
        self.hazard_events = []
        self.congestion_events = []
        self.state_transitions = defaultdict(list)
        self.harm_timeline = []
        self.max_hazard_timeline = []
        self.step_count = 0
        
    def record_step(self, env, info):
        """Record state for current step."""
        self.step_count += 1
        
        # Track agent positions
        for agent in env.agents:
            pos = agent.pos if not hasattr(agent, 'current_node') else env._node_to_cell(agent.current_node)
            self.agent_trajectories[agent.id].append(pos)
        
        # Track harm
        self.harm_timeline.append(info['harm_grid'].sum())
        self.max_hazard_timeline.append(env.hazard_field.concentration.max())
        
        # Track state transitions
        for agent in env.agents:
            if len(self.state_transitions[agent.id]) == 0 or \
               self.state_transitions[agent.id][-1] != agent.state:
                self.state_transitions[agent.id].append((self.step_count, agent.state))
        
        # Track hazard events
        for event in env.events_log:
            if event.get('timestep') == self.step_count:
                event_type = event.get('event_type')
                if 'hazard' in event_type or 'deploy' in event_type:
                    self.hazard_events.append(event)
                elif event_type == 'node_congestion':
                    self.congestion_events.append(event)
    
    def print_summary(self, env):
        """Print detailed terminal summary."""
        duration = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("SIMULATION SUMMARY")
        print("="*80)
        
        # Basic stats
        print(f"\n📊 Runtime Statistics:")
        print(f"  Duration: {duration:.1f}s ({self.step_count} steps, {self.step_count/duration:.1f} steps/s)")
        print(f"  Environment: {'OSM Graph Mode' if env.osm_graph else 'Grid Mode'}")
        if env.osm_graph:
            print(f"  Graph: {len(env.osm_graph.nodes)} nodes, {len(env.osm_graph.edges)} edges")
        
        # Agent outcomes
        print(f"\n👥 Agent Outcomes:")
        states = defaultdict(int)
        for agent in env.protesters:
            states[agent.state.value if hasattr(agent.state, 'value') else str(agent.state)] += 1
        
        total = len(env.protesters)
        for state, count in sorted(states.items()):
            pct = 100 * count / total
            print(f"  {state.upper()}: {count}/{total} ({pct:.1f}%)")
        
        # Movement statistics
        print(f"\n🚶 Movement Statistics:")
        total_distance = 0
        max_distance = 0
        max_agent = None
        
        for agent_id, positions in self.agent_trajectories.items():
            if len(positions) < 2:
                continue
            dist = sum(np.hypot(positions[i+1][0] - positions[i][0],
                               positions[i+1][1] - positions[i][1])
                      for i in range(len(positions) - 1))
            total_distance += dist
            if dist > max_distance:
                max_distance = dist
                max_agent = agent_id
        
        avg_distance = total_distance / len(self.agent_trajectories) if self.agent_trajectories else 0
        print(f"  Total distance traveled: {total_distance:.1f} cells")
        print(f"  Average per agent: {avg_distance:.1f} cells")
        print(f"  Longest journey: Agent {max_agent} ({max_distance:.1f} cells)")
        
        # Hazard exposure
        print(f"\n☁️ Hazard Exposure:")
        print(f"  Peak concentration: {max(self.max_hazard_timeline):.1f}")
        print(f"  Total harm incidents: {sum(self.harm_timeline)}")
        print(f"  Hazard deployment events: {len(self.hazard_events)}")
        
        if self.hazard_events:
            print(f"  Notable deployments:")
            for event in self.hazard_events[:5]:
                step = event.get('timestep', '?')
                etype = event.get('event_type', 'unknown')
                pos = event.get('position', event.get('cell', '?'))
                print(f"    Step {step}: {etype} at {pos}")
        
        # Congestion
        if env.osm_graph and self.congestion_events:
            print(f"\n🚧 Congestion Events:")
            print(f"  Total queue formations: {len(self.congestion_events)}")
            
            # Group by location
            location_counts = defaultdict(int)
            for event in self.congestion_events:
                loc = event.get('street_name', event.get('node_id', 'unknown'))
                location_counts[loc] += 1
            
            print(f"  Worst bottlenecks:")
            for loc, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {loc}: {count} events")
        
        # State transitions
        print(f"\n🔄 Agent State Transitions:")
        transition_counts = defaultdict(int)
        for agent_id, transitions in self.state_transitions.items():
            transition_counts[len(transitions)] += 1
        
        for n_transitions, count in sorted(transition_counts.items()):
            print(f"  {count} agents had {n_transitions} state change(s)")
        
        # Critical incidents
        print(f"\n⚠️ Critical Incidents:")
        incapacitated = [a for a in env.protesters if a.state == AgentState.INCAPACITATED]
        if incapacitated:
            print(f"  {len(incapacitated)} agents incapacitated:")
            for agent in incapacitated[:5]:
                harm = agent.cumulative_harm
                pos = agent.pos
                print(f"    Agent {agent.id} at {pos} (cumulative harm: {harm:.2f})")
        else:
            print(f"  No agents incapacitated ✓")
        
        print("\n" + "="*80 + "\n")


def run_live_visualization(config_path='configs/default_scenario.yaml', 
                          max_steps=200, 
                          seed=42,
                          trail_length=10,
                          show_paths=False):
    """
    Run live visualization with enhanced visuals and tracking.

    Args:
        config_path: path to YAML config for the ProtestEnv
        max_steps: maximum simulation steps to run
        seed: RNG seed for reproducible simulation
        trail_length: number of previous positions to keep as trail per agent
        show_paths: attempt to plot agent planned paths (best-effort)
    """
    # --- Load config and environment ---
    cfg = load_config(config_path)
    env = ProtestEnv(cfg)
    env.rng = np.random.default_rng(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        # Some env.reset implementations might accept different args
        env.reset()

    tracker = SimulationTracker()

    # Figure setup
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.tight_layout()

    is_graph_mode = getattr(env, "osm_graph", None) is not None and getattr(env, "cell_to_node", None) is not None

    # Elements we'll update
    hazard_img = None
    node_scatter = None
    edge_lines = None
    agent_scatter = None
    trail_lines = {}  # agent_id -> LineCollection
    text_overlay = {}

    # Maintain trails using deque for each agent
    trails = {agent.id: deque(maxlen=trail_length) for agent in env.agents}

    # Prepare base plot depending on mode
    if is_graph_mode:
        ax.set_title("Live Visualization — OSM Graph Mode")
        # draw edges as background lines
        edges = []
        node_positions = {}
        for n, data in env.osm_graph.nodes(data=True):
            node_positions[n] = (data.get("x", 0), data.get("y", 0))
        for u, v, data in env.osm_graph.edges(data=True):
            p1 = node_positions[u]
            p2 = node_positions[v]
            edges.append((p1, p2))
        if edges:
            edge_lines = LineCollection(edges, linewidths=0.5, alpha=0.5)
            ax.add_collection(edge_lines)

        # Node scatter: colormap by capacity utilization (updated later)
        node_xy = np.array([node_positions[n] for n in env.osm_graph.nodes])
        node_ids = list(env.osm_graph.nodes)
        node_scatter = ax.scatter(node_xy[:, 0], node_xy[:, 1], s=20, cmap='viridis')

        # set view limits with padding
        xmin, ymin = node_xy.min(axis=0) - 5
        xmax, ymax = node_xy.max(axis=0) + 5
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_title("Live Visualization — Grid Mode")
        # Show hazard field as background image if available
        try:
            hazard_arr = env.hazard_field.concentration
            hazard_img = ax.imshow(hazard_arr, origin='lower', interpolation='nearest', alpha=0.7)
        except Exception:
            hazard_img = None
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(-0.5, env.height - 0.5)
        ax.set_aspect('equal')

    # Agent scatter initial
    def get_agent_xy(agent):
        if is_graph_mode and hasattr(agent, "current_node"):
            # convert node -> cell coordinates for display if _node_to_cell exists
            try:
                return env._node_to_cell(agent.current_node)
            except Exception:
                ndata = env.osm_graph.nodes.get(agent.current_node, {})
                return (ndata.get("x", 0), ndata.get("y", 0))
        else:
            return agent.pos

    agent_xy = np.array([get_agent_xy(a) for a in env.agents])
    agent_colors = ['blue' if a.agent_type == 'protester' else 'red' for a in env.agents]
    agent_scatter = ax.scatter(agent_xy[:, 0], agent_xy[:, 1], c=agent_colors, s=40, edgecolors='k', zorder=5)

    # Create legend
    protesters_patch = mpatches.Patch(color='blue', label='Protester')
    police_patch = mpatches.Patch(color='red', label='Police')
    ax.legend(handles=[protesters_patch, police_patch], loc='upper right')

    # Overlay text boxes for stats
    stats_text = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    steps = {'count': 0}

    # Update function for animation
    def update(frame):
        if steps['count'] >= max_steps:
            # stop animation
            anim.event_source.stop()
            return

        try:
            obs, reward, terminated, truncated, info = env.step()
        except TypeError:
            # fallback if step expects args
            obs, reward, terminated, truncated, info = env.step(None)
        except Exception as e:
            print(f"[ERROR] env.step failed at step {steps['count']}: {e}")
            anim.event_source.stop()
            raise

        steps['count'] += 1

        # record stats
        tracker.record_step(env, info)

        # Update hazard visualization
        if not is_graph_mode:
            try:
                hazard_arr = env.hazard_field.concentration
                if hazard_img is None:
                    hazard_img = ax.imshow(hazard_arr, origin='lower', interpolation='nearest', alpha=0.7)
                else:
                    hazard_img.set_data(hazard_arr)
                # auto rescale color range
                hazard_img.set_clim(vmin=0, vmax=max(1.0, hazard_arr.max()))
            except Exception:
                pass
        else:
            # update node scatter color by utilization (if node_occupancy exists)
            try:
                caps = np.array([env.osm_graph.nodes[n].get('capacity', 6) for n in node_ids], dtype=float)
                occ = np.array([env.node_occupancy.get(n, 0) for n in node_ids], dtype=float)
                utilization = np.clip(occ / np.maximum(caps, 1.0), 0.0, 1.0)
                # Map utilization to color via scatter's array
                node_scatter.set_offsets(node_xy)
                node_scatter.set_array(utilization)
                node_scatter.set_clim(0.0, 1.0)
            except Exception:
                pass

        # Update agents and trails
        xs, ys = [], []
        for agent in env.agents:
            x, y = get_agent_xy(agent)
            trails.setdefault(agent.id, deque(maxlen=trail_length)).append((x, y))

            xs.append(x)
            ys.append(y)

            # draw/update trail
            pts = list(trails[agent.id])
            if len(pts) >= 2:
                segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
                if agent.id not in trail_lines:
                    lc = LineCollection(segs, linewidths=1.5, alpha=0.7, zorder=4)
                    ax.add_collection(lc)
                    trail_lines[agent.id] = lc
                else:
                    trail_lines[agent.id].set_segments(segs)
            else:
                if agent.id in trail_lines:
                    trail_lines[agent.id].set_segments([])

            # Optionally draw planned paths if requested and agent exposes one
            if show_paths and hasattr(agent, 'planned_path') and isinstance(agent.planned_path, (list, tuple)):
                try:
                    path_pts = agent.planned_path
                    path_segs = [(path_pts[i], path_pts[i+1]) for i in range(len(path_pts)-1)]
                    # lightweight drawing: reuse a temporary LineCollection per agent (not persisted)
                    lc_path = LineCollection(path_segs, linewidths=0.8, linestyles='dashed', alpha=0.6, zorder=3)
                    ax.add_collection(lc_path)
                except Exception:
                    pass

        # update scatter positions
        if env.agents:
            agent_scatter.set_offsets(np.column_stack((xs, ys)))
        else:
            agent_scatter.set_offsets(np.empty((0, 2)))

        # Update overlay text
        s = []
        s.append(f"Step: {steps['count']}/{max_steps}")
        s.append(f"Agents: {len(env.agents)} (Protesters: {len(getattr(env, 'protesters', []))}, Police: {len(getattr(env, 'police_agents', []))})")
        s.append(f"Events (recent): {len(info.get('events', []))}")
        s.append(f"Congestion events: {info.get('congestion_events', 0)}")
        s.append(f"Avg queue: {info.get('avg_queue_length', 0):.2f}")
        s.append(f"Peak hazard: {np.nanmax(getattr(env.hazard_field, 'concentration', np.array([0]))):.2f}")
        stats_text.set_text("\n".join(s))

        # Termination check
        if terminated or truncated or steps['count'] >= max_steps:
            anim.event_source.stop()
            # Final record (step may have already been recorded)
            tracker.print_summary(env)
            # show final frame for a moment then close
            plt.pause(0.5)
            try:
                plt.close(fig)
            except Exception:
                pass

    # Run animation
    anim = FuncAnimation(fig, update, interval=100, blit=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        print("[INFO] Visualization interrupted by user.")
        anim.event_source.stop()
        tracker.print_summary(env)
        plt.close(fig)


if __name__ == "__main__":
    run_live_visualization(
        config_path='configs/default_scenario.yaml',
        max_steps=200,
        seed=42,
        trail_length=10,
        show_paths=False
    )