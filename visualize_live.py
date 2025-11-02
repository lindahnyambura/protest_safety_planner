#!/usr/bin/env python3
"""
visualize_live.py — Production-Ready Protest Simulation Visualization
---------------------------------------------------------------------
Features:
✓ State-aware agent coloring (moving/fleeing/incapacitated)
✓ Enhanced hazard overlay with transparency
✓ Real-time statistics with intuitive metrics
✓ Detailed terminal summary with timing analysis
✓ Clean, publication-ready aesthetics
✓ OSM graph mode with road network visualization

Literature: Helbing et al. (2000) - Visualization best practices
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
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
logger.set_level(LogLevel.NORMAL)


# === COLOR SCHEME (Publication-Ready) ===
COLORS = {
    'protester_moving': '#2E86AB',      # Professional blue
    'protester_alert': '#F77F00',       # Alert orange
    'protester_panic': '#D62828',       # Panic red
    'protester_fleeing': '#8B0000',     # Deep red
    'protester_incap': '#6C757D',       # Grey
    'protester_safe': '#06A77D',        # Success green
    'police': '#1A1A1A',                # Police black
    'hazard_low': '#FFF3CD',            # Light yellow
    'hazard_high': '#8B0000',           # Deep red
    'road': '#E0E0E0',                  # Light grey
    'background': '#FFFFFF'             # White
}


class SimulationTracker:
    """Enhanced simulation tracker with timing and performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.agent_trajectories = defaultdict(list)
        self.hazard_events = []
        self.congestion_events = []
        self.state_transitions = defaultdict(list)
        self.harm_timeline = []
        self.max_hazard_timeline = []
        self.step_count = 0
        self.step_times = []
        
        # Behavioral state tracking
        self.behavioral_states = defaultdict(list)
        
    def record_step(self, env, info, step_time):
        """Record state with ENHANCED event categorization."""
        self.step_count += 1
        self.step_times.append(step_time) 
    
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
    
        # === ENHANCED EVENT TRACKING ===
        # NEW: Categorize events by type for better reporting
        if not hasattr(self, 'event_counts'):
            self.event_counts = {
                'gas_deployment': 0,
                'water_cannon': 0,
                'shooting': 0,
                'incapacitation': 0,
                'stun': 0
            }
    
        for event in env.events_log:
            if event.get('timestep') == self.step_count:
                event_type = event.get('event_type')
            
                # Categorize events
                if 'gas' in event_type or event_type == 'gas_deployment':
                    self.event_counts['gas_deployment'] += 1
                    self.hazard_events.append(event)
                elif event_type == 'water_cannon':
                    self.event_counts['water_cannon'] += 1
                    self.hazard_events.append(event)
                elif event_type == 'shooting':
                    self.event_counts['shooting'] += 1
                    self.hazard_events.append(event)
                elif event_type == 'incapacitation':
                    self.event_counts['incapacitation'] += 1
                elif 'stun' in event_type:
                    self.event_counts['stun'] += 1
            
                # Keep generic tracking
                if 'deploy' in event_type or 'gas' in event_type or 'water' in event_type or 'shoot' in event_type:
                    if event not in self.hazard_events:
                        self.hazard_events.append(event)
                elif event_type == 'node_congestion':
                    self.congestion_events.append(event)
    
    def print_summary(self, env):
        """Enhanced terminal summary with timing analysis."""
        duration = time.time() - self.start_time
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        
        print("\n" + "="*80)
        print(" SIMULATION SUMMARY")
        print("="*80)
        
        # === PERFORMANCE METRICS ===
        print(f"\n  Performance Metrics:")
        print(f"  Total duration: {duration:.1f}s ({self.step_count} steps)")
        print(f"  Average step time: {avg_step_time*1000:.1f}ms")
        print(f"  Throughput: {self.step_count/duration:.2f} steps/s")
        print(f"  Environment: {'OSM Graph Mode' if env.osm_graph else 'Grid Mode'}")
        if env.osm_graph:
            print(f"  Graph size: {len(env.osm_graph.nodes)} nodes, {len(env.osm_graph.edges)} edges")
            print(f"  Coverage: {100 * np.sum(~env.obstacle_mask) / env.obstacle_mask.size:.1f}% traversable")
        
        # === AGENT OUTCOMES ===
        print(f"\n👥 Agent Outcomes:")
        states = defaultdict(int)
        for agent in env.protesters:
            state = agent.state.value if hasattr(agent.state, 'value') else str(agent.state)
            states[state] += 1
        
        total = len(env.protesters)
        for state, count in sorted(states.items()):
            pct = 100 * count / total
            bar = '█' * int(pct / 2)  # Visual bar
            print(f"  {state.upper():12s}: {count:3d}/{total} ({pct:5.1f}%) {bar}")
        
        # === MOVEMENT STATISTICS ===
        print(f"\n Movement Statistics:")
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
        print(f"  Total distance: {total_distance:.1f} cells ({total_distance * 5:.0f}m)")
        print(f"  Average per agent: {avg_distance:.1f} cells ({avg_distance * 5:.0f}m)")
        print(f"  Longest journey: Agent {max_agent} ({max_distance:.1f} cells, {max_distance * 5:.0f}m)")
        
        # === BEHAVIORAL DYNAMICS ===
        if self.behavioral_states:
            print(f"\n Behavioral Dynamics:")
            state_counts = defaultdict(int)
            for agent_id, transitions in self.behavioral_states.items():
                for _, state in transitions:
                    state_counts[state] += 1
            
            print(f"  State distribution (total observations):")
            for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {state:10s}: {count:4d}")
        
        # === HAZARD EXPOSURE ===
        print(f"\n☁️  Hazard Exposure:")
        print(f"  Peak concentration: {max(self.max_hazard_timeline):.1f} mg/m³")
        print(f"  Total harm incidents: {sum(self.harm_timeline)}")
    
        # FIXED: Show all deployment types
        if hasattr(self, 'event_counts'):
            print(f"\n  📍 Deployment Breakdown:")
            print(f"    Gas canisters: {self.event_counts['gas_deployment']}")
            print(f"    Water cannon: {self.event_counts['water_cannon']}")
            print(f"    Shooting events: {self.event_counts['shooting']}")
            print(f"    Total deployments: {sum(self.event_counts[k] for k in ['gas_deployment', 'water_cannon', 'shooting'])}")
        
            if self.event_counts['water_cannon'] == 0:
                print(f"    ⚠️  WARNING: No water cannon events detected!")
            if self.event_counts['shooting'] == 0:
                print(f"    ⚠️  WARNING: No shooting events detected!")
        else:
            # Fallback to old counting
            print(f"  Deployment events: {len(self.hazard_events)}")
        
        # === CONGESTION ANALYSIS ===
        if env.osm_graph and self.congestion_events:
            print(f"\n Congestion Analysis:")
            print(f"  Total queue events: {len(self.congestion_events)}")
            
            # Group by location
            location_counts = defaultdict(int)
            for event in self.congestion_events:
                loc = event.get('street_name', event.get('node_id', 'unknown'))
                location_counts[loc] += 1
            
            print(f"\n   Worst Bottlenecks:")
            for i, (loc, count) in enumerate(sorted(location_counts.items(), 
                                                    key=lambda x: x[1], reverse=True)[:5]):
                print(f"    {i+1}. {loc:30s}: {count:3d} events")
        
        # === STATE TRANSITIONS ===
        print(f"\n Agent State Transitions:")
        transition_counts = defaultdict(int)
        for agent_id, transitions in self.state_transitions.items():
            transition_counts[len(transitions)] += 1
        
        for n_transitions, count in sorted(transition_counts.items()):
            print(f"  {count:3d} agents: {n_transitions} state change(s)")
        
        # === CRITICAL INCIDENTS ===
        print(f"\n Critical Incidents:")
        incapacitated = [a for a in env.protesters if a.state == AgentState.INCAPACITATED]
        if incapacitated:
            print(f"  {len(incapacitated)} agents incapacitated:")
            for i, agent in enumerate(incapacitated[:5]):
                harm = agent.cumulative_harm
                pos = agent.pos
                print(f"    {i+1}. Agent {agent.id:3d} at {pos} (cumulative harm: {harm:.2f})")
            if len(incapacitated) > 5:
                print(f"    ... and {len(incapacitated) - 5} more")
        else:
            print(f"   No agents incapacitated")
        
        # === SIMULATION QUALITY METRICS ===
        print(f"\n Simulation Quality:")
        
        # Exit convergence rate
        safe_agents = sum(1 for a in env.protesters if a.state == AgentState.SAFE)
        exit_rate = safe_agents / total if total > 0 else 0
        print(f"  Exit rate: {exit_rate*100:.1f}% ({safe_agents}/{total} safely exited)")
        
        # Harm realism (expect 15-30 incidents per 100 protesters over 200 steps)
        expected_harm = (total / 100) * (self.step_count / 200) * 22.5  # Midpoint
        harm_ratio = sum(self.harm_timeline) / max(expected_harm, 1)
        print(f"  Harm incidents: {sum(self.harm_timeline)} (expected: {expected_harm:.0f}, "
              f"ratio: {harm_ratio:.2f})")
        
        # Gas persistence
        if self.hazard_events:
            first_deployment = self.hazard_events[0].get('timestep', 0)
            last_high_hazard = max((i for i, h in enumerate(self.max_hazard_timeline) 
                                   if h > 10), default=0)
            persistence = last_high_hazard - first_deployment
            print(f"  Gas persistence: {persistence} steps ({persistence * 1}s)")
        
        print("\n" + "="*80 + "\n")


def get_agent_color(agent, env):
    """
    Get color for agent based on state and behavior.
    
    Color scheme:
    - MOVING (CALM): Blue
    - ALERT: Orange
    - PANIC: Red
    - FLEEING: Deep red
    - INCAPACITATED: Grey
    - SAFE: Green
    - POLICE: Black
    """
    if agent.agent_type == 'police':
        return COLORS['police']
    
    state = agent.state.value if hasattr(agent.state, 'value') else str(agent.state)
    
    if state == 'incapacitated':
        return COLORS['protester_incap']
    elif state == 'safe':
        return COLORS['protester_safe']
    
    # Use behavioral state if available
    if hasattr(agent, 'behavioral_state'):
        behavior = agent.behavioral_state
        if behavior == 'FLEEING':
            return COLORS['protester_fleeing']
        elif behavior == 'PANIC':
            return COLORS['protester_panic']
        elif behavior == 'ALERT':
            return COLORS['protester_alert']
    
    return COLORS['protester_moving']


def run_live_visualization(config_path='configs/default_scenario.yaml', 
                          max_steps=200, 
                          seed=42,
                          trail_length=15,
                          save_video=False):
    """
    Run enhanced live visualization.
    
    Args:
        config_path: Path to YAML config
        max_steps: Maximum simulation steps
        seed: RNG seed
        trail_length: Agent trail history
        save_video: Save animation as MP4 (requires ffmpeg)
    """
    # === INITIALIZATION ===
    print("\n" + "="*80)
    print(" INITIALIZING SIMULATION")
    print("="*80)
    
    cfg = load_config(config_path)
    env = ProtestEnv(cfg)
    env.rng = np.random.default_rng(seed)
    
    print(f"  Environment: {env.width}×{env.height} grid ({env.cell_size}m cells)")
    print(f"  Agents: {cfg['agents']['protesters']['count']} protesters, "
          f"{cfg['agents']['police']['count']} police")
    print(f"  Max steps: {max_steps}")
    print(f"  Seed: {seed}")
    
    try:
        obs, info = env.reset(seed=seed)
        print(f" Environment initialized successfully")
    except Exception as e:
        print(f" Initialization failed: {e}")
        raise

    tracker = SimulationTracker()

    # === FIGURE SETUP ===
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(14, 8), 
                                              gridspec_kw={'width_ratios': [4, 1]})
    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout()

    is_graph_mode = env.osm_graph is not None and env.cell_to_node is not None

    # Elements to update
    hazard_img = None
    node_scatter = None
    edge_lines = None
    agent_scatter = None
    trail_lines = {}
    
    # Agent trails
    trails = {agent.id: deque(maxlen=trail_length) for agent in env.agents}

    # === PREPARE BASE PLOT ===
    if is_graph_mode:
        ax_main.set_title("  Protest Simulation - OSM Graph Mode", 
                         fontsize=14, fontweight='bold', pad=15)
        
        # Draw road network
        edges = []
        node_positions = {}
        for n, data in env.osm_graph.nodes(data=True):
            node_positions[n] = (data.get("x", 0), data.get("y", 0))
        
        for u, v, data in env.osm_graph.edges(data=True):
            p1 = node_positions[u]
            p2 = node_positions[v]
            edges.append((p1, p2))
        
        if edges:
            edge_lines = LineCollection(edges, linewidths=1.0, 
                                       colors=COLORS['road'], alpha=0.6, zorder=1)
            ax_main.add_collection(edge_lines)

        # Node markers (capacity utilization)
        node_xy = np.array([node_positions[n] for n in env.osm_graph.nodes])
        node_ids = list(env.osm_graph.nodes)
        node_scatter = ax_main.scatter(node_xy[:, 0], node_xy[:, 1], 
                                      s=15, c='white', edgecolors='gray',
                                      linewidths=0.5, alpha=0.7, zorder=2)

        # Set limits with padding
        xmin, ymin = node_xy.min(axis=0) - 50
        xmax, ymax = node_xy.max(axis=0) + 50
        ax_main.set_xlim(xmin, xmax)
        ax_main.set_ylim(ymin, ymax)
    else:
        ax_main.set_title("  Protest Simulation - Grid Mode", 
                         fontsize=14, fontweight='bold', pad=15)
        
        # Hazard field background
        hazard_arr = env.hazard_field.concentration
        hazard_img = ax_main.imshow(hazard_arr, origin='lower', 
                                    interpolation='bilinear', alpha=0.6,
                                    cmap='YlOrRd', zorder=1)
        
        ax_main.set_xlim(-0.5, env.width - 0.5)
        ax_main.set_ylim(-0.5, env.height - 0.5)
        ax_main.set_aspect('equal')

    ax_main.set_facecolor('#F5F5F5')
    ax_main.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # === INITIAL AGENT SCATTER ===
    def get_agent_xy(agent):
        if is_graph_mode and hasattr(agent, "current_node"):
            try:
                return env._node_to_cell(agent.current_node)
            except:
                ndata = env.osm_graph.nodes.get(agent.current_node, {})
                return (ndata.get("x", 0), ndata.get("y", 0))
        return agent.pos

    agent_xy = np.array([get_agent_xy(a) for a in env.agents])
    agent_colors = [get_agent_color(a, env) for a in env.agents]
    agent_scatter = ax_main.scatter(agent_xy[:, 0], agent_xy[:, 1], 
                                   c=agent_colors, s=60, edgecolors='white',
                                   linewidths=1, zorder=5, alpha=0.9)

    # === LEGEND PANEL ===
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    
    # Title
    ax_legend.text(0.5, 0.95, " Real-Time Statistics", 
                  ha='center', va='top', fontsize=12, fontweight='bold',
                  transform=ax_legend.transAxes)
    
    stats_text = ax_legend.text(0.1, 0.88, '', transform=ax_legend.transAxes,
                               va='top', ha='left', fontsize=9, family='monospace',
                               bbox=dict(boxstyle='round,pad=0.5', 
                                        facecolor='white', alpha=0.9))

    # Color legend
    legend_y = 0.35
    legend_items = [
        ('Moving (Calm)', COLORS['protester_moving']),
        ('Alert', COLORS['protester_alert']),
        ('Panic', COLORS['protester_panic']),
        ('Fleeing', COLORS['protester_fleeing']),
        ('Incapacitated', COLORS['protester_incap']),
        ('Safe/Exited', COLORS['protester_safe']),
        ('Police', COLORS['police'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y - (i * 0.05)
        ax_legend.scatter([0.15], [y_pos], c=[color], s=80, 
                         edgecolors='white', linewidths=1,
                         transform=ax_legend.transAxes, zorder=5)
        ax_legend.text(0.25, y_pos, label, transform=ax_legend.transAxes,
                      va='center', ha='left', fontsize=8)

    steps = {'count': 0}
    last_step_time = [time.time()]

    # === UPDATE FUNCTION ===
    def update(frame):
        if steps['count'] >= max_steps:
            anim.event_source.stop()
            return

        step_start = time.time()
        
        try:
            obs, reward, terminated, truncated, info = env.step()
        except Exception as e:
            print(f"\n Step {steps['count']} failed: {e}")
            anim.event_source.stop()
            raise

        step_time = time.time() - step_start
        steps['count'] += 1

        # Record statistics
        tracker.record_step(env, info, step_time)

        # === UPDATE HAZARD VISUALIZATION ===
        if not is_graph_mode and hazard_img is not None:
            hazard_arr = env.hazard_field.concentration
            hazard_img.set_data(hazard_arr)
            hazard_img.set_clim(vmin=0, vmax=max(1.0, hazard_arr.max()))
        elif is_graph_mode and node_scatter is not None:
            # Update node colors by utilization
            try:
                caps = np.array([env.osm_graph.nodes[n].get('capacity', 6) 
                                for n in node_ids], dtype=float)
                occ = np.array([env.node_occupancy.get(str(n), 0) 
                               for n in node_ids], dtype=float)
                utilization = np.clip(occ / np.maximum(caps, 1.0), 0.0, 1.0)
                node_scatter.set_array(utilization)
                node_scatter.set_cmap('RdYlGn_r')
                node_scatter.set_clim(0.0, 1.0)
            except:
                pass

        # === UPDATE AGENTS AND TRAILS ===
        xs, ys, colors = [], [], []
        for agent in env.agents:
            x, y = get_agent_xy(agent)
            trails.setdefault(agent.id, deque(maxlen=trail_length)).append((x, y))
            
            xs.append(x)
            ys.append(y)
            colors.append(get_agent_color(agent, env))

            # Draw trail
            pts = list(trails[agent.id])
            if len(pts) >= 2:
                segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
                if agent.id not in trail_lines:
                    lc = LineCollection(segs, linewidths=1.5, alpha=0.4, 
                                       colors=[get_agent_color(agent, env)],
                                       zorder=4)
                    ax_main.add_collection(lc)
                    trail_lines[agent.id] = lc
                else:
                    trail_lines[agent.id].set_segments(segs)
                    trail_lines[agent.id].set_colors([get_agent_color(agent, env)])

        if env.agents:
            agent_scatter.set_offsets(np.column_stack((xs, ys)))
            agent_scatter.set_color(colors)

        # === UPDATE STATISTICS ===
        states = defaultdict(int)
        for agent in env.protesters:
            state = agent.state.value if hasattr(agent.state, 'value') else str(agent.state)
            states[state] += 1
        
        peak_hazard = env.hazard_field.concentration.max()
        
        stats_lines = [
            f"Step: {steps['count']}/{max_steps}",
            f"Time: {steps['count'] * env.delta_t:.0f}s",
            f"",
            f"Agents: {len(env.agents)}",
            f"  Moving: {states.get('moving', 0)}",
            f"  Safe: {states.get('safe', 0)}",
            f"  Incap: {states.get('incapacitated', 0)}",
            f"",
            f"Hazards:",
            f"  Peak: {peak_hazard:.1f} mg/m³",
            f"  Events: {len(tracker.hazard_events)}",
            f"  Harm: {sum(tracker.harm_timeline)}",
            f"",
            f"Performance:",
            f"  {1000*step_time:.1f}ms/step",
            f"  {1/step_time:.1f} FPS"
        ]
        
        if is_graph_mode:
            stats_lines.append(f"")
            stats_lines.append(f"Congestion: {len(tracker.congestion_events)}")
        
        stats_text.set_text("\n".join(stats_lines))

        # === TERMINATION CHECK ===
        if terminated or truncated or steps['count'] >= max_steps:
            print(f"\n Simulation complete at step {steps['count']}")
            anim.event_source.stop()
            tracker.print_summary(env)
            plt.pause(1.0)

    # === RUN ANIMATION ===
    print(f"\n Starting visualization...")
    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    
    if save_video:
        print(f" Saving video...")
        anim.save('simulation.mp4', writer='ffmpeg', fps=20, dpi=150)
        print(f" Video saved: simulation.mp4")
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print(f"\n  Visualization interrupted")
        anim.event_source.stop()
        tracker.print_summary(env)
        plt.close(fig)


def verify_hazard_systems(env):
    """
    Diagnostic function to verify water cannon and shooting are enabled.
    
    Call this after env.reset() to check configuration.
    """
    print("\n" + "="*70)
    print("🔍 HAZARD SYSTEM VERIFICATION")
    print("="*70)
    
    # Check water cannon
    wc_cfg = env.config.get('hazards', {}).get('water_cannon', {})
    print(f"\n💧 Water Cannon:")
    print(f"  Enabled: {wc_cfg.get('enabled', False)}")
    print(f"  Probability: {wc_cfg.get('prob', 0.0)}")
    print(f"  Cooldown: {wc_cfg.get('cooldown', 30)} steps")
    print(f"  Radius: {wc_cfg.get('radius', 6)} cells")
    print(f"  Stun prob: {wc_cfg.get('stun_prob', 0.1)}")
    
    if not wc_cfg.get('enabled', False):
        print(f"  ❌ DISABLED - Enable in config to use")
    else:
        expected_per_step = len(env.police_agents) * wc_cfg.get('prob', 0.0)
        print(f"  Expected: ~{expected_per_step:.2f} deployments/step")
        print(f"  Over 200 steps: ~{expected_per_step * 200:.0f} deployments")
    
    # Check shooting
    shoot_cfg = env.config.get('hazards', {}).get('shooting', {})
    print(f"\n🔫 Shooting:")
    print(f"  Enabled: {shoot_cfg.get('enabled', False)}")
    print(f"  Probability: {shoot_cfg.get('prob_per_step', 0.0)}")
    print(f"  Lethality: {shoot_cfg.get('lethality', 0.3)}")
    print(f"  Cooldown: {shoot_cfg.get('cooldown', 100)} steps")
    
    if not shoot_cfg.get('enabled', False):
        print(f"  ❌ DISABLED - Enable in config to use")
    else:
        expected_per_step = len(env.police_agents) * shoot_cfg.get('prob_per_step', 0.0)
        print(f"  Expected: ~{expected_per_step:.4f} events/step")
        print(f"  Over 200 steps: ~{expected_per_step * 200:.1f} events")
    
    # Check gas (for comparison)
    gas_cfg = env.config.get('hazards', {}).get('gas', {})
    print(f"\n☁️  Tear Gas:")
    print(f"  Deploy prob: {env.police_agents[0].deploy_prob if env.police_agents else 'N/A'}")
    print(f"  Expected: ~{len(env.police_agents) * env.police_agents[0].deploy_prob:.2f} deployments/step")
    
    print(f"\n" + "="*70 + "\n")

if __name__ == "__main__":
    run_live_visualization(
        config_path='configs/default_scenario.yaml',
        max_steps=200,
        seed=42,
        trail_length=15,
        save_video=False  # Set True to save MP4
    )