"""
demo_planner_integration.py - Complete Monte Carlo → Route Planner Pipeline

Demonstrates full integration:
1. Load Nairobi CBD environment
2. Run Monte Carlo simulation
3. Initialize route planner with p_sim
4. Compute safe routes
5. Compare with baseline
6. Export results

Usage:
    python demo_planner_integration.py (to run with p_sim.npy from default config)
    alternatively:
    python demo_planner_integration.py --n-rollouts 50 --start-node 123 --goal-node 456
"""

import numpy as np
import networkx as nx
from pathlib import Path
import json
import argparse
import yaml
from datetime import datetime

# Import your existing components
from src.env.real_nairobi_loader import RealNairobiLoader
from src.env.protest_env import ProtestEnv
from src.monte_carlo.aggregator import MonteCarloAggregator

# Import new planner module
from src.planner import RiskAwareRoutePlanner


def load_config(config_path: str = 'configs/default_scenario.yaml') -> dict:
    """Load simulation configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config: dict):
    """Initialize Nairobi CBD environment."""
    print("\n" + "=" * 60)
    print("STEP 1: Environment Setup")
    print("=" * 60)
    
    # --- 1. Load real map and graph ---
    from src.env.real_nairobi_loader import load_real_nairobi_cbd_map

    map_data = load_real_nairobi_cbd_map(config)
    if map_data is None or "graph" not in map_data:
        raise RuntimeError("Failed to load real Nairobi CBD map or graph")

    # --- 2. Initialize protest environment ---
    env = ProtestEnv(config=config)

    # Attach graph + metadata + obstacle mask
    env.G = map_data["graph"]
    env.metadata = map_data["metadata"]
    env.obstacle_mask = map_data["obstacle_mask"]
    env.is_real_osm = True

    print(f"✓ Loaded Nairobi CBD environment")
    print(f"  Grid: {env.height}x{env.width}")
    print(f"  Nodes: {len(env.G.nodes)}")
    print(f"  Edges: {len(env.G.edges)}")

    bounds = env.metadata["bounds"]
    print(f"  Bounds: ({bounds[0]:.0f}, {bounds[1]:.0f}) to ({bounds[2]:.0f}, {bounds[3]:.0f})")

    return env, map_data


def load_or_run_monte_carlo(env, config, n_rollouts=100, use_cached=True):
    """
    Either load precomputed Monte Carlo results or run fresh simulations.
    """
    from pathlib import Path
    import numpy as np

    run_dir = Path("artifacts/rollouts_test/test_run")  # temporarily running with 200x200 n=4 rollouts
    p_sim_path = run_dir / "p_sim.npy"

    if use_cached and p_sim_path.exists():
        print("\n" + "=" * 60)
        print("STEP 2: Load Cached Monte Carlo Results")
        print("=" * 60)
        p_sim = np.load(p_sim_path)
        print(f"✓ Loaded precomputed p_sim from {p_sim_path}")
        print(f"  Shape: {p_sim.shape}, Mean={p_sim.mean():.4f}, Max={p_sim.max():.4f}")
        return {"p_sim": p_sim}

    # Otherwise, run Monte Carlo fresh
    print("\nNo cached results found — running Monte Carlo simulation...")
    aggregator = MonteCarloAggregator(
        env_class=env.__class__,
        config=config,
        output_dir="artifacts/rollouts"
    )
    aggregator.n_rollouts = n_rollouts
    results = aggregator.run_monte_carlo(verbose=True)
    aggregator.save_results(results, run_id="production_run")
    return results


def extract_street_names(graph: nx.Graph) -> dict:
    """Extract street names from OSM graph."""
    street_names = {}
    
    for u, v, key, data in graph.edges(keys=True, data=True):
        name = data.get('name')
        if not name:
            continue
        
        # OSM can have multiple names (list or comma-separated)
        if isinstance(name, list):
            name = ", ".join(name)

        # Use all edge identifiers for uniqueness
        edge_key = f"{u}_{v}_{key}"
        street_names[edge_key] = name

    return street_names


def find_interesting_nodes(graph: nx.Graph, n_samples: int = 10):
    """
    Find interesting start/goal pairs for routing.
    
    Returns pairs that are:
    - In same connected component
    - Reasonably far apart (> 200m)
    """
    nodes = list(graph.nodes())
    node_positions = {n: (graph.nodes[n]['x'], graph.nodes[n]['y']) for n in nodes}
    
    # --- Handle directed graphs ---
    if graph.is_directed():
        undirected_graph = graph.to_undirected(as_view=True)
    else:
        undirected_graph = graph
    
    # Find largest connected component
    components = list(nx.connected_components(undirected_graph))
    largest_component = max(components, key=len)
    valid_nodes = list(largest_component)
    
    print(f"\n[Finding interesting routes]")
    print(f"  Valid nodes in largest component: {len(valid_nodes)}")
    
    pairs = []
    import random
    random.seed(42)
    
    for _ in range(n_samples * 10):  # Oversample
        if len(pairs) >= n_samples:
            break
        
        u, v = random.sample(valid_nodes, 2)
        
        # Check distance
        xu, yu = node_positions[u]
        xv, yv = node_positions[v]
        dist = np.sqrt((xu - xv)**2 + (yu - yv)**2)
        
        if 200 < dist < 800:  # 200–800m range
            pairs.append((u, v, dist))
    
    # Sort by distance
    pairs.sort(key=lambda x: x[2])
    
    print(f"  Selected {len(pairs)} valid pairs for routing.")
    return pairs[:n_samples]


def initialize_planner(env, map_data, results: dict, config: dict):
    """Initialize route planner with Monte Carlo results."""
    print("\n" + "=" * 60)
    print("STEP 3: Route Planner Initialization")
    print("=" * 60)
    
    # Extract street names
    street_names = extract_street_names(env.G)
    print(f"✓ Extracted {len(street_names)} street names")

    # Get metadata safely
    metadata = map_data.get("metadata", {})
    bounds = metadata.get("bounds", None)

    if bounds is not None:
        x_min, y_min, x_max, y_max = bounds
    else:
        # Fallback if using explicit ranges
        x_min, x_max = metadata.get("x_range", (0, 0))
        y_min, y_max = metadata.get("y_range", (0, 0))

    planner_config = {
        'cost_function': config.get('cost_function', 'log_odds'),
        'lambda_distance': config.get('lambda_distance', 1.0),
        'lambda_risk': config.get('lambda_risk', 10.0),
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'debug_mode': True,
        'cache_size': 10,
        'reroute_threshold': 0.1
    }

    planner = RiskAwareRoutePlanner(
        osm_graph=env.G,
        p_sim=results['p_sim'],
        config=planner_config,
        street_names=street_names
    )

    print(f"✓ Planner initialized")
    print(f"  Cost function: {planner_config['cost_function']}")
    print(f"  λ_distance: {planner_config['lambda_distance']}")
    print(f"  λ_risk: {planner_config['lambda_risk']}")
    print(f"  Bounds: x[{x_min:.0f}, {x_max:.0f}], y[{y_min:.0f}, {y_max:.0f}]")

    return planner


def compute_routes(planner: RiskAwareRoutePlanner, pairs: list, output_dir: Path):
    """Compute routes for multiple start/goal pairs."""
    print("\n" + "=" * 60)
    print("STEP 4: Route Computation")
    print("=" * 60)
    
    routes_dir = output_dir / 'routes'
    routes_dir.mkdir(parents=True, exist_ok=True)
    
    all_routes = []
    all_comparisons = []
    
    for i, (start, goal, dist) in enumerate(pairs):
        print(f"\n[Route {i+1}/{len(pairs)}] {start} → {goal} ({dist:.0f}m)")
        
        # Compute risk-aware route
        route = planner.plan_route(start, goal, algorithm='astar')
        
        if 'error' in route:
            print(f"  ✗ Failed: {route['error']}")
            continue
        
        print(f"  ✓ Path: {len(route['path'])} nodes")
        print(f"    Safety: {route['safety_score']:.3f}")
        print(f"    Distance: {route['metadata']['total_distance_m']:.0f}m")
        print(f"    Time: {route['metadata']['estimated_time_s']:.0f}s")
        print(f"    Turns: {route['metadata']['num_turns']}")
        
        # Compare with baseline
        comparison = planner.compare_routes(start, goal)
        
        if 'error' not in comparison:
            comp = comparison['comparison']

            # Safe printing
            dist_increase = comp.get('distance_increase_pct')
            safety_improvement = comp.get('safety_improvement')

            print("    vs Baseline:")
            if dist_increase is None:
                print("      Distance: N/A (baseline had zero distance)")
            else:
                print(f"      Distance +{dist_increase:.1f}%")

            if safety_improvement is None:
                print("      Safety improvement: N/A")
            else:
                print(f"      Safety +{safety_improvement:.3f}")

            all_comparisons.append(comparison)
        
        # Save route
        route_file = routes_dir / f"route_{i+1:03d}.json"
        planner.save_results(route, str(route_file))
        
        # Export debug visualization
        viz_file = routes_dir / f"route_{i+1:03d}_debug.geojson"
        planner.export_debug_visualization(route, str(viz_file))
        
        all_routes.append(route)
    
    return all_routes, all_comparisons


def generate_summary(all_routes: list, all_comparisons: list, output_dir: Path):
    """Generate summary statistics and report."""
    print("\n" + "=" * 60)
    print("STEP 5: Summary Statistics")
    print("=" * 60)
    
    if not all_routes:
        print("No routes computed")
        return
    
    # Aggregate metrics
    safety_scores = [r['safety_score'] for r in all_routes]
    distances = [r['metadata']['total_distance_m'] for r in all_routes]
    turns = [r['metadata']['num_turns'] for r in all_routes]
    
    print(f"\n[Route Statistics] ({len(all_routes)} routes)")
    print(f"  Safety score:")
    print(f"    Mean: {np.mean(safety_scores):.3f}")
    print(f"    Median: {np.median(safety_scores):.3f}")
    print(f"    Min: {np.min(safety_scores):.3f}")
    print(f"    Max: {np.max(safety_scores):.3f}")
    
    print(f"  Distance (m):")
    print(f"    Mean: {np.mean(distances):.0f}")
    print(f"    Median: {np.median(distances):.0f}")
    
    print(f"  Turns:")
    print(f"    Mean: {np.mean(turns):.1f}")
    print(f"    Median: {np.median(turns):.0f}")
    
    # Comparison statistics
    if all_comparisons:
        dist_increases = [
            c.get('comparison', {}).get('distance_increase_pct')
            for c in all_comparisons
            if c.get('comparison', {}).get('distance_increase_pct') is not None
        ]

        if dist_increases:
            print(f"    Mean: {np.mean(dist_increases):.1f}%")
            print(f"    Median: {np.median(dist_increases):.1f}%")
        else:
            print("    No valid distance comparisons (some baselines had zero distance).")

        safety_improvements = [c['comparison']['safety_improvement'] 
                              for c in all_comparisons]
        
        print(f"\n[Risk-Aware vs Baseline]")
        print(f"  Distance increase:")
        print(f"    Mean: {np.mean(dist_increases):.1f}%")
        print(f"    Median: {np.median(dist_increases):.1f}%")
        
        print(f"  Safety improvement:")
        print(f"    Mean: {np.mean(safety_improvements):.3f}")
        print(f"    Median: {np.median(safety_improvements):.3f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.utcnow().isoformat(),
        'n_routes': len(all_routes),
        'safety': {
            'mean': float(np.mean(safety_scores)),
            'median': float(np.median(safety_scores)),
            'std': float(np.std(safety_scores))
        },
        'distance_m': {
            'mean': float(np.mean(distances)),
            'median': float(np.median(distances)),
            'std': float(np.std(distances))
        },
        'comparisons': {
            'mean_distance_increase_pct': float(np.mean(dist_increases)) if dist_increases else None,
            'mean_safety_improvement': float(np.mean(safety_improvements)) if safety_improvements else None
        }
    }
    
    summary_file = output_dir / 'route_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_file}")


def main():
    """Run complete pipeline."""
    parser = argparse.ArgumentParser(description='Monte Carlo + Route Planner Demo')
    parser.add_argument('--n-rollouts', type=int, default=50,
                       help='Number of Monte Carlo rollouts')
    parser.add_argument('--n-routes', type=int, default=5,
                       help='Number of routes to compute')
    parser.add_argument('--start-node', type=str, default=None,
                       help='Specific start node (optional)')
    parser.add_argument('--goal-node', type=str, default=None,
                       help='Specific goal node (optional)')
    parser.add_argument('--config', type=str, default='configs/default_scenario.yaml',
                       help='Scenario configuration file')
    parser.add_argument('--cost-function', type=str, default='log_odds',
                       choices=['log_odds', 'linear', 'exponential', 'threshold'],
                       help='Cost function type')
    parser.add_argument('--lambda-risk', type=float, default=10.0,
                       help='Risk cost weight')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MONTE CARLO + ROUTE PLANNER INTEGRATION DEMO")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Rollouts: {args.n_rollouts}")
    print(f"Routes: {args.n_routes}")
    print(f"Cost function: {args.cost_function}")
    print(f"Lambda risk: {args.lambda_risk}")
    
    # Load config
    config = load_config(args.config)
    config['cost_function'] = args.cost_function
    config['lambda_risk'] = args.lambda_risk
    
    # Output directory
    output_dir = Path('artifacts/planner_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Setup environment
    env, loader = setup_environment(config)
    
    # Step 2: Run Monte Carlo
    results = load_or_run_monte_carlo(env, config, args.n_rollouts, use_cached=True)
    
    # Step 3: Initialize planner
    env, map_data = setup_environment(config)
    planner = initialize_planner(env, map_data, results, config)

    
    # Step 4: Select routes
    if args.start_node and args.goal_node:
        # User-specified route
        pairs = [(args.start_node, args.goal_node, 0.0)]
    else:
        # Auto-discover interesting routes
        pairs = find_interesting_nodes(env.G, args.n_routes)
        print(f"\n✓ Found {len(pairs)} route pairs")
    
    # Step 5: Compute routes
    all_routes, all_comparisons = compute_routes(planner, pairs, output_dir)
    
    # Step 6: Summary
    generate_summary(all_routes, all_comparisons, output_dir)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"  - Monte Carlo: {output_dir}/rollouts/")
    print(f"  - Routes: {output_dir}/routes/")
    print(f"  - Summary: {output_dir}/route_summary.json")
    print("\nNext steps:")
    print("  1. Visualize routes: Open .geojson files in QGIS")
    print("  2. Review summary: cat artifacts/planner_demo/route_summary.json")
    print("  3. Test re-routing: Run with updated p_sim")


if __name__ == '__main__':
    main()