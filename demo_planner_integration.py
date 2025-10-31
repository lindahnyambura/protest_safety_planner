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
from src.env.protest_env import ProtestEnvironment
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
    
    loader = RealNairobiLoader()
    
    env = ProtestEnvironment(
        grid_size=(config['grid_size'], config['grid_size']),
        real_nairobi_mode=True,
        real_nairobi_loader=loader
    )
    
    print(f"✓ Loaded Nairobi CBD environment")
    print(f"  Grid: {env.height}x{env.width}")
    print(f"  Nodes: {len(env.G.nodes)}")
    print(f"  Edges: {len(env.G.edges)}")
    print(f"  Bounds: ({loader.metadata['x_range'][0]:.0f}, {loader.metadata['y_range'][0]:.0f}) "
          f"to ({loader.metadata['x_range'][1]:.0f}, {loader.metadata['y_range'][1]:.0f})")
    
    return env, loader


def run_monte_carlo(env, config: dict, n_rollouts: int = 100):
    """Run Monte Carlo risk aggregation."""
    print("\n" + "=" * 60)
    print("STEP 2: Monte Carlo Simulation")
    print("=" * 60)
    
    aggregator = MonteCarloAggregator(
        env=env,
        scenario_config=config,
        output_dir='artifacts/planner_demo'
    )
    
    print(f"Running {n_rollouts} rollouts...")
    results = aggregator.run_parallel_rollouts(n=n_rollouts)
    
    print(f"✓ Monte Carlo complete")
    print(f"  Mean p(harm): {results['p_sim'].mean():.4f}")
    print(f"  Max p(harm): {results['p_sim'].max():.4f}")
    print(f"  Std dev: {results['sigma_sim'].mean():.4f}")
    
    return results


def extract_street_names(graph: nx.Graph) -> dict:
    """Extract street names from OSM graph."""
    street_names = {}
    
    for u, v, data in graph.edges(data=True):
        if 'name' in data:
            edge_key = f"{u}_{v}"
            street_names[edge_key] = data['name']
    
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
    
    # Find largest connected component
    components = list(nx.connected_components(graph))
    largest_component = max(components, key=len)
    valid_nodes = list(largest_component)
    
    print(f"\n[Finding interesting routes]")
    print(f"  Valid nodes: {len(valid_nodes)}")
    
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
        
        if dist > 200 and dist < 800:  # 200-800m range
            pairs.append((u, v, dist))
    
    # Sort by distance
    pairs.sort(key=lambda x: x[2])
    
    return pairs[:n_samples]


def initialize_planner(env, loader, results: dict, config: dict):
    """Initialize route planner with Monte Carlo results."""
    print("\n" + "=" * 60)
    print("STEP 3: Route Planner Initialization")
    print("=" * 60)
    
    # Extract street names
    street_names = extract_street_names(env.G)
    print(f"✓ Extracted {len(street_names)} street names")
    
    # Planner config
    planner_config = {
        'cost_function': config.get('cost_function', 'log_odds'),
        'lambda_distance': config.get('lambda_distance', 1.0),
        'lambda_risk': config.get('lambda_risk', 10.0),
        'x_min': loader.metadata['x_range'][0],
        'x_max': loader.metadata['x_range'][1],
        'y_min': loader.metadata['y_range'][0],
        'y_max': loader.metadata['y_range'][1],
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
            print(f"    vs Baseline:")
            print(f"      Distance +{comp['distance_increase_pct']:.1f}%")
            print(f"      Safety +{comp['safety_improvement']:.3f}")
            
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
        dist_increases = [c['comparison']['distance_increase_pct'] 
                         for c in all_comparisons]
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
    results = run_monte_carlo(env, config, args.n_rollouts)
    
    # Step 3: Initialize planner
    planner = initialize_planner(env, loader, results, config)
    
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