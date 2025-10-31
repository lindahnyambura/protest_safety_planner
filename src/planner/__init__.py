"""
planner - FR6: Risk-Aware Route Planner Module

Provides risk-aware routing functionality for protest navigation.

Usage:
    from planner import RiskAwareRoutePlanner, get_cost_function
    
    # Initialize planner
    planner = RiskAwareRoutePlanner(
        osm_graph=graph,
        p_sim=harm_grid,
        config={'cost_function': 'log_odds', 'lambda_risk': 10.0}
    )
    
    # Plan route
    route = planner.plan_route(start_node, goal_node, algorithm='astar')
    
    # Compare with baseline
    comparison = planner.compare_routes(start_node, goal_node)
"""

from .route_planner import RiskAwareRoutePlanner
from .cost_functions import (
    CostFunction,
    LogOddsCost,
    LinearCost,
    ExponentialCost,
    ThresholdCost,
    get_cost_function
)
from .route_optimizer import RouteOptimizer

__all__ = [
    'RiskAwareRoutePlanner',
    'RouteOptimizer',
    'CostFunction',
    'LogOddsCost',
    'LinearCost',
    'ExponentialCost',
    'ThresholdCost',
    'get_cost_function'
]

__version__ = '0.1.0'