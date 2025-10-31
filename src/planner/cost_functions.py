"""
cost_functions.py - Risk-aware cost models for route planning

Implements various cost functions that combine travel distance and harm risk.
"""

import numpy as np
from typing import Dict, Tuple


class CostFunction:
    """Base class for edge cost functions."""
    
    def __init__(self, config: Dict):
        """
        Initialize cost function.
        
        Args:
            config: Configuration dictionary with cost parameters
        """
        self.config = config
        self.lambda_distance = config.get('lambda_distance', 1.0)
        self.lambda_risk = config.get('lambda_risk', 10.0)
        
    def compute_cost(self, distance: float, p_harm: float) -> float:
        """
        Compute edge traversal cost.
        
        Args:
            distance: Physical distance (meters)
            p_harm: Probability of harm on this edge [0, 1]
            
        Returns:
            cost: Total cost (lower is better)
        """
        raise NotImplementedError


class LogOddsCost(CostFunction):
    """
    Log-odds based cost function (recommended).
    
    Cost(edge) = λ_d × distance + λ_r × (-log(1 - p_harm))
    
    Properties:
    - Logarithmic penalty for high-risk edges
    - Unbounded cost as p_harm → 1 (avoids certain death)
    - Configurable risk aversion via λ_r
    """
    
    def compute_cost(self, distance: float, p_harm: float) -> float:
        """
        Compute log-odds cost.
        
        Args:
            distance: Edge length (meters)
            p_harm: Harm probability [0, 1]
            
        Returns:
            cost: λ_d × d + λ_r × (-log(1 - p_harm))
        """
        # Numerical stability: clip p_harm away from 1.0
        p_harm_safe = np.clip(p_harm, 0.0, 0.9999)
        
        # Distance cost (linear)
        cost_distance = self.lambda_distance * distance
        
        # Risk cost (log-odds)
        if p_harm_safe < 1e-6:
            # Negligible risk → zero risk cost
            cost_risk = 0.0
        else:
            cost_risk = self.lambda_risk * (-np.log(1.0 - p_harm_safe))
        
        return cost_distance + cost_risk
    
    def compute_route_safety(self, p_harms: np.ndarray) -> float:
        """
        Compute overall route safety score.
        
        Assumes independence: P(safe) = ∏(1 - p_i) for all edges i
        
        Args:
            p_harms: Array of edge harm probabilities
            
        Returns:
            p_safe: Probability of completing route without harm [0, 1]
        """
        p_harms_safe = np.clip(p_harms, 0.0, 0.9999)
        p_safe_per_edge = 1.0 - p_harms_safe
        p_safe_route = np.prod(p_safe_per_edge)
        return float(p_safe_route)


class LinearCost(CostFunction):
    """
    Simple linear cost function (baseline).
    
    Cost(edge) = λ_d × distance + λ_r × p_harm
    
    Properties:
    - Simple and interpretable
    - Bounded cost
    - May not penalize high-risk edges enough
    """
    
    def compute_cost(self, distance: float, p_harm: float) -> float:
        """Compute linear cost."""
        cost_distance = self.lambda_distance * distance
        cost_risk = self.lambda_risk * p_harm
        return cost_distance + cost_risk
    
    def compute_route_safety(self, p_harms: np.ndarray) -> float:
        """Compute route safety (independence assumption)."""
        p_harms_safe = np.clip(p_harms, 0.0, 0.9999)
        p_safe_route = np.prod(1.0 - p_harms_safe)
        return float(p_safe_route)


class ExponentialCost(CostFunction):
    """
    Exponential risk penalty (high risk aversion).
    
    Cost(edge) = λ_d × distance + λ_r × exp(β × p_harm)
    
    Properties:
    - Strong aversion to any risk
    - May produce overly conservative routes
    - Useful for vulnerable populations
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.beta = config.get('beta', 5.0)  # Risk aversion parameter
    
    def compute_cost(self, distance: float, p_harm: float) -> float:
        """Compute exponential cost."""
        cost_distance = self.lambda_distance * distance
        cost_risk = self.lambda_risk * np.exp(self.beta * p_harm)
        return cost_distance + cost_risk
    
    def compute_route_safety(self, p_harms: np.ndarray) -> float:
        """Compute route safety."""
        p_harms_safe = np.clip(p_harms, 0.0, 0.9999)
        p_safe_route = np.prod(1.0 - p_harms_safe)
        return float(p_safe_route)


class ThresholdCost(CostFunction):
    """
    Threshold-based cost (binary risk classification).
    
    Cost(edge) = λ_d × distance + λ_r × I(p_harm > threshold)
    
    Properties:
    - Simple to explain to users
    - Binary "safe" vs "unsafe" classification
    - May miss nuanced risk differences
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.threshold = config.get('risk_threshold', 0.1)
    
    def compute_cost(self, distance: float, p_harm: float) -> float:
        """Compute threshold cost."""
        cost_distance = self.lambda_distance * distance
        cost_risk = self.lambda_risk if p_harm > self.threshold else 0.0
        return cost_distance + cost_risk
    
    def compute_route_safety(self, p_harms: np.ndarray) -> float:
        """Compute route safety."""
        p_harms_safe = np.clip(p_harms, 0.0, 0.9999)
        p_safe_route = np.prod(1.0 - p_harms_safe)
        return float(p_safe_route)


def get_cost_function(cost_type: str, config: Dict) -> CostFunction:
    """
    Factory function to create cost functions.
    
    Args:
        cost_type: One of ['log_odds', 'linear', 'exponential', 'threshold']
        config: Configuration dictionary
        
    Returns:
        CostFunction instance
    """
    cost_functions = {
        'log_odds': LogOddsCost,
        'linear': LinearCost,
        'exponential': ExponentialCost,
        'threshold': ThresholdCost
    }
    
    if cost_type not in cost_functions:
        raise ValueError(f"Unknown cost type: {cost_type}. "
                        f"Must be one of {list(cost_functions.keys())}")
    
    return cost_functions[cost_type](config)