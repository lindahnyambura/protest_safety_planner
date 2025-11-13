# src/fusion/simple_fusion.py
"""
Lightweight Risk Fusion - FR4 (Lite)
Combines simulation priors with report-based updates using weighted averaging
"""

import numpy as np
from typing import Dict, Tuple
import time


class SimpleFusionEngine:
    """
    Simplified Bayesian-inspired fusion engine.
    
    Uses confidence-weighted averaging instead of full log-odds Bayesian fusion.
    Computationally cheap, statistically reasonable, easy to explain.
    """
    
    def __init__(
        self,
        simulation_weight: float = 0.7,
        report_weight: float = 0.3,
        use_uncertainty_weighting: bool = True
    ):
        """
        Args:
            simulation_weight: Base weight for simulation priors
            report_weight: Base weight for report evidence
            use_uncertainty_weighting: Adjust weights by confidence intervals
        """
        self.sim_weight = simulation_weight
        self.report_weight = report_weight
        self.use_uncertainty = use_uncertainty_weighting
    
    def fuse_edge_probabilities(
        self,
        p_sim: float,
        p_report_delta: float,
        ci_report: Tuple[float, float] = None,
        num_reports: int = 1
    ) -> float:
        """
        Fuse simulation prior with report-based adjustment.
        
        Args:
            p_sim: Baseline simulation probability
            p_report_delta: Report-based adjustment (can be negative for 'safe' reports)
            ci_report: Confidence interval [lower, upper] for report
            num_reports: Number of reports contributing to this estimate
            
        Returns:
            p_fused: Fused probability
        """
        
        # Adaptive weighting based on number of reports
        # More reports → trust reports more
        report_confidence = min(1.0, num_reports / 10.0)  # Cap at 10 reports
        
        adjusted_report_weight = self.report_weight + (0.3 * report_confidence)
        adjusted_sim_weight = 1.0 - adjusted_report_weight
        
        # Uncertainty weighting (narrower CI → more weight)
        if self.use_uncertainty and ci_report is not None:
            ci_width = ci_report[1] - ci_report[0]
            uncertainty_factor = 1.0 / (1.0 + ci_width)  # Narrow CI → high factor
            adjusted_report_weight *= uncertainty_factor
        
        # Normalize weights
        total_weight = adjusted_sim_weight + adjusted_report_weight
        w_sim = adjusted_sim_weight / total_weight
        w_report = adjusted_report_weight / total_weight
        
        # Weighted combination
        # Note: p_report_delta is an adjustment, not a probability
        p_fused = p_sim + (w_report * p_report_delta)
        
        # Clip to valid probability range
        p_fused = np.clip(p_fused, 0.0, 1.0)
        
        return p_fused
    
    def conservative_fallback(
        self,
        p_sim: float,
        p_report_adjusted: float
    ) -> float:
        """
        Conservative fusion: take maximum (safer routing).
        Use when you want to be extra cautious.
        """
        return max(p_sim, p_report_adjusted)
    
    def get_fusion_metadata(
        self,
        p_sim: float,
        p_fused: float,
        num_reports: int
    ) -> Dict:
        """Return metadata about fusion decision"""
        
        delta = p_fused - p_sim
        
        return {
            'p_sim_baseline': round(p_sim, 4),
            'p_fused': round(p_fused, 4),
            'delta': round(delta, 4),
            'num_reports_used': num_reports,
            'interpretation': (
                'increased_risk' if delta > 0.01 else
                'decreased_risk' if delta < -0.01 else
                'no_significant_change'
            )
        }


def apply_fusion_to_graph(
    graph,
    all_reports: Dict[str, list],
    aggregator,  # ReportAggregator instance
    baseline_p_sim: Dict,
    fusion_engine: SimpleFusionEngine
) -> Dict:
    """
    Apply fusion across entire graph.
    
    This is an alternative to update_edge_harm_with_aggregation that
    uses explicit fusion instead of simple addition.
    
    Returns:
        Statistics about fusion operation
    """
    import networkx as nx
    
    edges_updated = 0
    total_delta = 0.0
    max_increase = 0.0
    max_decrease = 0.0
    
    # For each node with reports
    for node_id, reports in all_reports.items():
        if node_id not in graph.nodes:
            continue
        
        # Aggregate reports at this node
        p_report_delta, ci_lower, ci_upper = aggregator.aggregate_node_reports(
            node_id, all_reports
        )
        
        if abs(p_report_delta) < 0.001:
            continue
        
        num_reports = len([
            r for r in reports
            if time.time() - r['timestamp'] < aggregator.time_window
        ])
        
        # Update all edges from this node
        for neighbor in graph.neighbors(node_id):
            if not graph.has_edge(node_id, neighbor):
                continue
            
            # Get baseline probability
            if isinstance(graph, nx.MultiDiGraph):
                for key in graph[node_id][neighbor]:
                    edge_key = (node_id, neighbor, key)
                    p_sim_baseline = baseline_p_sim.get(edge_key, 0.0)
                    
                    # Fuse
                    p_fused = fusion_engine.fuse_edge_probabilities(
                        p_sim_baseline,
                        p_report_delta,
                        (ci_lower, ci_upper),
                        num_reports
                    )
                    
                    # Update graph
                    graph[node_id][neighbor][key]['p_harm'] = p_fused
                    
                    # Statistics
                    delta = p_fused - p_sim_baseline
                    edges_updated += 1
                    total_delta += abs(delta)
                    if delta > 0:
                        max_increase = max(max_increase, delta)
                    else:
                        max_decrease = min(max_decrease, delta)
            else:
                edge_key = (node_id, neighbor)
                p_sim_baseline = baseline_p_sim.get(edge_key, 0.0)
                
                p_fused = fusion_engine.fuse_edge_probabilities(
                    p_sim_baseline,
                    p_report_delta,
                    (ci_lower, ci_upper),
                    num_reports
                )
                
                graph[node_id][neighbor]['p_harm'] = p_fused
                
                delta = p_fused - p_sim_baseline
                edges_updated += 1
                total_delta += abs(delta)
                if delta > 0:
                    max_increase = max(max_increase, delta)
                else:
                    max_decrease = min(max_decrease, delta)
    
    return {
        'edges_updated': edges_updated,
        'mean_absolute_change': total_delta / max(edges_updated, 1),
        'max_risk_increase': max_increase,
        'max_risk_decrease': abs(max_decrease),
        'fusion_method': 'weighted_average_with_uncertainty'
    }


# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize fusion engine
    fusion = SimpleFusionEngine(
        simulation_weight=0.7,
        report_weight=0.3,
        use_uncertainty_weighting=True
    )
    
    # Example: Fuse a baseline probability with report evidence
    p_sim = 0.15  # Simulation says 15% risk
    p_report_delta = 0.25  # Reports suggest +25% increase
    ci_report = (0.20, 0.30)  # Confident estimate
    num_reports = 5
    
    p_fused = fusion.fuse_edge_probabilities(
        p_sim, p_report_delta, ci_report, num_reports
    )
    
    metadata = fusion.get_fusion_metadata(p_sim, p_fused, num_reports)
    
    print("Fusion Example:")
    print(f"  Simulation baseline: {p_sim:.3f}")
    print(f"  Report adjustment:   {p_report_delta:+.3f}")
    print(f"  Fused probability:   {p_fused:.3f}")
    print(f"  Interpretation:      {metadata['interpretation']}")
    print(f"  Net change:          {metadata['delta']:+.3f}")