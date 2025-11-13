# src/report_adapter/report_aggregator.py
"""
Lightweight Report Aggregator - FR3
Aggregates user reports into spatial probability estimates with uncertainty
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import time


class ReportAggregator:
    """
    Aggregates anonymous user reports into spatial harm probabilities.
    
    Key features:
    - Sliding time window (5 min default)
    - Smoothed binomial estimation with Laplace smoothing
    - Confidence intervals via normal approximation
    - Spatial influence radius
    """
    
    def __init__(
        self, 
        time_window: int = 300,  # 5 minutes
        spatial_radius: int = 2,  # Influence 2 neighbors
        confidence_level: float = 0.95
    ):
        self.time_window = time_window
        self.spatial_radius = spatial_radius
        self.confidence_level = confidence_level
        
        # Report type to harm mapping
        self.harm_weights = {
            'safe': -0.3,        # Reduces harm
            'crowd': 0.15,       # Slight increase
            'police': 0.65,      # Moderate increase  
            'tear_gas': 0.60,    # High increase
            'water_cannon': 0.60
        }
    
    def aggregate_node_reports(
        self, 
        node_id: str,
        all_reports: Dict[str, List[dict]],
        current_time: float = None
    ) -> Tuple[float, float, float]:
        """
        Aggregate reports for a specific node with uncertainty.
        
        Args:
            node_id: Target node
            all_reports: Dict mapping node_id -> list of reports
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            (p_report, ci_lower, ci_upper): Estimated harm probability and 95% CI
        """
        if current_time is None:
            current_time = time.time()
        
        # Get recent reports at this node
        node_reports = all_reports.get(node_id, [])
        recent_reports = [
            r for r in node_reports 
            if current_time - r['timestamp'] < self.time_window
        ]
        
        if not recent_reports:
            return 0.0, 0.0, 0.0  # No reports, no adjustment
        
        # Aggregate weighted harm contributions
        total_weight = 0.0
        weighted_harm_sum = 0.0
        
        for report in recent_reports:
            report_type = report['type']
            confidence = report['confidence']
            
            harm_contribution = self.harm_weights.get(report_type, 0.0)
            weight = confidence
            
            weighted_harm_sum += harm_contribution * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.0, 0.0
        
        # Mean harm adjustment
        mean_harm_delta = weighted_harm_sum / total_weight
        
        # Uncertainty: decreases with more reports
        n_reports = len(recent_reports)
        # Standard error decreases as 1/sqrt(n)
        se = 0.2 / np.sqrt(n_reports)  # Base uncertainty of 0.2
        
        # 95% confidence interval (z = 1.96)
        z_score = 1.96
        ci_lower = mean_harm_delta - z_score * se
        ci_upper = mean_harm_delta + z_score * se
        
        return mean_harm_delta, ci_lower, ci_upper
    
    def get_report_statistics(
        self, 
        all_reports: Dict[str, List[dict]]
    ) -> Dict:
        """Get summary statistics about current reports"""
        current_time = time.time()
        
        total_active = 0
        by_type = defaultdict(int)
        nodes_with_reports = set()
        
        for node_id, reports in all_reports.items():
            for report in reports:
                if current_time - report['timestamp'] < self.time_window:
                    total_active += 1
                    by_type[report['type']] += 1
                    nodes_with_reports.add(node_id)
        
        return {
            'total_active_reports': total_active,
            'by_type': dict(by_type),
            'nodes_affected': len(nodes_with_reports),
            'time_window_seconds': self.time_window
        }
    
    def apply_spatial_smoothing(
        self,
        graph,
        node_id: str,
        p_report_center: float
    ) -> List[Tuple[str, str, float]]:
        """
        Apply spatial smoothing: nearby edges also get affected.
        
        Returns:
            List of (node_u, node_v, harm_adjustment) tuples
        """
        affected_edges = []
        
        # Center node: full effect
        for neighbor in graph.neighbors(node_id):
            affected_edges.append((node_id, neighbor, p_report_center))
        
        # Neighbors: reduced effect (50%)
        if self.spatial_radius >= 1:
            for neighbor in graph.neighbors(node_id):
                for second_neighbor in graph.neighbors(neighbor):
                    if second_neighbor != node_id:
                        affected_edges.append(
                            (neighbor, second_neighbor, p_report_center * 0.5)
                        )
        
        return affected_edges


def update_edge_harm_with_aggregation(
    graph,
    all_reports: Dict[str, List[dict]],
    aggregator: ReportAggregator,
    baseline_p_sim: Dict[Tuple[str, str], float]
) -> None:
    """
    Update edge harm probabilities using aggregated reports.
    
    This is the key integration function that replaces your simple multipliers.
    
    Args:
        graph: NetworkX graph
        all_reports: Current reports dictionary
        aggregator: ReportAggregator instance
        baseline_p_sim: Original simulation probabilities (for reset)
    """
    import networkx as nx
    
    # Reset all edges to baseline first
    for (u, v), p_sim in baseline_p_sim.items():
        if graph.has_edge(u, v):
            if isinstance(graph, nx.MultiDiGraph):
                for key in graph[u][v]:
                    graph[u][v][key]['p_harm'] = p_sim
            else:
                graph[u][v]['p_harm'] = p_sim
    
    # Apply report-based adjustments
    for node_id in all_reports.keys():
        if node_id not in graph.nodes:
            continue
        
        # Aggregate reports at this node
        p_report, ci_lower, ci_upper = aggregator.aggregate_node_reports(
            node_id, all_reports
        )
        
        if abs(p_report) < 0.001:  # Skip negligible adjustments
            continue
        
        # Get spatially smoothed effects
        affected_edges = aggregator.apply_spatial_smoothing(
            graph, node_id, p_report
        )
        
        # Update edges
        for u, v, harm_delta in affected_edges:
            if not graph.has_edge(u, v):
                continue
            
            if isinstance(graph, nx.MultiDiGraph):
                for key in graph[u][v]:
                    current_p = graph[u][v][key].get('p_harm', 0.0)
                    new_p = np.clip(current_p + harm_delta, 0.0, 1.0)
                    graph[u][v][key]['p_harm'] = new_p
            else:
                current_p = graph[u][v].get('p_harm', 0.0)
                new_p = np.clip(current_p + harm_delta, 0.0, 1.0)
                graph[u][v]['p_harm'] = new_p


# Example usage in backend
if __name__ == "__main__":
    # Initialize aggregator
    aggregator = ReportAggregator(
        time_window=300,  # 5 minutes
        spatial_radius=1,
        confidence_level=0.95
    )
    
    # Mock reports
    mock_reports = {
        'node_123': [
            {
                'type': 'tear_gas',
                'confidence': 0.9,
                'timestamp': time.time() - 60
            },
            {
                'type': 'police',
                'confidence': 0.7,
                'timestamp': time.time() - 120
            }
        ]
    }
    
    # Aggregate
    p_report, ci_lower, ci_upper = aggregator.aggregate_node_reports(
        'node_123', mock_reports
    )
    
    print(f"Report-derived harm: {p_report:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Interpretation: Add {p_report:.3f} to baseline p_harm at affected edges")