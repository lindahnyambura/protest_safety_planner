# test_fusion_integration.py

import requests
import time

BASE_URL = "http://localhost:8000"

def test_fusion_pipeline():
    print("=== Testing Fusion Pipeline ===\n")
    
    # 1. Check fusion stats (baseline)
    print("1. Initial state:")
    res = requests.get(f"{BASE_URL}/fusion/stats")
    stats = res.json()
    print(f"   Active reports: {stats['reports']['total_active']}")
    print(f"   Edges modified: {stats['graph_state']['edges_modified']}")
    
    # 2. Submit a test report
    print("\n2. Submitting tear gas report...")
    report = {
        "type": "tear_gas",
        "lat": -1.2882,
        "lng": 36.8201,
        "confidence": 0.9
    }
    res = requests.post(f"{BASE_URL}/report", json=report)
    result = res.json()
    print(f"   ✓ Report ID: {result['report_id']}")
    print(f"   Edges updated: {result['graph_update']['edges_updated']}")
    
    # 3. Check aggregated data
    print("\n3. Checking aggregated reports:")
    res = requests.get(f"{BASE_URL}/reports/aggregated")
    agg = res.json()
    print(f"   Found {len(agg['aggregated_reports'])} aggregated report locations")
    
    if len(agg['aggregated_reports']) > 0:
        first = agg['aggregated_reports'][0]
        print(f"   Example: p_report={first['p_report']}, "
              f"CI=[{first['ci_lower']}, {first['ci_upper']}]")
    
    # 4. Compute route near report
    print("\n4. Computing route (should avoid report area):")
    res = requests.get(
        f"{BASE_URL}/route",
        params={
            "start": "123456",  # Replace with actual node near KICC
            "goal": "789012",   # Replace with actual node away from report
            "lambda_risk": 10.0
        }
    )
    
    if res.status_code == 200:
        route = res.json()
        fusion = route.get('fusion_metadata', {})
        print(f"   ✓ Route computed")
        print(f"   Active reports: {fusion['reports']['total_active']}")
        print(f"   Edges influenced: {fusion['route_analysis']['edges_influenced_by_reports']}")
        print(f"   Interpretation: {fusion['interpretation']}")
    else:
        print(f"   ✗ Route failed: {res.json()}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_fusion_pipeline()