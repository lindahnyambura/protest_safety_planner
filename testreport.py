#!/usr/bin/env python3
"""
Quick script to test the report endpoint
Run this to see exactly what error the backend is returning
"""

import requests
import json

# Test data
report_data = {
    "type": "crowd",
    "lat": -1.2875,
    "lng": 36.8225,
    "confidence": 0.8,
    "notes": "Test report",
    "timestamp": "2025-01-01T12:00:00Z"
}

print("Testing report endpoint...")
print(f"Sending: {json.dumps(report_data, indent=2)}")
print()

try:
    response = requests.post(
        "http://localhost:8000/report",
        json=report_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print()
    
    if response.status_code == 200:
        print("✅ SUCCESS!")
        print(json.dumps(response.json(), indent=2))
    else:
        print("❌ ERROR!")
        print(f"Response Text: {response.text}")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            pass
            
except requests.exceptions.ConnectionError:
    print("❌ CONNECTION ERROR!")
    print("Is the backend running? Start it with: python backend/main.py")
except Exception as e:
    print(f"❌ EXCEPTION: {e}")

print()
print("=" * 60)
print("If you see a 400 error, check:")
print("1. Backend logs for validation errors")
print("2. Data types match (lat/lng are floats, confidence 0-1)")
print("3. Report type is valid: safe, crowd, police, tear_gas, water_cannon")
print("=" * 60)