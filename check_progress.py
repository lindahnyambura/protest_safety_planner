#!/usr/bin/env python3
"""
check_progress.py - Monitor Monte Carlo progress

Run this in a separate terminal while Monte Carlo is running.
"""

import json
from pathlib import Path
import time

print("Monitoring Monte Carlo progress...\n")
print("Press Ctrl+C to stop monitoring\n")

rollouts_dir = Path("artifacts/rollouts/production_run")

try:
    while True:
        # Check if metadata file exists
        metadata_file = rollouts_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            n_rollouts = metadata.get('n_rollouts', 'N/A')
            runtime = metadata.get('runtime_seconds', 0)
            
            print(f"\r✓ Completed: {n_rollouts} rollouts in {runtime:.1f}s ({runtime/60:.1f} min)", end='')
        else:
            # Count numpy files to estimate progress
            npy_files = list(rollouts_dir.glob("*.npy"))
            
            if npy_files:
                print(f"\r⏳ Running... (found {len(npy_files)} output files)", end='')
            else:
                print(f"\r⏳ Waiting for Monte Carlo to start...", end='')
        
        time.sleep(5)  # Check every 5 seconds

except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
except Exception as e:
    print(f"\n\nError: {e}")