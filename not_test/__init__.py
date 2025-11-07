# tests/__init__.py
"""Test suite for protest_safety_planner"""

# ==========================================
# Quick test runner script
# tests/run_tests.py

"""
Quick test runner for Day 1 tests.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py --verbose
    python tests/run_tests.py --test test_deterministic_episode
"""

import sys
import pytest
from pathlib import Path

if __name__ == '__main__':
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Run tests
    args = sys.argv[1:] if len(sys.argv) > 1 else ['-v']
    pytest.main([str(project_root / 'tests')] + args)