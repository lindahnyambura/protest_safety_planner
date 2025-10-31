#!/usr/bin/env python3
"""
test_rng_isolation.py - Verify agent RNG independence

Run this BEFORE full validation to check RNG fix.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.agent import Agent

def test_agent_rng_independence():
    """Test that agents get independent RNGs."""
    print("\n" + "="*60)
    print("TESTING AGENT RNG INDEPENDENCE")
    print("="*60)
    
    # Create parent RNG
    parent_rng = np.random.default_rng(42)
    
    # Create two agents with same parent RNG
    agent1 = Agent(
        agent_id=0,
        agent_type='protester',
        pos=(0, 0),
        goal=(10, 10),
        speed=1.0,
        risk_tolerance=0.3,
        rng=parent_rng,
        profile_name='average'
    )
    
    agent2 = Agent(
        agent_id=1,
        agent_type='protester',
        pos=(1, 1),
        goal=(11, 11),
        speed=1.0,
        risk_tolerance=0.3,
        rng=parent_rng,
        profile_name='average'
    )
    
    # Generate random numbers from each agent's RNG
    agent1_samples = [agent1.rng.random() for _ in range(10)]
    agent2_samples = [agent2.rng.random() for _ in range(10)]
    
    print(f"\nAgent 1 samples: {agent1_samples[:3]}")
    print(f"Agent 2 samples: {agent2_samples[:3]}")
    
    # Check if samples differ
    if agent1_samples == agent2_samples:
        print("\n❌ FAIL: Agents share same RNG!")
        print("   Fix: Ensure Agent.__init__() creates independent RNG")
        return False
    else:
        print("\n✅ PASS: Agents have independent RNGs")
        return True

def test_seed_isolation():
    """Test that different parent seeds produce different agent behavior."""
    print("\n" + "="*60)
    print("TESTING SEED ISOLATION")
    print("="*60)
    
    def create_agents_with_seed(seed):
        parent_rng = np.random.default_rng(seed)
        agents = []
        for i in range(3):
            agent = Agent(
                agent_id=i,
                agent_type='protester',
                pos=(i, i),
                goal=(10+i, 10+i),
                speed=1.0,
                risk_tolerance=0.3,
                rng=parent_rng,
                profile_name='average'
            )
            agents.append(agent)
        return agents
    
    # Create agents with different seeds
    agents_42 = create_agents_with_seed(42)
    agents_43 = create_agents_with_seed(43)
    
    # Sample from each agent's RNG
    samples_42 = [a.rng.random() for a in agents_42]
    samples_43 = [a.rng.random() for a in agents_43]
    
    print(f"\nSeed 42 samples: {samples_42}")
    print(f"Seed 43 samples: {samples_43}")
    
    # Check if samples differ
    if samples_42 == samples_43:
        print("\n❌ FAIL: Different seeds produce same results!")
        print("   Fix: Verify Agent.__init__() uses parent RNG to derive seed")
        return False
    else:
        print("\n✅ PASS: Different seeds produce different results")
        return True

if __name__ == "__main__":
    test1 = test_agent_rng_independence()
    test2 = test_seed_isolation()
    
    print("\n" + "="*60)
    print("RNG ISOLATION TEST SUMMARY")
    print("="*60)
    print(f"  Independence: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Seed Isolation: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 RNG ISOLATION WORKING - Ready for Monte Carlo")
        sys.exit(0)
    else:
        print("\n⚠️ RNG ISOLATION FAILED - Fix agent.py")
        sys.exit(1)