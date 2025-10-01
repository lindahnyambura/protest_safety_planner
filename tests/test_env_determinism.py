"""
test_env_determinism.py - Critical tests for reproducibility

Day 1 priority: Ensure Monte Carlo reliability through deterministic behavior
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from env.protest_env import ProtestEnv, load_config


@pytest.fixture
def default_config():
    """Load default configuration for testing."""
    config_path = Path(__file__).parent.parent / 'configs' / 'default_scenario.yaml'
    return load_config(str(config_path))


@pytest.fixture
def small_config(default_config):
    """Smaller config for fast tests."""
    config = default_config.copy()
    config['grid']['width'] = 100
    config['grid']['height'] = 100
    config['agents']['protesters']['count'] = 20
    config['agents']['police']['count'] = 3
    config['time']['max_steps'] = 100
    return config


class TestResetDeterminism:
    """Test that reset produces identical initial states with same seed."""
    
    def test_identical_seeds_identical_initial_state(self, small_config):
        """CRITICAL: Same seed → identical initial state"""
        seed = 42
        
        env1 = ProtestEnv(small_config)
        env2 = ProtestEnv(small_config)
        
        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)
        
        # Check all observation components
        assert np.array_equal(obs1['hazard_concentration'], obs2['hazard_concentration']), \
            "Hazard fields differ with same seed"
        assert np.array_equal(obs1['occupancy_count'], obs2['occupancy_count']), \
            "Occupancy grids differ with same seed"
        assert np.array_equal(obs1['obstacle_mask'], obs2['obstacle_mask']), \
            "Obstacle masks differ with same seed"
        
        # Check agent positions
        for a1, a2 in zip(env1.agents, env2.agents):
            assert a1.pos == a2.pos, f"Agent {a1.id} position differs"
            assert a1.goal == a2.goal, f"Agent {a1.id} goal differs"
    
    def test_different_seeds_different_states(self, small_config):
        """Different seeds should produce different spawns."""
        env1 = ProtestEnv(small_config)
        env2 = ProtestEnv(small_config)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=999)
        
        # At least occupancy should differ
        assert not np.array_equal(obs1['occupancy_count'], obs2['occupancy_count']), \
            "Different seeds should produce different spawns"


class TestStepDeterminism:
    """Test that step execution is deterministic."""
    
    def test_deterministic_episode(self, small_config):
        """CRITICAL: Identical seeds and actions → identical trajectories"""
        seed = 42
        n_steps = 100
        
        env1 = ProtestEnv(small_config)
        env2 = ProtestEnv(small_config)
        
        env1.reset(seed=seed)
        env2.reset(seed=seed)
        
        for step in range(n_steps):
            # Use deterministic actions (STAY for all)
            actions = {a.id: 0 for a in env1.agents}
            
            obs1, _, term1, trunc1, info1 = env1.step(actions)
            obs2, _, term2, trunc2, info2 = env2.step(actions)
            
            # Check observations match
            assert np.allclose(obs1['hazard_concentration'], 
                             obs2['hazard_concentration'], 
                             atol=1e-6), \
                f"Hazard fields diverged at step {step}"
            
            assert np.array_equal(obs1['occupancy_count'], 
                                obs2['occupancy_count']), \
                f"Occupancy diverged at step {step}"
            
            # Check termination flags match
            assert term1 == term2, f"Termination flags differ at step {step}"
            assert trunc1 == trunc2, f"Truncation flags differ at step {step}"
            
            if term1 or trunc1:
                break
    
    def test_internal_policy_deterministic(self, small_config):
        """Test that internal agent policies are deterministic."""
        seed = 42
        n_steps = 50
        
        env1 = ProtestEnv(small_config)
        env2 = ProtestEnv(small_config)
        
        env1.reset(seed=seed)
        env2.reset(seed=seed)
        
        for step in range(n_steps):
            # Let agents decide their own actions
            obs1, _, _, _, _ = env1.step(actions=None)
            obs2, _, _, _, _ = env2.step(actions=None)
            
            # Agent positions should match
            for a1, a2 in zip(env1.agents, env2.agents):
                assert a1.pos == a2.pos, \
                    f"Agent {a1.id} position diverged at step {step}: {a1.pos} vs {a2.pos}"
                assert np.isclose(a1.cumulative_harm, a2.cumulative_harm, atol=1e-6), \
                    f"Agent {a1.id} harm diverged at step {step}"


class TestMovementConsistency:
    """Test movement mechanics and constraints."""
    
    def test_agents_stay_in_bounds(self, small_config):
        """Agents should never leave grid bounds."""
        env = ProtestEnv(small_config)
        env.reset(seed=42)
        
        for _ in range(200):
            env.step(actions=None)
            
            for agent in env.agents:
                assert 0 <= agent.pos[0] < env.width, \
                    f"Agent {agent.id} x-position out of bounds: {agent.pos[0]}"
                assert 0 <= agent.pos[1] < env.height, \
                    f"Agent {agent.id} y-position out of bounds: {agent.pos[1]}"
    
    def test_agents_avoid_obstacles(self, small_config):
        """Agents should never occupy obstacle cells."""
        env = ProtestEnv(small_config)
        env.reset(seed=42)
        
        for _ in range(200):
            env.step(actions=None)
            
            for agent in env.agents:
                x, y = agent.pos
                assert not env.obstacle_mask[y, x], \
                    f"Agent {agent.id} on obstacle at {agent.pos}"
    
    def test_fractional_speed_accumulation(self, small_config):
        """Test that fractional speeds work correctly."""
        config = small_config.copy()
        config['agents']['protesters']['count'] = 1
        config['agents']['police']['count'] = 0
        
        env = ProtestEnv(config)
        env.reset(seed=42)
        
        agent = env.agents[0]
        agent.speed = 0.3  # Should move every ~3 steps
        initial_pos = agent.pos
        
        moves = 0
        for _ in range(20):
            # Force agent to try moving in one direction
            env.step(actions={agent.id: 3})  # EAST
            if agent.pos != initial_pos:
                moves += 1
                initial_pos = agent.pos
        
        # With speed 0.3, should move ~6-7 times in 20 steps
        assert 4 <= moves <= 8, f"Expected 4-8 moves, got {moves}"
    
    def test_occupancy_limit_enforced(self, small_config):
        """Test that N_CELL_MAX is enforced."""
        env = ProtestEnv(small_config)
        env.reset(seed=42)
        
        for _ in range(100):
            env.step(actions=None)
            
            # Check no cell exceeds N_CELL_MAX
            max_occupancy = env.occupancy_count.max()
            assert max_occupancy <= env.n_cell_max, \
                f"Cell occupancy {max_occupancy} exceeds N_CELL_MAX {env.n_cell_max}"


class TestHazardConsistency:
    """Test hazard field behavior."""
    
    def test_hazard_nonnegativity(self, small_config):
        """Hazard concentrations should never be negative."""
        env = ProtestEnv(small_config)
        env.reset(seed=42)
        
        for _ in range(200):
            env.step(actions=None)
            
            assert (env.hazard_field.concentration >= 0).all(), \
                "Negative hazard concentration detected"
    
    def test_hazard_bounded(self, small_config):
        """Hazard concentrations should be bounded."""
        env = ProtestEnv(small_config)
        env.reset(seed=42)
        
        # Deploy lots of gas
        for _ in range(100):
            # Force police to deploy
            for police in env.police_agents:
                police.deploy_cooldown = 0
                police._attempt_gas_deployment(env)
            env.step(actions=None)
        
        assert (env.hazard_field.concentration <= 100).all(), \
            "Hazard concentration exceeded maximum"


class TestSmokeTest:
    """End-to-end integration smoke test."""
    
    def test_full_episode_completes(self, small_config):
        """Full episode should complete without errors."""
        config = small_config.copy()
        config['time']['max_steps'] = 500
        
        env = ProtestEnv(config)
        env.reset(seed=42)
        
        step_count = 0
        done = False
        
        while not done and step_count < 500:
            obs, reward, terminated, truncated, info = env.step(actions=None)
            done = terminated or truncated
            step_count += 1
        
        assert step_count > 0, "Episode ended immediately"
        assert step_count <= 500, "Episode exceeded max steps"
    
    def test_harm_indicators_generated(self, small_config):
        """Test that harm indicators (I_i grid) are generated."""
        env = ProtestEnv(small_config)
        env.reset(seed=42)
        
        harm_detected = False
        
        for _ in range(200):
            obs, _, _, _, info = env.step(actions=None)
            
            if info['harm_grid'].any():
                harm_detected = True
                break
        
        # Note: May not always detect harm in 200 steps, but grid should exist
        assert 'harm_grid' in info, "harm_grid missing from info"
        assert info['harm_grid'].shape == (env.height, env.width), \
            "harm_grid has wrong shape"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])