"""
aggregator.py - Monte Carlo aggregator for FR2

Runs N stochastic rollouts and produces:
- p_sim: Empirical per-cell harm probabilities
- Bootstrap confidence intervals
- Summary statistics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from joblib import Parallel, delayed
import json


class MonteCarloAggregator:
    """
    Monte Carlo aggregator for empirical harm probability estimation.
    
    Runs multiple stochastic rollouts of the same scenario and aggregates
    harm indicators to produce p_sim(x,y) = P(harm occurs in cell (x,y)).
    """
    
    def __init__(self, env_class, config: Dict, output_dir: str = "artifacts/rollouts"):
        """
        Initialize aggregator.
        
        Args:
            env_class: ProtestEnv class (not instance)
            config: Configuration dictionary
            output_dir: Directory for saving outputs
        """
        self.env_class = env_class
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Monte Carlo configuration
        mc_cfg = config.get('monte_carlo', {})
        self.n_rollouts = mc_cfg.get('n_rollouts', 200)
        self.n_bootstrap = mc_cfg.get('bootstrap_samples', 1000)
        self.n_jobs = mc_cfg.get('n_jobs', 8)
        self.use_memmap = mc_cfg.get('use_memmap', True)
        
        # Grid dimensions
        self.height = config['grid']['height']
        self.width = config['grid']['width']
        
        # Storage for I_i indicators
        self.I_rollouts = None
        
    def run_monte_carlo(self, base_seed: Optional[int] = None,
                       verbose: bool = True) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Args:
            base_seed: Base seed for rollouts (seed_i = base_seed + i)
            verbose: Print progress information
            
        Returns:
            Dictionary with p_sim, confidence intervals, and metadata
        """
        if base_seed is None:
            base_seed = self.config['simulation']['base_seed']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Monte Carlo Aggregation: {self.n_rollouts} rollouts")
            print(f"{'='*60}")
            print(f"Grid: {self.width}×{self.height}")
            print(f"Parallel jobs: {self.n_jobs}")
        
        start_time = time.time()
        
        # Run rollouts in parallel
        if verbose:
            print(f"\nRunning {self.n_rollouts} rollouts...")
        
        rollout_seeds = [base_seed + i for i in range(self.n_rollouts)]
        
        if self.n_jobs > 1:
            # Parallel execution
            results = Parallel(n_jobs=self.n_jobs, verbose=10 if verbose else 0)(
                delayed(self._run_single_rollout)(seed_i, i)
                for i, seed_i in enumerate(rollout_seeds)
            )
        else:
            # Serial execution (for debugging)
            results = [
                self._run_single_rollout(seed_i, i)
                for i, seed_i in enumerate(rollout_seeds)
            ]
        
        # Extract I_i grids
        self.I_rollouts = np.array([r['harm_grid'] for r in results], dtype=bool)
        
        rollout_time = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Completed {self.n_rollouts} rollouts in {rollout_time:.1f}s")
            print(f"  Average: {rollout_time/self.n_rollouts:.2f}s per rollout")
        
        # Compute p_sim
        if verbose:
            print("\nComputing p_sim and confidence intervals...")
        
        p_sim = self._compute_p_sim()
        p_sim_ci = self._compute_bootstrap_ci(verbose=verbose)
        
        # Compute summary statistics
        summary = self._compute_summary_statistics(results)
        
        # Package results
        output = {
            'p_sim': p_sim,
            'p_sim_ci_lower': p_sim_ci[0],
            'p_sim_ci_upper': p_sim_ci[1],
            'n_rollouts': self.n_rollouts,
            'base_seed': base_seed,
            'summary': summary,
            'config': self.config,
            'runtime_seconds': rollout_time
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print("Monte Carlo Results:")
            print(f"{'='*60}")
            print(f"Mean harm probability: {p_sim.mean():.4f}")
            print(f"Max harm probability: {p_sim.max():.4f}")
            print(f"Cells with p>0.1: {(p_sim > 0.1).sum()} "
                  f"({100*(p_sim > 0.1).sum()/p_sim.size:.1f}%)")
            print(f"Mean episode length: {summary['mean_episode_length']:.1f} steps")
            print(f"Mean agents harmed: {summary['mean_agents_harmed']:.1f}")
        
        return output
    
    def _run_single_rollout(self, seed: int, rollout_idx: int) -> Dict:
        """
        Run a single rollout.
        
        Args:
            seed: RNG seed for this rollout
            rollout_idx: Rollout index (for logging)
            
        Returns:
            Dictionary with harm_grid and episode statistics
        """
        # Create environment
        env = self.env_class(self.config)
        obs, info = env.reset(seed=seed)
        
        # Initialize harm grid
        harm_grid = np.zeros((self.height, self.width), dtype=bool)
        
        # Run episode
        done = False
        step_count = 0
        max_steps = self.config['time']['max_steps']
        
        agents_harmed = set()
        
        while not done and step_count < max_steps:
            obs, reward, terminated, truncated, info = env.step(actions=None)
            
            # Accumulate harm indicators (OR operation)
            harm_grid |= info['harm_grid']
            
            # Track harmed agents
            for agent in env.agents:
                if agent.harm_events > 0:
                    agents_harmed.add(agent.id)
            
            done = terminated or truncated
            step_count += 1
        
        return {
            'harm_grid': harm_grid,
            'episode_length': step_count,
            'n_agents_harmed': len(agents_harmed),
            'seed': seed
        }
    
    def _compute_p_sim(self) -> np.ndarray:
        """
        Compute empirical harm probabilities.
        
        Returns:
            p_sim: Array of shape (H, W) with harm probabilities
        """
        # p_sim(x,y) = (1/N) * sum_i I_i(x,y)
        p_sim = self.I_rollouts.mean(axis=0).astype(np.float32)
        return p_sim
    
    def _compute_bootstrap_ci(self, confidence: float = 0.95,
                              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bootstrap confidence intervals for p_sim.
        
        Args:
            confidence: Confidence level (default 0.95)
            verbose: Print progress
            
        Returns:
            (ci_lower, ci_upper): Lower and upper bounds, shape (H, W)
        """
        alpha = 1 - confidence
        
        if verbose:
            print(f"  Bootstrap resampling ({self.n_bootstrap} samples)...")
        
        # Bootstrap resampling
        rng = np.random.default_rng(self.config['simulation']['base_seed'] + 99999)
        
        bootstrap_p_sims = np.zeros((self.n_bootstrap, self.height, self.width), 
                                    dtype=np.float32)
        
        for b in range(self.n_bootstrap):
            # Resample rollouts with replacement
            indices = rng.choice(self.n_rollouts, size=self.n_rollouts, replace=True)
            resampled_I = self.I_rollouts[indices]
            bootstrap_p_sims[b] = resampled_I.mean(axis=0)
        
        # Compute percentiles
        ci_lower = np.percentile(bootstrap_p_sims, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(bootstrap_p_sims, 100 * (1 - alpha / 2), axis=0)
        
        return ci_lower.astype(np.float32), ci_upper.astype(np.float32)
    
    def _compute_summary_statistics(self, results: List[Dict]) -> Dict:
        """Compute summary statistics across rollouts."""
        episode_lengths = [r['episode_length'] for r in results]
        agents_harmed = [r['n_agents_harmed'] for r in results]
        
        return {
            'mean_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            'mean_agents_harmed': float(np.mean(agents_harmed)),
            'std_agents_harmed': float(np.std(agents_harmed)),
            'min_episode_length': int(np.min(episode_lengths)),
            'max_episode_length': int(np.max(episode_lengths))
        }
    
    def save_results(self, output: Dict, run_id: str = "default"):
        """
        Save Monte Carlo results to disk.
        
        Args:
            output: Results dictionary from run_monte_carlo()
            run_id: Identifier for this run
        """
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save p_sim as .npy
        np.save(run_dir / 'p_sim.npy', output['p_sim'])
        np.save(run_dir / 'p_sim_ci_lower.npy', output['p_sim_ci_lower'])
        np.save(run_dir / 'p_sim_ci_upper.npy', output['p_sim_ci_upper'])
        
        # Save I_rollouts (optional, large)
        if self.config.get('output', {}).get('save_I_grids', True):
            np.save(run_dir / 'I_rollouts.npy', self.I_rollouts)
        
        # Save metadata
        metadata = {
            'n_rollouts': output['n_rollouts'],
            'base_seed': output['base_seed'],
            'summary': output['summary'],
            'runtime_seconds': output['runtime_seconds'],
            'grid_shape': [self.height, self.width],
            'config': {k: v for k, v in self.config.items() 
                      if k not in ['osm']}  # Don't save large OSM data
        }
        
        with open(run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Results saved to {run_dir}/")
    
    @staticmethod
    def load_results(run_dir: str) -> Dict:
        """
        Load Monte Carlo results from disk.
        
        Args:
            run_dir: Directory containing saved results
            
        Returns:
            Dictionary with p_sim and metadata
        """
        run_dir = Path(run_dir)
        
        output = {
            'p_sim': np.load(run_dir / 'p_sim.npy'),
            'p_sim_ci_lower': np.load(run_dir / 'p_sim_ci_lower.npy'),
            'p_sim_ci_upper': np.load(run_dir / 'p_sim_ci_upper.npy')
        }
        
        # Load I_rollouts if available
        if (run_dir / 'I_rollouts.npy').exists():
            output['I_rollouts'] = np.load(run_dir / 'I_rollouts.npy')
        
        # Load metadata
        with open(run_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        output.update(metadata)
        
        return output


def run_monte_carlo_batch(env_class, config: Dict, 
                          n_rollouts: int = 200,
                          output_dir: str = "artifacts/rollouts",
                          run_id: str = "default") -> Dict:
    """
    Convenience function to run Monte Carlo aggregation.
    
    Args:
        env_class: ProtestEnv class
        config: Configuration dictionary
        n_rollouts: Number of rollouts
        output_dir: Output directory
        run_id: Run identifier
        
    Returns:
        Results dictionary
    """
    # Override n_rollouts in config
    config['monte_carlo']['n_rollouts'] = n_rollouts
    
    # Create aggregator
    aggregator = MonteCarloAggregator(env_class, config, output_dir)
    
    # Run
    results = aggregator.run_monte_carlo(verbose=True)
    
    # Save
    aggregator.save_results(results, run_id=run_id)
    
    return results