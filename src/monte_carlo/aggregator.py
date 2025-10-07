"""
aggregator.py - Monte Carlo aggregator for FR2

Runs N stochastic rollouts and produces:
- p_sim: Empirical per-cell harm probabilities
- Bootstrap confidence intervals with convergence checks
- Summary statistics including incapacitation rates
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from joblib import Parallel, delayed
import json
import hashlib

from src.env.agent import AgentState  # Import for state checking


class MonteCarloAggregator:
    """
    Monte Carlo aggregator for empirical harm probability estimation.
    Runs multiple stochastic rollouts of the same scenario and aggregates
    harm indicators to produce p_sim(x,y) = P(harm occurs in cell (x,y)).
    """
    
    def __init__(self, env_class, config: Dict, output_dir: str = "artifacts/rollouts"):
        """
        Initialize Monte Carlo aggregator from configuration.

        Args:
            env_class: Gymnasium environment class
            config: Configuration dictionary
            output_dir: Path to save rollout data (default="artifacts/rollouts")
        """
        self.env_class = env_class
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        mc_cfg = config.get('monte_carlo', {})
        self.n_rollouts = mc_cfg.get('n_rollouts', 200)
        self.n_bootstrap = mc_cfg.get('bootstrap_samples', 1000)
        self.n_jobs = mc_cfg.get('n_jobs', 4)
        
        self.height = config['grid']['height']
        self.width = config['grid']['width']
        self.I_rollouts = None
    
    def run_monte_carlo(self, base_seed: Optional[int] = None, verbose: bool = True) -> Dict:
        """
        Run Monte Carlo aggregation to estimate empirical harm probabilities.

        Args:
            base_seed: Optional[int] - Base seed for rollouts (default=None)
            verbose: bool - Print progress and summary statistics (default=True)

        Returns:
            Dict - Run results including p_sim, summary statistics, and convergence metrics
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
        rollout_seeds = [base_seed + i for i in range(self.n_rollouts)]
        
        if self.n_jobs > 1:
            results = Parallel(n_jobs=self.n_jobs, verbose=10 if verbose else 0)(
                delayed(self._run_single_rollout)(seed_i, i)
                for i, seed_i in enumerate(rollout_seeds)
            )
        else:
            results = [self._run_single_rollout(seed_i, i) 
                      for i, seed_i in enumerate(rollout_seeds)]
        
        # FIXED: Preallocate array
        self.I_rollouts = np.empty((self.n_rollouts, self.height, self.width), dtype=bool)
        for i, r in enumerate(results):
            self.I_rollouts[i] = r['harm_grid']
        
        rollout_time = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Completed {self.n_rollouts} rollouts in {rollout_time:.1f}s")
            print(f"  Average: {rollout_time/self.n_rollouts:.2f}s per rollout")
        
        # Compute p_sim
        p_sim = self._compute_p_sim()
        p_sim_ci = self._compute_bootstrap_ci(verbose=verbose)
        
        # NEW: Convergence check
        convergence_stats = self._check_convergence()
        
        summary = self._compute_summary_statistics(results)
        
        output = {
            'p_sim': p_sim,
            'p_sim_ci_lower': p_sim_ci[0],
            'p_sim_ci_upper': p_sim_ci[1],
            'n_rollouts': self.n_rollouts,
            'base_seed': base_seed,
            'summary': summary,
            'convergence': convergence_stats,  # NEW
            'runtime_seconds': rollout_time,
            'config_hash': self._compute_config_hash()  # NEW
        }
        
        if verbose:
            self._print_summary(p_sim, summary, convergence_stats)
        
        return output
    
    def _run_single_rollout(self, seed: int, rollout_idx: int) -> Dict:
        """
        Run a single rollout of the environment and track harm indicators.

        Args:
            seed: int - Base seed for rollout
            rollout_idx: int - Index of rollout (for debugging)

        Returns:
            Dict - Rollout results including harm grid, episode length,
                number of agents harmed, and number of agents incapacitated
        """
        
        env = self.env_class(self.config)
        obs, info = env.reset(seed=seed)
        
        # Track time series
        hazard_history = []
        incapacitated_timeline = []
        harm_events_timeline = []

        harm_grid = np.zeros((self.height, self.width), dtype=bool)
        done = False
        step_count = 0
        max_steps = self.config['time']['max_steps']
        
        # FIXED: Track all harm types
        agents_harmed = set()
        agents_incapacitated = set()
        
        while not done and step_count < max_steps:
            obs, reward, terminated, truncated, info = env.step(actions=None)
            harm_grid |= info['harm_grid']

            # Record time series
            hazard_history.append(env.hazard_field.concentration.max())
            incapacitated_timeline.append(sum(1 for a in env.agents 
                                            if a.state == AgentState.INCAPACITATED))
            harm_events_timeline.append(info['harm_grid'].sum())
            
            # Track by state (includes gas, water, shooting)
            for agent in env.agents:
                if agent.harm_events > 0 or agent.state in [AgentState.STUNNED, AgentState.INCAPACITATED]:
                    agents_harmed.add(agent.id)
                if agent.state == AgentState.INCAPACITATED:
                    agents_incapacitated.add(agent.id)
            
            done = terminated or truncated
            step_count += 1
        
        return {
            'harm_grid': harm_grid,
            'hazard_history': hazard_history,
            'incapacitated_timeline': incapacitated_timeline,
            'harm_events_timeline': harm_events_timeline,
            'episode_length': step_count,
            'n_agents_harmed': len(agents_harmed),
            'n_agents_incapacitated': len(agents_incapacitated),
            'seed': seed
        }
    
    def _compute_p_sim(self) -> np.ndarray:
        """
        Compute the mean of the rollout results along the first axis (i.e., the mean
        probability of harm for each cell in the environment).
        
        Returns:
            np.ndarray - Mean probability of harm for each cell, shape (height, width)
        """
        return self.I_rollouts.mean(axis=0).astype(np.float32)
    
    def _compute_bootstrap_ci(self, confidence: float = 0.95, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the confidence interval for the mean probability of harm.

        Args:
            confidence (float, optional): Confidence level (default=0.95)
            verbose (bool, optional): Print progress messages (default=True)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of the confidence interval, shape (height, width)
        """
        alpha = 1 - confidence
        
        if verbose:
            print(f"\n  Bootstrap resampling ({self.n_bootstrap} samples)...")
        
        # FIXED: Isolated seed space
        bootstrap_seed = hash(('bootstrap', self.config['simulation']['base_seed'])) % (2**32)
        rng = np.random.default_rng(bootstrap_seed)
        
        bootstrap_p_sims = np.zeros((self.n_bootstrap, self.height, self.width), dtype=np.float32)
        
        for b in range(self.n_bootstrap):
            indices = rng.choice(self.n_rollouts, size=self.n_rollouts, replace=True)
            bootstrap_p_sims[b] = self.I_rollouts[indices].mean(axis=0)
        
        ci_lower = np.percentile(bootstrap_p_sims, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(bootstrap_p_sims, 100 * (1 - alpha / 2), axis=0)
        
        return ci_lower.astype(np.float32), ci_upper.astype(np.float32)
    
    def _check_convergence(self) -> Dict:
        """Check if N rollouts is sufficient."""
        convergence = {}
        n_samples_list = [50, 100, 200]
        prev_p_sim = None
        
        for n in n_samples_list:
            if n > self.n_rollouts:
                break
            
            subset = self.I_rollouts[:n]
            p_sim_n = subset.mean(axis=0)
            
            if prev_p_sim is not None:
                mae = np.abs(p_sim_n - prev_p_sim).mean()
                convergence[f'mae_{prev_n}_to_{n}'] = float(mae)
            
            prev_p_sim = p_sim_n
            prev_n = n
        
        return convergence
    
    def _compute_summary_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute summary statistics from a list of rollout results.

        Args:
            results: List of rollout results

        Returns:
            Dict with summary statistics, including mean and standard deviation of episode lengths and number of agents harmed, as well as the minimum and maximum episode lengths and the incapacitation rate of protesters.

        """
        episode_lengths = [r['episode_length'] for r in results]
        agents_harmed = [r['n_agents_harmed'] for r in results]
        agents_incapacitated = [r['n_agents_incapacitated'] for r in results]
        
        return {
            'mean_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            'mean_agents_harmed': float(np.mean(agents_harmed)),
            'std_agents_harmed': float(np.std(agents_harmed)),
            'mean_agents_incapacitated': float(np.mean(agents_incapacitated)),
            'incapacitation_rate': float(np.mean(agents_incapacitated)) / self.config['agents']['protesters']['count'],
            'min_episode_length': int(np.min(episode_lengths)),
            'max_episode_length': int(np.max(episode_lengths))
        }
    
    def _compute_config_hash(self) -> str:
        """Hash config for reproducibility tracking."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def plot_convergence_analysis(self, save_path=None):
        """Plot p_sim convergence as N increases."""
        n_values = [50, 100, 150, 200]
        convergence_data = []
    
        for n in n_values:
            if n > self.n_rollouts:
                break
            subset = self.I_rollouts[:n]
            p_sim_n = subset.mean(axis=0)
        
            convergence_data.append({
                'n': n,
                'mean_p': p_sim_n.mean(),
                'max_p': p_sim_n.max(),
                'std_p': p_sim_n.std()
            })
    
        # Plot (will add to visualization.py)
        return convergence_data
    
    def compute_calibration_metrics(self):
        """Compute Brier score for p_sim calibration."""
        # For each cell, compare predicted p_sim to observed frequency
        brier_scores = []
        for i in range(self.height):
            for j in range(self.width):
                p_pred = self.I_rollouts[:, i, j].mean()  # Predicted
                outcomes = self.I_rollouts[:, i, j]       # Observed {0,1}
                brier = np.mean((p_pred - outcomes)**2)
                brier_scores.append(brier)
    
        return {
            'mean_brier': np.mean(brier_scores),
            'median_brier': np.median(brier_scores)
        }
    
    def _print_summary(self, p_sim, summary, convergence):
        """
        Print a summary of the Monte Carlo results.

        Args:
            p_sim (np.ndarray): Mean probability of harm for each cell
            summary (Dict): Summary statistics of the rollouts
            convergence (Dict, optional): Convergence metrics of the rollouts

        Prints a summary of the Monte Carlo results, including the mean and max probability of harm, the number of cells with p>0.1 and p>0.5, and the mean episode length, mean agents harmed, and incapacitation rate. If convergence metrics are provided, prints them as well.
        """
        print(f"\n{'='*60}")
        print("Monte Carlo Results:")
        print(f"{'='*60}")
        print(f"Mean harm probability: {p_sim.mean():.4f}")
        print(f"Max harm probability: {p_sim.max():.4f}")
        print(f"Cells with p>0.1: {(p_sim > 0.1).sum()} ({100*(p_sim > 0.1).sum()/p_sim.size:.1f}%)")
        print(f"Cells with p>0.5: {(p_sim > 0.5).sum()} ({100*(p_sim > 0.5).sum()/p_sim.size:.1f}%)")
        print(f"\nMean episode length: {summary['mean_episode_length']:.1f} steps")
        print(f"Mean agents harmed: {summary['mean_agents_harmed']:.1f}")
        print(f"Incapacitation rate: {summary['incapacitation_rate']:.1%}")
        
        if convergence:
            print(f"\nConvergence check:")
            for k, v in convergence.items():
                print(f"  {k}: {v:.6f}")
    
    def save_results(self, output: Dict, run_id: str = "production_run"):
        """
        Save the results of the Monte Carlo simulation to a directory.

        Args:
            output (Dict): Output of the Monte Carlo simulation, containing the mean probability of harm, confidence intervals, summary statistics, and convergence metrics.
            run_id (str, optional): Identifier for the run (default="production_run").

        Saves the following to the specified directory:
            - p_sim.npy: Mean probability of harm for each cell
            - p_sim_ci_lower.npy: Lower bounds of the confidence interval for the mean probability of harm
            - p_sim_ci_upper.npy: Upper bounds of the confidence interval for the mean probability of harm
            - I_rollouts.npy (optional): Rollout results for each cell
            - metadata.json: Dictionary containing the number of rollouts, base seed, summary statistics, convergence metrics, runtime, config hash, timestamp, and grid shape

        Prints a message indicating where the results were saved.
        """
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(run_dir / 'p_sim.npy', output['p_sim'])
        np.save(run_dir / 'p_sim_ci_lower.npy', output['p_sim_ci_lower'])
        np.save(run_dir / 'p_sim_ci_upper.npy', output['p_sim_ci_upper'])
        
        if self.config.get('output', {}).get('save_I_grids', True):
            np.save(run_dir / 'I_rollouts.npy', self.I_rollouts)
        
        metadata = {
            'n_rollouts': output['n_rollouts'],
            'base_seed': output['base_seed'],
            'summary': output['summary'],
            'convergence': output.get('convergence', {}),
            'runtime_seconds': output['runtime_seconds'],
            'config_hash': output['config_hash'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'grid_shape': [self.height, self.width]
        }
        
        with open(run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n Results saved to {run_dir}/")