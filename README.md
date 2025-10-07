# Protest Safety Planner

**Privacy-preserving, offline decision-support pipeline for urban protest harm reduction**

---

The Protest Safety Planner is a research prototype that simulates urban protest scenarios to generate **probability-of-harm maps** for mutual-aid coordination. The system combines:

- **Agent-based simulation** (FR1) with heterogeneous protesters and police
- **Physics-based hazard modeling** (tear gas, water cannons, projectiles)
- **Monte Carlo risk aggregation** (FR2) with bootstrap uncertainty quantification
- **Privacy-by-design architecture** (no person-level tracking)

---

## Key Features

### FR1: Stylized Digital Twin
- **100×100 grid** representing ~500m×500m urban area (5m cell resolution)
- **Heterogeneous agents**:
    - 3 protester profiles: cautious (30%), average (50%), bold (20%)
    - Dynamic goal selection with hazard avoidance
    - Multi-step lookahead for risk assessment

- **Multi-hazard modeling**:
  - Tear gas: Advection-diffusion PDE with wind, decay, obstacle blocking
  - Water cannons: Directional push with stun probability
  - Projectiles: Probabilistic incapacitation events

- **Deterministic seeding**: Reproducible rollouts for Monte Carlo analysis

### FR2: Monte Carlo Aggregator
- **Parallel execution**: 200 rollouts in ~46 minutes (4 cores)
- **Bootstrap confidence intervals**: 1000 samples per cell
- **Convergence validated**: MAE < 0.0004 between N=100 and N=200
- **Calibration metrics**: Brier score 0.0014 (excellent)
- **Output**: Per-cell harm probabilities with 95% CIs

---

## Installation

### Requirements
- Python 3.10+
- 4+ CPU cores recommended (for parallel Monte Carlo)
- 4GB RAM minimum
- ~2GB disk space

### Quick Start
```bash
# Clone repository
git clone https://github.com/lindahnyambura/protest_safety_planner.git
cd protest_safety_planner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src; print('Installation successful!')"
```

## Usage
### 1. Quick Demo (Single Episode)
Visualize a single protest scenario with live animation:
```bash
python visualize_live.py --steps 500 --seed 42
```

Output: Real-time matplotlib animation showing:

- Agent movements (color-coded by state)
- Hazard concentration fields
- Crowd density heatmap
- Post-episode diagnostic plots
**Expected runtime:** ~30 seconds for 500 steps

### 2. Full Validation (FR1 + FR2)
Run complete pipeline with Monte Carlo aggregation:

```
python run_complete_demo.py
```

Interactive prompts:

Select configuration: Development (50 rollouts, ~16 min) or Production (200 rollouts, ~46 min)

**Output artifacts:**
```
artifacts/
├── rollouts/production_run/
│   ├── p_sim.npy              # Per-cell harm probabilities (100×100)
│   ├── p_sim_ci_lower.npy     # Bootstrap CI lower bounds
│   ├── p_sim_ci_upper.npy     # Bootstrap CI upper bounds
│   ├── I_rollouts.npy         # Binary harm indicators (200×100×100)
│   └── metadata.json          # Run statistics and config hash
├── validation/
│   ├── 01_environment_state.png
│   ├── 02_monte_carlo_results.png
│   └── 03_agent_profiles.png
└── experiment_log.json
```

### 3. Custom Scenarios
Edit `configs/default_scenario.yaml` to customize:

```
# Example: Increase gas intensity
hazards:
  gas:
    inj_intensity: 15.0  # mg/m³/s (default: 12.0)
    diffusion_coeff: 0.3  # m²/s
    decay_rate: 0.01      # s⁻¹

# Example: Change agent distribution
agents:
  protesters:
    count: 150  # Increase crowd size
    types:
      cautious:
        ratio: 0.4  # 40% cautious (up from 30%)
```

Then run:

```
python run_complete_demo.py --config configs/my_scenario.yaml
```
---

## Project Structure
```
protest_safety_planner/
├── configs/
│   ├── default_scenario.yaml    # Production parameters
│   └── dev_scenario.yaml        # Fast testing config
├── src/
│   ├── env/
│   │   ├── agent.py             # Protester & police behavior
│   │   ├── hazards.py           # Gas diffusion physics
│   │   ├── hazard_manager.py   # Multi-hazard orchestration
│   │   ├── protest_env.py       # Gymnasium environment
│   │   └── map_loader.py        # Obstacle generation
│   ├── monte_carlo/
│   │   └── aggregator.py        # Parallel rollouts + bootstrap CIs
│   ├── utils/
│   │   └── visualization.py     # Plotting suite
│   └── __init__.py
├── tests/
│   ├── test_env_determinism.py  # Reproducibility checks
│   ├── test_agents.py           # Behavioral validation
│   └── test_hazards.py          # Physics unit tests
├── artifacts/                   # Generated outputs (gitignored)
├── run_complete_demo.py         # Main entry point (FR1+FR2)
├── visualize_live.py            # Real-time episode viewer
├── requirements.txt
├── Dockerfile
├── Makefile
└── README.md
```

## Configuration Reference

### Critical Parameters

| **Parameter** | **Default** | **Description** | **Justification** |
|----------------|-------------|-----------------|-------------------|
| **Grid** |  |  |  |
| `cell_size_m` | 5.0 | Physical cell size (meters) | Balances granularity vs. computational cost (Lovreglio et al., 2016) |
| `grid.width` / `grid.height` | 100 | Grid dimensions | 500 m × 500 m — typical central business district protest zone |
| **Gas Physics** |  |  |  |
| `diffusion_coeff` | 0.3 m²/s | Turbulent diffusion coefficient | Represents urban wind mixing (Zhu et al., 2024) |
| `decay_rate` | 0.01 s⁻¹ | Exponential gas decay rate | ≈70 s half-life (CDC CS gas data) |
| `k_harm` | 0.0083 (mg·s/m³)⁻¹ | Harm rate constant | Calibrated for H = 5 after 60 s at 10 mg/m³ exposure |
| `inj_intensity` | 12.0 mg/m³/s | Emission rate | Tuned for 15–25% incapacitation target |
| **Agent Behavior** |  |  |  |
| `speed (cautious)` | 1.0 m/s | Walking speed | Slow pace typical of risk-averse individuals |
| `speed (average)` | 1.2 m/s | Normal walking speed | Standard pedestrian velocity (Helbing et al., 2000) |
| `risk_tolerance` | 0.1 / 0.3 / 0.6 | Hazard sensitivity | Derived from evacuation behavior studies (Moussaïd et al., 2011) |
| **Monte Carlo** |  |  |  |
| `n_rollouts` | 200 | Monte Carlo sample size | Converges with MAE < 0.0004 at N = 100 |
| `bootstrap_samples` | 1000 | Confidence interval resampling | Standard bootstrap estimation parameter |
|  |  |  |  |
| |  |  |  |

## Validation Results

### Convergence Analysis

| **N Rollouts** | **Mean P(harm)** | **MAE from Previous** | **Runtime** |
|----------------|------------------|------------------------|-------------|
| 50  | 0.0031 | –        | 15.8 min |
| 100 | 0.0032 | 0.0004   | 25.6 min |
| 200 | 0.0032 | 0.0003   | 45.8 min |

**Conclusion:** Converged by N=100; N=200 provides marginal improvement (0.03%) at high compute cost.

#### Calibration

- **Brier Score:** 0.0014 (excellent; <0.01 threshold)
- **Coverage:** 95% CIs well-behaved across all cells
- **Ground-truth validation:** Pending (see Limitations)

#### Performance

- **Throughput:** 0.073 rollouts/sec (4 cores, N=200)
- **Episode speed:** 13-22 steps/sec (varies by density)
- **Memory:** ~2GB RAM peak usage

## Known Limitations

### FR1: Simulation Physics

1. **2D approximation:** No vertical dispersion or atmospheric stability

    - **Mitigation:** Conservative diffusion coefficient (upper range)
    - **Next:** Compare against 3D CFD (Cabello et al. 2025)


2. **Concentration cap:** Clipped at 200 mg/m³

    - **Justification:** CDC CS gas data (100-200 mg/m³ typical)
    - **Impact:** Minimal (natural peak: 162 mg/m³)


3. **Simplified social force:** Utility-based, not full Helbing model

    - Impact: Missing crowd pushing, panic contagion
    - Trade-off: 10× faster, captures key behaviors

4. **Static risk tolerance:** No panic escalation over time

    - **Next:** Dynamic risk adjustment based on cumulative harm

### FR2: Validation Gaps

5. **No ground-truth testing:** Calibration metrics computed but not validated

    - **Next:** Synthetic scenarios with analytical P(harm) solutions


6. **Single parameter set:** No sensitivity analysis yet

    - **Next:** Systematic sweep over diffusion_coeff, k_harm


7. **High incapacitation rate:** 57% in simulation vs. ~0.002% in real protests (Kenya 2023-2024)

    - **Gap factor:** ~28,000×
    - **Likely causes:** Spatial compression, aggressive k_harm
    - **Next:** Recalibrate to target 10-20% incapacitation

### Data & Assumptions

8. **Synthetic obstacles only:** No real Nairobi OSM data (disabled for stability)

    - **Current:** Stylized rectangular buildings
    - **Next:** Integrate real CBD geometry with careful validation


9. **Temporal aggregation:** Uses OR operation (harm at any timestep)

    - **Alternative:** Max, mean, 95th percentile (not yet compared)


### Roadmap
#### Current Status (v0.1)

 - FR1: Functional simulation with multi-hazard modeling
 - FR2: Monte Carlo aggregator with bootstrap CIs
 - Convergence analysis and calibration metrics
 - Visualization suite

#### Next Iteration (v0.2)

 - Ground-truth validation (synthetic test scenarios)
 - Reliability diagrams (calibration plots)
 - Parameter sensitivity analysis
 - Literature review completion (all parameters justified)
 - Fix: Recalibrate incapacitation rate (target 15-25%)

 #### Future Work (v1.0)

 - FR3: Risk-aware A* planner (survival-based routing)
 - FR4: Baseline comparisons (shortest path vs. risk-aware)
 - FR5: CV adapter (video-to-hazard pipeline)
 - Real Nairobi OSM integration
 - Mobile app integration (frontend)
 - Real-world validation (Kenya protest data)

 ### Acknowledgments
#### Literature Foundations

1. Crowd dynamics: Helbing et al. (2000, 2005) - Social force model
2. Evacuation behavior: Moussaïd et al. (2011), Lovreglio et al. (2016)
3. Gas dispersion: Cabello et al. (2025), Zhu et al. (2024)
4. Monte Carlo methods: Wolny-Dominiak & Zadło (2024)

#### Data Sources

- Protest incidents: ACLED (2024) - Kenya protest tracker
- Human rights: Amnesty International, Human Rights Watch (2024-2025)
- CS gas effects: CDC NIOSH physiological data

#### Tools

- Simulation: Gymnasium (OpenAI)
- Scientific computing: NumPy, SciPy
- Visualization: Matplotlib
- Parallelization: Joblib
- Testing: pytest