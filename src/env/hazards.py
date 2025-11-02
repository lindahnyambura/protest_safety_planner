"""
hazards.py - Hazard field dynamics (gas diffusion, decay, advection)

Physics-based tear gas dispersion model with:
- Turbulent diffusion (von Neumann stencil)
- Wind advection (upwind finite difference)
- Exponential decay (settling + dispersion)
- Sustained emission sources (multi-step canisters)
- Obstacle blocking (gas doesn't penetrate walls)

Literature: Cabello et al. (2025) - Gaussian plume + CFD validation
"""

import numpy as np
from scipy.signal import convolve2d


class HazardField:
    """
    Tear gas concentration field with physics-based dynamics.
    
    Solves: ∂C/∂t = D∇²C - v·∇C - γC + S(x,y,t)
    where:
        C: concentration (mg/m³)
        D: diffusion coefficient (m²/s)
        v: wind velocity (m/s)
        γ: decay rate (s⁻¹)
        S: source term (mg/m³/s)
    """
    
    def __init__(self, 
                 height: int,
                 width: int,
                 diffusion_coeff: float,
                 decay_rate: float,
                 k_harm: float,
                 delta_t: float,
                 cell_size_m: float,
                 wind_direction=(0, 0),
                 wind_speed_m_s=0.0,
                 obstacle_mask=None):
        """
        Initialize hazard field.
        
        Args:
            height, width: Grid dimensions (cells)
            diffusion_coeff: Turbulent diffusion coefficient (m²/s)
                             Typical: 0.1-0.5 for urban wind (Cabello 2025)
            decay_rate: Base decay rate (s⁻¹)
                       Typical: 0.01 for CS gas with wind
            k_harm: Harm rate coefficient ((mg·s/m³)⁻¹)
                   Calibrated: 0.0083 for incap after 60s @ 10mg/m³
            delta_t: Timestep duration (s)
            cell_size_m: Physical cell size (m)
            wind_direction: (dx, dy) unit vector
            wind_speed_m_s: Wind speed (m/s)
            obstacle_mask: Boolean array (True = walls block gas)
        """
        # Grid parameters
        self.height = height
        self.width = width
        self.cell_size_m = cell_size_m
        self.delta_t = delta_t
        
        # Physical parameters (store both physical and grid units)
        self.D_physical = diffusion_coeff  # m²/s
        self.D_grid = diffusion_coeff / (cell_size_m ** 2)  # grid²/step
        
        # Decay (wind-enhanced)
        self.gamma_base = decay_rate
        self.gamma_effective = decay_rate * (1.0 + 0.5 * wind_speed_m_s)
        
        # Harm model
        self.k_harm = k_harm
        
        # Wind parameters
        wind_dir_array = np.array(wind_direction, dtype=np.float32)
        wind_norm = np.linalg.norm(wind_dir_array)
        if wind_norm > 0:
            self.wind_direction = wind_dir_array / wind_norm  # Normalize
        else:
            self.wind_direction = np.zeros(2, dtype=np.float32)
        self.wind_speed_m_s = wind_speed_m_s
        
        # Obstacle masking
        if obstacle_mask is None:
            self.obstacle_mask = np.zeros((height, width), dtype=bool)
        else:
            self.obstacle_mask = obstacle_mask.astype(bool)
        
        # State arrays
        self.concentration = np.zeros((height, width), dtype=np.float32)
        self.active_sources = []  # List of dicts: {x, y, intensity, duration}
        
        # Laplacian kernel (von Neumann 4-neighbor)
        self.laplacian_kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)
    
    def update(self, delta_t: float):
        """
        Update concentration field for one timestep.
        
        Physics:
            1. Emission: +S (sustained sources) ← MUST COME FIRST
            2. Diffusion: D∇²C
            3. Advection: -v·∇C (wind transport)
            4. Decay: -γC (settling + dispersion)
            5. Obstacle blocking: C=0 inside wallslls
        
        Args:
            delta_t: Timestep (should match self.delta_t)
        """
        # 1. SOURCE EMISSION (FIRST - critical ordering)
        source_term = np.zeros((self.height, self.width), dtype=np.float32)
        remaining_sources = []
    
        for src in self.active_sources:
            x, y = src['x'], src['y']
        
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
        
            # SUSTAINED emission profile (Gaussian temporal distribution)
            # Peak at 30% through duration, then gradual decline
            t_frac = (src['initial_duration'] - src['duration']) / src['initial_duration']
        
            if t_frac < 0.3:
                # Ramp-up phase
                emission_multiplier = t_frac / 0.3
            elif t_frac < 0.7:
                # Peak phase
                emission_multiplier = 1.0
            else:
                # Decline phase
                emission_multiplier = (1.0 - t_frac) / 0.3
        
            intensity = src['intensity'] * emission_multiplier
            source_term[y, x] += intensity * delta_t
        
            # Decrement duration
            src['duration'] -= 1
            if src['duration'] > 0:
                remaining_sources.append(src)
    
        self.active_sources = remaining_sources
    
        # 2. ADD emissions to field (BEFORE diffusion)
        self.concentration += source_term
    
        # 3. DIFFUSION (with obstacle-aware boundaries)
        laplacian = convolve2d(
            self.concentration,
            self.laplacian_kernel,
            mode='same',
            boundary='fill',  # Changed from 'symm' to 'fill'
            fillvalue=0.0     # Gas disperses at boundaries (not reflected)
        )
    
        # 4. WIND ADVECTION (enhanced for urban channeling)
        advection = self._compute_advection()
    
        # 5. DECAY (reduced for realistic persistence)
        # Literature: CS gas half-life 60-300s depending on ventilation
        # Urban canyon: slower decay due to recirculation
        effective_decay = self.gamma_effective * 0.7  # 30% slower decay
    
        # 6. UPDATE CONCENTRATION
        self.concentration += delta_t * (
            self.D_grid * laplacian - advection - effective_decay * self.concentration
        )
    
        # 7. CONSTRAINTS
        self.concentration = np.clip(self.concentration, 0.0, 200.0)
    
        # 8. OBSTACLE HANDLING (gas pools near walls, doesn't penetrate)
        # NEW: Apply exponential decay only inside obstacles
        self.concentration[self.obstacle_mask] *= 0.5  # Rapid decay inside buildings
    
        # DEBUG: Enhanced logging
        if len(remaining_sources) > 0:
            total_emission = source_term.sum()
            peak_conc = self.concentration.max()
            mean_conc = self.concentration[self.concentration > 0.1].mean() if (self.concentration > 0.1).any() else 0.0
        
            print(f"[HAZARD] Emitted {total_emission:.1f} mg, {len(remaining_sources)} sources active, "
                f"peak={peak_conc:.1f}, mean={mean_conc:.1f}")
    
    def _compute_advection(self) -> np.ndarray:
        """
        Compute wind advection using upwind finite difference.
        
        Upwind scheme: ∂C/∂x ≈ (C[i] - C[i-1])/Δx if wind > 0
                               (C[i+1] - C[i])/Δx if wind < 0
        
        This ensures numerical stability (prevents oscillations).
        
        Returns:
            advection: Wind transport term (mg/m³/s)
        """
        if self.wind_speed_m_s == 0:
            return np.zeros_like(self.concentration)
        
        # Convert wind speed to grid units
        wind_grid = self.wind_speed_m_s / self.cell_size_m
        wx, wy = self.wind_direction
        
        # X-direction advection (upwind)
        if wx > 0:
            # Wind blowing east: use backward difference
            dC_dx = np.diff(self.concentration, axis=1, prepend=self.concentration[:, :1])
        elif wx < 0:
            # Wind blowing west: use forward difference
            dC_dx = np.diff(self.concentration, axis=1, append=self.concentration[:, -1:])
        else:
            dC_dx = np.zeros_like(self.concentration)
        
        # Y-direction advection (upwind)
        if wy > 0:
            # Wind blowing south (in image coords): use backward difference
            dC_dy = np.diff(self.concentration, axis=0, prepend=self.concentration[:1, :])
        elif wy < 0:
            # Wind blowing north: use forward difference
            dC_dy = np.diff(self.concentration, axis=0, append=self.concentration[-1:, :])
        else:
            dC_dy = np.zeros_like(self.concentration)
        
        # Combined advection term: v · ∇C
        advection = wind_grid * (wx * dC_dx + wy * dC_dy)
        return advection
    
    def add_source(self, x: int, y: int, intensity: float, duration_steps: int = 30):
        """
        Deploy tear gas canister with sustained emission.
        
        Models real canister behavior: emission over 30-60 seconds, not instant pulse.
        
        Args:
            x, y: Deployment location (grid coordinates)
            intensity: Emission rate (mg/m³/s)
            duration_steps: Emission duration (timesteps, default 30s)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.active_sources.append({
                'x': int(x),
                'y': int(y),
                'intensity': float(intensity),
                'duration': int(duration_steps)
            })
    
    def get_harm_probability(self, x: int, y: int) -> float:
        """
        Compute instantaneous harm probability at location.
        
        Used by agents for decision-making (local risk assessment).
        
        Args:
            x, y: Grid coordinates
            
        Returns:
            p_harm: Probability of harm this timestep [0, 1]
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        
        concentration = self.concentration[y, x]
        
        # Exponential dose-response: p = 1 - exp(-k·c·Δt)
        p_harm = 1.0 - np.exp(-self.k_harm * concentration * self.delta_t)
        
        # Numerical stability
        return float(np.clip(p_harm, 1e-6, 0.999999))