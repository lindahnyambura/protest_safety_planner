"""
hazards.py - Hazard field dynamics (gas diffusion, decay)

Hazard concentration field with discrete diffusion and decay.
Models gas dispersal using:
- Discrete Laplacian diffusion (von Neumann neighborhood)
- Exponential decay
- Source injection (police deployments)
- Probability of harm to agents based on local concentration
"""

import numpy as np
from scipy.signal import convolve2d


class HazardField:
    """
    Hazard concentration field with discrete diffusion and decay.
    
    Models gas dispersal using:
    - Discrete Laplacian diffusion (von Neumann neighborhood)
    - Exponential decay
    - Source injection (police deployments)
    """
    
    def __init__(self, 
                 height: int,
                 width: int,
                 diffusion_coeff: float,
                 decay_rate: float,
                 k_harm: float,
                 delta_t: float):
        """
        Initialize hazard field.
        
        Args:
            height: Grid height
            width: Grid width
            diffusion_coeff: Diffusion coefficient (grid units)
            decay_rate: Decay rate (per-step)
            k_harm: Harm rate parameter for p_harm calculation
            delta_t: Timestep duration
        """
        self.height = height
        self.width = width
        self.D = diffusion_coeff
        self.gamma = decay_rate
        self.k_harm = k_harm
        self.delta_t = delta_t
        
        # State arrays (locked data types)
        self.concentration = np.zeros((height, width), dtype=np.float32)
        self.sources = np.zeros((height, width), dtype=np.float32)
        
        # Precompute Laplacian kernel (von Neumann, 4-neighbor)
        self.laplacian_kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)
    
    def update(self, delta_t: float):
        """
        Update hazard field for one timestep.
        
        Performs:
        1. Diffusion (discrete Laplacian)
        2. Decay (exponential)
        3. Source injection
        
        Args:
            delta_t: Timestep duration (usually self.delta_t)
        """
        # 1. Diffusion: C_new = C + D * Δt * ∇²C
        # Use scipy convolve2d for vectorized operation (C-optimized)
        laplacian = convolve2d(
            self.concentration,
            self.laplacian_kernel,
            mode='same',
            boundary='fill',
            fillvalue=0
        )
        
        # 2. Update equation
        self.concentration += (
            self.D * delta_t * laplacian            # Diffusion
            - self.gamma * self.concentration * delta_t  # Decay
            + self.sources * delta_t                # Injection
        )
        
        # 3. Numerical stability: clamp to valid range
        self.concentration = np.clip(self.concentration, 0, 100)
        
        # 4. Reset sources (pulse injection model)
        self.sources.fill(0)
    
    def add_source(self, x: int, y: int, intensity: float):
        """
        Add gas source at location (for manual injection).
        
        Args:
            x, y: Grid coordinates
            intensity: Injection intensity
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.sources[y, x] = intensity
    
    def get_harm_probability(self, x: int, y: int) -> float:
        """
        Get harm probability at location (for agent decision-making).
        
        Args:
            x, y: Grid coordinates
            
        Returns:
            p_harm: Probability of harm per timestep
        """
        concentration = self.concentration[y, x]
        p_harm = 1 - np.exp(-self.k_harm * concentration * self.delta_t)
        return float(np.clip(p_harm, 1e-6, 0.999999))