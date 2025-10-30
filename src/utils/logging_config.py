"""
logging_config.py - Centralized logging configuration for protest simulation

Add this file to src/utils/logging_config.py
"""

import os
from enum import IntEnum
from typing import Optional

class LogLevel(IntEnum):
    """Logging verbosity levels."""
    SILENT = 0      # No output (for production Monte Carlo runs)
    MINIMAL = 1     # Only critical warnings and final summary
    NORMAL = 2      # Standard progress updates every 20 steps
    VERBOSE = 3     # Detailed diagnostics every 10 steps
    DEBUG = 4       # Full debug output (all agent decisions, validation)

class SimulationLogger:
    """
    Centralized logger for simulation with configurable verbosity.
    
    Usage:
        from src.utils.logging_config import logger
        logger.set_level(LogLevel.NORMAL)
        logger.info("Simulation started")
        logger.debug("Agent 5 chose action 3")
    """
    
    def __init__(self, level: LogLevel = LogLevel.NORMAL):
        self.level = level
        self._step_count = 0
    
    def set_level(self, level: LogLevel):
        """Set logging verbosity."""
        self.level = level
    
    def set_step(self, step: int):
        """Update current simulation step for conditional logging."""
        self._step_count = step
    
    def silent(self, msg: str):
        """Always print (for critical errors only)."""
        print(msg)
    
    def minimal(self, msg: str):
        """Print at MINIMAL level and above."""
        if self.level >= LogLevel.MINIMAL:
            print(msg)
    
    def info(self, msg: str):
        """Print at NORMAL level and above."""
        if self.level >= LogLevel.NORMAL:
            print(msg)
    
    def verbose(self, msg: str):
        """Print at VERBOSE level and above."""
        if self.level >= LogLevel.VERBOSE:
            print(msg)
    
    def debug(self, msg: str):
        """Print at DEBUG level only."""
        if self.level >= LogLevel.DEBUG:
            print(msg)
    
    def step_periodic(self, msg: str, interval: int = 20):
        """Print periodically based on step count (NORMAL level)."""
        if self.level >= LogLevel.NORMAL and self._step_count % interval == 0:
            print(msg)
    
    def step_verbose(self, msg: str, interval: int = 10):
        """Print periodically for verbose logging."""
        if self.level >= LogLevel.VERBOSE and self._step_count % interval == 0:
            print(msg)

# Global logger instance
logger = SimulationLogger(level=LogLevel.NORMAL)

# Environment variable override
if 'SIM_LOG_LEVEL' in os.environ:
    level_map = {
        'SILENT': LogLevel.SILENT,
        'MINIMAL': LogLevel.MINIMAL,
        'NORMAL': LogLevel.NORMAL,
        'VERBOSE': LogLevel.VERBOSE,
        'DEBUG': LogLevel.DEBUG
    }
    env_level = os.environ['SIM_LOG_LEVEL'].upper()
    if env_level in level_map:
        logger.set_level(level_map[env_level])