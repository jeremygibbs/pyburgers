#!/usr/bin/env python
#
# PyBurgers
#
# Copyright (c) 2017–2026 Jeremy A. Gibbs
#
# This file is part of PyBurgers.
#
# This software is free and is distributed under the WTFPL license.
# See accompanying LICENSE file or visit https://www.wtfpl.net.
#

"""Core Data Models for PyBurgers.

This module defines the core data structures used throughout PyBurgers. These 
structures are implemented as Python `dataclasses` to provide a clear and 
robust way to manage the model's state and configuration.
"""

from dataclasses import dataclass

# --- Configuration Data Models ---

@dataclass(frozen=True)
class DNSConfig:
    """Direct numerical simulation configuration.
    
    Attributes:
        nx: Number of grid points.
    """
    nx: int

@dataclass(frozen=True)
class FFTWConfig:
    """FFTW parameters.
    
    Attributes:
        planning: fftw planning approach
        threads: number of threads to use
    """
    planning: str
    threads: int

@dataclass(frozen=True)
class LESConfig:
    """Large-eddy simulation configuration.
    
    Attributes:
        nx: Number of grid points.
        model: Integer ID for the subgrid-scale model to use.
    """
    nx: int
    model: int

@dataclass(frozen=True)
class ModelConfig:
    """Models configurations.

    Attributes:
        dns: Direct numerical simulation configuration
        les: Large-eddy simulation configuration
    """
    dns: DNSConfig
    les: LESConfig

@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings.

    Attributes:
        level: Logging level for the simulation (e.g., 'info', 'debug').
    """
    level: str

@dataclass(frozen=True)
    class NoiseConfig:
        """Noise method parameters.
        
        Attributes:
            alpha: FBM exponent controlling the spectral slope.
            amplitude: noise amplitude
        """
        alpha: float
        amplitude: float

@dataclass(frozen=True)
class NumericsConfig:
    """Numerical method parameters.
    
    Attributes:
        fftw = FFTW parameters
    """
    fftw: FFTWConfig

@dataclass(frozen=True)
class OutputConfig:
    """Output file configuration.

    Attributes:
        step_save: Frequency of save calls in time steps
    """
    step_save: int

@dataclass(frozen=True)
class PhysicsConfig:
    """Output file configuration.

    Attributes:
        noise: NoiseConfig configuration
        viscosity: The fluid's kinematic viscosity [m^2/s].
    """
    noise: NoiseConfig
    viscosity: int

@dataclass(frozen=True)
class TimeConfig:
    """Time-related parameters for the simulation.

    Attributes:
        nt: Number of time iterations.
        dt: Change in time between iterations [s].
    """
    nt: int
    dt: int
