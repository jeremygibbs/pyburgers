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
        points: Number of grid points.
    """

    points: int


@dataclass(frozen=True)
class FFTWConfig:
    """FFTW parameters.

    Attributes:
        planning: FFTW planning approach.
        threads: Number of threads to use.
    """

    planning: str
    threads: int


@dataclass(frozen=True)
class LESConfig:
    """Large-eddy simulation configuration.

    Attributes:
        points: Number of grid points.
    """

    points: int


@dataclass(frozen=True)
class GridConfig:
    """Grid configurations.

    Attributes:
        length: Domain length (periodic) [m].
        dns: Direct numerical simulation grid configuration.
        les: Large-eddy simulation grid configuration.
    """

    length: float
    dns: DNSConfig
    les: LESConfig


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings.

    Attributes:
        level: Logging level for the simulation (e.g., 'info', 'debug').
        file: Optional log file path for file logging.
    """

    level: str
    file: str | None = None


@dataclass(frozen=True)
class HyperviscosityConfig:
    """Hyperviscosity parameters for high-k damping.

    Adds a -ν₄∇⁴u term that provides k⁴ dissipation at high wavenumbers
    to prevent spectral pile-up near the Nyquist frequency.

    When enabled, the coefficient is auto-computed as ν₄ = dx⁴ to provide
    appropriate damping that scales correctly with grid resolution and
    does not limit the simulation timestep.

    Attributes:
        enabled: Whether hyperviscosity is enabled.
    """

    enabled: bool = False


@dataclass(frozen=True)
class NoiseConfig:
    """Noise method parameters.

    Attributes:
        exponent: FBM exponent controlling the spectral slope.
        amplitude: Noise amplitude.
    """

    exponent: float
    amplitude: float


@dataclass(frozen=True)
class OutputConfig:
    """Output file configuration.

    Attributes:
        interval_save: Save interval in physical time [s].
        interval_print: Print progress interval in physical time [s].
    """

    interval_save: float
    interval_print: float


@dataclass(frozen=True)
class PhysicsConfig:
    """Physics configuration.

    Attributes:
        noise: NoiseConfig configuration.
        viscosity: The fluid's kinematic viscosity [m^2/s].
        subgrid_model: Subgrid-scale model ID (0-4) for LES.
        hyperviscosity: HyperviscosityConfig for high-k damping.
    """

    noise: NoiseConfig
    viscosity: float
    subgrid_model: int
    hyperviscosity: HyperviscosityConfig = HyperviscosityConfig()


@dataclass(frozen=True)
class TimeConfig:
    """Time-related parameters for the simulation.

    Attributes:
        duration: Total simulation time [s].
        cfl: Target CFL number for adaptive time stepping.
        max_step: Maximum allowed time step [s].
    """

    duration: float
    cfl: float
    max_step: float
