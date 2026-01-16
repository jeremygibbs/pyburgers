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

"""PyBurgers: 1D Stochastic Burgers Equation Solver.

A tool for studying turbulence using DNS and LES.
"""

# Core data structures from the namelist file
from .data_models import (
    DNSConfig,
    DomainConfig,
    FFTWConfig,
    GridConfig,
    LESConfig,
    LoggingConfig,
    NoiseConfig,
    NumericsConfig,
    OutputConfig,
    PhysicsConfig,
    TimeConfig,
)

# Core I/O classes for handling model input and output
from .utils.io import Input, Output

# Exceptions
from .exceptions import InvalidMode, NamelistError, PyBurgersError

# Core and simulation classes
from .core import Burgers
from .dns import DNS
from .les import LES

__all__ = [
    # Data models
    'DNSConfig',
    'DomainConfig',
    'FFTWConfig',
    'GridConfig',
    'LESConfig',
    'LoggingConfig',
    'NoiseConfig',
    'NumericsConfig',
    'OutputConfig',
    'PhysicsConfig',
    'TimeConfig',
    # I/O
    'Input',
    'Output',
    # Exceptions
    'InvalidMode',
    'NamelistError',
    'PyBurgersError',
    # Core and simulation classes
    'Burgers',
    'DNS',
    'LES',
]
