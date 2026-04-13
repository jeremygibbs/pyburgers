#!/usr/bin/env python
#
# PyBurgers
#
# Copyright (c) 2017–2026 Jeremy A. Gibbs
#
# This file is part of PyBurgers.
#
# This software is free and is distributed under the MIT License.
# See accompanying LICENSE file or visit https://opensource.org/licenses/MIT.
#

"""PyBurgers Numerics Subpackage.

This package provides the numerical method abstractions for PyBurgers,
including temporal integration schemes and spatial discretization operators.
"""

from .spatial import SpatialOperator, get_operator
from .temporal import TemporalIntegrator, get_integrator

__all__ = ["SpatialOperator", "get_operator", "TemporalIntegrator", "get_integrator"]
