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

This package provides the numerical method abstractions for PyBurgers.
It contains the time integration schemes, with the structure designed
to accommodate future spatial discretization abstractions.
"""

from .integrate import TimeIntegrator, get_integrator

__all__ = ["TimeIntegrator", "get_integrator"]
