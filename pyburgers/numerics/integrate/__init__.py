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
"""PyBurgers Time Integration Subpackage.

This module serves as the entry point for the time integration component
of the solver. It imports and exposes the main `TimeIntegrator` base class
and factory method, making them accessible to the rest of the model.
"""

from .integrate import TimeIntegrator

# Export factory method as module-level function
get_integrator = TimeIntegrator.get_integrator

__all__ = ["TimeIntegrator", "get_integrator"]
