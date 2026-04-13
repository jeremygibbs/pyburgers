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
"""PyBurgers Temporal Integration Subpackage."""

from .temporal import TemporalIntegrator

get_integrator = TemporalIntegrator.get_integrator

__all__ = ["TemporalIntegrator", "get_integrator"]
