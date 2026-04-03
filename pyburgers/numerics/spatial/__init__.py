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
"""Spatial discretization schemes for PyBurgers."""

from .spatial import SpatialOperator

get_operator = SpatialOperator.get_operator

__all__ = ["SpatialOperator", "get_operator"]
