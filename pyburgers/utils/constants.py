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
"""Defines constants for PyBurgers.

This module provides a centralized location for all constants used throughout
PyBurgers. Constants are grouped into logical namespaces.
"""
from types import SimpleNamespace

# Spectral algorithm constants
spectral = SimpleNamespace(
    DEALIAS_SCALE=3.0 / 2.0,
)

# SGS model constants
sgs = SimpleNamespace(
    TEST_FILTER_RATIO=2,
    SMAG_CONSTANT_CS=0.16,
    DEARDORFF_CE=0.70,
    DEARDORFF_C1=0.10,
    WONGLILLY_EXPONENT=4.0 / 3.0,
)
