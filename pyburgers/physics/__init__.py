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

"""PyBurgers Physics Subpackage.

This package provides the core physics modules for PyBurgers. It contains the 
abstract base classes for noise and subgrid-scale models, which define the 
common interface for different physics parameterizations.

By importing the base classes here, they are made directly accessible under
the `pyburgers.physics` namespace.
"""

from .noise import Noise
from .sgs import SGS

__all__ = ['Noise', 'SGS']
