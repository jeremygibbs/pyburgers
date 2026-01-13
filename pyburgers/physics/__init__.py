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
abstract base class for subgrid-scale models, which defines the common
interface for different SGS parameterizations.

By importing the base class here, it is made directly accessible under
the `pyburgers.physics` namespace.
"""

from .sgs import SGS

__all__ = ['SGS']
