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
"""pyBurgers SGS Physics Subpackage.

This module serves as the entry point for the subgrid-scale (SGS) model
component of the LES solver. It imports and exposes the main `SGS`
base class and factory method, making it accessible to the rest of the model.
"""

from .sgs import SGS

# Export factory method as module-level function
get_model = SGS.get_model

__all__ = ["SGS", "get_model"]
