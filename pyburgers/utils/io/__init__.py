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
"""PyBurgers Input/Output Subpackage.

This module serves as the entry point for the I/O utilities. It imports
and exposes the main `Input` and `Output` classes, making them directly
accessible under the `pyburgers.utils.io` namespace for convenience.
"""

from .input import Input
from .output import Output

__all__ = ["Input", "Output"]
