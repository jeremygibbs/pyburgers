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
"""Custom exception types for PyBurgers."""


class PyBurgersError(Exception):
    """Base class for all custom exceptions in PyBurgers."""


class NamelistError(PyBurgersError):
    """Raised for errors found in the namelist configuration."""


class InvalidMode(PyBurgersError):
    """Raised when an invalid simulation mode is specified."""
