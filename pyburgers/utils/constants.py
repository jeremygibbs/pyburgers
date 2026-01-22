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
PyBurgers. Constants are grouped into logical namespaces using frozen dataclasses.

Attributes:
    spectral (SpectralConstants): Spectral algorithm constants
    sgs (SGSConstants): Subgrid-scale model constants
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralConstants:
    """Spectral algorithm constants.

    Attributes:
        DEALIAS_SCALE: Scale factor for dealiasing using 3/2 padding rule
    """

    DEALIAS_SCALE: float = 3.0 / 2.0


@dataclass(frozen=True)
class SGSConstants:
    """Subgrid-scale model constants.

    Attributes:
        TEST_FILTER_RATIO: Ratio for test filter width in dynamic models
        SMAG_CONSTANT_CS: Smagorinsky constant (Cs) for constant-coefficient model
        DEARDORFF_CE: Deardorff model constant for TKE dissipation
        DEARDORFF_C1: Deardorff model constant for eddy viscosity
        WONGLILLY_EXPONENT: Exponent for Wong-Lilly dynamic procedure (4/3)
    """

    TEST_FILTER_RATIO: int = 2
    SMAG_CONSTANT_CS: float = 0.16
    DEARDORFF_CE: float = 0.70
    DEARDORFF_C1: float = 0.10
    WONGLILLY_EXPONENT: float = 4.0 / 3.0


# Create singleton instances
spectral = SpectralConstants()
sgs = SGSConstants()
