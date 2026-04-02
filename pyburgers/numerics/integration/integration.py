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
"""Time integration base class for PyBurgers.

This module provides the base TimeIntegrator class and factory method for
creating different time integration schemes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class TimeIntegrator(ABC):
    """Base class for time integration schemes.

    Provides the interface and factory method for time integrators used
    by the Burgers equation solver. Subclasses implement the `step` method
    to advance the solution by one timestep.

    Attributes:
        nx: Number of grid points.
    """

    @staticmethod
    def get_integrator(scheme: int, nx: int) -> TimeIntegrator:
        """Factory method to create the appropriate time integrator.

        Args:
            scheme: Time integration scheme identifier.
                1 = Williamson low-storage RK3
                2 = Adams-Bashforth 2nd order
            nx: Number of grid points for pre-allocating storage arrays.

        Returns:
            Instance of the requested TimeIntegrator subclass.
        """
        if scheme == 1:
            from .integration_rk3 import RK3

            return RK3(nx)
        if scheme == 2:
            from .integration_ab2 import AB2

            return AB2(nx)
        raise ValueError(f"--- Unknown time integrator ID: {scheme}. Valid options: 1-2.")

    def __init__(self, nx: int) -> None:
        """Initialize the time integrator.

        Args:
            nx: Number of grid points.
        """
        self.nx = nx

    @abstractmethod
    def step(
        self,
        u: np.ndarray,
        dt: float,
        compute_rhs: Callable[[], np.ndarray],
        zero_nyquist: Callable[[bool], None],
    ) -> None:
        """Advance the solution by one timestep.

        Args:
            u: Velocity field array (modified in-place).
            dt: Time step size.
            compute_rhs: Callable that computes and returns the RHS vector.
            zero_nyquist: Callable that zeros the Nyquist mode.
                Accepts a `restore_physical` boolean argument.
        """
