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
"""Williamson (1980) low-storage RK3 time integrator for PyBurgers."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .integrate import TimeIntegrator


class RK3(TimeIntegrator):
    """Williamson (1980) low-storage 3rd-order Runge-Kutta.

    Uses two coefficient vectors for a three-stage integration with
    a single auxiliary storage array Q.

    Attributes:
        Q: Low-storage register array.
    """

    _A = (0.0, -5.0 / 9.0, -153.0 / 128.0)
    _B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)

    def __init__(self, nx: int) -> None:
        """Initialize RK3 integrator.

        Args:
            nx: Number of grid points.
        """
        super().__init__(nx)
        self.Q = np.zeros(nx)

    def step(
        self,
        u: np.ndarray,
        dt: float,
        compute_rhs: Callable[[], np.ndarray],
        zero_nyquist: Callable[[bool], None],
    ) -> None:
        """Advance the solution by one timestep using 3-stage RK3.

        Args:
            u: Velocity field array (modified in-place).
            dt: Time step size.
            compute_rhs: Callable that computes and returns the RHS vector.
            zero_nyquist: Callable that zeros the Nyquist mode.
        """
        self.Q[:] = 0.0
        for stage in range(3):
            rhs = compute_rhs()
            self.Q[:] = self._A[stage] * self.Q + rhs
            u[:] = u + self._B[stage] * dt * self.Q

            # Zero Nyquist; restore physical space only on final stage
            zero_nyquist(restore_physical=(stage == 2))
