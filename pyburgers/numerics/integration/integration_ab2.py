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
"""Adams-Bashforth 2nd-order time integrator for PyBurgers."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .integration import TimeIntegrator


class AB2(TimeIntegrator):
    """Adams-Bashforth 2nd-order multistep method.

    Uses the standard constant-dt AB2 formula:
        u^{n+1} = u^n + dt * (3/2 * F^n - 1/2 * F^{n-1})

    The first timestep is bootstrapped with forward Euler since no
    previous RHS is available.

    Note: The standard AB2 coefficients assume constant dt. With
    CFL-based adaptive time stepping the dt changes slowly, so
    the error introduced is small and acceptable for educational use.

    Attributes:
        _rhs_prev: RHS from the previous timestep, or None before
            the first step (triggers Euler bootstrap).
    """

    def __init__(self, nx: int) -> None:
        """Initialize AB2 integrator.

        Args:
            nx: Number of grid points.
        """
        super().__init__(nx)
        self._rhs_prev: np.ndarray | None = None

    def step(
        self,
        u: np.ndarray,
        dt: float,
        compute_rhs: Callable[[], np.ndarray],
        zero_nyquist: Callable[[bool], None],
    ) -> None:
        """Advance the solution by one timestep using AB2.

        Args:
            u: Velocity field array (modified in-place).
            dt: Time step size.
            compute_rhs: Callable that computes and returns the RHS vector.
            zero_nyquist: Callable that zeros the Nyquist mode.
        """
        rhs = compute_rhs()

        if self._rhs_prev is None:
            # Bootstrap: forward Euler for the first step
            u[:] = u + dt * rhs
            self._rhs_prev = rhs.copy()
        else:
            # AB2: u^{n+1} = u^n + dt * (3/2 * F^n - 1/2 * F^{n-1})
            u[:] = u + dt * (1.5 * rhs - 0.5 * self._rhs_prev)
            self._rhs_prev[:] = rhs

        zero_nyquist(restore_physical=True)
