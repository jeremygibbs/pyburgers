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

from collections.abc import Callable

import numpy as np

from .integration import TimeIntegrator


class AB2(TimeIntegrator):
    """Adams-Bashforth 2nd-order multistep method.

    Uses the variable-step AB2 formula to maintain 2nd-order accuracy
    under CFL-based adaptive time stepping:

        ω = dt_n / dt_{n-1}
        u^{n+1} = u^n + dt_n · [(1 + ω/2) · F^n  −  (ω/2) · F^{n-1}]

    Reduces to the standard (3/2, -1/2) coefficients when dt is constant.
    The first timestep is bootstrapped with forward Euler since no
    previous RHS is available.

    Attributes:
        _rhs_prev: RHS from the previous timestep, or None before
            the first step (triggers Euler bootstrap).
        _dt_prev: Time step used on the previous timestep, or None before
            the first step.
    """

    def __init__(self, nx: int) -> None:
        """Initialize AB2 integrator.

        Args:
            nx: Number of grid points.
        """
        super().__init__(nx)
        self._rhs_prev: np.ndarray | None = None
        self._dt_prev: float | None = None

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
            # Variable-step AB2: ω = dt_n / dt_{n-1}
            omega = dt / self._dt_prev
            c0 = 1.0 + 0.5 * omega
            c1 = 0.5 * omega
            u[:] = u + dt * (c0 * rhs - c1 * self._rhs_prev)
            self._rhs_prev[:] = rhs

        self._dt_prev = dt
        zero_nyquist(restore_physical=True)
