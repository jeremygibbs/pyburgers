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

import logging
from collections.abc import Callable

import numpy as np

from ...utils import get_logger
from .temporal import TemporalIntegrator


class RK3(TemporalIntegrator):
    """Williamson (1980) low-storage 3rd-order Runge-Kutta.

    Uses two coefficient vectors for a three-stage integration with
    a single auxiliary storage array Q.

    RK3's stability region for real negative eigenvalues extends to
    |λ dt| ~ 2.5+, so the dissipative_stability_limit of 2.5 is set well
    above typical max_step · π⁴ values and will not constrain the time step
    in practice. The limit is retained for interface consistency.

    Attributes:
        dissipative_stability_limit: 2.5 — exceeds typical hyperviscous
            eigenvalue at max_step, so max_step governs instead.
        Q: Low-storage register array.
    """

    dissipative_stability_limit: float = 2.5

    _A = (0.0, -5.0 / 9.0, -153.0 / 128.0)
    _B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)

    def __init__(self, nx: int) -> None:
        """Initialize RK3 integrator.

        Args:
            nx: Number of grid points.
        """
        super().__init__(nx)
        self.logger: logging.Logger = get_logger("Temporal")
        self.logger.info("--- using Williamson RK3 time integration")
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
        for stage in range(3):
            rhs = compute_rhs()
            if stage == 0:
                self.Q.fill(0.0)           # avoid 0.0 * NaN poisoning
            else:
                self.Q *= self._A[stage]
            self.Q += rhs                  # in-place
            u += self._B[stage] * dt * self.Q

            # Zero Nyquist; must restore physical space on all stages so
            # SGS models see the Nyquist-zeroed u during intermediate stages.
            zero_nyquist(restore_physical=True)
