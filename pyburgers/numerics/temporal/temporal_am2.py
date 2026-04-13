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
"""Adams-Moulton 2nd-order predictor-corrector time integrator for PyBurgers."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ...utils import get_logger
from .temporal import TemporalIntegrator


class AM2(TemporalIntegrator):
    """Adams-Bashforth 2nd-order predictor / Adams-Moulton 2nd-order corrector.

    Implements a PECE predictor-corrector scheme pairing the variable-step AB2
    predictor with the AM2 (trapezoidal rule) corrector:

        Predict:  u^* = u^n + dt·[(1 + ω/2)·F^n − (ω/2)·F^{n-1}]   (AB2, variable-step)
        Evaluate: F^* = F(u^*)
        Correct:  u^{n+1} = u^n + dt/2·(F^* + F^n)                   (AM2)
        Evaluate: F^{n+1} = F(u^{n+1})                               [implicit in next step]

    The variable-step AB2 predictor maintains 2nd-order accuracy under
    CFL-based adaptive time stepping. The AM2 corrector (trapezoidal rule)
    uses constant coefficients since it spans only the current interval and
    is A-stable as a standalone implicit method.

    The scheme reduces to the variable-step Euler bootstrap on the first
    timestep, matching the AB2 bootstrap strategy.

    Stability: The AM2 corrector is A-stable. In PECE mode the effective
    stability region for real negative eigenvalues is slightly wider than AB2
    alone, but the AB2 predictor still constrains the overall stability.
    The AM2 corrector's A-stability allows a larger dissipative_stability_limit
    (0.4) than standalone AB2 (0.1), since the corrector damps the parasitic
    root from the predictor.

    Attributes:
        dissipative_stability_limit: 0.4 — the A-stable corrector permits a
            larger limit than standalone AB2.
        _rhs_prev: RHS from the previous timestep (F^{n-1}), or None before
            the first step (triggers Euler bootstrap).
        _rhs_n: Pre-allocated buffer holding F^n, copied from compute_rhs()
            before the corrector evaluation overwrites the shared RHS buffer.
        _u_save: Pre-allocated buffer for u^n, restored before applying the
            corrector so the predictor modification does not persist.
        _dt_prev: Time step used on the previous timestep, or None before
            the first step.
    """

    dissipative_stability_limit: float = 0.4

    def __init__(self, nx: int) -> None:
        """Initialize AM2 predictor-corrector integrator.

        Args:
            nx: Number of grid points.
        """
        super().__init__(nx)
        self.logger: logging.Logger = get_logger("Temporal")
        self.logger.info("--- using Adams-Moulton 2nd-order time integration")
        self._rhs_prev: np.ndarray | None = None
        self._rhs_n = np.zeros(nx)
        self._u_save = np.zeros(nx)
        self._dt_prev: float | None = None

    def step(
        self,
        u: np.ndarray,
        dt: float,
        compute_rhs: Callable[[], np.ndarray],
        zero_nyquist: Callable[[bool], None],
    ) -> None:
        """Advance the solution by one timestep using AB2-AM2 predictor-corrector.

        Args:
            u: Velocity field array (modified in-place).
            dt: Time step size.
            compute_rhs: Callable that computes and returns the RHS vector.
                Returns a reference to a shared buffer; callers must copy
                before invoking compute_rhs a second time.
            zero_nyquist: Callable that zeros the Nyquist mode.
                Accepts a `restore_physical` boolean argument.
        """
        # Copy F^n before a second compute_rhs call overwrites the shared buffer
        self._rhs_n[:] = compute_rhs()

        if self._rhs_prev is None:
            # Bootstrap: forward Euler for the first step
            u += dt * self._rhs_n
            self._rhs_prev = self._rhs_n.copy()
        else:
            if self._dt_prev is None:
                raise RuntimeError("AM2 missing previous dt on non-bootstrap step")
            # Save u^n; the predictor modifies u in-place
            self._u_save[:] = u

            # AB2 predictor (variable-step): ω = dt_n / dt_{n-1}
            # Reuse _rhs_prev as scratch (overwritten with F^n below).
            omega = dt / self._dt_prev
            c0 = 1.0 + 0.5 * omega
            c1 = 0.5 * omega
            self._rhs_prev *= -c1           # in-place: -c1 · F^{n-1}
            self._rhs_prev += c0 * self._rhs_n  # in-place: c0·F^n - c1·F^{n-1}
            u += dt * self._rhs_prev

            # Must restore physical space so SGS models see Nyquist-zeroed u.
            zero_nyquist(restore_physical=True)

            # Evaluate F^* at the predicted u^* (fu already current, FFT skipped)
            rhs_star = compute_rhs()

            # AM2 corrector: u^{n+1} = u^n + dt/2 · (F^* + F^n)
            u[:] = self._u_save
            u += 0.5 * dt * rhs_star
            u += 0.5 * dt * self._rhs_n

            # Store F^n as F^{n-1} for the next step's AB2 predictor
            self._rhs_prev[:] = self._rhs_n

        self._dt_prev = dt
        zero_nyquist(restore_physical=True)
