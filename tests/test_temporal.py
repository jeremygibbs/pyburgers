#!/usr/bin/env python
#
# PyBurgers
#
# Copyright (c) 2017-2026 Jeremy A. Gibbs
#
# This file is part of PyBurgers.
#
# This software is free and is distributed under the MIT License.
# See accompanying LICENSE file or visit https://opensource.org/licenses/MIT.
#
"""Tests for temporal integration schemes."""

from __future__ import annotations

import numpy as np
import pytest

from pyburgers.numerics.temporal.temporal import TemporalIntegrator
from pyburgers.numerics.temporal.temporal_ab2 import AB2
from pyburgers.numerics.temporal.temporal_am2 import AM2
from pyburgers.numerics.temporal.temporal_rk3 import RK3


class TestTemporalFactory:
    """Tests for the TemporalIntegrator factory method."""

    def test_scheme_1_returns_ab2(self) -> None:
        """Test that scheme 1 returns an AB2 instance."""
        integrator = TemporalIntegrator.get_integrator(1, 64)
        assert isinstance(integrator, AB2)

    def test_scheme_2_returns_am2(self) -> None:
        """Test that scheme 2 returns an AM2 instance."""
        integrator = TemporalIntegrator.get_integrator(2, 64)
        assert isinstance(integrator, AM2)

    def test_scheme_3_returns_rk3(self) -> None:
        """Test that scheme 3 returns an RK3 instance."""
        integrator = TemporalIntegrator.get_integrator(3, 64)
        assert isinstance(integrator, RK3)

    def test_invalid_scheme_raises(self) -> None:
        """Test that an invalid scheme ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown time integrator ID"):
            TemporalIntegrator.get_integrator(99, 64)


class TestStabilityLimits:
    """Tests for integrator stability limit attributes."""

    def test_ab2_stability_limit(self) -> None:
        """Test AB2 dissipative stability limit."""
        integrator = TemporalIntegrator.get_integrator(1, 64)
        assert integrator.dissipative_stability_limit == 0.2

    def test_am2_stability_limit(self) -> None:
        """Test AM2 dissipative stability limit."""
        integrator = TemporalIntegrator.get_integrator(2, 64)
        assert integrator.dissipative_stability_limit == 0.4

    def test_rk3_stability_limit(self) -> None:
        """Test RK3 dissipative stability limit."""
        integrator = TemporalIntegrator.get_integrator(3, 64)
        assert integrator.dissipative_stability_limit == 2.5


class TestDecayODE:
    """Test convergence on du/dt = -u (exact: u(t) = u0 * exp(-t)).

    Each integrator steps the decay ODE forward over a fixed interval
    and the result is compared to the analytical solution.
    """

    @staticmethod
    def _run_decay(
        integrator: TemporalIntegrator, nx: int, dt: float, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step the decay ODE and return (numerical, exact) solutions."""
        u = np.ones(nx)
        u0 = u.copy()

        def compute_rhs() -> np.ndarray:
            return -u.copy()

        def zero_nyquist(restore_physical: bool = True) -> None:
            pass

        for _ in range(n_steps):
            integrator.step(u, dt, compute_rhs, zero_nyquist)

        t_final = dt * n_steps
        exact = u0 * np.exp(-t_final)
        return u, exact

    def test_rk3_convergence(self) -> None:
        """Test RK3 converges on du/dt = -u."""
        nx = 4
        dt = 0.01
        n_steps = 100
        integrator = RK3(nx)
        u, exact = self._run_decay(integrator, nx, dt, n_steps)
        np.testing.assert_allclose(u, exact, rtol=1e-6)

    def test_ab2_convergence(self) -> None:
        """Test AB2 converges on du/dt = -u (with Euler bootstrap)."""
        nx = 4
        dt = 0.01
        n_steps = 100
        integrator = AB2(nx)
        u, exact = self._run_decay(integrator, nx, dt, n_steps)
        np.testing.assert_allclose(u, exact, rtol=1e-4)

    def test_am2_convergence(self) -> None:
        """Test AM2 converges on du/dt = -u (with Euler bootstrap)."""
        nx = 4
        dt = 0.01
        n_steps = 100
        integrator = AM2(nx)
        u, exact = self._run_decay(integrator, nx, dt, n_steps)
        np.testing.assert_allclose(u, exact, rtol=1e-4)

    def test_ab2_first_step_is_euler(self) -> None:
        """Test that AB2's first step uses forward Euler bootstrap."""
        nx = 4
        dt = 0.1
        u = np.ones(nx) * 2.0

        integrator = AB2(nx)

        def compute_rhs() -> np.ndarray:
            return -u.copy()

        def zero_nyquist(restore_physical: bool = True) -> None:
            pass

        integrator.step(u, dt, compute_rhs, zero_nyquist)
        # Forward Euler: u_new = u_old + dt * (-u_old) = 2.0 + 0.1 * (-2.0) = 1.8
        np.testing.assert_allclose(u, 1.8, rtol=1e-14)

    def test_am2_performs_two_rhs_evaluations(self) -> None:
        """Test that AM2 calls compute_rhs twice per step (after bootstrap)."""
        nx = 4
        dt = 0.01
        u = np.ones(nx)
        integrator = AM2(nx)
        call_count = 0

        def compute_rhs() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return -u.copy()

        def zero_nyquist(restore_physical: bool = True) -> None:
            pass

        # First step: bootstrap (1 RHS eval from _rhs_n copy + the initial compute)
        integrator.step(u, dt, compute_rhs, zero_nyquist)
        bootstrap_calls = call_count

        # Second step: full PECE (should call compute_rhs twice: F^n and F^*)
        call_count = 0
        integrator.step(u, dt, compute_rhs, zero_nyquist)
        assert call_count == 2, f"AM2 should call compute_rhs twice, got {call_count}"
        assert bootstrap_calls == 1
