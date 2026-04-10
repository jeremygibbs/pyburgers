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
"""Tests for spatial discretization operators."""

from __future__ import annotations

import numpy as np
import pytest

from pyburgers.numerics.spatial.spatial import SpatialOperator
from pyburgers.numerics.spatial.spatial_fd2 import FD2
from pyburgers.numerics.spatial.spatial_fd4 import FD4
from pyburgers.numerics.spatial.spatial_spectral import Spectral
from pyburgers.utils.spectral_workspace import SpectralWorkspace


@pytest.fixture
def workspace_small() -> SpectralWorkspace:
    """Small spectral workspace for unit tests."""
    nx = 64
    dx = 2 * np.pi / nx
    return SpectralWorkspace(nx=nx, dx=dx, fftw_planning="FFTW_ESTIMATE", fftw_threads=1)


@pytest.fixture
def workspace_medium() -> SpectralWorkspace:
    """Medium spectral workspace for convergence tests."""
    nx = 256
    dx = 2 * np.pi / nx
    return SpectralWorkspace(nx=nx, dx=dx, fftw_planning="FFTW_ESTIMATE", fftw_threads=1)


class TestSpatialFactory:
    """Tests for the SpatialOperator factory method."""

    def test_scheme_1_returns_fd2(self, workspace_small: SpectralWorkspace) -> None:
        """Test that scheme 1 returns an FD2 instance."""
        op = SpatialOperator.get_operator(1, 64, 2 * np.pi / 64, workspace_small)
        assert isinstance(op, FD2)

    def test_scheme_2_returns_fd4(self, workspace_small: SpectralWorkspace) -> None:
        """Test that scheme 2 returns an FD4 instance."""
        op = SpatialOperator.get_operator(2, 64, 2 * np.pi / 64, workspace_small)
        assert isinstance(op, FD4)

    def test_scheme_3_returns_spectral(self, workspace_small: SpectralWorkspace) -> None:
        """Test that scheme 3 returns a Spectral instance."""
        op = SpatialOperator.get_operator(3, 64, 2 * np.pi / 64, workspace_small)
        assert isinstance(op, Spectral)

    def test_invalid_scheme_raises(self, workspace_small: SpectralWorkspace) -> None:
        """Test that an invalid scheme ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown spatial scheme ID"):
            SpatialOperator.get_operator(99, 64, 2 * np.pi / 64, workspace_small)


class TestEigenvalues:
    """Tests for scheme-specific eigenvalue attributes."""

    def test_fd2_eigenvalues(self) -> None:
        """Test FD2 modified wavenumber eigenvalues."""
        assert FD2.viscous_eigenvalue == 4.0
        assert FD2.hyperviscous_eigenvalue == 16.0

    def test_fd4_eigenvalues(self) -> None:
        """Test FD4 modified wavenumber eigenvalues."""
        assert FD4.viscous_eigenvalue == pytest.approx(16.0 / 3.0)
        assert FD4.hyperviscous_eigenvalue == pytest.approx(80.0 / 3.0)

    def test_spectral_eigenvalues(self) -> None:
        """Test Spectral modified wavenumber eigenvalues."""
        assert Spectral.viscous_eigenvalue == pytest.approx(np.pi**2)
        assert Spectral.hyperviscous_eigenvalue == pytest.approx(np.pi**4)


class TestFirstDerivative:
    """Test d/dx(sin(x)) = cos(x) for each spatial scheme."""

    @staticmethod
    def _compute_first_deriv(
        scheme_id: int, nx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1st derivative of sin(x) and return (result, expected)."""
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)
        ws = SpectralWorkspace(nx=nx, dx=dx, fftw_planning="FFTW_ESTIMATE", fftw_threads=1)
        op = SpatialOperator.get_operator(scheme_id, nx, dx, ws)
        u = np.sin(x)
        result = op.compute(u, [1])
        expected = np.cos(x)
        return result["1"], expected

    def test_fd2_first_derivative(self) -> None:
        """Test FD2 1st derivative of sin(x) (2nd-order accuracy)."""
        result, expected = self._compute_first_deriv(1, 256)
        np.testing.assert_allclose(result, expected, atol=5e-4)

    def test_fd4_first_derivative(self) -> None:
        """Test FD4 1st derivative of sin(x) (4th-order accuracy)."""
        result, expected = self._compute_first_deriv(2, 256)
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_spectral_first_derivative(self) -> None:
        """Test Spectral 1st derivative of sin(x) (machine precision)."""
        result, expected = self._compute_first_deriv(3, 64)
        np.testing.assert_allclose(result, expected, atol=1e-13)


class TestSecondDerivative:
    """Test d2/dx2(sin(x)) = -sin(x) for each spatial scheme."""

    @staticmethod
    def _compute_second_deriv(
        scheme_id: int, nx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 2nd derivative of sin(x) and return (result, expected)."""
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)
        ws = SpectralWorkspace(nx=nx, dx=dx, fftw_planning="FFTW_ESTIMATE", fftw_threads=1)
        op = SpatialOperator.get_operator(scheme_id, nx, dx, ws)
        u = np.sin(x)
        result = op.compute(u, [2])
        expected = -np.sin(x)
        return result["2"], expected

    def test_fd2_second_derivative(self) -> None:
        """Test FD2 2nd derivative of sin(x) (2nd-order accuracy)."""
        result, expected = self._compute_second_deriv(1, 256)
        np.testing.assert_allclose(result, expected, atol=5e-3)

    def test_fd4_second_derivative(self) -> None:
        """Test FD4 2nd derivative of sin(x) (4th-order accuracy)."""
        result, expected = self._compute_second_deriv(2, 256)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_spectral_second_derivative(self) -> None:
        """Test Spectral 2nd derivative of sin(x) (machine precision)."""
        result, expected = self._compute_second_deriv(3, 64)
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestNyquistZeroing:
    """Test zero_nyquist behavior for each scheme."""

    def test_fd2_zero_nyquist_no_crash(self, workspace_small: SpectralWorkspace) -> None:
        """Test that FD2 zero_nyquist does not error."""
        nx = 64
        dx = 2 * np.pi / nx
        op = FD2(nx, dx, workspace_small)
        # Should not raise; FD2 only acts when restore_physical=True
        op.zero_nyquist(restore_physical=True)
        op.zero_nyquist(restore_physical=False)

    def test_spectral_zero_nyquist(self, workspace_small: SpectralWorkspace) -> None:
        """Test that Spectral zero_nyquist zeros the Nyquist mode."""
        nx = 64
        dx = 2 * np.pi / nx
        op = Spectral(nx, dx, workspace_small)
        # Put something in the Nyquist mode
        u = np.sin(np.arange(0, 2 * np.pi, dx))
        workspace_small.derivatives.compute(u, [1])
        op.zero_nyquist(restore_physical=True)
        # After zeroing, the Nyquist coefficient should be 0
        fu = workspace_small.derivatives.fu
        assert fu[-1] == 0.0


class TestDealiasedSquaredTerm:
    """Test the dealiased d(u^2)/dx computation for FD schemes."""

    def test_fd2_squared_term(self) -> None:
        """Test FD2 computes d(u^2)/dx."""
        nx = 256
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)
        ws = SpectralWorkspace(nx=nx, dx=dx, fftw_planning="FFTW_ESTIMATE", fftw_threads=1)
        op = FD2(nx, dx, ws)
        u = np.sin(x)
        result = op.compute(u, ["sq"])
        # d(sin^2(x))/dx = 2*sin(x)*cos(x) = sin(2x)
        expected = np.sin(2 * x)
        np.testing.assert_allclose(result["sq"], expected, atol=5e-3)
