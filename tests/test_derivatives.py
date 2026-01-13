"""Tests for spectral derivative calculations."""
from __future__ import annotations

import numpy as np
import pytest

from pyburgers.utils import Derivatives


class TestDerivatives:
    """Test cases for the Derivatives class."""

    def test_first_derivative_sine(self, grid_small: dict) -> None:
        """Test d/dx(sin(x)) = cos(x)."""
        derivs = Derivatives(grid_small["nx"], grid_small["dx"])
        u = np.sin(grid_small["x"])
        result = derivs.compute(u, [1])

        expected = np.cos(grid_small["x"])
        np.testing.assert_allclose(result["1"], expected, rtol=1e-10, atol=1e-14)

    def test_first_derivative_cosine(self, grid_small: dict) -> None:
        """Test d/dx(cos(x)) = -sin(x)."""
        derivs = Derivatives(grid_small["nx"], grid_small["dx"])
        u = np.cos(grid_small["x"])
        result = derivs.compute(u, [1])

        expected = -np.sin(grid_small["x"])
        np.testing.assert_allclose(result["1"], expected, rtol=1e-10, atol=1e-14)

    def test_second_derivative_sine(self, grid_small: dict) -> None:
        """Test d2/dx2(sin(x)) = -sin(x)."""
        derivs = Derivatives(grid_small["nx"], grid_small["dx"])
        u = np.sin(grid_small["x"])
        result = derivs.compute(u, [2])

        expected = -np.sin(grid_small["x"])
        np.testing.assert_allclose(result["2"], expected, rtol=1e-10, atol=1e-13)

    def test_third_derivative_sine(self, grid_small: dict) -> None:
        """Test d3/dx3(sin(x)) = -cos(x)."""
        derivs = Derivatives(grid_small["nx"], grid_small["dx"])
        u = np.sin(grid_small["x"])
        result = derivs.compute(u, [3])

        expected = -np.cos(grid_small["x"])
        np.testing.assert_allclose(result["3"], expected, rtol=1e-10, atol=1e-11)

    def test_multiple_derivatives(self, grid_small: dict) -> None:
        """Test computing multiple derivatives at once."""
        derivs = Derivatives(grid_small["nx"], grid_small["dx"])
        u = np.sin(grid_small["x"])
        result = derivs.compute(u, [1, 2, 3])

        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_squared_derivative(self, grid_small: dict) -> None:
        """Test dealiased d(u^2)/dx computation."""
        derivs = Derivatives(grid_small["nx"], grid_small["dx"])
        u = np.sin(grid_small["x"])
        result = derivs.compute(u, ["sq"])

        # d(sin^2(x))/dx = 2*sin(x)*cos(x) = sin(2x)
        expected = np.sin(2 * grid_small["x"])
        # Lower tolerance due to dealiasing approximation
        np.testing.assert_allclose(result["sq"], expected, rtol=1e-6, atol=1e-10)

    def test_higher_wavenumber(self, grid_medium: dict) -> None:
        """Test derivative accuracy for higher wavenumber signal."""
        k = 5  # wavenumber
        derivs = Derivatives(grid_medium["nx"], grid_medium["dx"])
        u = np.sin(k * grid_medium["x"])
        result = derivs.compute(u, [1])

        expected = k * np.cos(k * grid_medium["x"])
        np.testing.assert_allclose(result["1"], expected, rtol=1e-10, atol=1e-14)
