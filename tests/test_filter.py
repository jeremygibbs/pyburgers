"""Tests for spectral filtering operations."""
from __future__ import annotations

import numpy as np
import pytest

from pyburgers.utils import Filter


class TestFilter:
    """Test cases for the Filter class."""

    def test_cutoff_preserves_low_frequencies(self, grid_small: dict) -> None:
        """Test that cutoff filter preserves low frequencies."""
        filt = Filter(grid_small["nx"])
        # Low frequency signal (k=1) should pass through
        u = np.sin(grid_small["x"])
        result = filt.cutoff(u, ratio=4)

        np.testing.assert_allclose(result, u, rtol=1e-10, atol=1e-14)

    def test_cutoff_removes_high_frequencies(self, grid_small: dict) -> None:
        """Test that cutoff filter removes high frequencies."""
        filt = Filter(grid_small["nx"])
        k_high = grid_small["nx"] // 4  # High frequency
        u_high = np.sin(k_high * grid_small["x"])
        result = filt.cutoff(u_high, ratio=2)

        # High frequency should be almost completely removed
        assert np.max(np.abs(result)) < 1e-10

    def test_cutoff_mixed_frequencies(self, grid_small: dict) -> None:
        """Test filtering signal with mixed frequencies."""
        filt = Filter(grid_small["nx"])
        k_low = 2
        k_high = grid_small["nx"] // 4

        u_low = np.sin(k_low * grid_small["x"])
        u_high = np.sin(k_high * grid_small["x"])
        u_mixed = u_low + u_high

        result = filt.cutoff(u_mixed, ratio=4)

        # Result should be close to just the low frequency component
        np.testing.assert_allclose(result, u_low, rtol=1e-6, atol=1e-10)

    def test_downscale_preserves_amplitude(self) -> None:
        """Test that downscale preserves signal amplitude."""
        nx_les = 64
        nx_dns = 256
        ratio = nx_dns // nx_les

        dx_dns = 2 * np.pi / nx_dns
        x_dns = np.arange(0, 2 * np.pi, dx_dns)
        dx_les = 2 * np.pi / nx_les
        x_les = np.arange(0, 2 * np.pi, dx_les)

        filt = Filter(nx_les, nx2=nx_dns)

        # Low frequency signal at DNS resolution
        u_dns = np.sin(x_dns)
        result = filt.downscale(u_dns, ratio)

        # Compare to signal at LES resolution
        expected = np.sin(x_les)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-14)

    def test_downscale_removes_high_frequencies(self) -> None:
        """Test that downscale removes unresolved frequencies."""
        nx_les = 64
        nx_dns = 256
        ratio = nx_dns // nx_les

        dx_dns = 2 * np.pi / nx_dns
        x_dns = np.arange(0, 2 * np.pi, dx_dns)

        filt = Filter(nx_les, nx2=nx_dns)

        # High frequency signal that cannot be resolved on LES grid
        k_high = nx_les  # At Nyquist of LES
        u_dns = np.sin(k_high * x_dns)
        result = filt.downscale(u_dns, ratio)

        # High frequency should be removed
        assert np.max(np.abs(result)) < 1e-10

    def test_downscale_no_stale_data(self) -> None:
        """Test that downscale zeroes array properly between calls."""
        nx_les = 64
        nx_dns = 256
        ratio = nx_dns // nx_les

        dx_dns = 2 * np.pi / nx_dns
        x_dns = np.arange(0, 2 * np.pi, dx_dns)
        dx_les = 2 * np.pi / nx_les
        x_les = np.arange(0, 2 * np.pi, dx_les)

        filt = Filter(nx_les, nx2=nx_dns)

        # First call with large signal
        u1 = 10 * np.sin(x_dns)
        _ = filt.downscale(u1, ratio)

        # Second call with small signal
        u2 = 0.1 * np.sin(x_dns)
        result = filt.downscale(u2, ratio)

        expected = 0.1 * np.sin(x_les)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-14)
