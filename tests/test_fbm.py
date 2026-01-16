"""Tests for fractional Brownian motion noise generation."""
##from __future__ import annotations

import numpy as np
import pytest

from pyburgers.utils import FBM


class TestFBM:
    """Test cases for the FBM class."""

    def test_noise_shape(self, grid_small: dict) -> None:
        """Test that noise has correct shape."""
        fbm = FBM(0.75, grid_small["nx"])
        noise = fbm.compute_noise()

        assert noise.shape == (grid_small["nx"],)

    def test_noise_is_real(self, grid_small: dict) -> None:
        """Test that noise output is real-valued."""
        fbm = FBM(0.75, grid_small["nx"])
        noise = fbm.compute_noise()

        assert np.all(np.isreal(noise))

    def test_noise_zero_mean(self, grid_small: dict) -> None:
        """Test that noise has approximately zero mean."""
        np.random.seed(42)
        fbm = FBM(0.75, grid_small["nx"])

        # Average over multiple realizations
        means = []
        for _ in range(100):
            noise = fbm.compute_noise()
            means.append(np.mean(noise))

        # Mean of means should be close to zero
        assert abs(np.mean(means)) < 0.5

    def test_noise_finite(self, grid_small: dict) -> None:
        """Test that noise values are finite."""
        fbm = FBM(0.75, grid_small["nx"])
        noise = fbm.compute_noise()

        assert np.all(np.isfinite(noise))

    def test_noise_different_realizations(self, grid_small: dict) -> None:
        """Test that consecutive calls produce different noise."""
        np.random.seed(42)
        fbm = FBM(0.75, grid_small["nx"])

        noise1 = fbm.compute_noise().copy()
        noise2 = fbm.compute_noise()

        # Noise realizations should be different
        assert not np.allclose(noise1, noise2)

    def test_spectral_slope(self) -> None:
        """Test that noise has correct spectral slope."""
        np.random.seed(42)
        nx = 256
        alpha = 0.75
        fbm = FBM(alpha, nx)

        # Average power spectrum over many realizations
        n_realizations = 100
        power_sum = np.zeros(nx // 2)

        for _ in range(n_realizations):
            noise = fbm.compute_noise()
            spectrum = np.abs(np.fft.fft(noise)[:nx // 2]) ** 2
            power_sum += spectrum

        power_avg = power_sum / n_realizations

        # Fit spectral slope in log-log space (avoid DC and Nyquist)
        k = np.arange(2, nx // 4)
        log_k = np.log(k)
        log_power = np.log(power_avg[2:nx // 4])

        # Linear regression for slope
        slope, _ = np.polyfit(log_k, log_power, 1)

        # Slope should be approximately -alpha (within tolerance)
        # Power spectrum scales as k^(-alpha) for FBM
        assert abs(slope + alpha) < 0.3  # Some tolerance for finite samples

    def test_different_alpha(self) -> None:
        """Test that different alpha values produce different spectra."""
        nx = 128
        np.random.seed(42)

        fbm_low = FBM(0.5, nx)
        fbm_high = FBM(1.0, nx)

        # Generate multiple realizations and compare variance at high-k
        variances_low = []
        variances_high = []

        for _ in range(50):
            noise_low = fbm_low.compute_noise()
            noise_high = fbm_high.compute_noise()

            # High-k variance proxy: variance of differences
            variances_low.append(np.var(np.diff(noise_low)))
            variances_high.append(np.var(np.diff(noise_high)))

        # Higher alpha should give smoother noise (lower high-k variance)
        assert np.mean(variances_high) < np.mean(variances_low)
