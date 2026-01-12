#!/usr/bin/env python
#
# PyBurgers
#
# Copyright (c) 2017–2026 Jeremy A. Gibbs
#
# This file is part of PyBurgers.
#
# This software is free and is distributed under the WTFPL license.
# See accompanying LICENSE file or visit https://www.wtfpl.net.
#
"""Fractional Brownian Motion noise generation.

This module provides the FBM class for generating stochastic forcing
with fractional Brownian motion characteristics.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pyfftw
from scipy.stats import norm

from .noise import Noise


class FBM(Noise):
    """Generates fractional Brownian motion (FBM) noise.

    FBM noise is used as the stochastic forcing term in the Burgers
    equation. The noise has a power spectrum that scales as k^(-alpha).

    Attributes:
        n_pts: Number of grid points.
        alpha: FBM exponent, controls spectral slope.
        nyquist: Nyquist mode index (n/2).
        wavenumber: Wavenumber array for spectral coloring.
    """

    def __init__(
        self,
        alpha: float,
        n_pts: int,
        fftw_planning: str,
        fftw_threads: int,
    ) -> None:
        """Initialize the FBM noise generator.

        Args:
            alpha: FBM exponent controlling the spectral slope. Typical value
                is 0.75 for Burgers turbulence.
            n_pts: Number of grid points.
            fftw_planning: FFTW planning strategy.
            fftw_threads: Number of FFTW threads.
        """
        super().__init__(
            alpha,
            n_pts,
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads,
        )

        # Computed values
        self.nyquist = int(0.5 * n_pts)
        self.wavenumber = np.abs(np.fft.fftfreq(n_pts, d=1/n_pts))
        self.wavenumber[0] = 1  # Avoid /0; DC component is 0 in compute_noise()

        # pyfftw arrays
        self.x = pyfftw.empty_aligned(n_pts, np.complex128)
        self.fx = pyfftw.empty_aligned(n_pts, np.complex128)
        self.fxn = pyfftw.empty_aligned(n_pts, np.complex128)
        self.noise = pyfftw.empty_aligned(n_pts, np.complex128)

        # pyfftw functions
        self.fft = pyfftw.FFTW(
            self.x, self.fx,
            direction="FFTW_FORWARD",
            flags=(self.fftw_planning,),
            threads=self.fftw_threads
        )

        self.ifft = pyfftw.FFTW(
            self.fxn, self.noise,
            direction="FFTW_BACKWARD",
            flags=(self.fftw_planning,),
            threads=self.fftw_threads
        )

    def compute_noise(self) -> NDArray[np.float64]:
        """Generate a realization of FBM noise.

        Creates white noise, transforms to spectral space, applies
        the FBM spectral coloring (k^(-alpha/2)), and transforms back.

        Returns:
            Real-valued noise array with FBM spectral characteristics.
        """
        # Generate white noise input using inverse normal CDF
        self.x[:] = np.sqrt(self.n_pts) * norm.ppf(np.random.rand(self.n_pts))

        # Transform to spectral space
        self.fft()

        # Zero-out DC and Nyquist modes, apply spectral coloring
        self.fx[0] = 0
        self.fx[self.nyquist] = 0
        self.fxn[:] = self.fx * (self.wavenumber ** (-0.5 * self.alpha))

        # Transform back to physical space
        self.ifft()

        return np.real(self.noise)
