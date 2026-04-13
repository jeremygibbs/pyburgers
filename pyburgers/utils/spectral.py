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
"""Spectral methods for PyBurgers.

This module provides classes for computing spectral derivatives,
dealiasing nonlinear terms, and filtering operations using pyfftw.
"""

from __future__ import annotations

import numpy as np
import pyfftw

from ..utils import constants as c


class Derivatives:
    """Computes spectral derivatives using real FFT (rfft/irfft).

    Uses Fourier collocation to compute spatial derivatives of a field.
    Supports first, second, and third order derivatives, as well as
    the dealiased derivative of the squared field (d(u^2)/dx).

    Since velocity fields are real-valued, rfft is used for efficiency
    (~2x speedup, ~50% memory reduction for frequency arrays).

    Attributes:
        nx: Number of grid points.
        dx: Grid spacing.
        nk: Number of rfft output coefficients (nx//2 + 1).
        m: Nyquist mode index (nx/2).
        fac: Derivative scaling factor (2*pi/(nx*dx)).
        k: Wavenumber array (non-negative frequencies only).
    """

    def __init__(
        self, nx: int, dx: float, fftw_planning: str = "FFTW_MEASURE", fftw_threads: int = 1
    ) -> None:
        """Initialize the Derivatives calculator.

        Args:
            nx: Number of grid points (must be even).
            dx: Grid spacing.
            fftw_planning: FFTW planning strategy.
            fftw_threads: Number of threads for FFTW.
        """
        self.nx = nx
        self.dx = dx

        # computed values
        self.nk = self.nx // 2 + 1  # rfft output size
        self.m = self.nx // 2  # Nyquist index
        self.fac = 2 * np.pi / (self.nx * self.dx)

        # wavenumber array for rfft (non-negative frequencies only)
        self.k = np.fft.rfftfreq(self.nx, d=1 / self.nx)
        self.k[self.nk - 1] = 0  # Zero Nyquist mode (last element)

        # Precompute powers for efficiency
        self.fac2 = self.fac**2
        self.fac3 = self.fac**3
        self.fac4 = self.fac**4
        self.k2 = self.k * self.k
        self.k3 = self.k**3
        self.k4 = self.k**4

        # pyfftw arrays for real FFT
        # Physical space: float64, Frequency space: complex128
        self.u = pyfftw.empty_aligned(nx, np.float64)
        self.fu = pyfftw.empty_aligned(self.nk, np.complex128)
        self.fun = pyfftw.empty_aligned(self.nk, np.complex128)
        self.der = pyfftw.empty_aligned(nx, np.float64)

        # Pre-allocated output arrays for derivatives (reused across calls)
        self._out_1 = pyfftw.empty_aligned(nx, np.float64)
        self._out_2 = pyfftw.empty_aligned(nx, np.float64)
        self._out_3 = pyfftw.empty_aligned(nx, np.float64)
        self._out_4 = pyfftw.empty_aligned(nx, np.float64)
        self._out_sq = pyfftw.empty_aligned(nx, np.float64)

        # Separate forward-FFT buffers for external-array derivatives
        # (e.g., SGS stress tau, subgrid TKE). Avoids save/restore of the
        # primary velocity buffers (self.u / self.fu).
        self._ext_u = pyfftw.empty_aligned(nx, np.float64)
        self._ext_fu = pyfftw.empty_aligned(self.nk, np.complex128)
        self._ext_fft = pyfftw.FFTW(
            self._ext_u, self._ext_fu,
            direction="FFTW_FORWARD", flags=(fftw_planning,), threads=fftw_threads,
        )

        # padded pyfftw arrays for 3/2-rule dealiasing
        nx_padded = 3 * self.nx // 2
        nk_padded = nx_padded // 2 + 1
        self.up = pyfftw.empty_aligned(nx_padded, np.float64)
        self.fup = pyfftw.empty_aligned(nk_padded, np.complex128)

        # pyfftw functions (auto-detects real<->complex from dtypes)
        self.fft = pyfftw.FFTW(
            self.u, self.fu, direction="FFTW_FORWARD", flags=(fftw_planning,), threads=fftw_threads
        )

        self.ifft = pyfftw.FFTW(
            self.fun,
            self.der,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads,
        )

        # Inverse FFT for Nyquist zeroing: fu -> u
        # Used by core.py to zero the Nyquist mode and transform back
        self.ifft_nyquist = pyfftw.FFTW(
            self.fu, self.u, direction="FFTW_BACKWARD", flags=(fftw_planning,), threads=fftw_threads
        )

        self.fftp = pyfftw.FFTW(
            self.up,
            self.fup,
            direction="FFTW_FORWARD",
            flags=(fftw_planning,),
            threads=fftw_threads,
        )

        self.ifftp = pyfftw.FFTW(
            self.fup,
            self.up,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads,
        )

        # Flag indicating self.fu is current for self.u (skip redundant FFT)
        self._fu_valid = False

    def compute(self, u: np.ndarray, orders: list[int | str]) -> dict[str, np.ndarray]:
        """Compute spectral derivatives of the input field.

        When ``u`` is not the internal velocity buffer (``self.u``),
        a separate FFT buffer pair is used so the primary velocity
        state is never touched.

        Args:
            u: Input field array (real-valued).
            orders: List of derivative orders to compute. Can include
                integers (1, 2, 3, 4) for standard derivatives or 'sq'
                for the dealiased derivative of u^2.

        Returns:
            Dictionary mapping order keys ('1', '2', '3', '4', 'sq') to
            the corresponding derivative arrays. Arrays are reused
            internally, so callers should consume values before the
            next compute() call.
        """
        derivatives = {}

        # For external arrays (e.g., tau, tke_sgs), use the separate
        # FFT buffers so the primary velocity state is undisturbed.
        if u is not self.u:
            self._ext_u[:] = u
            self._ext_fft()
            fu = self._ext_fu
        else:
            # Skip FFT if fu is already current (set by zero_nyquist)
            if not self._fu_valid:
                self.u[:] = u
                self.fft()
            self._fu_valid = False
            fu = self.fu

        # Loop through requested derivative orders
        for key in orders:
            if key == 1 or key == "1":
                self.fun[:] = 1j * self.k * fu
                self.ifft()
                np.multiply(self.fac, self.der, out=self._out_1)
                derivatives["1"] = self._out_1
            elif key == 2 or key == "2":
                self.fun[:] = -self.k2 * fu
                self.ifft()
                np.multiply(self.fac2, self.der, out=self._out_2)
                derivatives["2"] = self._out_2
            elif key == 3 or key == "3":
                self.fun[:] = -1j * self.k3 * fu
                self.ifft()
                np.multiply(self.fac3, self.der, out=self._out_3)
                derivatives["3"] = self._out_3
            elif key == 4 or key == "4":
                self.fun[:] = self.k4 * fu
                self.ifft()
                np.multiply(self.fac4, self.der, out=self._out_4)
                derivatives["4"] = self._out_4
            elif key == "sq":
                # Dealiased computation of d(u^2)/dx using 3/2-rule zero-padding
                # With rfft, only non-negative frequencies are stored
                self.fup[:] = 0
                self.fup[0 : self.nk] = fu
                # Transform to padded physical space
                self.ifftp()
                # Correct for 3/2 padding normalization: irfft divides by 3n/2 instead of n
                self.up *= 1.5
                # Square in physical space
                np.square(self.up, out=self.up)
                # Transform back to spectral space
                self.fftp()
                # Truncate to original modes and differentiate; self.k zeros Nyquist
                self.fun[:] = 1j * self.k * (self.fup[0 : self.nk] / 1.5)
                self.ifft()
                np.multiply(self.fac, self.der, out=self._out_sq)
                derivatives["sq"] = self._out_sq

        return derivatives

    def zero_nyquist(self, restore_physical: bool = True) -> None:
        """FFT self.u, zero the Nyquist mode, and optionally restore physical space.

        Called at the end of each RK3 stage to enforce the Nyquist constraint.

        Args:
            restore_physical: If True (default), IFFT back so self.u reflects
                the Nyquist-zeroed velocity. If False, skip the IFFT and mark
                self.fu as valid so the next compute() call skips the redundant
                FFT. Use False for intermediate RK stages, True for the final stage.
        """
        self.fft()
        self.fu[self.m] = 0
        if restore_physical:
            self.ifft_nyquist()
        else:
            self._fu_valid = True


class Dealias:
    """Dealiases nonlinear products using the 3/2 rule.

    Computes |x| * x with proper dealiasing to avoid aliasing errors
    in the nonlinear term of the Burgers equation.

    Uses real FFT (rfft/irfft) for efficiency since input fields
    are real-valued.

    Attributes:
        nx: Number of grid points.
        nk: Number of rfft output coefficients (nx//2 + 1).
        m: Nyquist mode index (nx/2).
    """

    def __init__(self, nx: int, fftw_planning: str = "FFTW_MEASURE", fftw_threads: int = 1) -> None:
        """Initialize the Dealias calculator.

        Args:
            nx: Number of grid points.
            fftw_planning: FFTW planning strategy.
            fftw_threads: Number of threads for FFTW.
        """
        self.nx = nx
        self.m = self.nx // 2
        self.nk = self.nx // 2 + 1  # rfft output size

        # 3/2 rule padding sizes
        nx_padded = 3 * self.m  # = 3/2 * nx
        nk_padded = nx_padded // 2 + 1

        # pyfftw arrays for real FFT
        self.x = pyfftw.empty_aligned(self.nx, np.float64)
        self.fx = pyfftw.empty_aligned(self.nk, np.complex128)

        # padded pyfftw arrays
        self.xp = pyfftw.empty_aligned(nx_padded, np.float64)
        self.temp = pyfftw.empty_aligned(nx_padded, np.float64)
        self.fxp = pyfftw.empty_aligned(nk_padded, np.complex128)

        # pyfftw functions (auto-detects real<->complex from dtypes)
        self.fft = pyfftw.FFTW(
            self.x, self.fx, direction="FFTW_FORWARD", flags=(fftw_planning,), threads=fftw_threads
        )

        self.ifft = pyfftw.FFTW(
            self.fx, self.x, direction="FFTW_BACKWARD", flags=(fftw_planning,), threads=fftw_threads
        )

        self.fftp = pyfftw.FFTW(
            self.xp,
            self.fxp,
            direction="FFTW_FORWARD",
            flags=(fftw_planning,),
            threads=fftw_threads,
        )

        self.ifftp = pyfftw.FFTW(
            self.fxp,
            self.xp,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads,
        )

    def compute(self, x: np.ndarray) -> np.ndarray:
        """Compute the dealiased product |x| * x.

        Args:
            x: Input array (real-valued).

        Returns:
            Dealiased result of |x| * x. The returned array is an
            internal buffer; copy it if you need to preserve the values
            across calls.
        """
        scale = c.spectral.DEALIAS_SCALE

        self.x[:] = x
        self.fft()

        # Zero-pad fx
        self.fxp[:] = 0
        self.fxp[0 : self.nk] = self.fx
        self.ifftp()

        # Store padded x in temp
        self.temp[:] = self.xp

        # Compute padded |x|
        np.abs(x, out=self.x)
        self.fft()
        self.fxp[:] = 0
        self.fxp[0 : self.nk] = self.fx
        self.ifftp()

        # Multiply x * |x| in padded space
        np.multiply(self.xp, self.temp, out=self.xp)

        # Transform back and de-alias
        self.fftp()
        self.fx[:] = self.fxp[0 : self.nk]
        self.ifft()

        self.x *= scale
        return self.x


class Filter:
    """Spectral filtering operations.

    Provides spectral cutoff filtering and downscaling from DNS to LES
    resolution using Fourier methods.

    Uses real FFT (rfft/irfft) for efficiency since input fields
    are real-valued.

    Attributes:
        nx: Number of grid points for the filtered field.
        nk: Number of rfft output coefficients (nx//2 + 1).
        nx2: Number of grid points for the source field (DNS resolution).
    """

    def __init__(
        self,
        nx: int,
        nx2: int | None = None,
        fftw_planning: str = "FFTW_MEASURE",
        fftw_threads: int = 1,
    ) -> None:
        """Initialize the Filter.

        Args:
            nx: Number of grid points for the target (filtered) field.
            nx2: Optional number of grid points for source field
                (used for downscaling from DNS to LES).
            fftw_planning: FFTW planning strategy.
            fftw_threads: Number of threads for FFTW.
        """
        self.nx = nx
        self.nk = self.nx // 2 + 1  # rfft output size

        # pyfftw arrays for real FFT
        self.x = pyfftw.empty_aligned(self.nx, np.float64)
        self.fx = pyfftw.empty_aligned(self.nk, np.complex128)
        self.fxf = pyfftw.zeros_aligned(self.nk, np.complex128)

        # pyfftw functions (auto-detects real<->complex from dtypes)
        self.fft = pyfftw.FFTW(
            self.x, self.fx, direction="FFTW_FORWARD", flags=(fftw_planning,), threads=fftw_threads
        )

        self.ifft = pyfftw.FFTW(
            self.fxf,
            self.x,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads,
        )

        # check for optional larger nx (for downscaling from DNS->LES)
        if nx2:
            self.nx2 = nx2
            self.nk2 = self.nx2 // 2 + 1

            # pyfftw arrays for larger grid
            self.x2 = pyfftw.empty_aligned(self.nx2, np.float64)
            self.fx2 = pyfftw.empty_aligned(self.nk2, np.complex128)

            # pyfftw function for larger grid
            self.fft2 = pyfftw.FFTW(
                self.x2,
                self.fx2,
                direction="FFTW_FORWARD",
                flags=(fftw_planning,),
                threads=fftw_threads,
            )

    def cutoff(self, x: np.ndarray, ratio: int) -> np.ndarray:
        """Apply a spectral cutoff filter.

        Removes high-frequency modes above nx/ratio.

        Args:
            x: Input array to filter (real-valued).
            ratio: Cutoff ratio (keeps modes up to nx/ratio).

        Returns:
            Filtered array with high frequencies removed.
        """
        # signal size information
        m = int(self.nx / ratio)
        half = int(m / 2)

        # copy input array
        self.x[:] = x

        # compute rfft of x
        self.fft()

        # filter fx (keep low frequencies only)
        # With rfft, only non-negative frequencies exist
        self.fxf[:] = 0
        self.fxf[0:half] = self.fx[0:half]

        # compute irfft of fxf
        self.ifft()

        # return filtered x
        return self.x.copy()

    def downscale(self, x: np.ndarray, ratio: int) -> np.ndarray:
        """Downscale a field from DNS to LES resolution.

        Uses Fourier filtering to project a high-resolution field
        onto a coarser grid while preserving low-frequency content.

        Args:
            x: Input array at DNS resolution (real-valued).
            ratio: Downscaling ratio (nx2 / nx).

        Returns:
            Downscaled array at LES resolution.
        """
        # zero output array to prevent stale data
        self.fxf[:] = 0

        # copy input array
        self.x2[:] = x

        # signal shape information - keep up to (but not including) LES Nyquist
        half = self.nx // 2

        # compute rfft of larger series
        self.fft2()

        # filter - transfer low frequencies to smaller array
        # With rfft, only non-negative frequencies exist
        self.fxf[0:half] = self.fx2[0:half]
        self.fxf[half] = 0  # Zero Nyquist

        # compute the irfft
        self.ifft()

        # return filtered downscaled field
        # Scale by 1/ratio to preserve amplitude when downscaling
        return (1 / ratio) * self.x.copy()
