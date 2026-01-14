"""Spectral methods for pyBurgers.

This module provides classes for computing spectral derivatives,
dealiasing nonlinear terms, and filtering operations using pyfftw.
"""
from __future__ import annotations

import numpy as np
import pyfftw


class Derivatives:
    """Computes spectral derivatives using FFT.

    Uses Fourier collocation to compute spatial derivatives of a field.
    Supports first, second, and third order derivatives, as well as
    the dealiased derivative of the squared field (d(u^2)/dx).

    Attributes:
        nx: Number of grid points.
        dx: Grid spacing.
        m: Nyquist mode index (nx/2).
        fac: Derivative scaling factor (2*pi/(nx*dx)).
        k: Wavenumber array.
    """

    def __init__(
        self,
        nx: int,
        dx: float,
        fftw_planning: str = 'FFTW_MEASURE',
        fftw_threads: int = 1
    ) -> None:
        """Initialize the Derivatives calculator.

        Args:
            nx: Number of grid points.
            dx: Grid spacing.
            fftw_planning: FFTW planning strategy.
            fftw_threads: Number of threads for FFTW.
        """
        self.nx = nx
        self.dx = dx

        # computed values
        self.m = int(self.nx / 2)
        self.fac = 2 * np.pi / (self.nx * self.dx)
        self.k = np.fft.fftfreq(self.nx, d=1 / self.nx)
        self.k[self.m] = 0

        # pyfftw arrays
        self.u = pyfftw.empty_aligned(nx, np.complex128)
        self.fu = pyfftw.empty_aligned(nx, np.complex128)
        self.fun = pyfftw.empty_aligned(nx, np.complex128)
        self.der = pyfftw.empty_aligned(nx, np.complex128)

        # padded pyfftw arrays for dealiasing
        self.zp = pyfftw.zeros_aligned(nx, np.complex128)
        self.up = pyfftw.empty_aligned(nx * 2, np.complex128)
        self.up2 = pyfftw.empty_aligned(nx * 2, np.complex128)
        self.fup = pyfftw.empty_aligned(nx * 2, np.complex128)

        # pyfftw functions
        self.fft = pyfftw.FFTW(
            self.u, self.fu,
            direction="FFTW_FORWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        self.ifft = pyfftw.FFTW(
            self.fun, self.der,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        # Inverse FFT for Nyquist zeroing: fu -> u
        # Used by core.py to zero the Nyquist mode and transform back
        self.ifft_nyquist = pyfftw.FFTW(
            self.fu, self.u,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        self.fftp = pyfftw.FFTW(
            self.up, self.fup,
            direction="FFTW_FORWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        self.ifftp = pyfftw.FFTW(
            self.fup, self.up,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

    def compute(
        self,
        u: np.ndarray,
        order: list[int | str]
    ) -> dict[str, np.ndarray]:
        """Compute spectral derivatives of the input field.

        Args:
            u: Input velocity field array.
            order: List of derivative orders to compute. Can include
                integers (1, 2, 3) for standard derivatives or 'sq'
                for the dealiased derivative of u^2.

        Returns:
            Dictionary mapping order keys ('1', '2', '3', 'sq') to
            the corresponding derivative arrays.
        """
        derivatives = {}

        # copy input array
        self.u[:] = u

        # compute fft
        self.fft()

        # loop through order of derivative from user
        for key in order:
            if key == 1:
                self.fun[:] = 1j * self.k * self.fu
                self.ifft()
                derivatives['1'] = self.fac * np.real(self.der)
            if key == 2:
                self.fun[:] = -self.k * self.k * self.fu
                self.ifft()
                derivatives['2'] = self.fac**2 * np.real(self.der)
            if key == 3:
                self.fun[:] = -1j * self.k**3 * self.fu
                self.ifft()
                derivatives['3'] = self.fac**3 * np.real(self.der)
            if key == 'sq':
                # Dealiased computation of d(u^2)/dx using zero-padding
                # Optimized: use slice assignment instead of np.insert()
                # fup has size 2*nx, insert nx zeros at position m
                self.fup[0:self.m] = self.fu[0:self.m]
                self.fup[self.m:self.m + self.nx] = 0  # nx zeros in the middle
                self.fup[self.m + self.nx:] = self.fu[self.m:]
                self.ifftp()
                self.up[:] = self.up[:] ** 2
                self.fftp()
                self.fu[0:self.m] = self.fup[0:self.m]
                self.fu[self.m::] = self.fup[self.nx + self.m:]
                self.fun[:] = 1j * self.k * self.fu
                self.ifft()
                derivatives['sq'] = 2 * self.fac * np.real(self.der)

        return derivatives


class Dealias:
    """Dealiases nonlinear products using the 3/2 rule.

    Computes |x| * x with proper dealiasing to avoid aliasing errors
    in the nonlinear term of the Burgers equation.

    Attributes:
        nx: Number of grid points.
        m: Nyquist mode index (nx/2).
    """

    def __init__(
        self,
        nx: int,
        fftw_planning: str = 'FFTW_MEASURE',
        fftw_threads: int = 1
    ) -> None:
        """Initialize the Dealias calculator.

        Args:
            nx: Number of grid points.
            fftw_planning: FFTW planning strategy.
            fftw_threads: Number of threads for FFTW.
        """
        self.nx = nx
        self.m = int(self.nx / 2)

        # pyfftw arrays
        self.x = pyfftw.empty_aligned(self.nx, np.complex128)
        self.fx = pyfftw.empty_aligned(self.nx, np.complex128)

        # padded pyfftw arrays
        self.zp = pyfftw.zeros_aligned(self.m, np.complex128)
        self.xp = pyfftw.empty_aligned(3 * self.m, np.complex128)
        self.temp = pyfftw.empty_aligned(3 * self.m, np.complex128)
        self.fxp = pyfftw.empty_aligned(3 * self.m, np.complex128)

        # pyfftw functions
        self.fft = pyfftw.FFTW(
            self.x, self.fx,
            direction="FFTW_FORWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        self.ifft = pyfftw.FFTW(
            self.fx, self.x,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        self.fftp = pyfftw.FFTW(
            self.xp, self.fxp,
            direction="FFTW_FORWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        self.ifftp = pyfftw.FFTW(
            self.fxp, self.xp,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

    def compute(self, x: np.ndarray) -> np.ndarray:
        """Compute the dealiased product |x| * x.

        Args:
            x: Input array.

        Returns:
            Dealiased result of |x| * x.
        """
        # copy input array
        self.x[:] = x

        # compute fft of x
        self.fft()

        # zero-pad fx
        self.fxp[:] = np.concatenate((
            self.fx[0:self.m + 1],
            self.zp,
            self.fx[self.m + 1:self.nx]
        ))

        # compute ifft of fxp
        self.ifftp()

        # store xp in temp
        self.temp[:] = self.xp[:]

        # change x to abs(x)
        self.x[:] = np.abs(x)

        # compute fft of x
        self.fft()

        # zero-pad fx
        self.fxp[:] = np.concatenate((
            self.fx[0:self.m + 1],
            self.zp,
            self.fx[self.m + 1:self.nx]
        ))

        # compute ifft of fxp
        self.ifftp()

        # multiply xp[x] with xp[abs(x)]
        self.xp[:] = np.real(self.xp) * np.real(self.temp)

        # compute fft of xp
        self.fftp()

        # de-alias fxp
        self.fx[:] = np.concatenate((
            self.fxp[0:self.m + 1],
            self.fxp[2 * self.m + 1:self.m + self.nx]
        ))

        # compute ifft of fx
        self.ifft()

        # return de-aliased input
        return (3 / 2) * np.real(self.x)


class Filter:
    """Spectral filtering operations.

    Provides spectral cutoff filtering and downscaling from DNS to LES
    resolution using Fourier methods.

    Attributes:
        nx: Number of grid points for the filtered field.
        nx2: Number of grid points for the source field (DNS resolution).
    """

    def __init__(
        self,
        nx: int,
        nx2: int | None = None,
        fftw_planning: str = 'FFTW_MEASURE',
        fftw_threads: int = 1
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

        # pyfftw arrays
        self.x = pyfftw.empty_aligned(self.nx, np.complex128)
        self.fx = pyfftw.empty_aligned(self.nx, np.complex128)
        self.fxf = pyfftw.zeros_aligned(self.nx, np.complex128)

        # pyfftw functions
        self.fft = pyfftw.FFTW(
            self.x, self.fx,
            direction="FFTW_FORWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        self.ifft = pyfftw.FFTW(
            self.fxf, self.x,
            direction="FFTW_BACKWARD",
            flags=(fftw_planning,),
            threads=fftw_threads
        )

        # check for optional larger nx (for downscaling from DNS->LES)
        if nx2:
            self.nx2 = nx2

            # pyfftw arrays for larger grid
            self.x2 = pyfftw.empty_aligned(self.nx2, np.complex128)
            self.fx2 = pyfftw.empty_aligned(self.nx2, np.complex128)

            # pyfftw function for larger grid
            self.fft2 = pyfftw.FFTW(
                self.x2, self.fx2,
                direction="FFTW_FORWARD",
                flags=(fftw_planning,),
                threads=fftw_threads
            )

    def cutoff(self, x: np.ndarray, ratio: int) -> np.ndarray:
        """Apply a spectral cutoff filter.

        Removes high-frequency modes above nx/ratio.

        Args:
            x: Input array to filter.
            ratio: Cutoff ratio (keeps modes up to nx/ratio).

        Returns:
            Filtered array with high frequencies removed.
        """
        # signal size information
        m = int(self.nx / ratio)
        half = int(m / 2)

        # copy input array
        self.x[:] = x

        # compute fft of x
        self.fft()

        # filter fx (keep low frequencies only)
        self.fxf[:] = 0
        self.fxf[0:half] = self.fx[0:half]
        self.fxf[self.nx - half + 1:self.nx] = self.fx[self.nx - half + 1:self.nx]

        # compute ifft of fxf
        self.ifft()

        # return filtered x
        return np.real(self.x)

    def downscale(self, x: np.ndarray, ratio: int) -> np.ndarray:
        """Downscale a field from DNS to LES resolution.

        Uses Fourier filtering to project a high-resolution field
        onto a coarser grid while preserving low-frequency content.

        Args:
            x: Input array at DNS resolution.
            ratio: Downscaling ratio (nx2 / nx).

        Returns:
            Downscaled array at LES resolution.
        """
        # zero output array to prevent stale data
        self.fxf[:] = 0

        # copy input array
        self.x2[:] = x

        # signal shape information
        half = int(self.nx / 2)

        # compute fft of larger series
        self.fft2()

        # filter - transfer low frequencies to smaller array
        self.fxf[half] = 0
        self.fxf[0:half] = self.fx2[0:half]
        self.fxf[half + 1:self.nx] = self.fx2[self.nx2 - half + 1:self.nx2]

        # compute the ifft
        self.ifft()

        # return filtered downscaled field
        # pyfftw >= 0.15.0 defaults to normalise_idft=True (divides by nx).
        # To match old behavior with unnormalized IFFT, multiply by nx.
        return (self.nx / ratio) * np.real(self.x)
