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
"""2nd-order central finite-difference gradient operator for PyBurgers.

Implements SpatialOperator using 2nd-order central difference stencils
on a uniform periodic grid. Periodicity is handled via numpy.pad with wrap mode.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ...utils import get_logger
from .spatial import SpatialOperator

if TYPE_CHECKING:
    from ...utils.spectral_workspace import SpectralWorkspace


class FD2(SpatialOperator):
    """2nd-order central finite-difference gradient operator.

    All derivatives are approximated with 2nd-order accurate central
    difference stencils. Periodic boundary conditions are enforced
    implicitly via numpy.roll. The Nyquist constraint does not apply
    to finite-difference schemes, so zero_nyquist is a no-op.

    Modified-wavenumber maxima at Nyquist (ξ = π):
        viscous_eigenvalue      = 4       (max |k̃²|·dx², stencil gives -4/dx²)
        hyperviscous_eigenvalue = 16      (max |k̃⁴|·dx⁴, stencil gives +16/dx⁴)

    Stencils (h = dx):
        du/dx   = (u[i+1] - u[i-1]) / (2h)
        d²u/dx² = (u[i+1] - 2u[i] + u[i-1]) / h²
        d³u/dx³ = (u[i+2] - 2u[i+1] + 2u[i-1] - u[i-2]) / (2h³)
        d⁴u/dx⁴ = (u[i+2] - 4u[i+1] + 6u[i] - 4u[i-1] + u[i-2]) / h⁴
        sq      = d(u²)/dx via 2nd-order FD applied to v = u²
    """

    viscous_eigenvalue: float = 4.0
    hyperviscous_eigenvalue: float = 16.0

    def __init__(self, nx: int, dx: float, spectral: SpectralWorkspace) -> None:
        super().__init__(nx, dx)
        self.logger: logging.Logger = get_logger("Spatial")
        self.logger.info("--- using 2nd-order finite-difference spatial discretization")
        self._der = spectral.derivatives

        # Pre-allocated padded buffer (pad=2 on each side) and output arrays
        self._u_pad = np.empty(nx + 4, dtype=np.float64)
        self._v_sq = np.empty(nx, dtype=np.float64)
        self._v_pad = np.empty(nx + 2, dtype=np.float64)
        self._tmp = np.empty(nx, dtype=np.float64)
        self._out_1 = np.empty(nx, dtype=np.float64)
        self._out_2 = np.empty(nx, dtype=np.float64)
        self._out_3 = np.empty(nx, dtype=np.float64)
        self._out_4 = np.empty(nx, dtype=np.float64)
        self._out_sq = np.empty(nx, dtype=np.float64)

    def compute(
        self, u: np.ndarray, orders: list[int | str]
    ) -> dict[str, np.ndarray]:
        """Compute finite-difference derivatives.

        Args:
            u: Velocity field array.
            orders: Derivative orders to compute (1, 2, 3, 4, 'sq').

        Returns:
            Dictionary of derivative arrays keyed by string order.
        """
        result: dict[str, np.ndarray] = {}
        h = self.dx
        n = self.nx

        # Fill pre-allocated padded buffer (periodic wrap, pad=2)
        u_pad = self._u_pad
        u_pad[2:n + 2] = u
        u_pad[0:2] = u[-2:]
        u_pad[n + 2:n + 4] = u[:2]

        um2, um1 = u_pad[0:n], u_pad[1:n + 1]
        u0 = u_pad[2:n + 2]
        up1, up2 = u_pad[3:n + 3], u_pad[4:n + 4]

        for order in orders:
            if order == 1 or order == "1":
                # (up1 - um1) / (2h)
                np.subtract(up1, um1, out=self._out_1)
                self._out_1 /= 2.0 * h
                result["1"] = self._out_1

            elif order == 2 or order == "2":
                # (up1 + um1 - 2*u0) / h²
                np.add(up1, um1, out=self._out_2)
                np.multiply(2.0, u0, out=self._tmp)
                self._out_2 -= self._tmp
                self._out_2 /= h**2
                result["2"] = self._out_2

            elif order == 3 or order == "3":
                # (up2 - um2) - 2*(up1 - um1), divided by 2h³
                np.subtract(up2, um2, out=self._out_3)
                np.subtract(up1, um1, out=self._tmp)
                self._tmp *= 2.0
                self._out_3 -= self._tmp
                self._out_3 /= 2.0 * h**3
                result["3"] = self._out_3

            elif order == 4 or order == "4":
                # (up2 + um2) - 4*(up1 + um1) + 6*u0, divided by h⁴
                np.add(up2, um2, out=self._out_4)
                np.add(up1, um1, out=self._tmp)
                self._tmp *= 4.0
                self._out_4 -= self._tmp
                np.multiply(6.0, u0, out=self._tmp)
                self._out_4 += self._tmp
                self._out_4 /= h**4
                result["4"] = self._out_4

            elif order == "sq":
                # d(u²)/dx via 2nd-order FD on pre-allocated padded v=u² buffer
                np.multiply(u, u, out=self._v_sq)
                v_pad = self._v_pad
                v_pad[1:n + 1] = self._v_sq
                v_pad[0] = self._v_sq[-1]
                v_pad[n + 1] = self._v_sq[0]
                vm1, vp1 = v_pad[0:n], v_pad[2:n + 2]
                np.subtract(vp1, vm1, out=self._out_sq)
                self._out_sq /= 2.0 * h
                result["sq"] = self._out_sq

        return result

    def zero_nyquist(self, restore_physical: bool = True) -> None:
        """Zero the Nyquist mode via spectral transform.

        FD stencils have zero advective modified wavenumber at Nyquist,
        so multistep integrators (AB2) can amplify that mode through
        the parasitic root. Zeroing it each step prevents this.

        Only acts when restore_physical=True (final integrator stage),
        since FD derivatives need physical-space u and intermediate
        stages don't accumulate parasitic growth.

        Args:
            restore_physical: Whether to transform back to physical space.
        """
        if restore_physical:
            self._der.zero_nyquist(restore_physical=True)
