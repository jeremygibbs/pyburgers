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
"""4th-order central finite-difference gradient operator for PyBurgers.

Implements SpatialOperator using 4th-order central difference stencils
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


class FD4(SpatialOperator):
    """4th-order central finite-difference gradient operator.

    All derivatives are approximated with 4th-order accurate central
    difference stencils. Periodic boundary conditions are enforced
    implicitly via numpy.roll. The Nyquist constraint does not apply
    to finite-difference schemes, so zero_nyquist is a no-op.

    Modified-wavenumber maxima at Nyquist (ξ = π):
        viscous_eigenvalue      = 16/3 ≈ 5.33  (max |k̃²|·dx², stencil gives -64/(12dx²))
        hyperviscous_eigenvalue = 80/3 ≈ 26.67 (max |k̃⁴|·dx⁴, stencil gives 160/(6dx⁴))

    Stencils (h = dx):
        du/dx   = (-u[i+2] + 8u[i+1] - 8u[i-1] + u[i-2]) / (12h)
        d²u/dx² = (-u[i+2] + 16u[i+1] - 30u[i] + 16u[i-1] - u[i-2]) / (12h²)
        d³u/dx³ = (u[i+3] - 8u[i+2] + 13u[i+1] - 13u[i-1] + 8u[i-2] - u[i-3]) / (8h³)
        d⁴u/dx⁴ = (-u[i+3] + 12u[i+2] - 39u[i+1] + 56u[i] - 39u[i-1] + 12u[i-2] - u[i-3]) / (6h⁴)
        sq      = d(u²)/dx via 4th-order FD applied to v = u²
    """

    viscous_eigenvalue: float = 16.0 / 3.0
    hyperviscous_eigenvalue: float = 80.0 / 3.0

    def __init__(self, nx: int, dx: float, spectral: SpectralWorkspace) -> None:
        super().__init__(nx, dx)
        self.logger: logging.Logger = get_logger("Spatial")
        self.logger.info("--- using 4th-order finite-difference spatial discretization")
        self._der = spectral.derivatives

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

        # Periodic padding — slices are zero-copy views (no allocation)
        u_pad = np.pad(u, 3, mode="wrap")
        um3, um2, um1 = u_pad[0:n], u_pad[1:n+1], u_pad[2:n+2]
        u0 = u_pad[3:n+3]
        up1, up2, up3 = u_pad[4:n+4], u_pad[5:n+5], u_pad[6:n+6]

        for order in orders:
            if order == 1 or order == "1":
                result["1"] = (-up2 + 8.0 * up1 - 8.0 * um1 + um2) / (12.0 * h)

            elif order == 2 or order == "2":
                result["2"] = (
                    -up2 + 16.0 * up1 - 30.0 * u0 + 16.0 * um1 - um2
                ) / (12.0 * h**2)

            elif order == 3 or order == "3":
                result["3"] = (
                    up3 - 8.0 * up2 + 13.0 * up1 - 13.0 * um1 + 8.0 * um2 - um3
                ) / (8.0 * h**3)

            elif order == 4 or order == "4":
                result["4"] = (
                    -up3 + 12.0 * up2 - 39.0 * up1 + 56.0 * u0
                    - 39.0 * um1 + 12.0 * um2 - um3
                ) / (6.0 * h**4)

            elif order == "sq":
                v_pad = np.pad(u**2, 2, mode="wrap")
                vm2, vm1 = v_pad[0:n], v_pad[1:n+1]
                vp1, vp2 = v_pad[3:n+3], v_pad[4:n+4]
                result["sq"] = (-vp2 + 8.0 * vp1 - 8.0 * vm1 + vm2) / (12.0 * h)

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
