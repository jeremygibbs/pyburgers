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
on a uniform periodic grid. Periodicity is handled via numpy.roll.
"""

from __future__ import annotations

import logging

import numpy as np

from ...utils import get_logger
from .spatial import SpatialOperator


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

    def __init__(self, nx: int, dx: float) -> None:
        super().__init__(nx, dx)
        self.logger: logging.Logger = get_logger("Spatial")
        self.logger.info("--- using 2nd-order finite-difference spatial discretization")

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

        up1 = np.roll(u, -1)
        um1 = np.roll(u, 1)
        up2: np.ndarray | None = None
        um2: np.ndarray | None = None

        for order in orders:
            if order == 1 or order == "1":
                result["1"] = (up1 - um1) / (2.0 * h)

            elif order == 2 or order == "2":
                result["2"] = (up1 - 2.0 * u + um1) / h**2

            elif order == 3 or order == "3":
                if up2 is None:
                    up2 = np.roll(u, -2)
                    um2 = np.roll(u, 2)
                result["3"] = (up2 - 2.0 * up1 + 2.0 * um1 - um2) / (2.0 * h**3)

            elif order == 4 or order == "4":
                if up2 is None:
                    up2 = np.roll(u, -2)
                    um2 = np.roll(u, 2)
                result["4"] = (up2 - 4.0 * up1 + 6.0 * u - 4.0 * um1 + um2) / h**4

            elif order == "sq":
                v = u**2
                vp1 = np.roll(v, -1)
                vm1 = np.roll(v, 1)
                result["sq"] = (vp1 - vm1) / (2.0 * h)

        return result

    def zero_nyquist(self, restore_physical: bool = True) -> None:
        """No-op: finite-difference schemes have no Nyquist mode to zero."""
