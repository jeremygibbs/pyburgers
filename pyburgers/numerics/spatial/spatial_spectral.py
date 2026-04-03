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
"""Spectral gradient operator for PyBurgers.

Wraps the existing Derivatives class from SpectralWorkspace to implement
the SpatialOperator interface for Fourier collocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .spatial import SpatialOperator

if TYPE_CHECKING:
    from ...utils.spectral_workspace import SpectralWorkspace


class Spectral(SpatialOperator):
    """Spectral (Fourier collocation) gradient operator.

    Delegates all derivative computation and Nyquist enforcement to the
    existing Derivatives object in the SpectralWorkspace. No code is
    duplicated; this class is a thin adapter.
    """

    def __init__(self, nx: int, dx: float, spectral: SpectralWorkspace) -> None:
        """Initialize the spectral gradient operator.

        Args:
            nx: Number of grid points.
            dx: Grid spacing.
            spectral: SpectralWorkspace whose Derivatives object is reused.
        """
        super().__init__(nx, dx)
        self._der = spectral.derivatives

    def compute(
        self, u: np.ndarray, orders: list[int | str]
    ) -> dict[str, np.ndarray]:
        """Compute spectral derivatives via FFT.

        Args:
            u: Velocity field array.
            orders: Derivative orders to compute (1, 2, 3, 4, 'sq').

        Returns:
            Dictionary of derivative arrays keyed by order.
        """
        return self._der.compute(u, orders)

    def zero_nyquist(self, restore_physical: bool = True) -> None:
        """Zero the Nyquist mode in spectral space.

        Args:
            restore_physical: Whether to transform back to physical space.
        """
        self._der.zero_nyquist(restore_physical)
