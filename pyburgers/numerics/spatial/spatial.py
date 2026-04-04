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
"""Spatial discretization base class for PyBurgers.

This module provides the abstract SpatialOperator base class and factory
method for creating different spatial discretization schemes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...utils.spectral_workspace import SpectralWorkspace


class SpatialOperator(ABC):
    """Base class for spatial discretization (gradient computation) schemes.

    Provides the interface and factory method for gradient operators used
    by the Burgers equation solver. Subclasses implement the `compute` method
    to evaluate spatial derivatives and the `zero_nyquist` method to enforce
    spectral constraints (a no-op for finite-difference schemes).

    Class attributes encode the scheme's maximum modified-wavenumber magnitudes
    at the Nyquist frequency, normalised by dx:

        viscous_eigenvalue      max |k̃²| · dx²   (for d²/dx²)
        hyperviscous_eigenvalue max |k̃⁴| · dx⁴   (for d⁴/dx⁴)

    These are used by the core solver to set scheme-appropriate stability
    limits for the viscous and hyperviscous time step constraints, analogous
    to how TemporalIntegrator.dissipative_stability_limit works for the time scheme.

    Attributes:
        nx: Number of grid points.
        dx: Grid spacing.
    """

    viscous_eigenvalue: float = np.pi**2
    hyperviscous_eigenvalue: float = np.pi**4

    @staticmethod
    def get_operator(
        scheme: int,
        nx: int,
        dx: float,
        spectral: SpectralWorkspace,
    ) -> SpatialOperator:
        """Factory method to create the appropriate gradient operator.

        Args:
            scheme: Spatial discretization scheme identifier.
                1 = 2nd-order central finite differences
                2 = 4th-order central finite differences
                3 = Spectral (Fourier collocation)
            nx: Number of grid points.
            dx: Grid spacing.
            spectral: SpectralWorkspace used for spectral derivatives (scheme=3)
                and Nyquist zeroing (all schemes).

        Returns:
            Instance of the requested SpatialOperator subclass.
        """
        if scheme == 1:
            from .spatial_fd2 import FD2

            return FD2(nx, dx, spectral)
        if scheme == 2:
            from .spatial_fd4 import FD4

            return FD4(nx, dx, spectral)
        if scheme == 3:
            from .spatial_spectral import Spectral

            return Spectral(nx, dx, spectral)
        raise ValueError(f"Unknown spatial scheme ID: {scheme}. Valid options: 1-3.")

    def __init__(self, nx: int, dx: float) -> None:
        """Initialize the gradient operator.

        Args:
            nx: Number of grid points.
            dx: Grid spacing.
        """
        self.nx = nx
        self.dx = dx

    @abstractmethod
    def compute(
        self, u: np.ndarray, orders: list[int | str]
    ) -> dict[str, np.ndarray]:
        """Compute spatial derivatives of u.

        Args:
            u: Velocity field array.
            orders: List of derivative orders to compute. Supported values:
                1, 2, 3, 4 (integer derivative orders) and 'sq' (dealiased
                d(u²)/dx for the nonlinear advection term).

        Returns:
            Dictionary mapping order keys ('1', '2', '3', '4', 'sq') to
            derivative arrays.
        """

    @abstractmethod
    def zero_nyquist(self, restore_physical: bool = True) -> None:
        """Zero the Nyquist mode after each integration stage.

        For spectral schemes this enforces the spectral constraint on the
        highest resolved wavenumber. For finite-difference schemes this is
        a no-op.

        Args:
            restore_physical: Whether to transform back to physical space
                after zeroing (spectral schemes only).
        """
