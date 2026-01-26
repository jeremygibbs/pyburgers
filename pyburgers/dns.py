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
"""Direct Numerical Simulation (DNS) for PyBurgers.

Implements the DNS solver for the 1D stochastic Burgers equation
using spectral methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .core import Burgers
from .utils.spectral_workspace import SpectralWorkspace

if TYPE_CHECKING:
    from .utils.io import Input, Output


class DNS(Burgers):
    """Direct numerical simulation solver for the Burgers equation.

    Solves the 1D stochastic Burgers equation at full resolution using
    Fourier collocation for spatial derivatives and Adams-Bashforth
    time integration.

    This class inherits common functionality from Burgers and implements
    DNS-specific behavior for noise generation and diagnostics. Uses a
    SpectralWorkspace with Derivatives and Dealias utilities (no Filter
    needed for DNS).
    """

    mode_name = "DNS"

    def __init__(self, input_obj: Input, output_obj: Output) -> None:
        """Initialize the DNS solver.

        Args:
            input_obj: Input configuration containing simulation parameters.
            output_obj: Output handler for writing results to NetCDF.
        """
        super().__init__(input_obj, output_obj)

    def _get_nx(self) -> int:
        """Return the DNS grid resolution.

        Returns:
            Number of grid points from DNS configuration.
        """
        return self.input.grid.dns.nx

    def _create_spectral_workspace(self) -> SpectralWorkspace:
        """Create the spectral workspace for DNS mode.

        DNS mode does not need filtering (no nx2), but includes FBM
        noise generation at full resolution.

        Returns:
            SpectralWorkspace configured for DNS with noise.
        """
        return SpectralWorkspace(
            nx=self.nx,
            dx=self.dx,
            noise_alpha=self.noise_alpha,
            noise_nx=self.nx,
            fftw_planning=self.fftw_planning,
            fftw_threads=self.fftw_threads,
        )

    def _setup_mode_specific(self) -> None:
        """Initialize DNS-specific components.

        DNS mode has no additional setup beyond the spectral workspace.
        FBM noise is initialized as part of the workspace.
        """
        self.logger.info("DNS configuration:")
        self.logger.info("  Resolution: %d points", self.nx)

    def _setup_output_fields(self) -> dict[str, Any]:
        """Configure DNS output fields.

        Returns:
            Dictionary with grid, velocity, and TKE fields.
        """
        return {
            "x": self.x,
            "u": self.u,
            "tke": self.tke,
        }

    def _compute_derivatives(self, t: int) -> dict[str, np.ndarray]:
        """Compute spatial derivatives for DNS.

        DNS needs 2nd derivative for diffusion and du²/dx for advection.

        Args:
            t: Current time step index (unused in DNS).

        Returns:
            Dictionary with '2' and 'sq' derivatives.
        """
        return self.spectral.derivatives.compute(self.u, [2, "sq"])

    def _compute_noise(self) -> np.ndarray:
        """Generate FBM noise at full resolution.

        Returns:
            Noise array at DNS grid resolution.
        """
        return self.spectral.noise.compute_noise()

    def _compute_rhs(self, derivatives: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        """Compute the DNS right-hand side.

        RHS = ν∂²u/∂x² - ½∂u²/∂x + √(2ε/dt) * noise

        Args:
            derivatives: Dictionary with '2' and 'sq' derivatives.
            noise: FBM noise array.

        Returns:
            RHS array for time integration.
        """
        d2udx2 = derivatives["2"]
        du2dx = derivatives["sq"]

        return self.visc * d2udx2 - 0.5 * du2dx + np.sqrt(2 * self.noise_amp / self.dt) * noise

    def _save_diagnostics(
        self, derivatives: dict[str, np.ndarray], t_out: int, t_loop: float
    ) -> None:
        """Compute TKE and save DNS output.

        Args:
            derivatives: Dictionary of spatial derivatives (unused).
            t_out: Output time index.
            t_loop: Current simulation time.
        """
        self.tke[:] = np.var(self.u)
        self.output.save(self.output_fields, t_out, t_loop, initial=False)
