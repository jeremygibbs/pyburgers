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
"""Large-Eddy Simulation (LES) for pyBurgers.

Implements the LES solver for the 1D stochastic Burgers equation
with subgrid-scale modeling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .core import Burgers
from .physics.sgs import get_model as get_sgs_model
from .utils.spectral_workspace import SpectralWorkspace

if TYPE_CHECKING:
    from .utils.io import Input, Output


class LES(Burgers):
    """Large-eddy simulation solver for the Burgers equation.

    Solves the filtered 1D stochastic Burgers equation using Fourier
    collocation for spatial derivatives, Adams-Bashforth time integration,
    and subgrid-scale models for unresolved turbulence.

    This class inherits common functionality from Burgers and implements
    LES-specific behavior including SGS modeling and filtered noise.

    Attributes:
        nx_dns: Number of DNS grid points (for noise generation).
        sgs_model_id: SGS model type identifier (0-4).
        spectral: SpectralWorkspace with derivatives, dealias, and filter.
        subgrid: SGS model instance.
        tke_sgs: Subgrid TKE (for Deardorff model).
    """

    mode_name = "LES"

    def __init__(self, input_obj: Input, output_obj: Output) -> None:
        """Initialize the LES solver.

        Args:
            input_obj: Input configuration containing simulation parameters.
            output_obj: Output handler for writing results to NetCDF.
        """
        # Store LES-specific config before calling parent __init__
        # (needed because _setup_mode_specific is called during parent init)
        self._nx_dns = input_obj.models.dns.nx
        self._sgs_model_id = input_obj.models.les.sgs

        super().__init__(input_obj, output_obj)

    def _get_nx(self) -> int:
        """Return the LES grid resolution.

        Returns:
            Number of grid points from LES configuration.
        """
        return self.input.models.les.nx

    def _create_spectral_workspace(self) -> SpectralWorkspace:
        """Create the spectral workspace for LES mode.

        LES mode needs filtering capability for downscaling DNS noise
        to LES grid, and generates FBM noise at DNS resolution.

        Returns:
            SpectralWorkspace configured for LES with filtering and noise.
        """
        return SpectralWorkspace(
            nx=self.nx,
            dx=self.dx,
            nx2=self._nx_dns,  # DNS resolution for downscaling
            noise_alpha=self.noise_alpha,
            noise_nx=self._nx_dns,  # Generate noise at DNS resolution
            fftw_planning=self.fftw_planning,
            fftw_threads=self.fftw_threads
        )

    def _setup_mode_specific(self) -> None:
        """Initialize LES-specific components.

        Sets up the SGS model. FBM noise and filtering are handled
        by the spectral workspace.
        """
        self.nx_dns = self._nx_dns
        self.sgs_model_id = self._sgs_model_id

        # SGS model (pass spectral workspace for shared utilities)
        self.subgrid = get_sgs_model(self.sgs_model_id, self.input, self.spectral)

        # Initialize subgrid TKE for Deardorff model
        if self.sgs_model_id == 4:
            self.tke_sgs: np.ndarray | float = np.ones(self.nx)
        else:
            self.tke_sgs = 0.0

        # LES diagnostic fields
        self.C_sgs = np.zeros(1)
        self.diss_sgs = np.zeros(1)
        self.diss_mol = np.zeros(1)
        self.ens_prod = np.zeros(1)
        self.ens_dsgs = np.zeros(1)
        self.ens_dmol = np.zeros(1)

        # Store last computed values for diagnostics
        self._last_tau: np.ndarray | None = None
        self._last_coeff: float = 0.0

    def _setup_output_fields(self) -> dict[str, Any]:
        """Configure LES output fields.

        Returns:
            Dictionary with grid, velocity, TKE, and SGS diagnostic fields.
        """
        fields = {
            'x': self.x,
            'u': self.u,
            'tke': self.tke,
            'C_sgs': self.C_sgs,
            'diss_sgs': self.diss_sgs,
            'diss_mol': self.diss_mol,
            'ens_prod': self.ens_prod,
            'ens_diss_sgs': self.ens_dsgs,
            'ens_diss_mol': self.ens_dmol,
        }

        # Add subgrid TKE output for Deardorff model
        if self.sgs_model_id == 4:
            fields['tke_sgs'] = self.tke_sgs

        return fields

    def _compute_derivatives(self, t: int) -> dict[str, np.ndarray]:
        """Compute spatial derivatives for LES.

        LES needs 1st, 2nd derivatives and du²/dx. At output times,
        also computes 3rd derivative for enstrophy budget.

        Args:
            t: Current time step index.

        Returns:
            Dictionary with '1', '2', 'sq' (and '3' at output times).
        """
        if t % self.step_save == 0:
            return self.spectral.derivatives.compute(self.u, [1, 2, 3, 'sq'])
        return self.spectral.derivatives.compute(self.u, [1, 2, 'sq'])

    def _compute_noise(self) -> np.ndarray:
        """Generate and filter FBM noise from DNS to LES scales.

        Returns:
            Filtered noise array at LES grid resolution.
        """
        noise = self.spectral.noise.compute_noise()
        return self.spectral.filter.downscale(noise, self.nx_dns // self.nx)

    def _compute_rhs(
        self,
        derivatives: dict[str, np.ndarray],
        noise: np.ndarray
    ) -> np.ndarray:
        """Compute the LES right-hand side including SGS term.

        RHS = ν∂²u/∂x² - ½∂u²/∂x + √(2ε/dt) * noise - ½∂τ/∂x

        Args:
            derivatives: Dictionary with derivatives.
            noise: Filtered FBM noise array.

        Returns:
            RHS array for time integration.
        """
        dudx = derivatives['1']
        d2udx2 = derivatives['2']
        du2dx = derivatives['sq']

        # Compute SGS stress
        sgs = self.subgrid.compute(self.u, dudx, self.tke_sgs)
        tau = sgs["tau"]
        self._last_tau = tau
        self._last_coeff = sgs["coeff"]

        # Update subgrid TKE for Deardorff model
        if self.sgs_model_id == 4:
            self.tke_sgs = sgs["tke_sgs"]

        # Compute SGS stress divergence
        # Save u because compute() overwrites the internal buffer
        u_saved = self.u.copy()
        sgsder = self.spectral.derivatives.compute(tau, [1])
        dtaudx = sgsder['1']
        # Restore u
        self.u[:] = u_saved

        return (
            self.visc * d2udx2
            - 0.5 * du2dx
            + np.sqrt(2 * self.noise_amp / self.dt) * noise
            - 0.5 * dtaudx
        )

    def _save_diagnostics(
        self,
        derivatives: dict[str, np.ndarray],
        t_out: int,
        t_loop: float
    ) -> None:
        """Compute LES diagnostics and save output.

        Computes TKE, dissipation rates, enstrophy budget terms,
        and SGS coefficient.

        Args:
            derivatives: Dictionary of spatial derivatives.
            t_out: Output time index.
            t_loop: Current simulation time.
        """
        dudx = derivatives['1']
        d2udx2 = derivatives['2']
        d3udx3 = derivatives.get('3', np.zeros_like(dudx))
        tau = self._last_tau if self._last_tau is not None else np.zeros(self.nx)

        # Compute diagnostics
        self.tke[:] = np.var(self.u)
        self.diss_sgs[:] = np.mean(-tau * dudx)
        self.diss_mol[:] = np.mean(self.visc * dudx ** 2)
        self.ens_prod[:] = np.mean(dudx ** 3)
        self.ens_dsgs[:] = np.mean(-tau * d3udx3)
        self.ens_dmol[:] = np.mean(self.visc * d2udx2 ** 2)
        self.C_sgs[:] = self._last_coeff

        self.output.save(self.output_fields, t_out, t_loop, initial=False)
