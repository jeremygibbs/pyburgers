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
"""Deardorff 1.5-order TKE SGS model.

Implements the prognostic subgrid TKE model following Deardorff.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ...utils import constants as c
from ...utils import get_logger
from .sgs import SGS

if TYPE_CHECKING:
    from ...utils.io import Input
    from ...utils.spectral_workspace import SpectralWorkspace


class Deardorff(SGS):
    """Deardorff 1.5-order TKE subgrid-scale model.

    A prognostic SGS model that solves a transport equation for subgrid
    turbulent kinetic energy (TKE). The eddy viscosity is computed from
    the subgrid TKE as: nu_t = c1 * dx * sqrt(tke_sgs).

    Uses the shared spectral workspace for dealiasing and derivative operations.
    """

    def __init__(self, input_obj: Input, spectral: SpectralWorkspace) -> None:
        """Initialize the Deardorff TKE model.

        Args:
            input_obj: Input configuration object.
            spectral: SpectralWorkspace with shared Dealias and Derivatives utilities.
        """
        super().__init__(input_obj, spectral)
        self.logger: logging.Logger = get_logger("SGS")
        self.logger.info("--- using the Deardorff TKE model")

        # Precompute reciprocal for hot-loop multiplication instead of division
        self._inv_dx = 1.0 / self.dx

        # Pre-allocate scratch arrays to avoid temporaries in the hot loop
        nx = self.nx
        self._dudx_snap = np.zeros(nx)
        self._dkdx = np.zeros(nx)
        self._dkudx = np.zeros(nx)
        self._tke_safe = np.zeros(nx)
        self._Vt = np.zeros(nx)
        self._scratch_a = np.zeros(nx)
        self._scratch_b = np.zeros(nx)
        self._tke_tendency = np.zeros(nx)

    def compute(
        self, u: np.ndarray, dudx: np.ndarray, tke_sgs: np.ndarray | float, dt: float
    ) -> dict[str, Any]:
        """Compute the Deardorff SGS stress and update subgrid TKE.

        Solves the prognostic TKE equation and computes the SGS stress
        from the updated subgrid TKE.

        Args:
            u: Velocity field array.
            dudx: Velocity gradient array.
            tke_sgs: Current subgrid TKE array.
            dt: Current time step size for TKE tendency integration.

        Returns:
            Dictionary with 'tau' (SGS stress), 'coeff' (c1),
            and 'tke_sgs' (updated subgrid TKE).
        """

        # Model constants
        ce = c.sgs.DEARDORFF_CE  # Dissipation coefficient
        c1 = c.sgs.DEARDORFF_C1  # Eddy viscosity coefficient
        dx = self.dx

        # Snapshot dudx — the caller's array may alias the same
        # internal buffer that derivatives.compute() overwrites.
        self._dudx_snap[:] = dudx

        # Compute TKE gradients (copy each result before the next
        # compute() call overwrites the shared _out_1 buffer)
        derivs_k = self.spectral.derivatives.compute(tke_sgs, [1])
        self._dkdx[:] = derivs_k["1"]

        np.multiply(tke_sgs, u, out=self._scratch_a)
        derivs_ku = self.spectral.derivatives.compute(self._scratch_a, [1])
        self._dkudx[:] = derivs_ku["1"]

        # Eddy viscosity and SGS stress
        np.maximum(tke_sgs, 0.0, out=self._tke_safe)
        np.sqrt(self._tke_safe, out=self._Vt)
        np.multiply(c1 * dx, self._Vt, out=self._Vt)      # Vt = c1*dx*sqrt(tke_safe)
        np.multiply(-2.0, self._Vt, out=self._scratch_a)
        np.multiply(self._scratch_a, self._dudx_snap, out=self.result["tau"])  # tau

        # TKE diffusion term: d/dx(2*Vt*dkdx)
        np.multiply(self._Vt, self._dkdx, out=self._scratch_a)
        np.multiply(2.0, self._scratch_a, out=self._scratch_a)
        derivs_zz = self.spectral.derivatives.compute(self._scratch_a, [1])
        # dzzdx lives in the shared derivative buffer — used directly below

        # TKE tendency: -dkudx + production + diffusion - dissipation
        # Production: 2*Vt*dudx^2
        np.square(self._dudx_snap, out=self._scratch_a)
        np.multiply(self._Vt, self._scratch_a, out=self._scratch_a)
        np.multiply(2.0, self._scratch_a, out=self._scratch_a)  # prod
        prod_mean = float(np.mean(self._scratch_a))

        # Dissipation: -ce * tke_safe^1.5 / dx
        np.power(self._tke_safe, 1.5, out=self._scratch_b)
        np.multiply(-ce * self._inv_dx, self._scratch_b, out=self._scratch_b)  # diss
        diss_mean = float(np.mean(self._scratch_b))

        # Assemble tendency: -dkudx + prod + diff + diss
        diff = derivs_zz["1"]
        diff_mean = float(np.mean(diff))
        np.negative(self._dkudx, out=self._tke_tendency)
        np.add(self._tke_tendency, self._scratch_a, out=self._tke_tendency)
        np.add(self._tke_tendency, diff, out=self._tke_tendency)
        np.add(self._tke_tendency, self._scratch_b, out=self._tke_tendency)

        self.result["coeff"] = c1
        self.result["tke_tendency"] = self._tke_tendency
        self.result["tke_prod"] = prod_mean
        self.result["tke_diff"] = diff_mean
        self.result["tke_diss"] = diss_mean

        return self.result
