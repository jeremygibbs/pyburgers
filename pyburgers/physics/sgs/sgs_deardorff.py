"""Deardorff 1.5-order TKE SGS model.

Implements the prognostic subgrid TKE model following Deardorff.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .sgs import SGS
from ...utils import get_logger
from ...utils import constants as c

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

    def __init__(
        self,
        input_obj: Input,
        spectral: SpectralWorkspace
    ) -> None:
        """Initialize the Deardorff TKE model.

        Args:
            input_obj: Input configuration object.
            spectral: SpectralWorkspace with shared Dealias and Derivatives utilities.
        """
        super().__init__(input_obj, spectral)
        self.logger: logging.Logger = get_logger("SGS")
        self.logger.info("Using the Deardorff TKE model")

    def compute(
        self,
        u: np.ndarray,
        dudx: np.ndarray,
        tke_sgs: np.ndarray | float
    ) -> dict[str, Any]:
        """Compute the Deardorff SGS stress and update subgrid TKE.

        Solves the prognostic TKE equation and computes the SGS stress
        from the updated subgrid TKE.

        Args:
            u: Velocity field array.
            dudx: Velocity gradient array.
            tke_sgs: Current subgrid TKE array.

        Returns:
            Dictionary with 'tau' (SGS stress), 'coeff' (c1),
            and 'tke_sgs' (updated subgrid TKE).
        """
        
        # Model constants
        ce = c.sgs.DEARDORFF_CE  # Dissipation coefficient
        c1 = c.sgs.DEARDORFF_C1  # Eddy viscosity coefficient

        # Derivatives.compute uses the shared velocity buffer; preserve u.
        u_local = u.copy()

        # Strain rate squared (1D), used for production
        dudx2 = dudx * dudx

        # Compute TKE gradients
        derivs_k = self.spectral.derivatives.compute(tke_sgs, [1])
        dkdx = derivs_k['1']

        derivs_ku = self.spectral.derivatives.compute(tke_sgs * u_local, [1])
        dkudx = derivs_ku['1']

        # Eddy viscosity and SGS stress
        tke_sgs_safe = np.maximum(tke_sgs, 0.0)
        Vt = c1 * self.dx * np.sqrt(tke_sgs_safe)
        tau = -2.0 * Vt * dudx

        # TKE diffusion term
        zz = 2 * Vt * dkdx
        derivs_zz = self.spectral.derivatives.compute(zz, [1])
        dzzdx = derivs_zz["1"]

        # TKE tendency: advection + production + diffusion - dissipation
        prod = 2 * Vt * dudx2
        diff = dzzdx
        diss = -ce * (tke_sgs ** 1.5) / self.dx
        dtke = (
            (-1 * dkudx)
            + prod
            + diff
            + diss
        ) * self.dt

        # Update subgrid TKE
        tke_sgs_new = np.maximum(tke_sgs + dtke, 0.0)
        self.spectral.u[:] = u_local

        self.sgs['tau'] = tau
        self.sgs['coeff'] = c1
        self.sgs['tke_sgs'] = tke_sgs_new
        self.sgs['tke_prod'] = float(np.mean(prod))
        self.sgs['tke_diff'] = float(np.mean(diff))
        self.sgs['tke_diss'] = float(np.mean(diss))

        return self.sgs
