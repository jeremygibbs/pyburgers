"""Deardorff 1.5-order TKE SGS model.

Implements the prognostic subgrid TKE model following Deardorff.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .sgs import SGS
from utils import Dealias, Derivatives, Filter, get_logger

if TYPE_CHECKING:
    from utils.io import Input


class Deardorff(SGS):
    """Deardorff 1.5-order TKE subgrid-scale model.

    A prognostic SGS model that solves a transport equation for subgrid
    turbulent kinetic energy (TKE). The eddy viscosity is computed from
    the subgrid TKE as: nu_t = c1 * dx * sqrt(tke_sgs).

    Attributes:
        dealias: Dealias object for nonlinear terms.
        filter: Filter object (instantiated but unused).
        derivs: Derivatives object for TKE gradients.
    """

    def __init__(self, input_obj: Input, derivs: Derivatives | None = None) -> None:
        """Initialize the Deardorff TKE model.

        Args:
            input_obj: Input configuration object.
            derivs: Optional shared Derivatives object for performance.
                If not provided, creates a new instance.
        """
        super().__init__(input_obj)
        self.logger: logging.Logger = get_logger("SGS")
        self.logger.info("Using the Deardorff TKE model")
        self.dealias = Dealias(self.nx)
        self.filter = Filter(self.nx)
        # Use shared derivatives object if provided, otherwise create new one
        self.derivs = derivs if derivs is not None else Derivatives(self.nx, self.dx)

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
        ce = 0.70  # Dissipation coefficient
        c1 = 0.10  # Eddy viscosity coefficient

        # Dealiased strain rate squared
        dudx2 = self.dealias.compute(dudx)

        # Compute TKE gradients
        derivs_k = self.derivs.compute(tke_sgs, [1])
        dkdx = derivs_k['1']

        derivs_ku = self.derivs.compute(tke_sgs * u, [1])
        dkudx = derivs_ku['1']

        # Eddy viscosity and SGS stress
        Vt = c1 * self.dx * (tke_sgs ** 0.5)
        tau = -2.0 * Vt * dudx2

        # TKE diffusion term
        zz = 2 * Vt * dkdx
        derivs_zz = self.derivs.compute(zz, [1])
        dzzdx = derivs_zz["1"]

        # TKE tendency: advection + production + diffusion - dissipation
        dtke = (
            (-1 * dkudx)
            + (2 * Vt * dudx2 * dudx2)
            + dzzdx
            - (ce * (tke_sgs ** 1.5) / self.dx)
        ) * self.dt

        # Update subgrid TKE
        tke_sgs_new = tke_sgs + dtke

        self.sgs['tau'] = tau
        self.sgs['coeff'] = c1
        self.sgs['tke_sgs'] = tke_sgs_new

        return self.sgs
