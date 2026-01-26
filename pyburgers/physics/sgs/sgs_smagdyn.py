"""Dynamic Smagorinsky SGS model.

Implements the Germano dynamic procedure for computing the
Smagorinsky coefficient from the resolved field.
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


class SmagDynamic(SGS):
    """Dynamic Smagorinsky subgrid-scale model.

    Uses the Germano identity to dynamically compute the Smagorinsky
    coefficient from the resolved velocity field. This removes the
    need for tuning Cs and allows it to adapt to local flow conditions.

    Uses the shared spectral workspace for filtering and dealiasing operations.
    """

    def __init__(self, input_obj: Input, spectral: SpectralWorkspace) -> None:
        """Initialize the dynamic Smagorinsky model.

        Args:
            input_obj: Input configuration object.
            spectral: SpectralWorkspace with shared Dealias and Filter utilities.
        """
        super().__init__(input_obj, spectral)
        self.logger: logging.Logger = get_logger("SGS")
        self.logger.info("Using the Dynamic Smagorinsky model")

    def compute(
        self, u: np.ndarray, dudx: np.ndarray, tke_sgs: np.ndarray | float
    ) -> dict[str, Any]:
        """Compute the dynamic Smagorinsky SGS stress.

        Uses test filtering to compute the Leonard stress L and
        model tensor M, then determines Cs^2 from their contraction.

        Args:
            u: Velocity field array.
            dudx: Velocity gradient array.
            tke_sgs: Subgrid TKE (unused in this model).

        Returns:
            Dictionary with 'tau' (SGS stress) and 'coeff' (Cs).
        """
        # Model constants
        ratio = c.sgs.TEST_FILTER_RATIO

        # Leonard stress L11 = <uu> - <u><u>
        uf = self.spectral.filter.cutoff(u, ratio)
        uuf = self.spectral.filter.cutoff(u**2, ratio)
        L11 = uuf - uf * uf

        # Model tensor M11
        dudxf = self.spectral.filter.cutoff(dudx, ratio)
        T = np.abs(dudx) * dudx
        Tf = self.spectral.filter.cutoff(T, ratio)
        M11 = (self.dx**2) * ((ratio**2) * np.abs(dudxf) * dudxf - Tf)

        # Dealiased strain rate
        dudx2 = self.spectral.dealias.compute(dudx)

        # Dynamic Smagorinsky coefficient
        if np.mean(M11 * M11) == 0:
            cs2 = 0
        else:
            cs2 = 0.5 * np.mean(L11 * M11) / np.mean(M11 * M11)
            if cs2 < 0:
                cs2 = 0

        self.result["tau"] = -2 * cs2 * (self.dx**2) * dudx2
        self.result["coeff"] = np.sqrt(cs2)

        return self.result
