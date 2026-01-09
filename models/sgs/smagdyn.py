"""Dynamic Smagorinsky SGS model.

Implements the Germano dynamic procedure for computing the
Smagorinsky coefficient from the resolved field.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .sgs import SGS
from utils import Dealias, Filter, get_logger

if TYPE_CHECKING:
    from utils.io import Input


class SmagDynamic(SGS):
    """Dynamic Smagorinsky subgrid-scale model.

    Uses the Germano identity to dynamically compute the Smagorinsky
    coefficient from the resolved velocity field. This removes the
    need for tuning Cs and allows it to adapt to local flow conditions.

    Attributes:
        dealias: Dealias object for computing nonlinear terms.
        filter: Filter object for test filtering.
    """

    def __init__(self, input_obj: Input) -> None:
        """Initialize the dynamic Smagorinsky model.

        Args:
            input_obj: Input configuration object.
        """
        super().__init__(input_obj)
        self.logger: logging.Logger = get_logger("SGS")
        self.logger.info("Using the Dynamic Smagorinsky model")
        self.dealias = Dealias(self.nx)
        self.filter = Filter(self.nx)

    def compute(
        self,
        u: np.ndarray,
        dudx: np.ndarray,
        tke_sgs: np.ndarray | float
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
        # Leonard stress L11 = <uu> - <u><u>
        uf = self.filter.cutoff(u, 2)
        uuf = self.filter.cutoff(u ** 2, 2)
        L11 = uuf - uf * uf

        # Model tensor M11
        dudxf = self.filter.cutoff(dudx, 2)
        T = np.abs(dudx) * dudx
        Tf = self.filter.cutoff(T, 2)
        M11 = (self.dx ** 2) * (4 * np.abs(dudxf) * dudxf - Tf)

        # Dealiased strain rate
        dudx2 = self.dealias.compute(dudx)

        # Dynamic Smagorinsky coefficient
        if np.mean(M11 * M11) == 0:
            cs2 = 0
        else:
            cs2 = 0.5 * np.mean(L11 * M11) / np.mean(M11 * M11)
            if cs2 < 0:
                cs2 = 0

        self.sgs['tau'] = -2 * cs2 * (self.dx ** 2) * dudx2
        self.sgs['coeff'] = np.sqrt(cs2)

        return self.sgs
