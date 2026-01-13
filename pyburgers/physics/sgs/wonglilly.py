"""Dynamic Wong-Lilly SGS model.

Implements the Wong-Lilly scale-similarity model with dynamic
coefficient computation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .sgs import SGS
from ...utils import get_logger

if TYPE_CHECKING:
    from ...utils.io import Input
    from ...utils.spectral_workspace import SpectralWorkspace


class WongLilly(SGS):
    """Dynamic Wong-Lilly subgrid-scale model.

    A scale-similarity based SGS model that uses a different scaling
    for the SGS stress compared to the Smagorinsky model. The stress
    scales as dx^(4/3) rather than dx^2.

    Uses the shared spectral workspace for filtering operations.
    """

    def __init__(
        self,
        input_obj: Input,
        spectral: SpectralWorkspace
    ) -> None:
        """Initialize the Wong-Lilly model.

        Args:
            input_obj: Input configuration object.
            spectral: SpectralWorkspace with shared Filter utility.
        """
        super().__init__(input_obj, spectral)
        self.logger: logging.Logger = get_logger("SGS")
        self.logger.info("Using the Wong-Lilly model")

    def compute(
        self,
        u: np.ndarray,
        dudx: np.ndarray,
        tke_sgs: np.ndarray | float
    ) -> dict[str, Any]:
        """Compute the Wong-Lilly SGS stress.

        Args:
            u: Velocity field array.
            dudx: Velocity gradient array.
            tke_sgs: Subgrid TKE (unused in this model).

        Returns:
            Dictionary with 'tau' (SGS stress) and 'coeff' (C_WL).
        """
        # Leonard stress L11
        uf = self.spectral.filter.cutoff(u, 2)
        uuf = self.spectral.filter.cutoff(u ** 2, 2)
        L11 = uuf - uf * uf

        # Model tensor M11 (Wong-Lilly scaling)
        dudxf = self.spectral.filter.cutoff(dudx, 2)
        M11 = self.dx ** (4 / 3) * (1 - 2 ** (4 / 3)) * dudxf

        # Wong-Lilly coefficient
        if np.mean(M11 * M11) == 0:
            cwl = 0
        else:
            cwl = 0.5 * np.mean(L11 * M11) / np.mean(M11 * M11)
            if cwl < 0:
                cwl = 0

        self.sgs['tau'] = -2 * cwl * (self.dx ** (4 / 3)) * dudx
        self.sgs['coeff'] = cwl

        return self.sgs
