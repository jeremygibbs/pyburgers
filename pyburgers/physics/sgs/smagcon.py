"""Constant-coefficient Smagorinsky SGS model.

Implements the classic Smagorinsky model with a fixed coefficient.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .sgs import SGS
from ...utils import Dealias, get_logger

if TYPE_CHECKING:
    from ...utils.io import Input


class SmagConstant(SGS):
    """Constant-coefficient Smagorinsky subgrid-scale model.

    The classic Smagorinsky model computes SGS stress as:
        tau = -2 * Cs^2 * dx^2 * |S| * S

    where Cs is a constant (typically 0.16) and S is the strain rate.

    Attributes:
        dealias: Dealias object for computing |dudx| * dudx.
    """

    def __init__(self, input_obj: Input) -> None:
        """Initialize the constant Smagorinsky model.

        Args:
            input_obj: Input configuration object.
        """
        super().__init__(input_obj)
        self.logger: logging.Logger = get_logger("SGS")
        self.logger.info("Using the Smagorinsky model")
        self.dealias = Dealias(
            self.nx,
            fftw_planning=self.fftw_planning,
            fftw_threads=self.fftw_threads,
        )

    def compute(
        self,
        u: np.ndarray,
        dudx: np.ndarray,
        tke_sgs: np.ndarray | float
    ) -> dict[str, Any]:
        """Compute the Smagorinsky SGS stress.

        Args:
            u: Velocity field array (unused).
            dudx: Velocity gradient array.
            tke_sgs: Subgrid TKE (unused in this model).

        Returns:
            Dictionary with 'tau' (SGS stress) and 'coeff' (Cs).
        """
        cs2 = 0.16 ** 2
        dudx2 = self.dealias.compute(dudx)

        self.sgs['tau'] = -2 * cs2 * (self.dx ** 2) * dudx2
        self.sgs['coeff'] = np.sqrt(cs2)

        return self.sgs
