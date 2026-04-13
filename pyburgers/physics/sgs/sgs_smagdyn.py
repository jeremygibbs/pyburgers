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
        self.logger.info("--- using the Dynamic Smagorinsky model")

        # Pre-allocate scratch arrays to avoid temporaries in the hot loop
        nx = self.nx
        self._scratch_a = np.zeros(nx)
        self._scratch_b = np.zeros(nx)
        self._scratch_c = np.zeros(nx)

    def compute(
        self, u: np.ndarray, dudx: np.ndarray, tke_sgs: np.ndarray | float, dt: float
    ) -> dict[str, Any]:
        """Compute the dynamic Smagorinsky SGS stress.

        Uses test filtering to compute the Leonard stress L and
        model tensor M, then determines Cs^2 from their contraction.

        Args:
            u: Velocity field array.
            dudx: Velocity gradient array.
            tke_sgs: Subgrid TKE (unused in this model).
            dt: Current time step size (unused in this model).

        Returns:
            Dictionary with 'tau' (SGS stress) and 'coeff' (Cs).
        """
        # Model constants
        ratio = c.sgs.TEST_FILTER_RATIO
        dx2 = self.dx**2
        ratio2 = ratio**2

        # Leonard stress L11 = <uu> - <u><u>
        np.square(u, out=self._scratch_a)               # u^2
        uf = self.spectral.filter.cutoff(u, ratio)
        uuf = self.spectral.filter.cutoff(self._scratch_a, ratio)
        np.square(uf, out=self._scratch_a)               # uf^2
        np.subtract(uuf, self._scratch_a, out=self._scratch_a)  # L11

        # Model tensor M11
        dudxf = self.spectral.filter.cutoff(dudx, ratio)
        np.abs(dudx, out=self._scratch_b)
        np.multiply(self._scratch_b, dudx, out=self._scratch_b)  # |dudx|*dudx
        Tf = self.spectral.filter.cutoff(self._scratch_b, ratio)
        np.abs(dudxf, out=self._scratch_b)
        np.multiply(self._scratch_b, dudxf, out=self._scratch_b)  # |dudxf|*dudxf
        np.multiply(ratio2, self._scratch_b, out=self._scratch_b)
        np.subtract(self._scratch_b, Tf, out=self._scratch_b)
        np.multiply(dx2, self._scratch_b, out=self._scratch_b)    # M11

        # Dealiased strain rate
        dudx2 = self.spectral.dealias.compute(dudx)

        # Dynamic Smagorinsky coefficient
        np.multiply(self._scratch_b, self._scratch_b, out=self._scratch_c)
        M11_sq_mean = np.mean(self._scratch_c)
        if M11_sq_mean < 1e-30:
            cs2 = 0.0
        else:
            np.multiply(self._scratch_a, self._scratch_b, out=self._scratch_c)
            cs2 = -0.5 * np.mean(self._scratch_c) / M11_sq_mean
            if cs2 < 0:
                cs2 = 0.0

        np.multiply(-2 * cs2 * dx2, dudx2, out=self.result["tau"])
        self.result["coeff"] = np.sqrt(cs2)

        return self.result
