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
"""Dynamic Wong-Lilly SGS model.

Implements the Wong-Lilly scale-similarity model with dynamic
coefficient computation.
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


class WongLilly(SGS):
    """Dynamic Wong-Lilly subgrid-scale model.

    A scale-similarity based SGS model that uses a different scaling
    for the SGS stress compared to the Smagorinsky model. The stress
    scales as dx^(4/3) rather than dx^2.

    Uses the shared spectral workspace for filtering operations.
    """

    def __init__(self, input_obj: Input, spectral: SpectralWorkspace) -> None:
        """Initialize the Wong-Lilly model.

        Args:
            input_obj: Input configuration object.
            spectral: SpectralWorkspace with shared Filter utility.
        """
        super().__init__(input_obj, spectral)
        self.logger: logging.Logger = get_logger("SGS")
        self.logger.info("--- using the Wong-Lilly model")

        # Pre-allocate scratch arrays to avoid temporaries in the hot loop
        nx = self.nx
        self._scratch_a = np.zeros(nx)
        self._scratch_b = np.zeros(nx)
        self._scratch_c = np.zeros(nx)
        self._filt_a = np.zeros(nx)  # filter.cutoff output buffer
        self._filt_b = np.zeros(nx)  # filter.cutoff output buffer

    def compute(
        self, u: np.ndarray, dudx: np.ndarray, tke_sgs: np.ndarray | float, dt: float
    ) -> dict[str, Any]:
        """Compute the Wong-Lilly SGS stress.

        Args:
            u: Velocity field array.
            dudx: Velocity gradient array.
            tke_sgs: Subgrid TKE (unused in this model).
            dt: Current time step size (unused in this model).

        Returns:
            Dictionary with 'tau' (SGS stress) and 'coeff' (C_WL).
        """

        # Model constants
        ratio = c.sgs.TEST_FILTER_RATIO
        exponent = c.sgs.WONGLILLY_EXPONENT
        dx_exp = self.dx**exponent
        ratio_pow = ratio**exponent

        # Leonard stress L11 = <uu> - <u><u>
        np.square(u, out=self._scratch_a)                                    # u^2
        self.spectral.filter.cutoff(u, ratio, out=self._filt_a)              # uf
        self.spectral.filter.cutoff(self._scratch_a, ratio, out=self._filt_b)  # uuf
        np.square(self._filt_a, out=self._scratch_a)                         # uf^2
        np.subtract(self._filt_b, self._scratch_a, out=self._scratch_a)      # L11

        # Model tensor M11 (Wong-Lilly scaling; _filt_a now free for reuse)
        self.spectral.filter.cutoff(dudx, ratio, out=self._filt_a)           # dudxf
        np.multiply(dx_exp * (1 - ratio_pow), self._filt_a, out=self._scratch_b)  # M11

        # Wong-Lilly coefficient
        np.square(self._scratch_b, out=self._scratch_c)
        M11_sq_mean = np.mean(self._scratch_c)
        if M11_sq_mean < 1e-30:
            cwl = 0.0
        else:
            np.multiply(self._scratch_a, self._scratch_b, out=self._scratch_c)
            cwl = 0.5 * np.mean(self._scratch_c) / M11_sq_mean
            if cwl < 0:
                cwl = 0.0

        np.multiply(-2 * cwl * dx_exp, dudx, out=self.result["tau"])
        self.result["coeff"] = cwl

        return self.result
