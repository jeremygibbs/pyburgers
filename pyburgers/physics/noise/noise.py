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
"""Abstract base class for noise models in PyBurgers.

This module defines the interface for all noise generation used within the
PyBurgers framework. It provides the `Noise` abstract base class, which ensures
that any concrete noise model implements the necessary methods.
"""
import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ...utils.logging_helper import get_logger


class Noise(ABC):
    """Abstract base class for noise models.

    This class defines the standard interface for noise calculations.
    It cannot be instantiated directly.

    Attributes:
        logger: A logger for this class.
        n_pts: Number of grid points.
        alpha: Spectral exponent controlling noise characteristics.
        fftw_planning: FFTW planning strategy.
        fftw_threads: Number of threads for FFTW.
    """

    def __init__(
        self,
        alpha: float,
        n_pts: int,
        fftw_planning: str,
        fftw_threads: int
    ) -> None:
        """Initialize the Noise base class.

        Args:
            alpha: Spectral exponent for noise coloring.
            n_pts: Number of grid points.
            fftw_planning: FFTW planning strategy.
            fftw_threads: Number of threads for FFTW.
        """
        self.logger: logging.Logger = get_logger('Noise')
        self.alpha = alpha
        self.n_pts = n_pts
        self.fftw_planning = fftw_planning
        self.fftw_threads = fftw_threads

    @abstractmethod
    def compute_noise(self) -> NDArray[np.float64]:
        """Compute a realization of the noise field.

        This method must be implemented by subclasses to generate
        noise with the appropriate spectral characteristics.

        Returns:
            Noise array of shape (n_pts,).
        """
        raise NotImplementedError
