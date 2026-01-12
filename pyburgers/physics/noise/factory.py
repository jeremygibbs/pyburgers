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
"""Factory for creating noise model instances."""
from ...exceptions import NamelistError
from ...utils.logging_helper import get_logger
from .noise import Noise
from .noise_fbm import FBM

logger = get_logger('Noise')


def get_noise_model(
    model_id: int,
    alpha: float,
    n_pts: int,
    fftw_planning: str,
    fftw_threads: int,
) -> Noise:
    """Factory function to select and instantiate a noise model.

    Args:
        model_id: Integer ID for the noise model (1 = FBM).
        alpha: Spectral exponent for noise coloring.
        n_pts: Number of grid points.
        fftw_planning: FFTW planning strategy.
        fftw_threads: Number of FFTW threads.

    Returns:
        An instance of a concrete `Noise` subclass.

    Raises:
        NamelistError: If the provided `model_id` is not a valid model ID.
    """
    noise_models = {
        1: FBM,
    }

    try:
        model_class = noise_models[model_id]
        return model_class(
            alpha,
            n_pts,
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads,
        )
    except KeyError as e:
        error_msg = f'{model_id} is an invalid noise model.'
        logger.error('x' * 62)
        logger.error('Namelist Error: %s', error_msg)
        logger.error('Valid options are:')
        for k, v in noise_models.items():
            logger.error('\t%d (%s)', k, v.__name__)
        logger.error('x' * 62)
        raise NamelistError(error_msg) from e
