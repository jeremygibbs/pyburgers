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

from ...data_models import SurfaceConfig
from ...exceptions import NamelistError
from ...util.io import logging_helper
from .noise import Noise
from .noise_fbm import FBM

logger = logging_helper.get_logger('Noise')

def get_noise_model(surface: SurfaceConfig) -> Surface:
    """Factory function to select and instantiate a noise model.

    Args:
        surface: Surface layer configuration.

    Returns:
        An instance of a concrete `Surface` subclass.

    Raises:
        NamelistError: If the provided `key` is not a valid model ID.
    """
    # Dictionary to map keys to classes
    sfc_models = {
        1: SurfaceMOST,
    }

    try:
        if surface.model == 1:
            return SurfaceMOST(psi_stable=surface.psi_stable)
        return sfc_models[surface.model]()
    except KeyError as e:
        error_msg = f'{surface.model} is an invalid surface model.'
        logger.error('x' * 62)
        logger.error('Namelist Error: %s', error_msg)
        logger.error('Valid options are:')
        for k, v in sfc_models.items():
            logger.error('\t%d (%s)', k, v.__name__)
        logger.error('x' * 62)
        raise NamelistError(error_msg) from e
