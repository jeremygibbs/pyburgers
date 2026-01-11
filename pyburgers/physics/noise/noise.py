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
that any concrete noise model implements the necessary methods, and a factory 
function (`get_model`) for creating instances of those models.
"""

import logging
from abc import ABC, abstractmethod
from typing import TypeVar

from ...util.io import logging_helper
ST = TypeVar('NT', bound='Noise')
logger = logging_helper.get_logger('Noise')

class Noise(ABC):
    """Abstract base class for noise models.
    
    This class defines the standard interface for noise calculations and acts as a 
    factory for creating specific noise model instances. It cannot be instantiated 
    directly.
    
    Attributes:
        logger: A logger for this class.
        latitude: The site latitude in degrees.
        longitude: The site longitude in degrees.
        albedo: The surface albedo (dimensionless).
        emissivity: The surface emissivity (dimensionless).
    """
    def __init__(self) -> None:
        """Initializes the Noise base class."""
        self.logger: logging.Logger = logging_helper.get_logger('Noise')
    
    #--- Abstract Methods ---
    
    @abstractmethod
    def compute_noise(self) -> NDArray[np.float64]:
        """Computes noise.
    
        This method must be implemented by subclasses to compute the water
        conductivity as a function of soil moisture using the soil's
        water retention and conductivity relationships.
    
        Args:
            soil_q: Soil moisture content [m^3/m^3]. Can be a scalar value
                or an array of values.
            level: Optional specific soil layer index. If provided with a
                scalar soil_q, indicates which layer the moisture belongs to.
    
        Returns:
            Moisture conductivity [m/s]. Returns the same type as soil_q
            (float or NDArray).
        """
        raise NotImplementedError

