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

"""The core PyBurgers model orchestrator.

This module contains the main `PyBurgers` class, which serves as the central
controller for the models. It initializes the components and steps through the 
simulation in time.
"""

import logging

import numpy as np