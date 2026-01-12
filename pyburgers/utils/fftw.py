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
"""FFTW wisdom management for pyBurgers.

This module handles loading and saving FFTW wisdom to disk, which allows
FFT plans to be reused across runs for faster initialization.

The wisdom file is stored at ~/.pyburgers_fftw_wisdom.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pyfftw
import pyfftw.interfaces.cache as fftw_cache

# Enable interface cache for automatic FFT function reuse
fftw_cache.enable()
fftw_cache.set_keepalive_time(30)

# Wisdom cache file location
WISDOM_FILE = Path.home() / '.pyburgers_fftw_wisdom'


def load_wisdom() -> bool:
    """Load FFTW wisdom from cache file.

    FFTW wisdom contains optimized FFT plans from previous runs.
    Loading wisdom speeds up FFT initialization significantly.

    Returns:
        True if wisdom was loaded successfully, False otherwise.
    """
    if WISDOM_FILE.exists():
        try:
            with open(WISDOM_FILE, 'rb') as f:
                pyfftw.import_wisdom(pickle.load(f))
            return True
        except Exception:
            # Ignore corrupted or incompatible wisdom
            pass
    return False


def save_wisdom() -> bool:
    """Save FFTW wisdom to cache file.

    Saves the accumulated FFT plans so they can be reused in future runs.

    Returns:
        True if wisdom was saved successfully, False otherwise.
    """
    try:
        with open(WISDOM_FILE, 'wb') as f:
            pickle.dump(pyfftw.export_wisdom(), f)
        return True
    except Exception:
        return False


def warmup_fftw_plans(
    nx_dns: int,
    nx_les: int,
    noise_alpha: float,
    fftw_planning: str,
    fftw_threads: int,
) -> None:
    """Generate FFTW plans for common PyBurgers sizes.

    This creates representative FFTW plans for DNS/LES grids, filters,
    dealiasing, and FBM noise so wisdom can be saved once and reused.
    """
    import numpy as np

    from .spectral import Dealias, Derivatives, Filter
    from ..physics.noise import get_noise_model

    def warm_derivatives(nx: int) -> None:
        Derivatives(
            nx,
            2 * np.pi / nx,
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads,
        )

    if nx_dns > 0:
        warm_derivatives(nx_dns)
        get_noise_model(
            1,
            noise_alpha,
            nx_dns,
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads,
        )

    if nx_les > 0:
        warm_derivatives(nx_les)
        Dealias(
            nx_les,
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads,
        )
        Filter(
            nx_les,
            nx2=nx_dns,
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads,
        )
