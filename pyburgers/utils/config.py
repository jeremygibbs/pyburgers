"""Global configuration for pyBurgers.

This module provides centralized configuration for FFT operations
and other simulation parameters that should be consistent across
all modules.

FFTW Configuration:
Namelist settings (fftw.planning, fftw.threads in namelist.json)

The main script (burgers.py) will override these values with namelist
settings after the Input object is created.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import pyfftw
import pyfftw.interfaces.cache as fftw_cache

# FFT Configuration - will be set from namelist by burgers.py
# Default values if not set
FFTW_PLANNING = 'FFTW_MEASURE'
FFTW_THREADS = 4

# Enable interface cache for automatic FFT function reuse
fftw_cache.enable()
fftw_cache.set_keepalive_time(30)

# Wisdom cache file location
WISDOM_FILE = Path.home() / '.pyburgers_fftw_wisdom'

def load_wisdom() -> bool:
    """Load FFTW wisdom from cache file.

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

    Returns:
        True if wisdom was saved successfully, False otherwise.
    """
    try:
        with open(WISDOM_FILE, 'wb') as f:
            pickle.dump(pyfftw.export_wisdom(), f)
        return True
    except Exception:
        return False
