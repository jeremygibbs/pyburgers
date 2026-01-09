"""Global configuration for pyBurgers.

This module provides centralized configuration for FFT operations
and other simulation parameters that should be consistent across
all modules.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import pyfftw

# FFT Configuration
# Can be overridden via environment variables
FFTW_THREADS = int(os.environ.get('PYBURGERS_FFTW_THREADS', 4))
FFTW_PLANNING = os.environ.get('PYBURGERS_FFTW_PLANNING', 'FFTW_ESTIMATE')

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
