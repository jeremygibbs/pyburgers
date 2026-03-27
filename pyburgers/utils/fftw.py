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
"""FFTW wisdom management for PyBurgers.

This module handles loading and saving FFTW wisdom to disk, which allows
FFT plans to be reused across runs for faster initialization.

The wisdom file is stored at ~/.pyburgers_fftw_wisdom.

File locking is used to prevent race conditions when multiple PyBurgers
instances access the wisdom file concurrently.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pyfftw
from filelock import FileLock, Timeout

# Wisdom cache file location
WISDOM_FILE = Path.home() / ".pyburgers_fftw_wisdom"

# Lock timeout in seconds
LOCK_TIMEOUT = 10.0


def load_wisdom() -> tuple[bool, str]:
    """Load accumulated FFTW wisdom from cache file.

    FFTW wisdom is cumulative: a single file can hold optimal plans for
    many different transform sizes. This function imports whatever plans
    are present; pyfftw will use matching plans and re-plan any missing
    sizes on-demand.

    No parameter validation is performed — grid sizes and physical
    parameters do not affect whether a stored plan is valid.

    Uses file locking to prevent concurrent access conflicts.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if not WISDOM_FILE.exists():
        return False, "No wisdom file found"

    try:
        with FileLock(str(WISDOM_FILE) + ".lock", timeout=LOCK_TIMEOUT):
            with open(WISDOM_FILE, "rb") as f:
                data = pickle.load(f)

        # Support both legacy dict format and current raw wisdom format
        wisdom = data.get("wisdom") if isinstance(data, dict) else data
        if wisdom is None:
            return False, "No wisdom data in file"

        pyfftw.import_wisdom(wisdom)
        return True, "Wisdom loaded successfully"

    except Timeout:
        return False, f"Lock timeout after {LOCK_TIMEOUT}s"
    except Exception as e:
        return False, f"Error loading wisdom: {e}"


def save_wisdom() -> bool:
    """Save accumulated FFTW wisdom to cache file.

    Exports all plans that have been created in this session and writes
    them to the cache file, merging with any plans already present.
    Plans accumulate across runs so that each new configuration adds to
    the file rather than replacing it.

    Uses file locking to prevent concurrent write conflicts.

    Returns:
        True if wisdom was saved successfully, False otherwise.
    """
    try:
        with FileLock(str(WISDOM_FILE) + ".lock", timeout=LOCK_TIMEOUT):
            with open(WISDOM_FILE, "wb") as f:
                pickle.dump(pyfftw.export_wisdom(), f)
        return True
    except Timeout:
        return False
    except Exception:
        return False


def warmup_fftw_plans(
    nx_dns: int,
    nx_les: int,
    noise_beta: float,
    fftw_planning: str,
    fftw_threads: int,
    domain_length: float = 2 * 3.141592653589793,
) -> tuple[bool, str]:
    """Generate FFTW plans for common PyBurgers sizes.

    This creates representative FFTW plans for DNS/LES grids, filters,
    dealiasing, and FBM noise so wisdom can be saved once and reused.

    If warmup fails (e.g., out of memory, invalid parameters), the function
    returns False and a descriptive error message. The simulation can still
    continue - plans will be created on-demand during initialization.

    Args:
        nx_dns: DNS grid resolution.
        nx_les: LES grid resolution.
        noise_beta: FBM noise exponent.
        fftw_planning: FFTW planning strategy.
        fftw_threads: Number of FFTW threads.
        domain_length: Length of the periodic domain (default: 2π).

    Returns:
        Tuple of (success: bool, message: str) indicating whether warmup
        completed successfully and any relevant diagnostic information.
    """
    from .spectral_workspace import SpectralWorkspace

    try:
        # Warm DNS components using SpectralWorkspace
        if nx_dns > 0:
            try:
                SpectralWorkspace(
                    nx=nx_dns,
                    dx=domain_length / nx_dns,
                    noise_beta=noise_beta,
                    noise_nx=nx_dns,
                    fftw_planning=fftw_planning,
                    fftw_threads=fftw_threads,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create DNS workspace (nx={nx_dns}): {e}") from e

        # Warm LES components using SpectralWorkspace
        if nx_les > 0:
            try:
                SpectralWorkspace(
                    nx=nx_les,
                    dx=domain_length / nx_les,
                    nx2=nx_dns,
                    noise_beta=noise_beta,
                    noise_nx=nx_dns,
                    fftw_planning=fftw_planning,
                    fftw_threads=fftw_threads,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create LES workspace (nx={nx_les}, nx2={nx_dns}): {e}"
                ) from e

        return True, "FFTW plans created successfully"

    except Exception as e:
        # Warmup failed, but this is not fatal - plans will be created on demand
        return False, f"Warmup failed: {e}"
