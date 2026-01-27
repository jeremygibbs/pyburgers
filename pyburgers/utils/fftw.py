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
"""FFTW wisdom management for PyBurgers.

This module handles loading and saving FFTW wisdom to disk, which allows
FFT plans to be reused across runs for faster initialization.

The wisdom file is stored at ~/.pyburgers_fftw_wisdom.

File locking is used to prevent race conditions when multiple PyBurgers
instances access the wisdom file concurrently.
"""

from __future__ import annotations

import fcntl
import pickle
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pyfftw

# Wisdom cache file location
WISDOM_FILE = Path.home() / ".pyburgers_fftw_wisdom"

# Lock timeout in seconds
LOCK_TIMEOUT = 10.0


@contextmanager
def _file_lock(file_path: Path, exclusive: bool = False) -> Iterator[None]:
    """Context manager for file locking.

    Uses fcntl advisory locks to prevent concurrent access issues.
    Multiple readers can hold shared locks simultaneously, but
    only one writer can hold an exclusive lock at a time.

    Args:
        file_path: Path to file to lock.
        exclusive: If True, acquire exclusive lock (write).
                   If False, acquire shared lock (read).

    Yields:
        None

    Raises:
        TimeoutError: If lock cannot be acquired within timeout.
        OSError: If file operations fail.
    """
    # Create lock file (append .lock to original filename)
    lock_file_path = Path(str(file_path) + ".lock")
    lock_file_path.touch(exist_ok=True)

    lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    lock_file = None

    try:
        lock_file = open(lock_file_path, "r+")

        # Try to acquire lock with timeout
        start_time = time.time()
        while True:
            try:
                # Non-blocking lock attempt
                fcntl.flock(lock_file.fileno(), lock_type | fcntl.LOCK_NB)
                break
            except BlockingIOError as err:
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed > LOCK_TIMEOUT:
                    raise TimeoutError(
                        f"Could not acquire {'exclusive' if exclusive else 'shared'} "
                        f"lock on {file_path} within {LOCK_TIMEOUT}s"
                    ) from err
                # Wait a bit before retrying
                time.sleep(0.1)

        # Lock acquired, yield control
        yield

    finally:
        # Release lock and close file
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception:
                # Ignore errors during cleanup
                pass


def load_wisdom(
    nx_dns: int,
    nx_les: int,
    noise_alpha: float,
    fftw_planning: str,
    fftw_threads: int,
) -> tuple[bool, str]:
    """Load FFTW wisdom from cache file if parameters match.

    FFTW wisdom contains optimized FFT plans from previous runs.
    Loading wisdom speeds up FFT initialization significantly.

    This function validates that the cached wisdom was created with
    the same grid sizes and parameters. If parameters have changed,
    the wisdom is invalidated and False is returned to trigger re-warmup.

    Uses shared file locking to allow concurrent reads while preventing
    read/write conflicts.

    Args:
        nx_dns: DNS grid resolution.
        nx_les: LES grid resolution.
        noise_alpha: FBM noise exponent.
        fftw_planning: FFTW planning strategy.
        fftw_threads: Number of FFTW threads.

    Returns:
        Tuple of (success: bool, message: str) indicating whether wisdom
        was loaded and a descriptive message about the outcome.
    """
    if not WISDOM_FILE.exists():
        return False, "No wisdom file found"

    try:
        # Acquire shared lock for reading (multiple readers OK)
        with _file_lock(WISDOM_FILE, exclusive=False):
            with open(WISDOM_FILE, "rb") as f:
                data = pickle.load(f)

        # Handle legacy wisdom files (raw wisdom without metadata)
        if not isinstance(data, dict):
            return False, "Legacy wisdom format detected (no metadata)"

        # Extract wisdom and metadata
        wisdom = data.get("wisdom")
        metadata = data.get("metadata", {})

        # Check each parameter and build a detailed message
        mismatches = []
        if metadata.get("nx_dns") != nx_dns:
            mismatches.append(f"nx_dns ({metadata.get('nx_dns')} → {nx_dns})")
        if metadata.get("nx_les") != nx_les:
            mismatches.append(f"nx_les ({metadata.get('nx_les')} → {nx_les})")
        if metadata.get("noise_alpha") != noise_alpha:
            mismatches.append(f"noise_alpha ({metadata.get('noise_alpha')} → {noise_alpha})")
        if metadata.get("fftw_planning") != fftw_planning:
            mismatches.append(
                f"fftw_planning ({metadata.get('fftw_planning')} → {fftw_planning})"
            )
        if metadata.get("fftw_threads") != fftw_threads:
            mismatches.append(f"fftw_threads ({metadata.get('fftw_threads')} → {fftw_threads})")

        if mismatches:
            msg = "Parameter mismatch: " + ", ".join(mismatches)
            return False, msg

        # Import the validated wisdom
        pyfftw.import_wisdom(wisdom)
        return True, "Wisdom loaded successfully"

    except TimeoutError as e:
        return False, f"Lock timeout: {e}"
    except Exception as e:
        return False, f"Error loading wisdom: {e}"


def save_wisdom(
    nx_dns: int,
    nx_les: int,
    noise_alpha: float,
    fftw_planning: str,
    fftw_threads: int,
) -> bool:
    """Save FFTW wisdom with metadata to cache file.

    Saves the accumulated FFT plans along with the grid sizes and
    parameters used to create them. This allows validation on load
    to ensure the cached plans match the current configuration.

    Uses exclusive file locking to prevent concurrent writes and
    read/write conflicts.

    Args:
        nx_dns: DNS grid resolution.
        nx_les: LES grid resolution.
        noise_alpha: FBM noise exponent.
        fftw_planning: FFTW planning strategy.
        fftw_threads: Number of FFTW threads.

    Returns:
        True if wisdom was saved successfully, False otherwise.
    """
    try:
        # Package wisdom with metadata
        data = {
            "wisdom": pyfftw.export_wisdom(),
            "metadata": {
                "nx_dns": nx_dns,
                "nx_les": nx_les,
                "noise_alpha": noise_alpha,
                "fftw_planning": fftw_planning,
                "fftw_threads": fftw_threads,
            },
        }

        # Acquire exclusive lock for writing (blocks all other access)
        with _file_lock(WISDOM_FILE, exclusive=True):
            with open(WISDOM_FILE, "wb") as f:
                pickle.dump(data, f)
        return True
    except TimeoutError:
        # Could not acquire lock - another process is accessing file
        return False
    except Exception:
        return False


def warmup_fftw_plans(
    nx_dns: int,
    nx_les: int,
    noise_alpha: float,
    fftw_planning: str,
    fftw_threads: int,
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
        noise_alpha: FBM noise exponent.
        fftw_planning: FFTW planning strategy.
        fftw_threads: Number of FFTW threads.

    Returns:
        Tuple of (success: bool, message: str) indicating whether warmup
        completed successfully and any relevant diagnostic information.
    """
    import numpy as np

    from .spectral_workspace import SpectralWorkspace

    try:
        # Warm DNS components using SpectralWorkspace
        if nx_dns > 0:
            try:
                SpectralWorkspace(
                    nx=nx_dns,
                    dx=2 * np.pi / nx_dns,
                    noise_alpha=noise_alpha,
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
                    dx=2 * np.pi / nx_les,
                    nx2=nx_dns,
                    noise_alpha=noise_alpha,
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
