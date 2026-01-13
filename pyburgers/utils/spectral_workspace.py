"""Centralized spectral operations workspace for pyBurgers.

This module provides the SpectralWorkspace class that bundles all spectral
utilities (Derivatives, Dealias, Filter, and FBM noise) into a single,
reusable workspace. This eliminates redundant FFT plan creation and reduces
memory usage.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pyburgers.utils.spectral import Derivatives, Dealias, Filter
from pyburgers.utils.fbm import FBM

if TYPE_CHECKING:
    import numpy as np


class SpectralWorkspace:
    """Centralized workspace for all spectral operations.

    Manages FFT plans and aligned buffers for a simulation by bundling
    Derivatives, Dealias, Filter, and FBM noise utilities into a single
    workspace. All utilities share consistent FFTW settings (planning
    strategy, threads).

    This design eliminates redundant FFT plan creation and ensures that
    resources are shared efficiently across the simulation.

    Attributes:
        derivatives: Derivatives calculator for spatial derivatives.
        dealias: Dealias calculator for nonlinear products.
        filter: Filter for spectral cutoff and downscaling.
        noise: FBM noise generator (only if noise_alpha provided).
        u: Primary velocity buffer (reference to derivatives.u).
        fu: Primary Fourier space buffer (reference to derivatives.fu).

    Example:
        >>> # DNS workspace with noise
        >>> workspace = SpectralWorkspace(nx=512, dx=0.01, noise_alpha=0.75)
        >>> derivs = workspace.derivatives.compute(u, order=[1, 2])
        >>> noise = workspace.noise.compute_noise()
        >>> # LES workspace with filtering and noise at DNS resolution
        >>> workspace_les = SpectralWorkspace(
        ...     nx=512, dx=0.01, nx2=8192, noise_alpha=0.75, noise_nx=8192
        ... )
        >>> filtered = workspace_les.filter.cutoff(x, ratio=2)
    """

    def __init__(
        self,
        nx: int,
        dx: float,
        nx2: int | None = None,
        noise_alpha: float | None = None,
        noise_nx: int | None = None,
        fftw_planning: str = 'FFTW_MEASURE',
        fftw_threads: int = 1
    ) -> None:
        """Initialize the spectral workspace.

        Args:
            nx: Number of grid points for the simulation.
            dx: Grid spacing.
            nx2: Optional number of grid points for DNS resolution
                (used in LES mode for downscaling noise from DNS to LES grid).
                If provided, creates Filter with downscaling capability.
            noise_alpha: Optional FBM exponent for noise generation.
                If provided, creates FBM noise generator. Typical value is 0.75.
            noise_nx: Optional number of grid points for noise generation.
                If not provided and noise_alpha is given, uses nx.
                For LES, set this to nx_dns to generate noise at DNS resolution.
            fftw_planning: FFTW planning strategy. Options:
                - 'FFTW_ESTIMATE': Fast planning, slower execution
                - 'FFTW_MEASURE': Balanced (default)
                - 'FFTW_PATIENT': Slow planning, faster execution
            fftw_threads: Number of threads for FFTW operations.
        """
        # Store configuration
        self.nx = nx
        self.dx = dx
        self.nx2 = nx2
        self.noise_alpha = noise_alpha
        self.noise_nx = noise_nx if noise_nx is not None else nx
        self.fftw_planning = fftw_planning
        self.fftw_threads = fftw_threads

        # Initialize all spectral utilities with consistent settings
        self.derivatives = Derivatives(
            nx=nx,
            dx=dx,
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads
        )

        self.dealias = Dealias(
            nx=nx,
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads
        )

        # Always create Filter (SGS models need it for test filtering)
        # If nx2 provided, Filter can also do downscaling from DNS to LES grid
        self.filter = Filter(
            nx=nx,
            nx2=nx2,  # Optional: None for basic filtering, set for downscaling
            fftw_planning=fftw_planning,
            fftw_threads=fftw_threads
        )

        # Optionally create FBM noise generator
        # If noise_nx differs from nx (LES case), noise is generated at noise_nx
        # resolution and must be filtered down using self.filter.downscale()
        if noise_alpha is not None:
            self.noise = FBM(
                alpha=noise_alpha,
                n_pts=self.noise_nx,
                fftw_planning=fftw_planning,
                fftw_threads=fftw_threads
            )
        else:
            self.noise = None

        # Expose commonly used buffers for direct access
        # This allows code like: workspace.u[:] = initial_condition
        self.u: np.ndarray = self.derivatives.u
        self.fu: np.ndarray = self.derivatives.fu

    def __repr__(self) -> str:
        """String representation of the workspace."""
        filter_info = f", nx2={self.nx2}" if self.nx2 else ""
        noise_info = f", noise_alpha={self.noise_alpha}, noise_nx={self.noise_nx}" if self.noise_alpha else ""
        return (
            f"SpectralWorkspace(nx={self.nx}, dx={self.dx}{filter_info}{noise_info}, "
            f"fftw_planning='{self.fftw_planning}', fftw_threads={self.fftw_threads})"
        )
