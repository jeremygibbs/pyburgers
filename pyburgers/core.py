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
"""Core Burgers equation solver base class.

This module provides the abstract base class for solving the 1D stochastic
Burgers equation. Both DNS and LES solvers inherit from this class.
"""
from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pyfftw

from .utils import get_logger
from .utils.spectral_workspace import SpectralWorkspace

if TYPE_CHECKING:
    from .utils.io import Input, Output


class Burgers(ABC):
    """Abstract base class for Burgers equation solvers.

    Provides common functionality for solving the 1D stochastic Burgers
    equation using Fourier collocation for spatial derivatives and
    Adams-Bashforth time integration.

    Subclasses must implement:
        - _get_nx(): Return the grid resolution for this mode
        - _create_spectral_workspace(): Create the spectral workspace for this mode
        - _setup_mode_specific(): Initialize mode-specific components
        - _setup_output_fields(): Configure output fields dictionary
        - _compute_noise(): Generate/process noise for this time step
        - _compute_rhs(): Compute the right-hand side of the equation
        - _save_diagnostics(): Compute and save mode-specific diagnostics

    Attributes:
        input: Input configuration object.
        output: Output handler for NetCDF writing.
        nx: Number of grid points.
        dx: Grid spacing.
        dt: Time step.
        nt: Number of time steps.
        visc: Kinematic viscosity.
        noise_amp: Noise amplitude.
        step_save: Output save interval.
    """

    # Mode name for logging (override in subclasses)
    mode_name: str = "Burgers"

    def __init__(self, input_obj: Input, output_obj: Output) -> None:
        """Initialize the Burgers solver.

        Args:
            input_obj: Input configuration containing simulation parameters.
            output_obj: Output handler for writing results to NetCDF.
        """
        self.logger: logging.Logger = get_logger(self.mode_name)
        self.logger.info("You are running in %s mode", self.mode_name)

        # Initialize random number generator for reproducibility
        np.random.seed(1)

        # Store input/output objects
        self.logger.debug("Reading input settings")
        self.input = input_obj
        self.output = output_obj

        # Extract common configuration
        self.dt = input_obj.dt
        self.nt = input_obj.nt
        self.visc = input_obj.viscosity
        self.noise_amp = input_obj.physics.noise.amplitude
        self.noise_alpha = input_obj.physics.noise.alpha
        self.step_save = input_obj.step_save
        self.fftw_planning = input_obj.fftw_planning
        self.fftw_threads = input_obj.fftw_threads

        # Get mode-specific grid resolution
        self.nx = self._get_nx()
        self.mp = self.nx // 2
        self.dx = 2 * np.pi / self.nx

        # Create spectral workspace (bundles Derivatives, Dealias, Filter)
        self.spectral = self._create_spectral_workspace()

        # Grid coordinates
        self.x = np.arange(0, 2 * np.pi, self.dx)

        # Reference workspace buffers (zero-copy)
        self.u = self.spectral.u
        self.fu = self.spectral.fu

        # Initialize velocity field to zero
        self.u[:] = 0

        # Common output field
        self.tke = np.zeros(1)

        # Mode-specific setup (noise, SGS, etc.)
        self._setup_mode_specific()

        # Setup output
        self.output_dims = {'t': 0, 'x': self.nx}
        self.output.set_dims(self.output_dims)

        self.output_fields = self._setup_output_fields()
        self.output.set_fields(self.output_fields)

        # Write initial data
        self.output.save(self.output_fields, 0, 0, initial=True)

    @abstractmethod
    def _get_nx(self) -> int:
        """Return the grid resolution for this mode.

        Returns:
            Number of grid points.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_spectral_workspace(self) -> SpectralWorkspace:
        """Create the spectral workspace for this mode.

        This method is called during initialization to create the
        SpectralWorkspace that bundles all spectral utilities.
        Subclasses should configure the workspace based on their needs
        (e.g., LES needs nx2 for downscaling, DNS does not).

        Returns:
            Configured SpectralWorkspace instance.
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_mode_specific(self) -> None:
        """Initialize mode-specific components.

        Called during __init__ after common setup. Subclasses should
        set up noise generators, filters, SGS models, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_output_fields(self) -> dict[str, Any]:
        """Configure the output fields dictionary.

        Returns:
            Dictionary mapping field names to arrays for output.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_derivatives(self, t: int) -> dict[str, np.ndarray]:
        """Compute required spatial derivatives.

        Args:
            t: Current time step index.

        Returns:
            Dictionary of derivative arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_noise(self) -> np.ndarray:
        """Generate noise for the current time step.

        Returns:
            Noise array at the appropriate resolution.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_rhs(
        self,
        derivatives: dict[str, np.ndarray],
        noise: np.ndarray
    ) -> np.ndarray:
        """Compute the right-hand side of the Burgers equation.

        Args:
            derivatives: Dictionary of spatial derivatives.
            noise: Noise array for stochastic forcing.

        Returns:
            RHS array for time integration.
        """
        raise NotImplementedError

    @abstractmethod
    def _save_diagnostics(
        self,
        derivatives: dict[str, np.ndarray],
        t_out: int,
        t_loop: float
    ) -> None:
        """Compute and save mode-specific diagnostics.

        Args:
            derivatives: Dictionary of spatial derivatives.
            t_out: Output time index.
            t_loop: Current simulation time.
        """
        raise NotImplementedError

    def run(self) -> None:
        """Execute the time integration loop.

        Advances the simulation using 2nd-order Adams-Bashforth time
        stepping, with Euler for the first step. Writes output at
        intervals specified by step_save.
        """
        # Placeholder for previous RHS (Adams-Bashforth)
        rhsp: np.ndarray | int = 0

        # Time loop
        for t in range(1, int(self.nt)):
            # Current simulation time
            t_loop = t * self.dt

            # Progress reporting
            self._log_progress(t, t_loop)

            # Compute spatial derivatives
            derivatives = self._compute_derivatives(t)

            # Generate noise
            noise = self._compute_noise()

            # Compute RHS
            rhs = self._compute_rhs(derivatives, noise)

            # Time integration (Adams-Bashforth, Euler for t=1)
            if t == 1:
                self.u[:] = self.u + self.dt * rhs
            else:
                self.u[:] = self.u + self.dt * (1.5 * rhs - 0.5 * rhsp)

            # Zero Nyquist mode to prevent aliasing
            self.spectral.derivatives.fft()
            self.fu[self.mp] = 0
            self.spectral.derivatives.ifft_nyquist()

            # Store RHS for next step
            rhsp = rhs

            # Write output at save intervals
            if t % self.step_save == 0:
                t_out = t // self.step_save
                self._save_diagnostics(derivatives, t_out, t_loop)

        # Print newline after progress bar
        if self.logger.isEnabledFor(logging.INFO):
            print()

    def _log_progress(self, t: int, t_loop: float) -> None:
        """Log simulation progress.

        Args:
            t: Current time step.
            t_loop: Current simulation time.
        """
        total_time = self.nt * self.dt
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "Running for time %05.2f of %05.2f", t_loop, total_time
            )
        elif self.logger.isEnabledFor(logging.INFO):
            sys.stdout.write(
                f"\r[pyBurgers: pyBurgers.{self.mode_name}] \t "
                f"Running for time {t_loop:05.2f} of {total_time:05.2f}"
            )
            sys.stdout.flush()
