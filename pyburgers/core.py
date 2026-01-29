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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .utils import get_logger
from .utils.spectral_workspace import SpectralWorkspace

if TYPE_CHECKING:
    from .utils.io import Input, Output


class Burgers(ABC):
    """Abstract base class for Burgers equation solvers.

    Provides common functionality for solving the 1D stochastic Burgers
    equation using Fourier collocation for spatial derivatives and
    Williamson (1980) low-storage RK3 time integration with CFL-based
    adaptive time stepping.

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
        visc: Kinematic viscosity.
        noise_amp: Noise amplitude.
        cfl_target: Target CFL number.
        max_step: Maximum allowed time step.
        t_duration: Total simulation time.
        t_save: Output save interval in physical time.
        t_print: Progress print interval in physical time.
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
        self.visc = input_obj.viscosity
        self.noise_amp = input_obj.physics.noise.amplitude
        self.noise_alpha = input_obj.physics.noise.exponent
        self.fftw_planning = input_obj.fftw_planning
        self.fftw_threads = input_obj.fftw_threads
        self.domain_length = input_obj.domain_length

        # Adaptive time stepping parameters
        self.cfl_target = input_obj.cfl_target
        self.max_step = input_obj.max_step
        self.t_duration = input_obj.time.duration
        self.t_save = input_obj.t_save
        self.t_print = input_obj.t_print

        # Get mode-specific grid resolution
        self.nx = self._get_nx()
        self.mp = self.nx // 2
        self.dx = self.domain_length / self.nx

        # Precompute viscous stability limit (constant for the run)
        self._dt_visc = 0.2 * self.dx**2 / self.visc

        # Create spectral workspace (bundles Derivatives, Dealias, Filter)
        self.spectral = self._create_spectral_workspace()

        # Grid coordinates
        self.x = np.arange(0, self.domain_length, self.dx)

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
        self.output_dims = {"t": 0, "x": self.nx}
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
    def _compute_derivatives(self, is_output_step: bool) -> dict[str, np.ndarray]:
        """Compute required spatial derivatives.

        Args:
            is_output_step: Whether this is an output save step.

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
        self, derivatives: dict[str, np.ndarray], noise: np.ndarray, dt: float
    ) -> np.ndarray:
        """Compute the right-hand side of the Burgers equation.

        Args:
            derivatives: Dictionary of spatial derivatives.
            noise: Noise array for stochastic forcing.
            dt: Current time step size.

        Returns:
            RHS array for time integration.
        """
        raise NotImplementedError

    @abstractmethod
    def _save_diagnostics(
        self, derivatives: dict[str, np.ndarray], t_out: int, t_loop: float
    ) -> None:
        """Compute and save mode-specific diagnostics.

        Args:
            derivatives: Dictionary of spatial derivatives.
            t_out: Output time index.
            t_loop: Current simulation time.
        """
        raise NotImplementedError

    def _compute_dt(self) -> float:
        """Compute the adaptive time step from CFL and viscous constraints.

        Returns:
            Time step size satisfying CFL, viscous, and max_step limits.
        """
        u_max = np.max(np.abs(self.u))
        if u_max > 0:
            dt_adv = self.cfl_target * self.dx / u_max
        else:
            dt_adv = self.max_step
        return min(dt_adv, self._dt_visc, self.max_step)

    def run(self) -> None:
        """Execute the time integration loop.

        Advances the simulation using Williamson (1980) low-storage RK3
        with CFL-based adaptive time stepping. Output is written at
        exact multiples of t_save by clamping dt to hit output times.
        """
        # Williamson (1980) low-storage RK3 coefficients
        A = (0.0, -5.0 / 9.0, -153.0 / 128.0)
        B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)

        t_current = 0.0
        t_next_save = self.t_save
        t_next_print = self.t_print
        save_idx = 0
        Q = np.zeros_like(self.u)

        # Sample noise at fixed max_step intervals so that DNS and LES
        # consume the same random sequence regardless of adaptive dt.
        noise = self._compute_noise()
        t_next_noise = self.max_step

        while t_current < self.t_duration - 1e-14:
            dt = self._compute_dt()

            # Clamp to hit next output time or end time exactly
            if t_current + dt >= t_next_save - 1e-14:
                dt = t_next_save - t_current
            if t_current + dt > self.t_duration:
                dt = self.t_duration - t_current
            if dt < 1e-15:
                break

            is_output_step = abs(t_current + dt - t_next_save) < 1e-14

            # 3-stage RK3
            Q[:] = 0.0
            for stage in range(3):
                derivatives = self._compute_derivatives(False)
                rhs = self._compute_rhs(derivatives, noise, dt)
                Q[:] = A[stage] * Q + rhs
                self.u[:] = self.u + B[stage] * dt * Q

                # Zero Nyquist after each stage
                self.spectral.derivatives.fft()
                self.fu[self.mp] = 0
                self.spectral.derivatives.ifft_nyquist()

            t_current += dt

            # Refresh noise at fixed intervals
            if t_current >= t_next_noise - 1e-14:
                noise = self._compute_noise()
                t_next_noise += self.max_step

            # Progress logging
            self._log_progress(t_current, t_next_print)
            if t_current >= t_next_print - 1e-14:
                t_next_print += self.t_print

            # Output at exact save times
            if is_output_step:
                save_idx += 1
                derivatives = self._compute_derivatives(True)
                t_exact = save_idx * self.t_save
                self._save_diagnostics(derivatives, save_idx, t_exact)
                t_next_save += self.t_save

        if self.logger.isEnabledFor(logging.INFO):
            print()

    def _log_progress(self, t_current: float, t_next_print: float) -> None:
        """Log simulation progress.

        Args:
            t_current: Current simulation time.
            t_next_print: Next scheduled print time.
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "Running for time %05.2f of %05.2f", t_current, self.t_duration
            )
        elif self.logger.isEnabledFor(logging.INFO):
            if t_current < t_next_print - 1e-14 and t_current < self.t_duration - 1e-14:
                return
            self.logger.info(
                "Running for time %05.2f of %05.2f",
                t_current,
                self.t_duration,
                extra={"progress": True},
            )
