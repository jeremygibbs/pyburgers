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
"""Core Burgers equation solver base class.

This module provides the abstract base class for solving the 1D stochastic
Burgers equation. Both DNS and LES solvers inherit from this class.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .numerics import get_integrator, get_operator
from .utils import get_logger
from .utils.spectral_workspace import SpectralWorkspace

if TYPE_CHECKING:
    from .utils.io import Input, Output


class Burgers(ABC):
    """Abstract base class for Burgers equation solvers.

    Provides common functionality for solving the 1D stochastic Burgers
    equation using Fourier collocation for spatial derivatives and
    pluggable time integration with CFL-based adaptive time stepping.

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

        # Initialize random number generator (seed=None means random each run)
        self.rng = np.random.default_rng(input_obj.physics.noise.seed)

        # Store input/output objects
        self.logger.debug("Reading input settings")
        self.input = input_obj
        self.output = output_obj

        # Extract common configuration
        self.visc = input_obj.viscosity
        self.noise_amp = input_obj.physics.noise.amplitude
        self.noise_beta = input_obj.physics.noise.exponent
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

        # Create time integrator early so its stability coefficients inform dt limits
        self.integrator = get_integrator(input_obj.numerics.temporal, self.nx)

        # Create spectral workspace and gradient operator before computing dt limits
        # so that the operator's eigenvalue attributes can inform those limits.
        self.spectral = self._create_spectral_workspace()
        self.gradient_op = get_operator(
            input_obj.numerics.spatial, self.nx, self.dx, self.spectral
        )

        # Precompute viscous stability limit (constant for the run).
        # The gradient operator's viscous_eigenvalue (max |k̃²|·dx²) scales the
        # limit correctly across spectral and finite-difference schemes.
        # Use the integrator's dissipative_stability_limit (same coefficient used
        # for hyperviscosity) so the limit is consistent with the chosen scheme.
        _C = self.integrator.dissipative_stability_limit
        self._dt_visc = (
            _C * self.dx**2
            / (self.visc * self.gradient_op.viscous_eigenvalue)
        )

        # Hyperviscosity: coefficient normalized so the Nyquist damping rate
        # (ν₄ · k̃⁴_max) is equal across all spatial schemes.
        #
        # For the spectral scheme, k̃⁴_max = (π/dx)⁴, so ν₄ = dx⁴ gives a
        # Nyquist rate of π⁴.  FD stencils have smaller k̃⁴_max (16 for FD2,
        # 80/3 for FD4), so their coefficients are scaled up by π⁴/λ to
        # compensate:
        #
        #   ν₄ = dx⁴ · π⁴ / λ_hypervisc
        #
        # A beneficial side-effect: the dt stability limit reduces to C/π⁴ for
        # every scheme, so the hyperviscous timestep constraint is
        # scheme-independent.
        self.hypervisc = (
            self.dx**4 * np.pi**4 / self.gradient_op.hyperviscous_eigenvalue
        )
        self._dt_hypervisc = (
            _C * self.dx**4
            / (self.hypervisc * self.gradient_op.hyperviscous_eigenvalue)
        )
        self.logger.info(
            "Hyperviscosity: coefficient = %.2e",
            self.hypervisc,
        )

        # Grid coordinates
        self.x = np.linspace(0, self.domain_length, self.nx, endpoint=False)

        # Reference workspace buffers (zero-copy)
        self.u = self.spectral.u
        self.fu = self.spectral.fu

        # Initialize velocity field to zero
        self.u[:] = 0

        # Common output field
        self.tke = np.zeros(1)

        # Pre-allocated RHS buffer and precomputed noise scaling constant
        self.rhs = np.zeros(self.nx)
        self._noise_scale = np.sqrt(2.0 * self.noise_amp / self.max_step)

        # Step-scoped state for the RHS callable (avoids closure allocation in the loop)
        self._step_dt: float = 0.0
        self._step_noise: np.ndarray | None = None

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

    def _compute_diagnostic_derivatives(self) -> dict[str, np.ndarray]:
        """Compute only the derivatives needed for diagnostics at output time.

        By default falls back to ``_compute_derivatives(True)``. Subclasses
        can override to avoid recomputing derivatives that are only needed
        for the RHS (e.g. 4th derivative, du²/dx).

        Returns:
            Dictionary of derivative arrays for diagnostics.
        """
        return self._compute_derivatives(True)

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

    def _post_step(self, dt: float) -> None:  # noqa: B027
        """Hook called once after each completed integrator step.

        Override in subclasses to advance prognostic quantities (e.g.,
        subgrid TKE) that must be updated exactly once per physical
        timestep, regardless of the number of integrator stages. Default
        implementation is an intentional no-op so DNS does not need to
        override it.

        Args:
            dt: The physical time step just completed.
        """

    def _compute_dt(self) -> float:
        """Compute the adaptive time step from CFL and viscous constraints.

        Returns:
            Time step size satisfying CFL, viscous, hyperviscous, and max_step limits.
        """
        u_max = np.max(np.abs(self.u))
        if u_max > 0:
            dt_adv = self.cfl_target * self.dx / u_max
        else:
            dt_adv = self.max_step
        return min(dt_adv, self._dt_visc, self._dt_hypervisc, self.max_step)

    def _rhs_for_step(self) -> np.ndarray:
        """RHS callable passed to the integrator each timestep.

        Reads step-scoped state (_step_dt, _step_noise) set by run() before
        each integrator.step() call, avoiding closure allocation in the loop.
        """
        if self._step_noise is None:
            raise RuntimeError("_step_noise not set before _rhs_for_step call")
        derivatives = self._compute_derivatives(False)
        return self._compute_rhs(derivatives, self._step_noise, self._step_dt)

    def run(self) -> None:
        """Execute the time integration loop.

        Advances the simulation using the configured time integrator
        with CFL-based adaptive time stepping. Output is written at
        exact multiples of t_save by clamping dt to hit output times.
        """
        t_current = 0.0
        t_next_save = self.t_save
        t_next_print = self.t_print
        save_idx = 0

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

            is_output_step = abs(t_current + dt - t_next_save) <= 1e-12 * max(1.0, t_next_save)

            # Set step-scoped state and delegate to the integrator
            self._step_dt = dt
            self._step_noise = noise
            self.integrator.step(
                self.u, dt, self._rhs_for_step, self.gradient_op.zero_nyquist
            )

            # Post-step hook (e.g., advance prognostic SGS quantities once per step)
            self._post_step(dt)

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
                derivatives = self._compute_diagnostic_derivatives()
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
                "Running for time %6.2f of %6.2f", t_current, self.t_duration
            )
        elif self.logger.isEnabledFor(logging.INFO):
            if t_current < t_next_print - 1e-14 and t_current < self.t_duration - 1e-14:
                return
            self.logger.info(
                "Running for time %6.2f of %6.2f",
                t_current,
                self.t_duration,
                extra={"progress": True},
            )
