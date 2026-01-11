"""Direct Numerical Simulation (DNS) for pyBurgers.

Implements the DNS solver for the 1D stochastic Burgers equation
using spectral methods.
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import pyfftw

from utils import Derivatives, FBM, get_logger, Input, Output

class DNS:
    """Direct numerical simulation solver for the Burgers equation.

    Solves the 1D stochastic Burgers equation using Fourier collocation
    for spatial derivatives and Adams-Bashforth time integration.

    Attributes:
        input: Input configuration object.
        output: Output handler for NetCDF writing.
        nx: Number of grid points.
        dx: Grid spacing.
        dt: Time step.
        nt: Number of time steps.
        visc: Kinematic viscosity.
        namp: Noise amplitude.
        t_save: Output save interval.
    """

    def __init__(self, input_obj: Input, output_obj: Output) -> None:
        """Initialize the DNS solver.

        Args:
            input_obj: Input configuration containing simulation parameters.
            output_obj: Output handler for writing results to NetCDF.
        """
        self.logger: logging.Logger = get_logger("DNS")

        self.logger.info("You are running in DNS mode")

        # Initialize random number generator for reproducibility
        np.random.seed(1)

        # Read configuration variables
        self.logger.debug("Reading input settings")
        self.input = input_obj

        # Local input settings
        self.dt = self.input.dt
        self.nt = self.input.nt
        self.visc = self.input.visc
        self.namp = self.input.namp
        self.nx = self.input.nxDNS
        self.t_save = self.input.t_save
        self.mp = int(self.nx / 2)
        self.dx = 2 * np.pi / self.nx

        # Fractional Brownian motion noise instance
        self.fbm = FBM(0.75, self.nx)

        # Derivatives object
        self.derivs = Derivatives(self.nx, self.dx)

        # Grid coordinates
        self.x = np.arange(0, 2 * np.pi, self.dx)

        # Velocity field (complex for FFT operations)
        self.u = pyfftw.empty_aligned(self.nx, dtype='complex128')
        self.fu = pyfftw.empty_aligned(self.nx, dtype='complex128')

        # FFT functions
        self.fft = pyfftw.FFTW(
            self.u, self.fu,
            direction='FFTW_FORWARD',
            flags=(FFTW_PLANNING,),
            threads=FFTW_THREADS
        )
        self.ifft = pyfftw.FFTW(
            self.fu, self.u,
            direction='FFTW_BACKWARD',
            flags=(FFTW_PLANNING,),
            threads=FFTW_THREADS
        )

        # Output fields
        self.tke = np.zeros(1)

        # Setup output
        self.output = output_obj
        self.output_dims = {'t': 0, 'x': self.nx}
        self.output.set_dims(self.output_dims)

        self.output_fields = {
            'x': self.x,
            'u': self.u,
            'tke': self.tke,
        }
        self.output.set_fields(self.output_fields)

        # Write initial data
        self.output.save(self.output_fields, 0, 0, initial=True)

    def run(self) -> None:
        """Execute the DNS time integration loop.

        Advances the simulation using 2nd-order Adams-Bashforth time
        stepping, with Euler for the first step. Writes output at
        intervals specified by t_save.
        """
        # Local copies for performance
        dt = self.dt
        nt = self.nt
        mp = self.mp
        visc = self.visc
        namp = self.namp
        t_save = self.t_save

        # Placeholder for previous RHS (Adams-Bashforth)
        rhsp = 0

        # Time loop
        for t in range(1, int(nt)):
            # Current simulation time
            t_loop = t * self.dt

            # Progress reporting: DEBUG logs every step, INFO shows overwriting progress bar
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Running for time {t_loop:05.2f} of {int(nt) * dt:05.2f}")
            elif self.logger.isEnabledFor(logging.INFO):
                # Use carriage return for overwriting progress bar at INFO level
                sys.stdout.write(
                    f"\r[pyBurgers: pyBurgers.DNS] \t Running for time {t_loop:05.2f} "
                    f"of {int(nt) * dt:05.2f}"
                )
                sys.stdout.flush()

            # Compute spatial derivatives
            derivatives = self.derivs.compute(self.u, [2, 'sq'])
            d2udx2 = derivatives['2']
            du2dx = derivatives['sq']

            # Generate FBM noise forcing
            noise = self.fbm.compute_noise()

            # Compute RHS: viscous diffusion - advection + stochastic forcing
            rhs = visc * d2udx2 - 0.5 * du2dx + np.sqrt(2 * namp / dt) * noise

            # Time integration (Adams-Bashforth, Euler for t=0)
            if t == 0:
                self.u[:] = self.u[:] + dt * rhs
            else:
                self.u[:] = self.u[:] + dt * (1.5 * rhs - 0.5 * rhsp)

            # Zero Nyquist mode after time integration to prevent aliasing
            self.fft()
            self.fu[mp] = 0
            self.ifft()

            # Store RHS for next step
            rhsp = rhs

            # Write output at save intervals
            if t % t_save == 0:
                t_out = int(t / t_save)
                self.tke[:] = np.var(self.u)
                self.output.save(self.output_fields, t_out, t_loop, initial=False)

        # Close output file
        self.output.close()
