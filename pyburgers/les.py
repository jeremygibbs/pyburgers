"""Large-Eddy Simulation (LES) for pyBurgers.

Implements the LES solver for the 1D stochastic Burgers equation
with subgrid-scale modeling.
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import pyfftw

from .sgs import SGS
from utils import Derivatives, FBM, Filter, FFTW_PLANNING, FFTW_THREADS, get_logger

if TYPE_CHECKING:
    from utils.io import Input, Output


class LES:
    """Large-eddy simulation solver for the Burgers equation.

    Solves the filtered 1D stochastic Burgers equation using Fourier
    collocation for spatial derivatives, Adams-Bashforth time integration,
    and subgrid-scale models for unresolved turbulence.

    Attributes:
        input: Input configuration object.
        output: Output handler for NetCDF writing.
        nx: Number of LES grid points.
        nxDNS: Number of DNS grid points (for noise generation).
        dx: Grid spacing.
        dt: Time step.
        nt: Number of time steps.
        visc: Kinematic viscosity.
        namp: Noise amplitude.
        model: SGS model type (0-4).
        t_save: Output save interval.
    """

    def __init__(self, input_obj: Input, output_obj: Output) -> None:
        """Initialize the LES solver.

        Args:
            input_obj: Input configuration containing simulation parameters.
            output_obj: Output handler for writing results to NetCDF.
        """
        self.logger: logging.Logger = get_logger("LES")

        self.logger.info("You are running in LES mode")

        # Initialize random number generator for reproducibility
        np.random.seed(1)

        # Read configuration variables
        self.logger.debug("Reading input settings")
        self.input = input_obj
        self.dt = self.input.dt
        self.nt = self.input.nt
        self.visc = self.input.visc
        self.namp = self.input.namp
        self.nx = self.input.nxLES
        self.nxDNS = self.input.nxDNS
        self.model = self.input.sgs
        self.t_save = self.input.t_save
        self.mp = int(self.nx / 2)
        self.dx = 2 * np.pi / self.nx

        # FBM noise at DNS resolution (will be filtered down)
        self.fbm = FBM(0.75, self.nxDNS)

        # Derivatives object
        self.derivs = Derivatives(self.nx, self.dx)

        # Filter for downscaling DNS noise to LES grid
        self.filter = Filter(self.nx, nx2=self.nxDNS)

        # SGS model (pass derivatives to Deardorff for sharing)
        self.subgrid = SGS.get_model(self.model, self.input, self.derivs)

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

        # Initialize subgrid TKE for Deardorff model
        if self.model == 4:
            self.tke_sgs: np.ndarray | float = np.ones(self.nx)
        else:
            self.tke_sgs = 0

        # Output diagnostic fields
        self.tke = np.zeros(1)
        self.C_sgs = np.zeros(1)
        self.diss_sgs = np.zeros(1)
        self.diss_mol = np.zeros(1)
        self.ens_prod = np.zeros(1)
        self.ens_dsgs = np.zeros(1)
        self.ens_dmol = np.zeros(1)

        # Setup output
        self.output = output_obj
        self.output_dims = {'t': 0, 'x': self.nx}
        self.output.set_dims(self.output_dims)

        self.output_fields = {
            'x': self.x,
            'u': self.u,
            'tke': self.tke,
            'C_sgs': self.C_sgs,
            'diss_sgs': self.diss_sgs,
            'diss_mol': self.diss_mol,
            'ens_prod': self.ens_prod,
            'ens_diss_sgs': self.ens_dsgs,
            'ens_diss_mol': self.ens_dmol
        }

        # Add subgrid TKE output for Deardorff model
        if self.model == 4:
            self.output_fields['tke_sgs'] = self.tke_sgs

        self.output.set_fields(self.output_fields)

        # Write initial data
        self.output.save(self.output_fields, 0, 0, initial=True)

    def run(self) -> None:
        """Execute the LES time integration loop.

        Advances the simulation using 2nd-order Adams-Bashforth time
        stepping, with Euler for the first step. Computes SGS stress
        at each step and writes output at intervals specified by t_save.
        """
        # Placeholder for previous RHS (Adams-Bashforth)
        rhsp = 0

        # Time loop
        for t in range(1, int(self.nt)):
            # Current simulation time
            looptime = t * self.dt

            # Progress reporting: DEBUG logs every step, INFO shows overwriting progress bar
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Running for time {looptime:05.2f} of {int(self.nt) * self.dt:05.2f}")
            elif self.logger.isEnabledFor(logging.INFO):
                # Use carriage return for overwriting progress bar at INFO level
                sys.stdout.write(
                    f"\r[pyBurgers: pyburgers.LES] \t Running for time {looptime:05.2f} "
                    f"of {int(self.nt) * self.dt:05.2f}"
                )
                sys.stdout.flush()

            # Compute spatial derivatives
            # Include 3rd derivative at output times for enstrophy budget
            if t % self.t_save == 0:
                derivatives = self.derivs.compute(self.u, [1, 2, 3, 'sq'])
                d3udx3 = derivatives['3']
            else:
                derivatives = self.derivs.compute(self.u, [1, 2, 'sq'])

            dudx = derivatives['1']
            du2dx = derivatives['sq']
            d2udx2 = derivatives['2']

            # Generate and filter FBM noise from DNS to LES scales
            noise = self.fbm.compute_noise()
            noise = self.filter.downscale(noise, int(self.nxDNS / self.nx))

            # Compute SGS stress
            sgs = self.subgrid.compute(self.u, dudx, self.tke_sgs)
            tau = sgs["tau"]
            coeff = sgs["coeff"]

            # Update subgrid TKE for Deardorff model
            if self.model == 4:
                self.tke_sgs[:] = sgs["tke_sgs"]

            # Compute SGS stress divergence
            sgsder = self.derivs.compute(tau, [1])
            dtaudx = sgsder['1']

            # Compute RHS: diffusion - advection + forcing - SGS
            rhs = (
                self.visc * d2udx2
                - 0.5 * du2dx
                + np.sqrt(2 * self.namp / self.dt) * noise
                - 0.5 * dtaudx
            )

            # Time integration (Adams-Bashforth, Euler for t=0)
            if t == 0:
                self.u[:] = self.u[:] + self.dt * rhs
            else:
                self.u[:] = self.u[:] + self.dt * (1.5 * rhs - 0.5 * rhsp)

            # Zero Nyquist mode after time integration to prevent aliasing
            self.fft()
            self.fu[self.mp] = 0
            self.ifft()

            # Store RHS for next step
            rhsp = rhs

            # Write output at save intervals
            if t % self.t_save == 0:
                t_out = int(t / self.t_save)

                # Compute diagnostics
                self.tke[:] = np.var(self.u)
                self.diss_sgs[:] = np.mean(-tau * dudx)
                self.diss_mol[:] = np.mean(self.visc * dudx ** 2)
                self.ens_prod[:] = np.mean(dudx ** 3)
                self.ens_dsgs[:] = np.mean(-tau * d3udx3)
                self.ens_dmol[:] = np.mean(self.visc * d2udx2 ** 2)
                self.C_sgs[:] = coeff

                self.output.save(self.output_fields, t_out, looptime, initial=False)

        # Close output file
        self.output.close()
