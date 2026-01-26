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
"""PyBurgers: 1D Stochastic Burgers Equation Solver.

This script serves as the primary entry point for running PyBurgers simulations.
It handles command-line argument parsing for specifying simulation mode (DNS or
LES), sets up the necessary input and output files, initializes the solver,
and executes the main time-stepping loop.

To run a simulation, use:
    $ python burgers.py -m dns
    $ python burgers.py -m les
    $ python burgers.py -m dns -o output.nc
"""

import argparse
import atexit
import time

from pyburgers import DNS, LES, Input, Output
from pyburgers.exceptions import InvalidMode, NamelistError, PyBurgersError
from pyburgers.utils import (
    get_logger,
    load_wisdom,
    save_wisdom,
    warmup_fftw_plans,
)


def main() -> None:
    """Parse arguments, run the simulation, and print timing information.

    Raises:
        InvalidMode: If an invalid simulation mode is specified.
        NamelistError: If the namelist configuration is invalid.
        PyBurgersError: If an error occurs during model setup or execution.
        FileNotFoundError: If required input files cannot be found.
    """
    # Set up command-line argument parsing
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run a simulation with PyBurgers"
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        type=str,
        default="dns",
        help="Simulation mode: 'dns' or 'les' (default: dns)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="outfile",
        type=str,
        help="Output file name (default: pyburgers_<mode>.nc)",
    )
    args: argparse.Namespace = parser.parse_args()
    mode: str = args.mode.lower()
    outfile: str | None = args.outfile

    output_obj: Output | None = None
    # Welcome message
    print("##############################################################")
    print("#                                                            #")
    print("#                   Welcome to PyBurgers                     #")
    print("#     A toy to study Burgers turbulence with DNS and LES     #")
    print("#                      by: Jeremy Gibbs                      #")
    print("#                                                            #")
    print("##############################################################")

    try:
        # Create Input instance from namelist (configures logging)
        namelist = "namelist.json"
        input_obj: Input = Input(namelist)

        # Get logger after Input sets up logging
        logger = get_logger("Main")

        # Log FFTW configuration
        logger.debug(
            "FFTW Planning: %s, Threads: %d", input_obj.fftw_planning, input_obj.fftw_threads
        )

        # Load FFTW wisdom at startup for optimized FFT plans
        # Validates that wisdom matches current grid sizes and parameters
        wisdom_loaded, wisdom_msg = load_wisdom(
            input_obj.grid.dns.nx,
            input_obj.grid.les.nx,
            input_obj.physics.noise.alpha,
            input_obj.fftw_planning,
            input_obj.fftw_threads,
        )

        if wisdom_loaded:
            logger.debug("FFTW wisdom: %s", wisdom_msg)
        else:
            logger.debug("FFTW wisdom: %s", wisdom_msg)

        # Register save_wisdom to run at exit
        atexit.register(
            save_wisdom,
            input_obj.grid.dns.nx,
            input_obj.grid.les.nx,
            input_obj.physics.noise.alpha,
            input_obj.fftw_planning,
            input_obj.fftw_threads,
        )

        # Generate FFTW plans if no wisdom is available yet
        if not wisdom_loaded:
            logger.info("Building FFTW plans to populate wisdom cache...")
            warmup_success, warmup_msg = warmup_fftw_plans(
                input_obj.grid.dns.nx,
                input_obj.grid.les.nx,
                input_obj.physics.noise.alpha,
                input_obj.fftw_planning,
                input_obj.fftw_threads,
            )

            if warmup_success:
                logger.debug("FFTW warmup: %s", warmup_msg)
                # Save wisdom immediately after successful warmup
                save_wisdom(
                    input_obj.grid.dns.nx,
                    input_obj.grid.les.nx,
                    input_obj.physics.noise.alpha,
                    input_obj.fftw_planning,
                    input_obj.fftw_threads,
                )
                logger.debug("FFTW wisdom saved to cache")
            else:
                logger.warning("FFTW warmup: %s", warmup_msg)
                logger.warning("Continuing without pre-warmed plans (will plan on-demand)")

        # Create Output instance
        if not outfile:
            outfile = f"pyburgers_{mode}.nc"
        output_obj = Output(outfile)

        # Create simulation instance (includes FFTW planning)
        logger.info("Initializing simulation and planning FFTs...")
        if mode == "dns":
            burgers = DNS(input_obj, output_obj)
        elif mode == "les":
            burgers = LES(input_obj, output_obj)
        else:
            raise InvalidMode(f'Invalid mode "{mode}". Must be "dns" or "les".')

        # Initialization complete - now start timing the actual simulation
        logger.info("Initialization complete. Starting simulation run...")
        t1: float = time.time()

        # Run the simulation
        burgers.run()

        # Report timing
        t2: float = time.time()
        elapsed: float = t2 - t1
        logger.info("Done! Completed in %.2f seconds", elapsed)

    except InvalidMode as e:
        print(f"\nInvalid mode error: {e}")
        print("Use -m dns or -m les")
        raise SystemExit(1) from e
    except NamelistError as e:
        print(f"\nNamelist configuration error: {e}")
        print("Check namelist.json settings.")
        raise SystemExit(1) from e
    except PyBurgersError as e:
        print(f"\nAn error occurred: {e}")
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        raise SystemExit(1) from e
    finally:
        # Ensure the output file is properly closed, even if an error occurred
        if output_obj is not None:
            output_obj.close()
    print("##############################################################")


if __name__ == "__main__":
    main()
