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

from pyburgers import DNS, LES, Input, Output, __version__
from pyburgers.exceptions import NamelistError, PyBurgersError
from pyburgers.utils import (
    get_logger,
    load_wisdom,
    save_wisdom,
    warmup_fftw_plans,
)


def main() -> None:
    """Parse arguments, run the simulation, and print timing information.

    Raises:
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
        choices=["dns", "les"],
        required=True,
        help="Simulation mode: 'dns' or 'les'",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="namelist",
        type=str,
        default="namelist.json",
        help="Namelist configuration file (default: namelist.json)",
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

    # Welcome message
    print("#"*100)
    print("#"+(" "*98)+"#")
    print("#"+f"Welcome to PyBurgers (version {__version__})".center(98)+"#")
    print("#"+(" "*24)+"A toy to study Burgers turbulence with DNS and LES"+(" "*24)+"#")
    print("#"+(" "*40)+"by: Jeremy A Gibbs"+(" "*40)+"#")
    print("#"+(" "*98)+"#")
    print("#"*100)

    output_obj: Output | None = None
    logger = get_logger("Main")

    try:
        # Create Input instance from namelist (configures logging)
        input_obj: Input = Input(args.namelist)
        t_total: float = time.perf_counter()

        # Log FFTW configuration
        logger.debug(
            "FFTW Planning: %s, Threads: %d", input_obj.fftw_planning, input_obj.fftw_threads
        )

        # Load any accumulated wisdom from previous runs.
        # pyfftw wisdom is cumulative: plans for many sizes coexist in one file.
        # Matching plans are reused immediately; missing sizes are planned on-demand.
        wisdom_loaded, wisdom_msg = load_wisdom()
        logger.debug("FFTW wisdom: %s", wisdom_msg)

        # Register save_wisdom to run at exit (accumulates plans across runs)
        atexit.register(save_wisdom)

        # Warm up plans for the current configuration.
        # Fast when wisdom is already loaded; required on first run or new config.
        if not wisdom_loaded:
            logger.info("Building FFTW plans to populate wisdom cache...")
        warmup_success, warmup_msg = warmup_fftw_plans(
            input_obj.grid.dns.points,
            input_obj.grid.les.points,
            input_obj.physics.noise.exponent,
            input_obj.fftw_planning,
            input_obj.fftw_threads,
            input_obj.domain_length,
        )

        if warmup_success:
            logger.debug("FFTW warmup: %s", warmup_msg)
        else:
            logger.warning("FFTW warmup: %s", warmup_msg)
            logger.warning("Continuing without pre-warmed plans (will plan on-demand)")

        # Create Output instance
        if not outfile:
            outfile = f"pyburgers_{mode}.nc"
        elif not outfile.lower().endswith(".nc"):
            outfile = f"{outfile}.nc"
        output_obj = Output(outfile)

        # Create simulation instance (includes FFTW planning)
        logger.info("Initializing simulation and planning FFTs...")
        if mode == "dns":
            burgers = DNS(input_obj, output_obj)
        else:
            burgers = LES(input_obj, output_obj)

        # Initialization complete - now start timing the actual simulation
        logger.info("Initialization complete. Starting simulation...")
        t_solver: float = time.perf_counter()

        # Run the simulation
        burgers.run()

        # Report timing
        t_end: float = time.perf_counter()
        logger.info("Done! Solver: %06.2f s  |  Total: %06.2f s",
                    t_end - t_solver, t_end - t_total)

    except NamelistError as e:
        logger.error("Namelist configuration error: %s", e)
        logger.error("Check namelist.json settings.")
        raise SystemExit(1) from e
    except PyBurgersError as e:
        logger.error("An error occurred: %s", e)
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise SystemExit(1) from e
    finally:
        # Ensure the output file is properly closed, even if an error occurred
        if output_obj is not None:
            output_obj.close()
        print("#"*100)


if __name__ == "__main__":
    main()
