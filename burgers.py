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

A tool for studying turbulence using Direct Numerical Simulation (DNS)
and Large-Eddy Simulation (LES) of the 1D stochastic Burgers equation.
"""
import argparse
import atexit
import logging
import sys
import time

from models import DNS, LES
from utils import Input, Output, get_logger, setup_logging
from utils import config
from utils.config import load_wisdom, save_wisdom

# Load FFTW wisdom at startup for optimized FFT plans
load_wisdom()

# Save FFTW wisdom at exit for future runs
atexit.register(save_wisdom)


class InvalidMode(Exception):
    """Exception raised for invalid simulation mode selection."""

    pass

def main() -> None:
    """Run the pyBurgers simulation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a case with pyBurgers")
    parser.add_argument(
        "-m", "--mode",
        dest='mode',
        type=str,
        default="dns",
        help="Simulation mode: 'dns' or 'les'"
    )
    parser.add_argument(
        "-o", "--output",
        dest='outfile',
        type=str,
        help="Output file name (default: pyburgers_<mode>.nc)"
    )
    args = parser.parse_args()
    mode = args.mode
    outfile = args.outfile

    # Create Input instance from namelist (reads log level)
    namelist = 'namelist.json'
    input_obj = Input(namelist)

    # Apply FFTW settings from namelist to config module
    config.FFTW_PLANNING = input_obj.fftw_planning
    config.FFTW_THREADS = input_obj.fftw_threads

    # Configure logging based on namelist settings
    setup_logging(level=input_obj.log_level)
    logger: logging.Logger = get_logger("Main")

    # Log FFTW configuration
    logger.debug(f"FFTW Planning: {config.FFTW_PLANNING}, Threads: {config.FFTW_THREADS}")

    # Welcome message
    logger.info("##############################################################")
    logger.info("#                                                            #")
    logger.info("#                   Welcome to pyBurgers                     #")
    logger.info("#      A fun tool to study turbulence using DNS and LES      #")
    logger.info("#                                                            #")
    logger.info("##############################################################")

    # Create Output instance
    if not outfile:
        outfile = f'pyburgers_{mode}.nc'
    output_obj = Output(outfile)

    # Create simulation instance (includes FFTW planning)
    logger.info("Initializing simulation and planning FFTs...")
    try:
        if mode == "dns":
            burgers = DNS(input_obj, output_obj)
        elif mode == "les":
            burgers = LES(input_obj, output_obj)
        else:
            raise InvalidMode(f'Error: Invalid mode "{mode}" (must be "dns" or "les")')
    except InvalidMode as e:
        logger.error(str(e))
        sys.exit(1)

    # Initialization complete - now start timing the actual simulation
    logger.info("Initialization complete. Starting simulation run...")
    t1 = time.time()

    # Run the simulation
    burgers.run()

    # Report timing
    t2 = time.time()
    elapsed = t2 - t1
    logger.info(f"Done! Completed in {elapsed:0.2f} seconds")
    logger.info("##############################################################")


if __name__ == "__main__":
    main()
