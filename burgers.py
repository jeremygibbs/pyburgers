#!/usr/bin/env python
"""pyBurgers: 1D Stochastic Burgers Equation Solver.

A tool for studying turbulence using Direct Numerical Simulation (DNS)
and Large-Eddy Simulation (LES) of the 1D stochastic Burgers equation.
"""
import argparse
import logging
import sys
import time

from models import DNS, LES
from utils import Input, Output, get_logger, setup_logging


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

    # Configure logging based on namelist settings
    setup_logging(level=input_obj.log_level)
    logger: logging.Logger = get_logger("Main")

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

    # Create simulation instance
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

    # Time the simulation
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
