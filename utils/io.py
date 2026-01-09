"""Input/Output utilities for pyBurgers.

This module provides classes for reading simulation configuration
from JSON namelist files and writing output to NetCDF format.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

import netCDF4 as nc
import numpy as np

from .logging_helper import get_logger


class Input:
    """Reads and parses simulation configuration from a JSON namelist file.

    Attributes:
        dt: Time step size in seconds.
        nt: Number of time steps.
        nxDNS: Number of grid points for DNS.
        nxLES: Number of grid points for LES.
        sgs: SGS model type (0-4).
        visc: Kinematic viscosity.
        namp: Noise amplitude for FBM forcing.
        t_save: Output save interval in time steps.
        n_save: Number of saves (computed from t_save/dt).
        log_level: Logging level ("DEBUG", "INFO", "WARNING", etc.).
    """

    def __init__(self, namelist: str) -> None:
        """Initialize Input by parsing the namelist file.

        Args:
            namelist: Path to the JSON configuration file.

        Raises:
            SystemExit: If the file cannot be opened or parsed.
        """
        try:
            with open(namelist) as json_file:
                data = json.load(json_file)
        except FileNotFoundError as e:
            # Create logger only when needed for error reporting
            logger = get_logger("Input")
            logger.error(f"There was an issue opening '{namelist}'.")
            logger.error(f"Error: {e.strerror}")
            sys.exit(1)
        except json.decoder.JSONDecodeError as e:
            # Create logger only when needed for error reporting
            logger = get_logger("Input")
            logger.error(f"There was an issue parsing '{namelist}'.")
            logger.error(f"Error: {e}")
            sys.exit(1)
        else:
            try:
                # time section
                self.dt = data["time"]["dt"]
                self.nt = data["time"]["nt"]

                # DNS section
                self.nxDNS = data["dns"]["nx"]

                # LES section
                self.nxLES = data["les"]["nx"]
                self.sgs = data["les"]["sgs"]

                # physics section
                self.visc = data["physics"]["visc"]
                self.namp = data["physics"]["namp"]

                # output section
                self.t_save = data["output"]["t_save"]
                self.n_save = np.floor(self.t_save / self.dt)

                # logging section (optional, defaults to INFO)
                self.log_level: str = data.get("logging", {}).get("level", "INFO")

            except KeyError as e:
                # Create logger only when needed for error reporting
                logger = get_logger("Input")
                logger.error(f"There was an issue accessing data from '{namelist}'")
                logger.error(f"Error: The key {e} does not exist")
                sys.exit(1)


class Output:
    """Handles NetCDF output for simulation results.

    Attributes:
        outfile: The NetCDF4 Dataset object.
        fields_time: Dictionary of time-varying output variables.
        fields_static: Dictionary of static output variables.
        attributes: Metadata for each output field.
    """

    def __init__(self, outfile: str) -> None:
        """Initialize Output by creating a NetCDF file.

        Args:
            outfile: Path to the output NetCDF file.
        """
        self.outfile = nc.Dataset(outfile, 'w')
        self.outfile.description = "PyBurgers output"
        self.outfile.source = "Jeremy A. Gibbs"
        self.outfile.history = "Created " + time.ctime(time.time())

        self.fields_time: dict[str, Any] = {}
        self.fields_static: dict[str, Any] = {}

        self.attributes: dict[str, dict[str, Any]] = {
            'time': {
                'dimension': ("t",),
                'long_name': 'time',
                'units': 's'
            },
            'x': {
                'dimension': ("x",),
                'long_name': 'x-distance',
                'units': 'm'
            },
            'u': {
                'dimension': ("t", "x",),
                'long_name': 'u-component velocity',
                'units': 'm s-1'
            },
            'tke': {
                'dimension': ("t",),
                'long_name': 'turbulence kinetic energy',
                'units': 'm2 s-2'
            },
            'tke_sgs': {
                'dimension': ("t",),
                'long_name': 'subgrid turbulence kinetic energy',
                'units': 'm2 s-2'
            },
            'C_sgs': {
                'dimension': ("t",),
                'long_name': 'subgrid model coefficient',
                'units': '--'
            },
            'diss_sgs': {
                'dimension': ("t",),
                'long_name': 'subgrid dissipation',
                'units': 'm2 s-3'
            },
            'diss_mol': {
                'dimension': ("t",),
                'long_name': 'molecular dissipation',
                'units': 'm2 s-3'
            },
            'ens_prod': {
                'dimension': ("t",),
                'long_name': 'enstrophy production',
                'units': 's-3'
            },
            'ens_diss_sgs': {
                'dimension': ("t",),
                'long_name': 'subgrid enstrophy dissipation',
                'units': 's-3'
            },
            'ens_diss_mol': {
                'dimension': ("t",),
                'long_name': 'molecular enstrophy dissipation',
                'units': 's-3'
            }
        }

    def set_dims(self, dims: dict[str, int]) -> None:
        """Set the dimensions for the NetCDF file.

        Args:
            dims: Dictionary mapping dimension names to sizes.
                  A size of 0 indicates an unlimited dimension.
        """
        for dim, size in dims.items():
            if size == 0:
                self.outfile.createDimension(dim)
            else:
                self.outfile.createDimension(dim, size)

    def set_fields(self, fields: dict[str, np.ndarray]) -> None:
        """Create output variables in the NetCDF file.

        Args:
            fields: Dictionary mapping field names to their data arrays.
        """
        # add time manually
        dims = self.attributes['time']['dimension']
        name = self.attributes['time']['long_name']
        units = self.attributes['time']['units']
        ncvar = self.outfile.createVariable('time', "f4", dims)
        ncvar.long_name = name
        ncvar.units = units
        self.fields_time['time'] = ncvar

        # iterate through keys in dictionary
        for field in fields:
            dims = self.attributes[field]['dimension']
            name = self.attributes[field]['long_name']
            units = self.attributes[field]['units']
            ncvar = self.outfile.createVariable(field, "f4", dims)
            ncvar.long_name = name
            ncvar.units = units
            if 't' in dims:
                self.fields_time[field] = ncvar
            else:
                self.fields_static[field] = ncvar

    def save(
        self,
        fields: dict[str, np.ndarray],
        tidx: int,
        time: float,
        initial: bool = False
    ) -> None:
        """Save field data to the NetCDF file.

        Args:
            fields: Dictionary mapping field names to their data arrays.
            tidx: Time index for this save.
            time: Simulation time value.
            initial: If True, also save static fields.
        """
        # save static only for initial time
        if initial:
            for field in self.fields_static:
                self.fields_static[field][:] = fields[field]

        # save time-varying fields
        for field in self.fields_time:
            dim = self.attributes[field]['dimension']

            if len(dim) == 1:
                if field == 'time':
                    self.fields_time[field][tidx] = time
                else:
                    self.fields_time[field][tidx] = fields[field]
            else:
                self.fields_time[field][tidx, :] = np.real(fields[field])

        self.outfile.sync()

    def close(self) -> None:
        """Close the NetCDF output file."""
        self.outfile.close()
