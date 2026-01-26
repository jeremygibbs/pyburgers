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
"""Handles NetCDF output for PyBurgers.

This module defines the `Output` class, which is responsible for creating,
configuring, and writing simulation results to a NetCDF file. It manages
file dimensions, variables, and attributes, providing a simple interface
for saving the model's state at each time step.
"""

import logging
import time as time_module
from typing import Any

import netCDF4 as nc
import numpy as np

from ..logging_helper import get_logger


class Output:
    """Manages the creation and writing of NetCDF output files.

    This class handles all aspects of the output file, from its initial
    creation to writing data at each time step and final closing.

    Attributes:
        outfile: A `netCDF4.Dataset` object representing the output file.
        fields_time: A dictionary mapping time-varying field names to their
            NetCDF variable objects.
        fields_static: A dictionary mapping static field names to their
            NetCDF variable objects.
        attributes: A dictionary defining the metadata (dimensions, units, etc.)
            for each possible output variable.
    """

    def __init__(self, outfile: str, sync_interval: int = 100) -> None:
        """Initialize the Output class and create the NetCDF file.

        Args:
            outfile: The path and name for the output NetCDF file.
            sync_interval: Number of saves between disk syncs. Higher values
                improve performance but risk data loss on crash. Defaults to 100.
        """
        self.logger: logging.Logger = get_logger("Output")
        self.logger.info("Saving output to %s", outfile)
        self.outfile: nc.Dataset = nc.Dataset(outfile, "w")
        self._sync_interval = sync_interval
        self._save_count = 0

        self.outfile.description = "PyBurgers output"
        self.outfile.source = "PyBurgers - 1D Stochastic Burgers Equation Solver"
        self.outfile.history = "Created " + time_module.ctime(time_module.time())

        self.fields_time: dict[str, Any] = {}
        self.fields_static: dict[str, Any] = {}
        self.attributes: dict[str, dict[str, Any]] = {
            "time": {"dimension": ("t",), "long_name": "time", "units": "s"},
            "x": {"dimension": ("x",), "long_name": "x-distance", "units": "m"},
            "u": {"dimension": ("t", "x"), "long_name": "u-component velocity", "units": "m s-1"},
            "tke": {
                "dimension": ("t",),
                "long_name": "turbulence kinetic energy",
                "units": "m2 s-2",
            },
            "tke_sgs": {
                "dimension": ("t",),
                "long_name": "subgrid turbulence kinetic energy",
                "units": "m2 s-2",
            },
            "tke_sgs_prod": {
                "dimension": ("t",),
                "long_name": "subgrid TKE production",
                "units": "m2 s-3",
            },
            "tke_sgs_diff": {
                "dimension": ("t",),
                "long_name": "subgrid TKE diffusion",
                "units": "m2 s-3",
            },
            "tke_sgs_diss": {
                "dimension": ("t",),
                "long_name": "subgrid TKE dissipation",
                "units": "m2 s-3",
            },
            "C_sgs": {"dimension": ("t",), "long_name": "subgrid model coefficient", "units": "--"},
            "diss_sgs": {
                "dimension": ("t",),
                "long_name": "subgrid dissipation",
                "units": "m2 s-3",
            },
            "diss_mol": {
                "dimension": ("t",),
                "long_name": "molecular dissipation",
                "units": "m2 s-3",
            },
            "ens_prod": {"dimension": ("t",), "long_name": "enstrophy production", "units": "s-3"},
            "ens_diss_sgs": {
                "dimension": ("t",),
                "long_name": "subgrid enstrophy dissipation",
                "units": "s-3",
            },
            "ens_diss_mol": {
                "dimension": ("t",),
                "long_name": "molecular enstrophy dissipation",
                "units": "s-3",
            },
        }

    def set_dims(self, dims: dict[str, int]) -> None:
        """Set the dimensions in the NetCDF output file.

        Args:
            dims: A dictionary mapping dimension names to their sizes. A size
                of 0 indicates an unlimited dimension.
        """
        for dim, size in dims.items():
            if size == 0:
                self.outfile.createDimension(dim)
            else:
                self.outfile.createDimension(dim, size)

    def set_fields(self, fields: dict[str, Any]) -> None:
        """Create the variables (fields) in the NetCDF output file.

        Args:
            fields: A dictionary of fields to be created in the output file.
        """
        # Add time manually
        dims = self.attributes["time"]["dimension"]
        name = self.attributes["time"]["long_name"]
        units = self.attributes["time"]["units"]
        ncvar = self.outfile.createVariable("time", "f8", dims)
        ncvar.units = units
        ncvar.long_name = name
        self.fields_time["time"] = ncvar

        # Iterate through keys in dictionary
        for field in fields:
            if field not in self.attributes:
                self.logger.warning("Unknown field %s, skipping", field)
                continue
            dims = self.attributes[field]["dimension"]
            units = self.attributes[field]["units"]
            name = self.attributes[field]["long_name"]
            ncvar = self.outfile.createVariable(field, "f8", dims)
            ncvar.units = units
            ncvar.long_name = name
            if "t" in dims:
                self.fields_time[field] = ncvar
            else:
                self.fields_static[field] = ncvar

    def save(self, fields: dict[str, Any], tidx: int, time: float, initial: bool = False) -> None:
        """Save a snapshot of the simulation state to the output file.

        Args:
            fields: A dictionary of data fields to save.
            tidx: The time index for the current snapshot.
            time: The simulation time in seconds.
            initial: A boolean flag indicating if this is the initial save,
                in which case only static fields are written. Defaults to False.
        """
        # Save static fields only on initial save
        if initial:
            for field, static_var in self.fields_static.items():
                if field in fields:
                    static_var[:] = np.asarray(fields[field])

        # Save time-varying fields
        for field, field_var in self.fields_time.items():
            dim = self.attributes[field]["dimension"]
            if len(dim) == 1:
                if field == "time":
                    field_var[tidx] = time
                elif field in fields:
                    field_var[tidx] = fields[field]
            else:
                if field in fields:
                    field_var[tidx, :] = np.real(np.asarray(fields[field]))

        self._save_count += 1
        if self._sync_interval > 0 and self._save_count % self._sync_interval == 0:
            self.outfile.sync()

    def close(self) -> None:
        """Close the NetCDF output file.

        Performs a final sync to ensure all buffered data is written before
        closing the file.
        """
        self.outfile.sync()
        self.outfile.close()
        self.logger.info("Output file closed")
