#
# UtahLSM
#
# Copyright (c) 2017–2025 Jeremy A. Gibbs
# Copyright (c) 2017–2025 Rob Stoll
# Copyright (c) 2017–2025 Eric Pardyjak
# Copyright (c) 2017–2025 Pete Willemsen
#
# This file is part of UtahLSM.
#
# This software is free and is distributed under the MIT License.
# See accompanying LICENSE file or visit https://opensource.org/licenses/MIT.
#
"""Handles NetCDF output for the UtahLSM model.

This module defines the `Output` class, which is responsible for creating,
configuring, and writing simulation results to a NetCDF file. It manages
file dimensions, variables, and attributes, providing a simple interface
for saving the model's state at each time step.
"""
import logging
from typing import Any

import netCDF4 as nc
import numpy as np

from . import logging_helper


class Output:
    """Manages the creation and writing of NetCDF output files.

    This class handles all aspects of the output file, from its initial
    creation to writing data at each time step and final closing.

    Attributes:
        logger: A logger for this class.
        outfile: A `netCDF4.Dataset` object representing the output file.
        fields_time: A dictionary mapping time-varying field names to their
            NetCDF variable objects.
        fields_static: A dictionary mapping static field names to their
            NetCDF variable objects.
        attributes: A dictionary defining the metadata (dimensions, units, etc.)
            for each possible output variable.
    """
    def __init__(self, outfile: str, sync_interval: int = 100) -> None:
        """Initializes the Output class and creates the NetCDF file.

        Args:
            outfile: The path and name for the output NetCDF file.
            sync_interval: Number of saves between disk syncs. Higher values
                improve performance but risk data loss on crash. Defaults to 100.
        """
        self.logger: logging.Logger = logging_helper.get_logger('Output')
        self.logger.info('Saving output to %s', outfile)
        self.outfile: nc.Dataset = nc.Dataset(outfile, 'w')
        self._sync_interval = sync_interval
        self._save_count = 0
        # self.outfile.description = "UtahLSM output"
        # self.outfile.source      = "Jeremy A. Gibbs"
        # self.outfile.history     = "Created " + time.ctime(time.time())

        self.fields_time: dict[str, Any] = {}
        self.fields_static: dict[str, Any] = {}
        self.attributes: dict[str, dict[str, Any]] = {
            'time': {
                'dimension':('t',),
                'long_name':'time',
                'units':'s'
            },
            'soil_z': {
                'dimension':('z',),
                'long_name':'z-distance',
                'units':'m'
            },
            'soil_type': {
                'dimension':('z',),
                'long_name':'soil type',
                'units':''
            },
            'soil_T': {
                'dimension':('t','z',),
                'long_name':'soil temperature',
                'units':'K'
            },
            'soil_q': {
                'dimension':('t','z',),
                'long_name':'soil moisture',
                'units':'m3 m-3'
            },
            'ust': {
                'dimension':('t',),
                'long_name':'friction velocity',
                'units':'m s-1'
            },
            'obl': {
                'dimension':('t',),
                'long_name':'Obukhov length',
                'units':'m'
            },
            'shf': {
                'dimension':('t',),
                'long_name':'sensible heat flux',
                'units':'W m-2'
            },
            'lhf': {
                'dimension':('t',),
                'long_name':'latent heat flux',
                'units':'W m-2'
            },
            'ghf': {
                'dimension':('t',),
                'long_name':'ground heat flux',
                'units':'W m-2'
            },
        }

    def set_dims(self, dims: dict[str, int]) -> None:
        """Sets the dimensions in the NetCDF output file.

        Args:
            dims: A dictionary mapping dimension names to their sizes. A size
                of 0 indicates an unlimited dimension.
        """
        has_xy = ('x' in dims and 'y' in dims
                  and (dims['x'] > 1 or dims['y'] > 1))
        if has_xy:
            self.attributes['soil_T']['dimension'] = ('t', 'z', 'y', 'x')
            self.attributes['soil_q']['dimension'] = ('t', 'z', 'y', 'x')
            for field in ('ust', 'obl', 'shf', 'lhf', 'ghf'):
                self.attributes[field]['dimension'] = ('t', 'y', 'x')

        for dim, size in dims.items():
            if size == 0:
                self.outfile.createDimension(dim)
            else:
                self.outfile.createDimension(dim, size)

    def set_fields(self, fields: dict[str, Any]) -> None:
        """Creates the variables (fields) in the NetCDF output file.

        Args:
            fields: A dictionary of fields to be created in the output file.
        """
        dims = self.attributes['time']['dimension']
        name = self.attributes['time']['long_name']
        units = self.attributes['time']['units']
        ncvar = self.outfile.createVariable('time', 'f8', dims)
        ncvar.units = units
        ncvar.long_name = name
        self.fields_time['time'] = ncvar

        # iterate through keys in dictionary
        for field in fields:
            dims  = self.attributes[field]['dimension']
            units = self.attributes[field]['units']
            name  = self.attributes[field]['long_name']
            ncvar = self.outfile.createVariable(field, 'f8', dims)
            ncvar.units = units
            ncvar.long_name = name
            if 't' in dims:
                self.fields_time[field] = ncvar
            else:
                self.fields_static[field] = ncvar

    def save(self, fields: dict[str, Any], tidx: int, time: float,
             initial: bool = False) -> None:
        """Saves a snapshot of the model's state to the output file.

        Args:
            fields: A dictionary of data fields to save.
            tidx: The time index for the current snapshot.
            time: The simulation time in seconds.
            initial: A boolean flag indicating if this is the initial save,
                in which case only static fields are written. Defaults to False.
        """
        def _reshape_for_output(data: Any, target_shape: tuple[int, ...],
                                field_name: str) -> np.ndarray:
            arr = np.asarray(data)
            if arr.shape == target_shape:
                return arr
            if arr.size == int(np.prod(target_shape)):
                return arr.reshape(target_shape)
            raise ValueError(
                f"Field {field_name} has shape {arr.shape}, which cannot be "
                f"reshaped to {target_shape} for output."
            )

        if initial:
            for field, static_field in self.fields_static.items():
                target_shape = static_field.shape
                static_field[:] = _reshape_for_output(
                    fields[field], target_shape, field)

        for field, field_var in self.fields_time.items():
            dim = self.attributes[field]['dimension']
            if len(dim) == 1:
                if field == 'time':
                    field_var[tidx] = time
                else:
                    field_var[tidx] = fields[field]
            else:
                target_shape = field_var.shape[1:]
                field_var[tidx, :] = _reshape_for_output(
                    fields[field], target_shape, field)

        self._save_count += 1
        if self._sync_interval > 0 and self._save_count % self._sync_interval == 0:
            self.outfile.sync()

    def close(self) -> None:
        """Closes the NetCDF output file.

        Performs a final sync to ensure all buffered data is written before
        closing the file.
        """
        self.outfile.sync()
        self.outfile.close()
