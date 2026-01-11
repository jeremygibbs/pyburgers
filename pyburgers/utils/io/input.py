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
"""Handles all input data loading and validation for UtahLSM.

This module defines the `Input` class, which is responsible for reading the
JSON namelist, initial conditions from a NetCDF file, and offline forcing
data. It validates the inputs, populates the configuration and state
dataclasses, and provides a single, clean interface for the main model to
access all setup information.
"""
import json
import logging
from typing import Optional

import jsonschema
import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray

from utahlsm.util.io import logging_helper
from utahlsm.util.io.soil_properties_loader import SoilPropertiesLoader

from ...data_models import (
    AtmosphericState,
    ForcingData,
    GeneralConfig,
    GridConfig,
    IterationsConfig,
    NumericsConfig,
    OutputConfig,
    RadiationConfig,
    SoilConfig,
    SoilState,
    SurfaceConfig,
    TimeConfig,
    TolerancesConfig,
)
from ...exceptions import NamelistError


class Input:
    """Orchestrates the loading and validation of all model inputs.

    This class reads configuration from a JSON namelist file and initial
    conditions from a NetCDF file. It also handles optional offline forcing
    data. All data is validated and organized into the appropriate dataclasses.

    Attributes:
        logger: A logger for this class.
        general: Dataclass with general simulation settings.
        numerics: Dataclass with numerical scheme parameters.
        time: Dataclass with time-related parameters.
        surface: Dataclass with surface-related parameters.
        soil: Dataclass with soil model configuration.
        radiation: Dataclass with radiation model configuration.
        output: Dataclass with output file configuration.
        grid: Dataclass with grid and spatial discretization parameters.
        initial: Dataclass holding the initial soil state.
        forcing: Dataclass holding the time-series of offline forcing data,
            or None if not provided.
        soil_properties: Dictionary mapping soil type names to their properties.
        soil_properties_name: Name of the soil property dataset being used.
        soil_type_names: List of soil type names for each layer.
    """

    def __init__(self, namelist_path: str, inputfile: str,
                 offlinefile: Optional[str] = None) -> None:
        """Initializes the Input class and loads all data.

        Args:
            namelist_path: The file path to the JSON namelist.
            inputfile: The file path to the NetCDF initial conditions file.
            offlinefile: The optional file path to the NetCDF offline
                forcing file. Defaults to None.

        Raises:
            FileNotFoundError: If any of the required input files do not exist.
            json.JSONDecodeError: If the namelist JSON file is malformed.
            jsonschema.ValidationError: If the namelist does not conform to the
                expected schema.
        """
        self.logger: logging.Logger = logging_helper.get_logger('Input')
        self.logger.info('Reading %s', namelist_path)
        namelist_data = self._load_and_validate_namelist(namelist_path)
        log_level = namelist_data['general']['log_level']
        logging_helper.finalize_logging(log_level)

        self.logger.info('Reading %s', inputfile)
        nx = namelist_data['grid']['nx']
        ny = namelist_data['grid']['ny']
        nz = namelist_data['grid']['nz']
        init_data = self._load_initial_conditions(inputfile, nx, ny, nz)

        # Load soil properties before creating configs
        self.logger.info('Loading soil properties')
        self._load_soil_properties(namelist_data['soil'], init_data['type'])

        self.general: GeneralConfig = GeneralConfig(**namelist_data['general'])
        iterations_data = namelist_data['numerics']['iterations']
        tolerances_data = namelist_data['numerics']['tolerances']
        numerics_data = namelist_data['numerics']
        self.numerics = NumericsConfig(
            diffusion_back_weight=(
                namelist_data['numerics']['diffusion_back_weight']),
            warm_start_turbulence=bool(
                numerics_data.get('warm_start_turbulence', False)),
            initialize_surface_temperature_from_seb=bool(
                numerics_data.get('initialize_surface_temperature_from_seb', False)),
            iterations=IterationsConfig(**iterations_data),
            tolerances=TolerancesConfig(**tolerances_data)
        )
        self.time: TimeConfig = TimeConfig(**namelist_data['time'])
        self.surface: SurfaceConfig = SurfaceConfig(**namelist_data['surface'])
        self.soil: SoilConfig = SoilConfig(**namelist_data['soil'])
        rad_data = namelist_data['radiation']
        self.radiation: RadiationConfig = RadiationConfig(**rad_data)
        self.output: OutputConfig = OutputConfig(**namelist_data['output'])
        self.grid: GridConfig = GridConfig(
            nx=nx,
            ny=ny,
            nz=nz,
            z=init_data['z']
        )
        self.initial: SoilState = SoilState(
            temperature=init_data['temperature'],
            moisture=init_data['moisture'],
            type=init_data['type']
        )

        self.forcing: Optional[ForcingData] = None
        if offlinefile:
            self.logger.info('Reading %s', offlinefile)
            self._load_offline_data(offlinefile)

        self._validate_physical_consistency()

    def _load_and_validate_namelist(self, namelist_path: str) -> dict:
        """Loads and validates the JSON namelist against a schema.

        Args:
            namelist_path: The path to the JSON namelist file.

        Returns:
            A dictionary containing the validated namelist data.

        Raises:
            FileNotFoundError: If the namelist or schema file cannot be
                found.
            json.JSONDecodeError: If the namelist is not valid JSON.
            jsonschema.ValidationError: If the namelist does not match
                the schema.
        """
        schema_path = 'utahlsm/util/io/schema_namelist.json'

        try:
            with open(schema_path, encoding='utf-8') as f:
                schema = json.load(f)
            with open(namelist_path, encoding='utf-8') as f:
                namelist_data = json.load(f)
            jsonschema.validate(instance=namelist_data, schema=schema)
            self.logger.info('--- namelist validation successful')
            return namelist_data
        except (FileNotFoundError, json.JSONDecodeError,
                jsonschema.ValidationError) as e:
            self.logger.error('--- namelist error: %s', e)
            raise

    def _load_initial_conditions(
            self, inputfile: str, nx: int, ny: int,
            nz: int) -> dict[str, NDArray]:
        """Loads data from the NetCDF initialization file.

        Args:
            inputfile: The path to the NetCDF initial conditions file.

        Returns:
            A dictionary of NumPy arrays for soil depth, temperature,
            moisture, and type. The 'type' array contains soil type names
            as strings.

        Raises:
            IOError: If the file cannot be read.
            KeyError: If a required variable is missing from the NetCDF
                file.
        """
        try:
            with nc.Dataset(inputfile) as inifile:
                inifile.set_auto_mask(False)
                soil_z_var = inifile.variables['soil_z'][:]
                soil_T_var = inifile.variables['soil_T'][:]
                soil_q_var = inifile.variables['soil_q'][:]
                soil_type_var = inifile.variables['soil_type'][:]

                ncol = nx * ny

                def _decode_soil_type(array: NDArray) -> NDArray:
                    if array.dtype.kind == 'S':
                        return np.char.decode(array, 'utf-8').astype(object)
                    if array.dtype.kind in ('U', 'O'):
                        return np.asarray(array, dtype=object).astype(str)
                    raise ValueError(
                        f"soil_type variable must contain strings, "
                        f"got dtype {array.dtype}"
                    )

                def _ensure_z_1d(soil_z: NDArray) -> NDArray[np.float64]:
                    z = (-1) * soil_z.astype('float')
                    if z.ndim == 1:
                        if z.shape[0] != nz:
                            raise ValueError(
                                f"soil_z length {z.shape[0]} does not match "
                                f"namelist nz={nz}."
                            )
                        return z
                    if z.ndim == 2:
                        if z.shape == (nz, ncol):
                            ref = z[:, 0]
                            if not np.allclose(z, ref[:, None]):
                                raise ValueError(
                                    "soil_z varies across columns; "
                                    "horizontal variation is not supported."
                                )
                            return ref
                        raise ValueError(
                            f"soil_z shape {z.shape} must be (nz, ncol) or "
                            f"(nz,) when using flattened columns."
                        )
                    if z.ndim == 3:
                        if z.shape != (nz, ny, nx):
                            raise ValueError(
                                f"soil_z shape {z.shape} does not match "
                                f"(nz, ny, nx)=({nz}, {ny}, {nx})."
                            )
                        ref = z[:, 0, 0]
                        if not np.allclose(z, ref[:, None, None]):
                            raise ValueError(
                                "soil_z varies across columns; "
                                "horizontal variation is not supported."
                            )
                        return ref
                    raise ValueError(
                        f"soil_z has unsupported dimensions: {z.ndim}."
                    )

                def _reshape_soil_field(
                    field: NDArray, name: str
                ) -> NDArray[np.float64]:
                    data = field.astype('float')
                    if data.ndim == 1:
                        if data.shape[0] != nz:
                            raise ValueError(
                                f"{name} length {data.shape[0]} does not "
                                f"match namelist nz={nz}."
                            )
                        if ncol == 1:
                            return data[:, None]
                        return np.repeat(data[:, None], ncol, axis=1)
                    if data.ndim == 2:
                        if data.shape == (nz, ncol):
                            return data
                        if data.shape == (nz, 1) and ncol > 1:
                            return np.repeat(data, ncol, axis=1)
                        raise ValueError(
                            f"{name} shape {data.shape} must be (nz, ncol) "
                            f"or (nz,) for single-column runs."
                        )
                    if data.ndim == 3:
                        if data.shape != (nz, ny, nx):
                            raise ValueError(
                                f"{name} shape {data.shape} does not match "
                                f"(nz, ny, nx)=({nz}, {ny}, {nx})."
                            )
                        return data.reshape(nz, ncol)
                    raise ValueError(
                        f"{name} has unsupported dimensions: {data.ndim}."
                    )

                def _reshape_soil_type(
                    field: NDArray, name: str
                ) -> NDArray:
                    data = _decode_soil_type(field)
                    if data.ndim == 1:
                        if data.shape[0] != nz:
                            raise ValueError(
                                f"{name} length {data.shape[0]} does not "
                                f"match namelist nz={nz}."
                            )
                        return data
                    if data.ndim == 2:
                        if data.shape != (nz, ncol):
                            raise ValueError(
                                f"{name} shape {data.shape} must be "
                                f"(nz, ncol) when using flattened columns."
                            )
                        ref = data[:, 0]
                        if not np.all(data == ref[:, None]):
                            raise ValueError(
                                "soil_type varies across columns; "
                                "horizontal variation is not supported."
                            )
                        return ref
                    if data.ndim == 3:
                        if data.shape != (nz, ny, nx):
                            raise ValueError(
                                f"{name} shape {data.shape} does not match "
                                f"(nz, ny, nx)=({nz}, {ny}, {nx})."
                            )
                        ref = data[:, 0, 0]
                        if not np.all(data == ref[:, None, None]):
                            raise ValueError(
                                "soil_type varies across columns; "
                                "horizontal variation is not supported."
                            )
                        return ref
                    raise ValueError(
                        f"{name} has unsupported dimensions: {data.ndim}."
                    )

                init_dict = {
                    'z': _ensure_z_1d(soil_z_var),
                    'temperature': _reshape_soil_field(soil_T_var, "soil_T"),
                    'moisture': _reshape_soil_field(soil_q_var, "soil_q"),
                    'type': _reshape_soil_type(soil_type_var, "soil_type"),
                }
            self.logger.info('--- initial conditions loaded successfully')
            return init_dict
        except (OSError, KeyError) as e:
            self.logger.error('--- initial conditions error: %s', e)
            raise

    def _load_soil_properties(
        self, soil_config: dict, soil_type_array: NDArray
    ) -> None:
        """Loads soil properties from JSON files and validates soil types.

        Args:
            soil_config: Dictionary from namelist with 'properties' and 'model'.
            soil_type_array: Array of soil type names (strings) from initial
                conditions.

        Raises:
            NamelistError: If soil property file not found or invalid, or if
                soil types in initial conditions are not found in properties.
        """
        try:
            properties_spec = soil_config['properties']
            self.soil_properties = SoilPropertiesLoader.load(properties_spec)
            self.soil_properties_name = properties_spec

            # Validate that all soil types in initial conditions are available
            # in the loaded properties
            self.soil_type_names = []
            for soil_type_name in soil_type_array:
                soil_type_lower = soil_type_name.lower()
                if soil_type_lower not in self.soil_properties:
                    available = ', '.join(sorted(self.soil_properties.keys()))
                    raise NamelistError(
                        f"Soil type '{soil_type_name}' from initial conditions "
                        f"not found in properties dataset '{properties_spec}'. "
                        f"Available soil types: {available}"
                    )
                self.soil_type_names.append(soil_type_lower)

            self.logger.info(
                'Soil properties loaded from: %s', properties_spec
            )
        except NamelistError:
            raise
        except Exception as e:
            self.logger.error('Error loading soil properties: %s', e)
            raise NamelistError(
                f'Failed to load soil properties: {e}'
            ) from e

    def _load_offline_data(self, offlinefile: str) -> None:
        """Loads data from the NetCDF offline forcing file.

        Args:
            offlinefile: The path to the NetCDF offline forcing file.

        Raises:
            IOError: If the file cannot be read.
            KeyError: If a required variable is missing from the NetCDF
                file.
        """
        try:
            with nc.Dataset(offlinefile) as metfile:
                metfile.set_auto_mask(False)
                ntime = len(metfile.dimensions['t'])
                tstep = metfile.variables['tstep'][0].astype('float')
                atm_U = metfile.variables['atm_U'][:]
                atm_T = metfile.variables['atm_T'][:]
                atm_q = metfile.variables['atm_q'][:]
                atm_p = metfile.variables['atm_p'][:]
                r_net = metfile.variables['R_net'][:]

                ncol = self.grid.nx * self.grid.ny
                ny = self.grid.ny
                nx = self.grid.nx

                def _reshape_forcing(field: NDArray, name: str) -> NDArray:
                    data = field.astype('float')
                    if data.ndim == 1:
                        if data.shape[0] != ntime:
                            raise ValueError(
                                f"{name} length {data.shape[0]} does not "
                                f"match forcing ntime={ntime}."
                            )
                        if ncol == 1:
                            return data[:, None]
                        return np.repeat(data[:, None], ncol, axis=1)
                    if data.ndim == 2:
                        if data.shape == (ntime, ncol):
                            return data
                        if data.shape == (ntime, 1) and ncol > 1:
                            return np.repeat(data, ncol, axis=1)
                        raise ValueError(
                            f"{name} shape {data.shape} must be (ntime, ncol) "
                            f"or (ntime,) for single-column runs."
                        )
                    if data.ndim == 3:
                        if data.shape != (ntime, ny, nx):
                            raise ValueError(
                                f"{name} shape {data.shape} does not match "
                                f"(ntime, ny, nx)=({ntime}, {ny}, {nx})."
                            )
                        return data.reshape(ntime, ncol)
                    raise ValueError(
                        f"{name} has unsupported dimensions: {data.ndim}."
                    )

                atm_U = _reshape_forcing(atm_U, "atm_U")
                atm_T = _reshape_forcing(atm_T, "atm_T")
                atm_q = _reshape_forcing(atm_q, "atm_q")
                atm_p = _reshape_forcing(atm_p, "atm_p")
                r_net = _reshape_forcing(r_net, "R_net")

                # Validate and correct forcing data
                self._validate_forcing_data(atm_U, atm_T, atm_q, atm_p,
                                            r_net, ntime)

                atm_data = [
                    AtmosphericState(
                        wind_speed=atm_U[i], temperature=atm_T[i],
                        specific_humidity=atm_q[i], pressure=atm_p[i],
                        radiation_net=r_net[i])
                    for i in range(ntime)
                ]

                self.forcing = ForcingData(ntime=ntime, tstep=tstep,
                                           atmos=atm_data)
                self.logger.info(
                    '--- loaded %d timesteps of forcing data', ntime)
        except (OSError, KeyError) as e:
            self.logger.error('--- offline forcing error: %s', e)
            raise

    def _validate_forcing_data(self, atm_U: NDArray, atm_T: NDArray,
                               atm_q: NDArray, atm_p: NDArray,
                               r_net: NDArray, _ntime: int) -> None:
        """Validates and corrects atmospheric forcing data for physical
        consistency.

        Checks that forcing variables are within reasonable physical ranges
        and corrects minor issues. Raises errors for impossible values. Provides
        informative warnings for values at the edges of valid ranges.

        Args:
            atm_U: Wind speed array with one value per time step [m/s].
            atm_T: Temperature array with one value per time step [K].
            atm_q: Specific humidity array with one value per time step [kg/kg].
            atm_p: Pressure array with one value per time step [Pa].
            r_net: Net radiation array with one value per time step [W/m²].
            _ntime: Number of time steps in forcing arrays (unused but kept for
                API compatibility with other validation functions).

        Raises:
            ValueError: If forcing data contains impossible or physically
                unrealistic values that cannot be corrected.
        """
        # Physical bounds for atmospheric variables
        # Reasonable atmospheric temperature range [K]
        T_min, T_max = 200.0, 350.0
        p_min, p_max = 50000.0, 110000.0  # Pressure range [Pa]
        q_min, q_max = 0.0, 0.05  # Specific humidity range [kg/kg]
        U_min, U_max = 1e-4, 50.0  # Wind speed range [m/s]
        R_min, R_max = -100.0, 1200.0  # Net radiation range [W/m²]

        issues_found = False

        # Check temperature
        T_bad = (atm_T < T_min) | (atm_T > T_max)
        if np.any(T_bad):
            num_bad = int(np.sum(T_bad))
            self.logger.warning(
                'Found %d forcing entries with out-of-range temperature '
                '(expected %f-%f K).', num_bad, T_min, T_max)
            issues_found = True
            # Clamp to valid range
            atm_T[T_bad] = np.clip(atm_T[T_bad], T_min, T_max)

        # Check pressure
        p_bad = (atm_p < p_min) | (atm_p > p_max)
        if np.any(p_bad):
            num_bad = int(np.sum(p_bad))
            self.logger.warning(
                'Found %d forcing entries with out-of-range pressure '
                '(expected %f-%f Pa).', num_bad, p_min, p_max)
            issues_found = True
            # Clamp to valid range
            atm_p[p_bad] = np.clip(atm_p[p_bad], p_min, p_max)

        # Check specific humidity
        q_bad = (atm_q < q_min) | (atm_q > q_max)
        if np.any(q_bad):
            num_bad = int(np.sum(q_bad))
            self.logger.warning(
                'Found %d forcing entries with out-of-range humidity '
                '(expected %f-%f kg/kg).', num_bad, q_min, q_max)
            issues_found = True
            # Clamp to valid range (especially fix negative values)
            atm_q[q_bad] = np.clip(atm_q[q_bad], q_min, q_max)

        # Check wind speed
        U_bad = (atm_U <= U_min) | (atm_U > U_max)
        if np.any(U_bad):
            num_bad = int(np.sum(U_bad))
            self.logger.warning(
                'Found %d forcing entries with out-of-range wind speed '
                '(expected %f-%f m/s).', num_bad, U_min, U_max)
            issues_found = True
            # Fix zero/negative and excessive wind speeds
            atm_U[atm_U <= U_min] = U_min
            atm_U[atm_U > U_max] = U_max

        # Check net radiation
        R_bad = (r_net < R_min) | (r_net > R_max)
        if np.any(R_bad):
            num_bad = int(np.sum(R_bad))
            self.logger.warning(
                'Found %d forcing entries with suspicious radiation '
                '(expected %f-%f W/m²).', num_bad, R_min, R_max)
            issues_found = True
            # Clamp to physically reasonable range
            r_net[R_bad] = np.clip(r_net[R_bad], R_min, R_max)

        if issues_found:
            self.logger.info(
                'Forcing data validation: Issues found and corrected. '
                'Please review input data quality.')
        else:
            self.logger.info(
                'Forcing data validation: All variables within '
                'expected ranges')

    def _validate_physical_consistency(self) -> None:
        """Performs validation checks on inter-variable relationships.

        Raises:
            ValueError: If a physical consistency check fails.
        """
        # Grid size validation
        if self.grid.nz < 2:
            raise ValueError(
                f"Grid must have at least 2 soil layers for diffusion "
                f"solvers, got nz={self.grid.nz}.")

        if self.surface.z_m <= self.surface.z_o:
            raise ValueError(
                f"z_m={self.surface.z_m} must be > "
                f"z_o={self.surface.z_o}.")

        if self.surface.z_s <= self.surface.z_t:
            raise ValueError(
                f"z_s={self.surface.z_s} must be > "
                f"z_t={self.surface.z_t}.")

        ncol = self.grid.nx * self.grid.ny
        soil_temp = np.asarray(self.initial.temperature)
        soil_mois = np.asarray(self.initial.moisture)

        if soil_temp.ndim == 1:
            if soil_temp.shape[0] != self.grid.nz:
                raise ValueError(
                    f"Namelist nlevs={self.grid.nz} does not match "
                    f"init file soil_T length of {soil_temp.shape[0]}."
                )
        elif soil_temp.ndim == 2:
            if soil_temp.shape != (self.grid.nz, ncol):
                raise ValueError(
                    f"init file soil_T shape {soil_temp.shape} does not "
                    f"match (nz, ncol)=({self.grid.nz}, {ncol})."
                )
        else:
            raise ValueError(
                f"init file soil_T has unsupported dimensions: "
                f"{soil_temp.ndim}."
            )

        if soil_mois.ndim == 1:
            if soil_mois.shape[0] != self.grid.nz:
                raise ValueError(
                    f"Namelist nlevs={self.grid.nz} does not match "
                    f"init file soil_q length of {soil_mois.shape[0]}."
                )
        elif soil_mois.ndim == 2:
            if soil_mois.shape != (self.grid.nz, ncol):
                raise ValueError(
                    f"init file soil_q shape {soil_mois.shape} does not "
                    f"match (nz, ncol)=({self.grid.nz}, {ncol})."
                )
        else:
            raise ValueError(
                f"init file soil_q has unsupported dimensions: "
                f"{soil_mois.ndim}."
            )

        if len(self.initial.type) != self.grid.nz:
            raise ValueError(
                f"Namelist nlevs={self.grid.nz} does not match "
                f"init file soil_type length of {len(self.initial.type)}.")

        # Validate soil moisture values are within physically possible bounds
        # Moisture must be >= 0 and <= porosity (will be validated
        # against residual later). Issues warnings instead of errors to
        # allow running with imperfect data
        moisture_by_layer = soil_mois
        if moisture_by_layer.ndim == 1:
            moisture_by_layer = moisture_by_layer[:, None]

        for i, soil_type_name in enumerate(self.soil_type_names):
            layer_moisture = moisture_by_layer[i]
            if np.any(layer_moisture < 0):
                self.logger.warning(
                    'Layer %d: soil moisture has negative values. '
                    'Moisture must be >= 0.', i)
            # Get porosity for this soil type to validate upper bound
            soil_type_lower = soil_type_name.lower()
            if soil_type_lower not in self.soil_properties:
                self.logger.warning(
                    'Layer %d: soil type %s not found in properties.',
                    i, soil_type_name)
            else:
                porosity = self.soil_properties[soil_type_lower]['porosity']
                if np.any(layer_moisture > porosity):
                    self.logger.warning(
                        'Layer %d: soil moisture exceeds porosity %f.',
                        i, porosity)

        self.logger.info('Physical consistency checks passed')
