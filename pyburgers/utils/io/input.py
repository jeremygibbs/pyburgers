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
"""Handles all input data loading and validation for PyBurgers.

This module defines the `Input` class, which is responsible for reading the
JSON namelist and populating the configuration dataclasses. It validates
the inputs and provides a single, clean interface for the simulation to
access all setup information.
"""
import json
import logging
import math
from typing import Any

from ..logging_helper import get_logger, setup_logging
from ...data_models import (
    DNSConfig,
    FFTWConfig,
    LESConfig,
    LoggingConfig,
    ModelConfig,
    NoiseConfig,
    OutputConfig,
    PhysicsConfig,
    TimeConfig,
)
from ...exceptions import NamelistError


class Input:
    """Orchestrates the loading and validation of all model inputs.

    This class reads configuration from a JSON namelist file. All data is
    validated and organized into the appropriate dataclasses.

    Attributes:
        time: Dataclass with time-related parameters (nt, dt).
        physics: Dataclass with physics parameters (noise, viscosity).
        models: Dataclass with DNS and LES configurations.
        output: Dataclass with output file configuration.
        logging: Dataclass with logging settings.
        fftw: Dataclass with FFTW configuration.
        log_level: Convenience accessor for logging level string.
        fftw_planning: Convenience accessor for FFTW planning strategy.
        fftw_threads: Convenience accessor for FFTW thread count.
    """

    def __init__(self, namelist_path: str) -> None:
        """Initialize the Input class and load all configuration.

        Args:
            namelist_path: The file path to the JSON namelist.

        Raises:
            FileNotFoundError: If the namelist file does not exist.
            json.JSONDecodeError: If the namelist JSON file is malformed.
            NamelistError: If required configuration is missing or invalid.
        """
        # Set up basic logging before we can read the log level
        setup_logging(level="INFO")
        self.logger: logging.Logger = get_logger('Input')
        self.logger.info('Reading %s', namelist_path)

        namelist_data = self._load_namelist(namelist_path)
        self._validate_namelist(namelist_data)

        # Extract and finalize logging config first so we can adjust log level
        logging_data = namelist_data.get("logging", {})
        log_level = logging_data.get("level", "INFO")
        log_file = logging_data.get("file")
        if log_file == "":
            log_file = None
        self.logging: LoggingConfig = LoggingConfig(level=log_level, file=log_file)
        setup_logging(level=log_level, log_file=log_file)

        # Time configuration
        time_data = namelist_data['time']
        self.time: TimeConfig = TimeConfig(
            nt=int(time_data['nt']),
            dt=float(time_data['dt'])
        )

        # Physics configuration
        physics_data = namelist_data['physics']
        noise_data = physics_data.get('noise', {})
        self.physics: PhysicsConfig = PhysicsConfig(
            noise=NoiseConfig(
                alpha=float(noise_data.get('alpha', 0.75)),
                amplitude=float(noise_data.get('amplitude', 1e-6))
            ),
            viscosity=float(physics_data['viscosity'])
        )

        # Models configuration (DNS and LES)
        models_data = namelist_data['models']
        dns_data = models_data.get('dns', {})
        les_data = models_data.get('les', {})
        self.models: ModelConfig = ModelConfig(
            dns=DNSConfig(nx=int(dns_data.get('nx', 8192))),
            les=LESConfig(
                nx=int(les_data.get('nx', 512)),
                sgs=int(les_data.get('sgs', 1))
            )
        )

        # Output configuration
        output_data = namelist_data.get("output", {})
        default_t_save = 1000 * self.time.dt
        self.output: OutputConfig = OutputConfig(
            t_save=float(output_data.get("t_save", default_t_save))
        )
        self._step_save = max(1, int(round(self.output.t_save / self.time.dt)))
        self._t_save_effective = self._step_save * self.time.dt
        if not math.isclose(
            self._t_save_effective,
            self.output.t_save,
            rel_tol=0.0,
            abs_tol=0.5 * self.time.dt,
        ):
            self.logger.warning(
                "Requested t_save=%g not aligned with dt=%g; using %g (steps=%d)",
                self.output.t_save,
                self.time.dt,
                self._t_save_effective,
                self._step_save,
            )

        # FFTW configuration
        fftw_data = namelist_data.get('fftw', {})
        self.fftw: FFTWConfig = FFTWConfig(
            planning=str(fftw_data.get('planning', 'FFTW_MEASURE')),
            threads=int(fftw_data.get('threads', 4))
        )

        self._log_configuration()
        self.logger.info('--- namelist loaded successfully')

    @property
    def log_level(self) -> str:
        """Convenience accessor for log level."""
        return self.logging.level

    @property
    def fftw_planning(self) -> str:
        """Convenience accessor for FFTW planning strategy."""
        return self.fftw.planning

    @property
    def fftw_threads(self) -> int:
        """Convenience accessor for FFTW thread count."""
        return self.fftw.threads

    @property
    def dt(self) -> float:
        """Convenience accessor for time step."""
        return self.time.dt

    @property
    def nt(self) -> int:
        """Convenience accessor for number of time steps."""
        return self.time.nt

    @property
    def viscosity(self) -> float:
        """Convenience accessor for viscosity."""
        return self.physics.viscosity

    @property
    def step_save(self) -> int:
        """Convenience accessor for save interval (in time steps)."""
        return self._step_save

    @property
    def t_save(self) -> float:
        """Convenience accessor for save interval (in seconds)."""
        return self.output.t_save

    def _load_namelist(self, namelist_path: str) -> dict[str, Any]:
        """Load the JSON namelist file.

        Args:
            namelist_path: The path to the JSON namelist file.

        Returns:
            A dictionary containing the namelist data.

        Raises:
            FileNotFoundError: If the namelist file cannot be found.
            json.JSONDecodeError: If the namelist is not valid JSON.
        """
        try:
            with open(namelist_path, encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError as e:
            self.logger.error('Namelist file not found: %s', namelist_path)
            raise
        except json.JSONDecodeError as e:
            self.logger.error('Invalid JSON in namelist: %s', e)
            raise

    def _validate_namelist(self, data: dict[str, Any]) -> None:
        """Validate required namelist sections and values.

        Args:
            data: The namelist dictionary to validate.

        Raises:
            NamelistError: If required sections or values are missing.
        """
        required_sections = ['time', 'physics', 'models']
        for section in required_sections:
            if section not in data:
                raise NamelistError(f"Missing required section: '{section}'")

        # Validate time section
        time_data = data['time']
        if 'nt' not in time_data:
            raise NamelistError("Missing 'nt' in time section")
        if 'dt' not in time_data:
            raise NamelistError("Missing 'dt' in time section")
        if float(time_data['dt']) <= 0:
            raise NamelistError("'dt' must be positive")
        if int(time_data['nt']) <= 0:
            raise NamelistError("'nt' must be positive")

        # Validate physics section
        physics_data = data['physics']
        if 'viscosity' not in physics_data:
            raise NamelistError("Missing 'viscosity' in physics section")
        if float(physics_data['viscosity']) <= 0:
            raise NamelistError("'viscosity' must be positive")

        # Validate models section
        models_data = data['models']
        if 'dns' not in models_data and 'les' not in models_data:
            raise NamelistError(
                "At least one of 'dns' or 'les' must be in models section"
            )

        # Validate DNS config if present
        if 'dns' in models_data:
            dns_data = models_data['dns']
            if 'nx' in dns_data and int(dns_data['nx']) <= 0:
                raise NamelistError("DNS 'nx' must be positive")

        # Validate LES config if present
        if 'les' in models_data:
            les_data = models_data['les']
            if 'nx' in les_data and int(les_data['nx']) <= 0:
                raise NamelistError("LES 'nx' must be positive")
            if 'sgs' in les_data:
                sgs = int(les_data['sgs'])
                if sgs < 0 or sgs > 4:
                    raise NamelistError(
                        f"LES 'sgs' must be 0-4, got {sgs}"
                    )

        # Validate output config if present
        if "output" in data:
            output_data = data["output"]
            if "t_save" in output_data and float(output_data["t_save"]) <= 0:
                raise NamelistError("'t_save' must be positive")

        # Validate FFTW config if present
        if 'fftw' in data:
            fftw_data = data['fftw']
            valid_planning = [
                'FFTW_ESTIMATE', 'FFTW_MEASURE',
                'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'
            ]
            planning = fftw_data.get('planning', 'FFTW_MEASURE')
            if planning not in valid_planning:
                raise NamelistError(
                    f"Invalid FFTW planning: '{planning}'. "
                    f"Valid options: {valid_planning}"
                )
            threads = fftw_data.get('threads', 4)
            if int(threads) < 1:
                raise NamelistError("FFTW 'threads' must be at least 1")

    def _log_configuration(self) -> None:
        """Log the loaded configuration for debugging."""
        self.logger.debug('Time: nt=%d, dt=%g', self.time.nt, self.time.dt)
        self.logger.debug(
            'Physics: viscosity=%g, noise(alpha=%g, amp=%g)',
            self.physics.viscosity,
            self.physics.noise.alpha,
            self.physics.noise.amplitude
        )
        self.logger.debug('DNS: nx=%d', self.models.dns.nx)
        self.logger.debug(
            'LES: nx=%d, sgs=%d',
            self.models.les.nx,
            self.models.les.sgs
        )
        self.logger.debug(
            "Output: t_save=%g (steps=%d, effective=%g)",
            self.output.t_save,
            self._step_save,
            self._t_save_effective,
        )
        self.logger.debug(
            "Logging: level=%s, file=%s",
            self.logging.level,
            self.logging.file,
        )
        self.logger.debug(
            'FFTW: planning=%s, threads=%d',
            self.fftw.planning,
            self.fftw.threads
        )

    def get_dns_config(self) -> dict[str, Any]:
        """Get DNS-specific configuration as a dictionary.

        Returns:
            Dictionary with DNS configuration values.
        """
        return {
            'nx': self.models.dns.nx,
            'dt': self.time.dt,
            'nt': self.time.nt,
            'viscosity': self.physics.viscosity,
            'noise_alpha': self.physics.noise.alpha,
            'noise_amplitude': self.physics.noise.amplitude,
            't_save': self.output.t_save,
        }

    def get_les_config(self) -> dict[str, Any]:
        """Get LES-specific configuration as a dictionary.

        Returns:
            Dictionary with LES configuration values.
        """
        return {
            'nx': self.models.les.nx,
            'sgs': self.models.les.sgs,
            'dt': self.time.dt,
            'nt': self.time.nt,
            'viscosity': self.physics.viscosity,
            'noise_alpha': self.physics.noise.alpha,
            'noise_amplitude': self.physics.noise.amplitude,
            't_save': self.output.t_save,
        }
