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
"""Handles all input data loading and validation for PyBurgers.

This module defines the `Input` class, which is responsible for reading the
JSON namelist and populating the configuration dataclasses. It validates
the inputs and provides a single, clean interface for the simulation to
access all setup information.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

import jsonschema

from ...data_models import (
    DNSConfig,
    FFTWConfig,
    GridConfig,
    LESConfig,
    LoggingConfig,
    NoiseConfig,
    NumericsConfig,
    OutputConfig,
    PhysicsConfig,
    TimeConfig,
)
from ...exceptions import NamelistError
from ..logging_helper import get_logger, setup_logging


class Input:
    """Orchestrates the loading and validation of all model inputs.

    This class reads configuration from a JSON namelist file. All data is
    validated and organized into the appropriate dataclasses.

    Attributes:
        time: Dataclass with time-related parameters (duration).
        physics: Dataclass with physics parameters (noise, viscosity).
        grid: Dataclass with grid configuration (length, DNS, LES).
        numerics: Dataclass with numerical method selections (integration, cfl, max_step).
        output: Dataclass with output file configuration.
        logging: Dataclass with logging settings.
        fftw: Dataclass with FFTW configuration.
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
        self.logger: logging.Logger = get_logger("Input")
        self.logger.info("Reading %s", namelist_path)

        namelist_data = self._load_and_validate_namelist(namelist_path)

        # Extract and finalize logging config first so we can adjust log level
        logging_data = namelist_data.get("logging", {})
        log_level = logging_data.get("level", "INFO")
        log_file = logging_data.get("file")
        if log_file == "":
            log_file = None
        self.logging: LoggingConfig = LoggingConfig(level=log_level, file=log_file)
        setup_logging(level=log_level, log_file=log_file)

        # Time configuration
        time_data = namelist_data["time"]
        duration = float(time_data["duration"])
        self.time: TimeConfig = TimeConfig(duration=duration)

        # Numerics configuration
        numerics_data = namelist_data.get("numerics", {})
        self.numerics: NumericsConfig = NumericsConfig(
            temporal=int(numerics_data.get("temporal", 3)),
            spatial=int(numerics_data.get("spatial", 3)),
            cfl=float(numerics_data.get("cfl", 0.4)),
            max_step=float(numerics_data.get("max_step", 0.01)),
        )

        # AB2 has a limited stability region for hyperbolic terms; warn if CFL
        # is set higher than the recommended limit for that scheme.
        if self.numerics.temporal == 1 and self.numerics.cfl > 0.4:
            self.logger.warning(
                "CFL target %.2f exceeds the recommended limit of 0.4 for AB2 "
                "(temporal=1). Consider reducing cfl or switching to RK3 (temporal=3).",
                self.numerics.cfl,
            )

        # Grid configuration (DNS and LES)
        grid_data = namelist_data["grid"]
        dns_data = grid_data.get("dns", {})
        les_data = grid_data.get("les", {})
        self.grid: GridConfig = GridConfig(
            length=float(grid_data.get("length", math.tau)),
            dns=DNSConfig(points=int(dns_data.get("points", 8192))),
            les=LESConfig(points=int(les_data.get("points", 512))),
        )

        # Physics configuration
        physics_data = namelist_data["physics"]
        noise_data = physics_data.get("noise", {})
        self.physics: PhysicsConfig = PhysicsConfig(
            noise=NoiseConfig(
                exponent=float(noise_data.get("exponent", -0.75)),
                amplitude=float(noise_data.get("amplitude", 1e-6)),
                seed=noise_data.get("seed", None),
            ),
            viscosity=float(physics_data["viscosity"]),
            subgrid_model=int(physics_data.get("subgrid_model", 1)),
        )

        # Output configuration
        output_data = namelist_data.get("output", {})
        default_interval = 100 * self.numerics.max_step
        self.output: OutputConfig = OutputConfig(
            interval_save=float(output_data.get("interval_save", default_interval)),
            interval_print=float(output_data.get("interval_print", default_interval)),
        )

        # FFTW configuration
        fftw_data = namelist_data.get("fftw", {})
        self.fftw: FFTWConfig = FFTWConfig(
            planning=str(fftw_data.get("planning", "FFTW_MEASURE")),
            threads=int(fftw_data.get("threads", 4)),
        )

        self._log_configuration()
        self.logger.info("--- namelist loaded successfully")

    # --- Convenience accessors ---

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
    def cfl_target(self) -> float:
        """Target CFL number for adaptive time stepping."""
        return self.numerics.cfl

    @property
    def max_step(self) -> float:
        """Maximum allowed time step."""
        return self.numerics.max_step

    @property
    def domain_length(self) -> float:
        """Convenience accessor for domain length."""
        return self.grid.length

    @property
    def viscosity(self) -> float:
        """Convenience accessor for viscosity."""
        return self.physics.viscosity

    @property
    def t_save(self) -> float:
        """Save interval in seconds."""
        return self.output.interval_save

    @property
    def t_print(self) -> float:
        """Print interval in seconds."""
        return self.output.interval_print

    @property
    def temporal(self) -> int:
        """Time integration scheme identifier."""
        return self.numerics.temporal

    def _load_and_validate_namelist(self, namelist_path: str) -> dict[str, Any]:
        """Load and validate the JSON namelist file against the schema.

        Args:
            namelist_path: The path to the JSON namelist file.

        Returns:
            A dictionary containing the validated namelist data.

        Raises:
            FileNotFoundError: If the namelist file cannot be found.
            json.JSONDecodeError: If the namelist is not valid JSON.
            NamelistError: If the namelist fails schema validation.
        """
        schema_path = str(Path(__file__).parent / "schema_namelist.json")

        try:
            with open(schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            with open(namelist_path, encoding="utf-8") as f:
                namelist_data = json.load(f)
            # Normalize log level to uppercase before validation
            if "logging" in namelist_data and "level" in namelist_data["logging"]:
                namelist_data["logging"]["level"] = namelist_data["logging"]["level"].upper()
            jsonschema.validate(instance=namelist_data, schema=schema)
            self.logger.debug("Namelist validation successful")
            return namelist_data
        except FileNotFoundError:
            self.logger.error("File not found: %s", namelist_path)
            raise
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON in namelist: %s", e)
            raise
        except jsonschema.ValidationError as e:
            self.logger.error("Namelist validation error: %s", e.message)
            raise NamelistError(e.message) from e

    def _log_configuration(self) -> None:
        """Log the loaded configuration for debugging."""
        self.logger.debug(
            "Time: duration=%g",
            self.time.duration,
        )
        self.logger.debug(
            "Numerics: temporal=%d, spatial=%d, cfl=%g, max_step=%g",
            self.numerics.temporal,
            self.numerics.spatial,
            self.numerics.cfl,
            self.numerics.max_step,
        )
        self.logger.debug(
            "Physics: viscosity=%g, noise(exponent=%g, amplitude=%g)",
            self.physics.viscosity,
            self.physics.noise.exponent,
            self.physics.noise.amplitude,
        )
        self.logger.debug(
            "Grid: length=%g, DNS points=%d, LES points=%d",
            self.grid.length,
            self.grid.dns.points,
            self.grid.les.points,
        )
        self.logger.debug("Subgrid model: %d", self.physics.subgrid_model)
        self.logger.debug(
            "Output: interval_save=%g, interval_print=%g",
            self.output.interval_save,
            self.output.interval_print,
        )
        self.logger.debug(
            "Logging: level=%s, file=%s",
            self.logging.level,
            self.logging.file,
        )
        self.logger.debug("FFTW: planning=%s, threads=%d", self.fftw.planning, self.fftw.threads)

    def get_dns_config(self) -> dict[str, Any]:
        """Get DNS-specific configuration as a dictionary.

        Returns:
            Dictionary with DNS configuration values.
        """
        return {
            "nx": self.grid.dns.points,
            "cfl": self.numerics.cfl,
            "max_step": self.numerics.max_step,
            "temporal": self.numerics.temporal,
            "viscosity": self.physics.viscosity,
            "noise_beta": self.physics.noise.exponent,
            "noise_amplitude": self.physics.noise.amplitude,
            "t_save": self.output.interval_save,
            "domain_length": self.grid.length,
        }

    def get_les_config(self) -> dict[str, Any]:
        """Get LES-specific configuration as a dictionary.

        Returns:
            Dictionary with LES configuration values.
        """
        return {
            "nx": self.grid.les.points,
            "sgs_model": self.physics.subgrid_model,
            "cfl": self.numerics.cfl,
            "max_step": self.numerics.max_step,
            "temporal": self.numerics.temporal,
            "viscosity": self.physics.viscosity,
            "noise_beta": self.physics.noise.exponent,
            "noise_amplitude": self.physics.noise.amplitude,
            "t_save": self.output.interval_save,
            "domain_length": self.grid.length,
        }
