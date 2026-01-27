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

from ...data_models import (
    DNSConfig,
    FFTWConfig,
    GridConfig,
    LESConfig,
    LoggingConfig,
    NoiseConfig,
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
        time: Dataclass with time-related parameters (duration, step).
        physics: Dataclass with physics parameters (noise, viscosity).
        grid: Dataclass with grid configuration (length, DNS, LES).
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
        time_data = namelist_data["time"]
        duration = float(time_data["duration"])
        step = float(time_data["step"])
        self.time: TimeConfig = TimeConfig(duration=duration, step=step)

        # Compute number of time steps (derived quantity)
        self._nt = int(round(duration / step))
        if self._nt <= 0:
            raise NamelistError("Computed number of time steps must be positive")

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
                exponent=float(noise_data.get("exponent", 0.75)),
                amplitude=float(noise_data.get("amplitude", 1e-6)),
            ),
            viscosity=float(physics_data["viscosity"]),
            subgrid_model=int(physics_data.get("subgrid_model", 1)),
        )

        # Output configuration
        output_data = namelist_data.get("output", {})
        default_interval = 1000 * step
        self.output: OutputConfig = OutputConfig(
            interval_save=float(output_data.get("interval_save", default_interval)),
            interval_print=float(output_data.get("interval_print", default_interval)),
        )

        # Compute step intervals
        self._step_save = max(1, int(round(self.output.interval_save / step)))
        self._interval_save_effective = self._step_save * step
        if not math.isclose(
            self._interval_save_effective,
            self.output.interval_save,
            rel_tol=0.0,
            abs_tol=0.5 * step,
        ):
            self.logger.warning(
                "Requested interval_save=%g not aligned with step=%g; using %g (steps=%d)",
                self.output.interval_save,
                step,
                self._interval_save_effective,
                self._step_save,
            )

        self._step_print = max(1, int(round(self.output.interval_print / step)))
        self._interval_print_effective = self._step_print * step
        if not math.isclose(
            self._interval_print_effective,
            self.output.interval_print,
            rel_tol=0.0,
            abs_tol=0.5 * step,
        ):
            self.logger.warning(
                "Requested interval_print=%g not aligned with step=%g; using %g (steps=%d)",
                self.output.interval_print,
                step,
                self._interval_print_effective,
                self._step_print,
            )

        # FFTW configuration
        fftw_data = namelist_data.get("fftw", {})
        self.fftw: FFTWConfig = FFTWConfig(
            planning=str(fftw_data.get("planning", "FFTW_MEASURE")),
            threads=int(fftw_data.get("threads", 4)),
        )

        self._log_configuration()
        self.logger.info("--- namelist loaded successfully")

    # --- Convenience accessors (maintain backward compatibility) ---

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
        return self.time.step

    @property
    def nt(self) -> int:
        """Number of time steps (derived from duration / step)."""
        return self._nt

    @property
    def domain_length(self) -> float:
        """Convenience accessor for domain length."""
        return self.grid.length

    @property
    def viscosity(self) -> float:
        """Convenience accessor for viscosity."""
        return self.physics.viscosity

    @property
    def step_save(self) -> int:
        """Save interval in time steps."""
        return self._step_save

    @property
    def t_save(self) -> float:
        """Save interval in seconds."""
        return self.output.interval_save

    @property
    def step_print(self) -> int:
        """Print interval in time steps."""
        return self._step_print

    @property
    def t_print(self) -> float:
        """Print interval in seconds."""
        return self.output.interval_print

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
            with open(namelist_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error("Namelist file not found: %s", namelist_path)
            raise
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON in namelist: %s", e)
            raise

    def _validate_namelist(self, data: dict[str, Any]) -> None:
        """Validate required namelist sections and values.

        Args:
            data: The namelist dictionary to validate.

        Raises:
            NamelistError: If required sections or values are missing.
        """
        required_sections = ["time", "physics", "grid"]
        for section in required_sections:
            if section not in data:
                raise NamelistError(f"Missing required section: '{section}'")

        # Validate time section
        time_data = data["time"]
        if "duration" not in time_data:
            raise NamelistError("Missing 'duration' in time section")
        if "step" not in time_data:
            raise NamelistError("Missing 'step' in time section")
        if float(time_data["step"]) <= 0:
            raise NamelistError("time 'step' must be positive")
        if float(time_data["duration"]) <= 0:
            raise NamelistError("time 'duration' must be positive")

        # Validate grid section
        grid_data = data["grid"]
        if "length" in grid_data and float(grid_data["length"]) <= 0:
            raise NamelistError("grid 'length' must be positive")

        # Validate physics section
        physics_data = data["physics"]
        if "viscosity" not in physics_data:
            raise NamelistError("Missing 'viscosity' in physics section")
        if float(physics_data["viscosity"]) <= 0:
            raise NamelistError("'viscosity' must be positive")

        if "dns" not in grid_data and "les" not in grid_data:
            raise NamelistError("At least one of 'dns' or 'les' must be in grid section")

        # Validate DNS config if present
        if "dns" in grid_data:
            dns_data = grid_data["dns"]
            if "points" in dns_data and int(dns_data["points"]) <= 0:
                raise NamelistError("dns 'points' must be positive")

        # Validate LES config if present
        if "les" in grid_data:
            les_data = grid_data["les"]
            if "points" in les_data and int(les_data["points"]) <= 0:
                raise NamelistError("les 'points' must be positive")

        if "subgrid_model" in physics_data:
            sgs = int(physics_data["subgrid_model"])
            if sgs < 0 or sgs > 4:
                raise NamelistError(f"physics 'subgrid_model' must be 0-4, got {sgs}")

        # Validate output config if present
        if "output" in data:
            output_data = data["output"]
            if "interval_save" in output_data and float(output_data["interval_save"]) <= 0:
                raise NamelistError("output 'interval_save' must be positive")
            if "interval_print" in output_data and float(output_data["interval_print"]) <= 0:
                raise NamelistError("output 'interval_print' must be positive")

        # Validate FFTW config if present
        if "fftw" in data:
            fftw_data = data["fftw"]
            valid_planning = ["FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE"]
            planning = fftw_data.get("planning", "FFTW_MEASURE")
            if planning not in valid_planning:
                raise NamelistError(
                    f"Invalid FFTW planning: '{planning}'. Valid options: {valid_planning}"
                )
            threads = fftw_data.get("threads", 4)
            if int(threads) < 1:
                raise NamelistError("FFTW 'threads' must be at least 1")

    def _log_configuration(self) -> None:
        """Log the loaded configuration for debugging."""
        self.logger.debug(
            "Time: duration=%g, step=%g, nt=%d",
            self.time.duration,
            self.time.step,
            self._nt,
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
            "Output: interval_save=%g (steps=%d), interval_print=%g (steps=%d)",
            self.output.interval_save,
            self._step_save,
            self.output.interval_print,
            self._step_print,
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
            "dt": self.time.step,
            "nt": self._nt,
            "viscosity": self.physics.viscosity,
            "noise_alpha": self.physics.noise.exponent,
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
            "dt": self.time.step,
            "nt": self._nt,
            "viscosity": self.physics.viscosity,
            "noise_alpha": self.physics.noise.exponent,
            "noise_amplitude": self.physics.noise.amplitude,
            "t_save": self.output.interval_save,
            "domain_length": self.grid.length,
        }
