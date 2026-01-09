"""Logging configuration for pyBurgers.

This module provides centralized logging setup and helper functions
for consistent logging across all pyBurgers modules.
"""
from __future__ import annotations

import logging
import sys
from typing import Literal

# Valid log level names
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default format for log messages
DEFAULT_FORMAT = "[pyBurgers: %(name)s] \t %(message)s"
DEBUG_FORMAT = "[pyBurgers: %(name)s] \t %(levelname)s - %(message)s"

# Cache of created loggers
_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    level: str | int = "INFO",
    format_string: str | None = None,
) -> None:
    """Configure the root logger for pyBurgers.

    This should be called once at application startup, typically in
    the main entry point (burgers.py).

    Args:
        level: Log level as string ("DEBUG", "INFO", etc.) or int.
        format_string: Optional custom format string. If None, uses
            DEBUG_FORMAT for DEBUG level, DEFAULT_FORMAT otherwise.
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Choose format based on level
    if format_string is None:
        format_string = DEBUG_FORMAT if level <= logging.DEBUG else DEFAULT_FORMAT

    # Configure root logger
    root_logger = logging.getLogger("pyBurgers")
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))

    root_logger.addHandler(handler)

    # Prevent propagation to root logger
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified module/class.

    Creates a child logger under the pyBurgers namespace for consistent
    hierarchical logging.

    Args:
        name: Name for the logger, typically the class or module name.
            Examples: "DNS", "LES", "SGS", "Input", "Output"

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger("DNS")
        >>> logger.info("Starting simulation")
        [pyBurgers: DNS]     Starting simulation
    """
    full_name = f"pyBurgers.{name}"

    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)

    return _loggers[full_name]


def get_log_level(level_name: str) -> int:
    """Convert a log level name to its integer value.

    Args:
        level_name: Case-insensitive level name ("debug", "INFO", etc.)

    Returns:
        Integer log level value.

    Raises:
        ValueError: If level_name is not a valid log level.
    """
    level_name = level_name.upper()
    level = getattr(logging, level_name, None)

    if level is None:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        raise ValueError(
            f"Invalid log level: '{level_name}'. "
            f"Valid options: {', '.join(valid_levels)}"
        )

    return level
