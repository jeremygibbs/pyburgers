"""Logging configuration for pyBurgers.

This module provides centralized logging setup and helper functions
for consistent logging across all pyBurgers modules.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

# Valid log level names
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class _ShortNameFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        original_name = record.name
        record.name = original_name.rsplit(".", 1)[-1]
        try:
            return super().format(record)
        finally:
            record.name = original_name


# Log format
log_format = _ShortNameFormatter(
    '{asctime} [{levelname:^8s}] {name:.>10s}: {message}',
    datefmt='%Y-%m-%d %H:%M:%S', style='{'
)

# Cache of created loggers
_loggers: dict[str, logging.Logger] = {}


class _ProgressOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "progress", False)


class _SkipProgressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "progress", False)


class _ProgressHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.stream.write(f"\r{msg}")
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str | int = "INFO",
    format_string: str | None = None,
    log_file: str | None = None,
    file_mode: str = "w",
) -> None:
    """Configure the root logger for pyBurgers.

    This should be called once at application startup, typically in
    the main entry point (burgers.py).

    Args:
        level: Log level as string ("DEBUG", "INFO", etc.) or int.
        format_string: Optional custom format string. If None, uses
            DEBUG_FORMAT for DEBUG level, DEFAULT_FORMAT otherwise.
        log_file: Optional log file path for file logging.
        file_mode: File mode for log file handler (default: "w").
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Choose format based on level
    #if format_string is None:
    #    format_string = log_format

    # Configure root logger
    root_logger = logging.getLogger("PyBurgers")
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(log_format)
    handler.addFilter(_SkipProgressFilter())
    root_logger.addHandler(handler)

    # Create progress handler (same format, overwrites current line)
    progress_handler = _ProgressHandler(sys.stdout)
    progress_handler.setLevel(level)
    progress_handler.setFormatter(log_format)
    progress_handler.addFilter(_ProgressOnlyFilter())
    root_logger.addHandler(progress_handler)

    if log_file:
        log_path = Path(log_file).expanduser()
        if log_path.parent != Path("."):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        file_handler.addFilter(_SkipProgressFilter())
        root_logger.addHandler(file_handler)

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
    full_name = f"PyBurgers.{name}"

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
