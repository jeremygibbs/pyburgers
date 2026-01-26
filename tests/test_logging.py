"""Tests for logging functionality."""

import logging
from pathlib import Path

from pyburgers.utils import get_logger, setup_logging
from pyburgers.utils.io import Input


class TestLoggingHelper:
    """Test cases for logging_helper module."""

    def test_setup_logging_default(self) -> None:
        """Test default logging setup."""
        setup_logging()
        logger = get_logger("Test")
        assert logger.level == logging.NOTSET  # Inherits from parent
        assert logging.getLogger("PyBurgers").level == logging.INFO

    def test_setup_logging_custom_level(self) -> None:
        """Test logging setup with custom level."""
        setup_logging(level="DEBUG")
        assert logging.getLogger("PyBurgers").level == logging.DEBUG

    def test_setup_logging_string_level(self) -> None:
        """Test that string log levels work."""
        for level_str in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            setup_logging(level=level_str)
            expected_level = getattr(logging, level_str)
            assert logging.getLogger("PyBurgers").level == expected_level

    def test_setup_logging_int_level(self) -> None:
        """Test that integer log levels work."""
        setup_logging(level=logging.WARNING)
        assert logging.getLogger("PyBurgers").level == logging.WARNING

    def test_setup_logging_file_handler(self, tmp_path: Path) -> None:
        """Test that file logging writes to the specified log file."""
        log_file = tmp_path / "pyburgers.log"
        setup_logging(level="INFO", log_file=str(log_file))
        logger = get_logger("Test")

        logger.info("File log message")
        for handler in logging.getLogger("PyBurgers").handlers:
            handler.flush()

        assert log_file.exists()
        assert "File log message" in log_file.read_text()

    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("Test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "PyBurgers.Test"

    def test_get_logger_caching(self) -> None:
        """Test that get_logger caches loggers."""
        logger1 = get_logger("Test")
        logger2 = get_logger("Test")
        assert logger1 is logger2

    def test_logger_hierarchy(self) -> None:
        """Test that loggers follow hierarchy."""
        setup_logging(level="INFO")
        parent_logger = logging.getLogger("PyBurgers")
        child_logger = get_logger("DNS")

        assert child_logger.name.startswith(parent_logger.name)


class TestInputLogging:
    """Test cases for Input class logging."""

    def test_input_reads_log_level_from_namelist(self, tmp_path: Path) -> None:
        """Test that Input reads log level from namelist."""
        namelist_file = tmp_path / "test_namelist.json"
        namelist_content = """
        {
            "time": {"nt": 10, "dt": 0.001},
            "grid": {"domain_length": 6.283185307179586, "dns": {"nx": 64}, "les": {"nx": 32}},
            "physics": {
                "noise": {"alpha": 0.75, "amplitude": 0.1},
                "viscosity": 0.01,
                "sgs_model": 1
            },
            "output": {"t_save": 0.005},
            "logging": {"level": "DEBUG"},
            "fftw": {"planning": "FFTW_ESTIMATE", "threads": 1}
        }
        """
        namelist_file.write_text(namelist_content)

        input_obj = Input(str(namelist_file))
        assert input_obj.log_level == "DEBUG"

    def test_input_defaults_to_info(self, tmp_path: Path) -> None:
        """Test that Input defaults to INFO level."""
        namelist_file = tmp_path / "test_namelist.json"
        namelist_content = """
        {
            "time": {"nt": 10, "dt": 0.001},
            "grid": {"domain_length": 6.283185307179586, "dns": {"nx": 64}, "les": {"nx": 32}},
            "physics": {
                "noise": {"alpha": 0.75, "amplitude": 0.1},
                "viscosity": 0.01,
                "sgs_model": 1
            },
            "output": {"t_save": 0.005},
            "logging": {"level": "INFO"},
            "fftw": {"planning": "FFTW_ESTIMATE", "threads": 1}
        }
        """
        namelist_file.write_text(namelist_content)

        input_obj = Input(str(namelist_file))
        assert input_obj.log_level == "INFO"

    def test_input_case_insensitive_log_level(self, tmp_path: Path) -> None:
        """Test that Input handles mixed case log levels."""
        namelist_file = tmp_path / "test_namelist.json"
        namelist_content = """
        {
            "time": {"nt": 10, "dt": 0.001},
            "grid": {"domain_length": 6.283185307179586, "dns": {"nx": 64}, "les": {"nx": 32}},
            "physics": {
                "noise": {"alpha": 0.75, "amplitude": 0.1},
                "viscosity": 0.01,
                "sgs_model": 1
            },
            "output": {"t_save": 0.005},
            "logging": {"level": "debug"},
            "fftw": {"planning": "FFTW_ESTIMATE", "threads": 1}
        }
        """
        namelist_file.write_text(namelist_content)

        input_obj = Input(str(namelist_file))
        # Level is stored as-is, setup_logging will handle normalization
        assert input_obj.log_level == "debug"


class TestLoggingLevels:
    """Test cases for different logging levels."""

    def test_info_level_shows_info(self, capsys) -> None:
        """Test that INFO level shows info messages."""
        setup_logging(level="INFO")
        logger = get_logger("Test")

        logger.info("Test info message")
        logger.debug("Test debug message")

        captured = capsys.readouterr()
        assert "Test info message" in captured.out
        assert "Test debug message" not in captured.out

    def test_debug_level_shows_all(self, capsys) -> None:
        """Test that DEBUG level shows all messages."""
        setup_logging(level="DEBUG")
        logger = get_logger("Test")

        logger.debug("Test debug message")
        logger.info("Test info message")

        captured = capsys.readouterr()
        assert "Test debug message" in captured.out
        assert "Test info message" in captured.out

    def test_warning_level_hides_info(self, capsys) -> None:
        """Test that WARNING level hides info messages."""
        setup_logging(level="WARNING")
        logger = get_logger("Test")

        logger.info("Test info message")
        logger.warning("Test warning message")

        captured = capsys.readouterr()
        assert "Test info message" not in captured.out
        assert "Test warning message" in captured.out
