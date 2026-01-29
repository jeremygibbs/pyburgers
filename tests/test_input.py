"""Tests for Input class validation."""

import json
from pathlib import Path

import pytest

from pyburgers.exceptions import NamelistError
from pyburgers.utils.io import Input


def create_namelist(tmp_path: Path, data: dict) -> Path:
    """Helper to create a namelist JSON file."""
    namelist_file = tmp_path / "namelist.json"
    namelist_file.write_text(json.dumps(data))
    return namelist_file


def get_valid_namelist() -> dict:
    """Return a minimal valid namelist configuration."""
    return {
        "time": {"duration": 0.01, "cfl": 0.4, "max_step": 0.001},
        "grid": {"dns": {"points": 64}, "les": {"points": 32}},
        "physics": {"viscosity": 0.01},
    }


class TestMissingSections:
    """Tests for missing required sections."""

    def test_missing_time_section_raises_error(self, tmp_path: Path) -> None:
        """Test that missing 'time' section raises NamelistError."""
        data = get_valid_namelist()
        del data["time"]
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="Missing required section: 'time'"):
            Input(str(namelist_file))

    def test_missing_physics_section_raises_error(self, tmp_path: Path) -> None:
        """Test that missing 'physics' section raises NamelistError."""
        data = get_valid_namelist()
        del data["physics"]
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="Missing required section: 'physics'"):
            Input(str(namelist_file))

    def test_missing_grid_section_raises_error(self, tmp_path: Path) -> None:
        """Test that missing 'grid' section raises NamelistError."""
        data = get_valid_namelist()
        del data["grid"]
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="Missing required section: 'grid'"):
            Input(str(namelist_file))


class TestMissingRequiredFields:
    """Tests for missing required fields within sections."""

    def test_missing_duration_raises_error(self, tmp_path: Path) -> None:
        """Test that missing 'duration' in time section raises NamelistError."""
        data = get_valid_namelist()
        del data["time"]["duration"]
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="Missing 'duration' in time section"):
            Input(str(namelist_file))

    def test_missing_cfl_raises_error(self, tmp_path: Path) -> None:
        """Test that missing 'cfl' in time section raises NamelistError."""
        data = get_valid_namelist()
        del data["time"]["cfl"]
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="Missing 'cfl' in time section"):
            Input(str(namelist_file))

    def test_missing_max_step_raises_error(self, tmp_path: Path) -> None:
        """Test that missing 'max_step' in time section raises NamelistError."""
        data = get_valid_namelist()
        del data["time"]["max_step"]
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="Missing 'max_step' in time section"):
            Input(str(namelist_file))

    def test_missing_viscosity_raises_error(self, tmp_path: Path) -> None:
        """Test that missing 'viscosity' in physics section raises NamelistError."""
        data = get_valid_namelist()
        del data["physics"]["viscosity"]
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="Missing 'viscosity' in physics section"):
            Input(str(namelist_file))


class TestInvalidTimeValues:
    """Tests for invalid time configuration values."""

    def test_cfl_zero_raises_error(self, tmp_path: Path) -> None:
        """Test that CFL = 0 raises NamelistError."""
        data = get_valid_namelist()
        data["time"]["cfl"] = 0.0
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="time 'cfl' must be in \\(0, 0.55\\)"):
            Input(str(namelist_file))

    def test_cfl_negative_raises_error(self, tmp_path: Path) -> None:
        """Test that negative CFL raises NamelistError."""
        data = get_valid_namelist()
        data["time"]["cfl"] = -0.1
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="time 'cfl' must be in \\(0, 0.55\\)"):
            Input(str(namelist_file))

    def test_cfl_above_limit_raises_error(self, tmp_path: Path) -> None:
        """Test that CFL >= 0.55 raises NamelistError."""
        data = get_valid_namelist()
        data["time"]["cfl"] = 0.55
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="time 'cfl' must be in \\(0, 0.55\\)"):
            Input(str(namelist_file))

    def test_duration_negative_raises_error(self, tmp_path: Path) -> None:
        """Test that negative duration raises NamelistError."""
        data = get_valid_namelist()
        data["time"]["duration"] = -1.0
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="time 'duration' must be positive"):
            Input(str(namelist_file))

    def test_duration_zero_raises_error(self, tmp_path: Path) -> None:
        """Test that zero duration raises NamelistError."""
        data = get_valid_namelist()
        data["time"]["duration"] = 0.0
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="time 'duration' must be positive"):
            Input(str(namelist_file))

    def test_max_step_negative_raises_error(self, tmp_path: Path) -> None:
        """Test that negative max_step raises NamelistError."""
        data = get_valid_namelist()
        data["time"]["max_step"] = -0.001
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="time 'max_step' must be positive"):
            Input(str(namelist_file))

    def test_max_step_zero_raises_error(self, tmp_path: Path) -> None:
        """Test that zero max_step raises NamelistError."""
        data = get_valid_namelist()
        data["time"]["max_step"] = 0.0
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="time 'max_step' must be positive"):
            Input(str(namelist_file))


class TestInvalidPhysicsValues:
    """Tests for invalid physics configuration values."""

    def test_viscosity_negative_raises_error(self, tmp_path: Path) -> None:
        """Test that negative viscosity raises NamelistError."""
        data = get_valid_namelist()
        data["physics"]["viscosity"] = -0.01
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="'viscosity' must be positive"):
            Input(str(namelist_file))

    def test_viscosity_zero_raises_error(self, tmp_path: Path) -> None:
        """Test that zero viscosity raises NamelistError."""
        data = get_valid_namelist()
        data["physics"]["viscosity"] = 0.0
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="'viscosity' must be positive"):
            Input(str(namelist_file))

    def test_subgrid_model_negative_raises_error(self, tmp_path: Path) -> None:
        """Test that subgrid_model < 0 raises NamelistError."""
        data = get_valid_namelist()
        data["physics"]["subgrid_model"] = -1
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="physics 'subgrid_model' must be 0-4"):
            Input(str(namelist_file))

    def test_subgrid_model_above_max_raises_error(self, tmp_path: Path) -> None:
        """Test that subgrid_model > 4 raises NamelistError."""
        data = get_valid_namelist()
        data["physics"]["subgrid_model"] = 5
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="physics 'subgrid_model' must be 0-4, got 5"):
            Input(str(namelist_file))


class TestInvalidFFTWConfig:
    """Tests for invalid FFTW configuration values."""

    def test_invalid_fftw_planning_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid FFTW planning strategy raises NamelistError."""
        data = get_valid_namelist()
        data["fftw"] = {"planning": "FFTW_INVALID"}
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="Invalid FFTW planning: 'FFTW_INVALID'"):
            Input(str(namelist_file))

    def test_fftw_threads_zero_raises_error(self, tmp_path: Path) -> None:
        """Test that FFTW threads < 1 raises NamelistError."""
        data = get_valid_namelist()
        data["fftw"] = {"threads": 0}
        namelist_file = create_namelist(tmp_path, data)

        with pytest.raises(NamelistError, match="FFTW 'threads' must be at least 1"):
            Input(str(namelist_file))


class TestValidConfigurations:
    """Tests for valid namelist configurations."""

    def test_valid_minimal_namelist_loads(self, tmp_path: Path) -> None:
        """Test that a minimal valid namelist loads successfully."""
        data = get_valid_namelist()
        namelist_file = create_namelist(tmp_path, data)

        input_obj = Input(str(namelist_file))

        assert input_obj.time.duration == 0.01
        assert input_obj.time.cfl == 0.4
        assert input_obj.time.max_step == 0.001
        assert input_obj.physics.viscosity == 0.01
        assert input_obj.grid.dns.points == 64
        assert input_obj.grid.les.points == 32

    def test_valid_full_namelist_loads(self, tmp_path: Path) -> None:
        """Test that a full valid namelist loads successfully."""
        data = {
            "time": {"duration": 1.0, "cfl": 0.4, "max_step": 0.001},
            "grid": {
                "length": 6.283185307179586,
                "dns": {"points": 8192},
                "les": {"points": 512},
            },
            "physics": {
                "viscosity": 0.001,
                "subgrid_model": 2,
                "noise": {"exponent": -0.75, "amplitude": 1e-6},
            },
            "output": {"interval_save": 0.1, "interval_print": 0.01},
            "logging": {"level": "DEBUG"},
            "fftw": {"planning": "FFTW_MEASURE", "threads": 4},
        }
        namelist_file = create_namelist(tmp_path, data)

        input_obj = Input(str(namelist_file))

        assert input_obj.time.duration == 1.0
        assert input_obj.time.cfl == 0.4
        assert input_obj.time.max_step == 0.001
        assert input_obj.grid.length == 6.283185307179586
        assert input_obj.grid.dns.points == 8192
        assert input_obj.grid.les.points == 512
        assert input_obj.physics.viscosity == 0.001
        assert input_obj.physics.subgrid_model == 2
        assert input_obj.physics.noise.exponent == -0.75
        assert input_obj.physics.noise.amplitude == 1e-6
        assert input_obj.output.interval_save == 0.1
        assert input_obj.output.interval_print == 0.01
        assert input_obj.logging.level == "DEBUG"
        assert input_obj.fftw.planning == "FFTW_MEASURE"
        assert input_obj.fftw.threads == 4

    def test_all_valid_subgrid_models(self, tmp_path: Path) -> None:
        """Test that all valid subgrid model values (0-4) are accepted."""
        for model_id in range(5):
            data = get_valid_namelist()
            data["physics"]["subgrid_model"] = model_id
            namelist_file = create_namelist(tmp_path, data)

            input_obj = Input(str(namelist_file))
            assert input_obj.physics.subgrid_model == model_id

    def test_all_valid_fftw_planning_strategies(self, tmp_path: Path) -> None:
        """Test that all valid FFTW planning strategies are accepted."""
        valid_strategies = ["FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE"]
        for strategy in valid_strategies:
            data = get_valid_namelist()
            data["fftw"] = {"planning": strategy}
            namelist_file = create_namelist(tmp_path, data)

            input_obj = Input(str(namelist_file))
            assert input_obj.fftw.planning == strategy


class TestFileErrors:
    """Tests for file-related errors."""

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """Test that a non-existent file raises FileNotFoundError."""
        non_existent = tmp_path / "does_not_exist.json"

        with pytest.raises(FileNotFoundError):
            Input(str(non_existent))

    def test_invalid_json_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid JSON raises JSONDecodeError."""
        namelist_file = tmp_path / "invalid.json"
        namelist_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            Input(str(namelist_file))
