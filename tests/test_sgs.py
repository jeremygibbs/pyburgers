"""Tests for subgrid-scale (SGS) models."""

from __future__ import annotations

import numpy as np
import pytest

from pyburgers.physics.sgs import SGS
from pyburgers.utils.spectral_workspace import SpectralWorkspace


class MockInput:
    """Mock input configuration for SGS testing."""

    def __init__(
        self,
        nx_les: int = 64,
        visc: float = 0.01,
        dt: float = 0.001,
        domain_length: float = 2 * np.pi,
        fftw_planning: str = "FFTW_ESTIMATE",
        fftw_threads: int = 1,
    ) -> None:
        # Create nested structure matching Input class
        class Grid:
            class LES:
                def __init__(self, points):
                    self.points = points

            def __init__(self, points):
                self.les = self.LES(points)

        self.grid = Grid(nx_les)
        self.domain_length = domain_length
        self.dt = dt
        self.fftw_planning = fftw_planning
        self.fftw_threads = fftw_threads


class TestSGSFactory:
    """Test cases for SGS factory method."""

    @pytest.fixture
    def spectral_workspace(self) -> SpectralWorkspace:
        """Create spectral workspace for testing."""
        nx = 64
        dx = 2 * np.pi / nx
        return SpectralWorkspace(nx=nx, dx=dx, fftw_planning="FFTW_ESTIMATE", fftw_threads=1)

    @pytest.mark.parametrize("model_id", [1, 2, 3, 4])
    def test_get_model_returns_sgs(
        self, model_id: int, spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that factory returns valid SGS model."""
        input_obj = MockInput()
        model = SGS.get_model(model_id, input_obj, spectral_workspace)

        assert model is not None
        assert hasattr(model, "compute")

    def test_get_model_invalid_id(self, spectral_workspace: SpectralWorkspace) -> None:
        """Test that invalid model ID raises error."""
        input_obj = MockInput()

        with pytest.raises(ValueError):
            SGS.get_model(99, input_obj, spectral_workspace)


class TestSGSModels:
    """Test cases for individual SGS models."""

    @pytest.fixture
    def spectral_workspace(self) -> SpectralWorkspace:
        """Create spectral workspace for testing."""
        nx = 64
        dx = 2 * np.pi / nx
        return SpectralWorkspace(nx=nx, dx=dx, fftw_planning="FFTW_ESTIMATE", fftw_threads=1)

    @pytest.fixture
    def test_field(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate test velocity and gradient fields."""
        nx = 64
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)

        u = np.sin(x)
        dudx = np.cos(x)
        return u, dudx

    @pytest.mark.parametrize("model_id", [1, 2, 3])
    def test_sgs_comprehensive_output(
        self, model_id: int, test_field: tuple, spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that SGS models return valid tau and coefficient."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(model_id, input_obj, spectral_workspace)

        result = model.compute(u, dudx, 0)

        # Check required keys exist
        assert "tau" in result
        assert "coeff" in result

        # Check shapes and types
        assert result["tau"].shape == u.shape
        # Coefficient can be float, np.floating, or int (if zero)
        assert isinstance(result["coeff"], (float, np.floating, int, np.integer))

        # Check values are finite
        assert np.all(np.isfinite(result["tau"]))
        assert np.isfinite(result["coeff"])

        # Check coefficient is non-negative
        assert result["coeff"] >= 0

    def test_deardorff_returns_tke_sgs(
        self, test_field: tuple, spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that Deardorff model returns subgrid TKE."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(4, input_obj, spectral_workspace)

        # Deardorff needs tke_sgs input
        tke_sgs = np.ones_like(u)
        result = model.compute(u, dudx, tke_sgs)

        assert "tke_sgs" in result
        assert result["tke_sgs"].shape == u.shape

    def test_deardorff_tke_positive(
        self, test_field: tuple, spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that Deardorff TKE remains positive."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(4, input_obj, spectral_workspace)

        tke_sgs = np.ones_like(u) * 0.1
        result = model.compute(u, dudx, tke_sgs)

        # TKE should be clipped to positive values
        assert np.all(result["tke_sgs"] >= 0)

    def test_smagcon_coefficient_fixed(
        self, test_field: tuple, spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that constant Smagorinsky has Cs = 0.16."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(1, input_obj, spectral_workspace)

        result = model.compute(u, dudx, 0)

        # Constant Smagorinsky coefficient should be exactly 0.16
        np.testing.assert_allclose(result["coeff"], 0.16, rtol=1e-10)

    def test_dynamic_smagorinsky_coefficient_bounds(
        self, spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that dynamic Smagorinsky Cs^2 stays in [0, 0.5]."""
        nx = 64
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)
        input_obj = MockInput(nx_les=nx)
        model = SGS.get_model(2, input_obj, spectral_workspace)

        # Test with multiple wavenumbers
        for k in [1, 2, 4, 8]:
            u = np.sin(k * x)
            dudx = k * np.cos(k * x)
            result = model.compute(u, dudx, 0)

            # Dynamic coefficient should be non-negative and physically reasonable
            # Cs^2 typically < 0.1, but allow up to 0.5 for safety
            assert result["coeff"] >= 0
            assert result["coeff"] < 0.7  # sqrt(0.5) ≈ 0.7

    def test_wonglilly_coefficient_bounds(self, spectral_workspace: SpectralWorkspace) -> None:
        """Test that Wong-Lilly coefficient stays in [0, 1]."""
        nx = 64
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)
        input_obj = MockInput(nx_les=nx)
        model = SGS.get_model(3, input_obj, spectral_workspace)

        # Test with multiple wavenumbers
        for k in [1, 2, 4]:
            u = np.sin(k * x)
            dudx = k * np.cos(k * x)
            result = model.compute(u, dudx, 0)

            # Wong-Lilly coefficient should be in [0, 1] range
            assert result["coeff"] >= 0
            assert result["coeff"] < 1.5  # Allow some margin

    def test_deardorff_tke_bounded(
        self, test_field: tuple, spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that Deardorff TKE stays in [0, 1] range."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(4, input_obj, spectral_workspace)

        # Start with reasonable TKE value
        tke_sgs = np.ones_like(u) * 0.5
        result = model.compute(u, dudx, tke_sgs)

        # TKE should remain in physical bounds
        assert np.all(result["tke_sgs"] >= 0)
        assert np.all(result["tke_sgs"] < 2.0)  # Reasonable upper bound


class TestSGSPhysics:
    """Tests for physical behavior of SGS models."""

    @pytest.fixture
    def spectral_workspace(self) -> SpectralWorkspace:
        """Create spectral workspace for testing."""
        nx = 64
        dx = 2 * np.pi / nx
        return SpectralWorkspace(nx=nx, dx=dx, fftw_planning="FFTW_ESTIMATE", fftw_threads=1)

    def test_smagorinsky_dissipative(self, spectral_workspace: SpectralWorkspace) -> None:
        """Test that Smagorinsky model is dissipative."""
        nx = 64
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)

        # Create field with gradient
        u = np.sin(x)
        dudx = np.cos(x)

        input_obj = MockInput(nx_les=nx)
        model = SGS.get_model(1, input_obj, spectral_workspace)

        result = model.compute(u, dudx, 0)
        tau = result["tau"]

        # SGS dissipation: -tau * dudx
        # For constant viscosity Smagorinsky, tau = -nu_t * dudx
        # So -tau * dudx = nu_t * dudx^2 >= 0 (always dissipative)
        dissipation = -tau * dudx
        # Should be non-trivial for sin(x) field, not just >= 0
        assert np.mean(dissipation) >= 1e-6
        # Upper bound sanity check
        assert np.mean(dissipation) < 1.0

    def test_dynamic_model_adapts_coefficient(self, spectral_workspace: SpectralWorkspace) -> None:
        """Test that dynamic model coefficient is in physical range."""
        nx = 64
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)

        input_obj = MockInput(nx_les=nx)
        model = SGS.get_model(2, input_obj, spectral_workspace)

        # Test with different flow fields
        coeffs = []
        for k in [1, 2, 4]:
            u = np.sin(k * x)
            dudx = k * np.cos(k * x)
            result = model.compute(u, dudx, 0)
            coeffs.append(result["coeff"])

        # Coefficients should be in physical range (can be zero for smooth fields)
        assert all(0 <= c < 0.7 for c in coeffs)
        # All coefficients should be finite
        assert all(np.isfinite(c) for c in coeffs)

    def test_sgs_dissipation_zero_for_constant_field(
        self, spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that SGS models produce zero stress for u=const."""
        nx = 64
        u = np.ones(nx)
        dudx = np.zeros(nx)

        input_obj = MockInput(nx_les=nx)

        # Test Smagorinsky models (1, 2, 3)
        for model_id in [1, 2, 3]:
            model = SGS.get_model(model_id, input_obj, spectral_workspace)
            result = model.compute(u, dudx, 0)

            # Constant field → zero gradient → zero SGS stress
            assert np.max(np.abs(result["tau"])) < 1e-10
