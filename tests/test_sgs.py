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
        nxLES: int = 64,
        visc: float = 0.01,
        dt: float = 0.001,
        domain_length: float = 2 * np.pi,
        fftw_planning: str = "FFTW_ESTIMATE",
        fftw_threads: int = 1,
    ) -> None:
        # Create nested structure matching Input class
        class Grid:
            class LES:
                def __init__(self, nx):
                    self.nx = nx

            def __init__(self, nx):
                self.les = self.LES(nx)

        self.grid = Grid(nxLES)
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
        return SpectralWorkspace(
            nx=nx,
            dx=dx,
            fftw_planning='FFTW_ESTIMATE',
            fftw_threads=1
        )

    @pytest.mark.parametrize("model_id", [1, 2, 3, 4])
    def test_get_model_returns_sgs(
        self,
        model_id: int,
        spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that factory returns valid SGS model."""
        input_obj = MockInput()
        model = SGS.get_model(model_id, input_obj, spectral_workspace)

        assert model is not None
        assert hasattr(model, "compute")

    def test_get_model_invalid_id(
        self,
        spectral_workspace: SpectralWorkspace
    ) -> None:
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
        return SpectralWorkspace(
            nx=nx,
            dx=dx,
            fftw_planning='FFTW_ESTIMATE',
            fftw_threads=1
        )

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
    def test_sgs_returns_tau(
        self,
        model_id: int,
        test_field: tuple,
        spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that SGS models return tau field."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(model_id, input_obj, spectral_workspace)

        result = model.compute(u, dudx, 0)

        assert "tau" in result
        assert result["tau"].shape == u.shape

    @pytest.mark.parametrize("model_id", [1, 2, 3])
    def test_sgs_returns_coeff(
        self,
        model_id: int,
        test_field: tuple,
        spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that SGS models return coefficient."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(model_id, input_obj, spectral_workspace)

        result = model.compute(u, dudx, 0)

        assert "coeff" in result
        assert np.isfinite(result["coeff"])

    @pytest.mark.parametrize("model_id", [1, 2, 3])
    def test_sgs_tau_finite(
        self,
        model_id: int,
        test_field: tuple,
        spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that SGS tau values are finite."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(model_id, input_obj, spectral_workspace)

        result = model.compute(u, dudx, 0)

        assert np.all(np.isfinite(result["tau"]))

    def test_deardorff_returns_tke_sgs(
        self,
        test_field: tuple,
        spectral_workspace: SpectralWorkspace
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
        self,
        test_field: tuple,
        spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that Deardorff TKE remains positive."""
        u, dudx = test_field
        input_obj = MockInput()
        model = SGS.get_model(4, input_obj, spectral_workspace)

        tke_sgs = np.ones_like(u) * 0.1
        result = model.compute(u, dudx, tke_sgs)

        # TKE should be clipped to positive values
        assert np.all(result["tke_sgs"] >= 0)


class TestSGSPhysics:
    """Tests for physical behavior of SGS models."""

    @pytest.fixture
    def spectral_workspace(self) -> SpectralWorkspace:
        """Create spectral workspace for testing."""
        nx = 64
        dx = 2 * np.pi / nx
        return SpectralWorkspace(
            nx=nx,
            dx=dx,
            fftw_planning='FFTW_ESTIMATE',
            fftw_threads=1
        )

    def test_smagorinsky_dissipative(
        self,
        spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that Smagorinsky model is dissipative."""
        nx = 64
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)

        # Create field with gradient
        u = np.sin(x)
        dudx = np.cos(x)

        input_obj = MockInput(nxLES=nx)
        model = SGS.get_model(1, input_obj, spectral_workspace)

        result = model.compute(u, dudx, 0)
        tau = result["tau"]

        # SGS dissipation: -tau * dudx
        # For constant viscosity Smagorinsky, tau = -nu_t * dudx
        # So -tau * dudx = nu_t * dudx^2 >= 0 (always dissipative)
        dissipation = -tau * dudx
        assert np.mean(dissipation) >= 0

    def test_dynamic_model_adapts_coefficient(
        self,
        spectral_workspace: SpectralWorkspace
    ) -> None:
        """Test that dynamic model coefficient varies with flow."""
        nx = 64
        dx = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, dx)

        input_obj = MockInput(nxLES=nx)
        model = SGS.get_model(2, input_obj, spectral_workspace)

        # Different flow fields should give different coefficients
        coeffs = []
        for k in [1, 2, 4]:
            u = np.sin(k * x)
            dudx = k * np.cos(k * x)
            result = model.compute(u, dudx, 0)
            coeffs.append(result["coeff"])

        # Coefficients should vary (not all identical)
        assert not (coeffs[0] == coeffs[1] == coeffs[2])
