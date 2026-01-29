"""Integration tests for DNS and LES solvers."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest

from pyburgers import DNS, LES
from pyburgers.utils.io import Output


@pytest.fixture(autouse=True)
def isolate_tests():
    """Ensure test isolation by resetting random state and running garbage collection."""
    np.random.seed(42)  # Reset to known state before each test
    gc.collect()  # Clean up any lingering objects
    yield
    gc.collect()  # Clean up after test


class MockInput:
    """Mock input configuration for testing."""

    def __init__(
        self,
        cfl: float = 0.4,
        max_step: float = 0.001,
        duration: float = 0.01,
        visc: float = 0.01,
        namp: float = 0.1,
        nx_dns: int = 64,
        nx_les: int = 32,
        sgs_model: int = 1,
        t_save: float = 0.005,
        domain_length: float = 2 * np.pi,
    ) -> None:
        class Time:
            def __init__(self, duration, cfl, max_step):
                self.duration = duration
                self.cfl = cfl
                self.max_step = max_step

        class Noise:
            def __init__(self, amplitude):
                self.exponent = 0.75
                self.amplitude = amplitude

        class Physics:
            def __init__(self, viscosity, noise, subgrid_model):
                self.viscosity = viscosity
                self.noise = noise
                self.subgrid_model = subgrid_model

        class DNS:
            def __init__(self, points):
                self.points = points

        class LES:
            def __init__(self, points):
                self.points = points

        class Grid:
            def __init__(self, nx_dns, nx_les):
                self.dns = DNS(nx_dns)
                self.les = LES(nx_les)

        self.time = Time(duration, cfl, max_step)
        self.physics = Physics(visc, Noise(namp), subgrid_model=sgs_model)
        self.grid = Grid(nx_dns, nx_les)
        self.domain_length = domain_length
        self.fftw_planning = "FFTW_ESTIMATE"
        self.fftw_threads = 1
        self._t_save = t_save
        self._t_print = t_save

    @property
    def cfl_target(self) -> float:
        return self.time.cfl

    @property
    def max_step(self) -> float:
        return self.time.max_step

    @property
    def viscosity(self) -> float:
        return self.physics.viscosity

    @property
    def t_save(self) -> float:
        return self._t_save

    @property
    def t_print(self) -> float:
        return self._t_print


class TestDNSIntegration:
    """Integration tests for DNS solver."""

    def test_dns_runs_without_error(self, tmp_path: Path) -> None:
        """Test that DNS simulation runs without errors."""
        input_obj = MockInput(duration=0.01, t_save=0.005)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        assert output_file.exists()

    def test_dns_velocity_bounded(self, tmp_path: Path) -> None:
        """Test that DNS velocity remains bounded."""
        input_obj = MockInput(duration=0.02, t_save=0.01)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        # Velocity should remain finite and bounded
        assert np.all(np.isfinite(dns.u))
        u_rms = np.sqrt(np.mean(np.abs(dns.u) ** 2))
        assert 0.01 < u_rms < 2.0  # Physical bound for test params
        assert np.max(np.abs(dns.u)) < 5.0  # Peak velocity ~3-5 sigma

    def test_dns_tke_positive(self, tmp_path: Path) -> None:
        """Test that DNS TKE is in physical range."""
        input_obj = MockInput(duration=0.02, t_save=0.005)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        # TKE = variance of velocity, should be in physical range for test params
        assert 0.001 < dns.tke[0] < 1.0

    def test_dns_zero_mean_velocity(self, tmp_path: Path) -> None:
        """Test that DNS velocity has approximately zero mean."""
        input_obj = MockInput(duration=0.05, t_save=0.01, namp=0.01)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        # Mean should be close to zero (periodic domain + spectral methods)
        mean_u = np.mean(np.real(dns.u))
        assert abs(mean_u) < 1e-8

    def test_dns_nyquist_mode_zero(self, tmp_path: Path) -> None:
        """Test that Nyquist mode stays zero (dealiasing check)."""
        input_obj = MockInput(duration=0.02, t_save=0.01)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        # Nyquist mode should be zero to prevent aliasing
        u_fft = np.fft.rfft(dns.u)
        nyquist_idx = len(u_fft) - 1
        assert np.abs(u_fft[nyquist_idx]) < 1e-10


class TestLESIntegration:
    """Integration tests for LES solver."""

    @pytest.mark.parametrize("sgs_model", [1, 2, 3])
    def test_les_runs_all_sgs_models(self, tmp_path: Path, sgs_model: int) -> None:
        """Test that LES runs with all SGS model options."""
        input_obj = MockInput(duration=0.01, t_save=0.005, sgs_model=sgs_model)
        output_file = tmp_path / f"test_les_sgs{sgs_model}.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        assert output_file.exists()

    def test_les_velocity_bounded(self, tmp_path: Path) -> None:
        """Test that LES velocity remains bounded."""
        input_obj = MockInput(duration=0.02, t_save=0.01, sgs_model=1)
        output_file = tmp_path / "test_les.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        # LES velocity should be bounded similar to DNS
        assert np.all(np.isfinite(les.u))
        u_rms = np.sqrt(np.mean(np.abs(les.u) ** 2))
        assert 0.01 < u_rms < 2.0  # Physical bound for test params
        assert np.max(np.abs(les.u)) < 5.0  # Peak velocity ~3-5 sigma

    def test_les_diagnostics_computed(self, tmp_path: Path) -> None:
        """Test that LES computes all diagnostic fields."""
        input_obj = MockInput(duration=0.02, t_save=0.01, sgs_model=1)
        output_file = tmp_path / "test_les.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        # All diagnostics should be computed
        assert np.all(np.isfinite(les.tke))
        assert np.all(np.isfinite(les.diss_sgs))
        assert np.all(np.isfinite(les.diss_mol))

    def test_les_deardorff_model(self, tmp_path: Path) -> None:
        """Test that LES with Deardorff TKE model runs."""
        input_obj = MockInput(duration=0.01, t_save=0.005, sgs_model=4)
        output_file = tmp_path / "test_les_deardorff.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        # TKE_sgs should be an array for Deardorff model
        assert isinstance(les.tke_sgs, np.ndarray)
        assert np.all(np.isfinite(les.tke_sgs))

    def test_les_sgs_dissipation_positive(self, tmp_path: Path) -> None:
        """Test that LES SGS dissipation is non-negative."""
        input_obj = MockInput(duration=0.02, t_save=0.01, sgs_model=1)
        output_file = tmp_path / "test_les.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        # SGS models must be dissipative (Second Law of Thermodynamics)
        # Allow for tiny floating point errors (machine precision)
        assert np.all(les.diss_sgs >= -1e-15)

    def test_les_total_dissipation_bounds(self, tmp_path: Path) -> None:
        """Test that total dissipation matches energy input order of magnitude."""
        input_obj = MockInput(duration=0.05, t_save=0.01, sgs_model=1)
        output_file = tmp_path / "test_les.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        # Total dissipation = SGS + molecular
        total_diss = les.diss_sgs[-1] + les.diss_mol[-1]

        # Should be positive and in reasonable range
        assert total_diss > 0
        # Order of magnitude check: dissipation should be O(1e-5 to 1.0)
        assert 1e-6 < total_diss < 10.0

    @pytest.mark.parametrize("sgs_model", [1, 2, 3])
    def test_les_coefficient_physical_range(self, tmp_path: Path, sgs_model: int) -> None:
        """Test that SGS coefficients stay in physical bounds during run."""
        input_obj = MockInput(duration=0.02, t_save=0.005, sgs_model=sgs_model)
        output_file = tmp_path / f"test_les_coeff_{sgs_model}.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        # Check coefficient bounds based on model
        if sgs_model == 1:
            # Constant Smagorinsky: Cs = 0.16 (exact)
            np.testing.assert_allclose(les.C_sgs, 0.16, rtol=1e-10)
        elif sgs_model == 2:
            # Dynamic Smagorinsky: 0 ≤ Cs ≤ 0.5
            assert np.all(les.C_sgs >= 0)
            assert np.all(les.C_sgs <= 0.7)  # Allow some margin
        elif sgs_model == 3:
            # Wong-Lilly: 0 ≤ Cwl ≤ 1.0
            assert np.all(les.C_sgs >= 0)
            assert np.all(les.C_sgs <= 1.5)  # Allow some margin


class TestReproducibility:
    """Tests for simulation reproducibility."""

    def test_dns_reproducibility(self, tmp_path: Path) -> None:
        """Test that DNS produces identical results with same seed."""
        # Run 1
        np.random.seed(1)
        input_obj1 = MockInput(duration=0.02, t_save=0.01)
        output1 = tmp_path / "test_dns1.nc"
        dns1 = DNS(input_obj1, Output(str(output1)))
        dns1.run()
        u1 = dns1.u.copy()
        del dns1  # Ensure cleanup
        gc.collect()

        # Run 2
        np.random.seed(1)
        input_obj2 = MockInput(duration=0.02, t_save=0.01)
        output2 = tmp_path / "test_dns2.nc"
        dns2 = DNS(input_obj2, Output(str(output2)))
        dns2.run()
        u2 = dns2.u.copy()

        np.testing.assert_array_equal(u1, u2)

    def test_les_reproducibility(self, tmp_path: Path) -> None:
        """Test that LES produces identical results with same seed."""
        # Run 1
        np.random.seed(1)
        input_obj1 = MockInput(duration=0.02, t_save=0.01, sgs_model=1)
        output1 = tmp_path / "test_les1.nc"
        les1 = LES(input_obj1, Output(str(output1)))
        les1.run()
        u1 = les1.u.copy()
        del les1  # Ensure cleanup
        gc.collect()

        # Run 2
        np.random.seed(1)
        input_obj2 = MockInput(duration=0.02, t_save=0.01, sgs_model=1)
        output2 = tmp_path / "test_les2.nc"
        les2 = LES(input_obj2, Output(str(output2)))
        les2.run()
        u2 = les2.u.copy()

        np.testing.assert_array_equal(u1, u2)
