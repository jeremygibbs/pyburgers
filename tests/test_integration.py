"""Integration tests for DNS and LES solvers."""
from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from models import DNS, LES
from utils.io import Input, Output


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
        dt: float = 0.001,
        nt: int = 10,
        visc: float = 0.01,
        namp: float = 0.1,
        nxDNS: int = 64,
        nxLES: int = 32,
        sgs: int = 1,
        t_save: int = 5,
    ) -> None:
        self.dt = dt
        self.nt = nt
        self.visc = visc
        self.namp = namp
        self.nxDNS = nxDNS
        self.nxLES = nxLES
        self.sgs = sgs
        self.t_save = t_save


class TestDNSIntegration:
    """Integration tests for DNS solver."""

    def test_dns_runs_without_error(self, tmp_path: Path) -> None:
        """Test that DNS simulation runs without errors."""
        input_obj = MockInput(nt=10, t_save=5)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        assert output_file.exists()

    def test_dns_velocity_bounded(self, tmp_path: Path) -> None:
        """Test that DNS velocity remains bounded."""
        input_obj = MockInput(nt=20, t_save=10)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        # Velocity should remain finite and bounded
        assert np.all(np.isfinite(dns.u))
        assert np.max(np.abs(dns.u)) < 100

    def test_dns_tke_positive(self, tmp_path: Path) -> None:
        """Test that DNS TKE is non-negative."""
        input_obj = MockInput(nt=20, t_save=5)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        assert dns.tke[0] >= 0

    def test_dns_zero_mean_velocity(self, tmp_path: Path) -> None:
        """Test that DNS velocity has approximately zero mean."""
        input_obj = MockInput(nt=50, t_save=10, namp=0.01)
        output_file = tmp_path / "test_dns.nc"
        output_obj = Output(str(output_file))

        dns = DNS(input_obj, output_obj)
        dns.run()

        # Mean should be close to zero (periodic domain)
        mean_u = np.mean(np.real(dns.u))
        assert abs(mean_u) < 1.0


class TestLESIntegration:
    """Integration tests for LES solver."""

    @pytest.mark.parametrize("sgs_model", [1, 2, 3])
    def test_les_runs_all_sgs_models(self, tmp_path: Path, sgs_model: int) -> None:
        """Test that LES runs with all SGS model options."""
        input_obj = MockInput(nt=10, t_save=5, sgs=sgs_model)
        output_file = tmp_path / f"test_les_sgs{sgs_model}.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        assert output_file.exists()

    def test_les_velocity_bounded(self, tmp_path: Path) -> None:
        """Test that LES velocity remains bounded."""
        input_obj = MockInput(nt=20, t_save=10, sgs=1)
        output_file = tmp_path / "test_les.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        assert np.all(np.isfinite(les.u))
        assert np.max(np.abs(les.u)) < 100

    def test_les_diagnostics_computed(self, tmp_path: Path) -> None:
        """Test that LES computes all diagnostic fields."""
        input_obj = MockInput(nt=20, t_save=10, sgs=1)
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
        input_obj = MockInput(nt=10, t_save=5, sgs=4)
        output_file = tmp_path / "test_les_deardorff.nc"
        output_obj = Output(str(output_file))

        les = LES(input_obj, output_obj)
        les.run()

        # TKE_sgs should be an array for Deardorff model
        assert isinstance(les.tke_sgs, np.ndarray)
        assert np.all(np.isfinite(les.tke_sgs))


class TestReproducibility:
    """Tests for simulation reproducibility."""

    def test_dns_reproducibility(self, tmp_path: Path) -> None:
        """Test that DNS produces identical results with same seed."""
        # Run 1
        np.random.seed(1)
        input_obj1 = MockInput(nt=20, t_save=10)
        output1 = tmp_path / "test_dns1.nc"
        dns1 = DNS(input_obj1, Output(str(output1)))
        dns1.run()
        u1 = dns1.u.copy()
        del dns1  # Ensure cleanup
        gc.collect()

        # Run 2
        np.random.seed(1)
        input_obj2 = MockInput(nt=20, t_save=10)
        output2 = tmp_path / "test_dns2.nc"
        dns2 = DNS(input_obj2, Output(str(output2)))
        dns2.run()
        u2 = dns2.u.copy()

        np.testing.assert_array_equal(u1, u2)

    def test_les_reproducibility(self, tmp_path: Path) -> None:
        """Test that LES produces identical results with same seed."""
        # Run 1
        np.random.seed(1)
        input_obj1 = MockInput(nt=20, t_save=10, sgs=1)
        output1 = tmp_path / "test_les1.nc"
        les1 = LES(input_obj1, Output(str(output1)))
        les1.run()
        u1 = les1.u.copy()
        del les1  # Ensure cleanup
        gc.collect()

        # Run 2
        np.random.seed(1)
        input_obj2 = MockInput(nt=20, t_save=10, sgs=1)
        output2 = tmp_path / "test_les2.nc"
        les2 = LES(input_obj2, Output(str(output2)))
        les2.run()
        u2 = les2.u.copy()

        np.testing.assert_array_equal(u1, u2)
