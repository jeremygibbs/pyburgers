"""Pytest fixtures for PyBurgers tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def grid_small() -> dict:
    """Small grid for fast unit tests."""
    nx = 64
    dx = 2 * np.pi / nx
    x = np.arange(0, 2 * np.pi, dx)
    return {"nx": nx, "dx": dx, "x": x}


@pytest.fixture
def grid_medium() -> dict:
    """Medium grid for accuracy tests."""
    nx = 256
    dx = 2 * np.pi / nx
    x = np.arange(0, 2 * np.pi, dx)
    return {"nx": nx, "dx": dx, "x": x}


@pytest.fixture
def sine_field(grid_small: dict) -> np.ndarray:
    """Sine wave test field."""
    return np.sin(grid_small["x"])


@pytest.fixture
def cosine_field(grid_small: dict) -> np.ndarray:
    """Cosine wave test field."""
    return np.cos(grid_small["x"])
