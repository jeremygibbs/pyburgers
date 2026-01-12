"""Subgrid-scale (SGS) model base class for pyBurgers LES.

This module provides the base SGS class and factory method for
creating different subgrid-scale models.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ...utils.io import Input


class SGS:
    """Base class for subgrid-scale models.

    Provides the interface and factory method for SGS models used in
    Large-Eddy Simulation. The base class returns zero SGS stress,
    equivalent to no subgrid model.

    Attributes:
        input: Input configuration object.
        dt: Time step size.
        nx: Number of LES grid points.
        dx: Grid spacing.
        sgs: Dictionary containing SGS stress (tau) and coefficient.
    """

    @staticmethod
    def get_model(model: int, input_obj: Input, derivs: Any = None) -> SGS:
        """Factory method to create the appropriate SGS model.

        Args:
            model: SGS model type identifier.
                0 = No model (base SGS)
                1 = Constant-coefficient Smagorinsky
                2 = Dynamic Smagorinsky
                3 = Dynamic Wong-Lilly
                4 = Deardorff 1.5-order TKE
            input_obj: Input configuration object.
            derivs: Optional Derivatives object to share (for Deardorff model).

        Returns:
            Instance of the requested SGS model subclass.
        """
        if model == 0:
            return SGS(input_obj)
        if model == 1:
            from .smagcon import SmagConstant
            return SmagConstant(input_obj)
        if model == 2:
            from .smagdyn import SmagDynamic
            return SmagDynamic(input_obj)
        if model == 3:
            from .wonglilly import WongLilly
            return WongLilly(input_obj)
        if model == 4:
            from .deardorff import Deardorff
            return Deardorff(input_obj, derivs)
        raise ValueError(f"Unknown SGS model ID: {model}. Valid options: 0-4.")

    def __init__(self, input_obj: Input) -> None:
        """Initialize the SGS model.

        Args:
            input_obj: Input configuration object containing simulation
                parameters.
        """
        self.input = input_obj
        self.dt = input_obj.dt
        self.nx = input_obj.models.les.nx
        self.dx = 2 * np.pi / self.nx
        self.fftw_planning = input_obj.fftw_planning
        self.fftw_threads = input_obj.fftw_threads

        # SGS terms dictionary
        self.sgs: dict[str, Any] = {
            'tau': np.zeros(self.nx),
            'coeff': 0
        }

    def compute(
        self,
        u: np.ndarray,
        dudx: np.ndarray,
        tke_sgs: np.ndarray | float
    ) -> dict[str, Any]:
        """Compute the SGS stress tensor.

        Args:
            u: Velocity field array.
            dudx: Velocity gradient (du/dx) array.
            tke_sgs: Subgrid TKE (used by Deardorff model).

        Returns:
            Dictionary containing:
                - 'tau': SGS stress array
                - 'coeff': Model coefficient
                - 'tke_sgs': Updated subgrid TKE (Deardorff only)
        """
        return self.sgs
