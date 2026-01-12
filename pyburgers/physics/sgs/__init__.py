"""pyBurgers SGS Physics Subpackage.

This module serves as the entry point for the subgrid-scale (SGS) model
component of the LES solver. It imports and exposes the main `SGS`
base class and factory method, making it accessible to the rest of the model.
"""
from .sgs import SGS

# Export factory method as module-level function
get_model = SGS.get_model

__all__ = ['SGS', 'get_model']
