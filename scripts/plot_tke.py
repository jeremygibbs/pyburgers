#!/usr/bin/env python
#
# PyBurgers
#
# Copyright (c) 2017–2026 Jeremy A. Gibbs
#
# This file is part of PyBurgers.
#
# This software is free and is distributed under the WTFPL license.
# See accompanying LICENSE file or visit https://www.wtfpl.net.
#
"""Plot DNS and LES TKE output from PyBurgers NetCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


def _read_tke(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read TKE field from NetCDF file.
    
    Args:
        path: Path to NetCDF file.
    
    Returns:
        Tuple of (time, tke) arrays where tke has shape (nt).
    
    Raises:
        KeyError: If required variables are missing.
    """
    with nc.Dataset(path, "r") as ds:
        if "tke" not in ds.variables:
            raise KeyError(f"Missing 'tke' in {path}")
        tke = np.asarray(ds.variables["tke"][:])
        if "time" in ds.variables:
            time = np.asarray(ds.variables["time"][:])
        else:
            time = np.arange(tke.shape[0], dtype=float)
    return time, tke

def main() -> int:

    # Get command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot TKE time series from DNS and LES NetCDF outputs."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Path(s) to NetCDF output file(s).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Optional output image path (PNG/SVG/etc). If omitted, shows plot.",
    )
    args = parser.parse_args()
    files = args.files

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Loop through each file
    for file_path in files:
        time, tke = _read_tke(file_path)
        ax.plot(time, tke, label=file_path.stem, linewidth=2.0)

    # Configure plot
    ax.set_title("PyBurgers: Turbulence Kinetic Energy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("TKE (m^2 s^-2)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Save or display figure
    if args.out:
        fig.savefig(args.out, dpi=150)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
