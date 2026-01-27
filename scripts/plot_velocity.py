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
"""Plot velocity field in x-t plane from PyBurgers NetCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


def _read_velocity(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read velocity field from NetCDF file.

    Args:
        path: Path to NetCDF file.

    Returns:
        Tuple of (time, x, u) arrays where u has shape (nt, nx).

    Raises:
        KeyError: If required variables are missing.
    """
    with nc.Dataset(path, "r") as ds:
        if "u" not in ds.variables:
            raise KeyError(f"Missing 'u' in {path}")
        if "x" not in ds.variables:
            raise KeyError(f"Missing 'x' in {path}")

        u = np.asarray(ds.variables["u"][:])
        x = np.asarray(ds.variables["x"][:])

        if "time" in ds.variables:
            time = np.asarray(ds.variables["time"][:])
        else:
            time = np.arange(u.shape[0], dtype=float)

    return time, x, u


def main() -> int:

    # Get command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot velocity field(s) in x-t plane as space-time diagram."
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
    parser.add_argument(
        "--cmap",
        type=str,
        default="RdBu_r",
        help="Colormap for velocity field (default: RdBu_r).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for colorbar (default: auto from data).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for colorbar (default: auto from data).",
    )
    args = parser.parse_args()
    files = args.files
    n_files = len(files)

    # Make figure

    # Determine subplot layout: use rows for readability
    if n_files == 1:
        nrows, ncols = 1, 1
        figsize = (10, 4)
    elif n_files == 2:
        nrows, ncols = 2, 1
        figsize = (10, 7)
    elif n_files <= 4:
        nrows, ncols = 2, 2
        figsize = (12, 7)
    else:
        # More than 4: use 2 columns
        nrows = (n_files + 1) // 2
        ncols = 2
        figsize = (12, 3 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, sharex=True, sharey=True
    )
    axes = axes.flatten()

    # Determine global vmin/vmax if not specified
    if args.vmin is None or args.vmax is None:
        all_u = []
        for file_path in files:
            _, _, u = _read_velocity(file_path)
            all_u.append(u)
        all_u_concat = np.concatenate([u.ravel() for u in all_u])
        vmin = args.vmin if args.vmin is not None else np.min(all_u_concat)
        vmax = args.vmax if args.vmax is not None else np.max(all_u_concat)
    else:
        vmin = args.vmin
        vmax = args.vmax

    # Plot each file
    for idx, file_path in enumerate(files):
        ax = axes[idx]
        time, x, u = _read_velocity(file_path)

        # Create meshgrid for pcolormesh
        X, T = np.meshgrid(x, time)

        # Plot velocity field
        pcm = ax.pcolormesh(
            X, T, u, cmap=args.cmap, vmin=vmin, vmax=vmax, shading="auto"
        )

        # Set labels and title
        title = file_path.stem

        ax.set_title(title)
        ax.set_ylabel("time (s)")

        # Add colorbar
        fig.colorbar(pcm, ax=ax, label="u (m/s)")

    # Configure plot
    # Set xlabel on bottom plots only
    for idx in range(n_files):
        if idx >= (nrows - 1) * ncols:
            axes[idx].set_xlabel("x (m)")

    # Hide unused subplots
    for idx in range(n_files, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()

    # Save or display figure
    if args.out:
        fig.savefig(args.out, dpi=150)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
