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
"""Plot velocity power spectral density from PyBurgers NetCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


def _read_velocity(
    path: Path, t_start: float | None = None, t_end: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read velocity field from NetCDF file.

    Args:
        path: Path to NetCDF file.
        t_start: Optional start time for averaging window.
        t_end: Optional end time for averaging window.

    Returns:
        Tuple of (x, u, t) arrays where u has shape (nt, nx) and t has shape (nt,).

    Raises:
        KeyError: If required variables are missing.
    """
    with nc.Dataset(path, "r") as ds:
        if "u" not in ds.variables:
            raise KeyError(f"Missing 'u' in {path}")
        if "x" not in ds.variables:
            raise KeyError(f"Missing 'x' in {path}")
        if "time" not in ds.variables:
            raise KeyError(f"Missing 'time' in {path}")

        u = np.asarray(ds.variables["u"][:])
        x = np.asarray(ds.variables["x"][:])
        t = np.asarray(ds.variables["time"][:])

    # Filter time window if specified
    if t_start is not None or t_end is not None:
        t_start = t_start if t_start is not None else t[0]
        t_end = t_end if t_end is not None else t[-1]
        mask = (t >= t_start) & (t <= t_end)
        u = u[mask, :]
        t = t[mask]
        if len(t) == 0:
            raise ValueError(f"No data in time window [{t_start}, {t_end}]")

    return x, u, t


def _compute_psd(u: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-averaged one-sided autospectral density.

    Args:
        u: Velocity field with shape (nt, nx).
        dx: Grid spacing.

    Returns:
        Tuple of (wavenumbers, PSD) arrays.
    """
    nt, nx = u.shape
    nk = nx // 2 + 1

    # Compute wavenumber array (non-negative frequencies from rfft)
    k = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi

    # Compute one-sided autospectral density for each time step
    psd_time = np.zeros((nt, nk))
    for t in range(nt):
        u_t = u[t, :] - np.mean(u[t, :])
        fu = np.fft.rfft(u_t)
        # Autospectral density: |F(k)|^2 / N^2 (one-sided correction below)
        psd = np.abs(fu) ** 2 / (nx**2)
        # One-sided correction: double positive frequencies (exclude DC and Nyquist)
        if nx % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0
        psd_time[t, :] = psd

    # Time average
    psd = np.mean(psd_time, axis=0)

    # Exclude zero wavenumber
    return k[1:], psd[1:]


def main() -> int:

    # Get command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot velocity power spectral density from NetCDF outputs."
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
        "--threshold",
        type=float,
        default=1e-10,
        help="Clip PSD values below this threshold (default: 1e-10).",
    )
    parser.add_argument(
        "--check-variance",
        action="store_true",
        help="Print variance vs. summed PSD check for each file.",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=None,
        help="Start time for averaging window (default: use all times).",
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=None,
        help="End time for averaging window (default: use all times).",
    )
    args = parser.parse_args()
    files = args.files

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 6))
    psd_min = None
    psd_max = None

    # Plot each file
    for idx, file_path in enumerate(files):
        x, u, t = _read_velocity(file_path, t_start=args.t1, t_end=args.t2)
        dx = x[1] - x[0]

        # Print time window info
        print(
            f"{file_path.name}: Averaging over t=[{t[0]:.2f}, {t[-1]:.2f}] "
            f"({len(t)} time steps)"
        )

        # Compute PSD
        k, psd = _compute_psd(u, dx)

        # Plot PSD
        ax.loglog(k, psd, label=file_path.stem, linewidth=1.5)

        if idx == 0:
            # Add theoretical -5/3 slope across all wavenumbers
            # Fit scaling constant to inertial subrange (middle decade in log-space)
            # Exclude first and last decades to avoid energy-containing and dissipation ranges
            k_min = k[0]
            k_max = k[-1]
            k_fit_min = k_min * 10
            k_fit_max = k_max / 10

            # Select wavenumbers in fitting range
            fit_mask = (k >= k_fit_min) & (k <= k_fit_max)
            k_fit = k[fit_mask]
            psd_fit = psd[fit_mask]

            # Fit C * k^(-5/3) to data in log space
            # log(PSD) = log(C) - (5/3)*log(k)
            # Solve for C: C = exp(mean(log(PSD) + (5/3)*log(k)))
            if len(k_fit) > 0:
                log_C = np.mean(np.log(psd_fit) + (5 / 3) * np.log(k_fit))
                C = np.exp(log_C)
            else:
                # Fallback to middle point if fit range is empty
                k_mid = k[len(k) // 2]
                psd_mid = psd[len(k) // 2]
                C = psd_mid * k_mid ** (5 / 3)

            psd_theory = C * k ** (-5 / 3)
            ax.loglog(k, psd_theory, "k--", linewidth=1.5, alpha=0.7, label=r"$k^{-5/3}$")

        # Set y-axis limits to exclude near-zero high-frequency noise
        # Find minimum non-zero PSD value above threshold
        psd_threshold = args.threshold
        valid_psd = psd[psd > psd_threshold]
        if len(valid_psd) > 0:
            ymin = valid_psd.min() * 0.5
            ymax = psd.max() * 2.0
            psd_min = ymin if psd_min is None else min(psd_min, ymin)
            psd_max = ymax if psd_max is None else max(psd_max, ymax)

        if args.check_variance:
            u_fluct = u - np.mean(u, axis=1, keepdims=True)
            var_time = np.mean(u_fluct**2, axis=1)
            variance = np.mean(var_time)
            psd_sum = np.sum(psd)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = psd_sum / variance if variance > 0 else np.nan
            print(
                f"{file_path.name}: variance={variance:.6e}, "
                f"sum(PSD)={psd_sum:.6e}, ratio={ratio:.6f}"
            )

    if psd_min is not None and psd_max is not None:
        ax.set_ylim(psd_min, psd_max)

    # Set labels and title
    title = "PyBurgers: Velocity Power Spectral Density"
    if args.t1 is not None or args.t2 is not None:
        t_start_str = f"{args.t1:.1f}" if args.t1 is not None else "start"
        t_end_str = f"{args.t2:.1f}" if args.t2 is not None else "end"
        title += f"\n(Averaged over $t$ = [{t_start_str}, {t_end_str}])"
    ax.set_title(title)
    ax.set_xlabel(r"Wavenumber $k$ (m$^{-1}$)")
    ax.set_ylabel(r"PSD $E(k)$ (m$^3$ s$^{-2}$)")
    ax.grid(True, alpha=0.3, which="both")
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
