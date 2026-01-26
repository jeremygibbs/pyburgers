#!/usr/bin/env python
"""Compare DNS and LES TKE output from PyBurgers NetCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np


def _read_tke(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    parser = argparse.ArgumentParser(
        description="Compare TKE time series from DNS and LES NetCDF outputs."
    )
    parser.add_argument(
        "-d",
        "--dns",
        required=True,
        type=Path,
        help="Path to DNS NetCDF output file.",
    )
    parser.add_argument(
        "-l",
        "--les",
        action="extend",
        nargs="+",
        required=True,
        type=Path,
        help="Path(s) to LES NetCDF output file(s). Repeatable.",
    )
    parser.add_argument(
        "--les-labels",
        nargs="+",
        default=None,
        help="Optional labels for LES runs (must match number of LES files).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image path (PNG/SVG/etc). If omitted, shows plot.",
    )
    args = parser.parse_args()

    les_paths = args.les
    if args.les_labels and len(args.les_labels) != len(les_paths):
        raise SystemExit("Number of --les-labels must match number of LES files.")

    dns_time, dns_tke = _read_tke(args.dns)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dns_time, dns_tke, label=args.dns.stem, linewidth=2.0)

    for idx, les_path in enumerate(les_paths):
        les_time, les_tke = _read_tke(les_path)
        if args.les_labels:
            label = args.les_labels[idx]
        else:
            label = les_path.stem
        ax.plot(les_time, les_tke, label=label)

    ax.set_title("PyBurgers: Turbulence Kinetic Energy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("TKE (m^2 s^-2)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
