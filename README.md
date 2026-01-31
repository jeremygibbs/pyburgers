![PyBurgers Logo](https://gibbs.science/img/pyburgers_v2.png)

# PyBurgers

A high-performance solver for the 1D Stochastic Burgers Equation with DNS and LES capabilities.

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.gibbs.science/pyburgers)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![DOI](https://zenodo.org/badge/104835582.svg)](https://doi.org/10.5281/zenodo.18444178)

## Overview

PyBurgers implements direct numerical simulation (DNS) and large-eddy simulation (LES) for studying Burgers turbulence, following the procedures described in [Basu (2009)](https://doi.org/10.1080/14685240902852719). The solver uses Fourier collocation methods for spatial derivatives and Williamson (1980) low-storage RK3 time integration with CFL-based adaptive time stepping.

## Features

- **Dual Simulation Modes**: DNS for full resolution and LES for coarse-grained modeling
- **Four SGS Models**: Constant Smagorinsky, Dynamic Smagorinsky, Dynamic Wong-Lilly, and Deardorff 1.5-order TKE
- **Optimized FFTs**: FFTW with intelligent wisdom caching for fast repeated runs
- **Fractional Brownian Motion**: Configurable stochastic forcing with spectral control
- **NetCDF Output**: Standard format for analysis and visualization
- **Schema-Validated Configuration**: JSON namelist with comprehensive validation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jeremygibbs/pyburgers.git
cd pyburgers

# Install the package
pip install -e .
```

### Run a Simulation

```bash
# Run DNS simulation (default: 8192 grid points)
python burgers.py -m dns

# Run LES simulation with custom output file
python burgers.py -m les -o my_simulation.nc
```

Configuration is controlled via `namelist.json`. See the [documentation](https://docs.gibbs.science/pyburgers) for details.

### Compare DNS vs LES TKE

Install the optional visualization dependencies:

```bash
pip install -e ".[viz]"
```

Then compare TKE output from one DNS run and multiple LES runs:

```bash
python scripts/compare_tke.py -d pyburgers_dns.nc -l pyburgers_les.nc -l pyburgers_les_alt.nc
```

## Documentation

Full documentation is available at: **https://docs.gibbs.science/pyburgers**

- [Getting Started](https://docs.gibbs.science/pyburgers/getting-started/)
- [Namelist Configuration](https://docs.gibbs.science/pyburgers/namelist/)
- [API Reference](https://docs.gibbs.science/pyburgers/reference/)
- [Contributing Guide](https://docs.gibbs.science/pyburgers/contributing/)

## Performance

PyBurgers v2.0 delivers dramatic performance improvements through real FFTs, optimized FFTW planning, and efficient buffer management.

**Benchmark: Default namelist (8192 DNS / 512 LES grid points, 200s duration)**

| Version | DNS | LES |
|---------|-----|-----|
| Original Matlab | ~35 min | ~16 min |
| PyBurgers v1.0 | ~43 min | ~23 min |
| **PyBurgers v2.0** | **~40 sec** | **~7 sec** |

*Tested on a late 2023 MacBook Pro (M3 Max). Performance varies by system; results illustrate relative gains.*

## Requirements

- Python ≥ 3.10
- NumPy ≥ 2.1
- pyFFTW ≥ 0.15
- netCDF4 ≥ 1.7

## Citation

If you use PyBurgers in your research, please cite:

```bibtex
@software{pyburgers,
  author = {Gibbs, Jeremy A.},
  title = {PyBurgers: 1D Stochastic Burgers Equation Solver},
  year = {2026},
  url = {https://github.com/jeremygibbs/pyburgers},
  version = {2.0.0}
}
```

And reference the underlying methodology:

```bibtex
@article{basu2009,
  author = {Basu, Sukanta},
  title = {High-resolution large-eddy simulations of stably stratified flows:
           application to the Cooperative Atmosphere–Surface Exchange Study 1999 (CASES-99)},
  journal = {Journal of Turbulence},
  volume = {10},
  pages = {N12},
  year = {2009},
  doi = {10.1080/14685240902852719}
}

@article{williamson1980,
  author = {Williamson, J.H.},
  title = {Low-storage Runge-Kutta schemes},
  journal = {Journal of Computational Physics},
  volume = {35},
  number = {1},
  pages = {48--56},
  year = {1980},
  doi = {10.1016/0021-9991(80)90033-9}
}
```

## License

This software is free and is distributed under the **WTFPL** (Do What The Fuck You Want To Public License). It comes without any warranty, to the extent permitted by applicable law. You can redistribute it and/or modify it under the terms of the WTFPL, Version 2, as published by Sam Hocevar. See http://www.wtfpl.net for more details.

## Contact

Jeremy A. Gibbs - jeremy.gibbs@noaa.gov

For bug reports and feature requests, please [open an issue](https://github.com/jeremygibbs/pyburgers/issues).
