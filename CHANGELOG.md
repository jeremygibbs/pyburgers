# Changelog

All notable changes to PyBurgers will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-XX

Version 2.0 represents a complete rewrite of PyBurgers with modern Python practices, significant performance improvements, and enhanced usability.

### Added

- **Schema-validated namelist**: JSON configuration with comprehensive validation via `schema_namelist.json`
- **FFTW wisdom caching**: Intelligent caching system stores optimized FFT plans in `~/.pyburgers_fftw_wisdom` for instant startup on subsequent runs
- **File locking**: Thread-safe wisdom file access prevents corruption when running multiple instances
- **Comprehensive logging**: Configurable logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) with file and console output
- **Data models**: Pydantic-style dataclasses for type-safe configuration (`data_models.py`)
- **MkDocs documentation**: Professional documentation site with Material theme
- **Namelist documentation generator**: Script to auto-generate documentation from JSON schema
- **Comprehensive test suite**: Pytest-based tests for derivatives, FBM, filtering, SGS models, and integration
- **FFTW planning levels**: Support for ESTIMATE, MEASURE, PATIENT, and EXHAUSTIVE planning strategies
- **Multithreading**: Configurable thread count for FFT operations
- **Custom exceptions**: `PyBurgersError`, `NamelistError`, `InvalidMode` for better error handling
- **NetCDF metadata**: Enhanced output with comprehensive simulation metadata
- **pyproject.toml**: Modern Python packaging with optional dependencies for dev and docs

### Changed

- **Python requirement**: Now requires Python ≥ 3.10 (up from Python 2/3 compatibility)
- **FFT backend**: Switched to real FFTs (rfft/irfft) for ~2× speedup and 50% memory reduction
- **Project structure**: Complete reorganization into logical modules:
  - `pyburgers/core.py`: Abstract base class
  - `pyburgers/dns.py`: DNS solver
  - `pyburgers/les.py`: LES solver
  - `pyburgers/utils/`: Utilities (spectral, IO, logging, FFTW)
  - `pyburgers/physics/sgs/`: Subgrid-scale models
- **Spectral workspace**: Unified workspace object manages all FFT operations and buffers
- **Initialization**: Streamlined startup with warmup phase for FFTW planning
- **Time stepping**: Refactored Adams-Bashforth implementation in base class
- **Output organization**: NetCDF output with improved structure and metadata
- **Logging**: Moved from print statements to proper Python logging framework
- **Constants**: Centralized physical and numerical constants in `utils/constants.py`

### Improved

- **Performance**: Significant speedup through real FFTs, optimized FFTW planning, and reduced overhead
- **Code organization**: Clear separation of concerns with abstract base class and mode-specific implementations
- **Documentation**: Comprehensive docs covering theory, usage, API reference, and contribution guidelines
- **Error messages**: Descriptive error messages with context and suggestions
- **Type hints**: Full type annotations throughout the codebase
- **Reproducibility**: Fixed random seed for consistent results across runs
- **Memory efficiency**: Reduced memory footprint through real FFT usage and efficient buffer management

### Fixed

- **Deardorff model bugs**: Corrected implementation issues in the TKE-based SGS model
- **Nyquist mode handling**: Proper zeroing of Nyquist mode to prevent aliasing
- **Edge cases**: Improved handling of boundary conditions and special cases

### Developer Experience

- **Linting**: Configured Ruff for code quality (pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear)
- **Formatting**: Automated code formatting with Ruff
- **Testing**: Easy test execution with `pytest` and coverage reports
- **CI/CD**: GitHub Actions workflow for automated documentation deployment
- **Development install**: Simple `pip install -e ".[dev]"` for development dependencies

### Migration from v1.x

Version 2.0 is not backward compatible with v1.x. Key migration steps:

1. **Configuration**: Convert old configuration to JSON namelist format
2. **Python version**: Upgrade to Python 3.10 or newer
3. **Dependencies**: Update to numpy ≥ 2.1, pyfftw ≥ 0.15, netCDF4 ≥ 1.7
4. **Import paths**: Update imports to use new module structure
5. **Output files**: NetCDF structure has changed; update analysis scripts accordingly

### Notes

- First run with new grid sizes will take extra time for FFTW planning; subsequent runs will be fast
- FFTW wisdom file is stored at `~/.pyburgers_fftw_wisdom` and can be deleted to force re-planning
- Recommended FFTW planning level is `FFTW_PATIENT` for production runs

---

## [1.0.0] - 2017-XX-XX

Initial release of PyBurgers.

### Features

- Direct Numerical Simulation (DNS) mode
- Large-Eddy Simulation (LES) mode
- Four subgrid-scale models (Smagorinsky, Dynamic Smagorinsky, Wong-Lilly, Deardorff)
- Fourier collocation spatial methods
- Adams-Bashforth time integration
- Fractional Brownian motion forcing
- NetCDF output
