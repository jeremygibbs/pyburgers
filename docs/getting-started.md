# Getting Started

This guide will walk you through installing PyBurgers and running your first simulation.

## Prerequisites

PyBurgers requires:

- Python 3.10 or newer
- A C compiler (for building NumPy/FFTW dependencies if needed)
- Basic familiarity with command-line interfaces

### Dependencies

The following packages are installed automatically:

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥2.1 | Array operations and numerical computing |
| pyFFTW | ≥0.15 | Fast Fourier transforms via FFTW |
| netCDF4 | ≥1.7 | Output file format for simulation data |

## Installation

### Option 1: Local Development Install (Recommended)

Clone the repository and install in editable mode:

```bash
# Clone the repository
git clone https://github.com/jeremygibbs/pyburgers.git
cd pyburgers

# Install the package
pip install -e .
```

This installs PyBurgers and its runtime dependencies (NumPy, pyFFTW, netCDF4).

### Option 2: Install with Development Tools

If you plan to contribute or run tests:

```bash
pip install -e ".[dev]"
```

This adds pytest, pytest-cov, and ruff for testing and linting.

### Option 3: Install with Visualization Tools

To use the included plotting scripts for analyzing results:

```bash
pip install -e ".[viz]"
```

This adds matplotlib for creating plots and visualizations of simulation output.

### Option 4: Install with Documentation Tools

To build documentation locally:

```bash
pip install -e ".[docs]"
```

This adds MkDocs, Material theme, and mkdocstrings for building the documentation site.

### Verify Installation

Check that PyBurgers is installed correctly:

```bash
python -c "import pyburgers; print(pyburgers.__version__)"
```

You should see version information printed (or it won't error if `__version__` isn't defined yet).

## Configuration

PyBurgers is configured using a JSON namelist file. The repository includes a default `namelist.json`:

```json
{
    "time": {
        "duration": 200.0,
        "cfl": 0.4,
        "max_step": 0.01
    },
    "grid": {
        "length": 6.283185307179586,
        "dns": {
            "points": 8192
        },
        "les": {
            "points": 512
        }
    },
    "physics": {
        "viscosity": 1e-5,
        "subgrid_model": 2,
        "noise": {
            "exponent": -0.75,
            "amplitude": 1e-6
        }
    },
    "output": {
        "interval_save": 0.1,
        "interval_print": 0.1
    },
    "logging": {
        "level": "INFO",
        "file": "pyburgers.log"
    },
    "fftw": {
        "planning": "FFTW_PATIENT",
        "threads": 8
    }
}
```

For a quick test run, you might want to reduce the grid size and simulation duration:

```json
{
    "time": {
        "duration": 1.0,
        "cfl": 0.4,
        "max_step": 0.01
    },
    "grid": {
        "dns": { "points": 512 },
        "les": { "points": 128 }
    },
    "fftw": {
        "planning": "FFTW_ESTIMATE",
        "threads": 4
    }
}
```

See the [Namelist Configuration](namelist.md) documentation for detailed parameter descriptions.

## Running Your First Simulation

### DNS Mode

Run a Direct Numerical Simulation:

```bash
python burgers.py -m dns
```

This will:
1. Load and validate `namelist.json`
2. Initialize the DNS solver with the specified grid resolution
3. Generate FFTW plans (planning time is hardware-dependent)
4. Run the time-stepping loop
5. Save results to `pyburgers_dns.nc`

**First-run note**: The first time you run PyBurgers with a given grid size and FFTW planning level, it will spend time optimizing FFT plans. These plans are cached in `~/.pyburgers_fftw_wisdom`. Subsequent runs will load the cached plans and start immediately.

### LES Mode

Run a Large-Eddy Simulation:

```bash
python burgers.py -m les
```

This uses the LES grid resolution (`grid.les.points`) and applies the specified subgrid-scale model (`physics.subgrid_model`).

### Custom Output File

Specify a custom output filename:

```bash
python burgers.py -m dns -o my_simulation.nc
```

## Understanding the Output

### Console Output

During the simulation, you'll see log messages (exact form may change):

```
##############################################################
#                                                            #
#                   Welcome to PyBurgers                     #
#     A toy to study Burgers turbulence with DNS and LES     #
#                                                            #
##############################################################
INFO - You are running in DNS mode
INFO - Initializing simulation and planning FFTs...
INFO - Initialization complete. Starting simulation run...
INFO - Done! Completed in 42.15 seconds
##############################################################
```

With `logging.level: "DEBUG"`, you'll see detailed information about FFTW planning, array shapes, and intermediate steps.

### NetCDF Output

Results are saved in NetCDF format. The output file contains:

- **Variables**:
    - `u`: Velocity field (time, x)
    - `t`: Time values
    - `x`: Spatial coordinates
    - Diagnostic quantities (energy, dissipation, etc.)

- **Metadata**:
    - Simulation parameters
    - Timestamps
    - Version information

You can inspect the output using standard NetCDF tools:

```bash
# Using ncdump (if available)
ncdump -h pyburgers_dns.nc

# Using Python
python -c "from netCDF4 import Dataset; ds = Dataset('pyburgers_dns.nc'); print(ds)"
```

### Analyzing Results

PyBurgers includes three example plotting scripts in the `scripts/` directory for visualizing simulation output. These require the visualization dependencies:

```bash
pip install -e ".[viz]"
```

#### Available Plotting Scripts

1. **`plot_velocity.py`** - Visualize velocity field evolution in the x-t plane:
   ```bash
   python scripts/plot_velocity.py pyburgers_dns.nc
   ```
   Creates a space-time diagram showing the velocity field throughout the simulation.

2. **`plot_spectra.py`** - Plot velocity power spectral density:
   ```bash
   python scripts/plot_spectra.py pyburgers_dns.nc
   ```
   Generates power spectra averaged over a time window, useful for analyzing turbulent energy distribution across scales.

3. **`plot_tke.py`** - Compare turbulent kinetic energy (TKE) evolution:
   ```bash
   python scripts/plot_tke.py pyburgers_dns.nc pyburgers_les.nc
   ```
   Compares TKE time series from DNS and LES runs to evaluate subgrid-scale model performance.

Each script accepts `--help` for additional options:
```bash
python scripts/plot_velocity.py --help
```

## Next Steps

### Customize Your Simulation

Default simulations are crafted following published results and numerical best practices. However, users are free to adjust the simulation settings in `namelist.json` for their own purposes.

1. **Adjust grid resolution**: Modify `grid.dns.points` and `grid.les.points`
2. **Change simulation duration**: Adjust `time.duration` for longer/shorter runs
3. **Tune time stepping**: Adjust `time.cfl` and `time.max_step` to control adaptive stepping
4. **Try different SGS models**: Set `physics.subgrid_model` to 0-4 for LES runs
5. **Tune FFTW**: Experiment with planning levels (ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE)
6. **Control output**: Adjust `output.interval_save` to save more or fewer snapshots

### Learn More

- [Namelist Configuration](namelist.md) - Detailed parameter reference
- [API Reference](reference.md) - Code documentation
- [Contributing](contributing.md) - Development setup and guidelines

### Performance Tips

1. **Grid size**: Start small (nx=512) for testing, scale up for production
2. **FFTW planning**: Use ESTIMATE for quick tests, PATIENT for production
3. **Threading**: Set `fftw.threads` to tune performance
4. **Output frequency**: Higher `interval_save` values reduce I/O overhead
5. **Wisdom caching**: After the first run, subsequent runs are much faster

### Troubleshooting

**Problem**: Simulation is very slow

- Check that `fftw.planning` isn't EXHAUSTIVE (unless intentional)
- Reduce `time.duration` for testing
- Ensure FFTW wisdom is being cached (check for `~/.pyburgers_fftw_wisdom`)

**Problem**: "NamelistError" on startup

- Validate your JSON syntax (use a JSON linter)
- Ensure all required fields are present
- Check that numeric values aren't quoted as strings

**Problem**: Out of memory

- Reduce `grid.dns.points` and/or `grid.les.points`
- The default 8192 grid points requires several GB of RAM

**Problem**: FFTW planning takes forever

- Use `FFTW_ESTIMATE` or `FFTW_MEASURE` instead of `FFTW_PATIENT`/`FFTW_EXHAUSTIVE`
- This is normal on first run with PATIENT/EXHAUSTIVE; subsequent runs are instant

## Example Workflows

### Quick Test Run

Create a test namelist (`test_namelist.json`):

```json
{
    "time": { "duration": 1.0, "cfl": 0.4, "max_step": 0.01 },
    "physics": {
        "noise": { "exponent": -0.75, "amplitude": 1e-6 },
        "viscosity": 1e-5,
        "subgrid_model": 1
    },
    "grid": {
        "dns": { "points": 512 },
        "les": { "points": 128 }
    },
    "output": { "interval_save": 0.1 },
    "logging": { "level": "INFO" },
    "fftw": { "planning": "FFTW_ESTIMATE", "threads": 4 }
}
```

Then copy it over `namelist.json` to use it.

### Production DNS Run

Use the default `namelist.json` with:
- `grid.dns.points`: 8192 or 16384
- `fftw.planning`: FFTW_PATIENT
- `time.duration`: 200 to 500

### Production LES Comparison

Run both modes to compare:

```bash
# DNS reference
python burgers.py -m dns -o reference_dns.nc

# LES with different SGS models
sed -i 's/"subgrid_model": 1/"subgrid_model": 1/' namelist.json
python burgers.py -m les -o les_smagorinsky.nc

sed -i 's/"subgrid_model": 1/"subgrid_model": 2/' namelist.json
python burgers.py -m les -o les_dynamic.nc
```

Then compare the results to evaluate SGS model performance.

---

Happy simulating! If you encounter issues, please [open an issue](https://github.com/jeremygibbs/pyburgers/issues) on GitHub.
