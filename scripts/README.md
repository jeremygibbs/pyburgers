# PyBurgers Plotting Scripts

Utility scripts for visualizing PyBurgers simulation output.

## Available Scripts

### plot_tke.py

Plot turbulence kinetic energy (TKE) time series from DNS and LES simulations.

**Usage:**
```bash
# Single file
python scripts/plot_tke.py pyburgers_dns.nc

# Compare multiple runs (DNS and/or LES)
python scripts/plot_tke.py pyburgers_dns.nc pyburgers_les.nc

# Multiple LES runs
python scripts/plot_tke.py pyburgers_les_01.nc pyburgers_les_02.nc pyburgers_les_03.nc

# Save to file instead of displaying
python scripts/plot_tke.py pyburgers_dns.nc pyburgers_les.nc -o tke_comparison.png
```

**Output:** Line plot of TKE vs. time (uses filename as legend label)

**Options:**
- `-o/--out`: Save to file (PNG, SVG, PDF, etc.)

### plot_velocity.py

Visualize velocity field in the x-t plane as a space-time diagram using pcolormesh.

**Usage:**
```bash
# Single file
python scripts/plot_velocity.py pyburgers_dns.nc

# Multiple files (creates subplots)
python scripts/plot_velocity.py pyburgers_les_01.nc pyburgers_les_02.nc pyburgers_les_03.nc

# With custom colormap and save to file
python scripts/plot_velocity.py \
  pyburgers_les_01.nc pyburgers_les_02.nc pyburgers_les_03.nc \
  --cmap seismic \
  -o velocity_comparison.png

# Control colorbar range
python scripts/plot_velocity.py pyburgers_dns.nc --vmin -2 --vmax 2
```

**Output:** 2D pcolormesh plot(s) showing velocity evolution in space and time (uses filename as subplot title)

**Options:**
- `--cmap`: Matplotlib colormap (default: RdBu_r)
- `--vmin/--vmax`: Colorbar range (default: auto-scaled from data)
- `-o/--out`: Save to file (PNG, SVG, PDF, etc.)

### plot_spectra.py

Plot time-averaged power spectral density (PSD) of the velocity field with theoretical scaling.

**Usage:**
```bash
# Single file
python scripts/plot_spectra.py pyburgers_dns.nc

# Compare DNS and LES spectra
python scripts/plot_spectra.py pyburgers_dns.nc pyburgers_les.nc

# Multiple files (all on one panel)
python scripts/plot_spectra.py pyburgers_les_01.nc pyburgers_les_02.nc pyburgers_les_03.nc

# Save to file
python scripts/plot_spectra.py pyburgers_dns.nc -o spectra.png

# Adjust y-axis clipping threshold to exclude high-frequency noise
python scripts/plot_spectra.py pyburgers_dns.nc --threshold 1e-8

# Check variance vs. summed PSD
python scripts/plot_spectra.py pyburgers_dns.nc --check-variance
```

**Output:** Single log-log plot of power spectral density E(k) vs. wavenumber k, with theoretical k^(-5/3)
inertial range scaling shown as dashed black line across all wavenumbers (uses filename as legend label)

**Options:**
- `--t1`: Start time for averaging window (default: use all times)
- `--t2`: End time for averaging window (default: use all times)
- `--threshold`: Clip y-axis to exclude PSD values below this threshold (default: 1e-10)
- `--check-variance`: Print variance vs. summed PSD check for each file
- `-o/--out`: Save to file (PNG, SVG, PDF, etc.)

## Requirements

These scripts require the visualization dependencies:

```bash
pip install -e ".[viz]"
```

This installs matplotlib along with the core PyBurgers dependencies.
