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

## Requirements

These scripts require the visualization dependencies:

```bash
pip install -e ".[viz]"
```

This installs matplotlib along with the core PyBurgers dependencies.
