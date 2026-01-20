# Namelist Configuration

PyBurgers is configured using a JSON namelist file (default: `namelist.json`). The namelist is validated against a schema to ensure all required parameters are present and properly formatted.

## Quick Reference

A complete namelist has six required sections:

```json
{
    "time": { ... },
    "physics": { ... },
    "grid": { ... },
    "output": { ... },
    "logging": { ... },
    "fftw": { ... }
}
```

## Complete Example

Here's a typical configuration for running both DNS and LES:

```json
{
    "time" : {
        "nt" : 2E5,
        "dt" : 1E-4
    },
    "physics" : {
        "noise" : {
            "alpha" : 0.75,
            "amplitude" : 1E-6
        },
        "viscosity" : 1E-5,
        "sgs_model" : 1
    },
    "grid" : {
        "domain_length" : 6.283185307179586,
        "dns" : {
            "nx" : 8192
        },
        "les" : {
            "nx" : 512
        }
    },
    "output"  : {
        "t_save" : 0.1
    },
    "logging" : {
        "level" : "INFO",
        "file" : "pyburgers.log"
    },
    "fftw" : {
        "planning" : "FFTW_PATIENT",
        "threads" : 8
    }
}
```

---

## Section: `time`

Controls the temporal discretization and simulation duration.

### `nt` (required)

**Type:** Number

**Description:** Total number of time steps to simulate.

**Example:** `2E5` (200,000 time steps)

**Notes:** Total simulation time = `nt × dt`. Use scientific notation for large values.

### `dt` (required)

**Type:** Number

**Description:** Time step size in seconds.

**Example:** `1E-4` (0.0001 seconds)

**Notes:** Must satisfy CFL condition for numerical stability. Typical values: 1E-4 to 1E-5.

---

## Section: `physics`

Defines the physical parameters of the Burgers equation.

### `viscosity` (required)

**Type:** Number

**Description:** Kinematic viscosity in m²/s.

**Example:** `1E-5`

**Notes:** Controls the rate of diffusion. Lower values lead to sharper gradients and require finer resolution.

### `sgs_model` (optional)

**Type:** Integer (0-4)

**Description:** Subgrid-scale turbulence model for LES mode.

**Options:**
- `0` - No SGS model (DNS or inviscid LES)
- `1` - Constant-coefficient Smagorinsky
- `2` - Dynamic Smagorinsky
- `3` - Dynamic Wong-Lilly
- `4` - Deardorff 1.5-order TKE

**Default:** `0`

**Example:** `1`

**Notes:** Only affects LES runs (`-m les`). DNS mode ignores this setting. Dynamic models (2-4) are more computationally expensive but generally more accurate.

### Subsection: `physics.noise`

Configures the stochastic forcing term (fractional Brownian motion).

#### `alpha` (required)

**Type:** Number

**Description:** Spectral exponent for fractional Brownian motion.

**Example:** `0.75`

**Notes:** Controls the correlation structure of the noise. Values typically range from 0.5 to 1.5. A value of 0.75 produces realistic turbulent forcing.

#### `amplitude` (required)

**Type:** Number

**Description:** Amplitude of the stochastic forcing.

**Example:** `1E-6`

**Notes:** Controls the energy injection rate. Adjust based on viscosity and desired turbulence intensity.

---

## Section: `grid`

Defines the computational domain and spatial resolution.

### `domain_length` (optional)

**Type:** Number

**Description:** Length of the periodic domain in meters.

**Default:** `6.283185307179586` (2π)

**Example:** `6.283185307179586`

**Notes:** The domain is periodic, so this sets the fundamental wavelength. Default value of 2π is convenient for spectral methods.

### Subsection: `grid.dns`

DNS grid configuration (required even if only running LES, as it's used for noise generation).

#### `nx` (required)

**Type:** Integer (must be even)

**Description:** Number of grid points for DNS resolution.

**Example:** `8192`

**Notes:** Must be even for FFT efficiency. Higher resolution resolves smaller scales but increases computational cost. Typical values: 4096 to 16384.

### Subsection: `grid.les`

LES grid configuration.

#### `nx` (required)

**Type:** Integer (must be even)

**Description:** Number of grid points for LES resolution.

**Example:** `512`

**Notes:** Must be even and typically much smaller than DNS resolution. The ratio `dns.nx / les.nx` determines the scale separation. Typical values: 256 to 1024.

---

## Section: `output`

Controls output file writing.

### `t_save` (required)

**Type:** Number

**Description:** Output interval in simulation seconds (not wall-clock time).

**Example:** `0.1`

**Notes:** Data is saved to NetCDF every `t_save` seconds of simulated time. Smaller values produce more output but larger files. The actual number of outputs = `(nt × dt) / t_save`.

---

## Section: `logging`

Configures runtime logging behavior.

### `level` (required)

**Type:** String

**Description:** Logging verbosity level.

**Options:**
- `"DEBUG"` - Verbose diagnostics (FFTW planning details, array shapes, etc.)
- `"INFO"` - Normal runtime information (recommended)
- `"WARNING"` - Only warnings and errors
- `"ERROR"` - Only errors
- `"CRITICAL"` - Only critical failures

**Default:** None (must specify)

**Example:** `"INFO"`

**Notes:** Use `"DEBUG"` for troubleshooting, `"INFO"` for normal runs.

### `file` (optional)

**Type:** String

**Description:** Path to log file. If not specified, logs only to console.

**Example:** `"pyburgers.log"`

**Notes:** The file is created in the current working directory. Logs are appended, not overwritten.

---

## Section: `fftw`

Configures FFTW (Fastest Fourier Transform in the West) behavior.

### `planning` (required)

**Type:** String

**Description:** FFTW planning strategy. Controls the trade-off between planning time and FFT performance.

**Options:**
- `"FFTW_ESTIMATE"` - Fastest planning (~instant), decent performance. Good for quick tests.
- `"FFTW_MEASURE"` - Moderate planning (~seconds), good performance. Balanced choice.
- `"FFTW_PATIENT"` - Thorough planning (~30-60 seconds), better performance. Recommended for production.
- `"FFTW_EXHAUSTIVE"` - Extreme planning (~minutes), best performance. Only for repeated runs with fixed parameters.

**Example:** `"FFTW_PATIENT"`

**Notes:**
- Planning is only done on first run. Subsequent runs load cached "wisdom" and start instantly.
- Wisdom is stored in `~/.pyburgers_fftw_wisdom`.
- If you change grid sizes, wisdom is regenerated automatically.

### `threads` (required)

**Type:** Integer (≥1)

**Description:** Number of threads for FFT operations.

**Example:** `8`

**Notes:**
- Set to the number of physical CPU cores for best performance.
- Diminishing returns beyond ~8 threads for typical grid sizes.
- FFT threading overhead can dominate for very small grids (nx < 512).

---

## Validation

PyBurgers validates the namelist against `pyburgers/schema_namelist.json` at startup. Common errors:

- **Missing required field**: Add the missing key to your namelist
- **Invalid type**: Ensure numbers aren't quoted as strings
- **Invalid value**: Check that enums (like `sgs_model`) use allowed values
- **Invalid structure**: Ensure nested sections (e.g., `physics.noise`) are properly nested

Error messages will indicate the specific problem and location in the namelist.

---

## Performance Tuning

### For Quick Tests
```json
{
    "time": { "nt": 1000, "dt": 1E-3 },
    "grid": { "dns": { "nx": 512 }, "les": { "nx": 128 } },
    "fftw": { "planning": "FFTW_ESTIMATE", "threads": 4 }
}
```

### For Production DNS
```json
{
    "time": { "nt": 5E5, "dt": 5E-5 },
    "grid": { "dns": { "nx": 16384 }, "les": { "nx": 1024 } },
    "fftw": { "planning": "FFTW_PATIENT", "threads": 8 }
}
```

### For Production LES
```json
{
    "time": { "nt": 1E6, "dt": 1E-4 },
    "grid": { "dns": { "nx": 8192 }, "les": { "nx": 512 } },
    "physics": { "sgs_model": 2 },
    "fftw": { "planning": "FFTW_PATIENT", "threads": 8 }
}
```

---

## Tips

1. **Start small**: Test with small `nx` and `nt` values before running full-scale simulations
2. **Monitor first**: Use `logging.level: "DEBUG"` for your first run to ensure everything is configured correctly
3. **Save wisely**: Balance `t_save` between temporal resolution and disk space
4. **Thread carefully**: More threads isn't always better; benchmark your specific hardware
5. **Be patient**: Let FFTW take time to plan on the first run; subsequent runs will be much faster
