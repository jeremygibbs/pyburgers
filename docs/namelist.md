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
        "t_save" : 0.1,
        "t_print" : 0.01
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

## Section: time

Controls the temporal discretization and simulation duration.

`nt`
:   **Type:** Number (required)
    **Default:** None

    Total number of time steps to simulate.

    Total simulation time = `nt × dt`. Use scientific notation for large values.

    **Example:** `2E5` (200,000 time steps)

`dt`
:   **Type:** Number (required)
    **Default:** None

    Time step size in seconds.

    Must satisfy CFL condition for numerical stability. Typical values: 1E-4 to 1E-5.

    **Example:** `1E-4` (0.0001 seconds)

---

## Section: physics

Defines the physical parameters of the Burgers equation.

`viscosity`
:   **Type:** Number (required)
    **Default:** None

    Kinematic viscosity in m²/s.

    Controls the rate of diffusion. Lower values lead to sharper gradients and require finer resolution.

    **Example:** `1E-5`

`sgs_model`
:   **Type:** Integer (optional)
    **Default:** `0`

    Subgrid-scale turbulence model for LES mode.

    **Available models:**

    - `0` - No SGS model (DNS or inviscid LES)
    - `1` - Constant-coefficient Smagorinsky
    - `2` - Dynamic Smagorinsky
    - `3` - Dynamic Wong-Lilly
    - `4` - Deardorff 1.5-order TKE

    **Note:** Only affects LES runs (`-m les`). DNS mode ignores this setting. Dynamic models (2-4) are more computationally expensive but generally more accurate.

    **Example:** `1`

### Subsection: physics.noise

Configures the stochastic forcing term (fractional Brownian motion).

`alpha`
:   **Type:** Number (required)
    **Default:** None

    Spectral exponent for fractional Brownian motion.

    Controls the correlation structure of the noise. Values typically range from 0.5 to 1.5. A value of 0.75 produces realistic turbulent forcing.

    **Example:** `0.75`

`amplitude`
:   **Type:** Number (required)
    **Default:** None

    Amplitude of the stochastic forcing.

    Controls the energy injection rate. Adjust based on viscosity and desired turbulence intensity.

    **Example:** `1E-6`

---

## Section: grid

Defines the computational domain and spatial resolution.

`domain_length`
:   **Type:** Number (optional)
    **Default:** `6.283185307179586` (2π)

    Length of the periodic domain in meters.

    The domain is periodic, so this sets the fundamental wavelength. Default value of 2π is convenient for spectral methods.

    **Example:** `6.283185307179586`

### Subsection: grid.dns

DNS grid configuration (required even if only running LES, as it's used for noise generation).

`nx`
:   **Type:** Integer (required)
    **Default:** None

    Number of grid points for DNS resolution.

    Must be even for FFT efficiency. Higher resolution resolves smaller scales but increases computational cost. Typical values: 4096 to 16384.

    **Example:** `8192`

### Subsection: grid.les

LES grid configuration.

`nx`
:   **Type:** Integer (required)
    **Default:** None

    Number of grid points for LES resolution.

    Must be even and typically much smaller than DNS resolution. The ratio `dns.nx / les.nx` determines the filter width. Typical values: 256 to 1024.

    **Note:** The filter width is automatically computed as `dns.nx / les.nx` and displayed in the LES startup log.

    **Example:** `512` (with `dns.nx=8192`, this gives filter width of 16Δx)

---

## Section: output

Controls output file writing and progress reporting.

`t_save`
:   **Type:** Number (required)
    **Default:** None

    Output interval in simulation seconds (not wall-clock time).

    Data is saved to NetCDF every `t_save` seconds of simulated time. Smaller values produce more output but larger files. The actual number of outputs = `(nt × dt) / t_save`.

    **Example:** `0.1`

`t_print`
:   **Type:** Number (optional)
    **Default:** Same as `t_save`

    Progress reporting interval in simulation seconds (not wall-clock time).

    Progress messages are printed to the console every `t_print` seconds of simulated time. This is independent of `t_save`, allowing you to monitor progress more frequently without increasing file output. Smaller values provide more frequent updates but may clutter the console.

    **Example:** `0.01` (prints 10x more frequently than saving)

---

## Section: logging

Configures runtime logging behavior.

`level`
:   **Type:** String (required)
    **Default:** None

    Logging verbosity level.

    **Available levels:**

    - `"DEBUG"` - Verbose diagnostics (FFTW planning details, array shapes, etc.)
    - `"INFO"` - Normal runtime information (recommended)
    - `"WARNING"` - Only warnings and errors
    - `"ERROR"` - Only errors
    - `"CRITICAL"` - Only critical failures

    **Tip:** Use `"DEBUG"` for troubleshooting, `"INFO"` for normal runs.

    **Example:** `"INFO"`

`file`
:   **Type:** String (optional)
    **Default:** None (log to console only)

    Path to log file. If not specified, logs only to console.

    The file is created in the current working directory. Logs are appended, not overwritten.

    **Example:** `"pyburgers.log"`

---

## Section: fftw

Configures FFTW (Fastest Fourier Transform in the West) behavior.

`planning`
:   **Type:** String (required)
    **Default:** None

    FFTW planning strategy. Controls the trade-off between planning time and FFT performance.

    **Available strategies:**

    - `"FFTW_ESTIMATE"` - Fastest planning (~instant), decent performance. Good for quick tests.
    - `"FFTW_MEASURE"` - Moderate planning (~seconds), good performance. Balanced choice.
    - `"FFTW_PATIENT"` - Thorough planning (~30-60 seconds), better performance. Recommended for production.
    - `"FFTW_EXHAUSTIVE"` - Extreme planning (~minutes), best performance. Only for repeated runs with fixed parameters.

    **Note:** Planning is only done on first run. Subsequent runs load cached "wisdom" and start instantly. Wisdom is stored in `~/.pyburgers_fftw_wisdom`. If you change grid sizes, wisdom is regenerated automatically.

    **Example:** `"FFTW_PATIENT"`

`threads`
:   **Type:** Integer (required)
    **Default:** None

    Number of threads for FFT operations.

    Set to the number of physical CPU cores for best performance. Diminishing returns beyond ~8 threads for typical grid sizes. FFT threading overhead can dominate for very small grids (nx < 512).

    **Example:** `8`

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

!!! tip "Best Practices"
    1. **Start small**: Test with small `nx` and `nt` values before running full-scale simulations
    2. **Monitor first**: Use `logging.level: "DEBUG"` for your first run to ensure everything is configured correctly
    3. **Save wisely**: Balance `t_save` between temporal resolution and disk space
    4. **Print smartly**: Set `t_print` smaller than `t_save` to monitor progress without bloating output files
    5. **Thread carefully**: More threads isn't always better; benchmark your specific hardware
    6. **Be patient**: Let FFTW take time to plan on the first run; subsequent runs will be much faster
