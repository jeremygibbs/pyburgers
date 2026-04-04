# Namelist Configuration

PyBurgers is configured using a JSON namelist file (default: `namelist.json`). The namelist is validated at startup to ensure all required parameters are present and properly formatted.

## Quick Reference

A complete namelist has seven sections:

```json
{
    "time": { ... },
    "grid": { ... },
    "physics": { ... },
    "output": { ... },
    "numerics": { ... },
    "logging": { ... },
    "fftw": { ... }
}
```

## Complete Example

Here's a typical configuration for running both DNS and LES:

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
            "amplitude": 1e-6,
            "seed": 1
        },
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
        "threads": 1
    },
    "numerics": {
        "temporal": 3,
        "spatial": 3
    }
}
```

---

## Section: time

Controls the simulation duration and adaptive time stepping.

`duration`
:   **Type:** Number (required)
    **Default:** None

    Total simulation time in seconds.

    The simulation runs until this physical time is reached.

    **Example:** `200.0` (200 seconds of simulated time)

`cfl`
:   **Type:** Number (required)
    **Default:** None
    **Valid range:** (0, 0.55)

    Target CFL number for adaptive time stepping.

    The time step is automatically adjusted each iteration to satisfy dt ≤ cfl × dx / |u_max|. Lower values are more conservative but slower. Typical values: 0.3 to 0.5.

    **Example:** `0.4`

`max_step`
:   **Type:** Number (required)
    **Default:** None

    Maximum allowed time step in seconds.

    Caps the adaptive time step. Typical values: 0.001 to 0.01.

    **Note:** Stochastic noise is refreshed at `max_step` intervals in both DNS and LES modes. This ensures both simulations consume the same random sequence, making their results directly comparable even though adaptive time stepping may produce different sub-step sizes.

    **Example:** `0.01`

---

## Section: grid

Defines the computational domain and spatial resolution.

`length`
:   **Type:** Number (optional)
    **Default:** `6.283185307179586` (2π)

    Length of the periodic domain in meters.

    The domain is periodic, so this sets the fundamental wavelength. Default value of 2π is convenient for spectral methods.

    **Example:** `6.283185307179586`

### Subsection: grid.dns

DNS grid configuration (required even if only running LES, as it's used for noise generation).

`points`
:   **Type:** Integer (optional)
    **Default:** `8192`

    Number of grid points for DNS resolution.

    Must be even for FFT efficiency. Higher resolution resolves smaller scales but increases computational cost. Typical values: 4096 to 16384.

    **Example:** `8192`

### Subsection: grid.les

LES grid configuration.

`points`
:   **Type:** Integer (optional)
    **Default:** `512`

    Number of grid points for LES resolution.

    Must be even and typically much smaller than DNS resolution. The ratio `dns.points / les.points` determines the filter width. Typical values: 256 to 1024.

    **Note:** The filter width is automatically computed as `dns.points / les.points` and displayed in the LES startup log.

    **Example:** `512` (with `dns.points=8192`, this gives filter width of 16Δx)

---

## Section: physics

Defines the physical parameters of the Burgers equation.

`viscosity`
:   **Type:** Number (required)
    **Default:** None

    Kinematic viscosity in m²/s.

    Controls the rate of diffusion. Lower values lead to sharper gradients and require finer resolution.

    **Example:** `1e-5`

`subgrid_model`
:   **Type:** Integer (optional)
    **Default:** `1`

    Subgrid-scale turbulence model for LES mode.

    **Available models:**

    - `1` - Constant-coefficient Smagorinsky
    - `2` - Dynamic Smagorinsky
    - `3` - Dynamic Wong-Lilly
    - `4` - Deardorff 1.5-order TKE

    **Note:** Required for LES runs (`-m les`). DNS mode ignores this setting. Dynamic models (2-4) are more computationally expensive but generally more accurate.

    **Example:** `2`

### Subsection: physics.noise

Configures the stochastic forcing term (fractional Brownian motion).

`exponent`
:   **Type:** Number (optional)
    **Default:** `-0.75`

    Spectral exponent for fractional Brownian motion.

    Controls the correlation structure of the noise. Negative values produce red noise (energy concentrated at low wavenumbers), which is physically appropriate for turbulent forcing. A value of -0.75 produces realistic turbulent forcing following Basu (2009).

    **Example:** `-0.75`

`amplitude`
:   **Type:** Number (optional)
    **Default:** `1e-6`

    Amplitude of the stochastic forcing.

    Controls the energy injection rate. Adjust based on viscosity and desired turbulence intensity.

    **Example:** `1e-6`

`seed`
:   **Type:** Integer or null (optional)
    **Default:** `null`

    Seed for the random number generator used in noise generation.

    When set to an integer, the simulation is fully reproducible: every run with the same seed and namelist produces identical results. This is useful for debugging and regression testing. When set to `null` (or omitted), a random seed is drawn at startup. For ensemble studies, seeds can also be controlled externally by incrementing this value across runs.

    Uses NumPy's `default_rng` (PCG64 generator). Note that seeds are not portable across NumPy versions.

    **Example:** `1`

---

## Section: output

Controls output file writing and progress reporting.

`interval_save`
:   **Type:** Number (optional)
    **Default:** `100 × max_step`

    Output interval in simulation seconds (not wall-clock time).

    Data is saved to NetCDF every `interval_save` seconds of simulated time. Smaller values produce more output but larger files. The adaptive time stepper clamps dt to hit output times exactly.

    **Example:** `0.1`

`interval_print`
:   **Type:** Number (optional)
    **Default:** Same as `interval_save`

    Progress reporting interval in simulation seconds (not wall-clock time).

    Progress messages are printed to the console every `interval_print` seconds of simulated time. This is independent of `interval_save`, allowing you to monitor progress more frequently without increasing file output.

    **Example:** `0.1`

---

## Section: numerics

Configures numerical method selections.

`temporal`
:   **Type:** Integer (optional)
    **Default:** `3`
    **Valid values:** `1`, `2`, `3`

    Time integration scheme selector.

    **Available schemes:**

    - `1` - Adams-Bashforth 2nd order (2nd-order, 1 evaluation per step, bootstrapped with forward Euler)
    - `2` - Adams-Moulton 2nd order predictor-corrector (2nd-order, 2 evaluations per step: AB2 predictor + AM2 corrector, bootstrapped with forward Euler)
    - `3` - Williamson (1980) low-storage RK3 (3rd-order, 3 stages per step)

    **Note:** AB2 and AM2 are multistep methods that use the variable-step formula to maintain 2nd-order accuracy under CFL-based adaptive time stepping. AM2 uses AB2 as its predictor and applies the trapezoidal corrector, reducing the local truncation error at the cost of one additional RHS evaluation per step. RK3 is higher-order and self-starting (no bootstrap required), making it the recommended default for accuracy-critical simulations.

    **Example:** `3`

`spatial`
:   **Type:** Integer (optional)
    **Default:** `1`
    **Valid values:** `1`, `2`, `3`

    Spatial discretization scheme selector. Controls how all spatial derivatives (advection, viscosity, hyperviscosity) are computed.

    **Available schemes:**

    - `1` - 2nd-order central finite differences
    - `2` - 4th-order central finite differences
    - `3` - Spectral (Fourier collocation)

    **Note:** Hyperviscosity is applied automatically only when `spatial: 3`. Finite-difference schemes do not exhibit spectral pile-up and do not need it.

    **Example:** `3`

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
:   **Type:** String (optional)
    **Default:** `"FFTW_MEASURE"`

    FFTW planning strategy. Controls the trade-off between planning time and FFT performance.

    **Available strategies** (planning time is hardware-dependent):

    - `"FFTW_ESTIMATE"` - Fastest planning, decent performance. Good for quick tests.
    - `"FFTW_MEASURE"` - Moderate planning, good performance. Balanced choice.
    - `"FFTW_PATIENT"` - Thorough planning, better performance. Recommended for production.
    - `"FFTW_EXHAUSTIVE"` - Extensive planning, best performance. Only for repeated runs with fixed parameters.

    **Note:** Planning is only done on first run. Subsequent runs load cached "wisdom" and start instantly. Wisdom is stored in `~/.pyburgers_fftw_wisdom`. If you change grid sizes, wisdom is regenerated automatically.

    **Example:** `"FFTW_PATIENT"`

`threads`
:   **Type:** Integer (optional)
    **Default:** `4`

    Number of threads for FFT operations.

    Set to the number of physical CPU cores for best performance. Diminishing returns beyond ~8 threads for typical grid sizes. FFT threading overhead can dominate for very small grids (points < 512).

    **Example:** `8`

---

## Validation

PyBurgers validates the namelist at startup. Common errors:

- **Missing required field**: Add the missing key to your namelist
- **Invalid type**: Ensure numbers aren't quoted as strings
- **Invalid value**: Check that enums (like `subgrid_model`) use allowed values
- **Invalid CFL**: The `cfl` value must be in the range (0, 0.55)
- **Invalid structure**: Ensure nested sections (e.g., `physics.noise`) are properly nested

Error messages will indicate the specific problem and location in the namelist.

---

## Performance Tuning

### For Quick Tests

```json
{
    "time": { "duration": 1.0, "cfl": 0.4, "max_step": 0.01 },
    "grid": { "dns": { "points": 512 }, "les": { "points": 128 } },
    "fftw": { "planning": "FFTW_ESTIMATE", "threads": 4 }
}
```

### For Production DNS

```json
{
    "time": { "duration": 500.0, "cfl": 0.4, "max_step": 0.001 },
    "grid": { "dns": { "points": 16384 }, "les": { "points": 1024 } },
    "fftw": { "planning": "FFTW_PATIENT", "threads": 8 }
}
```

### For Production LES

```json
{
    "time": { "duration": 1000.0, "cfl": 0.4, "max_step": 0.01 },
    "grid": { "dns": { "points": 8192 }, "les": { "points": 512 } },
    "physics": { "subgrid_model": 2 },
    "fftw": { "planning": "FFTW_PATIENT", "threads": 8 }
}
```

---

## Tips

!!! tip "Best Practices"
    1. **Start small**: Test with small grid sizes and short durations before running full-scale simulations
    2. **Monitor first**: Use `logging.level: "DEBUG"` for your first run to ensure everything is configured correctly
    3. **Save wisely**: Balance `interval_save` between temporal resolution and disk space
    4. **Print smartly**: Set `interval_print` smaller than `interval_save` to monitor progress without bloating output files
    5. **Thread carefully**: More threads isn't always better; benchmark your specific hardware
    6. **Be patient**: Let FFTW take time to plan on the first run; subsequent runs will be much faster
