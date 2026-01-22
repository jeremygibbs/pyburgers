# PyBurgers Documentation

Welcome to the documentation for PyBurgers, a high-performance solver for the 1D Stochastic Burgers Equation.

## What is PyBurgers?

PyBurgers provides both Direct Numerical Simulation (DNS) and Large-Eddy Simulation (LES) capabilities for studying Burgers turbulence. The solver uses Fourier collocation methods in space and second-order Adams-Bashforth time integration, following the procedures described in [Basu (2009)](https://doi.org/10.1080/14685240902852719).

## Scientific Background

### The Burgers Equation

The Burgers equation was originally conceived by Dutch scientist J.M. Burgers in the 1930s as one of the first attempts to arrive at a statistical theory of turbulent fluid motion. The original equation shares many characteristics with the Navier-Stokes equations:

- **Advective non-linearity**: The convective term u∂u/∂x
- **Diffusion**: Viscous damping
- **Invariance**: Conservation laws and symmetries
- **Conservation properties**: Mass and momentum conservation

### Limitations and Extensions

While the Burgers equation is not an ideal model for the chaotic nature of turbulence (it can be integrated explicitly and is not sensitive to small changes in initial conditions), it remains valuable for understanding turbulent dynamics. In the limit of vanishing viscosity, shock waves form, providing insight into energy dissipation mechanisms.

A popular modification is the addition of a forcing term that accounts for neglected effects. By perturbing the system with a stochastic process that is stationary in time and space, we obtain the **1D Stochastic Burgers Equation (1D SBE)**.

### Why Study the 1D SBE?

The 1D SBE provides valuable insights into turbulence without the computational burden of generalizing to 3D:

- **Nonlinearity**: Captures the essence of turbulent energy transfer
- **Energy spectrum**: Exhibits spectral characteristics similar to 3D turbulence
- **Intermittent dissipation**: Shows bursts of energy dissipation
- **Computational efficiency**: Far cheaper than 3D Navier-Stokes simulations

## Numerical Methods

### Spatial Discretization

PyBurgers uses **Fourier collocation** for computing spatial derivatives. Since the velocity field is real-valued, the code uses real FFTs (rfft/irfft) for approximately 2× speedup and 50% memory reduction compared to complex FFTs.

Key features:
- Spectral accuracy for smooth solutions
- Efficient computation of derivatives of any order
- 2× padding for dealiasing nonlinear terms

### Time Integration

The solver employs **second-order Adams-Bashforth** time stepping:
- Explicit method suitable for the problem structure
- Good stability properties with reasonable CFL constraints
- Efficient for spectral methods

### Stochastic Forcing

The stochastic term uses **fractional Brownian motion (FBM)** noise:
- Spectral exponent α (typically 0.75) controls the correlation structure
- Amplitude controls the energy injection rate
- Generated in spectral space for efficiency

## Simulation Modes

### Direct Numerical Simulation (DNS)

DNS resolves all scales of motion down to the Kolmogorov scale. In PyBurgers:
- Typically uses 8192 grid points
- No subgrid-scale modeling
- Provides reference solutions for validating LES

### Large-Eddy Simulation (LES)

LES resolves large-scale motions and models the effect of small scales. PyBurgers supports four subgrid-scale (SGS) models:

1. **Constant-coefficient Smagorinsky** (model: 1)
    - Classic eddy-viscosity model
    - Single tunable coefficient

2. **Dynamic Smagorinsky** (model: 2)
   - Coefficient computed dynamically from resolved scales
   - Better adaptation to flow conditions

3. **Dynamic Wong-Lilly** (model: 3)
   - Alternative dynamic procedure
   - Different averaging approach

4. **Deardorff 1.5-order TKE** (model: 4)
   - Prognostic equation for subgrid turbulence kinetic energy
   - More sophisticated closure

## Performance Optimizations

### FFTW Wisdom Caching

PyBurgers includes an intelligent FFTW wisdom system:
- **First run**: Generates optimal FFT plans and saves them to `~/.pyburgers_fftw_wisdom`
- **Subsequent runs**: Loads pre-computed plans for instant startup
- **File locking**: Prevents corruption when running multiple instances
- **Validation**: Ensures cached plans match current configuration

Planning levels (set via `fftw.planning` in namelist):
- `FFTW_ESTIMATE`: Fastest planning, decent performance
- `FFTW_MEASURE`: Good balance (default)
- `FFTW_PATIENT`: Thorough search, better performance
- `FFTW_EXHAUSTIVE`: Extremely thorough (slow planning, best performance)

### Multithreading

FFT operations can use multiple threads (set via `fftw.threads` in namelist). Optimal thread count depends on grid size and CPU architecture.

## Output

Simulations write results to NetCDF files with:
- Time series of velocity fields
- Diagnostic quantities (energy, dissipation, etc.)
- Comprehensive metadata (simulation parameters, timestamps, etc.)
- CF-compliant conventions for compatibility with analysis tools

## Getting Started

Ready to run your first simulation? Head over to the [Getting Started](getting-started.md) guide.

For detailed configuration options, see the [Namelist Reference](namelist.md).

For API documentation, check the [API Reference](reference.md).

## References

Basu, S. (2009). High-resolution large-eddy simulations of stably stratified flows: application to the Cooperative Atmosphere–Surface Exchange Study 1999 (CASES-99). *Journal of Turbulence*, 10, N12. https://doi.org/10.1080/14685240902852719
