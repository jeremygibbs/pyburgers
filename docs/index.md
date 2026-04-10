# PyBurgers Documentation

Welcome to the documentation for PyBurgers, a high-performance solver for the 1D Stochastic Burgers Equation.

## What is PyBurgers?

PyBurgers provides both direct numerical simulation (DNS) and large-eddy simulation (LES) capabilities for studying Burgers turbulence. The solver offers pluggable spatial discretization (2nd-order FD, 4th-order FD, or Fourier collocation) and pluggable time integration (Adams-Bashforth 2, Adams-Moulton 2 predictor-corrector, or Williamson 1980 low-storage RK3) with CFL-based adaptive time stepping. Many settings follow or are inspired by the procedures described in [Basu (2009)](https://doi.org/10.1080/14685240902852719).

## Scientific Background

### The Burgers Equation

The Burgers equation was originally conceived by Dutch scientist J.M. Burgers in the 1930s as one of the first attempts to arrive at a statistical theory of turbulent fluid motion. 

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

This represents a very simplified model that describes the interaction of non-linear inertial terms and dissipation in the motion of a fluid. This original equation shares many characteristics with the Navier-Stokes equations: **advective non-linearity**, **diffusion**, **invariance**, and **conservation.**

### Limitations and Extensions

While the Burgers equation is not an ideal model for the chaotic nature of turbulence (it can be integrated explicitly and is not sensitive to small changes in initial conditions), it remains valuable for understanding turbulent dynamics. Shock waves form in the limit of vanishing viscosity, providing insight into energy dissipation mechanisms.

A popular modification is the addition of a forcing term that accounts for neglected effects. By perturbing the system with a stochastic process that is stationary in time and space, we obtain the **1D Stochastic Burgers Equation (1D SBE)**.

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} + \eta(x,t)$$

* In the above equation, the "new" term is $\eta$ -- which is called the stochastic term
* $\eta(x,t)$ should be white noise in time, but spatially correlated

Here we use fractional Brownian noise (FBM):

$$\eta(x,t) = \sqrt{\frac{2D_0}{\Delta t}} \mathfrak{F}^{-1} \left\lbrace|k|^{\beta/2}\hat{f}(k)\right\rbrace$$

- $D_0$ = noise amplitude ($10^{-6}$ by default)
- $\Delta t$ = time step
- $\mathfrak{F}^{-1}$ =inverse Fourier transform
- $f$ = Gaussian random noise with mean = 0, and standard deviation = $\sqrt{N}$ (where $N$ is the number of points)
- $\beta$ = spectral slope of the noise ($-0.75$ by default)

### Why Study the 1D SBE?

The 1D SBE provides valuable insights into turbulence without the computational burden of generalizing to 3D:

- **Nonlinearity**: Captures the essence of turbulent energy transfer
- **Energy spectrum**: Exhibits spectral characteristics similar to 3D turbulence
- **Intermittent dissipation**: Shows bursts of energy dissipation
- **Computational efficiency**: Far cheaper than 3D Navier-Stokes simulations

## Numerical Methods

### Spatial Discretization

PyBurgers supports three pluggable spatial discretization schemes, selected via `numerics.spatial` in the namelist:

- **`1` - 2nd-order central finite differences (FD2)**: Simple, robust stencils for all derivative orders
- **`2` - 4th-order central finite differences (FD4)**: Higher accuracy with wider stencils
- **`3` - Spectral (Fourier collocation)** (default): Exponential accuracy for smooth solutions

All schemes use real FFTs (rfft/irfft) for approximately 2x speedup and 50% memory reduction compared to complex FFTs. Nonlinear terms are dealiased using the 3/2-rule (zero-padding).

### Time Integration

PyBurgers supports three pluggable time integration schemes, selected via `numerics.temporal` in the namelist:

- **`1` - Adams-Bashforth 2nd order (AB2)**: 2nd-order multistep, 1 RHS evaluation per step, bootstrapped with forward Euler
- **`2` - Adams-Moulton 2nd order predictor-corrector (AM2)**: 2nd-order PECE scheme (AB2 predictor + trapezoidal corrector), 2 RHS evaluations per step
- **`3` - Williamson (1980) low-storage RK3** (default): 3rd-order, 3 stages per step, self-starting

All schemes use CFL-based adaptive time stepping that automatically adjusts dt based on the maximum velocity, viscous stability, and hyperviscous stability constraints. Output times are hit exactly by clamping dt to reach save intervals. Nyquist mode is zeroed after each integration stage to prevent aliasing accumulation.

### Hyperviscosity

Spectral methods can exhibit energy pile-up near the Nyquist frequency, where energy accumulates at the highest resolved wavenumbers instead of being properly dissipated. PyBurgers automatically applies **hyperviscosity** to all spatial schemes:

$$\frac{\partial u}{\partial t} = \ldots - \nu_4 \frac{\partial^4 u}{\partial x^4}$$

The hyperviscosity term provides $k^4$ dissipation that strongly damps high-wavenumber modes while leaving large scales essentially unaffected. The coefficient is **automatically computed** and **normalized per scheme** so that the Nyquist damping rate is equal across all spatial discretizations:

$$\nu_4 = \frac{\pi^4 \Delta x^4}{\lambda_\text{hypervisc}}$$

where $\lambda_\text{hypervisc}$ is the scheme's maximum modified wavenumber magnitude $\max|\tilde{k}^4| \cdot \Delta x^4$:

| Scheme | $\lambda_\text{hypervisc}$ | $\nu_4$ |
|--------|--------------------------|---------|
| FD2 (`spatial: 1`) | 16 | $\approx 6.09\,\Delta x^4$ |
| FD4 (`spatial: 2`) | 80/3 | $\approx 3.65\,\Delta x^4$ |
| Spectral (`spatial: 3`) | $\pi^4$ | $\Delta x^4$ |

This normalization ensures:

- **Equal Nyquist damping rate**: Every scheme damps its highest resolved mode at the same rate
- **Scheme-independent timestep limit**: The hyperviscous stability constraint reduces to $dt \le C/\pi^4$ for all schemes
- **Resolution-independent behavior**: Scaling with $\Delta x^4$ is consistent across grid resolutions

The active coefficient is logged at startup for reference.

### Stochastic Forcing

The stochastic term uses **fractional Brownian motion (FBM)** noise:

- Spectral exponent $\beta$ (default $-0.75$) controls the correlation structure
- Amplitude $D_0$ (default $10^{-6}$) controls the energy injection rate
- Generated in spectral space for efficiency

## Simulation Modes

### Direct Numerical Simulation (DNS)

DNS resolves all scales of motion down to the Kolmogorov scale. In PyBurgers:

- Typical setup uses 8192 grid points
- No subgrid-scale modeling
- Provides reference solutions for validating LES

### Large-Eddy Simulation (LES)

LES resolves large-scale motions and models the effect of small scales. PyBurgers includes four subgrid-scale (SGS) models:

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

PyBurgers easily supports more SGS models through extending the related class system.

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

FFT operations can use multiple threads (set via `fftw.threads` in namelist). Optimal thread count depends on grid size and CPU architecture. Note, for smaller problems, threading overhead can cause a degradation in performance. Users should tune to their respective systems.

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

Williamson, J.H. (1980). Low-storage Runge-Kutta schemes. *Journal of Computational Physics*, 35(1), 48-56. https://doi.org/10.1016/0021-9991(80)90033-9
