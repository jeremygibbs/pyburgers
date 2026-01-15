![Image of pyburgers logo](https://gibbs.science/img/pyburgers_v2.png)

# Burgers Turbulence (Burgulence)
* Originally conceived by Dutch scientist, J.M. Burgers in the 1930s
* One of the first attempts to arrive at the statistical theory of turbulent fluid motion
* The original equation shares a lot in common with the Navier-Stokes (N-S) equations:
  * Advective non-linearity, diffusion, invariance, and conservation laws
* This equation is not an ideal model for the chaotic nature of turbulence
  * Can be integrated explicitly, meaning it is not sensitive to small changes in initial conditions
  * Shock waves form in the limit of vanishing viscosity
* A popular modification is the addition of a forcing term that accounts for the neglected effects
  * An example is perturbing the system with a stochastic process that is stationary in time/space
* A popular version is called the 1D Stochastic Burgers Equation (1D SBE)
  * Allows insight into turbulence without having to generalize to 3D
* 1D SBE shares characteristics of 3D turbulence
  * nonlinearity, energy spectrum, intermittent energy dissipation
* 1D SBE is computationally cheap

# PyBurgers
* Many solutions exist for the 1D SBE
* PyBurgers follows the procedures in [Basu (2009)](https://doi.org/10.1080/14685240902852719)
* Fourier methods are used in space, and time is advanced in real space
  * Fourier collocation for spatial derivatives, 2nd-order Adams-Bashforth in time
* Offers a direct numerical simulation (DNS) mode
* Offers a large-eddy simulation (LES) mode, with 4 subgrid-scale (SGS) models
  * Constant-coefficient Smagorinsky
  * Dyanmic Smagorinsky
  * Dynamic Wong-Lilly
  * Deardorff 1.5-order TKE
* Stochastic term is fractional Brownian motion (FBM) noise
* Users can set how agressively they want to optimize FFT computations
* Output in NetCDF
* DNS took 70 minutes on a 2019 iMac
* LES took 62 minutes on a 2019 iMac

# Namelist Settings
* nx: number of grid points in the x-direction
* nt: number of time steps
* dt: time step (s)
* visc: kinematic viscosity (m^2 s^-3)
* damp: noise amplitude for FBM noise
* sgs: subgrid-scale model
  * 0 = no model
  * 1 = constant-coefficient Smagorinsky
  * 2 = dynamic Smagorinsky
  * 3 = dynamic Wong-Lilly
  * 4 = Deardorff 1.5-order TKE

# Requirements
* PyBurgers requires Python 3, NumPy, json, and netCDF4

# How to run
```
python burgers.py -m [dns, les] -o [optional output name]
```

# Disclaimer
I may have made errors! If you find one, [let me know][2]!

# License 
This template is free source code. It comes without any warranty, to the extent permitted by applicable law. You can redistribute it and/or modify it under the terms of the Do What The Fuck You Want To Public License, Version 2, as published by Sam Hocevar. See [http://www.wtfpl.net][1] for more details.

[1]: http://www.wtfpl.net
[2]: mailto:jeremy.gibbs@noaa.gov
