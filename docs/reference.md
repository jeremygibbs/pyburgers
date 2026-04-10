# API Reference

## Overview

::: pyburgers
    options:
      members: false
      show_root_heading: false

## Core Solvers

::: pyburgers.core.Burgers

::: pyburgers.dns.DNS

::: pyburgers.les.LES

## Numerical Methods

### Temporal Integrators

::: pyburgers.numerics.temporal.temporal.TemporalIntegrator

::: pyburgers.numerics.temporal.temporal_ab2.AB2

::: pyburgers.numerics.temporal.temporal_am2.AM2

::: pyburgers.numerics.temporal.temporal_rk3.RK3

### Spatial Operators

::: pyburgers.numerics.spatial.spatial.SpatialOperator

::: pyburgers.numerics.spatial.spatial_fd2.FD2

::: pyburgers.numerics.spatial.spatial_fd4.FD4

::: pyburgers.numerics.spatial.spatial_spectral.Spectral

## Data Models

::: pyburgers.data_models

## Exceptions

::: pyburgers.exceptions

## Utilities

::: pyburgers.utils.spectral.Derivatives

::: pyburgers.utils.spectral.Dealias

::: pyburgers.utils.spectral.Filter

::: pyburgers.utils.spectral_workspace.SpectralWorkspace

::: pyburgers.utils.fftw

::: pyburgers.utils.fbm.FBM

::: pyburgers.utils.constants
    options:
      show_if_no_docstring: true
      members: true

::: pyburgers.utils.logging_helper

## Input/Output

::: pyburgers.utils.io.input.Input

::: pyburgers.utils.io.output.Output

## Physics

::: pyburgers.physics.sgs.sgs.SGS

::: pyburgers.physics.sgs.sgs_smagcon.SmagConstant

::: pyburgers.physics.sgs.sgs_smagdyn.SmagDynamic

::: pyburgers.physics.sgs.sgs_wonglilly.WongLilly

::: pyburgers.physics.sgs.sgs_deardorff.Deardorff
