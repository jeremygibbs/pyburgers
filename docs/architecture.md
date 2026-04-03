# Architecture

This page provides visual diagrams showing how PyBurgers is structured and how the simulation flows from start to finish.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph entry[Entry Point]
        CLI[burgers.py<br/>CLI Interface]
    end

    subgraph config[Configuration]
        NL[namelist.json<br/>User Config]
        Schema[schema_namelist.json<br/>Validation Rules]
        Input[Input Class<br/>Parser & Validator]
    end

    subgraph core[Core Simulation]
        Base[Burgers<br/>Abstract Base Class]
        DNS[DNS Class<br/>Direct Simulation]
        LES[LES Class<br/>Large-Eddy Simulation]
    end

    subgraph numerics[Numerical Methods]
        TI[TimeIntegrator Base]
        AB2[AB2<br/>Adams-Bashforth 2]
        AM2[AM2<br/>Adams-Moulton 2 PC]
        RK3[RK3<br/>Low-Storage RK3]
    end

    subgraph physics[Physics Models]
        SGS[SGS Base Class]
        Smag[Smagorinsky Models]
        Wong[Wong-Lilly Model]
        Dear[Deardorff TKE Model]
    end

    subgraph utils[Utilities]
        Spectral[SpectralWorkspace<br/>FFT Operations]
        Deriv[Derivatives]
        Filter[Filter]
        Dealias[Dealias]
        FBM[FBM Noise]
        FFTW[FFTW Wisdom Cache]
    end

    subgraph output[Output]
        Output[Output Class<br/>NetCDF Writer]
        NC[pyburgers_*.nc<br/>Results]
    end

    CLI --> NL
    NL --> Input
    Schema --> Input
    Input --> DNS
    Input --> LES
    DNS --> Base
    LES --> Base
    LES --> SGS
    Base --> TI
    TI --> AB2
    TI --> AM2
    TI --> RK3
    SGS --> Smag
    SGS --> Wong
    SGS --> Dear
    Base --> Spectral
    Spectral --> Deriv
    Spectral --> Filter
    Spectral --> Dealias
    Spectral --> FBM
    Spectral --> FFTW
    Base --> Output
    Output --> NC

    style CLI fill:#e1f5ff
    style Base fill:#fff4e1
    style DNS fill:#e8f5e9
    style LES fill:#e8f5e9
    style Spectral fill:#f3e5f5
    style TI fill:#fce4ec
    style Output fill:#ffe0b2
```

## Simulation Execution Flow

```mermaid
flowchart TD
    Start([Start]) --> Parse[Parse CLI Arguments<br/>dns or les mode]
    Parse --> Load[Load namelist.json]
    Load --> Validate[Validate Configuration<br/>against schema]
    Validate --> Setup[Setup Logging]
    Setup --> LoadWisdom{Load FFTW<br/>Wisdom?}

    LoadWisdom -->|Cached| Reuse[Reuse Cached Plans]
    LoadWisdom -->|None/Mismatch| Warmup[Warmup FFTW Plans]

    Reuse --> InitSolver
    Warmup --> SaveWisdom[Save Wisdom Cache]
    SaveWisdom --> InitSolver

    InitSolver[Initialize Solver<br/>DNS or LES] --> CreateWorkspace[Create SpectralWorkspace<br/>FFT buffers & plans]
    CreateWorkspace --> InitSGS{LES Mode?}

    InitSGS -->|Yes| LoadSGS[Load SGS Model<br/>0-4]
    InitSGS -->|No| InitIC
    LoadSGS --> InitIC

    InitIC[Initialize Velocity Field<br/>Random + low-k energy] --> CreateOutput[Create NetCDF Output]
    CreateOutput --> SelectIntegrator{Integrator}

    SelectIntegrator -->|1| UseAB2[AB2 Integrator]
    SelectIntegrator -->|2| UseAM2[AM2 Integrator]
    SelectIntegrator -->|3| UseRK3[RK3 Integrator]

    UseAB2 --> TimeLoop
    UseAM2 --> TimeLoop
    UseRK3 --> TimeLoop

    TimeLoop{t < duration?}

    TimeLoop -->|Yes| TimeStep[Time Step<br/>via Integrator]
    TimeStep --> ComputeCFL[Compute CFL<br/>Adaptive Δt]
    ComputeCFL --> CheckSave{Time to<br/>save?}

    CheckSave -->|Yes| WriteNC[Write to NetCDF]
    CheckSave -->|No| CheckPrint
    WriteNC --> CheckPrint{Time to<br/>print?}

    CheckPrint -->|Yes| LogProgress[Log Progress]
    CheckPrint -->|No| TimeLoop
    LogProgress --> TimeLoop

    TimeLoop -->|No| CloseOutput[Close NetCDF File]
    CloseOutput --> End([End])

    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style TimeStep fill:#fff4e1
    style SelectIntegrator fill:#fce4ec
    style WriteNC fill:#e8f5e9
    style InitSolver fill:#f3e5f5
```

## RK3 Stage Details (Integrator ID 3)

Each Runge-Kutta stage computes the right-hand side (RHS) of the Burgers equation:

```mermaid
flowchart TD
    Start([Stage k]) --> Noise[Generate FBM Noise<br/>Stochastic forcing]
    Noise --> FFT1[FFT: u → û]
    FFT1 --> Deriv[Compute Derivatives<br/>∂u/∂x, ∂²u/∂x²]
    Deriv --> Dealias[Compute u²<br/>with dealiasing]
    Dealias --> NonLin[Compute ∂u²/∂x]
    NonLin --> SGSCheck{LES Mode?}
    SGSCheck -->|Yes| ComputeSGS[Compute SGS Stress<br/>τ = -2ν_t ∂u/∂x]
    SGSCheck -->|No| BuildRHS
    ComputeSGS --> BuildRHS[Build RHS:<br/>F = -∂u²/∂x + ν∂²u/∂x² + ∂τ/∂x + noise]
    BuildRHS --> UpdateU[Update u<br/>using RK3 coefficients]
    UpdateU --> ZeroNyquist[Zero Nyquist Mode]
    ZeroNyquist --> IFFT[IFFT: û → u]
    IFFT --> End([Next Stage])

    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style ComputeSGS fill:#fff4e1
    style Dealias fill:#ffe0b2
```

## Class Hierarchy

```mermaid
classDiagram
    class Burgers {
        <<abstract>>
        +SpectralWorkspace workspace
        +TimeIntegrator integrator
        +Input input_obj
        +Output output
        +run()
        +_compute_rhs()*
        +_initialize_velocity()
        +_compute_dt()
    }

    class DNS {
        +_compute_rhs()
        Override: No SGS model
    }

    class LES {
        +SGS sgs_model
        +_compute_rhs()
        Override: Includes SGS stress
    }

    class TimeIntegrator {
        <<abstract>>
        +step()*
        +get_integrator()$
    }

    class RK3 {
        +Q: array
        +step()
        3-stage low-storage
    }

    class AB2 {
        +rhs_prev: array
        +step()
        2nd-order multistep
    }

    class AM2 {
        +rhs_prev: array
        +rhs_n: array
        +u_save: array
        +step()
        2nd-order predictor-corrector
    }

    class SGS {
        <<abstract>>
        +compute()*
        +get_model()$
    }

    class SmagConstant {
        +Cs: float
        +compute()
        Constant coefficient
    }

    class SmagDynamic {
        +compute()
        Dynamic coefficient
    }

    class WongLilly {
        +compute()
        Dynamic with scale-similarity
    }

    class Deardorff {
        +tke_sgs: array
        +compute()
        1.5-order TKE equation
    }

    class SpectralWorkspace {
        +Derivatives deriv
        +Dealias dealias
        +Filter filter
        +FBM noise
    }

    Burgers <|-- DNS
    Burgers <|-- LES
    Burgers *-- SpectralWorkspace
    Burgers *-- TimeIntegrator
    TimeIntegrator <|-- AB2
    TimeIntegrator <|-- AM2
    TimeIntegrator <|-- RK3
    LES *-- SGS
    SGS <|-- SmagConstant
    SGS <|-- SmagDynamic
    SGS <|-- WongLilly
    SGS <|-- Deardorff
```

## Data Flow in SpectralWorkspace

```mermaid
flowchart TB
    subgraph "Physical Space"
        U[u: velocity field<br/>Real array nx]
    end

    subgraph "Spectral Operations"
        FFT[FFT: rfft<br/>pyfftw]
        UHat[û: velocity spectrum<br/>Complex array nx/2+1]

        D1[∂u/∂x<br/>Multiply by ik]
        D2[∂²u/∂x²<br/>Multiply by -k²]
        D3[∂³u/∂x³<br/>Multiply by -ik³]

        Filt[Filter<br/>Spectral cutoff]
        Deal[Dealiasing<br/>3/2 padding]
    end

    subgraph "Back to Physical"
        IFFT[IFFT: irfft<br/>pyfftw]
        Result[Derivatives in x-space]
    end

    U -->|Forward| FFT
    FFT --> UHat
    UHat --> D1
    UHat --> D2
    UHat --> D3
    UHat --> Filt
    UHat --> Deal

    D1 --> IFFT
    D2 --> IFFT
    D3 --> IFFT
    Filt --> IFFT
    Deal --> IFFT

    IFFT --> Result

    style U fill:#e1f5ff
    style UHat fill:#fff4e1
    style Result fill:#e8f5e9
```

## FFTW Wisdom Caching Strategy

```mermaid
flowchart TD
    Start([Initialization]) --> CheckFile{Wisdom file<br/>exists?}

    CheckFile -->|No| Warmup[Run Warmup:<br/>Create all plans]
    CheckFile -->|Yes| Lock[Acquire File Lock]

    Lock --> Load[Load & Import Wisdom]
    Load --> Import[Reuse Cached Plans<br/>Missing sizes planned on-demand]

    Import --> Continue([Continue])

    Warmup --> Save[Save Wisdom<br/>cumulative across runs]
    Save --> Continue

    style CheckFile fill:#e1f5ff
    style Import fill:#e8f5e9
    style Warmup fill:#fff4e1
    style Continue fill:#e1f5ff
```

## SGS Model Selection

```mermaid
flowchart TD
    Start([LES Mode]) --> ReadConfig[Read subgrid_model<br/>from namelist]
    ReadConfig --> Factory{Model ID}

    Factory -->|0| NoModel[No SGS Model<br/>τ = 0]
    Factory -->|1| SmagC[Constant Smagorinsky<br/>C_s = 0.16]
    Factory -->|2| SmagD[Dynamic Smagorinsky<br/>C_s computed]
    Factory -->|3| WongL[Wong-Lilly<br/>Dynamic + similarity]
    Factory -->|4| Dear[Deardorff TKE<br/>1.5-order closure]

    NoModel --> Compute[compute method]
    SmagC --> Compute
    SmagD --> Compute
    WongL --> Compute
    Dear --> Compute

    Compute --> Return[Return τ, C_s<br/>Optional: TKE]
    Return --> RHS[Add ∂τ/∂x to RHS]

    style Start fill:#e1f5ff
    style Factory fill:#fff4e1
    style Return fill:#e8f5e9
```

## Key Design Patterns

### Abstract Base Class Pattern
- `Burgers` defines common interface and time-stepping logic
- `DNS` and `LES` implement mode-specific RHS computation
- Eliminates code duplication while allowing specialization

### Factory Pattern
- `SGS.get_model()` creates appropriate SGS model instance
- `TimeIntegrator.get_integrator()` creates appropriate time integration scheme
- Centralizes model/scheme selection logic
- Easy to add new SGS models or time integrators

### Workspace Pattern
- `SpectralWorkspace` bundles all spectral operations
- Pre-allocates FFTW buffers for efficiency
- Provides clean interface to complex FFT operations

### Caching Pattern
- FFTW wisdom is cumulative across runs and configurations
- First run: slow planning, subsequent runs: instant
- File locking prevents concurrent access conflicts
