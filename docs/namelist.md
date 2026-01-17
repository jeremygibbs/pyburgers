# Namelist

This page documents the namelist configuration format and available options.

## Source files

- `namelist.json`: example configuration
- `pyburgers/schema_namelist.json`: JSON schema used for validation

## Example

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

## Reference

Required top-level sections: `time`, `physics`, `grid`, `output`, `logging`, `fftw`.

### time

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| nt | none | none | yes |
| dt | none | none | yes |

### physics

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| viscosity | none | none | yes |
| sgs_model | Subgrid-scale model selector. | 0: No SGS model<br>1: Constant Smagorinsky<br>2: Dynamic Smagorinsky<br>3: Dynamic Wong-Lilly<br>4: Deardorff 1.5-order TKE | no |

#### physics.noise

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| alpha | none | none | yes |
| amplitude | none | none | yes |

### grid

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| domain_length | none | none | no |

#### grid.dns

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| nx | none | none | yes |

#### grid.les

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| nx | none | none | yes |

### output

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| t_save | Output interval in seconds. | none | yes |

### logging

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| level | Logging severity. | DEBUG: Verbose diagnostics.<br>INFO: Normal runtime info.<br>WARNING: Potential issues.<br>ERROR: Errors that stop work.<br>CRITICAL: Serious failures. | yes |
| file | none | none | no |

### fftw

| Key | Description | Options | Required |
| --- | --- | --- | --- |
| planning | none | FFTW_ESTIMATE<br>FFTW_MEASURE<br>FFTW_PATIENT<br>FFTW_EXHAUSTIVE | yes |
| threads | none | none | yes |
