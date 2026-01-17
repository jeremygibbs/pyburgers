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

| Key | Description | Type | Required |
| --- | --- | --- | --- |
| time.nt | none | integer | yes |
| time.dt | none | number | yes |

### physics

| Key | Description | Type | Required |
| --- | --- | --- | --- |
| physics.noise.alpha | none | number | yes |
| physics.noise.amplitude | none | number | yes |
| physics.viscosity | none | number | yes |
| physics.sgs_model<br>options:<br>0: No SGS model<br>1: Constant Smagorinsky<br>2: Dynamic Smagorinsky<br>3: Dynamic Wong-Lilly<br>4: Deardorff 1.5-order TKE | Subgrid-scale model selector. | integer | no |

### grid

| Key | Description | Type | Required |
| --- | --- | --- | --- |
| grid.domain_length | none | number | no |
| grid.dns.nx | none | integer | yes |
| grid.les.nx | none | integer | yes |

### output

| Key | Description | Type | Required |
| --- | --- | --- | --- |
| output.t_save | Output interval in seconds. | number | yes |

### logging

| Key | Description | Type | Required |
| --- | --- | --- | --- |
| logging.level<br>options:<br>DEBUG: Verbose diagnostics.<br>INFO: Normal runtime info.<br>WARNING: Potential issues.<br>ERROR: Errors that stop work.<br>CRITICAL: Serious failures. | Logging severity. | string | yes |
| logging.file | none | string | no |

### fftw

| Key | Description | Type | Required |
| --- | --- | --- | --- |
| fftw.planning<br>options:<br>FFTW_ESTIMATE<br>FFTW_MEASURE<br>FFTW_PATIENT<br>FFTW_EXHAUSTIVE | none | string | yes |
| fftw.threads | none | integer | yes |
