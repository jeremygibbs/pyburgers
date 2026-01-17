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

| Key | Type | Required | Description | Constraints |
| --- | --- | --- | --- | --- |
| time.nt | integer | yes | none | minimum: 1 |
| time.dt | number | yes | none | exclusiveMinimum: 0 |

### physics

| Key | Type | Required | Description | Constraints |
| --- | --- | --- | --- | --- |
| physics.noise.alpha | number | yes | none | none |
| physics.noise.amplitude | number | yes | none | exclusiveMinimum: 0 |
| physics.viscosity | number | yes | none | exclusiveMinimum: 0 |
| physics.sgs_model<br>options:<br>0: No SGS model<br>1: Constant Smagorinsky<br>2: Dynamic Smagorinsky<br>3: Dynamic Wong-Lilly<br>4: Deardorff 1.5-order TKE | integer | no | Subgrid-scale model selector. | minimum: 0; maximum: 4 |

### grid

| Key | Type | Required | Description | Constraints |
| --- | --- | --- | --- | --- |
| grid.domain_length | number | no | none | exclusiveMinimum: 0 |
| grid.dns.nx | integer | yes | none | minimum: 1 |
| grid.les.nx | integer | yes | none | minimum: 1 |

### output

| Key | Type | Required | Description | Constraints |
| --- | --- | --- | --- | --- |
| output.t_save | number | yes | Output interval in seconds. | exclusiveMinimum: 0 |

### logging

| Key | Type | Required | Description | Constraints |
| --- | --- | --- | --- | --- |
| logging.level<br>options:<br>DEBUG: Verbose diagnostics.<br>INFO: Normal runtime info.<br>WARNING: Potential issues.<br>ERROR: Errors that stop work.<br>CRITICAL: Serious failures. | string | yes | Logging severity. | none |
| logging.file | string | no | none | none |

### fftw

| Key | Type | Required | Description | Constraints |
| --- | --- | --- | --- | --- |
| fftw.planning<br>options:<br>FFTW_ESTIMATE<br>FFTW_MEASURE<br>FFTW_PATIENT<br>FFTW_EXHAUSTIVE | string | yes | none | none |
| fftw.threads | integer | yes | none | minimum: 1 |
