# Getting Started

## Install for docs

```bash
python -m pip install -e ".[docs]"
```

## Run a simulation

```bash
python burgers.py -m dns
python burgers.py -m les -o output.nc
```

## Build the docs

```bash
mkdocs build
```

## Preview the docs

```bash
mkdocs serve
```
