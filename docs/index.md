# PyBurgers

PyBurgers solves the 1D stochastic Burgers equation with DNS and LES modes. It
uses Fourier collocation in space and Adams-Bashforth time integration.

## Quick start

```bash
python -m pip install -e .
python burgers.py -m dns
```

## Docs workflow

The API reference is built from docstrings with `mkdocstrings`. See
[Contributing](contributing.md) for docs tooling and commands.
