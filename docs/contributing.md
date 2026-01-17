# Contributing

## Local development

Install runtime dependencies to run the solver from the repo:

```bash
python -m pip install -e .
```

## Documentation

Install the docs toolchain (no package install required):

```bash
python -m pip install mkdocs mkdocs-material mkdocstrings[python]
```

Regenerate the namelist reference, then preview the site:

```bash
python scripts/build_namelist_docs.py
mkdocs serve
```

## Tests and lint

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format
```
