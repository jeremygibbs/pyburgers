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

Preview the documentation site locally:

```bash
mkdocs serve
```

Then visit http://127.0.0.1:8000 in your browser.

## Tests and lint

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format
```
