# Contributing to PyBurgers

Thank you for your interest in contributing to PyBurgers!

## Quick Start

For detailed contributing guidelines, please see our full [Contributing Guide](https://docs.gibbs.science/pyburgers/contributing/).

## Essential Information

### Code Style
This project uses `ruff` for linting and formatting. Please run these before submitting a PR:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format .
```

### Testing
Ensure all tests pass:

```bash
pytest
```

### Documentation
Use **Google-style docstrings** for all code. Documentation is auto-generated from docstrings using MkDocs.

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with clear commit messages
4. Update docstrings and add tests
5. Run tests and linting
6. Submit a pull request

## Resources

- [Full Documentation](https://docs.gibbs.science/pyburgers/)
- [Getting Started Guide](https://docs.gibbs.science/pyburgers/getting-started/)
- [API Reference](https://docs.gibbs.science/pyburgers/reference/)

## Questions?

Open an [issue](https://github.com/jeremygibbs/pyburgers/issues) on GitHub if you have questions or need clarification.
