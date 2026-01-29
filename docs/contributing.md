# Contributing

Thank you for your interest in contributing to PyBurgers! This guide will help you get set up for development and understand the project's workflow.

## Development Setup

### Install Dependencies

Install runtime dependencies to run the solver from the repo:

```bash
python -m pip install -e .
```

### Install Development Tools

For testing and linting:

```bash
python -m pip install -e ".[dev]"
```

This installs:
- `pytest` and `pytest-cov` for testing
- `ruff` for linting and formatting

### Install Documentation Tools

To build and preview documentation locally:

```bash
python -m pip install -e ".[docs]"
```

This installs:
- `mkdocs` - Documentation generator
- `mkdocs-material` - Material theme
- `mkdocstrings[python]` - Auto-generate API docs from docstrings

## Code Style

This project uses **Ruff** for both linting and formatting. Before submitting a pull request, ensure your code passes all checks:

```bash
# Check for linting issues
ruff check .

# Auto-format code
ruff format .
```

### Style Guidelines
- Generally follow the [Google Python style guide](https://google.github.io/styleguide/pyguide.html) 
- Line length: 100 characters (enforced by Ruff)
- Follow PEP 8 conventions
- Use type hints for function signatures
- Use descriptive variable names

## Documentation

### Docstring Style

PyBurgers uses [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). All public functions, classes, and methods should have comprehensive docstrings.

**Example:**

```python
def compute_derivative(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Compute the first derivative using spectral methods.

    Args:
        u: Velocity field in physical space.
        k: Wavenumber array.

    Returns:
        First derivative of u in physical space.

    Raises:
        ValueError: If array shapes don't match.
    """
    # implementation
```

### What to Document

- **Modules**: Add a module-level docstring explaining the module's purpose
- **Classes**: Purpose, key attributes, usage examples
- **Functions/Methods**: What it does, parameters, return values, exceptions
- **Complex algorithms**: Add inline comments explaining non-obvious logic

### Building Documentation Locally

Preview the documentation site locally:

```bash
# Serve with live reload at http://127.0.0.1:8000
mkdocs serve

# Build static site to site/ directory
mkdocs build
```

### Documentation Structure

- `docs/index.md` - Landing page with overview and scientific background
- `docs/getting-started.md` - Installation and first simulation guide
- `docs/namelist.md` - Complete namelist parameter reference
- `docs/reference.md` - API reference (auto-generated from docstrings)
- `docs/contributing.md` - This file

## Testing

### Running Tests

Run the full test suite:

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=pyburgers --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Use descriptive test names that explain what is being tested
- Add tests for new features and bug fixes
- Ensure tests are independent and can run in any order

**Example:**

```python
def test_derivative_accuracy():
    """Test that spectral derivatives are accurate for smooth functions."""
    # Test implementation
```

## Submitting Changes

### Workflow

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pyburgers.git
   cd pyburgers
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-feature
   ```
4. **Make your changes** with clear, focused commits
5. **Update documentation** and add tests
6. **Run tests and linting**:
   ```bash
   pytest
   ruff check .
   ruff format .
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```
8. **Open a pull request** on GitHub

### Commit Messages

Write clear, concise commit messages:

- Use present tense ("Add feature" not "Added feature")
- First line should be a short summary (50 chars or less)
- Add a blank line, then detailed explanation if needed
- Reference issue numbers when applicable

**Good examples:**
```
Add dynamic Smagorinsky SGS model

Implement the dynamic procedure following Germano et al. (1991)
for computing the Smagorinsky coefficient adaptively.

Fixes #42
```

```
Fix FFTW wisdom cache corruption

Prevents race condition when multiple processes attempt to write
wisdom simultaneously by using file locking.
```

### Pull Request Guidelines

- Fill out the PR template completely
- Link to related issues
- Ensure all CI checks pass
- Respond to review feedback promptly
- Keep PRs focused on a single feature/fix
- Update documentation if your changes affect user-facing behavior

## Development Tips

### Testing Different Configurations

Create test namelists for quick iterations:

```json
{
    "time": { "duration": 0.1, "cfl": 0.4, "max_step": 0.01 },
    "grid": { "dns": { "points": 256 }, "les": { "points": 64 } },
    "fftw": { "planning": "FFTW_ESTIMATE", "threads": 2 }
}
```

### Debugging

Use Python's built-in debugger:

```python
import pdb; pdb.set_trace()
```

Or use `logging` with DEBUG level:

```json
{
    "logging": { "level": "DEBUG", "file": "debug.log" }
}
```

### Performance Profiling

Use `cProfile` to identify bottlenecks:

```bash
python -m cProfile -o profile.stats burgers.py -m dns
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

## Getting Help

- **Documentation**: [docs.gibbs.science/pyburgers](https://docs.gibbs.science/pyburgers/)
- **Issues**: [github.com/jeremygibbs/pyburgers/issues](https://github.com/jeremygibbs/pyburgers/issues)
- **Discussions**: [github.com/jeremygibbs/pyburgers/discussions](https://github.com/jeremygibbs/pyburgers/discussions)

## Code of Conduct

Do not be a jerk.
