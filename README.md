# Learnability on Information-theoretic Continuum

Research on learnability on information-theoretic continuum.

## Setup

### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

```bash
# Install dependencies
uv sync

# Install with development dependencies (includes ruff)
uv sync --extra dev
```

## Usage

### Running Scripts
```bash
# Activate virtual environment and run a script
uv run python src/your_script.py

# Or run with specific arguments
uv run python src/your_script.py --config config/example.yaml
```

### Development

#### Code Quality (Ruff)
```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Fix auto-fixable issues
uv run ruff check --fix

# Run both format and lint
uv run ruff format && uv run ruff check
```

#### Managing Dependencies
```bash
# Add a new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update dependencies
uv sync --upgrade
```

## Project Structure
- `src/` - Main source code
- `config/` - Configuration files
- `data/` - Data files
- `experiments/` - Experimental code
- `notebooks/` - Jupyter notebooks
- `scripts/` - Utility scripts
