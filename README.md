# ContinuousBayesianNE

Python project for experiments related to continuous Bayesian Nash equilibrium models.

## Structure

- `main.py`: current top-level entry point
- `Symmetric/`: symmetric-game utilities, losses, and optimizers
- `tests/`: minimal test suite

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Test

```bash
pytest
```

## Notes

This baseline keeps the existing code layout intact and adds the files typically needed to work on the project comfortably.
