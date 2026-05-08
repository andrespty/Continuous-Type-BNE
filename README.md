# ContinuousBayesianNE

A Python framework for computing Bayesian Nash Equilibria (BNE) in continuous-type games using neural networks. The solver uses a primal-dual neural network architecture trained to satisfy equilibrium conditions, supporting both symmetric and asymmetric games.

## Method

Each player's strategy is parameterized by a neural network (primal net) trained alongside a dual network that enforces equilibrium constraints. The networks are jointly optimized using a custom equilibrium loss derived from the game's best-response conditions.

## Project Structure

```
├── games/                        # Game definitions
│   ├── rent_seeking/
│   │   ├── SymmetricRentSeeking.py
│   │   └── AsymmetricRentSeeking.py
│   └── cournot-oligopoly/
│       └── SymmetricAirlinePricing.py
│
├── neural_nets/                  # Network architectures
│   ├── base_networks/
│   │   ├── primal_NN.py          # Strategy network
│   │   └── dual_NN.py            # Dual variable network
│   ├── symmetric_models.py       # Combined model for symmetric games
│   └── asymmetric_models.py      # Combined model for asymmetric games
│
├── utils/                        # Core framework
│   ├── game.py                   # Abstract Game base class
│   ├── solvers.py                # SymmetricSolver / AsymmetricSolver
│   ├── loss.py                   # Equilibrium loss (EquilibriumLoss)
│   └── stepsize.py               # Learning rate utilities
│
├── examples/                     # Runnable examples
│   ├── SymmetricRentSeeking2P.py
│   ├── SymmetricRentSeeking3P.py
│   └── AsymmetricRentSeeking2P.py
│
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run examples from the project root:

```bash
python -m examples.SymmetricRentSeeking2P
python -m examples.SymmetricRentSeeking3P
python -m examples.AsymmetricRentSeeking2P
```

### Basic usage

```python
from games.rent_seeking.SymmetricRentSeeking import SymmetricRentSeeking
from utils.solvers import SymmetricSolver

game   = SymmetricRentSeeking(n_players=2)
solver = SymmetricSolver(game)

solver.fit(n_samples=1000, epochs=1000, batch_size=500, type_bounds=(0.001, 1.001))
solver.evaluate()
solver.plot_strategy()
solver.save('saved_models/Symmetric2P', '1')
```

### Loading a saved model

```python
solver.load('saved_models/Symmetric2P', '1')
solver.evaluate()
solver.plot_strategy()
```

## Adding a New Game

1. Create a new file under `games/your_game/YourGame.py`
2. Inherit from `Game` (defined in `utils/game.py`) and implement `nn_model()`, `loss_fn()`, and `optimizer()`
3. Add a corresponding example under `examples/`

## Requirements

- Python 3.10+
- TensorFlow 2.21.0
- Keras 3.14.0
- NumPy 2.4.4
- Matplotlib 3.10.9