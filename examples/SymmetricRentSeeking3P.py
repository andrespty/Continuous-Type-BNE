"""
Symmetric Rent-Seeking 3 Players BNE
==========================
Computes the Bayesian Nash Equilibrium for a symmetric n-player
rent-seeking game where player types are drawn from U[0, 1].

Usage:
    python -m examples.SymmetricRentSeeking3P
"""

from games.rent_seeking.SymmetricRentSeeking import SymmetricRentSeeking
from utils.solvers import SymmetricSolver

# --- Configuration ---
N_PLAYERS    = 3
N_SAMPLES    = 1000
EPOCHS       = 1000
BATCH_SIZE   = 500
TYPE_BOUNDS  = (0.001, 1.001)
SAVE_PATH    = 'saved_models/Symmetric3P'
VERSION      = '1'

# --- Train ---
game   = SymmetricRentSeeking(n_players=N_PLAYERS)
solver = SymmetricSolver(game)

solver.fit(
    n_samples=N_SAMPLES,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    type_bounds=TYPE_BOUNDS
)

solver.evaluate()
solver.plot_strategy()
solver.save(SAVE_PATH, VERSION)

# --- Load & verify ---
solver.load(SAVE_PATH, VERSION)
solver.evaluate()
solver.plot_strategy()