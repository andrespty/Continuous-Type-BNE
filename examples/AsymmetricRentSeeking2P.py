"""
Asymmetric Rent-Seeking BNE
==========================
Computes the Bayesian Nash Equilibrium for an asymmetric n-player
rent-seeking game where player types are drawn from U[0, 1].

Usage:
    python -m examples.AsymmetricRentSeeking2P
"""

from games.rent_seeking.AsymmetricRentSeeking import AsymmetricRentSeeking
from utils.solvers import AsymmetricSolver

# --- Configuration ---
N_SAMPLES    = 1000
EPOCHS       = 1000
BATCH_SIZE   = 500
TYPE_BOUNDS  = [(0.001, 1.001), (0.51, 1.51)]
SAVE_PATH    = 'saved_models/Asymmetric2P'
VERSION      = '1'

game   = AsymmetricRentSeeking()
solver = AsymmetricSolver(game)

# --- Train ---
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