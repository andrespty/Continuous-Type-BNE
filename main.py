import tensorflow as tf
import timeit

from games.rent_seeking.SymmetricRentSeeking import SymmetricRentSeeking
from games.rent_seeking.AsymmetricRentSeeking import AsymmetricRentSeeking
from utils.solvers import SymmetricSolver, AsymmetricSolver

from utils.loader import ModelLoader

if __name__ == '__main__':
    # Symmetric 2-player
    # game = SymmetricRentSeeking(n_players=2)
    # solver = SymmetricSolver(game)
    # solver.fit(n_samples=1000, epochs=1000, batch_size=500, type_bounds=(0.001, 1.001))
    # solver.plot_strategy()

    # Symmetric 3-player
    # game = SymmetricRentSeeking(n_players=3)
    # solver = SymmetricSolver(game)
    # solver.fit(n_samples=1000, epochs=100, batch_size=500, type_bounds=(0.001, 1.001))
    # solver.plot_strategy()

    # Asymmetric 2-player
    # game = AsymmetricRentSeeking()
    # solver = AsymmetricSolver(game)
    # solver.fit(n_samples=1000, epochs=1000, batch_size=500, type_bounds=[(0.001, 1.001), (0.51, 1.51)])
    # solver.evaluate()
    # solver.plot_strategy()

    loader = ModelLoader(game=SymmetricRentSeeking(n_players=2))
    model = loader.load('saved_models/S2P_lambda_C1')