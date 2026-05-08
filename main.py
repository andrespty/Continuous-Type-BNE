from games.rent_seeking.SymmetricRentSeeking import SymmetricRentSeeking
from utils.solvers import SymmetricSolver

if __name__ == '__main__':
    # Train a symmetric 2-player rent-seeking BNE
    game = SymmetricRentSeeking(n_players=2)
    solver = SymmetricSolver(game)
    solver.fit(n_samples=1000, epochs=1000, batch_size=500, type_bounds=(0.001, 1.001))
    solver.plot_strategy()
    solver.evaluate()