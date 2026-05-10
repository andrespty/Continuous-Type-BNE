from games.rent_seeking.SymmetricRentSeeking import SymmetricRentSeeking
from utils.solvers import SymmetricSolver


from games.bertrand.SymmetricAirlinePricing import SymmetricAirlinePricing 

if __name__ == '__main__':
    # Train a symmetric 2-player rent-seeking BNE
    # game = SymmetricRentSeeking(n_players=2)
    # solver = SymmetricSolver(game)
    # solver.fit(n_samples=1000, epochs=1000, batch_size=500, type_bounds=(0.001, 1.001))
    # solver.plot_strategy()
    # solver.evaluate()


    game = SymmetricAirlinePricing(n_players=2)
    solver = SymmetricSolver(game)
    # solver.fit(n_samples=1000, epochs=10000, batch_size=500, type_bounds=(80.0, 250.0))
    # solver.plot_strategy(type_bounds=(80.0, 250.0))
    # solver.evaluate()
    # solver.save('saved_models/airline_symmetric_2p', '1')
    solver.load('saved_models/airline_symmetric_2p', '5')
    solver.fit(n_samples=1000, epochs=100000, batch_size=500, type_bounds=(80.0, 250.0))
    solver.evaluate()
    solver.save('saved_models/airline_symmetric_2p', '6')
    solver.plot_strategy(type_bounds=(80.0, 250.0))

