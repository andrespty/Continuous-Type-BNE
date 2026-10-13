import json
import numpy as np
from utils.config import GameConfig


if __name__ == '__main__':
    # Create a GameConfig instance from the loaded data
    game_config = GameConfig(
        game='sym2',
        optimizer='sgd',
        run_n='uniform_oct6',
        max_iters=1_000_000,
        seed=0,
        m=1000,
        batch_size=500,
        sgd_alpha_0=1e-3,
        two_sample=True,                # When True, uses two-sample training with own_dist and opp_dist
        own_dist='stratified_pareto',
        opp_dist='pareto',
        strat_floor=1,
    )

    # Print the configuration to verify
    print(game_config)