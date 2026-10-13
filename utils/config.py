import os
import numpy as np
from utils.lossFunctions import *
import tensorflow as tf
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # config.py is in utils/, so go up one level
GAME_DEFS = {
    "asym2": {
        "n": 2,
        "n_players": 2,
        "z": [1, 1],                     # scalar action per player
        "d": [1, 1],                     # scalar type per player
        "theta_low":  np.array([0.01, 0.51], dtype=np.float32),
        "theta_high": np.array([1.01, 1.51], dtype=np.float32),
        "action_low":  np.array([0.0, 0.0], dtype=np.float32),
        "action_high": np.array([10.0, 5], dtype=np.float32),
        "loss_func": CustomLoss_A2(),
        "symmetric": False,
        "dual_output": "relu"
    },
    "sym2": {
        "n": 1,
        "n_players": 2,
        "z": [1],                     # scalar action per player
        "d": [1],                     # scalar type per player
        "theta_low":  np.array([0.01], dtype=np.float32),
        "theta_high": np.array([1.01], dtype=np.float32),
        "action_low":  np.array([0.0], dtype=np.float32),
        "action_high": np.array([20.0], dtype=np.float32),
        "loss_func": CustomLoss(),
        "symmetric": True,
        "dual_output": "relu"
    },
    "sym3":{
        "n": 1,
        "n_players": 3,
        "z": [1],                     # scalar action per player
        "d": [1],                     # scalar type per player
        "theta_low":  np.array([0.01], dtype=np.float32),
        "theta_high": np.array([1.01], dtype=np.float32),
        "action_low":  np.array([0.0], dtype=np.float32),
        "action_high": np.array([15.0], dtype=np.float32),
        "loss_func": CustomLossP3(m=500),
        "symmetric": True,
        "dual_output": "relu"
    },
    "cournot": {
        "n": 3,
        "n_players": 3,
        "z": [3, 3, 3],                  # quantity in each of 3 markets
        "d": [2, 2, 2],                  # 2-dimensional type per player
        "theta_low":  np.array([[0.5, 1.0], [1.0, 1.5], [1.5, 2.0]], dtype=np.float32),
        "theta_high": np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32),
        "action_low":  np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "action_high": np.array([5.0, 5.0, 5.0], dtype=np.float32),
        "loss_func": NMultiMarketCournotLoss(
            alpha_m=(10.0, 15.0, 20.0),
            beta_m=(1.0, 1.0, 1.0),
            qbar_m=(100.0, 100.0, 100.0),
            lambda_m=(1.0, 1.5, 2.0),
            Gammas=[
                [[1.0, 0.3], [0.3, 1.0], [1.0, 1.0]],
                [[1.0, 0.5], [0.5, 1.0], [0.5, 0.5]],
                [[1.0, 0.2], [0.2, 1.0], [0.8, 0.7]],
            ],
        ),
        "symmetric": False,
        "dual_output": "softplus"
    },
}


@dataclass
class GameConfig:
    game: str = "sym2"              # one of: "asym2", "sym2", "sym3", "cournot"
    optimizer: str = "adam"         # "adam" or "sgd"
    seed: int = 0
    

    # --- sampling (shared across games/optimizers for comparability) ---
    m: int = 1000                   # type samples per player (training set)
    batch_size: int = 500           # matches the manuscript's Adam runs
    tilde_m: int = 1000             # MC samples per opponent in the gradient estimator

    adam_lr: float = 1e-5

    # --- SGD settings: tau_t = tau1 / t^r, r in (0.5, 1) per Corollary 2 ---
    sgd_alpha_0: float = 0.01
    sgd_a: float = 1.0
    sgd_b: float = 1e-4
    sgd_r: float = 0.7

    # --- iteration budget + logging ---
    max_iters: int = 10000          # PILOT value only; full HPC runs use round-1 reference counts
    log_every_decades: int = 20     # geometrically spaced checkpoints per decade of iterations

    # --- fixed evaluation set used for the merit/relative-loss curve ---
    eval_m: int = 1000              # size of the held-out type grid for logging

    
    # outdir: str = "/gpfs/home/aeh19f/Game_Theory/ContinuousTypeBNE/logs/"
    model_path: str = os.path.join(BASE_DIR, "saved_models") + os.sep
    run_n: str = "pareto_19"

    # --- Pareto Distribution ----------
    two_sample: bool = False        # True: own types and opponent types are sampled separately
    own_dist: str = "uniform"       # own-type sampling (coverage): "uniform" or "pareto"
    opp_dist: str = "uniform"       # opponent-type prior (defines the game): "uniform" or "pareto"
    pareto_a: float = 3.0
    cap_quantile: float = 0.999
    opp_batch_size: int = 500 
    
    # --- stratified sampling (own_dist="stratified_pareto") ---
    strat_bins: int = 100           # equal-width intervals over [theta_low, theta_high]
    strat_floor: int = 2            # minimum own types per interval

    @property
    def run_dir(self):
        """Folder holding everything for this run: checkpoints, data, training log."""
        return os.path.join(self.model_path, self.game, f"{self.optimizer}_{self.run_n}")

    def __post_init__(self):
        os.makedirs(self.run_dir, exist_ok=True)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    # --- dict-style access, so CONFIG["key"] keeps working unchanged ---
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)