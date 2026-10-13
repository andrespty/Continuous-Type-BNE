import numpy as np 
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
from utils.config import GameConfig, GAME_DEFS
from utils.models import build_main_model, display_strat
from utils.utils import make_optimizer, make_training_data, make_two_sample_data
from utils.training import train
from utils.evaluation import *
from utils.lossFunctions import *
from utils.loading import load_training_data
from utils.metrics import merits
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--game", required=True)
parser.add_argument("--optimizer", default="adam")
parser.add_argument("--run_n", required=True)
parser.add_argument("--max_iters", type=int, default=1_000_000)
parser.add_argument("--alpha", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sgd_alpha_0", type=float, default=1e-3)
parser.add_argument("--adam_lr", type=float, default=None)

parser.add_argument("--pareto", action="store_true",
                    help="shortcut for --two_sample --own_dist uniform --opp_dist pareto")
parser.add_argument("--two_sample", action="store_true")
parser.add_argument("--own_dist", default="uniform", choices=["uniform", "pareto", "uniform+pareto", "stratified_pareto"])
parser.add_argument("--opp_dist", default="uniform", choices=["uniform", "pareto"])
parser.add_argument("--m", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=250)
parser.add_argument("--strat_floor", type=int, default=1)

args = parser.parse_args()

if args.pareto:
    args.two_sample, args.own_dist, args.opp_dist = True, "uniform", "pareto"

if args.two_sample and args.game != "sym2":
    parser.error("two-sample / Pareto runs are only set up for sym2")
    
l2_alpha = args.alpha
save_weight = 100_000

config_kwargs = dict(
    game=args.game,
    optimizer=args.optimizer,
    run_n=args.run_n,
    max_iters=args.max_iters,
    seed=args.seed,
    two_sample=args.two_sample,
    own_dist=args.own_dist,
    opp_dist=args.opp_dist,
    m=args.m,
    batch_size=args.batch_size,
    strat_floor=args.strat_floor
)
if args.sgd_alpha_0 is not None:
    config_kwargs["sgd_alpha_0"] = args.sgd_alpha_0
if args.adam_lr is not None:
    config_kwargs["adam_lr"] = args.adam_lr

CONFIG = GameConfig(**config_kwargs)
print(CONFIG)


game_def = GAME_DEFS[CONFIG['game']]
designed_initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=CONFIG['seed'])
main_model = build_main_model(game_def, designed_initializer, l2_alpha)

optimizer = make_optimizer(CONFIG)

if CONFIG.two_sample: # pareto
    theta_own, theta_opp = make_two_sample_data(game_def, CONFIG)
    checkpointer = train(main_model, optimizer, game_def, CONFIG,
                         theta_own=theta_own, theta_opp=theta_opp, save_weight_every=save_weight)
else:
    training_data = make_training_data(game_def, CONFIG['m'], CONFIG['seed'])
    checkpointer = train(main_model, optimizer, game_def, CONFIG,
                         training_data=training_data, save_weight_every=save_weight)


x_own, x_opp = load_training_data(checkpointer, CONFIG)
total, per = merits(main_model, game_def, x_own, x_opp)
print(f"Final training merit: {total:.6e}" + (f"   per player: {per}" if per else ""))