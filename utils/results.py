"""Load the latest checkpoint of each run and compute everything once into a RunResult.
Nothing is cached between calls: rerun evaluate_experiment to pick up newer checkpoints."""
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils.config import GameConfig, GAME_DEFS
from utils.evaluation import make_evaluator, CournotEvaluator
from experiments import experiment_runs
from utils.loading import load_model, load_training_data, merit_log_path, read_merit_log
from utils.metrics import (make_test_set, merits, layer_table, rent_seeking_curves, relative_losses,
                     cournot_curves, cournot_relative_losses, true_epsilon_cournot, spectral_norm_table)
from utils.epsilon import epsilon_report


@dataclass
class RunResult:
    label: str
    config: GameConfig
    game_def: dict
    l2_alpha: float
    step: int
    checkpoint: str
    curves: dict
    merit_log: Optional[pd.DataFrame]
    weights: pd.DataFrame
    train_merit: dict
    test: dict
    fill_distance: Optional[float]
    epsilon: Optional[dict]
    train_types: dict                 # {"own": array, "opp": array or None}, training types per player
    spectral: pd.DataFrame


# ---------- one test set ----------

def _test_set_rent_seeking(model, config, game_def, seed, weightings, ev_kwargs):
    """Relative losses (own types on the grid, fresh opponent pool) and the true epsilon."""
    ev = make_evaluator(model, config, game_def, seed=seed, **ev_kwargs)
    curves = rent_seeking_curves(ev)
    losses = {}
    for w in weightings:
        per, pooled = relative_losses(curves, ev, w)
        losses[w] = {"per_player": per, "strategy": pooled}
    eps_true = max(p["max_loss"] for p in losses["uniform"]["per_player"].values())
    return losses, eps_true


def _test_set_cournot(model, config, ev, seed, n_types):
    """Relative losses against the closed-form equilibrium, and the true epsilon
    (opponents playing the trained networks)."""
    per, pooled, pooled_rms = cournot_relative_losses(ev, seed, n_types)
    _, eps_true = true_epsilon_cournot(model, config, seed=seed)
    return {"uniform": {"per_player": per, "strategy": pooled, "rms_strategy": pooled_rms}}, eps_true


# ---------- one run ----------

def evaluate_run(label, config, l2_alpha=0.0, n_sets=5, base_seed=10_000,
                 cournot_n_types=10_000, cournot_grid=60, **ev_kwargs):
    """Load the latest checkpoint of one run and compute everything for figures and tables.
    ev_kwargs go to the evaluator (e.g. w_t=2000 for a faster best-response grid)."""
    game_def = GAME_DEFS[config.game]
    model, checkpointer = load_model(config, l2_alpha)
    checkpoint = checkpointer.manager.latest_checkpoint
    step = int(checkpointer.optimizer.iterations.numpy())
    print(f"[{label}] {checkpoint} (step {step:,})")

    ev = make_evaluator(model, config, game_def, **ev_kwargs)
    is_cournot = isinstance(ev, CournotEvaluator)

    log_path = merit_log_path(config)
    merit_log = read_merit_log(log_path) if os.path.exists(log_path) else None

    x_own, x_opp = load_training_data(checkpointer, config)
    train_total, train_per = merits(model, game_def, x_own, x_opp)
    eps_theory = epsilon_report(model, checkpointer, config)

    weightings = ["uniform"] + (["prior"] if config.opp_dist != "uniform" else [])
    test = {"seeds": [], "merit": [], "merit_per_player": [], "true_epsilon": [],
            "relative_loss": {w: [] for w in weightings}}
    for k in range(n_sets):
        seed = base_seed + k
        t_own, t_opp = make_test_set(config, game_def, seed)
        total, per = merits(model, game_def, t_own, t_opp)
        if is_cournot:
            losses, eps_true = _test_set_cournot(model, config, ev, seed, cournot_n_types)
        else:
            losses, eps_true = _test_set_rent_seeking(model, config, game_def, seed,
                                                      weightings, ev_kwargs)
        test["seeds"].append(seed)
        test["merit"].append(total)
        test["merit_per_player"].append(per)
        for w, v in losses.items():
            test["relative_loss"][w].append(v)
        test["true_epsilon"].append(eps_true)

    curves = cournot_curves(ev, cournot_grid) if is_cournot else rent_seeking_curves(ev)

    return RunResult(
        label=label,
        config=config,
        game_def=game_def,
        l2_alpha=l2_alpha,
        step=step,
        checkpoint=checkpoint,
        curves=curves,
        merit_log=merit_log,
        weights=layer_table(model, game_def),
        train_merit={"total": train_total, "per_player": train_per},
        test=test,
        fill_distance=max(p["eps_theta"] for p in eps_theory["per_player"]),
        epsilon=eps_theory,
        train_types={"own": x_own.numpy(), "opp": None if x_opp is None else x_opp.numpy()},
        spectral=spectral_norm_table(model, game_def),
    )


def evaluate_experiment(name, n_sets=5, base_seed=10_000, **ev_kwargs):
    """evaluate_run for every run in EXPERIMENTS[name]. Always reads the latest checkpoints."""
    return [evaluate_run(label, config, l2_alpha, n_sets=n_sets, base_seed=base_seed, **ev_kwargs)
            for label, config, l2_alpha in experiment_runs(name)]