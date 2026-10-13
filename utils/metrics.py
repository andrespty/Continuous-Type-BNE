"""Merits, coverage, network weights, and utility-based metrics (relative loss, true epsilon)."""
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.config import GAME_DEFS
from utils.utils import make_sampler
from utils.evaluation import make_evaluator


# ====================== samples and merit ======================

def make_test_set(config, game_def, seed):
    """Fresh types from the run's training distribution: own types (m) and,
    for two-sample runs, opponent types (tilde_m)."""
    rng = np.random.default_rng(seed)
    n, d = game_def["n"], game_def["d"]
    own = make_sampler(config.own_dist, config, game_def)
    x_own = np.stack([own(rng, i, (config.m, d[i])) for i in range(n)], axis=1)
    x_opp = None
    if config.two_sample:
        opp = make_sampler(config.opp_dist, config, game_def)
        x_opp = tf.cast(np.stack([opp(rng, i, (config.tilde_m, d[i])) for i in range(n)], axis=1),
                        tf.float32)
    return tf.cast(x_own, tf.float32), x_opp


def merits(model, game_def, x_own, x_opp=None):
    """Total merit (no L2 term) and per-player merits.
    Symmetric games: per-player merit = total / n_players."""
    loss = game_def["loss_func"]
    y_own = model(x_own, training=False)
    if x_opp is None:
        total = float(loss(x_own, y_own))
    else:
        total = float(loss.call_two_sample(x_own, y_own, model(x_opp, training=False)))

    n_players = game_def.get("n_players", game_def["n"])
    if game_def["symmetric"]:
        return total, [total / n_players] * n_players
    if hasattr(loss, "player_merits") and x_opp is None:
        return total, [float(m) for m in loss.player_merits(x_own, y_own)]
    return total, None


def fill_distance(samples, lo, hi):
    """Largest distance from any type in [lo, hi] to its nearest sample (1-D types, exact)."""
    s = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    return float(max(s[0] - lo, hi - s[-1], np.diff(s).max() / 2))


def fill_distance_box(samples, lo, hi, g=200, chunk=4000):
    """Largest Euclidean distance from a g^d grid of the box [lo, hi] to the nearest sample
    (d-dimensional types; slightly underestimates the true supremum)."""
    axes = [np.linspace(l, h, g) for l, h in zip(lo, hi)]
    pts = np.stack(np.meshgrid(*axes), -1).reshape(-1, len(lo))
    s = np.asarray(samples, dtype=np.float64)
    worst = 0.0
    for k in range(0, len(pts), chunk):
        d2 = ((pts[k:k + chunk, None, :] - s[None]) ** 2).sum(-1)
        worst = max(worst, float(np.sqrt(d2.min(axis=1)).max()))
    return worst


def layer_table(model, game_def):
    """Max absolute parameter (weight or bias) per layer of each player's strategy network."""
    cols = {}
    for i in range(game_def["n"]):
        sub = model.get_layer(f"Player{i + 1}")
        name = "Player 1 (shared)" if game_def["symmetric"] else f"Player {i + 1}"
        for layer in sub.layers:
            ws = layer.get_weights()
            if ws:
                cols.setdefault(name, {})[layer.name] = max(float(np.abs(w).max()) for w in ws)
    return pd.DataFrame(cols)


# ====================== rent-seeking games ======================

def rent_seeking_curves(ev):
    """NN action, best response and utilities on each player's type grid."""
    curves = {}
    for i, j in ev.matchups():
        theta = ev.theta_grid(i)
        pool = ev.opponent_pool(j)
        a_nn = ev.action(i, theta)
        a_hat, rho_hat = ev.best_response(i, pool, theta)
        rho_nn = ev.rho(a_nn, pool, theta)
        curves[i] = {
            "theta": theta.numpy().ravel(),
            "a_nn": a_nn.numpy().ravel(),
            "a_hat": a_hat.numpy().ravel(),
            "rho_nn": rho_nn.numpy().ravel().astype(np.float64),
            "rho_hat": rho_hat.numpy().ravel().astype(np.float64),
        }
    return curves


def relative_losses(curves, ev, weighting="uniform", tail_from=0.4):
    """Per-player relative utility loss (1 - rho_bar / (rho_bar + l_bar)), max loss, tail losses,
    and the pooled loss of the strategy profile. weighting='prior' weights the grid by the prior density."""
    per = {}
    for i, c in curves.items():
        w = np.ones_like(c["rho_nn"])
        if weighting == "prior" and ev.density is not None:
            w = np.asarray(ev.density(i, c["theta"]), dtype=np.float64).ravel()
        w = w / w.sum()

        l = np.maximum(0.0, c["rho_hat"] - c["rho_nn"])
        l_bar, l_max = float((w * l).sum()), float(l.max())
        rho_bar = float((w * c["rho_nn"]).sum())
        tail = c["theta"] > tail_from
        per[i] = {
            "mean_loss": l_bar,
            "max_loss": l_max,
            "rho_bar": rho_bar,
            "relative_loss": l_bar / (rho_bar + l_bar),
            "max_relative_loss": l_max / (rho_bar + l_max),
            "max_loss_tail": float(l[tail].max()),
            "mean_loss_tail": float(l[tail].mean()),
        }
    pooled = (sum(p["mean_loss"] for p in per.values())
              / sum(p["rho_bar"] + p["mean_loss"] for p in per.values()))
    return per, pooled


# ====================== Cournot ======================

def cournot_curves(ev, n_points=60):
    """Per player, on the type grid: NN and equilibrium actions (g, g, z), and the utilities of
    the NN strategy and the best response (g, g), opponents at the closed-form equilibrium."""
    curves = {}
    for i in range(ev.n):
        th1, th2, theta = ev.theta_grid(i, n_points)
        pool = ev.opponent_pool(i, against="equilibrium")
        a_nn = ev.action(i, theta)
        a_hat, rho_hat = ev.best_response(i, pool, theta)
        rho_nn = ev.rho(a_nn, pool, theta, i)
        g = th1.shape
        curves[i] = {
            "th1": th1, "th2": th2,
            "a_nn": a_nn.numpy().reshape(g + (ev.z,)),
            "a_hat": a_hat.numpy().reshape(g + (ev.z,)),
            "rho_nn": rho_nn.numpy().reshape(g),
            "rho_hat": rho_hat.numpy().reshape(g),
        }
    return curves


def cournot_relative_losses(ev, seed, n_types=10_000):
    """Relative loss and strategy deviation per player on n_types sampled own types,
    opponents at the closed-form equilibrium; plus the pooled values."""
    rng = np.random.default_rng(seed)
    per, sq_all = {}, []
    for i in range(ev.n):
        theta = ev.sample_types(i, n_types, rng)
        pool = ev.opponent_pool(i, against="equilibrium")
        a_nn = ev.action(i, theta)
        a_hat, rho_hat = ev.best_response(i, pool, theta)
        rho_nn = ev.rho(a_nn, pool, theta, i).numpy().ravel().astype(np.float64)
        rho_hat = rho_hat.numpy().ravel().astype(np.float64)
        sq = tf.reduce_sum((a_nn - a_hat) ** 2, axis=1).numpy().astype(np.float64)

        l = np.maximum(0.0, rho_hat - rho_nn)
        l_bar, l_max, rho_bar = l.mean(), l.max(), rho_nn.mean()
        per[i] = {
            "mean_loss": l_bar,
            "max_loss": l_max,
            "rho_bar": rho_bar,
            "relative_loss": l_bar / (rho_bar + l_bar),
            "max_relative_loss": l_max / (rho_bar + l_max),
            "rms_deviation": np.sqrt(sq.mean()),
            "max_deviation": np.sqrt(sq.max()),
        }
        sq_all.append(sq)
    pooled = (sum(p["mean_loss"] for p in per.values())
              / sum(p["rho_bar"] + p["mean_loss"] for p in per.values()))
    pooled_rms = float(np.sqrt(np.concatenate(sq_all).mean()))
    return per, pooled, pooled_rms

def spectral_norm_table(model, game_def):
    """Spectral norm (largest singular value) of each layer's kernel in each player's strategy
    network, plus the maximum over layers, their product, and the resulting Lipschitz bound
    B_a * 1/4 * product (tanh hidden layers, sigmoid output, output scaled by B_a)."""
    cols = {}
    for i in range(game_def["n"]):
        sub = model.get_layer(f"Player{i + 1}")
        name = "Player 1 (shared)" if game_def["symmetric"] else f"Player {i + 1}"
        norms = {layer.name: float(np.linalg.norm(layer.get_weights()[0], 2))
                 for layer in sub.layers if layer.get_weights()}
        layer_norms = list(norms.values())
        product = float(np.prod(layer_norms))
        norms["max"] = max(layer_norms)
        norms["product"] = product
        norms["Lipschitz bound"] = 0.25 * float(game_def["action_high"][i]) * product
        cols[name] = norms
    return pd.DataFrame(cols)

def true_epsilon_cournot(model, config, m_opp=1000, seed=0):
    """max over players and types of rho_hat - rho, opponents playing the trained networks."""
    game_def = GAME_DEFS[config.game]
    ev = make_evaluator(model, config, game_def, m_opp=m_opp, seed=seed)
    per = {}
    for i in range(ev.n):
        _, _, theta = ev.theta_grid(i)
        pool = ev.opponent_pool(i, against="nn")
        _, rho_hat = ev.best_response(i, pool, theta)
        rho_nn = ev.rho(ev.action(i, theta), pool, theta, i)
        per[i + 1] = float(tf.reduce_max(rho_hat - rho_nn))
    return per, max(per.values())