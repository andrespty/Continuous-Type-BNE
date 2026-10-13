"""Theoretical epsilon:
    eps_Psi = m * eps_J + L_Psi * eps_theta
    epsilon = max_i (2 + sqrt 2) z_i sqrt(eps_Psi) max(B_lambda, B_a) + sqrt(eps_Psi) sqrt(z_i) B_a
epsilon_report computes it (per player); print_epsilon prints it; epsilon_scenarios explores m and eps_J."""
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.config import GAME_DEFS
from utils.loading import load_training_data
from utils.metrics import fill_distance, fill_distance_box


# ====================== building blocks ======================

def epsilon_bound(eps_psi, z, B_a, B_lambda):
    """The bound for one player."""
    return ((2 + np.sqrt(2)) * z * np.sqrt(eps_psi) * max(B_lambda, B_a)
            + np.sqrt(eps_psi) * np.sqrt(z) * B_a)


def lipschitz_1d(psi, grid):
    """Largest slope between neighbouring grid points."""
    return float(np.max(np.abs(np.diff(psi)) / np.diff(grid)))


def lipschitz_2d(psi, t1, t2):
    """Largest gradient norm on a 2-D mesh (t1, t2 from np.meshgrid)."""
    d2, d1 = np.gradient(psi, t2[:, 0], t1[0, :])
    return float(np.sqrt(d1 ** 2 + d2 ** 2).max())


def per_type_psi(model, loss, theta, x_opp):
    """Symmetric games: Psi at each own type = the loss on that single type against opponents x_opp."""
    y = model(theta, training=False)
    y_opp = model(x_opp, training=False)

    @tf.function
    def run():
        return tf.map_fn(lambda k: loss.call_two_sample(theta[k:k + 1], y[k:k + 1], y_opp),
                         tf.range(tf.shape(theta)[0]), fn_output_signature=tf.float32)
    return run().numpy()


def _grid_input(x_train, i, values):
    """Model input with player i's types set to `values` (shape (N, d)); the other players'
    slots hold any valid type, since only player i's outputs are used."""
    x = np.repeat(x_train.numpy()[:1], len(values), axis=0)
    x[:, i, :] = values
    return tf.constant(x, dtype=tf.float32)


# ====================== Psi per player, by game type ======================
# Each returns one dict per player: max of Psi on the grid, L_Psi, and eps_theta.

def _players_symmetric(model, loss, game_def, x_own, x_opp, n_grid):
    n_players = game_def["n_players"]
    lo, hi = float(game_def["theta_low"][0]), float(game_def["theta_high"][0])
    grid = np.linspace(lo, hi, n_grid)
    # the symmetric losses sum over players; Psi_i is per player
    psi = per_type_psi(model, loss, tf.constant(grid.reshape(-1, 1, 1), dtype=tf.float32), x_opp) / n_players
    psi_train = per_type_psi(model, loss, x_own, x_opp) / n_players
    return [dict(psi_max=float(psi.max()), psi_train_max=float(psi_train.max()),
                 L_psi=lipschitz_1d(psi, grid),
                 eps_theta=fill_distance(x_own.numpy()[:, 0, 0], lo, hi))]


def _players_asym(model, loss, game_def, x_own, n_grid):
    y_train = model(x_own, training=False)
    players = []
    for i in range(game_def["n"]):
        lo, hi = float(game_def["theta_low"][i]), float(game_def["theta_high"][i])
        grid = np.linspace(lo, hi, n_grid)
        x = _grid_input(x_own, i, grid.reshape(-1, 1))
        psi = loss._terms_two_sample(x, model(x, training=False), y_train)[i].numpy().ravel()   # on the grid
        psi_train = loss._terms_two_sample(x_own, y_train, y_train)[i].numpy().ravel()          # at training types
        players.append(dict(psi_max=float(psi.max()), psi_train_max=float(psi_train.max()),
                            L_psi=lipschitz_1d(psi, grid),
                            eps_theta=fill_distance(x_own.numpy()[:, i, 0], lo, hi)))
    return players

def _players_cournot(model, loss, game_def, x_own, g):
    n, z = game_def["n"], game_def["z"][0]
    off = n * z
    y_train = model(x_own, training=False)
    mean_p = [tf.reduce_mean(y_train[:, j * z:(j + 1) * z], axis=0, keepdims=True) for j in range(n)]
    players = []
    for i in range(n):
        lo = np.asarray(game_def["theta_low"][i], dtype=np.float64)
        hi = np.asarray(game_def["theta_high"][i], dtype=np.float64)
        t1, t2 = np.meshgrid(np.linspace(lo[0], hi[0], g), np.linspace(lo[1], hi[1], g))
        th = np.stack([t1.ravel(), t2.ravel()], axis=1).astype(np.float32)
        opp = tf.add_n([mean_p[j] for j in range(n) if j != i])      # opponents' mean quantities

        # Psi on the grid
        y = model(_grid_input(x_own, i, th), training=False)
        cost = tf.matmul(th, loss.Gammas[i], transpose_b=True)
        psi = tf.reduce_sum(loss._player_terms(
            y[:, i * z:(i + 1) * z], cost, opp,
            y[:, off + 2 * z * i: off + 2 * z * i + z],
            y[:, off + 2 * z * i + z: off + 2 * z * (i + 1)]), axis=1).numpy().reshape(g, g)

        # Psi at the training types
        cost_tr = tf.matmul(x_own[:, i, :], loss.Gammas[i], transpose_b=True)
        psi_train = tf.reduce_sum(loss._player_terms(
            y_train[:, i * z:(i + 1) * z], cost_tr, opp,
            y_train[:, off + 2 * z * i: off + 2 * z * i + z],
            y_train[:, off + 2 * z * i + z: off + 2 * z * (i + 1)]), axis=1).numpy()

        players.append(dict(psi_max=float(psi.max()), psi_train_max=float(psi_train.max()),
                            L_psi=lipschitz_2d(psi, t1, t2),
                            eps_theta=fill_distance_box(x_own.numpy()[:, i, :], lo, hi)))
    return players


# ====================== the report ======================

def epsilon_report(model, checkpointer, config, B_lambda=None, n_grid=2000, g=100):
    """Theoretical epsilon and its ingredients, per player.
    B_lambda=None means B_lambda <= B_a, so max(B_lambda, B_a) = B_a."""
    game_def = GAME_DEFS[config.game]
    loss = game_def["loss_func"]
    x_own, x_opp = load_training_data(checkpointer, config)
    m = int(x_own.shape[0])

    y_own = model(x_own, training=False)
    if x_opp is None:
        eps_J = float(loss(x_own, y_own))
    else:
        eps_J = float(loss.call_two_sample(x_own, y_own, model(x_opp, training=False)))

    if config.game == "cournot":
        players = _players_cournot(model, loss, game_def, x_own, g)
    elif game_def["symmetric"]:
        players = _players_symmetric(model, loss, game_def, x_own,
                                     x_own if x_opp is None else x_opp, n_grid)
    else:
        players = _players_asym(model, loss, game_def, x_own, n_grid)

    for i, p in enumerate(players):
        z = game_def["z"][i]
        B_a = float(game_def["action_high"][i])
        B_lam = B_a if B_lambda is None else B_lambda
        eps_psi = m * eps_J + p["L_psi"] * p["eps_theta"]                     # theorem as stated
        eps_psi_ref = p["psi_train_max"] + p["L_psi"] * p["eps_theta"]        # max training residual instead of m * eps_J
        p.update(player=i + 1, z=z, B_a=B_a, B_lambda=B_lam,
                 eps_psi=eps_psi, epsilon=epsilon_bound(eps_psi, z, B_a, B_lam),
                 eps_psi_refined=eps_psi_ref, epsilon_refined=epsilon_bound(eps_psi_ref, z, B_a, B_lam))

    return dict(m=m, eps_J=eps_J, per_player=players,
                epsilon=max(p["epsilon"] for p in players),
                epsilon_refined=max(p["epsilon_refined"] for p in players))


def print_epsilon(result):
    m, eps_J = result["m"], result["eps_J"]
    print(f"m {m}   eps_J {eps_J:.3e}   m * eps_J {m * eps_J:.3e}\n")
    print(f"{'player':<8}{'max Psi tr':>12}{'eps_theta':>11}{'L_Psi':>11}{'eps_Psi':>11}"
          f"{'eps_Psi ref':>12}{'epsilon':>11}{'eps ref':>11}")
    for p in result["per_player"]:
        print(f"P{p['player']:<7}{p['psi_train_max']:>12.3e}{p['eps_theta']:>11.3e}{p['L_psi']:>11.3e}"
              f"{p['eps_psi']:>11.3e}{p['eps_psi_refined']:>12.3e}"
              f"{p['epsilon']:>11.3e}{p['epsilon_refined']:>11.3e}")
    print(f"\nepsilon (theorem) = {result['epsilon']:.3e}   "
          f"epsilon (refined) = {result['epsilon_refined']:.3e}")


def epsilon_scenarios(result, config, ms, eps_Js):
    """epsilon for each assumed merit eps_J and sample size m, with eps_theta(m) calibrated from
    the current sample as eps_theta * (m_now / m)^(1/d), and L_Psi as measured."""
    game_def = GAME_DEFS[config.game]
    m_now = result["m"]
    rows = {}
    for eJ in eps_Js:
        row = {}
        for m in ms:
            worst = 0.0
            for p in result["per_player"]:
                d = game_def["d"][p["player"] - 1]
                eps_theta = p["eps_theta"] * (m_now / m) ** (1 / d)
                eps_psi = m * eJ + p["L_psi"] * eps_theta
                worst = max(worst, epsilon_bound(eps_psi, p["z"], p["B_a"], p["B_lambda"]))
            row[m] = worst
        rows[f"eps_J = {eJ:.0e}"] = row
    df = pd.DataFrame(rows).T
    df.columns.name = "m"
    return df