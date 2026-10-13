import tensorflow as tf
import numpy as np
from utils.lossFunctions import DecaySGD

def make_training_data(game_def, m_samples, seed=None):
    """Draws m_samples i.i.d. type samples per player from each player's own
    Theta_i (Section 6.2's baseline uniform sampler). Shape (m_samples, N, d_i)."""
    rng = np.random.default_rng(seed)
    n, d = game_def["n"], game_def["d"]
    data = np.stack([
        rng.uniform(game_def["theta_low"][i], game_def["theta_high"][i],
                    size=(m_samples, d[i]))
        for i in range(n)
    ], axis=1)
    return tf.cast(data, dtype=tf.float32)

def make_optimizer(config):
    if config["optimizer"] == "adam":
        return tf.keras.optimizers.Adam(learning_rate=config["adam_lr"])
    if config["optimizer"] == "sgd":
        schedule = DecaySGD(config["sgd_alpha_0"], config["sgd_a"], config["sgd_b"], config["sgd_r"])
        return tf.keras.optimizers.SGD(learning_rate=schedule)
    raise ValueError(f"unknown optimizer {config['optimizer']}")

def uniform_type_sampler(game_def):
    """Uniform on player i's [theta_low, theta_high]."""
    def sampler(rng, i, size):
        return rng.uniform(game_def["theta_low"][i], game_def["theta_high"][i], size=size)
    return sampler

def pareto_type_sampler(game_def, pareto_a=3.0, cap_quantile=0.999):
    """Lomax truncated at cap_quantile, rescaled to player i's [theta_low, theta_high]."""
    cap = (1 - cap_quantile) ** (-1 / pareto_a) - 1
    def sampler(rng, i, size):
        lo, hi = game_def["theta_low"][i], game_def["theta_high"][i]
        u = rng.uniform(0.0, cap_quantile, size=size)
        y = (1 - u) ** (-1 / pareto_a) - 1
        return lo + (hi - lo) * y / cap
    return sampler

def make_sampler(dist, config, game_def):
    if dist == "uniform":
        return uniform_type_sampler(game_def)
    if dist == "pareto":
        return pareto_type_sampler(game_def, config.pareto_a, config.cap_quantile)
    if dist == "uniform+pareto":
        return mixture_sampler([
            uniform_type_sampler(game_def),
            pareto_type_sampler(game_def, config.pareto_a, config.cap_quantile),
        ])
    if dist == "stratified_pareto":
        return stratified_pareto_sampler(game_def, config.pareto_a, config.cap_quantile,
                                         config.strat_bins, config.strat_floor)
    raise ValueError(f"unknown distribution {dist}")

def make_two_sample_data(game_def, config):
    """theta_own: (m, N, d) from own_dist, for coverage of each player's type space.
    theta_opp: (tilde_m, N, d) from opp_dist, the prior used for the expectation over opponents."""
    rng = np.random.default_rng(config.seed)
    own_sampler = make_sampler(config.own_dist, config, game_def)
    opp_sampler = make_sampler(config.opp_dist, config, game_def)
    n, d = game_def["n"], game_def["d"]
    theta_own = np.stack([own_sampler(rng, i, (config.m, d[i])) for i in range(n)], axis=1)
    theta_opp = np.stack([opp_sampler(rng, i, (config.tilde_m, d[i])) for i in range(n)], axis=1)
    return tf.cast(theta_own, tf.float32), tf.cast(theta_opp, tf.float32)

def pareto_type_density(game_def, pareto_a=3.0, cap_quantile=0.999):
    """Unnormalized density of pareto_type_sampler on player i's type range."""
    cap = (1 - cap_quantile) ** (-1 / pareto_a) - 1
    def density(i, theta):
        lo, hi = game_def["theta_low"][i], game_def["theta_high"][i]
        y = (np.asarray(theta) - lo) / (hi - lo) * cap
        return (1 + y) ** (-(pareto_a + 1))
    return density


def make_density(dist, config, game_def):
    """None means uniform (equal weights)."""
    if dist == "uniform":
        return None
    if dist == "pareto":
        return pareto_type_density(game_def, config.pareto_a, config.cap_quantile)
    raise ValueError(f"unknown distribution {dist}")


def mixture_sampler(samplers):
    """Draws an equal share of the samples from each sampler, then shuffles them together."""
    def sampler(rng, i, size):
        n, k = size[0], len(samplers)
        counts = [n // k + (1 if r < n % k else 0) for r in range(k)]
        parts = [s(rng, i, (c,) + tuple(size[1:])) for s, c in zip(samplers, counts)]
        out = np.concatenate(parts, axis=0)
        return out[rng.permutation(n)]
    return sampler

def stratified_pareto_sampler(game_def, pareto_a=3.0, cap_quantile=0.999, n_bins=100, floor=2):
    """Own types: at least `floor` samples in each of n_bins equal-width intervals of
    [theta_low, theta_high]; the rest allocated in proportion to the Pareto probability of each
    interval. Within an interval, samples follow the Pareto restricted to it.
    Guarantees a fill distance of at most the interval width."""
    cap = (1 - cap_quantile) ** (-1 / pareto_a) - 1

    def cdf(theta, lo, hi):            # CDF of the rescaled, truncated Lomax
        y = (theta - lo) / (hi - lo) * cap
        return (1 - (1 + y) ** (-pareto_a)) / cap_quantile

    def inv_cdf(u, lo, hi):
        y = (1 - cap_quantile * u) ** (-1 / pareto_a) - 1
        return lo + (hi - lo) * y / cap

    def sampler(rng, i, size):
        lo, hi = float(game_def["theta_low"][i]), float(game_def["theta_high"][i])
        n = size[0]
        if n < floor * n_bins:
            raise ValueError(f"need at least {floor * n_bins} samples for {n_bins} bins x {floor}")
        edges = np.linspace(lo, hi, n_bins + 1)
        F = cdf(edges, lo, hi)
        probs = np.diff(F) / (F[-1] - F[0])
        counts = floor + rng.multinomial(n - floor * n_bins, probs)
        u = np.concatenate([rng.uniform(F[k], F[k + 1], c) for k, c in enumerate(counts)])
        theta = inv_cdf(u, lo, hi)
        return theta[rng.permutation(n)].reshape(size)
    return sampler
