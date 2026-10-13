"""Game definitions for evaluation: each evaluator knows its game's utility rho, its best response,
and how to sample own types and opponents. Metrics, tables, and figures live elsewhere."""
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from utils.utils import make_sampler, make_density


class ModelEvaluator(ABC):
    def __init__(self, model, game_def, w_t=10000, m_opp=1000, sampler=None, density=None, seed=0):
        self.model = model
        self.game_def = game_def
        self.w_t = w_t          # grid size for own types and for actions
        self.m_opp = m_opp      # number of sampled opponent types
        self.sampler = sampler  # opponent prior: sampler(rng, i, size); None = uniform
        self.density = density  # opponent prior density(i, theta); None = uniform
        self.seed = seed

    @abstractmethod
    def rho(self, a_own, opponent_pool, theta): ...

    def matchups(self):
        """(own player, opponent player) pairs to evaluate."""
        if self.game_def["symmetric"]:
            return [(0, 0)]
        return [(0, 1), (1, 0)]

    def action(self, i, theta):
        player_model = self.model.get_layer(f"Player{i + 1}")
        return player_model(tf.reshape(theta, [-1, 1]), training=False)[:, 0:1]

    def theta_grid(self, i):
        lo, hi = self.game_def["theta_low"][i], self.game_def["theta_high"][i]
        return tf.constant(np.linspace(lo, hi, self.w_t).reshape(-1, 1), dtype=tf.float32)

    def a_w(self, i):
        lo, hi = self.game_def["action_low"][i], self.game_def["action_high"][i]
        return tf.constant(np.linspace(lo, hi, self.w_t).reshape(-1, 1), dtype=tf.float32)

    def opponent_pool(self, j):
        """Actions of player j's network at m_opp types drawn from the prior."""
        rng = np.random.default_rng(self.seed)
        if self.sampler is None:
            theta_j = rng.uniform(self.game_def["theta_low"][j], self.game_def["theta_high"][j],
                                  size=(self.m_opp, 1))
        else:
            theta_j = self.sampler(rng, j, (self.m_opp, 1))
        return self.action(j, tf.constant(theta_j, dtype=tf.float32))

    def best_response(self, i, opponent_pool, theta):
        """Grid search over player i's actions [action_low, action_high]."""
        a_w = self.a_w(i)
        rho_hat_matrix = self.rho(a_w, opponent_pool, tf.reshape(theta, [1, -1]))  # (actions, types)
        max_index = tf.argmax(rho_hat_matrix, axis=0)
        a_hat = tf.gather(tf.reshape(a_w, [-1]), max_index)
        rho_hat = tf.reduce_max(rho_hat_matrix, axis=0)
        return tf.reshape(a_hat, [-1, 1]), tf.reshape(rho_hat, [-1, 1])


class RentSeeking2PlayerEvaluator(ModelEvaluator):
    """sym2 and asym2: u = E[a1 / (a1 + a2)] - theta * a1."""

    def rho(self, a_own, opponent_pool, theta):
        a1 = tf.reshape(a_own, [-1, 1])
        a2 = tf.reshape(opponent_pool, [1, -1])
        d = tf.reduce_mean(tf.math.divide_no_nan(a1, a1 + a2), axis=1, keepdims=True)
        return d - a1 * theta


class RentSeeking3PlayerEvaluator(ModelEvaluator):
    """sym3: u = E[a1 / (a1 + a2 + a3)] - theta * a1, both opponents from the same pool.
    Memory grows as w_t * m_opp^2, so keep m_opp small (default 100)."""

    def __init__(self, model, game_def, w_t=10000, m_opp=100, sampler=None, density=None, seed=0):
        super().__init__(model, game_def, w_t=w_t, m_opp=m_opp, sampler=sampler,
                         density=density, seed=seed)

    def rho(self, a_own, opponent_pool, theta):
        a1 = tf.reshape(a_own, [-1, 1, 1])
        a2 = tf.reshape(opponent_pool, [1, -1, 1])
        a3 = tf.reshape(opponent_pool, [1, 1, -1])
        d = tf.reshape(tf.reduce_mean(tf.math.divide_no_nan(a1, a1 + a2 + a3), axis=[1, 2]), [-1, 1])
        return d - tf.reshape(a_own, [-1, 1]) * theta


class CournotEvaluator(ModelEvaluator):
    """Multi-market Cournot. Profit per market:
        (alpha - beta*(q + Q_opp)) q - c(theta) q - (lambda/2) q^2,   c(theta) = Gamma @ theta.
    Linear in opponents' quantities -> only E[Q_opp] matters.
    Separable across markets -> exact best response per market.
    With opponents at the equilibrium means mu, the best response IS the closed-form BNE."""

    def __init__(self, model, game_def, n_points=100, m_opp=1000, sampler=None, density=None, seed=0):
        super().__init__(model, game_def, m_opp=m_opp, sampler=sampler, density=density, seed=seed)
        self.n_points = n_points
        loss = game_def["loss_func"]
        self.alpha_m, self.beta_m = loss.alpha_m, loss.beta_m
        self.lambda_m, self.qbar_m = loss.lambda_m, loss.qbar_m
        self.Gammas = loss.Gammas
        self.n = game_def["n"]
        self.z = game_def["z"][0]

    # ---------- types and actions ----------

    def sample_types(self, i, size, rng):
        if self.sampler is None:
            theta = rng.uniform(self.game_def["theta_low"][i], self.game_def["theta_high"][i],
                                size=(size, self.game_def["d"][i]))
        else:
            theta = self.sampler(rng, i, (size, self.game_def["d"][i]))
        return tf.constant(theta, dtype=tf.float32)

    def theta_grid(self, i, n_points=None):
        """g x g grid over player i's (theta1, theta2): the two mesh grids and the (g*g, 2) tensor."""
        n_points = n_points or self.n_points
        lo, hi = self.game_def["theta_low"][i], self.game_def["theta_high"][i]
        th1_grid, th2_grid = np.meshgrid(np.linspace(lo[0], hi[0], n_points),
                                         np.linspace(lo[1], hi[1], n_points))
        theta = tf.constant(np.stack([th1_grid.ravel(), th2_grid.ravel()], axis=1), dtype=tf.float32)
        return th1_grid, th2_grid, theta

    def action(self, i, theta):
        player_model = self.model.get_layer(f"Player{i + 1}")
        return player_model(theta, training=False)[:, :self.game_def["z"][i]]

    # ---------- closed-form equilibrium ----------

    def mean_types(self):
        """E[theta_i] per player, shape (n, d)."""
        if self.sampler is None:
            return (np.asarray(self.game_def["theta_low"]) + np.asarray(self.game_def["theta_high"])) / 2
        rng = np.random.default_rng(self.seed)
        return np.stack([self.sample_types(i, 100_000, rng).numpy().mean(axis=0) for i in range(self.n)])

    def equilibrium_means(self):
        """Equilibrium expected quantities mu, shape (n, z). Per market m, solves the interior BNE
        (2b + l) mu_i + b * sum_{j != i} mu_j = alpha - Gamma_i[m] @ thetabar_i."""
        alpha, beta, lam = self.alpha_m.numpy(), self.beta_m.numpy(), self.lambda_m.numpy()
        thetabar = self.mean_types()
        mu = np.zeros((self.n, self.z))
        for m in range(self.z):
            A = beta[m] * np.ones((self.n, self.n)) + (beta[m] + lam[m]) * np.eye(self.n)
            b = np.array([alpha[m] - self.Gammas[i].numpy()[m] @ thetabar[i] for i in range(self.n)])
            mu[:, m] = np.linalg.solve(A, b)
        return mu

    def opponent_pool(self, i, against="equilibrium"):
        """Expected total opponent quantity per market, shape (1, z).
        against='equilibrium': opponents at the closed-form means mu.
        against='nn': opponents play their trained networks."""
        others = [j for j in range(self.n) if j != i]
        if against == "equilibrium":
            mu = self.equilibrium_means()
            return tf.constant(mu[others].sum(axis=0, keepdims=True), dtype=tf.float32)
        rng = np.random.default_rng(self.seed)
        return tf.add_n([
            tf.reduce_mean(self.action(j, self.sample_types(j, self.m_opp, rng)), axis=0, keepdims=True)
            for j in others
        ])

    # ---------- utility and best response ----------

    def rho(self, a_own, opponent_pool, theta, i):
        cost = tf.matmul(theta, self.Gammas[i], transpose_b=True)
        price = self.alpha_m - self.beta_m * (a_own + opponent_pool)
        profit = (price - cost) * a_own - 0.5 * self.lambda_m * a_own ** 2
        return tf.reduce_sum(profit, axis=1, keepdims=True)

    def best_response(self, i, opponent_pool, theta):
        cost = tf.matmul(theta, self.Gammas[i], transpose_b=True)
        q = (self.alpha_m - cost - self.beta_m * opponent_pool) / (2 * self.beta_m + self.lambda_m)
        a_hat = tf.clip_by_value(q, 0.0, self.qbar_m)
        return a_hat, self.rho(a_hat, opponent_pool, theta, i)

    def player_grids(self, i, n_points=None, against="equilibrium"):
        """NN and best-response actions on the type grid, each shape (g, g, z)."""
        th1_grid, th2_grid, theta = self.theta_grid(i, n_points)
        a_nn = self.action(i, theta).numpy()
        a_hat, _ = self.best_response(i, self.opponent_pool(i, against), theta)
        shape = th1_grid.shape + (self.z,)
        return th1_grid, th2_grid, a_nn.reshape(shape), a_hat.numpy().reshape(shape)


EVALUATORS = {
    "sym2": RentSeeking2PlayerEvaluator,
    "asym2": RentSeeking2PlayerEvaluator,
    "sym3": RentSeeking3PlayerEvaluator,
    "cournot": CournotEvaluator,
}


def make_evaluator(model, config, game_def, **kwargs):
    """Opponent pool and prior weights follow config.opp_dist (uniform by default)."""
    kwargs.setdefault("sampler", make_sampler(config.opp_dist, config, game_def))
    kwargs.setdefault("density", make_density(config.opp_dist, config, game_def))
    return EVALUATORS[config.game](model, game_def, **kwargs)