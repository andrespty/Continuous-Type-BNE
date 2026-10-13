import matplotlib.pyplot as plt
import tensorflow as tf

class DecaySGD(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, alpha_0, a, b, r):
        super().__init__()
        self.alpha_0 = alpha_0
        self.a = a
        self.b = b
        self.r = r

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.alpha_0 /(self.a + self.b * tf.pow(step, self.r))
    
    def get_config(self):
        return {
            'alpha_0': self.alpha_0,
            'a': self.a,
            'b': self.b,
            'r': self.r
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='custom_loss', alpha=1e-5, upper_b = 100):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        self.alpha = alpha
        self.upper_b = upper_b

    def complementarity_func(self, a, b):
        return tf.sqrt(tf.square(a) + tf.square(b) + 1e-6) - (a + b)

    def call_two_sample(self, theta_own, y_own, y_opp):
        """theta_own, y_own: own types and the model's output at them (Bo rows).
        y_opp: the model's output at the opponent-prior types (Bp rows).
        Output columns: [action, d_ub, d_lb]."""
        a = tf.reshape(y_own[:, 0], [-1, 1])
        d_ub = tf.reshape(y_own[:, 1], [-1, 1])
        d_lb = tf.reshape(y_own[:, 2], [-1, 1])
        a_o = tf.reshape(y_opp[:, 0], [1, -1])                                   # (1, Bp)

        v = tf.reduce_mean(a_o / tf.square(a + a_o), axis=1, keepdims=True)      # (Bo, 1)
        grad = v - tf.reshape(theta_own, [-1, 1])

        # eng_primal = (grad + d_ub - d_lb) ** 2
        eng_primal = (grad - d_ub + d_lb) ** 2 # claude suggested to use this. says it was wrong
        eng_dual = self.complementarity_func(d_ub, -(a - self.upper_b)) ** 2
        eng_dual2 = self.complementarity_func(d_lb, a) ** 2
        return 2 * (tf.reduce_mean(eng_primal) + tf.reduce_mean(eng_dual + eng_dual2))

    def call(self, types, actions):
        # single-sample: the batch is its own opponent pool
        return self.call_two_sample(types, actions, actions)
    
class CustomLossP3(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='custom_loss',alpha=1e-5, m=250, upper_b=100):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        self.alpha = alpha
        self.m = m
        self.upper_b = upper_b
        
    def func_to_be_integrated3(self, actions):
        actions_1 = tf.reshape(actions, [-1,1,1])
        actions_2 = tf.reshape(actions[:self.m], [1,-1,1])
        actions_3 = tf.reshape(actions[-self.m:], [1,1,-1])
        sum_squared = tf.square(actions_1+actions_2+actions_3)
        return (actions_2+actions_3)/(sum_squared)
    
    def complementarity_func(self, a, b):
        return tf.sqrt(tf.square(a) + tf.square(b) + 1e-6) - (a + b)

    def call_two_sample(self, theta_own, y_own, y_opp):
        """Own types/outputs against an opponent pool. As before, the two opponents come from
        the first m and the last m rows of the pool. Output columns: [action, d_ub, d_lb]."""
        a = tf.reshape(y_own[:, 0], [-1, 1])
        d_ub = tf.reshape(y_own[:, 1], [-1, 1])
        d_lb = tf.reshape(y_own[:, 2], [-1, 1])

        a1 = tf.reshape(a, [-1, 1, 1])
        a2 = tf.reshape(y_opp[:self.m, 0], [1, -1, 1])
        a3 = tf.reshape(y_opp[-self.m:, 0], [1, 1, -1])
        v = (a2 + a3) / tf.square(a1 + a2 + a3)
        grad = tf.reshape(tf.reduce_mean(v, axis=[1, 2]), [-1, 1]) - tf.reshape(theta_own, [-1, 1])

        eng_primal = (grad - d_ub + d_lb) ** 2
        eng_dual = self.complementarity_func(d_ub, -(a - self.upper_b)) ** 2
        eng_dual2 = self.complementarity_func(d_lb, a) ** 2
        return 3 * (tf.reduce_mean(eng_primal) + tf.reduce_mean(eng_dual + eng_dual2))

    def call(self, types, actions):
        return self.call_two_sample(types, actions, actions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "m":self.m,
        })
        return config

class CustomLoss_A2(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='custom_loss', alpha=1e-5):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        self.alpha = alpha

    def complementarity_func(self, a, b):
        return tf.sqrt(tf.square(a) + tf.square(b)) - (a + b)

    def _terms_two_sample(self, types_own, y_own, y_opp):
        """Per-type Psi for each player, shape (N, 1) each. Own types/outputs in types_own/y_own;
        opponents' actions from y_opp (player 2's for player 1, player 1's for player 2).
        Output columns: [a1, a2, d_lb1, d_ub1, d_lb2, d_ub2]."""
        a1, a2 = y_own[:, 0:1], y_own[:, 1:2]
        d_lb1, d_ub1 = y_own[:, 2:3], y_own[:, 3:4]
        d_lb2, d_ub2 = y_own[:, 4:5], y_own[:, 5:6]
        th1 = tf.reshape(types_own[:, 0], [-1, 1])
        th2 = tf.reshape(types_own[:, 1], [-1, 1])
        b1 = tf.reshape(y_opp[:, 0], [1, -1])          # player 1's actions (opponents of player 2)
        b2 = tf.reshape(y_opp[:, 1], [1, -1])          # player 2's actions (opponents of player 1)

        grad1 = tf.reduce_mean(b2 / tf.square(a1 + b2), axis=1, keepdims=True) - th1
        grad2 = tf.reduce_mean(b1 / tf.square(a2 + b1), axis=1, keepdims=True) - th2

        # keep the same multiplier sign convention as the loss these models were trained with
        eng1 = (grad1 + d_lb1 - d_ub1) ** 2
        eng2 = (grad2 + d_lb2 - d_ub2) ** 2

        dual1 = (self.complementarity_func(d_lb1, a1) ** 2
                 + self.complementarity_func(d_ub1, 100 - a1) ** 2)
        dual2 = (self.complementarity_func(d_lb2, a2) ** 2
                 + self.complementarity_func(d_ub2, (100 / 51) - a2) ** 2)
        return eng1 + dual1, eng2 + dual2

    def player_merits(self, types, actions):
        psi1, psi2 = self._terms_two_sample(types, actions, actions)
        return [tf.reduce_mean(psi1), tf.reduce_mean(psi2)]

    def call(self, types, actions):
        return tf.add_n(self.player_merits(types, actions))


class CustomLossP3Fast(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='custom_loss',
                 alpha=1e-5, n_shifts=8, action_high=100.0):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        self.alpha = alpha
        self.n_shifts = n_shifts
        self.action_high = action_high

    def complementarity_func(self, a, b):
        return tf.sqrt(tf.square(a) + tf.square(b)) - (a + b)

    def call_two_sample(self, theta_own, y_own, y_opp):
        """Own types against opponent pairs (i, i+k) of the pool, k = 1..n_shifts.
        Output columns: [action, d_ub, d_lb]."""
        a = tf.reshape(y_own[:, 0], [-1, 1])
        d_ub = tf.reshape(y_own[:, 1], [-1, 1])
        d_lb = tf.reshape(y_own[:, 2], [-1, 1])

        b = y_opp[:, 0]                                             # opponent bids, (N,)
        S = tf.concat([b + tf.roll(b, shift=k, axis=0)              # total of each opponent pair
                       for k in range(1, self.n_shifts + 1)], axis=0)
        S = tf.reshape(S, [1, -1])                                  # (1, n_shifts * N)

        grad = tf.reduce_mean(S / tf.square(a + S), axis=1, keepdims=True) - tf.reshape(theta_own, [-1, 1])

        eng_primal = (grad - d_ub + d_lb) ** 2
        eng_dual = self.complementarity_func(d_ub, self.action_high - a) ** 2
        eng_dual2 = self.complementarity_func(d_lb, a) ** 2
        return 3 * (tf.reduce_mean(eng_primal) + tf.reduce_mean(eng_dual + eng_dual2))

    def call(self, types, actions):
        return self.call_two_sample(types, actions, actions)

    def get_config(self):
        config = super().get_config()
        config.update({"n_shifts": self.n_shifts, "action_high": self.action_high})
        return config


class NMultiMarketCournotLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='custom_loss',
                 alpha_m=(10.0, 10.0), beta_m=(1.0, 1.0), qbar_m=(10.0, 10.0),
                 lambda_m=(0.0, 0.0), Gammas=None):
        super().__init__(reduction=reduction, name=name)
        self.alpha_m_val, self.beta_m_val = alpha_m, beta_m
        self.qbar_m_val, self.lambda_m_val, self.Gammas_val = qbar_m, lambda_m, Gammas

        self.alpha_m = tf.constant(alpha_m, dtype=tf.float32)
        self.beta_m = tf.constant(beta_m, dtype=tf.float32)
        self.qbar_m = tf.constant(qbar_m, dtype=tf.float32)
        self.lambda_m = tf.constant(lambda_m, dtype=tf.float32)
        self.Gammas = [tf.constant(G, dtype=tf.float32) for G in Gammas]  # one (z, d) matrix per player

    def get_config(self):
        config = super().get_config()
        config.update({'alpha_m': self.alpha_m_val, 'beta_m': self.beta_m_val,
                       'qbar_m': self.qbar_m_val, 'lambda_m': self.lambda_m_val,
                       'Gammas': self.Gammas_val})
        return config

    def player_merits(self, types, actions):
        """Per-player merit: batch mean of the sum over markets of that player's terms."""
        n, z = len(self.Gammas), int(self.alpha_m.shape[0])
        off = n * z
        p = [actions[:, i * z:(i + 1) * z] for i in range(n)]
        d_ub = [actions[:, off + 2 * z * i: off + 2 * z * i + z] for i in range(n)]
        d_lb = [actions[:, off + 2 * z * i + z: off + 2 * z * (i + 1)] for i in range(n)]
        mean_p = [tf.reduce_mean(p_i, axis=0, keepdims=True) for p_i in p]

        merits = []
        for i in range(n):
            cost_i = tf.matmul(types[:, i, :], self.Gammas[i], transpose_b=True)
            opp_sum = tf.add_n([mean_p[j] for j in range(n) if j != i])
            terms = self._player_terms(p[i], cost_i, opp_sum, d_ub[i], d_lb[i])
            merits.append(tf.reduce_mean(tf.reduce_sum(terms, axis=1)))
        return merits


    def complementarity_func(self, a, b):
        return tf.sqrt(tf.square(a) + tf.square(b) + 1e-6) - (a + b)

    def _player_terms(self, p_i, cost_i, opp_sum, d_ub_i, d_lb_i):
        grad = self.alpha_m - cost_i - (2 * self.beta_m + self.lambda_m) * p_i - self.beta_m * opp_sum
        grad_dual = -(p_i - self.qbar_m)   # upper bound slack
        grad_dual2 = -(0.0 - p_i)          # lower bound slack

        eng_primal = (grad - d_ub_i + d_lb_i) ** 2
        eng_dual = self.complementarity_func(d_ub_i, grad_dual) ** 2
        eng_dual2 = self.complementarity_func(d_lb_i, grad_dual2) ** 2
        return eng_primal + eng_dual + eng_dual2

    
    def call(self, types, actions):
        return tf.add_n(self.player_merits(types, actions))