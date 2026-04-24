from utils.game import Game
from utils.loss import EquilibriumLoss
from utils.stepsize import CustomSchedule1
from neural_nets.symmetric_models import PrimalDualNet
import tensorflow as tf


class SymmetricRentSeeking(Game):
    """
    Symmetric N-player rent-seeking game with a continuous type space.

    Each player observes a private type θ ~ Uniform(type_bounds)
    and chooses an action a ∈ action_bounds. The equilibrium
    conditions are encoded in CustomLoss (2-player) or
    CustomLossP3 (3-player).

    Parameters
    ----------
    n_players : int
        Number of players. Currently supports 2 or 3.
    action_bounds : tuple[float, float]
        (lower, upper) bounds on the action space.
    type_bounds : tuple[float, float]
        (lower, upper) support of the type distribution.
    """

    def __init__(self, n_players: int = 2):
        if n_players not in (2, 3):
            raise ValueError(f"RentSeekingGame supports 2 or 3 players, got {n_players}.")
        self._n_players = n_players

    def nn_model(self):
        return PrimalDualNet(input_dim=1, output_dim=1, constraints=2, hidden=64, alpha=1e-6, seed=15)

    def loss_fn(self):
        if self._n_players == 2:
            class CustomLoss(EquilibriumLoss):
                def func_to_be_integrated2(self, actions):
                    actions_1 = tf.reshape(actions, [-1, 1, 1])
                    actions_2 = tf.reshape(actions, [1, -1, 1])
                    sum_squared = tf.square(actions_1 + actions_2)
                    return actions_2 / sum_squared

                def primal_energy(self, types, actions):
                    d_ub = tf.reshape(actions[:,1], [-1,1])
                    d_lb = tf.reshape(actions[:,2], [-1,1])
                    primal_actions = tf.reshape(actions[:,0], [-1,1])
                    v = self.func_to_be_integrated2(primal_actions)
                    grad = tf.reshape(tf.reduce_mean(v, axis=1, keepdims=False), [-1, 1]) - types
                    return tf.reduce_mean(tf.square(grad + d_ub - d_lb))  

                def dual_energy(self, types, actions):
                    primal_actions = tf.reshape(actions[:,0], [-1,1])
                    d_ub = tf.reshape(actions[:,1], [-1,1])
                    d_lb = tf.reshape(actions[:,2], [-1,1])

                    grad_dual =  -(primal_actions - 100) 
                    grad_dual2 =  -(0 - primal_actions) 

                    eng_dual = self.complementarity_func(d_ub, grad_dual) ** 2
                    eng_dual2 = self.complementarity_func(d_lb, grad_dual2) ** 2 
                    return tf.reduce_mean(eng_dual + eng_dual2) 
            return CustomLoss(n_players=self._n_players, isSymmetric=True)
        elif self._n_players == 3:
            class CustomLossP3(EquilibriumLoss):
                def __init__(self, name='custom_loss', alpha=1e-5, m=250, **kwargs):
                    super().__init__(name=name, **kwargs)
                    self.alpha = alpha
                    self.m = m
                
                def get_config(self):
                    config = super().get_config()
                    config.update({
                        "m":self.m,
                    })
                    return config

                def func_to_be_integrated3(self, actions):
                    actions_1 = tf.reshape(actions, [-1,1,1])
                    actions_2 = tf.reshape(actions[:self.m], [1,-1,1])
                    actions_3 = tf.reshape(actions[-self.m:], [1,1,-1])
                    sum_squared = tf.square(actions_1+actions_2+actions_3)
                    return (actions_2+actions_3)/(sum_squared)

                def primal_energy(self, types, actions):
                    d_ub = tf.reshape(actions[:,1], [-1,1])
                    d_lb = tf.reshape(actions[:,2], [-1,1])
                    primal_actions = tf.reshape(actions[:,0], [-1,1])
                    v = self.func_to_be_integrated3(primal_actions)
                    grad = tf.reshape(tf.reduce_mean(v, axis=[1,2], keepdims=False), [-1, 1]) - types
                    return tf.reduce_mean(tf.square(grad + d_ub - d_lb))  

                def dual_energy(self, types, actions):
                    primal_actions = tf.reshape(actions[:,0], [-1,1])
                    d_ub = tf.reshape(actions[:,1], [-1,1])
                    d_lb = tf.reshape(actions[:,2], [-1,1])

                    grad_dual =  -(primal_actions - 100) 
                    grad_dual2 =  -(0 - primal_actions) 

                    eng_dual = self.complementarity_func(d_ub, grad_dual) ** 2
                    eng_dual2 = self.complementarity_func(d_lb, grad_dual2) ** 2 
                    return tf.reduce_mean(eng_dual + eng_dual2) 
            return CustomLossP3(n_players=self._n_players, isSymmetric=True)
        
    def optimizer(self, lr=1e-5, isAdam=True):
        if isAdam:
            return tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            alpha_0 = 0.01
            a = 1
            b = 1e-4
            r = 0.7
            lr_schedule = CustomSchedule1(alpha_0, a, b, r)
            return tf.keras.optimizers.SGD(learning_rate=lr_schedule)