from utils.game import Game
from utils.loss import EquilibriumLoss
from utils.stepsize import CustomSchedule1
from neural_nets.asymmetric_models import PrimalDualAsymmetricNet
import tensorflow as tf


class AsymmetricRentSeeking(Game):
    """
    Asymmetric 2-player rent-seeking game with a continuous type space.

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

    def nn_model(self):
        return PrimalDualAsymmetricNet(input_dim=[1,1], output_dim=[1,1], constraints=[2,2], hidden=[64, 64], alpha=1e-6, seed=15)

    def loss_fn(self):
        @tf.keras.utils.register_keras_serializable()
        class LossA2(EquilibriumLoss):
            def func_to_be_integrated2(self, actions):
                actions_1 = tf.reshape(actions[:,0], [-1,1])
                actions_2 = tf.reshape(actions[:,1], [1,-1])
                sum_squared = tf.square(actions_1+actions_2)    
                return [(actions_2)/(sum_squared), (actions_1)/(sum_squared)]
            
            def primal_energy(self, types, actions):
                P_actions  = tf.reshape(actions[:,0:2], [-1,2,1])
                d_lb_P1    = tf.reshape(actions[:,2], [-1,1])
                d_ub_P1    = tf.reshape(actions[:,3], [-1,1])
                d_lb_P2    = tf.reshape(actions[:,4], [-1,1])
                d_ub_P2    = tf.reshape(actions[:,5], [-1,1])
                
                P1_theta = tf.reshape(types[:,0],[-1,1])
                P2_theta = tf.reshape(types[:,1],[-1,1])

                v1, v2 = self.func_to_be_integrated2(P_actions)

                grad1 = tf.reshape(tf.reduce_mean(v1, axis=[1],keepdims=False), [-1, 1]) - P1_theta
                grad2 = tf.reshape(tf.reduce_mean(v2, axis=[0],keepdims=False), [-1, 1]) - P2_theta
                
                eng_primal_p1 = tf.reduce_mean(tf.square(grad1 + d_ub_P1 - d_lb_P1))  
                eng_primal_p2 = tf.reduce_mean(tf.square(grad2 + d_ub_P2 - d_lb_P2))  
                
                return eng_primal_p1 + eng_primal_p2
            
            def dual_energy(self, types, actions):
                P1_actions = tf.reshape(actions[:,0], [-1,1])
                P2_actions = tf.reshape(actions[:,1], [-1,1])
                d_lb_P1    = tf.reshape(actions[:,2], [-1,1])
                d_ub_P1    = tf.reshape(actions[:,3], [-1,1])
                d_lb_P2    = tf.reshape(actions[:,4], [-1,1])
                d_ub_P2    = tf.reshape(actions[:,5], [-1,1])

                g_dual_lb_p1 =  -(0 - P1_actions)
                g_dual_lb_p2 =  -(0 - P2_actions)
                g_dual_ub_p1 =  -(P1_actions - 100) 
                g_dual_ub_p2 =  -(P2_actions - (100/51)) 

                eng_dual1 = self.complementarity_func(d_lb_P1, g_dual_lb_p1) ** 2
                eng_dual2 = self.complementarity_func(d_lb_P2, g_dual_lb_p2) ** 2
                eng_dual3 = self.complementarity_func(d_ub_P1, g_dual_ub_p1) ** 2 
                eng_dual4 = self.complementarity_func(d_ub_P2, g_dual_ub_p2) ** 2

                return tf.reduce_mean(eng_dual1 + eng_dual2 + eng_dual3 + eng_dual4)
        return LossA2(n_players=2, isSymmetric=False)
        
    def optimizer(self, lr=1e-4, isAdam=True):
        if isAdam:
            return tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            alpha_0 = 0.01
            a = 1
            b = 1e-4
            r = 0.7
            lr_schedule = CustomSchedule1(alpha_0, a, b, r)
            return tf.keras.optimizers.SGD(learning_rate=lr_schedule)