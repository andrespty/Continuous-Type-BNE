import tensorflow as tf
from neural_nets.base_networks.dual_NN import DualNet
from neural_nets.base_networks.primal_NN import PrimalNet

class PrimalDualAsymmetricNet:
    def __init__(self, input_dim=[1,1], output_dim=[1,1], constraints=[2,2], hidden=[64, 64], alpha=1e-6, seed=15):
        self.primal_net_p1 = PrimalNet(input_dim[0], output_dim[0], hidden[0], alpha=alpha, seed=seed, scaling_factor=100, name="Primal_P1")
        self.primal_net_p2 = PrimalNet(input_dim[1], output_dim[1], hidden[1], alpha=alpha, seed=seed+1, scaling_factor=100/51, name="Primal_P2")
        self.dual_net_1 = DualNet(n=constraints[0])
        self.dual_net_2 = DualNet(n=constraints[1])
        self.model = self._build(sum(input_dim))

    def _build(self, input_dim):
        inputs = tf.keras.Input(shape=(input_dim,), name='main_input_layer')

        slice_1 =  tf.keras.layers.Lambda(lambda x: x[:, 0:1], name='slice_1')(inputs)
        slice_2 =  tf.keras.layers.Lambda(lambda x: x[:, 1:2], name='slice_2')(inputs)
        primal_p1 = self.primal_net_p1.model(slice_1)
        primal_p2 = self.primal_net_p2.model(slice_2)
        
        dual_1 = self.dual_net_1(slice_1)
        dual_2 = self.dual_net_2(slice_2)

        y = tf.keras.layers.Concatenate(axis=-1)([primal_p1, primal_p2, dual_1, dual_2])
        return tf.keras.Model(inputs=inputs, outputs=[y], name='DUAL_ASYMMETRIC_NN')
    