import tensorflow as tf
from neural_nets.base_networks.dual_NN import DualNet
from neural_nets.base_networks.primal_NN import PrimalNet


class PrimalDualNet:
    def __init__(self, input_dim=1, output_dim=1, constraints=2, hidden=64, alpha=1e-6, seed=15):
        self.primal_net = PrimalNet(input_dim, output_dim, hidden, alpha=alpha, seed=seed)
        self.dual_net = DualNet(n=constraints)
        self.model = self._build(input_dim)

    def _build(self, input_dim):
        inputs = tf.keras.Input(shape=(input_dim,), name='main_input_layer')
        primal = self.primal_net.model(inputs)
        dual = self.dual_net(inputs)
        y = tf.keras.layers.Concatenate(axis=-1)([primal, dual])
        return tf.keras.Model(inputs=inputs, outputs=[y], name='DUAL_NN')

