import tensorflow as tf

class DualNet(tf.keras.layers.Layer):
    def __init__(self, n, hidden=64):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(n, activation='softplus')  # ensures λ ≥ 0
        ])

    def call(self, theta):
        return self.mlp(theta)
