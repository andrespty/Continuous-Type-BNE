import tensorflow as tf

# @tf.keras.utils.register_keras_serializable()
# class DualNet(tf.keras.layers.Layer):
#     def __init__(self, n, hidden=64, **kwargs):
#         super().__init__(**kwargs)
#         self.mlp = tf.keras.Sequential([
#             tf.keras.layers.Dense(hidden, activation='tanh'),
#             tf.keras.layers.Dense(hidden, activation='tanh'),
#             tf.keras.layers.Dense(hidden, activation='tanh'),
#             tf.keras.layers.Dense(hidden, activation='tanh'),
#             tf.keras.layers.Dense(n, activation='softplus')  # ensures λ ≥ 0
#         ])

#     def call(self, theta):
#         return self.mlp(theta)

#     def get_config(self):
#         config = super().get_config()
#         config.update({'n': self.n, 'hidden': self.hidden})
#         return config
    
@tf.keras.utils.register_keras_serializable()
class DualNet(tf.keras.layers.Layer):
    def __init__(self, n, hidden=64, **kwargs):
        super().__init__(**kwargs)
        self.n = n          # store these
        self.hidden = hidden

    def build(self, input_shape):
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden, activation='tanh'),
            tf.keras.layers.Dense(self.hidden, activation='tanh'),
            tf.keras.layers.Dense(self.hidden, activation='tanh'),
            tf.keras.layers.Dense(self.hidden, activation='tanh'),
            tf.keras.layers.Dense(self.n, activation='softplus')
        ])
        super().build(input_shape)

    def call(self, theta):
        return self.mlp(theta)

    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n, 'hidden': self.hidden})
        return config
