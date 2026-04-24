import tensorflow as tf
from tensorflow.keras.constraints import UnitNorm

class PrimalNet:
    def __init__(self, 
        input_dim=1, 
        output_dim=1, 
        hidden=64, 
        name="Symmetric",
        alpha=1e-6, 
        seed=15,
        scaling_factor=100
    ):
        self.model = self._build(input_dim, output_dim, hidden, name, alpha, seed, scaling_factor)

    def _build(self, input_dim, output_dim, hidden, name, alpha, seed, scaling_factor):
        init = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed)
        kernel_c = UnitNorm(axis=0)
        bias_c = UnitNorm(axis=0)
        reg = tf.keras.regularizers.l2(alpha)

        inputs = tf.keras.Input(shape=(input_dim,), name='input_layer')
        x = tf.keras.layers.Dense(hidden, kernel_initializer=init, kernel_constraint=kernel_c, bias_constraint=bias_c, kernel_regularizer=reg, name='dense_1')(inputs)
        x = tf.keras.layers.Dense(hidden, activation='tanh', kernel_initializer=init, kernel_constraint=kernel_c, bias_constraint=bias_c, kernel_regularizer=reg, name='dense_2')(x)
        x = tf.keras.layers.Dense(hidden, activation='tanh', kernel_initializer=init, kernel_constraint=kernel_c, bias_constraint=bias_c, kernel_regularizer=reg, name='dense_3')(x)
        x = tf.keras.layers.Dense(hidden, activation='tanh', kernel_initializer=init, kernel_constraint=kernel_c, bias_constraint=bias_c, kernel_regularizer=reg, name='dense_4')(x)
        x = tf.keras.layers.Dense(hidden, activation='tanh', kernel_initializer=init, kernel_constraint=kernel_c, bias_constraint=bias_c, kernel_regularizer=reg, name='dense_5')(x)
        x = tf.keras.layers.Dense(hidden, activation='tanh', kernel_initializer=init, kernel_constraint=kernel_c, bias_constraint=bias_c, kernel_regularizer=reg, name='dense_6')(x)
        x = tf.keras.layers.Dense(hidden, activation='tanh', kernel_initializer=init, kernel_constraint=kernel_c, bias_constraint=bias_c, kernel_regularizer=reg, name='dense_7')(x)
        x = tf.keras.layers.Dense(output_dim, activation='sigmoid', kernel_initializer=init, kernel_constraint=kernel_c, bias_constraint=bias_c, kernel_regularizer=reg, name='output_layer')(x)
        outputs = tf.keras.layers.Lambda(lambda x: x * scaling_factor, name='scaling_layer')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
