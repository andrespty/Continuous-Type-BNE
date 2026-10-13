import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras import layers, regularizers

def display_strat(model, game_def, n_points=500):
    n = game_def['n']
    d = game_def['d']
    theta_low, theta_high = game_def['theta_low'], game_def['theta_high']
    symmetric = game_def.get('symmetric', False)
    max_d = max(d)

    # One joint batch: slot i sweeps player i's own domain independently.
    # Valid because each player's output only depends on its own slot --
    # row k mixes unrelated theta values across players, but that's fine
    # since the model never lets them interact.
    batch = np.zeros((n_points, n, max_d), dtype=np.float32)
    theta_per_player = []
    for i in range(n):
        theta_vals = np.linspace(theta_low[i], theta_high[i], n_points).astype(np.float32)
        batch[:, i, :d[i]] = theta_vals.reshape(-1, 1)
        theta_per_player.append(theta_vals)

    outputs = model(batch)   # single forward pass for every player

    players = [0] if symmetric else list(range(n))
    fig, axes = plt.subplots(1, len(players), figsize=(6 * len(players), 5), squeeze=False)
    axes = axes[0]

    for ax, i in zip(axes, players):
        ax.plot(theta_per_player[i], outputs[:, i], color='b')
        ax.set_title("Symmetric strategy" if symmetric else f"Player {i+1}")
        ax.set_xlabel("Type")
        ax.set_ylabel("Network Output (Action)")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

class LambdaNet(tf.keras.layers.Layer):
    def __init__(self, n, output='relu' ,hidden=64, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.hidden = hidden
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(n, activation=output)
        ])

    def call(self, theta):
        return self.mlp(theta)

    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n, 'hidden': self.hidden})
        return config

def build_player_model(z_i, input_features, name, initializer, alpha, scale=100):
    inputs = tf.keras.Input(shape=(input_features,), name='input_layer')
    x = tf.keras.layers.Dense(64, kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(alpha), name='dense_1')(inputs)
    for i in range(2, 8):
        x = tf.keras.layers.Dense(64, 
                                  activation='tanh', 
                                  kernel_initializer=initializer, 
                                  kernel_regularizer=tf.keras.regularizers.l2(alpha), 
                                  name=f'dense_{i}'
                                 )(x)
    out = tf.keras.layers.Dense(z_i, 
                                activation='sigmoid', 
                                kernel_initializer=initializer, 
                                kernel_regularizer=tf.keras.regularizers.l2(alpha), 
                                name='output_layer'
                               )(x)
    out = tf.keras.layers.Lambda(lambda x: x * scale, name='scaling_layer')(out)
    return tf.keras.Model(inputs=inputs, outputs=out, name=name)

def build_main_model(game_def, designed_initializer, alpha):
    N, z_i, d_i = game_def['n'], game_def['z'], game_def['d']
    action_high = game_def['action_high']
    output_act = game_def['dual_output']
    p_models = [
        build_player_model(z_i[i], d_i[i], f'Player{i+1}', designed_initializer, alpha, action_high[i])
        for i in range(N)
    ]

    inputs_main = tf.keras.Input(shape=(N, max(d_i)), name='main_input_layer')
    slices = [
        tf.keras.layers.Lambda(lambda x, i=i: tf.squeeze(x[:, i:i+1, :], axis=1),
                                name=f'slice_{i+1}')(inputs_main)
        for i in range(N)
    ]

    primal_outs = [p_models[i](slices[i]) for i in range(N)]
    dual_outs = [
        tf.keras.layers.Lambda(lambda x: x, name=f"dual_p{i+1}")(LambdaNet(n=2*z_i[i], output=output_act)(slices[i]))
        for i in range(N)
    ]

    concat_layer = tf.keras.layers.Concatenate(axis=-1, name='concat_layer')(primal_outs + dual_outs)
    return tf.keras.Model(inputs=inputs_main, outputs=[concat_layer], name=f'Model_M{max(z_i)}')