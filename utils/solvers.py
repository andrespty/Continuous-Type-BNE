from .game import Game
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Solver:
    """Base solver — handles compile, evaluate, save, checkpoint."""

    def __init__(self, game: Game):
        self.pdn = game.nn_model()
        self.model = self.pdn.model
        self.optimizer = game.optimizer()
        self.history = None
        self.training_data = None
        self.model.compile(loss=game.loss_fn(), optimizer=self.optimizer)

    def _sample_types(self, n_samples, type_bounds) -> tf.Tensor:
        raise NotImplementedError

    def fit(self, n_samples=1000, epochs=5000, batch_size=500, type_bounds=None):
        self.training_data = self._sample_types(n_samples, type_bounds)
        self.history = self.model.fit(
            self.training_data, self.training_data,
            batch_size=batch_size, shuffle=False, epochs=epochs,
        )
        return self

    def plot_strategy(self, type_bounds=None):
        raise NotImplementedError

    def evaluate(self):
        if self.training_data is None:
            raise RuntimeError("Call fit() before evaluate().")
        energy = self.model.evaluate(self.training_data, self.training_data)
        print(f"Energy:               {energy}")
        print(f"Optimizer iterations: {self.model.optimizer.iterations.numpy()}")
        lr = self.model.optimizer.learning_rate
        try:
            print(f"Current LR:           {lr(self.model.optimizer.iterations).numpy()}")
        except TypeError:
            print(f"LR:                   {lr.numpy()}")
        return energy

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(path, include_optimizer=True)
        if self.training_data is not None:
            np.save(os.path.join(path, 'training_data.npy'), self.training_data.numpy())
        print(f"Saved to {path}")

    def checkpoint(self, path, name='ckpt'):
        os.makedirs(path, exist_ok=True)
        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt.write(os.path.join(path, name))
        print(f"Checkpoint written to {os.path.join(path, name)}")


class SymmetricSolver(Solver):

    def _sample_types(self, n_samples, type_bounds=(0.001, 1.001)):
        lower, upper = type_bounds
        return tf.cast(tf.random.uniform((n_samples, 1), lower, upper), tf.float32)

    def plot_strategy(self, type_bounds=(0.01, 1.0), output_index=0, n_points=500):
        input_values = np.linspace(*type_bounds, n_points).reshape(-1, 1)
        outputs = self.model(input_values)[:, output_index].numpy()
        label = "Primal action" if output_index == 0 else f"Dual variable λ{output_index}"
        plt.figure(figsize=(10, 5))
        plt.plot(input_values, outputs, color='steelblue', label=label)
        plt.title(f"Learned strategy — output {output_index}")
        plt.xlabel("Type θ"), plt.ylabel("Value")
        plt.legend(), plt.grid(True), plt.tight_layout(), plt.show()


class AsymmetricSolver(Solver):

    def _sample_types(self, n_samples, type_bounds=[(0.001, 1.001), (0.51, 1.51)]):
        players = [tf.cast(tf.random.uniform((n_samples, 1), lo, hi), tf.float32)
                   for lo, hi in type_bounds]
        return tf.reshape(tf.concat(players, axis=-1), [-1, len(players), 1])

    def plot_strategy(self, type_bounds=[(0.01, 1.0), (0.51, 1.51)], n_points=1000):
        players = [np.linspace(lo, hi, n_points).reshape(-1, 1) for lo, hi in type_bounds]
        input_values = tf.reshape(tf.concat(players, axis=-1), [-1, len(players), 1])
        outputs = self.model.predict(input_values)

        fig, axs = plt.subplots(1, len(type_bounds), figsize=(6 * len(type_bounds), 4))
        for i, ax in enumerate(axs):
            ax.plot(tf.reshape(input_values[:, i, :], [-1, 1]),
                    tf.reshape(outputs[:, i], [-1, 1]), color='b')
            ax.set(title=f'Player {i+1}', xlabel='Type', ylabel='Action')
            ax.legend(['Predictions']), ax.grid(True)
        plt.tight_layout(), plt.show()