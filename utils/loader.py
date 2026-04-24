import tensorflow as tf
import os
from .game import Game

class ModelLoader:
    """
    Loads a saved Keras model and optionally restores a checkpoint.

    Parameters
    ----------
    path : str
        Directory where the model is saved.
    game : Game
        The game instance — used to resolve custom objects (loss, lr schedule).
    """

    def __init__(self, game: Game):
        self.game = game

    def load(self, path: str, checkpoint: str = None):
        custom_objects = self.game.custom_objects()

        model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        optimizer = model.optimizer

        if checkpoint is not None:
            ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
            ckpt.restore(os.path.join(path, 'checkpoint_dir', checkpoint)).expect_partial()

        return model