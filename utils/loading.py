"""Rebuild a run's model and read its saved files."""
import os

import pandas as pd
import tensorflow as tf

from utils.config import GAME_DEFS
from utils.models import build_main_model
from utils.utils import make_optimizer
from utils.training import ModelCheckpointer


def load_model(config, l2_alpha=0.0):
    """Rebuild the model and restore the latest checkpoint for config."""
    game_def = GAME_DEFS[config.game]
    init = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=config.seed)
    model = build_main_model(game_def, init, l2_alpha)
    optimizer = make_optimizer(config)
    optimizer.build(model.trainable_variables)
    checkpointer = ModelCheckpointer(model, optimizer, config)
    if checkpointer.manager.latest_checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint for {config.game}/{config.optimizer}_{config.run_n}; check the config.")
    checkpointer.restore_latest()
    return model, checkpointer


def load_training_data(checkpointer, config):
    """(own types, opponent types) saved by train(); opponent types are None for single-sample runs."""
    if config.two_sample:
        own, opp = checkpointer.load_two_sample_data()
        return tf.cast(own, tf.float32), tf.cast(opp, tf.float32)
    return tf.cast(checkpointer.load_training_data(), tf.float32), None


def merit_log_path(config):
    """Training log in the run's folder."""
    return os.path.join(config.run_dir, "training_log.csv")


def read_merit_log(path):
    """Merit log; duplicate iterations from resumed runs are dropped."""
    return pd.read_csv(path).drop_duplicates("iteration", keep="last").sort_values("iteration")