import tensorflow as tf
import math
import numpy as np
from utils.config import GameConfig
import os
import pandas as pd
import time


class ModelCheckpointer:
    def __init__(self, model, optimizer, config: GameConfig, max_to_keep=5):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        ckpt_dir = os.path.join(config.run_dir, "checkpoint_dir")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=ckpt_dir, max_to_keep=max_to_keep)

    def save(self):
        save_path = self.manager.save()
        print(f"Saved checkpoint at step {int(self.optimizer.iterations.numpy())}: {save_path}")
        return save_path

    def restore_latest(self):
        status = self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            status.expect_partial()
            print(f"Restored from {self.manager.latest_checkpoint} "
                  f"(step {int(self.optimizer.iterations.numpy())})")
        else:
            print("No checkpoint found, starting from scratch.")
        return status

    def training_data_path(self):
        return os.path.join(self.config.run_dir, "training_data.npy")

    def save_training_data(self, training_data):
        path = self.training_data_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.asarray(training_data))
        print(f"Saved training data to {path}")

    def load_training_data(self):
        return np.load(self.training_data_path())

    def two_sample_data_path(self):
        return os.path.join(self.config.run_dir, "training_data_two_sample.npz")
        
    def save_two_sample_data(self, theta_own, theta_opp):
        path = self.two_sample_data_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, theta_own=np.asarray(theta_own), theta_opp=np.asarray(theta_opp))
        print(f"Saved two-sample training data to {path}")
    
    def load_two_sample_data(self):
        with np.load(self.two_sample_data_path()) as f:
            return f["theta_own"], f["theta_opp"]




        

def geometric_checkpoints(max_iters, points_per_decade=20):
    """Iteration numbers to log at, spaced evenly on a log scale from 1 to
    max_iters, instead of every fixed N steps. Gives you dense sampling early
    (where the loss moves fast) and sparse sampling late (where it's flat),
    which is what you want for a log-x convergence plot."""
    n_decades = math.log10(max_iters)
    n_points = max(2, int(n_decades * points_per_decade))
    return sorted(set(np.round(np.geomspace(1, max_iters, num=n_points)).astype(int).tolist()))


def _batches(data, batch_size, seed):
    """Shuffled, repeating batches; reshuffled every pass."""
    n = int(data.shape[0])
    return (tf.data.Dataset.from_tensor_slices(data)
            .shuffle(n, seed=seed, reshuffle_each_iteration=True)
            .repeat()
            .batch(batch_size))


def train(main_model, optimizer, game_def, config: GameConfig, training_data=None,
          theta_own=None, theta_opp=None, max_to_keep=5, save_weight_every=50_000, print_every=10_000):
    """Single-sample: pass training_data. Two-sample (config.two_sample=True): pass theta_own and theta_opp.
    Resumes from the latest checkpoint and saved data for this run if they exist."""
    loss_fn = game_def["loss_func"]
    two_sample = config.two_sample

    optimizer.build(main_model.trainable_variables)
    checkpointer = ModelCheckpointer(main_model, optimizer, config, max_to_keep=max_to_keep)
    checkpointer.restore_latest()

    # ---------- data: load if this run already has it, otherwise save ----------
    if two_sample:
        if not hasattr(loss_fn, "call_two_sample"):
            raise ValueError(f"{type(loss_fn).__name__} has no call_two_sample method.")
        if os.path.exists(checkpointer.two_sample_data_path()):
            own_saved, opp_saved = checkpointer.load_two_sample_data()
            if own_saved.shape != tuple(np.shape(theta_own)) or opp_saved.shape != tuple(np.shape(theta_opp)):
                raise ValueError("Saved two-sample data has different shapes. Use a new run_n.")
            theta_own, theta_opp = own_saved, opp_saved
            print(f"Loaded existing two-sample data from {checkpointer.two_sample_data_path()}")
        else:
            checkpointer.save_two_sample_data(theta_own, theta_opp)
        theta_own = tf.cast(theta_own, tf.float32)
        theta_opp = tf.cast(theta_opp, tf.float32)
        data_iter = iter(tf.data.Dataset.zip((
            _batches(theta_own, config.batch_size, config.seed),
            _batches(theta_opp, config.opp_batch_size, config.seed + 1),
        )))
        eval_own, eval_opp = theta_own, theta_opp
    else:
        if os.path.exists(checkpointer.training_data_path()):
            loaded = checkpointer.load_training_data()
            if loaded.shape != tuple(np.shape(training_data)):
                raise ValueError("Saved training data has a different shape. Use a new run_n.")
            training_data = loaded
            print(f"Loaded existing training data from {checkpointer.training_data_path()}")
        else:
            checkpointer.save_training_data(training_data)
        training_data = tf.cast(training_data, tf.float32)
        data_iter = iter(_batches(training_data, config.batch_size, config.seed).map(lambda x: (x, x)))
        eval_own, eval_opp = training_data, training_data

    # ---------- loss ----------
    def compute_loss(x_own, x_opp, training):
        y_own = main_model(x_own, training=training)
        if two_sample:
            y_opp = main_model(x_opp, training=training)
            return loss_fn.call_two_sample(x_own, y_own, y_opp)
        return loss_fn(x_own, y_own)

    def _step(x_own, x_opp):
        with tf.GradientTape() as tape:
            loss = compute_loss(x_own, x_opp, training=True) + sum(main_model.losses)
        grads = tape.gradient(loss, main_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, main_model.trainable_variables))

    @tf.function
    def train_steps(n_steps):
        """Run n_steps training steps inside one TensorFlow call."""
        for _ in tf.range(n_steps):
            x_own, x_opp = next(data_iter)
            _step(x_own, x_opp)

    @tf.function
    def eval_step(x_own, x_opp):
        return compute_loss(x_own, x_opp, training=False)

    # ---------- logging ----------
    global_step = int(optimizer.iterations.numpy())
    new_total = global_step + config.max_iters
    checkpoint_iters = set(
        c for c in geometric_checkpoints(new_total, points_per_decade=config.log_every_decades)
        if c > global_step
    )

    log_path = os.path.join(config.run_dir, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iteration,merit\n")

    # ---------- loop: train in stretches between the steps where something happens ----------
    print_iters = set(range((global_step // print_every + 1) * print_every, new_total + 1, print_every))
    save_iters = set(range((global_step // save_weight_every + 1) * save_weight_every,
                           new_total + 1, save_weight_every))
    events = sorted(checkpoint_iters | print_iters | save_iters | {new_total})

    print(f"Training steps {global_step + 1:,} to {new_total:,}", flush=True)
    t_start = t_last = time.perf_counter()
    step = last_print_step = global_step
    for event in events:
        train_steps(tf.constant(event - step, dtype=tf.int64))
        step = event

        log_now = step in checkpoint_iters
        print_now = step in print_iters
        if log_now or print_now:
            merit = float(eval_step(eval_own, eval_opp))
            if log_now:
                with open(log_path, "a") as f:
                    f.write(f"{step},{merit:.6e}\n")
            if print_now:
                now = time.perf_counter()
                rate = (step - last_print_step) / (now - t_last)
                hours_left = (new_total - step) / rate / 3600
                print(f"step {step:,}  merit {merit:.3e}  {rate:.1f} it/s  "
                      f"elapsed {(now - t_start) / 3600:.2f} h  ~{hours_left:.1f} h left", flush=True)
                t_last, last_print_step = now, step

        if step in save_iters:
            checkpointer.save()

    if new_total not in save_iters:
        checkpointer.save()
    print(f"Trained through iteration {int(optimizer.iterations.numpy())}")
    return checkpointer