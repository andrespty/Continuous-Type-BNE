# Continuous-Type-BNE

Neural-network approximation of Bayesian Nash equilibria (BNE) in games with continuous types and actions, using a Fischer-Burmeister merit function. Each player has a strategy network and a Lagrange network. Training minimizes the merit; evaluation measures how far the learned profile is from a best response.

Games included: symmetric rent-seeking (2 and 3 players), asymmetric rent-seeking (2 players), a 3-player multi-market Cournot oligopoly, and a Pareto (heavy-tailed) variant of the symmetric 2-player game.

## Project layout

```
Continuous-Type-BNE/
├── utils/                  all code lives here (imported by the notebooks and the HPC scripts)
│   ├── config.py           GameConfig and GAME_DEFS (one entry per game)
│   ├── models.py           build_main_model: strategy and Lagrange networks
│   ├── lossFunctions.py    merit-function losses, one per game
│   ├── helpers.py          optimizers (SGD schedule, Adam), type samplers, training-data generators
│   ├── training.py         train(), ModelCheckpointer
│   ├── evaluation.py       game evaluators: utility rho, best response, opponent sampling
│   ├── loading.py          rebuild a model, load checkpoints, training data, merit logs
│   ├── metrics.py          merit, relative loss, fill distance, weights, Cournot metrics
│   ├── epsilon.py          theoretical epsilon bound (and refined version)
│   ├── results.py          evaluate_run / evaluate_experiment -> RunResult
│   ├── experiments.py      EXPERIMENTS: which runs to compare
│   ├── tables.py           tables for an evaluated experiment
│   ├── figures.py          figures for an evaluated experiment
│   └── reports.py          thin entry point re-exporting tables and figures
├── notebooks/              notebooks only call functions from utils
├── example/                scripts for training on the HPC cluster (HPC_TRAINING.py, submit script)
├── saved_models/           one folder per trained run (created by training)
├── logs/
├── images/                 saved figures
├── tables/                 saved tables (.csv and .tex)
└── requirements.txt
```

If a file name above differs from the repository, the descriptions still apply to the module that holds that function.

## Setup

Python 3.8-3.11 and TensorFlow 2.13 (Keras 2; TensorFlow 2.16+ uses Keras 3 and breaks the loss classes).

```bash
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`:

```text
tensorflow==2.13.1
numpy<1.25
pandas
matplotlib
plotly
jupyter
```

In a notebook, add the project folder to the import path first:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))   # notebooks/ -> project folder
```

All project imports are absolute (`from utils.training import train`); never use `..`.

Paths (`outdir`, `model_path`) are computed from the location of `config.py`, so `saved_models/` and `logs/` sit in the project folder on any machine.

## Games

Defined in `GAME_DEFS` (`utils/config.py`):

| Name | Game |
|---|---|
| `sym2` | symmetric rent-seeking, 2 players (one shared network) |
| `sym3`, `sym3fast` | symmetric rent-seeking, 3 players. `sym3fast` estimates the expectation over opponent pairs by cyclic shifts (`n_shifts`), much cheaper than all pairs |
| `asym2` | asymmetric rent-seeking, 2 players |
| `cournot` | 3 players, 3 markets, 2-D types |

`sym2` also runs with Pareto types (two-sample mode, below). Each game definition sets the action bound `action_high`; the matching bound is passed to the loss, so keep the two equal.

## Training

`train()` in `utils/training.py` runs one training run and:

- builds the optimizer from the config (`make_optimizer`): SGD with the decaying step size `alpha_0 / (a + b * t^r)`, or Adam with a fixed rate;
- trains in stretches inside one TensorFlow call, and prints progress with speed and estimated time left;
- saves checkpoints (weights, optimizer state and step count) every `save_weight_every` steps;
- writes `training_log.csv` (merit at geometrically spaced iterations);
- resumes automatically if a checkpoint for the same run exists.

Everything for one run is stored in `saved_models/<game>/<optimizer>_<run_n>/`:

```
checkpoint_dir/                      weights and optimizer state
training_data.npy                    training types (single-sample runs)
training_data_two_sample.npz         own and opponent types (two-sample runs)
training_log.csv                     merit against iteration
train_output.log                     screen output of the job (HPC)
```

Important rules:

- **A run is identified by game, optimizer and `run_n`.** Use a new `run_n` for every new attempt. Reusing a name resumes the old run and reloads its saved data.
- **`max_iters` is added to the saved step count** when resuming, unless `train()` uses `new_total = max(global_step, max_iters)`. Check before resubmitting a job that hit the time limit.
- **The learning-rate settings and `l2_alpha` are not saved in the checkpoint.** Reuse the same settings when resuming and when loading a run for evaluation.
- The seed sets the initial weights, the training samples and the shuffling order, so different seeds give independent models.

### Pareto and two-sample runs

Types follow a Lomax distribution (shape 3), truncated at its 0.999 quantile by conditioning and rescaled affinely to [0.01, 1.01]. Two samples are used:

- **opponent types** (Monte Carlo): drawn from the Pareto prior, which defines the game;
- **own types** (training points): `stratified_pareto`, with at least `strat_floor` points in each of `strat_bins` equal-width intervals and the rest drawn from the Pareto. This bounds the fill distance by the interval width.

Run with `--two_sample --own_dist stratified_pareto --opp_dist pareto`.

## Training on the HPC

The `example/` folder has the scripts used on the cluster:

- `HPC_TRAINING.py` takes `--game --optimizer --run_n --seed --max_iters --sgd_alpha_0 --adam_lr --alpha --m --batch_size --two_sample --own_dist --opp_dist --strat_floor`. It trains, then prints the final training merit.
- The SLURM script runs a job array (one task per game and seed), writes each run's output to `train_output.log` in its run folder, and sets the TensorFlow thread counts from `$SLURM_CPUS_PER_TASK`.

```bash
sbatch submit.sh
```

To train several seeds of one game, give each seed its own `--seed` and `--run_n`.

## Evaluating

Everything is computed by `evaluate_run` / `evaluate_experiment` (`utils/results.py`), which load the latest checkpoint of each run and return `RunResult` objects. Nothing is cached, so rerun it to pick up newer checkpoints.

Define which runs to compare in `utils/experiments.py`:

```python
EXPERIMENTS["sym2_uniform"] = {
    "common": dict(game="sym2"),
    "runs": {
        "SGD": dict(optimizer="sgd", run_n="uniform_oct8_s1", l2_alpha=0.0),
    },
}
```

`"common"` must match how the runs were trained (game, `two_sample`, `own_dist`, `opp_dist`, `m`, `strat_floor`, ...). A mismatch changes the test samples and the weighted losses. Each run's first log lines show the exact config it was trained with.

```python
from utils.results import evaluate_experiment
from utils.reports import experiment_tables, experiment_figures

results = evaluate_experiment("sym2_uniform")
tables = experiment_tables(results, "sym2_uniform")     # printed, saved to tables/ as .csv and .tex
experiment_figures(results, "sym2_uniform")             # saved to images/
```

### What is computed

| Quantity | Meaning |
|---|---|
| Merit | the training loss (no L2 term), on the training data and on fresh test sets |
| Relative loss | `l / (rho_bar + l)` with `l` the mean utility loss against the best response; `uniform` weights every grid type equally, `prior` weights by the type density |
| Max utility loss | largest `rho_hat - rho` on the type grid, overall and for `theta > 0.4`; with opponents playing the trained network this is the true epsilon |
| Fill distance | largest distance from any type to the nearest training type |
| Theoretical epsilon | the bound of Theorem 2 / 3 from `eps_J`, fill distance, the Lipschitz constant of Psi, `B_a`, `z`; also a refined version using the largest per-type residual in place of `m * eps_J` |
| Layer weights, spectral norms | maximum absolute parameter and largest singular value per layer |

The best response is found by grid search over the action space (rent-seeking) or from its closed form (Cournot). For Cournot, the relative loss is measured against the closed-form equilibrium. The test-set standard deviations come from independent samples: 5 opponent pools for the relative losses, and 5 fresh type samples for the test merit.

### Evaluating a single model in a notebook

```python
from utils.config import GameConfig
from utils.results import evaluate_run

r = evaluate_run("model", GameConfig(game="sym2", optimizer="sgd", run_n="uniform_oct8_s1"), l2_alpha=0.0)
```

For several seeds of the same game, evaluate each run and report the mean and standard deviation across models.

## Notebooks

The notebooks only import and call functions from `utils`. Typical uses: evaluate an experiment and make the tables and figures, check specific types, plot gradients of the strategy, compare several seeds, run the scalability study (training time against the number of markets).

After editing any file in `utils/`, restart the kernel: Jupyter keeps modules cached, and `GAME_DEFS` creates the loss objects when `config.py` is first imported.

## Conventions and pitfalls

- Match the evaluation config to the training config, including `l2_alpha` (it changes the architecture's regularizers, not the loaded weights).
- Keep `action_high` in `GAME_DEFS` equal to the bound used inside the loss, and well above the largest best response.
- The merit includes the number of players as a factor in the symmetric games (2 for `sym2`, 3 for `sym3`); the per-player merit divides it out.
- Do not pin `typing_extensions` or install `tensorflow-probability` in the same environment: TensorFlow 2.13 needs `typing-extensions<4.6`.