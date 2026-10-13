"""Tables for an evaluated experiment. Everything is computed by evaluate_experiment;
the functions here only read from its results."""
import os

import numpy as np
import pandas as pd


def is_cournot(result):
    return "th1" in next(iter(result.curves.values()))


def _mean_std(values):
    v = np.asarray(values, dtype=np.float64)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


# ====================== the tables ======================

def weight_table(results):
    """Max absolute parameter per layer, one column per player, one block per run."""
    return pd.concat({r.label: r.weights for r in results}, axis=1)


def merit_table(results):
    """Fill distance, merit on the training data, and merit on the test sets (mean, std)."""
    rows = {}
    for r in results:
        row = {"Step": r.step, "Fill distance": r.fill_distance, "Train merit": r.train_merit["total"]}
        if r.train_merit["per_player"] is not None:
            for p, m in enumerate(r.train_merit["per_player"]):
                row[f"Train P{p + 1}"] = m
        row["Merit mean"], row["Merit std"] = _mean_std(r.test["merit"])
        per = r.test["merit_per_player"]
        if per[0] is not None:
            for p in range(len(per[0])):
                row[f"P{p + 1} mean"], row[f"P{p + 1} std"] = _mean_std([s[p] for s in per])
        rows[r.label] = row
    return pd.DataFrame(rows).T


def relative_loss_table(results, weighting="uniform"):
    """Relative loss on the test sets (mean, std), per player and for the strategy profile."""
    rows = {}
    for r in results:
        sets = r.test["relative_loss"][weighting]
        row = {"Step": r.step}
        for i in sets[0]["per_player"]:
            row[f"P{i + 1} mean"], row[f"P{i + 1} std"] = _mean_std(
                [s["per_player"][i]["relative_loss"] for s in sets])
        row["Strategy mean"], row["Strategy std"] = _mean_std([s["strategy"] for s in sets])
        rows[r.label] = row
    return pd.DataFrame(rows).T


def tail_table(results, weighting="uniform"):
    """Rent-seeking games: max utility loss on the grid, overall and in the tail (mean, std)."""
    rows = {}
    for r in results:
        sets = r.test["relative_loss"][weighting]
        row = {"Step": r.step}
        for i in sets[0]["per_player"]:
            for key, name in [("max_loss", "Max loss"), ("max_loss_tail", "Max loss θ>0.4"),
                              ("mean_loss_tail", "Mean loss θ>0.4")]:
                row[f"{name} P{i + 1} mean"], row[f"{name} P{i + 1} std"] = _mean_std(
                    [s["per_player"][i][key] for s in sets])
        rows[r.label] = row
    return pd.DataFrame(rows).T


def deviation_table(results):
    """Cournot: RMS strategy deviation from the closed-form equilibrium (mean, std)."""
    rows = {}
    for r in results:
        sets = r.test["relative_loss"]["uniform"]
        row = {"Step": r.step}
        for i in sets[0]["per_player"]:
            row[f"P{i + 1} mean"], row[f"P{i + 1} std"] = _mean_std(
                [s["per_player"][i]["rms_deviation"] for s in sets])
        row["Strategy mean"], row["Strategy std"] = _mean_std([s["rms_strategy"] for s in sets])
        rows[r.label] = row
    return pd.DataFrame(rows).T


def epsilon_table(results):
    """Theoretical epsilon as stated in the theorem (m * eps_J bounds the training residuals)."""
    rows = {}
    for r in results:
        th, per = r.epsilon, r.epsilon["per_player"]
        row = {"Step": r.step, "m": th["m"], "eps_J": th["eps_J"], "m * eps_J": th["m"] * th["eps_J"],
               "eps_theta": max(p["eps_theta"] for p in per), "L_Psi": max(p["L_psi"] for p in per),
               "eps_Psi": max(p["eps_psi"] for p in per), "Epsilon (theory)": th["epsilon"]}
        row["Epsilon (true) mean"], row["Epsilon (true) std"] = _mean_std(r.test["true_epsilon"])
        row["Theory / true"] = th["epsilon"] / row["Epsilon (true) mean"]
        rows[r.label] = row
    return pd.DataFrame(rows).T


def epsilon_refined_table(results):
    """Refined theoretical epsilon: the largest training residual in place of m * eps_J."""
    rows = {}
    for r in results:
        th, per = r.epsilon, r.epsilon["per_player"]
        row = {"max Psi (train)": max(p["psi_train_max"] for p in per),
               "eps_theta": max(p["eps_theta"] for p in per), "L_Psi": max(p["L_psi"] for p in per),
               "eps_Psi (refined)": max(p["eps_psi_refined"] for p in per),
               "Epsilon (refined)": th["epsilon_refined"]}
        row["Epsilon (true) mean"], row["Epsilon (true) std"] = _mean_std(r.test["true_epsilon"])
        row["Refined / true"] = th["epsilon_refined"] / row["Epsilon (true) mean"]
        rows[r.label] = row
    return pd.DataFrame(rows).T

def spectral_table(results):
    """Spectral norms per layer (and max, product, Lipschitz bound), one column block per run."""
    return pd.concat({r.label: r.spectral for r in results}, axis=1)
    
def build_tables(results):
    """All tables that apply to this game, in display order."""
    tables = {"weights": weight_table(results), "spectral": spectral_table(results),
              "merit": merit_table(results)}
    for w in results[0].test["relative_loss"]:
        tables[f"relative_loss_{w}"] = relative_loss_table(results, w)
    if is_cournot(results[0]):
        tables["deviation"] = deviation_table(results)
    else:
        tables["tail"] = tail_table(results)
    tables["epsilon"] = epsilon_table(results)
    tables["epsilon_refined"] = epsilon_refined_table(results)
    return tables


# ====================== formatting and saving ======================

INT_COLUMNS = ("Step", "m")
LATEX_SYMBOLS = {"α": r"$\alpha$", "θ": r"$\theta$", "±": r"$\pm$"}


def as_mean_std(df, fmt="{:.3e}", pm=" ± "):
    """Turn 'X mean' / 'X std' column pairs into one 'mean ± std' column per X;
    integer columns (Step, m) are shown as integers."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col.endswith(" mean"):
            name = col[:-len(" mean")]
            out[name] = [f"{fmt.format(m)}{pm}{fmt.format(s)}"
                         for m, s in zip(df[col], df[f"{name} std"])]
        elif not col.endswith(" std"):
            out[col] = df[col]
    for col in INT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(int).map("{:,}".format)
    return out


def _to_latex(df):
    """LaTeX table; escapes underscores in labels and replaces Unicode symbols."""
    escape = lambda s: str(s).replace("_", r"\_")
    df = df.rename(index=escape, columns=escape)
    tex = df.to_latex(escape=False, float_format=lambda v: f"{v:.3e}")
    for symbol, latex in LATEX_SYMBOLS.items():
        tex = tex.replace(symbol, latex)
    return tex


def experiment_tables(results, name, table_dir="tables", show=True):
    """Build all tables; save each as .csv (raw numbers) and .tex (mean ± std); print if show."""
    os.makedirs(table_dir, exist_ok=True)
    tables = build_tables(results)
    for tname, df in tables.items():
        shown = df if tname in ("weights", "spectral") else as_mean_std(df)
        df.to_csv(f"{table_dir}/{name}_{tname}.csv")
        with open(f"{table_dir}/{name}_{tname}.tex", "w") as f:
            f.write(_to_latex(shown))
        if show:
            print(f"\n=== {name}: {tname} ===")
            print(shown.to_string(float_format=lambda v: f"{v:.3e}"))
    return tables