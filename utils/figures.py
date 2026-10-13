"""Figures for an evaluated experiment. Everything is computed by evaluate_experiment;
the functions here only read from its results."""
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from utils.tables import is_cournot


# ====================== styles ======================


OPT_STYLE = {
    "sgd":  dict(color="#0072B2", linestyle="-", linewidth=1.8, zorder=2),  # blue, solid
    "adam": dict(color="#D55E00", linestyle=(0, (1, 1.2)), linewidth=2.0, zorder=3,
                 dash_capstyle="round"),                                     # vermillion, dotted
}

BR_STYLE = dict(color="#D55E00", linestyle="-", linewidth=2.5, zorder=1)                 # bright red
OPT_STYLE["sgd"] = dict(color="#0072B2", linestyle="-", linewidth=2.5, zorder=2)  # bright blue

OPT_NAME = {"adam": "Adam", "sgd": "SGD"}
FONT = dict(label=9, title=9.5, tick=8, legend=8) 
PAPER_WIDTH = 5.46 

def _l2_title(alpha):
    return "No L2" if alpha == 0 else f"L2 (α = {alpha:g})"


def _slug(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def _style_ax(ax, xlabel=None, ylabel=None, title=None, legend=True):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT["label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT["label"])
    if title:
        ax.set_title(title, fontsize=FONT["title"], pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT["tick"])
    if legend:
        ax.legend(fontsize=FONT["legend"], frameon=False)
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.3, color="0.92")


def _save(fig, outfile):
    if outfile:
        fig.savefig(outfile, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.show()


# ====================== rent-seeking games ======================

def plot_curves(results, key_nn, key_br, ylabel, outfile=None):
    """Columns: L2 settings. Rows: players (asymmetric games). Each panel: the best response
    (band, from the last run in the panel) and each optimizer's NN curve.
    key_nn/key_br: 'a_nn'/'a_hat' for strategies, 'rho_nn'/'rho_hat' for utilities."""
    alphas = sorted({r.l2_alpha for r in results})
    players = list(results[0].curves)
    symmetric = results[0].game_def["symmetric"]

    fig, axes = plt.subplots(len(players), len(alphas),
                             figsize=(3.2 * len(alphas), 2.8 * len(players)),
                             dpi=300, sharey="row", squeeze=False, constrained_layout=True)
    for row, i in enumerate(players):
        for col, a in enumerate(alphas):
            ax = axes[row, col]
            group = [r for r in results if r.l2_alpha == a]
            ref = group[-1]
            ax.plot(ref.curves[i]["theta"], ref.curves[i][key_br], label="Best response", **BR_STYLE)
            for r in group:
                opt = r.config.optimizer
                ax.plot(r.curves[i]["theta"], r.curves[i][key_nn], label=OPT_NAME.get(opt, opt),
                        **OPT_STYLE.get(opt, dict(color="0.4", linestyle=":")))
            _style_ax(ax,
                      xlabel=r"$\theta_i$" if row == len(players) - 1 else None,
                      ylabel=ylabel if col == 0 else None,
                      title=_l2_title(a) if symmetric else f"Player {i + 1}, {_l2_title(a)}")
    _save(fig, outfile)


def plot_training_curves(results, outfile=None, max_iter=None):
    """Merit against iteration (log-log), one panel per L2 setting; solid lines, optimizer colors."""
    alphas = sorted({r.l2_alpha for r in results})
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.2 * len(alphas), 2.8), dpi=300,
                             sharey=True, squeeze=False, constrained_layout=True)
    for col, a in enumerate(alphas):
        ax = axes[0, col]
        for r in [r for r in results if r.l2_alpha == a]:
            if r.merit_log is None:
                print(f"[{r.label}] no merit log")
                continue
            df = r.merit_log if max_iter is None else r.merit_log[r.merit_log["iteration"] <= max_iter]
            opt = r.config.optimizer
            style = {**OPT_STYLE.get(opt, dict(color="0.4")), "linestyle": "-"}
            style.pop("dash_capstyle", None)
            ax.plot(df["iteration"], df["merit"], label=OPT_NAME.get(opt, opt), **style)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style_ax(ax, xlabel="Iteration", ylabel=r"Merit $\bar J$" if col == 0 else None,
                  title=_l2_title(a))
    _save(fig, outfile)

def plot_training_histogram(result, outfile=None, player=0, log_y=True):
    """Histogram of the training types: own types and, for two-sample runs, the opponent pool.
    Bins match the stratification intervals; the dashed line marks the minimum per interval."""
    cfg, gd = result.config, result.game_def
    lo, hi = float(gd["theta_low"][player]), float(gd["theta_high"][player])
    edges = np.linspace(lo, hi, getattr(cfg, "strat_bins", 100) + 1)

    panels = [("Own types", result.train_types["own"][:, player, 0], cfg.own_dist)]
    if result.train_types["opp"] is not None:
        panels.append(("Opponent types", result.train_types["opp"][:, player, 0], cfg.opp_dist))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.6), dpi=300,
                             sharey=True, squeeze=False, constrained_layout=True)
    for col, (title, data, dist) in enumerate(panels):
        ax = axes[0, col]
        ax.hist(data, bins=edges, color="#0072B2", alpha=0.85, edgecolor="white", linewidth=0.3)
        stratified = dist == "stratified_pareto"
        if stratified:
            ax.axhline(cfg.strat_floor, color="0.3", linestyle="--", linewidth=0.8,
                       label=f"minimum per interval ({cfg.strat_floor})")
        if log_y:
            ax.set_yscale("log")
        _style_ax(ax, xlabel=r"$\theta_i$", ylabel="Count" if col == 0 else None,
                  title=f"{title} (m = {len(data):,})", legend=stratified)
    _save(fig, outfile)


# ====================== Cournot ======================

def _heatmap_with_contours(fig, ax, th1, th2, nn, ref, lo, hi, cbar_title):
    """Heatmap of nn, its contours (solid white), and ref's contours at the same levels (dashed cyan)."""
    n_points = th1.shape[0]
    mid = (n_points - 1) / 2

    im = ax.imshow(nn, origin='lower', aspect='auto', cmap='plasma',
                   extent=[lo[0], hi[0], lo[1], hi[1]])
    cs = ax.contour(th1, th2, nn, colors='white', linewidths=0.5, alpha=0.6)
    ax.contour(th1, th2, ref, levels=cs.levels, colors='cyan', linewidths=0.5, linestyles='--', alpha=0.9)

    # contour labels along the diagonal, nearest the middle
    diag_vals = np.diag(nn)
    diag_x, diag_y = np.diag(th1), np.diag(th2)
    manual_pos = []
    for level in cs.levels:
        sc = np.where(np.diff(np.sign(diag_vals - level)))[0]
        if len(sc) == 0:
            continue
        k = sc[np.argmin(np.abs(sc - mid))]
        if k < 0.10 * n_points or k > 0.90 * n_points:
            continue
        t = (level - diag_vals[k]) / (diag_vals[k + 1] - diag_vals[k])
        manual_pos.append((diag_x[k] + t * (diag_x[k + 1] - diag_x[k]),
                           diag_y[k] + t * (diag_y[k + 1] - diag_y[k])))
    for lbl in ax.clabel(cs, inline=True, fontsize=5, fmt='%.1f', manual=manual_pos):
        lbl.set_rotation(0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.set_title(cbar_title, fontsize=7, pad=4)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.5)
    ax.tick_params(labelsize=5.5)


def plot_cournot_heatmap(result, outfile=None):
    """Strategies: NN heatmap with NN contours (solid) and equilibrium contours (dashed),
    per player (rows) and market (columns)."""
    gd = result.game_def
    n, z = gd["n"], gd["z"][0]
    fig, axes = plt.subplots(n, z, figsize=(6.2, 5.6), dpi=300, constrained_layout=True)
    for i, c in result.curves.items():
        lo, hi = gd["theta_low"][i], gd["theta_high"][i]
        for m in range(z):
            ax = axes[i, m]
            _heatmap_with_contours(fig, ax, c["th1"], c["th2"], c["a_nn"][:, :, m], c["a_hat"][:, :, m],
                                   lo, hi, rf'$a_{{i,{m + 1}}}$')
            if i == n - 1:
                ax.set_xlabel(r'$\theta_{i,1}$', fontsize=7)
            if m == 0:
                ax.set_ylabel(r'$\theta_{i,2}$', fontsize=7)
                ax.annotate(f'Player {i + 1}', xy=(0, 0.5), xytext=(-42, 0),
                            xycoords='axes fraction', textcoords='offset points',
                            fontsize=8, rotation=90, va='center', ha='center')
            if i == 0:
                ax.set_title(f'Market {m + 1}', fontsize=8, pad=6)
    _save(fig, outfile)


def plot_cournot_rho(result, outfile=None):
    """Utilities: heatmap of the NN strategy's utility with its contours (solid), and the
    best-response utility's contours at the same levels (dashed); one panel per player."""
    gd = result.game_def
    n = gd["n"]
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.0), dpi=300, squeeze=False,
                             constrained_layout=True)
    for i, c in result.curves.items():
        ax = axes[0, i]
        lo, hi = gd["theta_low"][i], gd["theta_high"][i]
        _heatmap_with_contours(fig, ax, c["th1"], c["th2"], c["rho_nn"], c["rho_hat"],
                               lo, hi, r'$\rho_i$')
        ax.set_xlabel(r'$\theta_{i,1}$', fontsize=7)
        if i == 0:
            ax.set_ylabel(r'$\theta_{i,2}$', fontsize=7)
        ax.set_title(f'Player {i + 1}', fontsize=8, pad=6)
    _save(fig, outfile)


def plot_cournot_3d(result):
    """Interactive surfaces: NN (Viridis) vs closed-form equilibrium (Reds)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    gd = result.game_def
    n, z = gd["n"], gd["z"][0]
    fig = make_subplots(
        rows=n, cols=z,
        specs=[[{'type': 'surface'}] * z] * n,
        subplot_titles=[f'Player {i + 1}, Market {m + 1}' for i in range(n) for m in range(z)],
        horizontal_spacing=0.02, vertical_spacing=0.05,
    )
    for i, c in result.curves.items():
        for m in range(z):
            fig.add_trace(go.Surface(x=c["th1"], y=c["th2"], z=c["a_nn"][:, :, m],
                                     colorscale='Viridis', opacity=0.85, showscale=False, name='NN'),
                          row=i + 1, col=m + 1)
            fig.add_trace(go.Surface(x=c["th1"], y=c["th2"], z=c["a_hat"][:, :, m],
                                     colorscale='Reds', opacity=0.5, showscale=False, name='Equilibrium'),
                          row=i + 1, col=m + 1)
    fig.update_layout(
        title=f'{result.label}: NN (Viridis) vs closed-form equilibrium (Reds)',
        height=1100, width=1150, margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.show()


# ====================== everything ======================

def experiment_figures(results, name, fig_dir="images", max_iter=None, cournot_3d=False):
    """Strategies and utilities (rent-seeking) or heatmaps (Cournot), plus training curves."""
    os.makedirs(fig_dir, exist_ok=True)
    if is_cournot(results[0]):
        for r in results:
            plot_cournot_heatmap(r, f"{fig_dir}/{name}_{_slug(r.label)}_strategy.pdf")
            plot_cournot_rho(r, f"{fig_dir}/{name}_{_slug(r.label)}_rho.pdf")
            if cournot_3d:
                plot_cournot_3d(r)
    else:
        plot_curves(results, "a_nn", "a_hat", r"$a_i$", f"{fig_dir}/{name}_strategy.pdf")
        plot_curves(results, "rho_nn", "rho_hat", r"$\rho_i$", f"{fig_dir}/{name}_rho.pdf")
    plot_training_curves(results, f"{fig_dir}/{name}_training.pdf", max_iter=max_iter)


# ====================== paper figures ======================

def pick_run(results, optimizer="sgd", l2_alpha=0.0):
    """The run with the given optimizer and L2 setting."""
    for r in results:
        if r.config.optimizer == optimizer and r.l2_alpha == l2_alpha:
            return r
    raise ValueError(f"No run with optimizer={optimizer}, l2_alpha={l2_alpha}")


def paper_strategy_utility(result, outfile=None, titles=("Strategy", "Expected utility")):
    """Left: strategy vs best response. Right: expected utility of the strategy vs the best response.
    One row for symmetric games, one row per player for asymmetric games."""
    players = list(result.curves)
    symmetric = result.game_def["symmetric"]
    n_rows = len(players)

    fig, axes = plt.subplots(n_rows, 2, figsize=(PAPER_WIDTH, 0.4 * PAPER_WIDTH * n_rows), dpi=300,
                             squeeze=False, constrained_layout=True)
    for row, i in enumerate(players):
        c = result.curves[i]
        sub = "i" if symmetric else str(i + 1)
        panels = [("a_nn", "a_hat", rf"$a_{sub}$", titles[0]),
                  ("rho_nn", "rho_hat", rf"$\rho_{sub}$", titles[1])]
        for col, (key_nn, key_br, ylabel, title) in enumerate(panels):
            ax = axes[row, col]
            ax.plot(c["theta"], c[key_br], label="Best response", **BR_STYLE)
            ax.plot(c["theta"], c[key_nn], label="Neural network", **OPT_STYLE["sgd"])
            _style_ax(ax,
                      xlabel=rf"$\theta_{sub}$" if row == n_rows - 1 else None,
                      ylabel=ylabel,
                      title=title if symmetric else f"Player {i + 1}: {title.lower()}",
                      legend=True)
    _save(fig, outfile)


def paper_pareto_samples(result, outfile=None, player=0, bins=None, yscale="sqrt",
                         yticks=(0, 5, 10, 25, 50, 100, 200, 300)):
    """Side by side: own training types (stratified) and opponent types (Pareto), same bins and y-axis."""
    cfg, gd = result.config, result.game_def
    lo, hi = float(gd["theta_low"][player]), float(gd["theta_high"][player])
    if bins is None:
        bins = getattr(cfg, "strat_bins", 100)
    edges = np.linspace(lo, hi, bins + 1) if np.isscalar(bins) else np.asarray(bins)

    panels = [(result.train_types["own"][:, player, 0], "#0072B2", "Own types (stratified)"),
              (result.train_types["opp"][:, player, 0], "#D55E00", "Opponent types (Pareto)")]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), dpi=300, sharey=True, constrained_layout=True)
    for col, (data, color, title) in enumerate(panels):
        ax = axes[col]
        ax.hist(data, bins=edges, color=color, alpha=0.8, linewidth=0)
        if yscale == "log":
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.8)
        elif yscale == "sqrt":
            ax.set_yscale("function", functions=(np.sqrt, np.square))
            ax.set_ylim(bottom=0)
        ax.set_xlim(lo, hi)
        _style_ax(ax, xlabel=r"$\theta_i$", ylabel="Count" if col == 0 else None,
                  title=title, legend=False)
    if yticks is not None:
        top = axes[0].get_ylim()[1]
        axes[0].set_yticks([t for t in yticks if t <= top])
    _save(fig, outfile)