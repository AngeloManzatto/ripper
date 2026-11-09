"""
Created on Thu Oct 22 22:04:39 2026

@author: Angelo Antonio Manzatto

"""

###############################################################################
# libraries
###############################################################################

import matplotlib.pyplot as plt

###############################################################################
# Constants
###############################################################################

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BAND_LIGHT = "#9cc3f0"   # species bands alternate between two tones of one hue;
BAND_DARK = BLUE         # identity comes from the printed id, not from color
LABEL_MIN_SHARE = 0.04   # a band gets its id printed only if it holds >= 4% of the population


def _check(history):
    if not history:
        raise ValueError("history is empty: nothing to plot")


def fitness_series(history):
    """{"generation", "best", "mean", "best_ever"} as lists, one value per record."""
    _check(history)
    return {
        "generation": [r["generation"] for r in history],
        "best": [r["best_fitness"] for r in history],
        "mean": [r["mean_fitness"] for r in history],
        "best_ever": [r["best_ever_fitness"] for r in history],
    }


def species_matrix(history):
    """Sizes of every species in every generation.

    Returns (generations, ids, sizes): ids sorted ascending (oldest species first),
    sizes[k][t] = size of species ids[k] in generation t (0 when it does not exist then).
    """
    _check(history)
    generations = [r["generation"] for r in history]
    ids = sorted({s["id"] for r in history for s in r["species"]})
    sizes = [[0] * len(history) for _ in ids]
    row = {sid: k for k, sid in enumerate(ids)}
    for t, r in enumerate(history):
        for s in r["species"]:
            sizes[row[s["id"]]][t] = s["size"]
    return generations, ids, sizes


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK_SOFT)
    ax.set_xlabel("generation", color=INK_SOFT)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SOFT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_fitness(history, ax=None):
    """Best, mean and best-ever fitness per generation."""
    s = fitness_series(history)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    g = s["generation"]
    ax.plot(g, s["best_ever"], color=INK_SOFT, linewidth=1.5, linestyle="--", label="best ever")
    ax.plot(g, s["best"], color=BLUE, linewidth=2, label="best")
    ax.plot(g, s["mean"], color=ORANGE, linewidth=2, label="mean")
    _style(ax, "Fitness", "fitness")
    ax.legend(frameon=False, labelcolor=INK_SOFT, loc="lower right")
    return ax


def plot_species(history, ax=None):
    """Stacked area of species sizes: the width of a band is how many genomes it holds."""
    generations, ids, sizes = species_matrix(history)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    colors = [BAND_DARK if k % 2 == 0 else BAND_LIGHT for k in range(len(ids))]
    ax.stackplot(generations, sizes, colors=colors, edgecolor=SURFACE, linewidth=1.0)
    _style(ax, "Species (band height = members, number = species id)", "genomes")
    if len(generations) > 1:
        ax.set_xlim(generations[0], generations[-1])
    total = sum(sizes[k][-1] for k in range(len(ids)))
    top = 0
    for k, sid in enumerate(ids):
        size = sizes[k][-1]
        if total and size / total >= LABEL_MIN_SHARE:
            ax.text(generations[-1], top + size / 2, f" {sid}", va="center", ha="left",
                    fontsize=8, color=INK_SOFT, clip_on=False)
        top += size
    return ax


def complexity_series(history):
    """{"generation", "hidden_mean", "hidden_max", "connections_mean", "connections_max"} as lists."""
    _check(history)
    return {
        "generation": [r["generation"] for r in history],
        "hidden_mean": [r["mean_hidden"] for r in history],
        "hidden_max": [r["max_hidden"] for r in history],
        "connections_mean": [r["mean_connections"] for r in history],
        "connections_max": [r["max_connections"] for r in history],
    }


def plot_complexity(history, ax=None):
    """How big the genomes are: hidden nodes and connections, mean (solid) and largest (dashed)."""
    s = complexity_series(history)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    g = s["generation"]
    ax.plot(g, s["connections_max"], color=ORANGE, linewidth=1.5, linestyle="--", label="connections, largest")
    ax.plot(g, s["connections_mean"], color=ORANGE, linewidth=2, label="connections, mean")
    ax.plot(g, s["hidden_max"], color=BLUE, linewidth=1.5, linestyle="--", label="hidden nodes, largest")
    ax.plot(g, s["hidden_mean"], color=BLUE, linewidth=2, label="hidden nodes, mean")
    _style(ax, "Genome size", "count")
    ax.legend(frameon=False, labelcolor=INK_SOFT, loc="upper left")
    return ax


def plot_overview(history, path=None):
    """Fitness, species and genome size in one figure. Saves a PNG when path is given."""
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(8, 10), sharex=True, facecolor=SURFACE)
    plot_fitness(history, a)
    plot_species(history, b)
    plot_complexity(history, c)
    a.set_xlabel("")
    b.set_xlabel("")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    return fig