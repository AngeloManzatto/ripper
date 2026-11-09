"""
Created on Fri Oct 23 07:24:43 2026

@author: Angelo Antonio Manzatto

"""

###############################################################################
# libraries
###############################################################################

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neat.diagnostics.plots import (fitness_series, species_matrix, plot_fitness,
                                    plot_species, plot_overview,
                                    complexity_series, plot_complexity)


def rec(gen, best, mean, best_ever, sizes):
    return {"mean_hidden": gen + 0.5, "max_hidden": gen + 2, "mean_connections": 2.0 * gen + 3, "max_connections": 3 * gen + 4,
            "generation": gen, "best_fitness": best, "mean_fitness": mean, "best_ever_fitness": best_ever,
            "species": [{"id": i, "size": n, "best": 0.0, "mean": 0.0, "since_improved": 0} for i, n in sizes.items()]}


HISTORY = [
    rec(0, 1.0, 0.5, 1.0, {0: 10}),
    rec(1, 2.0, 0.8, 2.0, {0: 6, 4: 4}),
    rec(2, 1.5, 0.9, 2.0, {4: 7, 9: 3}),   # species 0 is gone, 9 is new
]


def test_fitness_series_values():
    s = fitness_series(HISTORY)
    assert s["generation"] == [0, 1, 2]
    assert s["best"] == [1.0, 2.0, 1.5]
    assert s["mean"] == [0.5, 0.8, 0.9]
    assert s["best_ever"] == [1.0, 2.0, 2.0]


def test_species_matrix_values_and_zero_for_missing():
    generations, ids, sizes = species_matrix(HISTORY)
    assert generations == [0, 1, 2]
    assert ids == [0, 4, 9]
    assert sizes == [[10, 6, 0], [0, 4, 7], [0, 0, 3]]


def test_species_matrix_columns_sum_to_population():
    _, _, sizes = species_matrix(HISTORY)
    for t in range(3):
        assert sum(row[t] for row in sizes) == 10


def test_ids_are_sorted_even_when_they_appear_out_of_order():
    h = [rec(0, 1, 1, 1, {7: 5, 2: 5})]
    assert species_matrix(h)[1] == [2, 7]


def test_empty_history_fails_loudly():
    for f in (fitness_series, species_matrix, complexity_series, plot_complexity, plot_fitness, plot_species, plot_overview,
                                    complexity_series, plot_complexity):
        try:
            f([])
        except ValueError:
            continue
        raise AssertionError(f.__name__)


def test_plot_fitness_draws_three_labelled_lines_with_the_data():
    ax = plot_fitness(HISTORY)
    lines = {l.get_label(): list(l.get_ydata()) for l in ax.get_lines()}
    assert lines == {"best ever": [1.0, 2.0, 2.0], "best": [1.0, 2.0, 1.5], "mean": [0.5, 0.8, 0.9]}
    plt.close("all")


def test_plot_species_draws_one_band_per_species_and_labels_big_ones():
    ax = plot_species(HISTORY)
    assert len(ax.collections) == 3
    labels = sorted(t.get_text().strip() for t in ax.texts)
    assert labels == ["4", "9"]            # species 0 is empty in the last generation
    plt.close("all")


def test_small_bands_get_no_label():
    h = [rec(0, 1, 1, 1, {0: 99, 1: 1})]
    ax = plot_species(h)
    assert [t.get_text().strip() for t in ax.texts] == ["0"]
    plt.close("all")


def test_complexity_series_values():
    s = complexity_series(HISTORY)
    assert s["hidden_mean"] == [0.5, 1.5, 2.5] and s["hidden_max"] == [2, 3, 4]
    assert s["connections_mean"] == [3.0, 5.0, 7.0] and s["connections_max"] == [4, 7, 10]


def test_plot_complexity_draws_four_lines_with_the_data():
    ax = plot_complexity(HISTORY)
    lines = {l.get_label(): list(l.get_ydata()) for l in ax.get_lines()}
    assert lines["hidden nodes, mean"] == [0.5, 1.5, 2.5]
    assert lines["connections, largest"] == [4, 7, 10]
    assert len(lines) == 4
    plt.close("all")


def test_plot_functions_use_a_given_axes():
    fig, (a, b) = plt.subplots(1, 2)
    assert plot_fitness(HISTORY, a) is a
    assert plot_species(HISTORY, b) is b
    plt.close("all")


def test_overview_saves_a_png():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "overview.png")
        fig = plot_overview(HISTORY, path)
        assert os.path.getsize(path) > 1000
        assert len(fig.axes) == 3
    plt.close("all")


def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"diagnostics plots: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()