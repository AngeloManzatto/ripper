"""
Created on Fri Oct 23 08:44:49 2026

@author: Angelo Antonio Manzatto

"""

###############################################################################
# libraries
###############################################################################


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neat.diagnostics.distances import pairwise_distances, share_below, plot_distances

CFG = {"compatibility_c1": 2.0, "compatibility_c2": 3.0, "compatibility_c3": 0.5,
       "compatibility_n_min": 20, "compatibility_threshold": 4.0}


def genome(weights):
    return {"n_inputs": 1, "n_outputs": 1, "nodes": {},
            "connections": {k: {"src": 0, "dst": 1, "weight": w, "enabled": True} for k, w in weights.items()}}


A = genome({0: 1.0, 1: 2.0})
B = genome({0: 2.0, 2: 5.0})      # vs A: 1 matching (diff 1.0), 1 disjoint (gene 1), 1 excess (gene 2)
C = genome({0: 1.0, 1: 2.0})      # identical to A


def test_distance_of_one_pair_uses_the_config_weights():
    assert pairwise_distances([A, B], CFG) == [2.0 * 1 + 3.0 * 1 + 0.5 * 1.0]


def test_identical_genomes_are_at_distance_zero():
    assert pairwise_distances([A, C], CFG) == [0.0]


def test_every_pair_once_in_order():
    d = pairwise_distances([A, B, C], CFG)
    assert len(d) == 3
    assert d == [5.5, 0.0, 5.5]          # (A,B) (A,C) (B,C)


def test_needs_two_genomes():
    for g in ([], [A]):
        try:
            pairwise_distances(g, CFG)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_share_below_is_strict():
    assert share_below([1.0, 2.0, 3.0, 4.0], 3.0) == 0.5
    try:
        share_below([], 1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_plot_counts_all_pairs_and_marks_the_threshold():
    ax = plot_distances([A, B, C], CFG, bins=5)
    assert sum(p.get_height() for p in ax.patches) == 3
    line = ax.get_lines()[0]
    assert list(line.get_xdata()) == [4.0, 4.0]
    assert "33% of 3 pairs" in ax.get_title(loc="left")      # only the identical pair is below 4.0
    plt.close("all")


def test_plot_uses_a_given_axes():
    fig, ax = plt.subplots()
    assert plot_distances([A, B], CFG, ax=ax) is ax
    plt.close("all")


def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"diagnostics distances: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()