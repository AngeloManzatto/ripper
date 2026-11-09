"""
Created on Fri Oct 23 07:23:26 2026

@author: Angelo Antonio Manzatto

"""

###############################################################################
# libraries
###############################################################################

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neat.diagnostics.genome_view import node_depths, node_layout, genome_table, draw_genome


def conn(src, dst, w=1.0, on=True):
    return {"src": src, "dst": dst, "weight": w, "enabled": on}


def node(kind, act="sigmoid"):
    return {"type": kind} if kind == "input" else {"type": kind, "bias": 0.5, "activation": act}


def make(hidden, conns, n_in=2):
    nodes = {i: node("input") for i in range(n_in)}
    nodes[n_in] = node("output")
    for h in hidden:
        nodes[h] = node("hidden", "tanh")
    return {"n_inputs": n_in, "n_outputs": 1, "nodes": nodes, "connections": conns}


# 0,1 -> 3 -> 4 -> 2 (chain of two hidden nodes) plus a direct 0 -> 2
CHAIN = make([3, 4], {0: conn(0, 3), 1: conn(1, 3, -2.0), 2: conn(3, 4), 3: conn(4, 2), 4: conn(0, 2), 5: conn(1, 2, 1.0, False)})
FLAT = make([], {0: conn(0, 2), 1: conn(1, 2)})


def test_depths_of_a_chain():
    d = node_depths(CHAIN)
    assert d == {0: 0, 1: 0, 3: 1, 4: 2, 2: 3}


def test_depths_without_hidden_nodes():
    assert node_depths(FLAT) == {0: 0, 1: 0, 2: 1}


def test_outputs_are_always_in_the_last_column():
    g = make([3], {0: conn(0, 3), 1: conn(3, 2), 2: conn(1, 2)})
    d = node_depths(g)
    assert d[2] == max(d.values()) and d[2] == 2


def test_disabled_connections_do_not_set_depth():
    g = make([3], {0: conn(0, 3, 1.0, False), 1: conn(3, 2)})
    assert node_depths(g)[3] == 1
    g = make([3, 4], {0: conn(0, 3), 1: conn(3, 2), 2: conn(1, 4), 3: conn(3, 4, 1.0, False), 4: conn(4, 2)})
    assert node_depths(g)[4] == 1          # the disabled 3 -> 4 would have pushed it to 2


def test_cycle_fails_loudly():
    g = make([3, 4], {0: conn(0, 3), 1: conn(3, 4), 2: conn(4, 3), 3: conn(4, 2)})
    try:
        node_depths(g)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_layout_x_runs_from_inputs_to_outputs():
    lay = node_layout(CHAIN)
    assert lay[0][0] == 0.0 and lay[1][0] == 0.0 and lay[2][0] == 1.0
    assert lay[0][0] < lay[3][0] < lay[4][0] < lay[2][0]


def test_layout_spreads_a_column_evenly():
    lay = node_layout(CHAIN)
    assert lay[0][1] == 1 / 3 and lay[1][1] == 2 / 3
    assert lay[2][1] == 0.5


def test_layout_has_every_node_once():
    assert sorted(node_layout(CHAIN)) == [0, 1, 2, 3, 4]


def test_table_lists_every_node_and_connection():
    text = genome_table(CHAIN)
    lines = text.splitlines()
    assert len(lines) == 2 + 5 + 1 + 2 + 6
    assert "tanh" in text and "-2.000" in text
    assert sum(1 for l in lines if l.endswith(" no")) == 1


def test_draw_has_a_circle_per_node_and_an_arrow_per_connection():
    ax = draw_genome(CHAIN)
    assert len(ax.collections) == 5
    arrows = [c for c in ax.get_children() if type(c).__name__ == "Annotation" and c.arrow_patch is not None]
    assert len(arrows) == 6
    plt.close("all")


def test_draw_colors_sign_and_disabled():
    ax = draw_genome(CHAIN)
    arrows = [c.arrow_patch for c in ax.get_children() if type(c).__name__ == "Annotation" and c.arrow_patch is not None]
    styles = [(a.get_linestyle(), a.get_linewidth()) for a in arrows]
    assert sum(1 for s in styles if s[0] == "--") == 1
    solid = [w for st, w in styles if st != "--"]
    assert max(solid) > min(solid)   # |weight| changes thickness (enabled lines only)
    colors = {matplotlib.colors.to_hex(a.get_edgecolor()) for a in arrows}
    assert {"#2a78d6", "#eb6834"} <= colors
    plt.close("all")


def test_draw_works_on_a_flat_genome_and_uses_a_given_axes():
    fig, ax = plt.subplots()
    assert draw_genome(FLAT, ax) is ax
    plt.close("all")


def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"diagnostics genome view: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()