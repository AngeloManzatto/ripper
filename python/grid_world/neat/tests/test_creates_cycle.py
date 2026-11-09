"""
Created on Sat Oct 17 22:32:55 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.genome import init_genome
from neat.graph import creates_cycle


def genome_from_edges(n_nodes, edges, disabled=()):
    """A genome with n_nodes nodes and one connection per (src, dst) pair."""
    nodes = {}
    for node_id in range(n_nodes):
        nodes[node_id] = {"type": "hidden", "bias": 0.0, "activation": "tanh"}
    connections = {}
    for innovation, (src, dst) in enumerate(edges):
        connections[innovation] = {
            "src": src,
            "dst": dst,
            "weight": 1.0,
            "enabled": (src, dst) not in disabled,
        }
    return {"n_inputs": 0, "n_outputs": 0, "nodes": nodes, "connections": connections}

###############################################################################
# The simplest cases
###############################################################################

def test_self_loop_is_a_cycle():
    genome = genome_from_edges(3, [])
    assert creates_cycle(genome, 1, 1) is True


def test_self_loop_is_a_cycle_even_with_connections_elsewhere():
    genome = genome_from_edges(3, [(0, 1), (1, 2)])
    assert creates_cycle(genome, 2, 2) is True


def test_empty_genome_has_no_cycle_for_two_different_nodes():
    genome = genome_from_edges(3, [])
    assert creates_cycle(genome, 0, 1) is False
    assert creates_cycle(genome, 1, 0) is False


def test_returns_a_real_bool():
    genome = genome_from_edges(3, [(0, 1)])
    assert type(creates_cycle(genome, 0, 2)) is bool
    assert type(creates_cycle(genome, 1, 0)) is bool

###############################################################################
# Direct and indirect loops
###############################################################################

def test_reverse_of_an_existing_connection_is_a_cycle():
    genome = genome_from_edges(2, [(0, 1)])
    assert creates_cycle(genome, 1, 0) is True


def test_closing_a_chain_is_a_cycle():
    genome = genome_from_edges(3, [(0, 1), (1, 2)])
    assert creates_cycle(genome, 2, 0) is True


def test_going_back_inside_a_chain_is_a_cycle():
    genome = genome_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert creates_cycle(genome, 2, 1) is True
    assert creates_cycle(genome, 3, 1) is True
    assert creates_cycle(genome, 3, 0) is True


def test_shortcut_forward_is_not_a_cycle():
    genome = genome_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert creates_cycle(genome, 0, 2) is False
    assert creates_cycle(genome, 0, 3) is False
    assert creates_cycle(genome, 1, 3) is False


def test_new_forward_edge_in_a_different_branch_is_not_a_cycle():
    # 0 -> 1 -> 3 and 0 -> 2 -> 3: adding 1 -> 2 or 2 -> 1 is fine
    genome = genome_from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert creates_cycle(genome, 1, 2) is False
    assert creates_cycle(genome, 2, 1) is False


def test_diamond_closing_back_to_the_top_is_a_cycle():
    genome = genome_from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert creates_cycle(genome, 3, 0) is True
    assert creates_cycle(genome, 3, 1) is True
    assert creates_cycle(genome, 3, 2) is True


def test_unrelated_components_do_not_create_a_cycle():
    genome = genome_from_edges(6, [(0, 1), (1, 2), (3, 4), (4, 5)])
    assert creates_cycle(genome, 2, 3) is False       # joins two separate chains
    assert creates_cycle(genome, 5, 0) is False

###############################################################################
# Disabled connections and duplicates
###############################################################################

def test_disabled_connections_still_count():
    genome = genome_from_edges(3, [(0, 1), (1, 2)], disabled=[(1, 2)])
    assert creates_cycle(genome, 2, 0) is True


def test_a_pair_that_already_exists_is_not_a_cycle():
    # duplicates are somebody else's job
    genome = genome_from_edges(3, [(0, 1), (1, 2)])
    assert creates_cycle(genome, 0, 1) is False
    assert creates_cycle(genome, 1, 2) is False

###############################################################################
# Real genomes
###############################################################################

def test_fresh_genome_input_to_output_is_not_a_cycle():
    genome = init_genome(3, 2, np.random.default_rng(0))
    for src in range(3):
        for dst in range(3, 5):
            assert creates_cycle(genome, src, dst) is False


def test_fresh_genome_output_to_input_is_a_cycle():
    genome = init_genome(3, 2, np.random.default_rng(0))
    for src in range(3, 5):
        for dst in range(3):
            assert creates_cycle(genome, src, dst) is True


def test_fresh_genome_output_to_output_is_not_a_cycle():
    genome = init_genome(3, 2, np.random.default_rng(0))
    assert creates_cycle(genome, 3, 4) is False
    assert creates_cycle(genome, 4, 3) is False

###############################################################################
# Robustness
###############################################################################

def test_genome_is_not_modified():
    genome = genome_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    snapshot = copy.deepcopy(genome)
    creates_cycle(genome, 3, 0)
    creates_cycle(genome, 0, 3)
    assert genome == snapshot


def test_very_long_chain_does_not_hit_the_recursion_limit():
    n = 5000
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    genome = genome_from_edges(n, edges)
    assert creates_cycle(genome, n - 1, 0) is True
    assert creates_cycle(genome, 0, n - 1) is False


def test_terminates_when_the_genome_already_has_a_loop_elsewhere():
    # nodes 1 and 2 loop on each other; asking about node 0 -> 3 must still finish
    genome = genome_from_edges(4, [(1, 2), (2, 1), (0, 1)])
    assert creates_cycle(genome, 0, 3) is False
    assert creates_cycle(genome, 2, 0) is True

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"creates_cycle: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()