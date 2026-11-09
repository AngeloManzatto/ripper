"""
Created on Sun Oct 18 09:03:14 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.genome import init_genome
from neat.innovations import init_innovations, get_connection_innovation

###############################################################################
# init_innovations: shape
###############################################################################

def test_keys():
    innovations = init_innovations(3, 2)
    assert set(innovations.keys()) == {"connections", "next_innovation",
                                       "next_node_id", "splits"}


def test_counters_for_a_small_network():
    innovations = init_innovations(3, 2)
    assert len(innovations["connections"]) == 6
    assert innovations["next_innovation"] == 6
    assert innovations["next_node_id"] == 5
    assert innovations["splits"] == {}


def test_counters_for_the_project_size():
    innovations = init_innovations(29, 4)
    assert len(innovations["connections"]) == 116
    assert innovations["next_innovation"] == 116
    assert innovations["next_node_id"] == 33


def test_smallest_network():
    innovations = init_innovations(1, 1)
    assert innovations["connections"] == {(0, 1): 0}
    assert innovations["next_innovation"] == 1
    assert innovations["next_node_id"] == 2

###############################################################################
# init_innovations: the starting numbers
###############################################################################

def test_pairs_are_every_input_with_every_output():
    innovations = init_innovations(4, 3)
    expected = []
    for src in range(4):
        for dst in range(4, 7):
            expected.append((src, dst))
    assert sorted(innovations["connections"].keys()) == expected


def test_numbers_are_zero_to_n_minus_one_without_repeats():
    innovations = init_innovations(4, 3)
    assert sorted(innovations["connections"].values()) == list(range(12))


def test_numbering_follows_position():
    n_inputs = 4
    n_outputs = 3
    innovations = init_innovations(n_inputs, n_outputs)
    for (src, dst), number in innovations["connections"].items():
        assert number == src * n_outputs + (dst - n_inputs)


def test_numbers_match_the_connections_of_init_genome():
    # the registry and the genome builder must agree on every starting number
    sizes = [(1, 1), (3, 2), (4, 3), (29, 4)]
    for n_inputs, n_outputs in sizes:
        genome = init_genome(n_inputs, n_outputs, np.random.default_rng(0))
        innovations = init_innovations(n_inputs, n_outputs)
        for number, connection in genome["connections"].items():
            pair = (connection["src"], connection["dst"])
            assert innovations["connections"][pair] == number


def test_keys_are_int_pairs_and_values_are_plain_ints():
    innovations = init_innovations(3, 2)
    for pair, number in innovations["connections"].items():
        assert type(pair) is tuple and len(pair) == 2
        assert type(pair[0]) is int and type(pair[1]) is int
        assert type(number) is int
    assert type(innovations["next_innovation"]) is int
    assert type(innovations["next_node_id"]) is int


def test_two_registries_are_independent():
    a = init_innovations(3, 2)
    b = init_innovations(3, 2)
    assert a == b
    get_connection_innovation(a, 3, 4)
    a["splits"][0] = 99
    assert b == init_innovations(3, 2)

###############################################################################
# get_connection_innovation: a pair that already exists
###############################################################################

def test_existing_pair_returns_its_number():
    innovations = init_innovations(3, 2)
    assert get_connection_innovation(innovations, 0, 3) == 0
    assert get_connection_innovation(innovations, 0, 4) == 1
    assert get_connection_innovation(innovations, 2, 4) == 5


def test_existing_pair_does_not_change_the_registry():
    innovations = init_innovations(3, 2)
    snapshot = copy.deepcopy(innovations)
    get_connection_innovation(innovations, 1, 3)
    get_connection_innovation(innovations, 1, 3)
    assert innovations == snapshot

###############################################################################
# get_connection_innovation: a new pair
###############################################################################

def test_new_pair_gets_the_next_free_number():
    innovations = init_innovations(3, 2)
    assert get_connection_innovation(innovations, 3, 4) == 6


def test_new_pair_is_stored_and_the_counter_moves_by_one():
    innovations = init_innovations(3, 2)
    get_connection_innovation(innovations, 3, 4)
    assert innovations["connections"][(3, 4)] == 6
    assert innovations["next_innovation"] == 7
    assert len(innovations["connections"]) == 7


def test_asking_twice_gives_the_same_number_and_counts_once():
    innovations = init_innovations(3, 2)
    first = get_connection_innovation(innovations, 3, 4)
    second = get_connection_innovation(innovations, 3, 4)
    assert first == second == 6
    assert innovations["next_innovation"] == 7


def test_different_new_pairs_get_consecutive_numbers():
    innovations = init_innovations(3, 2)
    a = get_connection_innovation(innovations, 3, 4)
    b = get_connection_innovation(innovations, 4, 3)
    c = get_connection_innovation(innovations, 5, 6)
    assert (a, b, c) == (6, 7, 8)


def test_direction_matters():
    innovations = init_innovations(3, 2)
    forward = get_connection_innovation(innovations, 5, 6)
    backward = get_connection_innovation(innovations, 6, 5)
    assert forward != backward


def test_result_is_a_plain_int():
    innovations = init_innovations(3, 2)
    assert type(get_connection_innovation(innovations, 3, 4)) is int
    assert type(get_connection_innovation(innovations, 0, 3)) is int


def test_many_new_pairs_stay_unique_and_contiguous():
    innovations = init_innovations(3, 2)
    numbers = []
    for src in range(5, 25):
        numbers.append(get_connection_innovation(innovations, src, src + 1))
    assert numbers == list(range(6, 26))
    assert innovations["next_innovation"] == 26


def test_the_same_pair_asked_from_two_places_agrees():
    # two genomes of one population ask for the same new connection
    innovations = init_innovations(3, 2)
    from_genome_a = get_connection_innovation(innovations, 3, 4)
    from_genome_b = get_connection_innovation(innovations, 3, 4)
    assert from_genome_a == from_genome_b


def test_other_fields_are_not_touched():
    innovations = init_innovations(3, 2)
    get_connection_innovation(innovations, 3, 4)
    get_connection_innovation(innovations, 0, 3)
    assert innovations["next_node_id"] == 5
    assert innovations["splits"] == {}


def test_same_sequence_of_calls_gives_same_numbers():
    a = init_innovations(3, 2)
    b = init_innovations(3, 2)
    pairs = [(3, 4), (5, 6), (3, 4), (6, 5), (0, 5)]
    numbers_a = [get_connection_innovation(a, s, d) for s, d in pairs]
    numbers_b = [get_connection_innovation(b, s, d) for s, d in pairs]
    assert numbers_a == numbers_b
    assert a == b

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"innovations: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()