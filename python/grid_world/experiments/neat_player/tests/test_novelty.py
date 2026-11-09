"""
Tests for experiments/neat_player/novelty.py.

jaccard_distance(a, b)               1 - |a & b| / |a | b|, 0.0 for two empty sets
behaviour_distance(visits_a, visits_b)   mean Jaccard distance over the games
novelty_scores(behaviours, k)        mean distance to the k nearest other genomes
mix_fitness(goal_rates, novelty, weight)   (1 - w) * goal + w * novelty / biggest novelty

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

from experiments.neat_player.novelty import jaccard_distance, behaviour_distance, novelty_scores, mix_fitness

###############################################################################
# helpers
###############################################################################

def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def games(*sets):
    """A behaviour: the list of cell sets, one per game, cells given as small integers."""
    return [set(s) for s in sets]

###############################################################################
# jaccard_distance
###############################################################################

def test_jaccard_equal_sets_is_zero():
    assert jaccard_distance({1, 2, 3}, {3, 2, 1}) == 0.0


def test_jaccard_disjoint_sets_is_one():
    assert jaccard_distance({1, 2}, {3, 4}) == 1.0


def test_jaccard_partial_overlap():
    # {1,2,3} and {2,3,4,5}: common 2, union 5
    assert close(jaccard_distance({1, 2, 3}, {2, 3, 4, 5}), 1 - 2 / 5)


def test_jaccard_is_symmetric():
    a, b = {1, 2, 3}, {3, 4}
    assert jaccard_distance(a, b) == jaccard_distance(b, a)


def test_jaccard_empty_cases():
    assert jaccard_distance(set(), set()) == 0.0
    assert jaccard_distance(set(), {1, 2}) == 1.0


def test_jaccard_accepts_frozensets_and_lists_without_changing_them():
    a = frozenset({1, 2})
    assert close(jaccard_distance(a, [2, 3]), 1 - 1 / 3)
    assert a == frozenset({1, 2})

###############################################################################
# behaviour_distance
###############################################################################

def test_behaviour_distance_is_the_mean_over_games():
    a = games({1, 2}, {5})
    b = games({1, 2}, {6})                  # game 0: 0.0, game 1: 1.0
    assert close(behaviour_distance(a, b), 0.5)


def test_behaviour_distance_compares_game_i_with_game_i():
    a = games({1}, {2})
    b = games({2}, {1})                     # same cells but in the other game: nothing in common
    assert behaviour_distance(a, b) == 1.0


def test_behaviour_distance_same_behaviour_is_zero():
    a = games({1, 2}, {3})
    assert behaviour_distance(a, copy.deepcopy(a)) == 0.0


def test_behaviour_distance_errors():
    assert raises(ValueError, behaviour_distance, games({1}), games({1}, {2}))
    assert raises(ValueError, behaviour_distance, [], [])

###############################################################################
# novelty_scores
###############################################################################

def test_novelty_of_a_genome_that_goes_elsewhere_is_biggest():
    crowd = games({1, 2}, {1, 2})
    odd = games({8, 9}, {8, 9})
    scores = novelty_scores([crowd, copy.deepcopy(crowd), copy.deepcopy(crowd), odd], k=2)
    # a member of the crowd: nearest two are the other two members of the crowd (0.0, 0.0)
    assert scores[:3] == [0.0, 0.0, 0.0]
    # the odd one: distance 1.0 to everybody
    assert scores[3] == 1.0


def test_novelty_uses_only_the_k_nearest():
    base = games({1, 2, 3, 4})
    near = games({1, 2, 3})                 # distance to base 1 - 3/4 = 0.25
    far = games({9})                        # distance to base 1.0, to near 1.0
    scores = novelty_scores([base, near, far], k=1)
    assert close(scores[0], 0.25) and close(scores[1], 0.25) and close(scores[2], 1.0)


def test_novelty_k_larger_than_the_population_uses_all_others():
    a, b, c = games({1}), games({1, 2}), games({3})
    big = novelty_scores([a, b, c], k=50)
    exact = novelty_scores([a, b, c], k=2)
    assert big == exact


def test_novelty_mean_of_k_neighbours():
    a = games({1})
    b = games({1, 2})                       # a-b: 0.5
    c = games({3})                          # a-c: 1.0, b-c: 1.0
    scores = novelty_scores([a, b, c], k=2)
    assert close(scores[0], (0.5 + 1.0) / 2) and close(scores[1], (0.5 + 1.0) / 2) and close(scores[2], 1.0)


def test_novelty_order_follows_the_input_order():
    a, odd = games({1}), games({7})
    assert novelty_scores([odd, a, copy.deepcopy(a)], k=1) == [1.0, 0.0, 0.0]


def test_novelty_two_genomes():
    assert novelty_scores([games({1}), games({2})], k=5) == [1.0, 1.0]


def test_novelty_errors_and_input_not_changed():
    b = [games({1}), games({2})]
    before = copy.deepcopy(b)
    novelty_scores(b, 1)
    assert b == before
    assert raises(ValueError, novelty_scores, [games({1})], 1)
    assert raises(ValueError, novelty_scores, [], 1)
    assert raises(ValueError, novelty_scores, b, 0)

###############################################################################
# mix_fitness
###############################################################################

def test_mix_weight_zero_is_exactly_the_goal_rates():
    goals = [0.15, 0.0, 0.7]
    assert mix_fitness(goals, [0.3, 0.9, 0.0], 0.0) == goals


def test_mix_weight_one_is_the_scaled_novelty():
    assert mix_fitness([0.9, 0.1], [0.2, 0.4], 1.0) == [0.5, 1.0]


def test_mix_in_between():
    result = mix_fitness([0.5, 0.0], [0.25, 0.5], 0.2)
    assert close(result[0], 0.8 * 0.5 + 0.2 * 0.5) and close(result[1], 0.0 + 0.2 * 1.0)


def test_mix_novelty_is_scaled_by_the_biggest():
    result = mix_fitness([0.0, 0.0, 0.0], [1.0, 2.0, 4.0], 0.5)
    assert result == [0.125, 0.25, 0.5]


def test_mix_no_novelty_at_all():
    assert mix_fitness([0.5, 0.25], [0.0, 0.0], 0.4) == [0.3, 0.15]


def test_mix_errors_and_inputs_not_changed():
    goals, nov = [0.5, 0.25], [0.2, 0.4]
    mix_fitness(goals, nov, 0.2)
    assert goals == [0.5, 0.25] and nov == [0.2, 0.4]
    assert raises(ValueError, mix_fitness, [0.5], [0.1, 0.2], 0.2)
    assert raises(ValueError, mix_fitness, [], [], 0.2)
    assert raises(ValueError, mix_fitness, [0.5], [0.1], -0.01)
    assert raises(ValueError, mix_fitness, [0.5], [0.1], 1.01)

###############################################################################
# run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"novelty: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()