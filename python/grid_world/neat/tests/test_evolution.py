"""
Created on Mon Oct 19 13:34:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################


"""
Tests for neat/evolution.py: init_population, evolve_one_generation, run_evolution.

This is the generation loop. It is GAME AGNOSTIC: it never plays anything. The
caller evaluates the genomes (in parallel if it wants) and hands the fitness
values in.

The evolution state is plain data:

    {"generation": int,           # number of generations already evolved
     "genomes": [genome, ...],    # the current population, to be evaluated
     "species_set": dict,         # from speciate
     "stagnation": dict,          # from update_stagnation
     "innovations": dict,         # the shared registry
     "best": None or {"fitness": float, "genome": genome, "generation": int}}

"best" is the best genome EVER evaluated (strictly better replaces it).

init_population(n_inputs, n_outputs, rng, cfg=None)
    A state at generation 0 with cfg["population_size"] minimal genomes, a fresh
    registry, no species, no stagnation record and best = None. rng required.

evolve_one_generation(state, fitnesses, rng, cfg=None) -> (new_state, stats)
    fitnesses[i] is the fitness of state["genomes"][i]. One step:
    speciate, update stagnation, remove stagnant species, update "best",
    compute offspring, reproduce. The new state has generation + 1 and a new
    population of the same size. stats describes the generation that was just
    evaluated:
        generation, best_fitness, mean_fitness, best_ever_fitness,
        n_species (species found by speciate), removed_species (stagnant ids).
    A wrong number of fitnesses or a NaN raises ValueError. The state given is
    never modified (the registry is copied, not shared).

run_evolution(state, evaluate, rng, cfg=None, on_generation=None) -> (state, history)
    Repeats: fitnesses = evaluate(state["genomes"]); evolve_one_generation; until
    state["generation"] reaches cfg["max_generations"] or the best-ever fitness
    reaches cfg["fitness_threshold"] (None = never stop early). history is the
    list of stats; on_generation(stats) is called after every generation.
    Starting from a state whose generation is already max_generations does
    nothing. The result is in state["best"].

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.config import default_config
from neat.genome import init_genome
from neat.innovations import init_innovations
from neat.speciation import init_species
from neat.evolution import init_population, evolve_one_generation, run_evolution

###############################################################################
# helpers
###############################################################################

def small_cfg(**overrides):
    cfg = default_config()
    cfg["population_size"] = 30
    cfg["max_generations"] = 10
    cfg["fitness_threshold"] = None
    cfg["compatibility_threshold"] = 1.5
    cfg["node_add_prob"] = 0.15
    cfg["conn_add_prob"] = 0.25
    cfg.update(overrides)
    return cfg


def toy_fitness(genome):
    """Sum of the weights of the enabled connections, plus 0.1 per hidden node."""
    total = 0.0
    for c in genome["connections"].values():
        if c["enabled"]:
            total += c["weight"]
    for node in genome["nodes"].values():
        if node["type"] == "hidden":
            total += 0.1
    return total


def toy_evaluate(genomes):
    return [toy_fitness(g) for g in genomes]


def tagged(weight):
    return {"n_inputs": 1, "n_outputs": 1,
            "nodes": {0: {"type": "input"},
                      1: {"type": "output", "bias": 0.0, "activation": "sigmoid"}},
            "connections": {0: {"src": 0, "dst": 1, "weight": weight, "enabled": True}}}


def two_group_state(cfg):
    """Half the population has weight 0.0, half 5.0: two clear species."""
    n = cfg["population_size"]
    genomes = [tagged(0.0) for _ in range(n // 2)] + [tagged(5.0) for _ in range(n - n // 2)]
    return {"generation": 0, "genomes": genomes, "species_set": init_species(),
            "stagnation": {}, "innovations": init_innovations(1, 1), "best": None}


def frozen_cfg(**overrides):
    """cfg for the two-group state: no mutation, children are copies of a group,
    two species that are far apart."""
    cfg = small_cfg(population_size=12, compatibility_threshold=1.0, compatibility_c3=1.0,
                    weight_mutate_prob=0.0, bias_mutate_prob=0.0, activation_mutate_prob=0.0,
                    node_add_prob=0.0, conn_add_prob=0.0, mate_only_prob=1.0,
                    survival_threshold=1.0, max_stagnation=3, min_species_to_keep=1)
    cfg.update(overrides)
    return cfg


def is_acyclic(genome):
    incoming = {n: 0 for n in genome["nodes"]}
    for c in genome["connections"].values():
        incoming[c["dst"]] += 1
    ready = [n for n, count in incoming.items() if count == 0]
    removed = 0
    while ready:
        current = ready.pop()
        removed += 1
        for c in genome["connections"].values():
            if c["src"] == current:
                incoming[c["dst"]] -= 1
                if incoming[c["dst"]] == 0:
                    ready.append(c["dst"])
    return removed == len(genome["nodes"])

###############################################################################
# init_population
###############################################################################

def test_init_population_size_comes_from_cfg():
    rng = np.random.default_rng(1)
    state = init_population(4, 2, rng, small_cfg(population_size=17))
    assert len(state["genomes"]) == 17


def test_init_population_defaults_to_150():
    state = init_population(4, 2, np.random.default_rng(2))
    assert len(state["genomes"]) == 150


def test_init_population_state_shape():
    state = init_population(4, 2, np.random.default_rng(3), small_cfg())
    assert set(state.keys()) == {"generation", "genomes", "species_set", "stagnation",
                                 "innovations", "best"}
    assert state["generation"] == 0
    assert state["species_set"] == init_species()
    assert state["stagnation"] == {}
    assert state["best"] is None
    assert state["innovations"] == init_innovations(4, 2)


def test_init_population_genomes_are_minimal_and_all_different():
    state = init_population(4, 2, np.random.default_rng(4), small_cfg())
    weights = set()
    for genome in state["genomes"]:
        assert genome["n_inputs"] == 4 and genome["n_outputs"] == 2
        assert len(genome["connections"]) == 8
        assert len(genome["nodes"]) == 6
        weights.add(genome["connections"][0]["weight"])
    assert len(weights) == len(state["genomes"])
    assert len({id(g) for g in state["genomes"]}) == len(state["genomes"])


def test_init_population_uses_the_cfg_for_the_genomes():
    cfg = small_cfg(weight_init_mean=3.0, weight_init_stdev=0.0, output_activation="tanh")
    state = init_population(2, 1, np.random.default_rng(5), cfg)
    for genome in state["genomes"]:
        for c in genome["connections"].values():
            assert c["weight"] == 3.0
        assert genome["nodes"][2]["activation"] == "tanh"


def test_init_population_is_reproducible_and_needs_rng():
    a = init_population(3, 2, np.random.default_rng(6), small_cfg())
    b = init_population(3, 2, np.random.default_rng(6), small_cfg())
    c = init_population(3, 2, np.random.default_rng(7), small_cfg())
    assert a == b
    assert a != c
    try:
        init_population(3, 2)
    except TypeError:
        return
    raise AssertionError("init_population must require rng")

###############################################################################
# evolve_one_generation: one step
###############################################################################

def one_step(seed=1, cfg=None, fitnesses=None):
    cfg = cfg or small_cfg()
    rng = np.random.default_rng(seed)
    state = init_population(4, 2, rng, cfg)
    if fitnesses is None:
        fitnesses = toy_evaluate(state["genomes"])
    new_state, stats = evolve_one_generation(state, fitnesses, rng, cfg)
    return state, fitnesses, new_state, stats


def test_generation_goes_up_by_one():
    state, fitnesses, new_state, stats = one_step()
    assert state["generation"] == 0
    assert new_state["generation"] == 1


def test_population_size_stays_the_same():
    cfg = small_cfg(population_size=37)
    state, fitnesses, new_state, stats = one_step(cfg=cfg)
    assert len(new_state["genomes"]) == 37


def test_new_state_has_the_same_shape():
    state, fitnesses, new_state, stats = one_step()
    assert set(new_state.keys()) == set(state.keys())


def test_the_state_given_is_not_modified():
    cfg = small_cfg(node_add_prob=0.5, conn_add_prob=0.5)
    rng = np.random.default_rng(2)
    state = init_population(4, 2, rng, cfg)
    snapshot = copy.deepcopy(state)
    fitnesses = toy_evaluate(state["genomes"])
    for _ in range(3):
        evolve_one_generation(state, fitnesses, rng, cfg)
    assert state == snapshot


def test_the_registry_is_copied_not_shared():
    cfg = small_cfg(node_add_prob=1.0)
    state, fitnesses, new_state, stats = one_step(cfg=cfg)
    assert new_state["innovations"] is not state["innovations"]
    assert new_state["innovations"]["next_node_id"] > state["innovations"]["next_node_id"]
    assert state["innovations"]["next_node_id"] == 6


def test_new_genomes_are_new_objects_and_not_the_old_ones():
    state, fitnesses, new_state, stats = one_step()
    old_ids = {id(g) for g in state["genomes"]}
    for genome in new_state["genomes"]:
        assert id(genome) not in old_ids


def test_wrong_number_of_fitnesses_raises():
    cfg = small_cfg()
    rng = np.random.default_rng(3)
    state = init_population(4, 2, rng, cfg)
    for bad in ([1.0] * 29, [1.0] * 31, []):
        try:
            evolve_one_generation(state, bad, rng, cfg)
        except ValueError:
            continue
        raise AssertionError("a wrong number of fitnesses must raise ValueError")


def test_nan_fitness_raises():
    cfg = small_cfg()
    rng = np.random.default_rng(4)
    state = init_population(4, 2, rng, cfg)
    fitnesses = toy_evaluate(state["genomes"])
    fitnesses[7] = float("nan")
    try:
        evolve_one_generation(state, fitnesses, rng, cfg)
    except ValueError:
        return
    raise AssertionError("NaN fitness must raise ValueError")


def test_numpy_fitnesses_are_accepted():
    cfg = small_cfg()
    rng = np.random.default_rng(5)
    state = init_population(4, 2, rng, cfg)
    fitnesses = np.array(toy_evaluate(state["genomes"]))
    new_state, stats = evolve_one_generation(state, fitnesses, rng, cfg)
    assert type(stats["best_fitness"]) is float
    assert len(new_state["genomes"]) == 30


def test_the_default_cfg_is_used_when_none():
    rng = np.random.default_rng(6)
    state = init_population(3, 1, rng)
    new_state, stats = evolve_one_generation(state, toy_evaluate(state["genomes"]), rng)
    assert len(new_state["genomes"]) == 150

###############################################################################
# stats
###############################################################################

def test_stats_keys():
    state, fitnesses, new_state, stats = one_step()
    assert set(stats.keys()) == {"generation", "best_fitness", "mean_fitness",
                                 "best_ever_fitness", "n_species", "removed_species"}


def test_stats_describe_the_generation_that_was_evaluated():
    state, fitnesses, new_state, stats = one_step()
    assert stats["generation"] == 0
    assert stats["best_fitness"] == max(fitnesses)
    assert abs(stats["mean_fitness"] - sum(fitnesses) / len(fitnesses)) < 1e-12
    assert stats["best_ever_fitness"] == max(fitnesses)
    assert stats["n_species"] >= 1
    assert stats["removed_species"] == []


def test_stats_types():
    state, fitnesses, new_state, stats = one_step()
    assert type(stats["best_fitness"]) is float
    assert type(stats["mean_fitness"]) is float
    assert type(stats["best_ever_fitness"]) is float
    assert type(stats["n_species"]) is int
    assert type(stats["generation"]) is int


def test_n_species_counts_the_species_found():
    cfg = frozen_cfg()
    rng = np.random.default_rng(7)
    state = two_group_state(cfg)
    new_state, stats = evolve_one_generation(state, [1.0] * 12, rng, cfg)
    assert stats["n_species"] == 2

###############################################################################
# the best genome ever
###############################################################################

def test_best_is_a_copy_of_the_best_genome():
    state, fitnesses, new_state, stats = one_step()
    index = fitnesses.index(max(fitnesses))
    best = new_state["best"]
    assert best["fitness"] == max(fitnesses)
    assert best["genome"] == state["genomes"][index]
    assert best["genome"] is not state["genomes"][index]
    assert best["generation"] == 0
    assert type(best["fitness"]) is float


def test_the_best_genome_is_kept_in_the_next_population():
    for seed in (1, 2, 3, 4):
        state, fitnesses, new_state, stats = one_step(seed=seed)
        assert new_state["best"]["genome"] in new_state["genomes"]


def test_a_worse_generation_does_not_replace_the_best():
    cfg = small_cfg()
    rng = np.random.default_rng(8)
    state = init_population(4, 2, rng, cfg)
    state, _ = evolve_one_generation(state, [10.0] * 30, rng, cfg)
    best_before = copy.deepcopy(state["best"])
    state, stats = evolve_one_generation(state, [1.0] * 30, rng, cfg)
    assert state["best"] == best_before
    assert stats["best_fitness"] == 1.0
    assert stats["best_ever_fitness"] == 10.0


def test_an_equal_fitness_does_not_replace_the_best():
    cfg = small_cfg()
    rng = np.random.default_rng(9)
    state = init_population(4, 2, rng, cfg)
    state, _ = evolve_one_generation(state, [5.0] * 30, rng, cfg)
    first = copy.deepcopy(state["best"])
    state, _ = evolve_one_generation(state, [5.0] * 30, rng, cfg)
    assert state["best"] == first
    assert state["best"]["generation"] == 0


def test_a_better_generation_replaces_the_best():
    cfg = small_cfg()
    rng = np.random.default_rng(10)
    state = init_population(4, 2, rng, cfg)
    state, _ = evolve_one_generation(state, [5.0] * 30, rng, cfg)
    fitnesses = [5.0] * 30
    fitnesses[12] = 9.0
    expected = copy.deepcopy(state["genomes"][12])
    state, stats = evolve_one_generation(state, fitnesses, rng, cfg)
    assert state["best"]["fitness"] == 9.0
    assert state["best"]["genome"] == expected
    assert state["best"]["generation"] == 1


def test_best_does_not_share_the_old_best_object():
    cfg = small_cfg()
    rng = np.random.default_rng(11)
    state = init_population(4, 2, rng, cfg)
    state, _ = evolve_one_generation(state, [5.0] * 30, rng, cfg)
    new_state, _ = evolve_one_generation(state, [1.0] * 30, rng, cfg)
    assert new_state["best"] is not state["best"]
    assert new_state["best"]["genome"] is not state["best"]["genome"]

###############################################################################
# stagnation inside the loop
###############################################################################

def test_a_stagnant_species_is_removed_after_max_stagnation_generations():
    cfg = frozen_cfg(max_stagnation=3, min_species_to_keep=1)
    rng = np.random.default_rng(12)
    state = two_group_state(cfg)
    removed = []
    for _ in range(6):
        state, stats = evolve_one_generation(state, [1.0] * 12, rng, cfg)
        removed.append(stats["removed_species"])
    # constant fitness: nobody improves after generation 0; at generation 3
    # both species have waited 3 generations, the newer one (id 1) goes
    assert removed == [[], [], [], [1], [], []]


def test_the_removed_species_is_gone_from_state_and_population():
    cfg = frozen_cfg(max_stagnation=3, min_species_to_keep=1)
    rng = np.random.default_rng(13)
    state = two_group_state(cfg)
    for _ in range(4):
        state, stats = evolve_one_generation(state, [1.0] * 12, rng, cfg)
    assert set(state["species_set"]["species"]) == {0}
    assert set(state["stagnation"]) == {0}
    assert len(state["genomes"]) == 12
    assert all(g["connections"][0]["weight"] == 0.0 for g in state["genomes"])


def test_min_species_to_keep_protects_the_last_species():
    cfg = frozen_cfg(max_stagnation=3, min_species_to_keep=2)
    rng = np.random.default_rng(14)
    state = two_group_state(cfg)
    for _ in range(8):
        state, stats = evolve_one_generation(state, [1.0] * 12, rng, cfg)
        assert stats["removed_species"] == []
    assert len(state["species_set"]["species"]) == 2


def test_the_stagnation_record_follows_the_species():
    cfg = frozen_cfg(max_stagnation=50)
    rng = np.random.default_rng(15)
    state = two_group_state(cfg)
    fitness = lambda genomes: [1.0 if g["connections"][0]["weight"] < 2.0 else 1.0 for g in genomes]
    state, _ = evolve_one_generation(state, fitness(state["genomes"]), rng, cfg)
    assert state["stagnation"] == {0: {"best_fitness": 1.0, "last_improved": 0},
                                   1: {"best_fitness": 1.0, "last_improved": 0}}
    state, _ = evolve_one_generation(state, [2.0] * 12, rng, cfg)
    assert state["stagnation"][0] == {"best_fitness": 2.0, "last_improved": 1}

###############################################################################
# reproducibility
###############################################################################

def test_same_seed_gives_the_same_evolution():
    results = []
    for _ in range(2):
        cfg = small_cfg()
        rng = np.random.default_rng(16)
        state = init_population(4, 2, rng, cfg)
        for _ in range(6):
            state, stats = evolve_one_generation(state, toy_evaluate(state["genomes"]), rng, cfg)
        results.append(state)
    assert results[0] == results[1]


def test_different_seeds_give_different_evolutions():
    results = []
    for seed in (17, 18):
        cfg = small_cfg()
        rng = np.random.default_rng(seed)
        state = init_population(4, 2, rng, cfg)
        for _ in range(4):
            state, stats = evolve_one_generation(state, toy_evaluate(state["genomes"]), rng, cfg)
        results.append(state)
    assert results[0] != results[1]

###############################################################################
# run_evolution
###############################################################################

def test_run_stops_at_max_generations():
    cfg = small_cfg(max_generations=5)
    rng = np.random.default_rng(19)
    state = init_population(4, 2, rng, cfg)
    state, history = run_evolution(state, toy_evaluate, rng, cfg)
    assert state["generation"] == 5
    assert len(history) == 5
    assert [s["generation"] for s in history] == [0, 1, 2, 3, 4]


def test_evaluate_gets_the_current_population_every_generation():
    cfg = small_cfg(max_generations=4)
    rng = np.random.default_rng(20)
    state = init_population(4, 2, rng, cfg)
    calls = []

    def evaluate(genomes):
        calls.append(len(genomes))
        return toy_evaluate(genomes)

    run_evolution(state, evaluate, rng, cfg)
    assert calls == [30, 30, 30, 30]


def test_the_first_evaluation_is_the_initial_population():
    cfg = small_cfg(max_generations=1)
    rng = np.random.default_rng(21)
    state = init_population(4, 2, rng, cfg)
    seen = []

    def evaluate(genomes):
        seen.append(copy.deepcopy(genomes))
        return toy_evaluate(genomes)

    run_evolution(state, evaluate, rng, cfg)
    assert seen[0] == state["genomes"]


def test_run_stops_early_at_the_fitness_threshold():
    cfg = small_cfg(max_generations=50, fitness_threshold=3.0)
    rng = np.random.default_rng(22)
    state = init_population(4, 2, rng, cfg)
    counter = {"generation": 0}

    def evaluate(genomes):
        counter["generation"] += 1
        return [float(counter["generation"])] * len(genomes)

    state, history = run_evolution(state, evaluate, rng, cfg)
    assert len(history) == 3
    assert state["best"]["fitness"] == 3.0


def test_threshold_none_never_stops_early():
    cfg = small_cfg(max_generations=6, fitness_threshold=None)
    rng = np.random.default_rng(23)
    state = init_population(4, 2, rng, cfg)
    state, history = run_evolution(state, lambda genomes: [1e9] * len(genomes), rng, cfg)
    assert len(history) == 6


def test_the_callback_gets_every_stats():
    cfg = small_cfg(max_generations=4)
    rng = np.random.default_rng(24)
    state = init_population(4, 2, rng, cfg)
    received = []
    state, history = run_evolution(state, toy_evaluate, rng, cfg, on_generation=received.append)
    assert received == history
    assert len(received) == 4


def test_a_finished_state_is_not_evolved_again():
    cfg = small_cfg(max_generations=3)
    rng = np.random.default_rng(25)
    state = init_population(4, 2, rng, cfg)
    state, _ = run_evolution(state, toy_evaluate, rng, cfg)
    calls = []

    def evaluate(genomes):
        calls.append(1)
        return toy_evaluate(genomes)

    again, history = run_evolution(state, evaluate, rng, cfg)
    assert calls == []
    assert history == []
    assert again == state


def test_a_run_can_be_continued_from_its_state():
    cfg_all = small_cfg(max_generations=6)
    rng = np.random.default_rng(26)
    straight = init_population(4, 2, rng, cfg_all)
    straight, _ = run_evolution(straight, toy_evaluate, rng, cfg_all)

    rng = np.random.default_rng(26)
    part = init_population(4, 2, rng, cfg_all)
    part, _ = run_evolution(part, toy_evaluate, rng, small_cfg(max_generations=3))
    part, _ = run_evolution(part, toy_evaluate, rng, small_cfg(max_generations=6))
    assert part == straight


def test_wrong_evaluate_output_raises():
    cfg = small_cfg(max_generations=2)
    rng = np.random.default_rng(27)
    state = init_population(4, 2, rng, cfg)
    try:
        run_evolution(state, lambda genomes: [1.0] * 3, rng, cfg)
    except ValueError:
        return
    raise AssertionError("evaluate returning the wrong number of values must raise ValueError")


def test_the_default_cfg_runs():
    rng = np.random.default_rng(28)
    cfg = default_config()
    cfg["population_size"] = 20
    state = init_population(3, 1, rng, cfg)
    state, history = run_evolution(state, toy_evaluate, rng, small_cfg(population_size=20, max_generations=2))
    assert state["generation"] == 2

###############################################################################
# a few generations together
###############################################################################

def run_toy(seed, generations):
    cfg = small_cfg(max_generations=generations, population_size=40)
    rng = np.random.default_rng(seed)
    state = init_population(4, 2, rng, cfg)
    return run_evolution(state, toy_evaluate, rng, cfg)


def test_best_ever_never_goes_down():
    for seed in (1, 2, 3):
        state, history = run_toy(seed, 25)
        values = [s["best_ever_fitness"] for s in history]
        for before, after in zip(values, values[1:]):
            assert after >= before


def test_the_toy_problem_improves():
    state, history = run_toy(4, 30)
    assert history[-1]["best_ever_fitness"] > history[0]["best_ever_fitness"]
    assert state["best"]["fitness"] == history[-1]["best_ever_fitness"]


def test_the_population_stays_valid_over_generations():
    state, history = run_toy(5, 25)
    assert len(state["genomes"]) == 40
    innovations = state["innovations"]
    for genome in state["genomes"]:
        assert is_acyclic(genome)
        for key, c in genome["connections"].items():
            assert c["src"] in genome["nodes"] and c["dst"] in genome["nodes"]
            assert genome["nodes"][c["dst"]]["type"] != "input"
            assert innovations["connections"][(c["src"], c["dst"])] == key
    assert max(len(g["nodes"]) for g in state["genomes"]) > 6
    assert max(s["n_species"] for s in history) > 1


def test_the_best_genome_is_a_valid_genome_that_gets_its_fitness():
    state, history = run_toy(6, 20)
    best = state["best"]
    assert abs(toy_fitness(best["genome"]) - best["fitness"]) < 1e-9
    assert is_acyclic(best["genome"])

###############################################################################
# the config is checked at every entry point
###############################################################################

def test_a_bad_config_is_refused_by_every_entry_point():
    rng = np.random.default_rng(40)
    good = small_cfg()
    state = init_population(4, 2, rng, good)
    bad = small_cfg(mate_only_prob=1.5)
    for call in (lambda: init_population(4, 2, rng, bad),
                 lambda: evolve_one_generation(state, [1.0] * 30, rng, bad),
                 lambda: run_evolution(state, toy_evaluate, rng, bad)):
        try:
            call()
        except ValueError as error:
            assert "mate_only_prob" in str(error)
            continue
        raise AssertionError("a bad config must raise ValueError")


def test_a_typo_in_a_config_key_is_refused():
    cfg = small_cfg()
    cfg["populaton_size"] = 10
    try:
        init_population(4, 2, np.random.default_rng(41), cfg)
    except ValueError as error:
        assert "populaton_size" in str(error)
        return
    raise AssertionError("an unknown key must raise ValueError")


###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"evolution: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()