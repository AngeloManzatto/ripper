"""
Created on Thu Oct 22 21:27:40 2026

@author: Angelo Antonio Manzatto

"""

###############################################################################
# libraries
###############################################################################

"""Tests for neat/diagnostics/record.py and table.py (diagnostics step 1).

Run: python tests/neat/test_diagnostics_record.py
The last test needs the real library and checks run_recorded against run_evolution.
"""


"""Tests for neat/diagnostics/record.py and table.py (diagnostics step 1).

Run: python tests/neat/test_diagnostics_record.py
The last test needs the real library and checks run_recorded against run_evolution.
"""


import numpy as np

from neat.diagnostics.record import genome_sizes, species_rows, make_record, run_recorded
from neat.diagnostics.table import generation_table, species_table


def genome(n_hidden, enabled_flags):
    nodes = {0: {"type": "input"}, 1: {"type": "input"}, 2: {"type": "output", "bias": 0.0, "activation": "sigmoid"}}
    for k in range(n_hidden):
        nodes[3 + k] = {"type": "hidden", "bias": 0.0, "activation": "sigmoid"}
    conns = {i: {"src": 0, "dst": 2, "weight": 1.0, "enabled": f} for i, f in enumerate(enabled_flags)}
    return {"n_inputs": 2, "n_outputs": 1, "nodes": nodes, "connections": conns}


GENOMES = [genome(0, [True, True]), genome(2, [True, False, True, True]), genome(1, [False])]
FIT = [1.0, 3.0, 2.0]
SPECIES = {"species": {5: {"representative": 0, "members": [0, 2]}, 7: {"representative": 1, "members": [1]}}, "next_id": 8}
STAG = {5: {"best_fitness": 2.0, "last_improved": 1}, 7: {"best_fitness": 3.0, "last_improved": 4}}
STATE = {"species_set": SPECIES, "stagnation": STAG, "innovations": {"next_innovation": 9, "next_node_id": 6}}
STATS = {"generation": 4, "best_fitness": 3.0, "mean_fitness": 2.0, "best_ever_fitness": 3.0, "n_species": 2, "removed_species": []}


def test_genome_sizes_exact_values():
    s = genome_sizes(GENOMES)
    assert s["mean_nodes"] == (3 + 5 + 4) / 3 and s["max_nodes"] == 5
    assert s["mean_hidden"] == 1.0 and s["max_hidden"] == 2
    assert s["mean_connections"] == (2 + 4 + 1) / 3 and s["max_connections"] == 4
    assert s["mean_enabled"] == (2 + 3 + 0) / 3 and s["max_enabled"] == 3


def test_genome_sizes_empty_fails():
    try:
        genome_sizes([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_species_rows_values():
    rows, n_removed = species_rows(GENOMES, FIT, SPECIES, STAG, 4)
    assert rows == [
        {"id": 5, "size": 2, "best": 2.0, "mean": 1.5, "since_improved": 3},
        {"id": 7, "size": 1, "best": 3.0, "mean": 3.0, "since_improved": 0},
    ]
    assert n_removed == 0


def test_species_rows_missing_stagnation_gives_none():
    rows, _ = species_rows(GENOMES, FIT, SPECIES, {}, 4)
    assert all(r["since_improved"] is None for r in rows)


def test_species_rows_with_a_removed_species_counts_its_genomes():
    kept = {"species": {7: {"representative": 1, "members": [1]}}, "next_id": 8}   # species 5 (genomes 0, 2) was removed
    rows, n_removed = species_rows(GENOMES, FIT, kept, STAG, 4, removed=[5])
    assert [r["id"] for r in rows] == [7] and n_removed == 2


def test_species_rows_uncovered_genomes_without_removal_fail():
    kept = {"species": {7: {"representative": 1, "members": [1]}}, "next_id": 8}
    try:
        species_rows(GENOMES, FIT, kept, STAG, 4, removed=[])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_species_rows_rejects_bad_members_and_lengths():
    cases = [
        (GENOMES, FIT[:2], SPECIES),                                                                    # fitness count
        (GENOMES, FIT, {"species": {1: {"representative": 0, "members": [0, 1, 1, 2]}}, "next_id": 2}),  # duplicate
        (GENOMES, FIT, {"species": {1: {"representative": 0, "members": [0, 1, 5]}}, "next_id": 2}),     # out of range
        (GENOMES, FIT, {"species": {1: {"representative": 0, "members": [-1, 1, 2]}}, "next_id": 2}),    # negative
    ]
    for g, f, sp in cases:
        try:
            species_rows(g, f, sp, STAG, 4, removed=[9])
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_make_record_combines_everything():
    r = make_record(GENOMES, FIT, STATE, STATS)
    assert r["generation"] == 4 and r["best_fitness"] == 3.0
    assert r["mean_hidden"] == 1.0
    assert len(r["species"]) == 2
    assert r["next_innovation"] == 9 and r["next_node_id"] == 6
    assert r["removed_genomes"] == 0


def test_generation_table_has_one_line_per_generation():
    r = make_record(GENOMES, FIT, STATE, STATS)
    text = generation_table([r, r])
    assert len(text.splitlines()) == 4  # header, rule, two generations
    assert "3.0000" in text and "  9" in text


def test_generation_table_enabled_percent():
    r = make_record(GENOMES, FIT, STATE, STATS)
    assert "71.4" in generation_table([r])  # 5 enabled of 7 connections


def test_species_table_names_removed_species():
    r = make_record(GENOMES, FIT, STATE, dict(STATS))
    r["removed_species"] = [3, 11]
    assert "removed for stagnation: 3, 11" in species_table(r)
    r["removed_species"] = []
    assert "removed" not in species_table(r)


def test_species_table_lists_every_species():
    r = make_record(GENOMES, FIT, STATE, STATS)
    text = species_table(r)
    assert "generation 4" in text
    assert len(text.splitlines()) == 5  # title, column names, rule, two species


def _xor_pieces():
    from neat.config import default_config
    from neat.network import compile_genome, activate
    cases = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

    def evaluate(genomes):
        out = []
        for g in genomes:
            net = compile_genome(g)
            err = sum((activate(net, i)[0] - t) ** 2 for i, t in cases)
            out.append(4.0 - err)
        return out

    def make_cfg(max_generations=15, threshold=3.9):
        cfg = default_config()
        cfg["population_size"] = 60
        cfg["max_generations"] = max_generations
        cfg["fitness_threshold"] = threshold
        return cfg
    return evaluate, make_cfg


def _same_stats(history, recorded):
    assert len(history) == len(recorded)
    for a, b in zip(history, recorded):
        for key in a:
            assert a[key] == b[key], key


def test_run_recorded_matches_run_evolution_on_xor():
    """Needs the real library: same seed gives the same stats as run_evolution."""
    from neat.evolution import init_population, run_evolution
    evaluate, make_cfg = _xor_pieces()
    cfg = make_cfg()
    rng = np.random.default_rng(3)
    _, history = run_evolution(init_population(2, 1, rng, cfg), evaluate, rng, cfg)
    cfg = make_cfg()
    rng = np.random.default_rng(3)
    _, recorded = run_recorded(init_population(2, 1, rng, cfg), evaluate, rng, cfg)
    _same_stats(history, recorded)


def test_run_recorded_matches_run_evolution_without_threshold_and_when_resumed():
    """Needs the real library: no early stop, and a run continued from generation 4."""
    from neat.evolution import init_population, run_evolution
    evaluate, make_cfg = _xor_pieces()
    cfg = make_cfg(max_generations=4, threshold=None)
    rng_a = np.random.default_rng(5)
    state_a, _ = run_evolution(init_population(2, 1, rng_a, cfg), evaluate, rng_a, cfg)
    rng_b = np.random.default_rng(5)
    state_b, _ = run_evolution(init_population(2, 1, rng_b, cfg), evaluate, rng_b, cfg)
    cfg = make_cfg(max_generations=9, threshold=None)
    _, history = run_evolution(state_a, evaluate, rng_a, cfg)
    _, recorded = run_recorded(state_b, evaluate, rng_b, cfg)
    assert len(history) == 5          # generations 4..8, not 9 more
    _same_stats(history, recorded)


def test_run_recorded_keeps_genomes_only_when_asked():
    """Needs the real library."""
    from neat.evolution import init_population
    evaluate, make_cfg = _xor_pieces()
    cfg = make_cfg(max_generations=3, threshold=None)
    rng = np.random.default_rng(1)
    _, plain = run_recorded(init_population(2, 1, rng, cfg), evaluate, rng, cfg)
    rng = np.random.default_rng(1)
    _, kept = run_recorded(init_population(2, 1, rng, cfg), evaluate, rng, cfg, keep_genomes=True)
    assert all("genomes" not in r for r in plain)
    assert all(len(r["genomes"]) == 60 for r in kept)


def test_run_recorded_survives_species_removal():
    """Needs the real library: a tiny stagnation limit forces species to be removed."""
    from neat.evolution import init_population
    evaluate, make_cfg = _xor_pieces()
    cfg = make_cfg(max_generations=40, threshold=None)
    cfg["compatibility_threshold"] = 1.0
    for key in ("max_stagnation", "stagnation_limit"):
        if key in cfg:
            cfg[key] = 2
    rng = np.random.default_rng(2)
    _, history = run_recorded(init_population(2, 1, rng, cfg), evaluate, rng, cfg)
    removed = [r for r in history if r["removed_species"]]
    assert removed, "no species was removed: lower the stagnation limit in this test"
    assert all(r["removed_genomes"] > 0 for r in removed)


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"diagnostics record/table: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()