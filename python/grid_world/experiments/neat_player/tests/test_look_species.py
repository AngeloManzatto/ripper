"""
Tests for experiments/neat_player/look_species.py: species_by_threshold and species_text.

species_by_threshold(genomes, cfg, thresholds, speciate_fn, new_species_set)
    - speciates from scratch once per threshold, in the order given, with cfg["compatibility_threshold"]
      set to that threshold on a COPY of cfg;
    - returns {"threshold", "n_species", "sizes"} per threshold, sizes biggest first;
    - empty genomes or empty thresholds raise; genomes and cfg are not changed.

species_text(rows, show) : one line per threshold; sizes cut at `show` with "(+k more)".

The real speciation is replaced by a small fake that groups numbers on a line: a number joins
the first species whose first member is closer than the threshold.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

from experiments.neat_player.look_species import species_by_threshold, species_text

###############################################################################
# helpers
###############################################################################

def fake_speciate(genomes, species_set, cfg):
    """Groups genomes (dicts with "x") on a line. Records the thresholds it was called with."""
    fake_speciate.thresholds.append(cfg["compatibility_threshold"])
    species = []
    for index, genome in enumerate(genomes):
        for members in species:
            if abs(genomes[members[0]]["x"] - genome["x"]) < cfg["compatibility_threshold"]:
                members.append(index)
                break
        else:
            species.append([index])
    return {"species": {k: {"members": m} for k, m in enumerate(species)}, "next_id": len(species)}


fake_speciate.thresholds = []


def fake_new():
    return {"species": {}, "next_id": 0}


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


GENOMES = [{"x": 0.0}, {"x": 0.1}, {"x": 0.2}, {"x": 1.0}, {"x": 1.05}, {"x": 5.0}]

###############################################################################
# tests
###############################################################################

def test_counts_and_sizes_for_each_threshold():
    rows = species_by_threshold(GENOMES, {"compatibility_threshold": 3.0}, [0.5, 2.0, 0.01], fake_speciate, fake_new)
    assert rows == [
        {"threshold": 0.5, "n_species": 3, "sizes": [3, 2, 1]},
        {"threshold": 2.0, "n_species": 2, "sizes": [5, 1]},
        {"threshold": 0.01, "n_species": 6, "sizes": [1, 1, 1, 1, 1, 1]},
    ]


def test_sizes_are_biggest_first():
    genomes = [{"x": 0.0}, {"x": 10.0}, {"x": 10.1}, {"x": 10.2}]
    rows = species_by_threshold(genomes, {"compatibility_threshold": 3.0}, [0.5], fake_speciate, fake_new)
    assert rows[0]["sizes"] == [3, 1]


def test_each_threshold_is_used_in_the_given_order():
    fake_speciate.thresholds.clear()
    species_by_threshold(GENOMES, {"compatibility_threshold": 3.0}, [0.5, 2.0, 0.01], fake_speciate, fake_new)
    assert fake_speciate.thresholds == [0.5, 2.0, 0.01]


def test_each_call_starts_from_a_fresh_species_set():
    seen = []

    def spy(genomes, species_set, cfg):
        seen.append(species_set)
        species_set["species"]["dirty"] = {"members": [0]}       # a spy that dirties what it is given
        return {"species": {0: {"members": list(range(len(genomes)))}}, "next_id": 1}

    species_by_threshold(GENOMES, {"compatibility_threshold": 3.0}, [0.5, 1.0], spy, fake_new)
    assert seen[0] is not seen[1]
    assert seen[1] == {"species": {"dirty": {"members": [0]}}, "next_id": 0}   # only its own dirt, nothing from call 1


def test_genomes_and_cfg_are_not_changed():
    genomes = copy.deepcopy(GENOMES)
    cfg = {"compatibility_threshold": 3.0, "other": [1, 2]}
    before = copy.deepcopy(cfg)
    species_by_threshold(genomes, cfg, [0.5, 2.0], fake_speciate, fake_new)
    assert genomes == GENOMES and cfg == before


def test_empty_inputs_raise():
    assert raises(ValueError, species_by_threshold, [], {"compatibility_threshold": 3.0}, [0.5], fake_speciate, fake_new)
    assert raises(ValueError, species_by_threshold, GENOMES, {"compatibility_threshold": 3.0}, [], fake_speciate, fake_new)


def test_text_lines():
    rows = [{"threshold": 0.5, "n_species": 3, "sizes": [3, 2, 1]}, {"threshold": 3.0, "n_species": 1, "sizes": [6]}]
    lines = species_text(rows, show=10).split("\n")
    assert lines[0].split()[:3] == ["threshold", "species", "sizes"]
    assert lines[2].split() == ["0.5", "3", "3", "2", "1"]
    assert lines[3].split() == ["3", "1", "6"]


def test_text_cuts_long_size_lists():
    rows = [{"threshold": 0.1, "n_species": 5, "sizes": [9, 8, 7, 6, 5]}]
    line = species_text(rows, show=3).split("\n")[2]
    assert line.split()[2:] == ["9", "8", "7", "(+2", "more)"]
    assert "more" not in species_text(rows, show=5).split("\n")[2]

###############################################################################
# run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"look_species: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()