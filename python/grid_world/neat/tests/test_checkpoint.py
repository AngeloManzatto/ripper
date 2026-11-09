"""
Created on Tue Oct 20 08:24:35 2026

@author: Angelo Antonio Manzatto
"""

"""
Tests for neat/checkpoint.py: save_checkpoint, load_checkpoint, save_genome, load_genome.

CHECKPOINT (a whole run, pickle)
    save_checkpoint(path, state, rng, cfg)  ->  None
    load_checkpoint(path)                   ->  (state, rng, cfg)
    The loaded state and cfg are equal to what was saved (tuple keys of the
    innovation registry and the tuple of activation_options included) and share
    nothing with it. The loaded rng continues exactly where the saved one was,
    so a run that is stopped, saved, loaded and continued gives the SAME result
    as a run that never stopped.
    save refuses an invalid cfg (ValueError). load raises ValueError for a file
    that is not a checkpoint or has another version.

GENOME (one genome, JSON)
    save_genome(path, genome)  ->  None
    load_genome(path)          ->  genome equal to the one saved (integer keys
                                   restored, floats exactly the same)
    The file is plain JSON (keys are text). NaN or infinite numbers are refused
    when saving (ValueError). load raises ValueError for invalid JSON, a file
    that is not a genome, another version, or missing parts.

Both: the target is replaced atomically. If saving fails half way, the previous
file is still intact and no temporary file is left. str and Path both work.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from neat.config import default_config
from neat.evolution import init_population, run_evolution, evolve_one_generation
from neat.genome import init_genome
from neat.innovations import init_innovations
from neat.mutation import mutate
from neat.network import compile_genome, activate
from neat.checkpoint import save_checkpoint, load_checkpoint, save_genome, load_genome

###############################################################################
# helpers
###############################################################################

def small_cfg(**overrides):
    cfg = default_config()
    cfg["population_size"] = 30
    cfg["max_generations"] = 6
    cfg["fitness_threshold"] = None
    cfg["compatibility_threshold"] = 1.5
    cfg["node_add_prob"] = 0.15
    cfg["conn_add_prob"] = 0.25
    cfg.update(overrides)
    return cfg


def toy_evaluate(genomes):
    result = []
    for genome in genomes:
        total = 0.0
        for c in genome["connections"].values():
            if c["enabled"]:
                total += c["weight"]
        for node in genome["nodes"].values():
            if node["type"] == "hidden":
                total += 0.1
        result.append(total)
    return result


def evolved_run(seed, generations, cfg=None):
    cfg = cfg or small_cfg(max_generations=generations)
    rng = np.random.default_rng(seed)
    state = init_population(4, 2, rng, cfg)
    state, history = run_evolution(state, toy_evaluate, rng, cfg)
    return state, rng, cfg


def evolved_genome(seed, steps=40, n_in=5, n_out=3):
    cfg = default_config()
    cfg["node_add_prob"] = 0.5
    cfg["conn_add_prob"] = 0.5
    cfg["activation_mutate_prob"] = 0.5
    rng = np.random.default_rng(seed)
    genome = init_genome(n_in, n_out, rng, cfg)
    innovations = init_innovations(n_in, n_out)
    for _ in range(steps):
        genome = mutate(genome, rng, innovations, cfg)
    return genome


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


class Folder:
    """A temporary folder for one test."""
    def __enter__(self):
        self.handle = tempfile.TemporaryDirectory()
        return Path(self.handle.name)

    def __exit__(self, *args):
        self.handle.cleanup()

###############################################################################
# checkpoint: round trip
###############################################################################

def test_checkpoint_round_trip_state_and_cfg():
    state, rng, cfg = evolved_run(1, 5)
    with Folder() as folder:
        path = folder / "run.ckpt"
        assert save_checkpoint(path, state, rng, cfg) is None
        loaded_state, loaded_rng, loaded_cfg = load_checkpoint(path)
    assert loaded_state == state
    assert loaded_cfg == cfg
    assert type(loaded_cfg["activation_options"]) is tuple


def test_checkpoint_keeps_the_tuple_keys_of_the_registry():
    state, rng, cfg = evolved_run(2, 6)
    pairs = list(state["innovations"]["connections"])
    assert len(pairs) > 8
    with Folder() as folder:
        save_checkpoint(folder / "run.ckpt", state, rng, cfg)
        loaded_state, _, _ = load_checkpoint(folder / "run.ckpt")
    assert list(loaded_state["innovations"]["connections"]) == pairs
    assert all(type(pair) is tuple for pair in loaded_state["innovations"]["connections"])
    assert loaded_state["innovations"]["splits"] == state["innovations"]["splits"]


def test_checkpoint_of_a_fresh_population():
    cfg = small_cfg()
    rng = np.random.default_rng(3)
    state = init_population(4, 2, rng, cfg)
    with Folder() as folder:
        save_checkpoint(folder / "a.ckpt", state, rng, cfg)
        loaded_state, _, _ = load_checkpoint(folder / "a.ckpt")
    assert loaded_state == state
    assert loaded_state["best"] is None


def test_loaded_objects_share_nothing_with_the_originals():
    state, rng, cfg = evolved_run(4, 4)
    with Folder() as folder:
        save_checkpoint(folder / "run.ckpt", state, rng, cfg)
        loaded_state, loaded_rng, loaded_cfg = load_checkpoint(folder / "run.ckpt")
    assert loaded_state is not state
    assert loaded_state["genomes"][0] is not state["genomes"][0]
    assert loaded_cfg is not cfg
    assert loaded_rng is not rng


def test_saving_does_not_change_what_is_saved():
    state, rng, cfg = evolved_run(5, 4)
    snapshot = copy.deepcopy(state)
    rng_before = copy.deepcopy(rng.bit_generator.state)
    with Folder() as folder:
        save_checkpoint(folder / "run.ckpt", state, rng, cfg)
    assert state == snapshot
    assert rng.bit_generator.state == rng_before

###############################################################################
# checkpoint: exact resume
###############################################################################

def test_the_loaded_rng_continues_where_the_saved_one_was():
    state, rng, cfg = evolved_run(6, 3)
    with Folder() as folder:
        save_checkpoint(folder / "run.ckpt", state, rng, cfg)
        _, loaded_rng, _ = load_checkpoint(folder / "run.ckpt")
    assert list(rng.random(5)) == list(loaded_rng.random(5))
    assert list(rng.integers(0, 1000, 5)) == list(loaded_rng.integers(0, 1000, 5))


def test_stop_save_load_continue_equals_never_stopping():
    cfg_all = small_cfg(max_generations=8)
    rng = np.random.default_rng(7)
    straight = init_population(4, 2, rng, cfg_all)
    straight, history_straight = run_evolution(straight, toy_evaluate, rng, cfg_all)

    rng = np.random.default_rng(7)
    part = init_population(4, 2, rng, cfg_all)
    part, history_a = run_evolution(part, toy_evaluate, rng, small_cfg(max_generations=4))
    with Folder() as folder:
        save_checkpoint(folder / "half.ckpt", part, rng, cfg_all)
        part, rng, cfg_loaded = load_checkpoint(folder / "half.ckpt")
    part, history_b = run_evolution(part, toy_evaluate, rng, cfg_loaded)

    assert part == straight
    assert history_a + history_b == history_straight


def test_resuming_several_times_is_still_exact():
    cfg = small_cfg(max_generations=9)
    rng = np.random.default_rng(8)
    straight = init_population(4, 2, rng, cfg)
    straight, _ = run_evolution(straight, toy_evaluate, rng, cfg)

    rng = np.random.default_rng(8)
    state = init_population(4, 2, rng, cfg)
    with Folder() as folder:
        path = folder / "run.ckpt"
        for _ in range(9):
            fitnesses = toy_evaluate(state["genomes"])
            state, _ = evolve_one_generation(state, fitnesses, rng, cfg)
            save_checkpoint(path, state, rng, cfg)
            state, rng, cfg = load_checkpoint(path)
    assert state == straight

###############################################################################
# checkpoint: refusals
###############################################################################

def test_save_refuses_an_invalid_config_and_writes_nothing():
    state, rng, cfg = evolved_run(9, 2)
    bad = dict(cfg)
    bad["mate_only_prob"] = 2.0
    with Folder() as folder:
        path = folder / "run.ckpt"
        assert raises(ValueError, save_checkpoint, path, state, rng, bad)
        assert list(folder.iterdir()) == []


def test_load_refuses_files_that_are_not_checkpoints():
    with Folder() as folder:
        other = folder / "other.pkl"
        with open(other, "wb") as file:
            pickle.dump({"hello": 1}, file)
        assert raises(ValueError, load_checkpoint, other)
        listing = folder / "list.pkl"
        with open(listing, "wb") as file:
            pickle.dump([1, 2, 3], file)
        assert raises(ValueError, load_checkpoint, listing)


def test_load_refuses_a_wrong_format_name_even_with_the_right_version():
    state, rng, cfg = evolved_run(10, 2)
    with Folder() as folder:
        path = folder / "run.ckpt"
        save_checkpoint(path, state, rng, cfg)
        with open(path, "rb") as file:
            contents = pickle.load(file)
        contents["format"] = "something-else"
        with open(path, "wb") as file:
            pickle.dump(contents, file)
        assert raises(ValueError, load_checkpoint, path)


def test_load_refuses_another_version():
    state, rng, cfg = evolved_run(10, 2)
    with Folder() as folder:
        path = folder / "run.ckpt"
        save_checkpoint(path, state, rng, cfg)
        with open(path, "rb") as file:
            contents = pickle.load(file)
        contents["version"] = 99
        with open(path, "wb") as file:
            pickle.dump(contents, file)
        assert raises(ValueError, load_checkpoint, path)


def test_load_of_a_missing_file_raises_file_not_found():
    with Folder() as folder:
        assert raises(FileNotFoundError, load_checkpoint, folder / "nope.ckpt")

###############################################################################
# atomic writes
###############################################################################

def test_a_failed_checkpoint_save_keeps_the_previous_file():
    state, rng, cfg = evolved_run(11, 3)
    with Folder() as folder:
        path = folder / "run.ckpt"
        save_checkpoint(path, state, rng, cfg)
        before = path.read_bytes()
        broken = dict(state)
        broken["oops"] = lambda x: x          # cannot be pickled
        assert raises(Exception, save_checkpoint, path, broken, rng, cfg)
        assert path.read_bytes() == before
        assert [p.name for p in folder.iterdir()] == ["run.ckpt"]


def test_a_save_that_fails_at_the_last_step_leaves_no_temporary_file():
    state, rng, cfg = evolved_run(16, 2)
    genome = state["genomes"][0]
    with Folder() as folder:
        target = folder / "sub"
        target.mkdir()                    # replacing a folder with a file fails
        assert raises(OSError, save_checkpoint, target, state, rng, cfg)
        assert raises(OSError, save_genome, target, genome)
        assert [p.name for p in folder.iterdir()] == ["sub"]


def test_saving_over_an_existing_checkpoint_replaces_it():
    first, rng, cfg = evolved_run(12, 2)
    second, rng2, cfg2 = evolved_run(13, 5)
    with Folder() as folder:
        path = folder / "run.ckpt"
        save_checkpoint(path, first, rng, cfg)
        save_checkpoint(path, second, rng2, cfg2)
        loaded, _, _ = load_checkpoint(path)
        assert loaded == second
        assert [p.name for p in folder.iterdir()] == ["run.ckpt"]


def test_str_and_path_both_work():
    state, rng, cfg = evolved_run(14, 2)
    with Folder() as folder:
        as_text = str(folder / "text.ckpt")
        save_checkpoint(as_text, state, rng, cfg)
        assert load_checkpoint(as_text)[0] == state
        genome = state["genomes"][0]
        save_genome(str(folder / "g.json"), genome)
        assert load_genome(folder / "g.json") == genome

###############################################################################
# genome: round trip
###############################################################################

def test_genome_round_trip_with_hidden_nodes():
    for seed in range(10):
        genome = evolved_genome(seed)
        with Folder() as folder:
            save_genome(folder / "g.json", genome)
            loaded = load_genome(folder / "g.json")
        assert loaded == genome, seed
        assert all(type(k) is int for k in loaded["nodes"])
        assert all(type(k) is int for k in loaded["connections"])


def test_genome_round_trip_keeps_floats_exactly():
    genome = evolved_genome(1, steps=0, n_in=2, n_out=1)
    awkward = [0.1 + 0.2, 1e-300, -1e300, 123456789.123456789, 5e-324, -0.0, 1.0 / 3.0]
    for value, key in zip(awkward, genome["connections"]):
        genome["connections"][key]["weight"] = value
    with Folder() as folder:
        save_genome(folder / "g.json", genome)
        loaded = load_genome(folder / "g.json")
    for value, key in zip(awkward, genome["connections"]):
        assert loaded["connections"][key]["weight"] == value


def test_genome_round_trip_keeps_flags_and_types():
    genome = evolved_genome(2)
    with Folder() as folder:
        save_genome(folder / "g.json", genome)
        loaded = load_genome(folder / "g.json")
    for key, c in genome["connections"].items():
        assert loaded["connections"][key]["enabled"] is c["enabled"]
        assert type(loaded["connections"][key]["enabled"]) is bool
    assert any(not c["enabled"] for c in genome["connections"].values())
    for node_id, node in genome["nodes"].items():
        assert loaded["nodes"][node_id] == node


def test_a_loaded_genome_behaves_the_same():
    rng = np.random.default_rng(0)
    for seed in range(8):
        genome = evolved_genome(seed)
        with Folder() as folder:
            save_genome(folder / "g.json", genome)
            loaded = load_genome(folder / "g.json")
        original, copy_ = compile_genome(genome), compile_genome(loaded)
        for _ in range(3):
            x = list(rng.normal(0.0, 2.0, 5))
            assert activate(original, x) == activate(copy_, x)


def test_saving_a_genome_does_not_change_it():
    genome = evolved_genome(3)
    snapshot = copy.deepcopy(genome)
    with Folder() as folder:
        save_genome(folder / "g.json", genome)
    assert genome == snapshot
    assert all(type(k) is int for k in genome["nodes"])


def test_the_genome_file_is_plain_readable_json():
    genome = evolved_genome(4)
    with Folder() as folder:
        save_genome(folder / "g.json", genome)
        text = (folder / "g.json").read_text(encoding="utf-8")
    contents = json.loads(text)
    assert contents["format"] == "neat-genome"
    assert contents["version"] == 1
    assert set(contents["genome"]) == {"n_inputs", "n_outputs", "nodes", "connections"}
    assert all(type(k) is str for k in contents["genome"]["nodes"])
    assert "\n" in text and "weight" in text

###############################################################################
# genome: refusals
###############################################################################

def test_saving_nan_or_infinite_numbers_is_refused_and_writes_nothing():
    for bad in (float("nan"), float("inf"), float("-inf")):
        genome = evolved_genome(5)
        genome["connections"][0]["weight"] = bad
        with Folder() as folder:
            assert raises(ValueError, save_genome, folder / "g.json", genome)
            assert list(folder.iterdir()) == []


def test_load_refuses_invalid_json():
    with Folder() as folder:
        path = folder / "g.json"
        path.write_text("{ this is not json", encoding="utf-8")
        assert raises(ValueError, load_genome, path)


def test_load_refuses_files_that_are_not_genomes():
    with Folder() as folder:
        path = folder / "g.json"
        for text in ('{"hello": 1}', "[1, 2]", '{"format": "other", "version": 1, "genome": {}}'):
            path.write_text(text, encoding="utf-8")
            assert raises(ValueError, load_genome, path), text


def test_load_refuses_a_wrong_format_name_even_with_everything_else_right():
    genome = evolved_genome(8)
    with Folder() as folder:
        path = folder / "g.json"
        save_genome(path, genome)
        contents = json.loads(path.read_text(encoding="utf-8"))
        contents["format"] = "something-else"
        path.write_text(json.dumps(contents), encoding="utf-8")
        assert raises(ValueError, load_genome, path)


def test_load_refuses_another_version_and_missing_parts():
    genome = evolved_genome(6)
    with Folder() as folder:
        path = folder / "g.json"
        save_genome(path, genome)
        contents = json.loads(path.read_text(encoding="utf-8"))
        wrong = dict(contents, version=7)
        path.write_text(json.dumps(wrong), encoding="utf-8")
        assert raises(ValueError, load_genome, path)
        for part in ("n_inputs", "n_outputs", "nodes", "connections"):
            broken = copy.deepcopy(contents)
            del broken["genome"][part]
            path.write_text(json.dumps(broken), encoding="utf-8")
            assert raises(ValueError, load_genome, path), part


def test_a_failed_genome_save_keeps_the_previous_file():
    genome = evolved_genome(7)
    with Folder() as folder:
        path = folder / "g.json"
        save_genome(path, genome)
        before = path.read_bytes()
        genome["connections"][0]["weight"] = float("nan")
        assert raises(ValueError, save_genome, path, genome)
        assert path.read_bytes() == before
        assert [p.name for p in folder.iterdir()] == ["g.json"]


def test_a_best_genome_from_a_run_can_be_exported():
    state, rng, cfg = evolved_run(15, 6)
    best = state["best"]["genome"]
    with Folder() as folder:
        save_genome(folder / "best.json", best)
        assert load_genome(folder / "best.json") == best

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"checkpoint: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()