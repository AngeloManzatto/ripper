"""
Created on Fri Oct 23 08:40:13 2026

@author: Angelo Antonio Manzatto

Tests for the checkpoint options of run_evolution.

Run: python tests/neat/test_evolution_checkpointing.py
Uses the real library with a small population. Fitness is deterministic and varied, so
runs are comparable; the point is the saving, not the problem being solved.

"""

###############################################################################
# libraries
###############################################################################

import os
import tempfile

import numpy as np

import neat.evolution as evolution
from neat.config import default_config
from neat.checkpoint import load_checkpoint
from neat.evolution import init_population, run_evolution


def evaluate(genomes):
    return [sum(1 for n in g["nodes"].values() if n["type"] == "hidden") + 0.001 * len(g["connections"])
            for g in genomes]


def make_cfg(max_generations=8, threshold=None):
    cfg = default_config()
    cfg["population_size"] = 20
    cfg["max_generations"] = max_generations
    cfg["fitness_threshold"] = threshold
    return cfg


def fresh(seed=7, max_generations=8, threshold=None):
    cfg = make_cfg(max_generations, threshold)
    rng = np.random.default_rng(seed)
    return init_population(2, 1, rng, cfg), rng, cfg


class Spy:
    """Replaces save_checkpoint in neat.evolution and remembers the generation of every save."""
    def __init__(self):
        self.saved = []
        self.real = evolution.save_checkpoint

    def __enter__(self):
        evolution.save_checkpoint = lambda path, state, rng, cfg: self.saved.append(state["generation"])
        return self

    def __exit__(self, *exc):
        evolution.save_checkpoint = self.real


def test_off_by_default():
    state, rng, cfg = fresh()
    with Spy() as spy:
        run_evolution(state, evaluate, rng, cfg)
    assert spy.saved == []


def test_saves_every_n_generations_and_at_the_end():
    state, rng, cfg = fresh()
    with Spy() as spy:
        run_evolution(state, evaluate, rng, cfg, checkpoint_path="x", checkpoint_every=3)
    assert spy.saved == [3, 6, 8]


def test_no_duplicate_save_when_the_last_generation_is_a_multiple():
    state, rng, cfg = fresh()
    with Spy() as spy:
        run_evolution(state, evaluate, rng, cfg, checkpoint_path="x", checkpoint_every=4)
    assert spy.saved == [4, 8]


def test_early_stop_still_saves_the_final_state():
    state, rng, cfg = fresh(threshold=-1.0)       # reached after the first generation
    with Spy() as spy:
        _, history = run_evolution(state, evaluate, rng, cfg, checkpoint_path="x", checkpoint_every=5)
    assert len(history) == 1 and spy.saved == [1]


def test_options_must_come_together_and_be_sane():
    state, rng, cfg = fresh()
    bad = [dict(checkpoint_path="x"), dict(checkpoint_every=2),
           dict(checkpoint_path="x", checkpoint_every=0), dict(checkpoint_path="x", checkpoint_every=-1),
           dict(checkpoint_path="x", checkpoint_every=2.5), dict(checkpoint_path="x", checkpoint_every=True)]
    for kwargs in bad:
        try:
            run_evolution(state, evaluate, rng, cfg, **kwargs)
        except ValueError:
            continue
        raise AssertionError(kwargs)


def test_on_generation_is_still_called():
    state, rng, cfg = fresh()
    seen = []
    with Spy():
        run_evolution(state, evaluate, rng, cfg, on_generation=seen.append, checkpoint_path="x", checkpoint_every=2)
    assert [s["generation"] for s in seen] == list(range(8))


def test_resuming_from_the_file_gives_exactly_the_same_run():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "run.ckpt")
        # straight run, 8 generations
        state, rng, cfg = fresh()
        straight, straight_history = run_evolution(state, evaluate, rng, cfg)
        straight_draw = rng.random()
        # run 4 generations with checkpointing, then resume from the file for the rest
        state, rng, cfg = fresh(max_generations=4)
        _, first = run_evolution(state, evaluate, rng, cfg, checkpoint_path=path, checkpoint_every=2)
        state, rng, cfg = load_checkpoint(path)
        assert state["generation"] == 4
        cfg["max_generations"] = 8
        resumed, second = run_evolution(state, evaluate, rng, cfg)
        assert resumed["generation"] == straight["generation"] == 8
        assert resumed["genomes"] == straight["genomes"]
        assert resumed["best"] == straight["best"]
        assert first + second == straight_history
        assert rng.random() == straight_draw


def test_the_file_holds_the_final_state():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "run.ckpt")
        state, rng, cfg = fresh()
        final, _ = run_evolution(state, evaluate, rng, cfg, checkpoint_path=path, checkpoint_every=3)
        loaded, _, _ = load_checkpoint(path)
        assert loaded["generation"] == 8 and loaded["genomes"] == final["genomes"]
        assert sorted(os.listdir(folder)) == ["run.ckpt"]       # no temporary files left


def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"evolution checkpointing: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()