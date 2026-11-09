"""
Created on Mon Nov  2 08:18:46 2026

@author: Angelo Antonio Manzatto
"""

"""Tests of experiments/poet/environment.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.

The first group uses stand-ins for the engine, so it checks how the functions talk to the engine.
The second group runs the real engine and is skipped when it cannot be imported.
"""

###############################################################################
# Libraries
###############################################################################

import sys
import traceback
import types
import unittest
from contextlib import contextmanager

from poet import environment
from poet.environment import make_world, mutate_layout

###############################################################################
# Small helpers
###############################################################################

@contextmanager
def patched(module, **replacements):
    """Replace attributes of `module` and ALWAYS put the originals back, even when a test fails,
    so a crash cannot leave stand-ins behind in a console that stays open."""
    originals = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


@contextmanager
def raises(exception, match=None):
    """The block must raise `exception` (with `match` inside the message when given)."""
    try:
        yield
    except exception as error:
        if match is not None and match not in str(error):
            raise AssertionError(f"{exception.__name__} message {str(error)!r} does not contain {match!r}")
    else:
        raise AssertionError(f"{exception.__name__} was not raised")


###############################################################################
# Stand-ins
###############################################################################

class FakeWorld:
    instances = []

    def __init__(self, layout, config, seed):
        self.args = (layout, config, seed)
        self.resets = []
        FakeWorld.instances.append(self)

    def reset(self, *args, **kwargs):
        self.resets.append((args, kwargs))
        observation = types.SimpleNamespace(player_id=7, enemy_ids=[3, 4])
        return types.SimpleNamespace(observation=observation)


class FakeMutationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@contextmanager
def fake_engine():
    """Stand-ins for World, MutationConfig and mutate_valid_layout; yields the list of mutate calls."""
    FakeWorld.instances.clear()
    calls = []

    def fake_mutate(layout, config):
        calls.append((layout, config))
        return f"{layout}+mutated(seed={config.kwargs['seed']})"

    with patched(environment, World=FakeWorld, MutationConfig=FakeMutationConfig,
                 mutate_valid_layout=fake_mutate):
        yield calls


###############################################################################
# make_world
###############################################################################

def test_make_world_builds_the_world_from_layout_and_seed():
    with fake_engine():
        world, player_id, enemy_ids = make_world("LAYOUT", seed=5)
    assert world is FakeWorld.instances[0]
    assert world.args == ("LAYOUT", None, 5)
    assert (player_id, enemy_ids) == (7, [3, 4])


def test_make_world_resets_once_and_without_repositioning():
    with fake_engine():
        world, _, _ = make_world("LAYOUT", seed=5)
    assert world.resets == [((), {})]


def test_every_call_builds_a_new_world():
    with fake_engine():
        a, _, _ = make_world("LAYOUT", seed=1)
        b, _, _ = make_world("LAYOUT", seed=1)
    assert a is not b and len(FakeWorld.instances) == 2


###############################################################################
# mutate_layout
###############################################################################

def test_mutate_layout_gives_the_settings_and_the_seed_to_the_engine():
    settings = {"n_wall_mutations": 12, "max_wall_density": 0.4}
    with fake_engine() as calls:
        result = mutate_layout("LAYOUT", settings, seed=9)
    layout, config = calls[0]
    assert layout == "LAYOUT"
    assert config.kwargs == {"n_wall_mutations": 12, "max_wall_density": 0.4, "seed": 9}
    assert result == "LAYOUT+mutated(seed=9)"


def test_mutate_layout_does_not_modify_the_settings():
    settings = {"n_wall_mutations": 1}
    with fake_engine():
        mutate_layout("LAYOUT", settings, seed=3)
    assert settings == {"n_wall_mutations": 1}


def test_a_seed_inside_the_settings_is_an_error():
    with fake_engine():
        with raises(TypeError):
            mutate_layout("LAYOUT", {"n_wall_mutations": 1, "seed": 4}, seed=3)


def test_no_valid_mutation_raises():
    with patched(environment, MutationConfig=FakeMutationConfig,
                 mutate_valid_layout=lambda layout, config: None):
        with raises(RuntimeError, match="no valid mutation"):
            mutate_layout("LAYOUT", {}, seed=1)


###############################################################################
# The real engine (not run in my sandbox: it needs your engine)
###############################################################################

GENERATION = {"width": 16, "height": 16, "wall_density": 0.1, "num_enemies": 1,
              "num_traps": 0, "min_player_goal_distance": 10}

MUTATION = {"n_wall_mutations": 1, "max_wall_density": 0.3, "max_enemies": 1,
            "enemy_add_weight": 0.0, "enemy_remove_weight": 0.0, "enemy_nothing_weight": 1.0,
            "max_traps": 3, "trap_add_weight": 0.0, "trap_remove_weight": 0.0, "trap_nothing_weight": 1.0,
            "n_repositions": 1, "min_player_goal_distance": 10}


def real_layout():
    """A layout from the real engine; skips the test when the engine is not there."""
    try:
        from engine import GenerationConfig, generate_valid_layout
    except ImportError:
        raise unittest.SkipTest("the real engine is not available")
    return generate_valid_layout(GenerationConfig(**GENERATION))


def picture(layout):
    return make_world(layout, seed=0)[0].render_world()


def test_real_world_has_one_player_and_one_enemy():
    _, player_id, enemy_ids = make_world(real_layout(), seed=1)
    assert len(enemy_ids) == 1 and player_id not in enemy_ids


def test_real_mutation_is_reproducible():
    layout = real_layout()
    assert picture(mutate_layout(layout, MUTATION, seed=1)) == picture(mutate_layout(layout, MUTATION, seed=1))


def test_real_mutation_changes_the_layout_and_the_seed_matters():
    layout = real_layout()
    pictures = {picture(mutate_layout(layout, MUTATION, seed=s)) for s in range(10)}
    assert picture(layout) not in pictures
    assert len(pictures) > 1


###############################################################################
# Runner
###############################################################################

def run_all():
    tests = [(name, test) for name, test in globals().items() if name.startswith("test_") and callable(test)]
    passed, skipped, failed = 0, 0, []

    for name, test in tests:
        try:
            test()
        except unittest.SkipTest as reason:
            skipped += 1
            print(f"skip  {name}  ({reason})")
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok    {name}")

    print(f"\n{passed} passed, {skipped} skipped, {len(failed)} failed" + (f": {failed}" if failed else ""))

    try:
        get_ipython()                    # inside Spyder / IPython: just report
    except NameError:
        if failed:
            sys.exit(1)                  # from a plain terminal: a failing exit code


if __name__ == "__main__":
    run_all()