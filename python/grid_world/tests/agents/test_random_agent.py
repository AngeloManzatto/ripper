"""
Created on Fri Oct  9 10:02:11 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from types import SimpleNamespace

from agents.random_agent.agent import RandomAgent
from core.actions import ACTIONS
from engine import GenerationConfig, World, generate_valid_layout
from runtime.runner import run_episode

OBS = SimpleNamespace()     # RandomAgent ignores the observation

###############################################################################
# Group A: the agent alone (no engine involved)
###############################################################################


def test_returns_valid_actions():
    agent = RandomAgent(seed=0)
    assert all(agent.act(OBS) in ACTIONS for _ in range(200))


def test_roughly_uniform():
    agent = RandomAgent(seed=0)
    counts = {i: 0 for i in range(len(ACTIONS))}
    for _ in range(4000):
        counts[ACTIONS.index(agent.act(OBS))] += 1
    # expected 1000 each, standard deviation ~27: this band is about +-3.5 sigma
    assert all(900 <= n <= 1100 for n in counts.values()), counts


def test_seed_determines_sequence():
    def sequence(seed):
        agent = RandomAgent(seed=seed)
        return [agent.act(OBS) for _ in range(50)]

    assert sequence(1) == sequence(1)
    assert sequence(1) != sequence(2)


def test_reset_does_not_reseed():
    agent = RandomAgent(seed=0)
    first = [agent.act(OBS) for _ in range(20)]
    agent.reset()
    second = [agent.act(OBS) for _ in range(20)]
    assert first != second      # identical by chance: 4 ** -20


def test_history_is_bounded_and_cleared_on_reset():
    agent = RandomAgent(seed=0, history_size=8)
    for _ in range(20):
        agent.act(OBS)
    assert len(agent.history) == 8
    agent.reset()
    assert len(agent.history) == 0


###############################################################################
# Group B: through the runner, with on_step as a recorder
###############################################################################


def _record_episode(world_seed, agent_seed):
    layout = generate_valid_layout(GenerationConfig(width=16, height=16, seed=world_seed))
    assert layout is not None, f"no valid layout for seed {world_seed}"
    world = World(layout, seed=world_seed)
    player_id = world.reset().observation.player_id

    records = []

    def on_step(world, step, actions):
        records.append((step, dict(actions)))

    result = run_episode(world, {player_id: RandomAgent(seed=agent_seed)}, on_step=on_step)
    return player_id, records, result


def test_recorded_steps_are_consistent():
    player_id, records, result = _record_episode(world_seed=0, agent_seed=3)

    assert records[0] == (0, {})                                  # the reset call
    assert len(records) == result.steps + 1
    assert [step for step, _ in records] == list(range(result.steps + 1))

    for step, actions in records[1:]:
        assert set(actions) == {player_id}, (step, actions)       # only the player has an agent
        assert actions[player_id] in ACTIONS


def test_same_seeds_reproduce_the_episode():
    _, records_a, result_a = _record_episode(world_seed=5, agent_seed=3)
    _, records_b, result_b = _record_episode(world_seed=5, agent_seed=3)
    assert records_a == records_b
    assert result_a == result_b


def test_different_agent_seeds_give_different_actions():
    pid, records_a, _ = _record_episode(world_seed=5, agent_seed=3)
    _, records_b, _ = _record_episode(world_seed=5, agent_seed=4)
    actions_a = [a[pid] for _, a in records_a[1:]]
    actions_b = [a[pid] for _, a in records_b[1:]]
    n = min(len(actions_a), len(actions_b))
    assert n >= 5, "episodes too short to compare"
    assert actions_a[:n] != actions_b[:n]


def show_episode(world_seed=0, agent_seed=3, n_lines=10):
    """Eyeball check: print the first steps of one episode."""
    layout = generate_valid_layout(GenerationConfig(width=16, height=16, seed=world_seed))
    world = World(layout, seed=world_seed)
    player_id = world.reset().observation.player_id
    shown = []

    def on_step(world, step, actions):
        if len(shown) < n_lines:
            shown.append(step)
            print(step, actions)

    return run_episode(world, {player_id: RandomAgent(seed=agent_seed)}, on_step=on_step)


###############################################################################
# Run
###############################################################################


def run():
    test_returns_valid_actions()
    test_roughly_uniform()
    test_seed_determines_sequence()
    test_reset_does_not_reseed()
    test_history_is_bounded_and_cleared_on_reset()
    test_recorded_steps_are_consistent()
    test_same_seeds_reproduce_the_episode()
    test_different_agent_seeds_give_different_actions()
    print("random agent: all checks passed")


if __name__ == "__main__":
    run()