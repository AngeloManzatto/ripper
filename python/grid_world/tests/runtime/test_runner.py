"""
Tests for runtime/runner.py: run_episode and EpisodeResult.

run_episode(world, agents) plays one episode and returns an EpisodeResult whose agent_outcomes
holds one entry per agent given, keyed by entity id. The fake world below follows the rules of
the Rust engine (state.rs) that matter to the runner:

    - a dying entity is still in the observation of the step it dies in, with its final status
      ("Trapped", or "GoalReached" for an enemy that caught the player), and gone from the next one;
    - an enemy that dies does not end the game; the player's goal, trap or capture does
      (terminated), and the clock does (truncated, nothing is applied on that step);
    - entities without an agent are moved by the engine and have no entry in agent_outcomes.

What each agent's outcome is:
    - its own final status when that is not "Alive" ("GoalReached", "Trapped", "Caught"),
      whenever it was reached, even if the entity was removed from the world afterwards;
    - otherwise "Timeout" when the clock ended the game, or "Alive" when the game ended by
      termination while the agent was still alive.

Also checked: an agent never acts after its entity died, its last transition is terminal, the
step count and the terminated / truncated flags, and the errors for missing or unknown agents.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

from types import SimpleNamespace

from agents.base_agent import BaseAgent
from core.actions import ACTIONS
from runtime.runner import run_episode

###############################################################################
# fake world, following the engine rules
###############################################################################

PLAYER, ENEMY_A, ENEMY_B = 1, 2, 3


class FakeTimeStep:
    def __init__(self, observation, terminated, truncated):
        self.observation, self.terminated, self.truncated = observation, terminated, truncated

    def done(self):
        return self.terminated or self.truncated


class FakeWorld:
    """Plays a script: one dict per world step.

        {"status": {id: new status}, "terminated": bool, "truncated": bool, "remove": [ids]}

    The observation of a step is built with the new statuses, then the ids in "remove" leave the
    world, like Rust's remove_dead. After the script ends the clock runs out (truncated)."""

    def __init__(self, script, ids=(PLAYER, ENEMY_A)):
        self.script, self.ids = script, ids

    def _observation(self):
        entities = [SimpleNamespace(id=i, status=self.status[i], position=(0, i)) for i in self.present]
        return SimpleNamespace(entities=entities, player_id=PLAYER)

    def reset(self, reposition=False):
        self.status = {i: "Alive" for i in self.ids}
        self.present = list(self.ids)
        self.tick = 0
        self.received = []
        return FakeTimeStep(self._observation(), False, False)

    def step(self, actions):
        self.received.append(dict(actions))
        if self.tick >= len(self.script):
            return FakeTimeStep(self._observation(), False, True)
        event = self.script[self.tick]
        self.tick += 1
        self.status.update(event.get("status", {}))
        observation = self._observation()
        self.present = [i for i in self.present if i not in event.get("remove", [])]
        return FakeTimeStep(observation, event.get("terminated", False), event.get("truncated", False))


class Walker(BaseAgent):
    name = "walker"

    def __init__(self):
        super().__init__()
        self.transitions = []
        self.acts = 0

    def select_action(self, obs):
        self.acts += 1
        return ACTIONS[0]

    def on_transition(self, t):
        self.transitions.append(t)


def play(script, agents, ids=(PLAYER, ENEMY_A)):
    world = FakeWorld(script, ids)
    return run_episode(world, agents), world


def idle(n):
    return [{} for _ in range(n)]

###############################################################################
# helpers
###############################################################################

def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False

###############################################################################
# outcomes
###############################################################################

def test_a_game_that_runs_out_of_time_is_a_timeout_for_every_agent():
    result, _ = play(idle(3), {PLAYER: Walker(), ENEMY_A: Walker()})
    assert result.agent_outcomes == {PLAYER: "Timeout", ENEMY_A: "Timeout"}
    assert result.truncated is True and result.terminated is False


def test_the_player_reaching_the_goal_ends_the_game_and_the_enemy_is_still_alive():
    script = idle(2) + [{"status": {PLAYER: "GoalReached"}, "terminated": True}]
    result, _ = play(script, {PLAYER: Walker(), ENEMY_A: Walker()})
    assert result.agent_outcomes == {PLAYER: "GoalReached", ENEMY_A: "Alive"}
    assert result.terminated is True and result.truncated is False and result.steps == 3


def test_the_player_stepping_on_a_trap_ends_the_game():
    script = [{"status": {PLAYER: "Trapped"}, "terminated": True}]
    result, _ = play(script, {PLAYER: Walker(), ENEMY_A: Walker()})
    assert result.agent_outcomes == {PLAYER: "Trapped", ENEMY_A: "Alive"}


def test_an_enemy_that_catches_the_player_reaches_its_goal():
    script = idle(1) + [{"status": {PLAYER: "Caught", ENEMY_A: "GoalReached"}, "terminated": True,
                         "remove": [ENEMY_A]}]
    result, _ = play(script, {PLAYER: Walker(), ENEMY_A: Walker()})
    assert result.agent_outcomes == {PLAYER: "Caught", ENEMY_A: "GoalReached"}


def test_only_the_enemy_that_caught_the_player_reaches_its_goal():
    script = [{"status": {PLAYER: "Caught", ENEMY_B: "GoalReached"}, "terminated": True, "remove": [ENEMY_B]}]
    result, _ = play(script, {PLAYER: Walker(), ENEMY_A: Walker(), ENEMY_B: Walker()}, ids=(PLAYER, ENEMY_A, ENEMY_B))
    assert result.agent_outcomes == {PLAYER: "Caught", ENEMY_A: "Alive", ENEMY_B: "GoalReached"}


def test_an_enemy_that_dies_keeps_its_outcome_when_the_game_times_out():
    script = idle(1) + [{"status": {ENEMY_A: "Trapped"}, "remove": [ENEMY_A]}] + idle(2)
    result, _ = play(script, {PLAYER: Walker(), ENEMY_A: Walker()})
    assert result.agent_outcomes == {PLAYER: "Timeout", ENEMY_A: "Trapped"}


def test_an_enemy_that_dies_keeps_its_outcome_when_the_game_ends_later_by_termination():
    script = ([{"status": {ENEMY_A: "Trapped"}, "remove": [ENEMY_A]}] + idle(2)
              + [{"status": {PLAYER: "GoalReached"}, "terminated": True}])
    result, _ = play(script, {PLAYER: Walker(), ENEMY_A: Walker()})
    assert result.agent_outcomes == {PLAYER: "GoalReached", ENEMY_A: "Trapped"}


def test_an_enemy_without_an_agent_has_no_entry_even_when_it_dies():
    script = [{"status": {ENEMY_A: "Trapped"}, "remove": [ENEMY_A]}] + idle(1)
    result, _ = play(script, {PLAYER: Walker()})
    assert result.agent_outcomes == {PLAYER: "Timeout"}


def test_the_engine_keeps_moving_an_enemy_without_an_agent():
    result, world = play(idle(2), {PLAYER: Walker()})
    assert all(set(actions) == {PLAYER} for actions in world.received)

###############################################################################
# what the agents experience
###############################################################################

def test_an_enemy_acts_until_the_step_it_dies_and_never_after():
    enemy = Walker()
    script = idle(2) + [{"status": {ENEMY_A: "Trapped"}, "remove": [ENEMY_A]}] + idle(3)
    result, world = play(script, {PLAYER: Walker(), ENEMY_A: enemy})
    assert enemy.acts == 3                                   # steps 1, 2 and the one it died in
    assert [ENEMY_A in actions for actions in world.received[:4]] == [True, True, True, False]
    assert all(ENEMY_A not in actions for actions in world.received[3:])


def test_the_last_transition_of_a_dead_enemy_is_terminal_and_the_earlier_ones_are_not():
    enemy = Walker()
    script = idle(1) + [{"status": {ENEMY_A: "Trapped"}, "remove": [ENEMY_A]}] + idle(2)
    play(script, {PLAYER: Walker(), ENEMY_A: enemy})
    assert [t.terminated for t in enemy.transitions] == [False, True]
    assert enemy.transitions[-1].next_obs.status == "Trapped"
    assert all(t.truncated is False for t in enemy.transitions)


def test_a_timeout_is_a_truncation_and_not_a_termination_for_the_transitions():
    player = Walker()
    play(idle(1), {PLAYER: player})
    assert [(t.terminated, t.truncated) for t in player.transitions] == [(False, False), (False, True)]


def test_a_game_that_ends_by_termination_is_terminal_for_every_agent_that_acted():
    player, enemy = Walker(), Walker()
    script = [{"status": {PLAYER: "GoalReached"}, "terminated": True}]
    play(script, {PLAYER: player, ENEMY_A: enemy})
    assert player.transitions[-1].terminated is True and enemy.transitions[-1].terminated is True


def test_every_agent_is_reset_before_the_episode():
    player, enemy = Walker(), Walker()
    player.positions.append((9, 9)); enemy.history.append("old")
    play(idle(1), {PLAYER: player, ENEMY_A: enemy})
    assert player.positions[0] != (9, 9) and "old" not in list(enemy.history)

###############################################################################
# errors and bookkeeping
###############################################################################

def test_the_player_must_have_an_agent():
    assert raises(ValueError, play, idle(1), {ENEMY_A: Walker()})


def test_an_agent_for_an_entity_that_is_not_in_the_world_is_refused():
    assert raises(ValueError, play, idle(1), {PLAYER: Walker(), 99: Walker()})


def test_steps_count_the_world_steps():
    result, _ = play(idle(4), {PLAYER: Walker()})
    assert result.steps == 5                                 # four scripted steps and the clock step


def test_on_step_is_called_once_after_the_reset_and_once_per_step():
    calls = []
    world = FakeWorld(idle(2))
    run_episode(world, {PLAYER: Walker()}, on_step=lambda w, step, actions: calls.append((step, sorted(actions))))
    assert [step for step, _ in calls] == [0, 1, 2, 3]
    assert calls[0][1] == []

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as error:
            failed.append((name, type(error).__name__, str(error)[:80]))
    for item in failed:
        print("FAILED", *item)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(tests)} runner checks failed")
    print(f"runner: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()