"""
Created on Wed Nov 11 09:45:00 2026

@author: Angelo Antonio Manzatto

The DDQN learner: one role (player or enemy) that POET can train, with a DdqnAgent inside.

POET only sees agent(), train() and clone() (poet.learner.Learner). Everything DDQN stays here: how
many games one train() call plays, which seeds they use, and the rule that the agent which learns is
never the one that plays for POET.

    agent()  a NEW FROZEN copy of the learning agent on every call (eval mode, no learning, empty
             replay buffer), as NeatLearner builds a new agent on every call. POET plays, measures
             and compares with it; nothing it does reaches the learner, and the learner keeps none.
    train()  the learning agent plays games_per_call games on the layout against a frozen clone of
             the opponent, and learns from them as it plays (DdqnAgent.on_transition).
    clone()  an independent learner: the networks are copied, the optimizer and the replay buffer
             start fresh (see DdqnAgent.clone).

The reward each agent learns from is the agent's own reward_fn: build the player's agent with a
reward that pays for reaching the goal and the enemy's with one that pays for catching.

"""

###############################################################################
# Libraries
###############################################################################

import copy

import numpy as np

from poet.entry import OPPONENT_OF
from poet.games import goal_rate, play_games

###############################################################################
# DDQN Learner
###############################################################################

class DdqnLearner:
    """A DdqnAgent that learns one role.

    role            : "player" or "enemy_0".
    agent           : the DdqnAgent that learns (the learner takes it over and puts it in train mode).
    games_per_call  : games played by one train() call. Every game is also up to max_tick updates of
                      the network, so this is the learner's budget; NEAT's is about 10,000 games.
    train_seed      : first seed of the training games.
    seed_span       : how many seeds the training games may use: train_seed ... train_seed + span - 1.
                      The games walk through them in order and start again at the first when they run
                      out, so they never reach the validation games (keep the span below
                      valid_seed - train_seed).

    After the training calls:
        games_played : training games so far (this decides the seeds of the next call)
        history      : one dict per call: call, games, goal_rate (of the learning agent in those games,
                       exploring), epsilon, learn_steps, loss (the last one)
    """

    def __init__(self, role, agent, games_per_call=500, train_seed=200_000, seed_span=10_000):
        if role not in OPPONENT_OF:
            raise ValueError(f"role must be one of {tuple(OPPONENT_OF)}, got {role!r}")
        if games_per_call < 1:
            raise ValueError(f"games_per_call must be >= 1, got {games_per_call}")
        if seed_span < 1:
            raise ValueError(f"seed_span must be >= 1, got {seed_span}")

        agent.set_train_mode()

        self.role = role
        self.games_per_call = games_per_call
        self.train_seed = train_seed
        self.seed_span = seed_span

        self._agent = agent
        self.games_played = 0
        self.history = []

    def agent(self):
        """A new frozen copy of the learning agent: eval mode, no replay buffer, learns nothing.

        Playing with it never changes the learner, and the learner keeps no reference to it.
        """
        playing = self._agent.clone()
        playing.set_eval_mode()
        return playing

    def train(self, layout, opponent, rng):
        """Play games_per_call games on this layout against `opponent`, learning as the agent plays.

        `rng` (a random.Random) is the only source of randomness: it reseeds the agent's own
        generator (exploration and batches) at the start of the call. The opponent is played against
        as a frozen clone and is not changed. Training happens in place: when the engine fails
        halfway, what the agent learned until then stays (games_played and history do not advance),
        and the agent holds no observation of the engine, so the learner can still be trained,
        cloned and pickled. Returns None.
        """
        opponent = opponent.clone()
        opponent.set_eval_mode()

        agent = self._agent
        agent.set_train_mode()
        agent.rng = np.random.default_rng(rng.getrandbits(64))

        agents = {self.role: agent, OPPONENT_OF[self.role]: opponent}

        outcomes, played, remaining = [], self.games_played, self.games_per_call
        try:
            while remaining > 0:
                offset = played % self.seed_span
                chunk = min(remaining, self.seed_span - offset)       # never past the last seed of the span

                result = play_games(layout, agents, n_games=chunk, first_seed=self.train_seed + offset)
                outcomes += result[self.role]
                played += chunk
                remaining -= chunk
        finally:
            # The games leave the last episode in the agent (observations of the engine, which cannot
            # be copied or pickled): clear it, also when the games failed
            agent.reset()

        self.games_played = played
        self.history = self.history + [{
            "call": len(self.history) + 1,
            "games": self.games_per_call,
            "goal_rate": goal_rate(outcomes),
            "epsilon": agent.epsilon,
            "learn_steps": agent.learn_steps,
            "loss": agent.last_loss,
        }]

    def clone(self):
        """An independent learner with the same networks, the same count of games and the same
        history. Its optimizer and replay buffer are new (DdqnAgent.clone)."""
        other = DdqnLearner(self.role, self._agent.clone(), self.games_per_call, self.train_seed, self.seed_span)
        other.games_played = self.games_played
        other.history = copy.deepcopy(self.history)
        return other