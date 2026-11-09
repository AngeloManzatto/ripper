"""
Created on Wed Oct 21 10:30:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import copy
from collections.abc import Sequence

import numpy as np

from agents.base_agent import BaseAgent
from agents.features.frame import features, FEATURE_NAMES, N_FEATURES
from agents.features.temporal import memory_flags, N_MEMORY

from core.actions import ACTIONS, N_ACTIONS
from core.danger import hostile_proximity, N_PROXIMITY

from engine import Action, EntityObservation

from neat.checkpoint import save_genome, load_genome
from neat.network import compile_genome, activate

###############################################################################
# Input
###############################################################################

# What the network sees: the frame features, the recent-visit flags, then the
# hostile-proximity flags. Each agent chooses its own input from the feature functions.
N_INPUTS = N_FEATURES + N_MEMORY + N_PROXIMITY

HOSTILE_FOUND = FEATURE_NAMES.index("hostile_found")
HOSTILE_DROW  = FEATURE_NAMES.index("hostile_drow")
HOSTILE_DCOL  = FEATURE_NAMES.index("hostile_dcol")

def build_input(history: Sequence[EntityObservation]) -> np.ndarray:
    """Input vector for the state at the end of `history`.

    history: observations, oldest first, current one last (at least one). Any
             sequence works, a list or the deque of a BaseAgent.
    Returns a float32 array of length N_INPUTS: the N_FEATURES features of the
    current observation, then the N_MEMORY recent-visit flags of the history,
    then the N_PROXIMITY hostile-proximity flags.
    Raises ValueError for an empty history. Nothing is modified.
    """
    if len(history) == 0:
        raise ValueError("history must contain at least the current observation")

    obs = history[-1]

    # Features of the observation
    x = features(obs)

    # Which neighbors were visited recently
    visited_flags = np.asarray(memory_flags(history), dtype=np.float32)

    # Hostile proximity flags
    cells = max(len(obs.grid), len(obs.grid[0])) - 1
    hostile_flags = np.asarray(
        hostile_proximity(
            x[HOSTILE_FOUND] > 0.5,
            x[HOSTILE_DROW] * cells,
            x[HOSTILE_DCOL] * cells,
        ),
        dtype=np.float32,
    )

    return np.concatenate([x, visited_flags, hostile_flags])

###############################################################################
# NEAT Agent
###############################################################################

class NeatAgent(BaseAgent):
    """
    A frozen policy that runs one NEAT genome.

    The genome is compiled once, when the agent is built. Every step the agent
    builds its input vector (features + recent-visit flags + hostile proximity),
    runs the network and takes the action whose output is largest. Ties go to
    the first action in ACTIONS.

    The agent does not learn: evolution happens outside, in the NEAT library.
    on_transition is the default no-op, and train / eval mode change nothing.

    save(path) writes the genome with the NEAT genome file format (the same file
    as neat.checkpoint.save_genome); load(path) puts the genome of a file into this
    agent, and a refused file (wrong sizes, not a genome) leaves the agent as it
    was. clone() gives an agent with its own copy of the genome and no recorder.

    """

    name = "neat"

    def __init__(self, genome, seed=42, history_size=8):
        super().__init__(seed=seed, history_size=history_size)

        self.recorder = None
        self._set_genome(genome)

    def _set_genome(self, genome):
        """Compile and check first, replace afterwards: a refused genome changes nothing."""
        network = compile_genome(genome)

        if network["n_inputs"] != N_INPUTS:
            raise ValueError(
                f"genome has {network['n_inputs']} inputs, "
                f"this agent builds {N_INPUTS}"
            )
        if network["n_outputs"] != N_ACTIONS:
            raise ValueError(
                f"genome has {network['n_outputs']} outputs, "
                f"there are {N_ACTIONS} actions"
            )

        # The agent shares nothing with the genome it was given: changing that
        # genome afterwards does not change this agent.
        self.network = network
        self.genome = copy.deepcopy(genome)

    def select_action(self, obs: EntityObservation) -> Action:

        # BaseAgent already appended `obs`: history[-1] is the current state
        x = build_input(self.history)
        outputs = activate(self.network, x)

        if self.recorder is not None:
            self.recorder.log_decision(x, outputs, self.training)

        return ACTIONS[int(np.argmax(outputs))]

    ###########################################################################
    # Persistence and copies
    ###########################################################################

    def save(self, path) -> None:
        save_genome(path, self.genome)

    def load(self, path) -> None:
        self._set_genome(load_genome(path))

    def clone(self) -> "NeatAgent":
        # No recorder in the copy: it would be shared or duplicated logs.
        other = NeatAgent(self.genome, history_size=self.history.maxlen)
        other.rng = copy.deepcopy(self.rng)
        other.training = self.training
        return other