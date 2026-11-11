"""
Created on Wed Nov 11 10:05:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

###############################################################################
# Replay buffer
###############################################################################

class ReplayBuffer:
    """The last `capacity` transitions of an agent, as numpy arrays.

    push(state, action, reward, next_state, terminated) stores one transition. When the
    buffer is full the oldest one is overwritten. sample(batch_size, rng) returns
    `batch_size` different stored transitions, drawn with the rng you give (so the draw
    is repeatable); the buffer has no random generator of its own.

    The arrays are allocated at the first push, with the size of the first state. An empty
    buffer (a fresh clone, an agent that only plays) therefore costs nothing.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self.capacity = capacity
        self._size = 0
        self._next = 0          # the row the next push writes to
        self._data = None       # (states, actions, rewards, next_states, terminated)

    def __len__(self) -> int:
        return self._size

    def _allocate(self, n_inputs: int):
        c = self.capacity
        self._data = (
            np.zeros((c, n_inputs), dtype=np.float32),   # states
            np.zeros(c, dtype=np.int64),                 # actions (index into ACTIONS)
            np.zeros(c, dtype=np.float32),               # rewards
            np.zeros((c, n_inputs), dtype=np.float32),   # next states
            np.zeros(c, dtype=np.float32),               # terminated: 1.0 or 0.0
        )

    def push(self, state, action: int, reward: float, next_state, terminated: bool) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        if self._data is None:
            self._allocate(len(state))

        n_inputs = self._data[0].shape[1]
        if state.shape != (n_inputs,) or next_state.shape != (n_inputs,):
            raise ValueError(
                f"states must have {n_inputs} numbers, got {state.shape} and {next_state.shape}"
            )

        states, actions, rewards, next_states, ended = self._data
        row = self._next
        states[row], actions[row], rewards[row] = state, action, reward
        next_states[row], ended[row] = next_state, float(terminated)

        self._next = (row + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        """(states, actions, rewards, next_states, terminated): numpy arrays, `batch_size` rows."""
        if batch_size < 1 or batch_size > self._size:
            raise ValueError(f"cannot sample {batch_size} from a buffer with {self._size} transitions")

        rows = rng.choice(self._size, size=batch_size, replace=False)

        return tuple(array[rows] for array in self._data)