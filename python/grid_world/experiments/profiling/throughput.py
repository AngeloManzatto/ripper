"""
Created on Wed Oct  7 07:32:37 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Step 0: where does the time go?

Measures steps/second for increasingly realistic loops, so the cost of each
layer shows up as the difference between two rows:

    A  world.step only                      Rust step + building the TimeStep
    B  A + ts.observation                   clone of the observation
    C  B + entities / ids (what the runner reads)
    D  C + player grid -> one-hot           what a learner or NEAT agent pays
    E  full run_episode with RandomAgent    everything, incl. agent + history

and the fixed per-episode cost (generate layout + build World + reset).

Run from the console (cwd = python/grid_world/):

    from experiments.profiling.throughput import main
    main()
"""

###############################################################################
# Libraries
###############################################################################

from time import perf_counter

import numpy as np

from agents.random_agent import ACTIONS, RandomAgent
from agents.runner import run_episode
from core.observation import one_hot_encode_grid
from engine import GenerationConfig, World, generate_valid_layout

###############################################################################
# Measurements
###############################################################################


def _setup(size: int, seed: int = 0):
    layout = generate_valid_layout(GenerationConfig(width=size, height=size, seed=seed))
    if layout is None:
        raise RuntimeError(f"No valid layout for size {size}, seed {seed}")
    world = World(layout, seed=seed)
    player_id = world.reset().observation.player_id
    return world, player_id


def _time_steps(size: int, n_steps: int, read) -> float:
    """Seconds for n_steps random steps; `read(ts)` is what the loop reads."""
    world, pid = _setup(size)
    choices = np.random.default_rng(0).integers(len(ACTIONS), size=n_steps).tolist()

    t0 = perf_counter()
    for i in range(n_steps):
        if world.done:
            world.reset()
        ts = world.step({pid: ACTIONS[choices[i]]})
        if read is not None:
            read(ts, pid)
    return perf_counter() - t0


def _read_observation(ts, pid):
    ts.observation


def _read_runner_like(ts, pid):
    obs = ts.observation
    {e.id: e for e in obs.entities}
    obs.player_id


def _read_one_hot(ts, pid):
    obs = ts.observation
    for e in obs.entities:
        if e.id == pid:
            one_hot_encode_grid(e.grid)


def _time_full_pipeline(size: int, n_steps: int) -> float:
    world, pid = _setup(size)
    agent = RandomAgent(seed=0)

    steps = 0
    t0 = perf_counter()
    while steps < n_steps:
        steps += run_episode(world, {pid: agent}).steps
    elapsed = perf_counter() - t0
    return elapsed * n_steps / steps  # normalise to n_steps


def _time_episode_setup(size: int, n_episodes: int = 100) -> float:
    """Seconds per episode for layout generation + World + reset."""
    t0 = perf_counter()
    for seed in range(n_episodes):
        layout = generate_valid_layout(GenerationConfig(width=size, height=size, seed=seed))
        if layout is None:
            raise RuntimeError(f"No valid layout for size {size}, seed {seed}")
        World(layout, seed=seed).reset()
    return (perf_counter() - t0) / n_episodes

###############################################################################
# Main
###############################################################################


def main(sizes=(16, 32), n_steps: int = 20_000) -> None:
    for size in sizes:
        rows = [
            ("A  step only", _time_steps(size, n_steps, None)),
            ("B  + ts.observation", _time_steps(size, n_steps, _read_observation)),
            ("C  + entities / ids (runner)", _time_steps(size, n_steps, _read_runner_like)),
            ("D  + player grid one-hot", _time_steps(size, n_steps, _read_one_hot)),
            ("E  full run_episode (random)", _time_full_pipeline(size, n_steps)),
        ]

        print(f"\n{size}x{size}, {n_steps} steps per row")
        print(f"{'':<32}{'steps/s':>10}{'µs/step':>10}")
        for name, seconds in rows:
            print(f"{name:<32}{n_steps / seconds:>10,.0f}{1e6 * seconds / n_steps:>10.1f}")

        setup = _time_episode_setup(size)
        print(f"per-episode setup: {1e3 * setup:.2f} ms")

        # Link back to the NEAT budget: 300 training episodes of ~80 steps
        full_us = 1e6 * rows[-1][1] / n_steps
        per_genome = 300 * (80 * full_us * 1e-6 + setup)
        print(f"~{per_genome:.1f} s per genome for 300 episodes x 80 steps (full pipeline)")


if __name__ == "__main__":
    main()