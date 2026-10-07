"""
Created on Tue Oct  6 21:11:40 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tasks and suite constants for the baseline benchmark.

A Task is a named environment setting (data, not code), so it can be saved
into a run's config.json and compared across agents.

A suite = the tasks below + a FIXED range of evaluation layout seeds.
Every agent is judged on exactly the same layouts. Training / evolution must
use seeds OUTSIDE this range (below EVAL_SEED), so evaluation stays held out.
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass, field
from pathlib import Path

###############################################################################
# Suite constants
###############################################################################

SUITE = "v0"

# Episode i of every evaluation uses layout seed EVAL_SEED + i
EVAL_SEED = 10_000

# 
RESULTS_DIR = Path("experiments") / "results"

###############################################################################
# Tasks
###############################################################################


@dataclass(frozen=True)
class Task:
    name: str
    generation_kwargs: dict = field(default_factory=dict)  # -> GenerationConfig
    world_kwargs: dict = field(default_factory=dict)       # -> WorldConfig


TASKS: dict[str, Task] = {
    t.name: t
    for t in [
        # Engine defaults: 16x16, 1 enemy, no traps
        Task("default"),
        # Goal far from the player. 12 is a first guess: calibrate it so the
        # random agent lands around 2-3% goal rate.
        Task("far", generation_kwargs={"min_player_goal_distance": 12}),
        # Exercises the Trapped outcome
        Task("traps", generation_kwargs={"num_traps": 3}),
    ]
}