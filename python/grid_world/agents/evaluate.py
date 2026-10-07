"""
Created on Mon Oct  5 09:35:02 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Evaluation: play many episodes of one agent on seeded layouts and summarise
the outcomes.

    evaluate(agent, n_episodes, seed)
        episode i uses layout seed (seed + i) and world seed (seed + i)
        one agent instance plays every episode (reset() between episodes,
        its rng is NOT reseeded), so the same agent seed + the same `seed`
        reproduces the exact same evaluation.

To measure noise, vary the agent seed and keep `seed` fixed (same layouts),
or the reverse.
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass

from agents.base_agent import BaseAgent
from agents.runner import EpisodeResult, run_episode
from engine import GenerationConfig, World, WorldConfig, generate_valid_layout

###############################################################################
# Summary
###############################################################################

OUTCOMES = ("GoalReached", "Trapped", "Caught", "Timeout")

@dataclass(frozen=True)
class EvalSummary:
    n_episodes: int
    outcome_counts: dict[str, int]
    mean_steps: float
    results: list[EpisodeResult]

    def rate(self, outcome: str) -> float:
        """Fraction of episodes that ended with `outcome`."""
        return self.outcome_counts.get(outcome, 0) / self.n_episodes

    def __str__(self) -> str:
        parts = [f"{o}: {self.rate(o):.1%}" for o in self.outcome_counts]
        return (
            f"{self.n_episodes} episodes | "
            + " | ".join(parts)
            + f" | mean steps: {self.mean_steps:.1f}"
        )

###############################################################################
# Evaluate
###############################################################################

def evaluate(
    agent: BaseAgent,
    n_episodes: int = 100,
    seed: int = 0,
    generation_kwargs: dict | None = None,
    world_config: WorldConfig | None = None,
) -> EvalSummary:
    """
    Play `n_episodes` episodes with `agent` controlling the player.

    generation_kwargs: forwarded to GenerationConfig (width, height,
                       wall_density, ...). The seed is set per episode.
    world_config:      forwarded to World (max_tick, perception ranges, ...).
    """

    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    generation_kwargs = generation_kwargs or {}

    results: list[EpisodeResult] = []

    for i in range(n_episodes):
        episode_seed = seed + i

        layout = generate_valid_layout(
            GenerationConfig(**generation_kwargs, seed=episode_seed)
        )
        if layout is None:
            raise RuntimeError(
                f"Could not generate a valid layout for seed {episode_seed}; "
                "loosen the generation settings"
            )

        world = World(layout, world_config, episode_seed)

        # The player id can differ between layouts
        player_id = world.reset().observation.player_id

        results.append(run_episode(world, {player_id: agent}))

    outcome_counts = {o: 0 for o in OUTCOMES}
    for r in results:
        outcome_counts[r.outcome] = outcome_counts.get(r.outcome, 0) + 1

    return EvalSummary(
        n_episodes=n_episodes,
        outcome_counts=outcome_counts,
        mean_steps=sum(r.steps for r in results) / n_episodes,
        results=results,
    )