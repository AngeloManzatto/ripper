"""
Created on Tue Oct  6 21:19:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Benchmark runner: one agent x one task x several agent seeds.

Each agent seed builds a FRESH agent (via the registry) and evaluates it on
the same layouts (EVAL_SEED ... EVAL_SEED + n_episodes - 1). So differences
between agent seeds are agent noise only, and different agents are compared
on identical layouts.

Every run writes a new folder (never overwrites):

    results/<suite>/<task>/<agent>/<run_id>/
        config.json    everything needed to reproduce the run
        episodes.csv   one row per episode
        summary.json   per-seed rates + mean/std across agent seeds
"""

import csv
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from agents.evaluate import OUTCOMES, evaluate
from agents.registry import make_agent
from engine import WorldConfig
from experiments.baselines.tasks import EVAL_SEED, RESULTS_DIR, SUITE, TASKS, Task

###############################################################################
# Helpers
###############################################################################


def _code_version() -> str | None:
    """Short git hash of the code that produced the run (None if unavailable)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _mean_std(values: list[float]) -> dict[str, float]:
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(np.mean(values)), "std": std}

###############################################################################
# Run benchmark
###############################################################################


def run_benchmark(
    agent_name: str,
    task: Task,
    agent_params: dict | None = None,
    agent_seeds=range(5),
    n_episodes: int = 200,
    results_dir: Path | None = None,
) -> Path:
    """Run and save one benchmark. Returns the run folder."""

    agent_params = agent_params or {}
    agent_seeds = list(agent_seeds)
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR

    world_config = WorldConfig(**task.world_kwargs) if task.world_kwargs else None

    rows = []
    per_seed = []

    for agent_seed in agent_seeds:
        agent = make_agent(agent_name, agent_seed, **agent_params)

        summary = evaluate(
            agent,
            n_episodes=n_episodes,
            seed=EVAL_SEED,
            generation_kwargs=task.generation_kwargs,
            world_config=world_config,
        )

        per_seed.append({
            "agent_seed": agent_seed,
            "rates": {o: summary.rate(o) for o in OUTCOMES},
            "mean_steps": summary.mean_steps,
        })

        for i, r in enumerate(summary.results):
            rows.append([agent_seed, i, EVAL_SEED + i, r.outcome, r.steps])

    aggregate = {
        "rates": {o: _mean_std([s["rates"][o] for s in per_seed]) for o in OUTCOMES},
        "mean_steps": _mean_std([s["mean_steps"] for s in per_seed]),
    }

    # ---- save -------------------------------------------------------------
    now = datetime.now()
    run_dir = results_dir / SUITE / task.name / agent_name / now.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "suite": SUITE,
        "task": asdict(task),
        "agent": {"name": agent_name, "params": agent_params},
        "agent_seeds": agent_seeds,
        "n_episodes": n_episodes,
        "eval_seed": EVAL_SEED,
        "code_version": _code_version(),
        "created": now.isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    with open(run_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_seed", "episode", "layout_seed", "outcome", "steps"])
        writer.writerows(rows)

    (run_dir / "summary.json").write_text(
        json.dumps({"per_seed": per_seed, "aggregate": aggregate}, indent=2)
    )

    return run_dir

###############################################################################
# Main: random baseline on every task
###############################################################################


def main(agent_name: str = "random", **kwargs) -> None:
    for task in TASKS.values():
        run_dir = run_benchmark(agent_name, task, **kwargs)
        print(f"{task.name:<8} -> {run_dir}")


if __name__ == "__main__":
    main()