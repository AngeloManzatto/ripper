"""
Created on Tue Oct  6 21:23:29 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################


"""
Compare saved benchmark runs: one row per (task, agent), using the latest
run of each. Values are mean +- std (in %) across agent seeds.

    from experiments.baselines.compare import compare
    compare()
"""

###############################################################################
# Libraries
###############################################################################

import json
from pathlib import Path

from agents.evaluate import OUTCOMES
from experiments.baselines.tasks import RESULTS_DIR, SUITE

###############################################################################
# Compare
###############################################################################


def _fmt(stat: dict, scale: float = 100.0) -> str:
    return f"{scale * stat['mean']:5.1f} ± {scale * stat['std']:3.1f}"


def compare(results_dir: Path | None = None) -> None:
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    suite_dir = results_dir / SUITE

    if not suite_dir.exists():
        raise FileNotFoundError(f"No results for suite '{SUITE}' in {results_dir}")

    print(f"suite {SUITE}  (% of episodes, mean ± std across agent seeds)\n")
    print(f"{'task':<9}{'agent':<9}" + "".join(f"{o:>14}" for o in OUTCOMES) + f"{'steps':>14}")

    for task_dir in sorted(suite_dir.iterdir()):
        for agent_dir in sorted(task_dir.iterdir()):
            runs = sorted(agent_dir.iterdir())  # run ids sort by time
            if not runs:
                continue

            agg = json.loads((runs[-1] / "summary.json").read_text())["aggregate"]

            cells = "".join(f"{_fmt(agg['rates'][o]):>14}" for o in OUTCOMES)
            steps = _fmt(agg["mean_steps"], scale=1.0)
            print(f"{task_dir.name:<9}{agent_dir.name:<9}{cells}{steps:>14}")