"""
Created on Tue Oct 13 06:26:31 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import time
import csv
import itertools

import numpy as np

from engine import GenerationConfig, World, WorldConfig, generate_valid_layout
from core.rewards import ShapedGoalReward

from agents.linear_q_agent.agent import LinearQAgent
from agents.linear_q_agent.diagnostics import Recorder
from experiments.baselines.tasks import TASKS, EVAL_SEED
from runtime.train import train
from runtime.play import play
from runtime.evaluate import evaluate

###############################################################################DEFAULTS = dict(w_goal=1.0, w_explore=0.0, w_bump=0.0, w_revisit=0.0, w_danger=0.0)

# What to run
###############################################################################

RUN_SINGLE      = False   # A) one run: table, outcome counts, summary, play
RUN_DETERMINISM = False   # B) same config twice must print the same numbers
RUN_SWEEP       = True    # C) CONFIGS x AGENT_SEEDS -> one comparison table

###############################################################################
# Settings: the ONLY place where the agent and the training are defined
###############################################################################

AGENT_KWARGS = dict(alpha=0.02, epsilon=0.1, eval_epsilon=0.0)
AGENT_SEEDS  = (0, 1, 2)

# Reward parameters. A config only needs the keys it wants to change.
DEFAULTS = dict(w_goal=1.0, w_explore=0.0, w_bump=0.0, w_revisit=0.0, w_danger=0.0)

GRID = dict(
    w_explore = (0.0, 0.5, 1.0),
    w_bump    = (0.0, 0.5, 1.0),
    w_revisit = (0.0, 0.5, 1.0),
    w_danger  = (0.0, 0.5, 1.0),
)


CONFIGS = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]

RESULTS_CSV = time.strftime("sweep_%Y%m%d_%H%M%S.csv")   # new file per run
TOP         = 15                                         # rows shown in the final table

task         = TASKS["default"]
world_config = WorldConfig(**task.world_kwargs)

n_blocks   = 100
block_size = 25
n_eval     = 100

###############################################################################
# Building blocks
###############################################################################

def make_agent(agent_seed, **reward_params):
    reward_fn = ShapedGoalReward(**reward_params)
    return LinearQAgent(reward_fn, seed=agent_seed, **AGENT_KWARGS)


def train_agent(agent):
    return train(
        agent,
        n_blocks=n_blocks,
        block_size=block_size,
        n_eval=n_eval,
        train_seed=0,
        eval_seed=EVAL_SEED,
        generation_kwargs=task.generation_kwargs,
        world_config=world_config,
    )


def evaluate_agent(agent, n_episodes=100):
    """Evaluate with learning OFF, whatever mode train() left the agent in."""
    agent.set_eval_mode()
    return evaluate(
        agent=agent,
        n_episodes=n_episodes,
        seed=EVAL_SEED,
        generation_kwargs=task.generation_kwargs,
        world_config=world_config,
    )

###############################################################################
# Sweep
###############################################################################

METRICS = [            # (column title, key in the per-seed row, decimals)
    ("AUC",        "auc",     3),
    ("online AUC", "online",  3),
    ("goal",       "goal",    1),
    ("caught",     "caught",  1),
    ("timeout",    "timeout", 1),
]


def validate_configs(configs):
    """Fail BEFORE the long run, not after ten minutes."""
    for i, cfg in enumerate(configs):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"config {i} has unknown keys {sorted(unknown)}; "
                             f"valid keys: {sorted(DEFAULTS)}")


def run_config(cfg, seeds=AGENT_SEEDS):
    """Train + frozen-evaluate one reward config on every seed."""
    params = {**DEFAULTS, **cfg}
    rows = []
    for seed in seeds:
        start = time.time()
        agent = make_agent(seed, **params)
        result = train_agent(agent)
        counts = evaluate_agent(agent).outcome_counts
        rows.append(dict(
            auc=result.auc,
            online=result.online_auc,
            goal=counts["GoalReached"],
            caught=counts["Caught"],
            timeout=counts["Timeout"],
            seconds=time.time() - start,
        ))
    return params, rows


def varying_keys(all_params):
    """Parameters that differ between configs (the only interesting columns)."""
    keys = [k for k in DEFAULTS if len({p[k] for p in all_params}) > 1]
    return keys or list(DEFAULTS)


def mean_sd(values, decimals):
    if len(values) == 1:
        return f"{values[0]:.{decimals}f}"
    return f"{np.mean(values):.{decimals}f} ± {np.std(values, ddof=1):.{decimals}f}"


def append_csv(params, rows):
    """One line per finished config, so a crash does not lose the sweep."""
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(list(DEFAULTS) + [key for _, key, _ in METRICS])
        writer.writerow([params[k] for k in DEFAULTS] +
                        [float(np.mean([r[key] for r in rows])) for _, key, _ in METRICS])


def print_sweep(results, top=None):
    keys = varying_keys([params for params, _ in results])
    head = ["#"] + keys + [title for title, _, _ in METRICS]

    # best mean AUC first
    order = sorted(range(len(results)),
                   key=lambda i: -np.mean([r["auc"] for r in results[i][1]]))
    shown = order[:top] if top else order

    table = []
    for i in shown:
        params, rows = results[i]
        cells = [str(i)]
        cells += [f"{params[k]:g}" for k in keys]
        cells += [mean_sd([r[key] for r in rows], d) for _, key, d in METRICS]
        table.append(cells)

    widths = [max(len(row[j]) for row in [head] + table) for j in range(len(head))]

    def line(cells):
        return "  ".join(c.rjust(w) for c, w in zip(cells, widths))

    print()
    print(line(head))
    print("-" * len(line(head)))
    for cells in table:
        print(line(cells))

    n_seeds = len(results[0][1])
    print(f"\nTop {len(shown)} of {len(results)} configs, sorted by mean AUC.  "
          f"# = config index.  ± = sd over {n_seeds} seeds.  "
          f"goal/caught/timeout = mean of 100 frozen eval episodes.")
    
def load_csv(path):
    """Finished sweep from disk: list of (params, metrics), config means only."""
    entries = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            params  = {k: float(row[k]) for k in DEFAULTS}
            metrics = {key: float(row[key]) for _, key, _ in METRICS}
            entries.append((params, metrics))
    return entries


def to_means(results):
    """Same format as load_csv, from the in-memory results of a sweep."""
    return [(params, {key: float(np.mean([r[key] for r in rows]) )
                      for _, key, _ in METRICS})
            for params, rows in results]


def print_marginals(entries, metric="auc"):
    keys   = varying_keys([p for p, _ in entries])
    values = sorted({p[k] for p, _ in entries for k in keys})

    print(f"\nMean {metric} with each weight fixed at a value "
          f"(averaged over all other weights)")
    print("weight".ljust(12) + "".join(f"{v:>9g}" for v in values) + f"{'range':>9}")
    for k in keys:
        means = []
        for v in values:
            xs = [m[metric] for p, m in entries if p[k] == v]
            means.append(np.mean(xs) if xs else None)
        present = [x for x in means if x is not None]
        cells = "".join(f"{x:9.3f}" if x is not None else f"{'-':>9}" for x in means)
        print(k.ljust(12) + cells + f"{max(present) - min(present):9.3f}")


def print_interaction(entries, a, b, metric="auc"):
    va = sorted({p[a] for p, _ in entries})
    vb = sorted({p[b] for p, _ in entries})

    print(f"\nMean {metric}: rows = {a}, columns = {b} "
          f"(averaged over the other weights)")
    print(a.ljust(12) + "".join(f"{b}={v:g}".rjust(16) for v in vb))
    for x in va:
        cells = []
        for y in vb:
            xs = [m[metric] for p, m in entries if p[a] == x and p[b] == y]
            cells.append(f"{np.mean(xs):16.3f}" if xs else f"{'-':>16}")
        print(f"{x:<12g}" + "".join(cells))


def report(entries, metric="auc"):
    print_marginals(entries, metric)
    keys = varying_keys([p for p, _ in entries])
    for a, b in itertools.combinations(keys, 2):
        print_interaction(entries, a, b, metric)

###############################################################################
# A) One run, with everything visible
###############################################################################

if RUN_SINGLE:

    agent = make_agent(agent_seed=0, **DEFAULTS)

    rec = Recorder()
    agent.recorder = rec

    start = time.time()
    train_result = train_agent(agent)
    print(f"trained in {time.time() - start:.0f} s")

    print("episodes  eval_goal  eval_steps")
    for i in range(len(train_result.checkpoints)):
        print(f"{train_result.checkpoints[i]:8d}  "
              f"{train_result.eval_goal_rates[i]:9.3f}  "
              f"{train_result.eval_mean_steps[i]:10.1f}")
    print("AUC:", round(train_result.auc, 3))
    print("Online AUC:", round(train_result.online_auc, 3))

    final = evaluate_agent(agent)
    print(final.outcome_counts)

    agent.recorder = None
    for k, v in rec.summary().items():
        print(k, v)

    # Play one episode, watch it with your own eyes
    episode_seed = 10
    layout = generate_valid_layout(
        GenerationConfig(**task.generation_kwargs, seed=episode_seed)
    )
    world = World(layout, config=world_config, seed=episode_seed)
    player_id = world.reset().observation.player_id

    play(world, {player_id: agent})

###############################################################################
# B) Determinism: the same config twice must print the same two numbers
###############################################################################

if RUN_DETERMINISM:

    for _ in range(2):
        a = make_agent(agent_seed=0, **DEFAULTS)
        r = train_agent(a)
        print("determinism:", round(r.auc, 3), round(r.online_auc, 3))

###############################################################################
# C) Sweep: every config x every seed, one table at the end
###############################################################################

if RUN_SWEEP:

    validate_configs(CONFIGS)
    n_runs = len(CONFIGS) * len(AGENT_SEEDS)
    print(f"{len(CONFIGS)} configs x {len(AGENT_SEEDS)} seeds = {n_runs} runs "
          f"(~{n_runs * 21 / 60:.0f} min).  Saving to {RESULTS_CSV}")

    results = []
    sweep_start = time.time()
    for i, cfg in enumerate(CONFIGS):
        params, rows = run_config(cfg)
        results.append((params, rows))
        append_csv(params, rows)
        elapsed = (time.time() - sweep_start) / 60
        print(f"[{i + 1}/{len(CONFIGS)}] {cfg}  AUC "
              f"{np.mean([r['auc'] for r in rows]):.3f}  ({elapsed:.0f} min elapsed)")

    print_sweep(results, top=TOP)
    report(to_means(results))