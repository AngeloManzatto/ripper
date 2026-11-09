"""
Created on Sat Oct 24 18:10:00 2026

@author: Angelo Antonio Manzatto

Watch the Player that evolution produced, instead of only reading its scores.

  1. For each saved genome, play N_GAMES games on layouts that played no part in training
     and print one line per game: outcome, steps, how many different cells it visited, how
     often it stayed on its cell (walked into a wall) and how often it stepped back to the cell
     it was on two decisions before (flip-flop). The summary lines at the end say which
     outcome comes with which behaviour.
  2. watch(genome, seed) draws one game frame by frame with your play(). Pick a seed from the
     table: a Timeout with a high stay or flip share is the kind of game worth watching.

Run this file (F5 in Spyder) to get the tables, then call watch(...) in the console, e.g.

    watch("neat_player_control_period4_seed1.json", 220_017)

The genome files are the ones written by run_curriculum.py (one per mode, period and seed).
"""

###############################################################################
# Libraries
###############################################################################

from engine import GenerationConfig, World, generate_valid_layout
from experiments.neat_player.fitness import RecordingAgent
from experiments.neat_player.moves import move_stats, behaviour_table
from neat.checkpoint import load_genome
from runtime.play import play          # <- change this line if play.py lives somewhere else
from runtime.runner import run_episode
import experiments.neat_player.run_rotating as rr

###############################################################################
# Settings (change here)
###############################################################################

# label -> genome file. The first is the best run so far, the second the run whose population
# got stuck (same seed, trained with the curriculum).
GENOME_FILES = {
    "control seed 1 (best)":     "neat_player_control_period4_seed1.json",
    "curriculum seed 1 (stuck)": "neat_player_curriculum_period4_seed1.json",
}

TARGET = {"wall_density": 0.2, "num_enemies": 1, "num_traps": 0}

FIRST_SEED = rr.TEST_SEED      # games use the seeds FIRST_SEED, FIRST_SEED + 1, ... (never trained on)
N_GAMES = 40

###############################################################################
# Playing
###############################################################################

def make_world(seed, generation_kwargs=None):
    """The world of layout `seed`, built exactly as runtime.evaluate builds it. Returns (world, player_id)."""
    layout = generate_valid_layout(GenerationConfig(**(generation_kwargs or {}), seed=seed))
    if layout is None:
        raise RuntimeError(f"Could not generate a valid layout for seed {seed}")
    world = World(layout, None, seed)
    player_id = world.reset().observation.player_id
    return world, player_id


def play_recorded(genome, seed, generation_kwargs=None):
    """Play one game without drawing. Returns (EpisodeResult, positions)."""
    world, player_id = make_world(seed, generation_kwargs)
    agent = RecordingAgent(genome)
    result = run_episode(world, {player_id: agent})
    return result, agent.positions


def behaviour_rows(genome, first_seed, n_games, generation_kwargs=None):
    """One dict per game: seed, outcome and the numbers of move_stats."""
    rows = []
    for seed in range(first_seed, first_seed + n_games):
        result, positions = play_recorded(genome, seed, generation_kwargs)
        rows.append({"seed": seed, "outcome": result.outcome, **move_stats(positions)})
    return rows


def as_genome(genome_or_path):
    return load_genome(genome_or_path) if isinstance(genome_or_path, str) else genome_or_path


def watch(genome_or_path, seed, generation_kwargs=TARGET, delay=0.15, clear=True):
    """Draw one game frame by frame (your play()), then print its move numbers.
    clear=False keeps every frame so you can scroll back."""
    genome = as_genome(genome_or_path)
    world, player_id = make_world(seed, generation_kwargs)
    agent = RecordingAgent(genome)
    result = play(world, {player_id: agent}, delay=delay, clear=clear)
    print(move_stats(agent.positions))
    return result

###############################################################################
# Run
###############################################################################

def main():
    all_rows = {}
    for label, path in GENOME_FILES.items():
        genome = load_genome(path)
        rows = behaviour_rows(genome, FIRST_SEED, N_GAMES, TARGET)
        all_rows[label] = rows
        print(f"\n{label}   ({path})   {N_GAMES} unseen games, target world\n")
        print(behaviour_table(rows))
    return all_rows


if __name__ == "__main__":
    all_rows = main()