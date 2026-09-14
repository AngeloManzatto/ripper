"""
Created on Fri Sep 11 07:14:57 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from pathlib import Path
import ripper

from agents.ddqn.agent import build_agent
from poet.lineage import run_lineage

###############################################################################
# Globals
###############################################################################

width  = 16
height = 16

max_tick=50
generations = 200
train_episodes_per_gen = 500

# model
lr = 0.0001

###############################################################################
# Files and folders
###############################################################################

model_type = "ddpg"

base_path = Path("checkpoint") / model_type
base_path.mkdir(parents=True, exist_ok=True)

checkpoint_path  = base_path / "agent_checkpoint.pt"
archive_path     = base_path / "archive.pkl"

###############################################################################
# Configs
###############################################################################

generation_config = ripper.GenerationConfig(
    width=width, 
    height=height,
    wall_density=0.1,
    num_enemies=1,
    num_traps=0
)

mutation_config = ripper.MutationConfig(
    n_wall_mutations=3,
    max_wall_density= 0.3,

    max_enemies=3,
    enemy_add_weight=0.4,
    enemy_remove_weight=0.2,
    enemy_nothing_weight=0.4,

    max_traps=3,
    trap_add_weight=0.2,
    trap_remove_weight=0.1,
    trap_nothing_weight=0.7,

    n_repositions=2,
)

###############################################################################
# Layout
###############################################################################

seed_layout = ripper.generate_valid_layout_py(
    config=generation_config, 
    max_attempts=100
    )

###############################################################################
# Agent
###############################################################################

agent = build_agent(width=width, height=height, lr=lr)

###############################################################################
# Run lineage
###############################################################################

agent, archive = run_lineage(
        agent, 
        seed_layout, 
        generations=generations, 
        mutation_config=mutation_config,
        train_episodes_per_gen=train_episodes_per_gen,
        max_tick=max_tick,
        mc_episodes=100,
        mc_min_rate=0.2, 
        mc_max_rate=0.7, 
        mutate_max_attempts=100,
        checkpoint_path=checkpoint_path, 
        archive_path=checkpoint_path,
        verbose=True
    )