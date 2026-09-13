"""
Created on Fri Sep 11 07:14:57 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import ripper

from grid_world.training.agent import build_agent
from grid_world.poet.lineage import run_lineage

###############################################################################
# Globals
###############################################################################

width  = 16
height = 16

generations = 200
train_episodes_per_gen = 500

# model

lr = 0.0001

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
        mc_episodes=20,
        mc_min_rate=0.2, 
        mc_max_rate=0.7, 
        mutate_max_attempts=100,
        verbose=True
    )