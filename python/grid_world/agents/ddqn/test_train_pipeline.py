"""
Created on Mon Sep 21 07:03:58 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################


import ripper

from agents.ddqn.agent import build_ddqn_agent
from agents.ddqn.train import train_agents
from agents.play import play_match

###############################################################################
# Layout Generation
###############################################################################


width  = 16
height = 16

generation_config = ripper.GenerationConfig(
    width=width, 
    height=height,
    wall_density=0.1,
    num_enemies=1,
    num_traps=0,
    min_player_goal_distance=10
)


layout = ripper.generate_valid_layout_py(
    config=generation_config, 
    max_attempts=100
)

###############################################################################
# World Generation
###############################################################################

world_config = ripper.WorldConfig()
world = ripper.PyWorld(layout, world_config)

world.print_world()

###############################################################################
# Agents
###############################################################################

lr         = 0.0001
batch_size = 32
episodes   = 2000
verbose    = True

player_agent = build_ddqn_agent(
    width=width, 
    height=height,
    channels=7,
    lr=lr, 
    batch_size=batch_size
)

enemy_agent = build_ddqn_agent(
    width=width, 
    height=height,
    channels=7,
    lr=lr, 
    batch_size=batch_size
)

agents = [player_agent, enemy_agent]
agent_ids = [world.get_player_id(), world.get_enemies_ids()[0]]

###############################################################################
# Train
###############################################################################

train_agents(agents, agent_ids, world, episodes, verbose=verbose)

###############################################################################
# Play match
###############################################################################

play_match(agents, agent_ids, world, delay=0.3)