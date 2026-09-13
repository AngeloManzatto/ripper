"""
Created on Fri Sep 11 20:26:29 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import ripper

from grid_world.training.train import train
from grid_world.poet.minimal_criterion import  satisfies_minimal_criterion

###############################################################################
# Run lineage
###############################################################################

def run_lineage(
        agent, 
        seed_layout, 
        generations, 
        mutation_config,
        train_episodes_per_gen=100, 
        mc_episodes=20,
        mc_min_rate=0.2, 
        mc_max_rate=0.7, 
        mutate_max_attempts=100,
        epsilon_start=1.0, 
        epsilon_min=0.05, 
        epsilon_decay=0.9999,
        verbose=False):

    current_layout = seed_layout
    archive = [current_layout]
    epsilon = epsilon_start   

    for generation in range(generations):
        
        agent, epsilon = train(
            agent, current_layout, episodes=train_episodes_per_gen,
            epsilon_start=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
            verbose=verbose
        )

        candidate = ripper.mutate_valid_layout_py(current_layout, mutation_config, mutate_max_attempts)
        
        if verbose:
            print(f"Candidate for Gen {generation}:\n{candidate}")
    
        accepted, result = satisfies_minimal_criterion(
            agent, candidate, episodes=mc_episodes, min_rate=mc_min_rate, max_rate=mc_max_rate
        )

        if accepted:
            current_layout = candidate
            archive.append(current_layout)

        if verbose:
            print(f"Gen {generation}: {'ACCEPTED' if accepted else 'rejected'}, "
                  f"win_rate={result['win_rate']:.2f}, epsilon={epsilon:.3f}, archive_size={len(archive)}")

    return agent, archive