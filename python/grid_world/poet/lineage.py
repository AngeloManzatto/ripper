"""
Created on Fri Sep 11 20:26:29 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import pickle
import ripper

from poet.minimal_criterion import satisfies_minimal_criterion

###############################################################################
# Run lineage
###############################################################################

def run_lineage(
        agent, 
        seed_layout, 
        generations, 
        mutation_config,
        train_episodes_per_gen=100,
        max_tick=100,
        mc_episodes=50,
        mc_min_rate=0.2, 
        mc_max_rate=0.7, 
        mutate_max_attempts=100,
        checkpoint_path="agent_checkpoint.pt", 
        archive_path="archive.pkl",
        verbose=False):

    current_layout = seed_layout
    archive = [current_layout]

    for generation in range(generations):
        
        if verbose:
            print(f"Trained for Gen {generation}:\n{current_layout}")
        
        # Train agent on the current layout
        agent.train(current_layout, 
                    episodes=train_episodes_per_gen, 
                    max_tick=max_tick,
                    verbose=verbose)

        # Apply mutation
        candidate_layout = ripper.mutate_valid_layout_py(
            current_layout, 
            mutation_config, 
            mutate_max_attempts
        )
        
        # Check how well the agent trained on a previous env goes into a new candidate
        accepted, result = satisfies_minimal_criterion(
            agent, 
            candidate_layout,
            episodes=mc_episodes,
            min_rate=mc_min_rate,
            max_rate=mc_max_rate,
            max_tick=max_tick
        )

        if accepted:
            current_layout = candidate_layout
            archive.append(current_layout)

            agent.save(checkpoint_path)
            with open(archive_path, "wb") as f:
                pickle.dump(archive, f)

        if verbose:
            print(f"Candidate for Gen {generation}:\n{candidate_layout}")
            print(f"Gen {generation}: {'ACCEPTED' if accepted else 'rejected'}, win_rate={result['win_rate']:.2f}, outcomes={result['outcomes']}, archive_size={len(archive)}")

    agent.save(checkpoint_path)
    with open(archive_path, "wb") as f:
        pickle.dump(archive, f)

    return agent, archive