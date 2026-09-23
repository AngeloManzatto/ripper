"""
Created on Tue Sep 22 13:15:24 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import ripper

from agents.ddqn.agent import build_ddqn_agent
from agents.ddqn.train import train_agents
from agents.evaluate import evaluate_match, passes_minimal_criterion

###############################################################################
# Configs
###############################################################################

#####################################
# World config
#####################################

max_tick = 100
player_perception_range = 5
enemy_perception_range  = 5

world_config = ripper.WorldConfig(
    max_tick=max_tick,
    player_perception_range=player_perception_range,
    enemy_perception_range=enemy_perception_range
)

#####################################
# Generation config
#####################################

width  = 16
height = 16
wall_density = 0.1
num_enemies = 1
num_traps = 0
min_player_goal_distance = 10

generation_config = ripper.GenerationConfig(
    width=width, 
    height=height,
    wall_density=wall_density,
    num_enemies=num_enemies,
    num_traps=num_traps,
    min_player_goal_distance=min_player_goal_distance
)

#####################################
# Mutation config
#####################################

n_wall_mutations = 1
max_wall_density = 0.3

max_enemies = 1
enemy_add_weight     = 0.4
enemy_remove_weight  = 0.2
enemy_nothing_weight = 0.4

max_traps = 3
trap_add_weight     = 0.2
trap_remove_weight  = 0.1
trap_nothing_weight = 0.7

n_repositions = 1
min_player_goal_distance = 10

mutation_config = ripper.MutationConfig(
    n_wall_mutations=n_wall_mutations,
    max_wall_density=max_wall_density,
    max_enemies=max_enemies,
    enemy_add_weight=enemy_add_weight,
    enemy_remove_weight=enemy_remove_weight,
    enemy_nothing_weight=enemy_nothing_weight,
    max_traps=max_traps,
    trap_add_weight=trap_add_weight,
    trap_remove_weight=trap_remove_weight,
    trap_nothing_weight=trap_nothing_weight,
    n_repositions=n_repositions,
    min_player_goal_distance=min_player_goal_distance
)

###############################################################################
# Layout Generation
###############################################################################

layout = ripper.generate_valid_layout_py(
    config=generation_config, 
    max_attempts=100
)

world = ripper.PyWorld(layout, world_config)
world.print_world()

###############################################################################
# Agents
###############################################################################

lr         = 0.0001
batch_size = 32
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
# Mutate Envs
###############################################################################

def mutate_envs(
        EA_list, 
        world_config,
        mutation_config, 
        mutate_max_attempts=100, 
        eval_episodes=30,
        win_low=0.2, 
        win_high=0.7, 
        timeout_cap=0.3
    ):
    """
    For each reproduction-eligible entry in EA_list, attempt one mutation of
    its layout. If the mutated layout passes the minimal criterion (scored
    with the *parent's current* agents), add it to EA_list as a new entry
    with cloned agents. Returns the (possibly grown) EA_list.
    """
    new_entries = []

    for parent in EA_list:

        # If we still never evaluate an agent play skip
        if parent.get("last_eval") is None:
            print(f"parent skipped, last_eval={parent['last_eval']}")
            continue
        
        # Check if 
        is_accepted = passes_minimal_criterion(
            parent["last_eval"], 
            win_low, 
            win_high, 
            timeout_cap
        )
        
        if not is_accepted:
            continue
        
        # Create a mutated layout from parent to put on test with the trained agents
        candidate = ripper.mutate_valid_layout_py(
            parent["layout"], 
            mutation_config, 
            mutate_max_attempts
        )    
        
        # Initialize a test env for evaluation
        world = ripper.PyWorld(candidate, world_config)
        agents = [parent["player_agent"], parent["enemy_agent"]]
        agent_ids = [world.get_player_id(), world.get_enemies_ids()[0]]
        
        # Check the performance of both trained agents on this new candidate
        results = evaluate_match(agents, agent_ids, world, eval_episodes)
        
        # If passes the minimal criterion increment the list with this trained agent clone
        if passes_minimal_criterion(results, win_low, win_high, timeout_cap):
            child_entry = {
                "layout": candidate,
                "world": world,
                "player_agent": parent["player_agent"].clone(),
                "enemy_agent": parent["enemy_agent"].clone(),
                "player_id": agent_ids[0],
                "enemy_id": agent_ids[1],
                "last_eval": None,
            }
            new_entries.append(child_entry)

    return EA_list + new_entries

def best_transfer_candidate(EA_list, m, entry, world, eval_episodes, role):
    """
    Find the best other entry's agent for `role` ("player" or "enemy") when
    dropped into `entry`'s world, holding the other role fixed at entry's
    own current agent. Returns the winning agent object (not a clone -- the
    caller clones it), or None if nothing beat entry's own current agent.
    """
    own_agent_key = f"{role}_agent"
    own_id_key = f"{role}_id"

    best_candidate = None
    best_win_rate = entry["last_eval"][entry[own_id_key]]["win_rate"] \
        if entry["last_eval"] else 0.0

    for other_m, other in enumerate(EA_list):
        if other_m == m:
            continue

        if role == "player":
            candidate_agents = [other["player_agent"], entry["enemy_agent"]]
        else:
            candidate_agents = [entry["player_agent"], other["enemy_agent"]]
        candidate_agent_ids = [entry["player_id"], entry["enemy_id"]]

        results = evaluate_match(candidate_agents, candidate_agent_ids, world, eval_episodes)
        candidate_win_rate = results[entry[own_id_key]]["win_rate"]

        if candidate_win_rate > best_win_rate:
            best_candidate = other[own_agent_key]
            best_win_rate = candidate_win_rate

    return best_candidate


def attempt_transfer(EA_list, eval_episodes=30):
    """
    For each entry, try substituting every OTHER entry's player_agent (resp.
    enemy_agent) into this entry's world, holding the other role fixed. If a
    candidate scores a higher win_rate than the entry's own current agent,
    swap it in (a clone, not the original object -- entries must stay
    independent). Mutates EA_list in place and also returns it.
    """
    for m, entry in enumerate(EA_list):
        if len(EA_list) <= 1:
            break

        world = entry["world"]

        best_player = best_transfer_candidate(EA_list, m, entry, world, eval_episodes, role="player")
        if best_player is not None:
            entry["player_agent"] = best_player.clone()

        # enemy transfer runs against entry["player_agent"] as it now stands --
        # possibly already updated by the player transfer just above
        best_enemy = best_transfer_candidate(EA_list, m, entry, world, eval_episodes, role="enemy")
        if best_enemy is not None:
            entry["enemy_agent"] = best_enemy.clone()

    return EA_list

###############################################################################
# EA_list initialization
###############################################################################

def run_poet(
        EA_list,
        world_config,
        mutation_config,
        T=20,
        N_mutate=5,
        N_transfer=5,
        train_episodes=100,
        eval_episodes=50,
        mutate_max_attempts=100,
        win_low=0.2,
        win_high=0.7,
        timeout_cap=0.3,
        verbose=False,
    ):
    for t in range(T):

        # TODO 1: mutation, every N_mutate iterations, skipped on t == 0
        #         (matches Algorithm 2 line 5: "if t > 0 and t mod N_mutate == 0")
        if t > 0 and t % N_mutate == 0:
            EA_list = mutate_envs(
                EA_list, world_config, mutation_config,
                mutate_max_attempts=mutate_max_attempts,
                eval_episodes=eval_episodes,
                win_low=win_low, win_high=win_high, timeout_cap=timeout_cap,
            )

        # TODO 2: optimize every entry -- this is your existing train+eval
        #         block from step 3, just moved inside the outer loop.
        for e in EA_list:
            agents    = [e["player_agent"], e["enemy_agent"]]
            agent_ids = [e["player_id"], e["enemy_id"]]
            world     = e["world"]

            train_agents(agents, agent_ids, world, train_episodes, verbose=verbose)

            results = evaluate_match(agents, agent_ids, world, eval_episodes)
            e["last_eval"] = results

        # TODO 3: transfer, every N_transfer iterations, only if population > 1
        #         (matches line 14: "if M > 1 and t mod N_transfer == 0")
        if len(EA_list) > 1 and t % N_transfer == 0:
            EA_list = attempt_transfer(EA_list, eval_episodes=eval_episodes)

        # TODO 4: a short per-iteration status line -- population size, and
        #         each entry's latest win_rate(s), so you can watch it run.
        print(f"--- t={t} | population={len(EA_list)} ---")
        for i, e in enumerate(EA_list):
            print(f"  entry {i}: {e['last_eval']}")

    return EA_list

###############################################################################
# Run POET
###############################################################################

entry = {
    "layout": layout,
    "world": world,
    "player_agent": player_agent,
    "enemy_agent": enemy_agent,
    "player_id": agent_ids[0],
    "enemy_id": agent_ids[1],
    "last_eval": None,
}

EA_list = [entry]

#####################################
# Params
#####################################

T                   = 2000
N_mutate            = 5
N_transfer          = 5
train_episodes      = 100
eval_episodes       = 50
mutate_max_attempts = 100
win_low             = 0.2
win_high            = 0.7
timeout_cap         = 0.3
verbose             = True

#####################################
# Run
#####################################

EA_list = run_poet(
    EA_list=EA_list,
    world_config=world_config,
    mutation_config=mutation_config, 
    T=T,
    N_mutate=N_mutate,
    N_transfer=N_transfer,
    train_episodes=train_episodes,
    eval_episodes=eval_episodes,
    mutate_max_attempts=mutate_max_attempts,
    win_low=win_low,
    win_high=win_high,
    timeout_cap=timeout_cap,
    verbose=verbose
)
    
