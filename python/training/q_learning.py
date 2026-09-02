"""
Created on Sat Aug 29 07:26:26 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import random
import ripper

###############################################################################
# Globals
###############################################################################

ACTIONS = ["up", "down", "left", "right"]

###############################################################################
# Q Learning (# maps (row, col) -> [q_up, q_down, q_left, q_right])
###############################################################################

def get_q(state):
    if state not in Q:
        Q[state] = [0.0, 0.0, 0.0, 0.0]
    return Q[state]

# Get player position
def get_entity_pos(obs, entitiy_kind="player"):
    
    # channel 0 -> Player, # channel 1 -> Enemy, channel 2 -> Goal
    
    if entitiy_kind == "player":
        grid = obs[0]
        
    elif entitiy_kind == "enemy":
        grid = obs[1]
        
    else:
        raise Exception(f'player or enemy entity kind allower, but found: {entitiy_kind}')
        
    
    
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    player_pos = None
    
    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] == 1.0:
                
                player_pos = (i, j)
                
                return player_pos
            
    return player_pos

# Choose and action based on q value
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)  # explore
    else:
        q_values = get_q(state)
        return q_values.index(max(q_values))  # exploit — best known action
    

# Update Q table
# Q[s][a] ← Q[s][a] + α * (reward + γ * max(Q[s']) − Q[s][a])
def update_q(state, action, reward, next_state, done, alpha=0.1, gamma=0.99):
    q_values = get_q(state)
    next_q_values = get_q(next_state)
    
    target = reward if done else reward + gamma * max(next_q_values)
    q_values[action] += alpha * (target - q_values[action])

###############################################################################
# Execute Training
###############################################################################

Q = {}  

g = ripper.PyGrid()
epsilon = 0.8
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999  # multiply epsilon by this after every episode
episodes = 50000
step = 0

for episode in range(episodes):
    g.reset()
    obs = g.observation()
    player_pos = get_entity_pos(obs, "player")
    enemy_pos  = get_entity_pos(obs, "enemy")
    state = (player_pos, enemy_pos)
    done = False
    total_reward = 0
    
    while not done:
        
        action_idx = choose_action(state, epsilon)
        reward, done = g.step(ACTIONS[action_idx])
        
        next_obs = g.observation()
        player_pos = get_entity_pos(next_obs, "player")
        enemy_pos  = get_entity_pos(next_obs, "enemy")
        
        next_state = (player_pos, enemy_pos)
        
        update_q(state, action_idx, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if step % 100000 == 0:
            print(action_idx, step, next_state)
        
        step +=1
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # <- moved here, once per episode
    
    if episode % 50 == 0:
        print(f"Episode {episode}, reward: {total_reward}, epsilon: {epsilon:.3f}")


###############################################################################
# Evaluation
###############################################################################

def evaluate(g, Q, max_steps=50):
    g.reset()
    obs = g.observation()
    player_pos = get_entity_pos(obs, "player")
    enemy_pos  = get_entity_pos(obs, "enemy")
    state = (player_pos, enemy_pos)
    
    path = [state]
    
    g.render()
    
    for step_num in range(max_steps):
        q_values = get_q(state)  # uses existing Q dict
        action_idx = q_values.index(max(q_values))
        
        reward, done = g.step(ACTIONS[action_idx])
        
        next_obs = g.observation()
        player_pos = get_entity_pos(next_obs, "player")
        enemy_pos  = get_entity_pos(next_obs, "enemy")
        
        next_state = (player_pos, enemy_pos)
        
        path.append(next_state)
        state = next_state
        
        g.render()
        
        if done:
            print(f"Reached goal in {step_num + 1} steps!")
            break
    else:
        print(f"Did not reach goal within {max_steps} steps.")
    
    print("Path:", path)
    return path

evaluate(g, Q, 100)