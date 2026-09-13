"""
Created on Thu Sep 10 13:19:19 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from collections import deque, Counter
import random
import torch

from rl_core.replay_buffer import ReplayBuffer

import ripper
from ripper import Action
from ripper import EndReason

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Choose and action based on q value
###############################################################################

def choose_action(state, epsilon, policy_net, device):
    if random.random() < epsilon:
        return random.randint(0, 3)  # explore
    
    with torch.no_grad():
        
        # Create a batch dimension for a grid state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get actions 
        q_values = policy_net(state_tensor)
        
        # Select action with max value (best)
        action = q_values.argmax(dim=1).item()
    
    return action

###############################################################################
# Reward
###############################################################################

def reward_for(reason):
    
    reward = -0.1
    
    if reason == EndReason.GoalReached:
        reward = 1.0
    elif reason in (EndReason.Caught, EndReason.Trapped, EndReason.Timeout):
        reward = -1.0
    
    return reward

###############################################################################
# Tensor utils 
###############################################################################
    
def sample_to_tensor(sample, device):
    
    states, actions, rewards, next_states, dones = zip(*sample)
    
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_state_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
    
    return states_tensor, actions_tensor, rewards_tensor, next_state_tensor, dones_tensor
    
###############################################################################
# Train
###############################################################################

def train_step(agent, batch, gamma):
    
    # Sample a batch from buffer
    states, actions, rewards, next_states, dones = sample_to_tensor(batch, agent.device)

    # Get predicted actions from model for the current state [B, n_actions]
    q_values = agent.policy_net(states)
    
    # Select the chosen action [B, 1]
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        
        # Get the action with max value from the next state for policy model
        next_actions = agent.policy_net(next_states).argmax(dim=1)
        
        # Get the action with max value from the next state for policy model
        next_q_values = agent.target_net(next_states)
        
        max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + gamma * max_next_q * (1 - dones)

    # Calculate loss 
    loss = agent.loss_fn(q_taken, target_q)
    
    # Update weights
    agent.optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
    loss.backward()
    agent.optimizer.step()

    return loss.item()


def train(
    agent, 
    layout, 
    episodes, 
    max_tick=100,
    epsilon_start=1.0,
    epsilon_min=0.10, 
    epsilon_decay=0.999,
    batch_size=128, 
    min_buffer_size=1000, 
    target_update_freq=500,
    gamma=0.99, 
    verbose=False, 
    log_every=50
    
):

    # Initialize env with layout and episode max expiration tick
    world = ripper.PyWorld(layout, max_tick)
    
    # Get player id so we can pass the action 
    player_id = world.get_player_id()
    
    # Initialize replay buffer for training
    replay_buffer = ReplayBuffer(capacity=50000)
    
    # Initialize epsilon for exploration Vs exploitation
    epsilon = epsilon_start
    
    # Initialize step count
    train_step_count = 0
    recent_reasons = deque(maxlen=100)
    recent_losses = deque(maxlen=100) 

    
    for episode in range(episodes):
        
        # Reset env
        world.reset()
        
        # Get observation containing grid, player_id and enemies_ids
        observation = world.observation()
        
        # Initialize done 
        done = False

        while not done:
            
            # Chose action index based on grid state
            action_idx = choose_action(
                observation.grid, 
                epsilon, 
                agent.policy_net, 
                agent.device
            )
            
            # Advance step into the env
            step_result = world.step({player_id: ACTIONS[action_idx]})
            
            # Done, reason what happened during step phase
            done, reason = step_result.done, step_result.reason
            
            # Calculate reward based on current reason
            reward = reward_for(reason)

            # Get next observation after reacting to the step
            next_observation = world.observation()
            
            # Store observation on buffer
            replay_buffer.push(observation.grid, action_idx, reward, next_observation.grid, done)
            
            # Update current observation
            observation = next_observation

            if len(replay_buffer) >= min_buffer_size:
                
                batch = replay_buffer.sample(batch_size)
                loss = train_step(agent, batch, gamma)
                recent_losses.append(loss)
                train_step_count += 1
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                if train_step_count % target_update_freq == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())

        recent_reasons.append(str(reason))

        if verbose and episode % log_every == 0 and len(recent_reasons) > 0:
            win_rate = recent_reasons.count("EndReason.GoalReached") / len(recent_reasons)
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            breakdown = Counter(recent_reasons)
            print(f"Episode {episode}, train_steps: {train_step_count}, epsilon: {epsilon:.3f}, "
                  f"win_rate: {win_rate:.2%}, avg_loss: {avg_loss:.4f}, outcomes: {dict(breakdown)}")
                    
    return agent, epsilon 