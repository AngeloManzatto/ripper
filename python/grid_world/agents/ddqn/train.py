"""
Created on Mon Sep 21 21:53:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from collections import deque, Counter

from agents.reward import calculate_reward
from agents.observation import get_one_hot_grid_for_entity, get_position_for_entity

from ripper import Action

import torch
import numpy as np

import os
import psutil

process = psutil.Process(os.getpid())

def memory_usage():
    rss = process.memory_info().rss / 1024**3
    system = psutil.virtual_memory()

    print(
        f"Python RAM: {rss:.2f} GB | "
        f"System RAM: {system.percent:.1f}% | "
        f"Available: {system.available / 1024**3:.2f} GB"
    )
    
###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Tensor utils
###############################################################################

def sample_to_tensor(sample, device):

    states, actions, rewards, next_states, dones = zip(*sample)

    states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
    

    return states_tensor, actions_tensor, rewards_tensor, next_state_tensor, dones_tensor

###############################################################################
# Log training progress
###############################################################################

def log_progress(episode, agents, agent_ids, player_id, recent_reasons, recent_losses):

    total = len(recent_reasons)
    breakdown = Counter(recent_reasons)

    print(f"\n--- Episode {episode} (last {total} episodes) ---")
    print(f"{'Agent':<10}{'Role':<8}{'Epsilon':>9}{'WinRate':>10}{'AvgLoss':>10}")

    for agent, agent_id in zip(agents, agent_ids):
        is_player = agent_id == player_id
        role = "Player" if is_player else "Enemy"
        success_reason = "EndReason.GoalReached" if is_player else "EndReason.Caught"

        win_rate = breakdown.get(success_reason, 0) / total if total else 0.0

        losses = recent_losses[agent_id]
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        print(f"{agent_id:<10}{role:<8}{agent.epsilon:>9.3f}{win_rate:>10.2%}{avg_loss:>10.4f}")

    print(f"Outcomes: {dict(breakdown)}")
            
###############################################################################
# Optimize step 
###############################################################################

def optimize_step(agent, batch, gamma):

    states, actions, rewards, next_states, dones = sample_to_tensor(batch, agent.device)

    q_values = agent.policy_net(states)
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = agent.policy_net(next_states).argmax(dim=1)
        next_q_values = agent.target_net(next_states)
        max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = agent.loss_fn(q_taken, target_q)

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
    agent.optimizer.step()

    return loss.item()

###############################################################################
# Learn from transition
###############################################################################

def learn_from_transition(agent, state, action_idx, reward, next_state, done):
    
    # Store transition into buffer
    agent.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    # Compute loss
    loss = None
    
    # Train if enough buffer
    if len(agent.replay_buffer) >= agent.min_buffer_size:
        
        # Sample an experience from buffer
        batch = agent.replay_buffer.sample(agent.batch_size)
        
        # Calculate loss
        loss = optimize_step(agent, batch, agent.gamma)
        
        # Advance train step count
        agent.train_step_count += 1
        
        # Decay exploration coefficience
        agent.decay_epsilon()
        
        # Update target network 
        if agent.train_step_count % agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    return loss

###############################################################################
# Train step
###############################################################################

def train_step(agents, agent_ids, world, observation):
    
    player_id = world.get_player_id()  # hoisted out of the loop below

    action_indices = {}
    actions = {}
    states = {}
    
    for agent, agent_id in zip(agents, agent_ids):
        
        # Get current state for agent
        state = get_one_hot_grid_for_entity(observation, agent_id)
        
        states[agent_id] = state
        
        # Play an action given current env state
        action_idx = agent.act(state)
                               
        # Register action index for agent
        action_indices[agent_id] = action_idx
        
        # Regist action for agent
        actions[agent_id] = ACTIONS[action_idx]
        
    # Act on env
    step_result = world.step(actions)
    
    # Get env state
    done, reason = step_result.done, step_result.reason
     
    # Get state observation after action
    next_observation = world.observation()
    
    # Train step for agents
    losses = {}
    for agent, agent_id in zip(agents, agent_ids):
        
        # Get current state for agent
        state = states[agent_id]
        next_state = get_one_hot_grid_for_entity(next_observation, agent_id)
        
        position = get_position_for_entity(observation, agent_id)
        next_position  = get_position_for_entity(next_observation, agent_id)
        
        # Get reward
        reward = calculate_reward(
            reason=reason, 
            is_player=agent_id==player_id, 
            position=position, 
            state=state, 
            next_position=next_position, 
            next_state=next_state,
            k_novel=0.02, 
            k_sight=0.15, 
            k_dist=0.05
        )
        
        loss = learn_from_transition(
            agent, 
            states[agent_id], 
            action_indices[agent_id], 
            reward, 
            next_state, 
            done
        )
        
        losses[agent_id] = loss
        
    return next_observation, done, reason, losses
    
###############################################################################
# Train Agents
###############################################################################

def train_agents(agents, agent_ids, world, episodes, verbose=False, log_every=50):
    
    assert len(agents) == len(agent_ids)
    
    player_id = world.get_player_id()
    recent_reasons = deque(maxlen=100)
    recent_losses = {key: deque(maxlen=100) for key in agent_ids}
    
    for episode in range(episodes):
        
        # Reset env
        world.reset(reposition=True)
        
        # Take observation 
        observation = world.observation()
        
        # Initialize done
        done = False
        
        while not done:
            
            # Execute train step
            observation, done, reason, losses = train_step(
                agents, agent_ids, world, observation
            )
            
            # Update losses
            for agent_id in losses:
                if losses[agent_id] is not None:
                    recent_losses[agent_id].append(losses[agent_id])
                    
        recent_reasons.append(str(reason))

        if verbose and episode % log_every == 0 and len(recent_reasons) > 0:
            memory_usage()
            log_progress(episode, agents, agent_ids, player_id, recent_reasons, recent_losses)
            