"""
Created on Mon Sep 21 22:54:17 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.observation import get_one_hot_grid_for_entity
from agents.reward import reward_for

from ripper import Action

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Compute GAE
###############################################################################

def compute_gae(rewards, values, dones, final_value, gamma, gae_lambda):
    advantages = [0.0] * len(rewards)

    gae = 0.0
    next_value = float(final_value)

    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[t])

        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae

        advantages[t] = gae
        next_value = values[t]

    return advantages

###############################################################################
# Log progress
###############################################################################

def log_progress(iteration, agents, agent_ids, recent_reasons, recent_losses, player_id, log_every):
    if iteration % log_every != 0:
        return

    print(f"--- Iteration {iteration} ---")
    for agent, agent_id in zip(agents, agent_ids):
        reasons = recent_reasons[agent_id]
        losses = recent_losses[agent_id]

        if len(reasons) == 0:
            continue

        role = "player" if agent_id == player_id else "enemy"
        win_reason = "EndReason.GoalReached" if role == "player" else "EndReason.Caught"

        win_rate = reasons.count(win_reason) / len(reasons)
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        breakdown = Counter(reasons)

        print(f"  agent {agent_id} ({role}): win_rate(last {len(reasons)}): {win_rate:.2%}, "
              f"avg_loss: {avg_loss:.4f}, outcomes: {dict(breakdown)}")

###############################################################################
# Collect rollout (all agents step the shared world in lockstep)
###############################################################################

def collect_rollout(agents, agent_ids, world, observation, rollout_steps):
    player_id = world.get_player_id()

    # per-agent trajectory buffers
    trajectories = {
        agent_id: {"states": [], "actions": [], "rewards": [], "log_probs": [], "values": [], "dones": []}
        for agent_id in agent_ids
    }
    episode_reasons = {agent_id: [] for agent_id in agent_ids}

    for _ in range(rollout_steps):
        actions = {}
        step_states = {}
        step_action_idx = {}

        for agent, agent_id in zip(agents, agent_ids):
            state = get_one_hot_grid_for_entity(observation, agent_id)
            action_idx = agent.act(state)

            step_states[agent_id] = state
            step_action_idx[agent_id] = action_idx
            actions[agent_id] = ACTIONS[action_idx]

        step_result = world.step(actions)
        done, reason = step_result.done, step_result.reason

        next_observation = world.observation()

        for agent, agent_id in zip(agents, agent_ids):
            is_player = agent_id == player_id
            reward = reward_for(reason, is_player)

            traj = trajectories[agent_id]
            traj["states"].append(step_states[agent_id])
            traj["actions"].append(step_action_idx[agent_id])
            traj["log_probs"].append(agent.last_log_prob)
            traj["values"].append(agent.last_value)
            traj["rewards"].append(reward)
            traj["dones"].append(done)

        if done:
            for agent_id in agent_ids:
                episode_reasons[agent_id].append(str(reason))
            world.reset()
            next_observation = world.observation()

        observation = next_observation

    final_values = {}
    for agent, agent_id in zip(agents, agent_ids):
        final_state = get_one_hot_grid_for_entity(observation, agent_id)
        final_values[agent_id] = agent.value_of(final_state)

    return trajectories, episode_reasons, final_values, observation

###############################################################################
# Optimize step (PPO clipped-objective epoch/minibatch update for one agent)
###############################################################################

def optimize_step(agent, states, actions, rewards, log_probs, values, dones, final_value):
    advantages = compute_gae(rewards, values, dones, final_value, agent.gamma, agent.gae_lambda)
    returns = [adv + val for adv, val in zip(advantages, values)]

    states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=agent.device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=agent.device)
    old_log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32, device=agent.device)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=agent.device)

    advantages_tensor = (
        advantages_tensor - advantages_tensor.mean()
    ) / (advantages_tensor.std(unbiased=False) + 1e-8)

    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=agent.device)

    num_samples = states_tensor.size(0)
    losses = []

    for _ in range(agent.epochs_per_update):
        indices = torch.randperm(num_samples, device=agent.device)

        for start in range(0, num_samples, agent.batch_size):
            idx = indices[start:start + agent.batch_size]

            logits, values_pred = agent.model(states_tensor[idx])
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions_tensor[idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_tensor[idx])

            mb_advantages = advantages_tensor[idx]

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - agent.clip_epsilon, 1 + agent.clip_epsilon) * mb_advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.functional.mse_loss(values_pred.squeeze(-1), returns_tensor[idx])

            loss = policy_loss + 0.5 * critic_loss - agent.entropy_coef * entropy

            agent.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)

            agent.optimizer.step()

            losses.append(loss.item())

    return losses

###############################################################################
# Train agents (N-agent PPO training loop over a shared, pre-built world)
###############################################################################

def train_agents(agents, agent_ids, world, iterations, rollout_steps=512, log_every=50):
    assert len(agents) == len(agent_ids)

    player_id = world.get_player_id()

    world.reset()
    observation = world.observation()

    recent_reasons = {agent_id: deque(maxlen=100) for agent_id in agent_ids}
    recent_losses = {agent_id: deque(maxlen=100) for agent_id in agent_ids}

    for iteration in range(iterations):
        trajectories, episode_reasons, final_values, observation = collect_rollout(
            agents, agent_ids, world, observation, rollout_steps
        )

        for agent, agent_id in zip(agents, agent_ids):
            recent_reasons[agent_id].extend(episode_reasons[agent_id])

            traj = trajectories[agent_id]
            losses = optimize_step(
                agent,
                traj["states"], traj["actions"], traj["rewards"],
                traj["log_probs"], traj["values"], traj["dones"],
                final_values[agent_id],
            )
            recent_losses[agent_id].extend(losses)

        log_progress(iteration, agents, agent_ids, recent_reasons, recent_losses, player_id, log_every)

    return recent_reasons, recent_losses