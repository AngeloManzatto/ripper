"""
Created on Thu Sep 10 13:19:19 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import torch
import ripper

###############################################################################
# Train
###############################################################################

def train_step(agent, batch):
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
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
    loss.backward()
    agent.optimizer.step()

    return loss.item()


def train(agent, layout, episodes, max_tick=100,
          epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9999,
          batch_size=128, min_buffer_size=1000, target_update_freq=500,
          gamma=0.99):

    world = ripper.PyWorld(layout, max_tick)
    player_id = world.get_player_id()
    replay_buffer = ReplayBuffer(capacity=10000)
    epsilon = epsilon_start
    train_step_count = 0

    for episode in range(episodes):
        world.reset()
        observation = world.observation()
        done = False

        while not done:
            action_idx = choose_action(observation.grid, epsilon, agent.policy_net, agent.device)
            step_result = world.step({player_id: ACTIONS[action_idx]})
            done, reason = step_result.done, step_result.reason
            reward = reward_for(reason)

            next_observation = world.observation()
            replay_buffer.push(observation.grid, action_idx, reward, next_observation.grid, done)
            observation = next_observation

            if len(replay_buffer) >= min_buffer_size:
                batch = replay_buffer.sample(batch_size)
                train_step(agent, batch)
                train_step_count += 1
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                if train_step_count % target_update_freq == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())

    return agent