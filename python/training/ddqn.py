"""
Created on Tue Sep  1 08:00:05 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from pathlib import Path

import random
import ripper

import torch
import torch.nn as nn

from collections import deque

###############################################################################
# Globals
###############################################################################

ACTIONS = ["up", "down", "left", "right"]

###############################################################################
# Files and folders
###############################################################################

checkpoint_dir = Path("python") / "training" / "checkpoints" / "ddqn"
checkpoint_dir.mkdir(parents=True,exist_ok=True)

checkpoint_path      = checkpoint_dir / "ddqn_weights.pt"
best_checkpoint_path = checkpoint_dir / "ddqn_weights_best.pt"

###############################################################################
# Replay buffer
###############################################################################

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

###############################################################################
# Model
###############################################################################

class DDQN(nn.Module):
    def __init__(self, grid_size=10, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * grid_size * grid_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

###############################################################################
# Policy 
###############################################################################

# Choose and action based on q value
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
# Train step 
###############################################################################

def train_step(policy_net, target_net, batch, optimizer, loss_fn, device, gamma=0.99):
    
    states, actions, rewards, next_states, dones = sample_to_tensor(batch, device)
    
    q_values = policy_net(states)
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(dim=1)                    # policy_net picks the action
        next_q_values = target_net(next_states)                                  # target_net evaluates it
        max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = loss_fn(q_taken, target_q)
    
    optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    loss.backward()
    optimizer.step()
    
    return loss.item()

###############################################################################
# Execute DDQN pipeline
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g = ripper.PyGrid()
replay_buffer = ReplayBuffer(capacity=10000)

policy_net = DDQN().to(device)
target_net = DDQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9999

batch_size = 128
min_buffer_size = 1000   # don't train until the buffer has real diversity
target_update_freq = 500  # sync target_net every N training steps
max_steps_per_episode = 200

episodes = 5000
train_step_count = 0

recent_rewards = deque(maxlen=100)

# Reload weights

best_avg_reward = float("-inf")

if checkpoint_path.exists():
    policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    print(f"Resumed training from checkpoint: {checkpoint_path}")

for episode in range(episodes):
    g.reset()
    state = g.observation()
    done = False
    total_reward = 0

    for t in range(max_steps_per_episode):
        action_idx = choose_action(state, epsilon, policy_net, device)
        reward, done = g.step(ACTIONS[action_idx])
        next_state = g.observation()

        replay_buffer.push(state, action_idx, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= min_buffer_size:
            
            batch = replay_buffer.sample(batch_size)
            loss = train_step(policy_net, target_net, batch, optimizer, loss_fn, device)
            train_step_count += 1
            
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            if train_step_count % 500 == 0:
                print(f"train_step {train_step_count}, loss: {loss:.4f}")

            if train_step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if done:
            break
    
    recent_rewards.append(total_reward)
    
    if episode % 50 == 0 and len(recent_rewards) >= 100:
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        print(f"Episode {episode}, avg_reward(last 100): {avg_reward:.3f}, epsilon: {epsilon:.3f}")
    
        # Periodic checkpoint — always overwrite, so you can resume from the latest state
        torch.save(policy_net.state_dict(), checkpoint_path)
    
        # Best-so-far checkpoint — only overwrite when genuinely improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(policy_net.state_dict(), best_checkpoint_path)
            print(f"  New best avg_reward: {best_avg_reward:.3f} — saved to {best_checkpoint_path}")

###############################################################################
# Evaluation
###############################################################################

def evaluate(g, policy_net, device, max_steps=50, render=True):
    g.reset()
    state = g.observation()
    
    path = [state]
    
    if render:
        g.render()
    
    for step_num in range(max_steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
        
        reward, done = g.step(ACTIONS[action_idx])
        next_state = g.observation()
        
        path.append(next_state)
        state = next_state
        
        if render:
            g.render()
        
        if done:
            print(f"Episode ended in {step_num + 1} steps, reward: {reward}")
            break
    else:
        print(f"Did not finish within {max_steps} steps.")
    
    return path

path = evaluate(g, policy_net.eval(), device, max_steps = 100)