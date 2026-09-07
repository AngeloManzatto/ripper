"""
Created on Tue Sep  1 08:00:05 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from pathlib import Path
import textwrap
import random
import ripper

import torch
import torch.nn as nn

from collections import deque

from ripper import Action
from ripper import EndReason

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

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
    def __init__(self, height=8, width=10, n_actions=4, channels=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * height * width, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
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

max_tick = 100

layout = textwrap.dedent("""
        ##########
        #P...#...#
        #....#...#
        #........#
        #...##..G#
        #...T....#
        #......E.#
        ##########
        """).strip()

g = ripper.PyWorld(layout, max_tick)
player_id = g.get_player_id()

replay_buffer = ReplayBuffer(capacity=10000)

height, width = g.get_grid_shape()
policy_net = DDQN(height=height, width=width).to(device)
target_net = DDQN(height=height, width=width).to(device)
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

episodes = 10000
train_step_count = 0

recent_rewards = deque(maxlen=100)
recent_reasons = deque(maxlen=100)

# Reload weights

best_avg_reward = float("-inf")

if checkpoint_path.exists():
    policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    print(f"Resumed training from checkpoint: {checkpoint_path}")

for episode in range(episodes):
    g.reset()
    observation = g.observation()
    done = False
    total_reward = 0

    while not done:
        action_idx = choose_action(observation.grid, epsilon, policy_net, device)
        
        step_result = g.step({player_id : ACTIONS[action_idx]})
        
        done, reason = step_result.done, step_result.reason
        reward = 0.0

        if reason == EndReason.GoalReached:
            reward = 1.0
        elif reason in (EndReason.Caught, EndReason.Trapped, EndReason.Timeout):
            reward = -1.0
        
        
        next_observation = g.observation()

        replay_buffer.push(observation.grid, action_idx, reward, next_observation.grid, done)
        observation = next_observation
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

    recent_rewards.append(total_reward)
    recent_reasons.append(str(reason))
    
    if episode % 50 == 0 and len(recent_rewards) >= 100:
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        win_rate = recent_reasons.count("EndReason.GoalReached") / len(recent_reasons)
        
        print(f"Episode {episode}, avg_reward(last 100): {avg_reward:.3f}, "
              f"win_rate(last 100): {win_rate:.2%}, epsilon: {epsilon:.3f}")
    
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

def evaluate(g, policy_net, device, render=True):
    g.reset()
    player_id = g.get_player_id()
    observation = g.observation()
    
    path = []
    
    if render:
        g.print_world()
    
    reason = None
    done = False
    step_num = 1
    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(observation.grid, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
        
        step_result = g.step({player_id : ACTIONS[action_idx]})
        
        done, reason = step_result.done, step_result.reason
        
        next_observation = g.observation()
        
        path.append(ACTIONS[action_idx])
        observation = next_observation
        
        if render:
            g.print_world()
            
        step_num += 1
        
    if render:
        print(f"Finished the game in {step_num} steps for the following reason: {reason}")
        print("Path:", path)
    return path, reason

reasons = {}

for i in range(1000):

    path, reason = evaluate(g, policy_net.eval(), device, render=False)
    
    reasons[str(reason)] = reasons.get(str(reason), 0) + 1
   
print(reasons)

path, reason = evaluate(g, policy_net, device, render=True)