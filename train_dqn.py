# train_dqn.py
"""
Train a DQN agent on CartPole-v1 for 100 episodes.
Saves:
 - dqn_cartpole.pth  (model weights)
 - rewards.npy       (episode rewards list)
 - training_rewards.png (plot)
"""

import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------
# Hyperparameters
# ---------------------
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 100            # as requested
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
MIN_REPLAY_SIZE = 200        # small so learning happens within 100 episodes
TARGET_UPDATE_EPISODES = 5   # copy policy -> target every N episodes
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995            # multiply per episode
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Utilities (gym compat)
# ---------------------
def reset_env(env):
    out = env.reset()
    # gymnasium returns (obs, info)
    if isinstance(out, tuple):
        return out[0]
    return out

def step_env(env, action):
    out = env.step(action)
    # gymnasium returns (obs, reward, terminated, truncated, info)
    if len(out) == 5:
        next_state, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        next_state, reward, done, info = out
    return next_state, reward, done, info

# ---------------------
# Replay Buffer
# ---------------------
Transition = collections.namedtuple('Transition',
                                    ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ---------------------
# Q-Network
# ---------------------
class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------
# Helper: select action (epsilon-greedy)
# ---------------------
def select_action(policy_net, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(0, n_actions)
    state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    qvals = policy_net(state_v)
    action = int(torch.argmax(qvals, dim=1).item())
    return action

# ---------------------
# Optimize step
# ---------------------
def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return None
    trans = memory.sample(BATCH_SIZE)
    state_batch = torch.tensor(np.array(trans.state), dtype=torch.float32).to(DEVICE)
    action_batch = torch.tensor(trans.action, dtype=torch.int64).unsqueeze(1).to(DEVICE)
    reward_batch = torch.tensor(trans.reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    next_state_batch = torch.tensor(np.array(trans.next_state), dtype=torch.float32).to(DEVICE)
    done_batch = torch.tensor(trans.done, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # Q(s,a) predictions
    q_values = policy_net(state_batch).gather(1, action_batch)

    # max_a' Q_target(s', a')
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q = reward_batch + (1.0 - done_batch) * GAMMA * next_q_values

    loss = F.smooth_l1_loss(q_values, expected_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ---------------------
# Training loop
# ---------------------
def train():
    env = gym.make(ENV_NAME)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = QNetwork(obs_size, n_actions).to(DEVICE)
    target_net = QNetwork(obs_size, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(REPLAY_CAPACITY)

    epsilon = EPS_START
    episode_rewards = []

    # Pre-fill memory with some random transitions
    state = reset_env(env)
    for _ in range(500):
        action = env.action_space.sample()
        next_state, reward, done, _ = step_env(env, action)
        memory.push(state, action, reward, next_state, float(done))
        state = next_state if not done else reset_env(env)

    print(f"Device: {DEVICE}. Starting training for {NUM_EPISODES} episodes...")

    for ep in range(1, NUM_EPISODES + 1):
        state = reset_env(env)
        ep_reward = 0
        done = False
        losses = []
        steps = 0

        while not done:
            action = select_action(policy_net, state, epsilon, n_actions)
            next_state, reward, done, _ = step_env(env, action)
            memory.push(state, action, reward, next_state, float(done))

            loss = optimize_model(policy_net, target_net, memory, optimizer)
            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_reward += reward
            steps += 1

            # safety cap (CartPole episodes typically limited by env)
            if steps > 1000:
                break

        episode_rewards.append(ep_reward)

        # epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # update target network periodically
        if ep % TARGET_UPDATE_EPISODES == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_loss = np.mean(losses) if losses else 0.0
        print(f"Episode {ep:3d}  Reward: {ep_reward:6.1f}  Eps: {epsilon:.3f}  AvgLoss: {avg_loss:.4f}  ReplaySize: {len(memory)}")

    env.close()

    # Save model and rewards
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "dqn_cartpole.pth")
    torch.save(policy_net.state_dict(), model_path)
    np.save(os.path.join("models", "rewards.npy"), np.array(episode_rewards))

    # Plot and save figure
    plt.figure(figsize=(8,5))
    plt.plot(episode_rewards, label="episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN CartPole Training (100 eps)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("models", "training_rewards.png"))
    plt.show()

    print(f"Training finished. Model and rewards saved to ./models/")

if __name__ == "__main__":
    train()
