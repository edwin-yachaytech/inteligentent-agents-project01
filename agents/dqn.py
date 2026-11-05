"""DQN implemented directly with NumPy replay and PyTorch."""
import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.obs = np.empty((capacity, obs_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.terminated = np.empty(capacity, np.float32)
        self.capacity, self.pos, self.size = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(self.size, size=batch_size)
        return tuple(torch.from_numpy(a[idx]) for a in
                     (self.obs, self.actions, self.rewards, self.next_obs, self.terminated))


class DQNAgent:
    def __init__(self, obs_dim, actions, lr=0.001, width=64, gamma=0.99,
                 buffer_size=50000, target_update=1000, no_target=False):
        self.online = nn.Sequential(nn.Linear(obs_dim, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU(), nn.Linear(width, actions))
        self.target = nn.Sequential(nn.Linear(obs_dim, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU(), nn.Linear(width, actions))
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size, obs_dim)
        self.gamma, self.target_update, self.no_target = gamma, target_update, no_target

    def action(self, obs, rng=None):
        with torch.no_grad():
            return int(self.online(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    def learn(self, batch_size, rng):
        obs, actions, rewards, next_obs, terminated = self.replay.sample(batch_size, rng)
        with torch.no_grad():
            bootstrap_net = self.online if self.no_target else self.target
            targets = rewards + self.gamma * (1 - terminated) * bootstrap_net(next_obs).max(1).values
        predictions = self.online(obs).gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
