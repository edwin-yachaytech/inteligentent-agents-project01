"""Q-learning with finite, explicitly bounded CartPole discretization."""
import numpy as np


class TabularAgent:
    def __init__(self, obs_dim, actions, bins=10, alpha=0.15, gamma=0.99):
        if obs_dim != 4:
            raise ValueError('This discretizer is defined only for CartPole.')
        # Position/angle cover the failure region; velocities are clipped.
        self.bounds = np.array([[-2.4, 2.4], [-3., 3.], [-0.2095, 0.2095], [-3.5, 3.5]])
        self.edges = [np.linspace(low, high, bins + 1)[1:-1] for low, high in self.bounds]
        self.q = np.zeros((bins,) * obs_dim + (actions,), dtype=np.float32)
        self.alpha, self.gamma = alpha, gamma

    def state(self, obs):
        return tuple(int(np.digitize(x, edges)) for x, edges in zip(obs, self.edges))

    def action(self, obs, rng=None):
        values = self.q[self.state(obs)]
        if rng is None:
            return int(np.argmax(values))
        # Random tie breaking prevents initial action bias during training.
        return int(rng.choice(np.flatnonzero(values == values.max())))

    def update(self, obs, action, reward, next_obs, terminated):
        index = self.state(obs) + (action,)
        target = reward + self.gamma * (not terminated) * self.q[self.state(next_obs)].max()
        self.q[index] += self.alpha * (target - self.q[index])
