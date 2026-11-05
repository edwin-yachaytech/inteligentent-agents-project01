"""Small numerical checks for Bellman targets and replay semantics."""
import unittest
import numpy as np
import torch
from agents.tabular import TabularAgent
from agents.dqn import DQNAgent, ReplayBuffer


class AgentTests(unittest.TestCase):
    def test_tabular_terminal_and_time_limit(self):
        a = TabularAgent(4, 2, bins=6, alpha=1., gamma=.9)
        obs, nxt = np.array([-1.,0.,0.,0.]), np.array([1.,0.,0.,0.])
        a.q[a.state(nxt)] = [10., 5.]
        a.update(obs, 0, 1., nxt, True)
        self.assertAlmostEqual(float(a.q[a.state(obs)+(0,)]), 1.)
        a.update(obs, 0, 1., nxt, False)
        self.assertAlmostEqual(float(a.q[a.state(obs)+(0,)]), 10.)
        self.assertTrue(all(0 <= i < 6 for i in a.state([1e6]*4)))

    def test_replay_wrap(self):
        b = ReplayBuffer(2, 1)
        for i in range(3): b.add([i], 0, i, [i+1], False)
        self.assertEqual(b.size, 2)
        self.assertEqual(set(b.rewards.tolist()), {1., 2.})

    def test_dqn_target_frozen_then_sync(self):
        torch.manual_seed(0)
        a = DQNAgent(4, 2)
        before = [p.clone() for p in a.target.parameters()]
        for _ in range(64): a.replay.add(np.zeros(4), 0, 1., np.ones(4), True)
        self.assertTrue(np.isfinite(a.learn(64, np.random.default_rng(0))))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(before,a.target.parameters())))
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a.online.parameters(),a.target.parameters())))
        a.sync_target()
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.online.parameters(),a.target.parameters())))

    def test_dqn_terminal_mask_and_detached_ablation(self):
        torch.set_num_threads(1)
        for terminal, expected in [(True, .5), (False, 10.4)]:
            a = DQNAgent(4, 2, lr=0.)
            with torch.no_grad():
                for net in [a.online, a.target]:
                    for param in net.parameters(): param.zero_()
                a.target[-1].bias.fill_(10.)
            a.replay.add(np.zeros(4), 0, 1., np.zeros(4), terminal)
            loss = a.learn(8, np.random.default_rng(0))
            self.assertAlmostEqual(loss, expected, places=4)
        a = DQNAgent(4, 2, no_target=True)
        def forbidden(*args): raise AssertionError('Ablation must not use target network')
        a.target.forward = forbidden
        a.replay.add(np.zeros(4), 0, 1., np.ones(4), False)
        self.assertTrue(np.isfinite(a.learn(8, np.random.default_rng(0))))

if __name__ == '__main__': unittest.main()
