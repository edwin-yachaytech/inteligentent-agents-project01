"""One reproducible run. Evaluation never consumes the training RNG or env."""
import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from agents.tabular import TabularAgent
from agents.dqn import DQNAgent


def evaluate(agent, env_name, seeds):
    env = gym.make(env_name)
    returns = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        total = 0.
        while True:
            obs, reward, terminated, truncated, _ = env.step(agent.action(obs))
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return returns


def run(config, seed, root='results', steps_override=None):
    cfg = dict(config)
    if steps_override is not None:
        cfg['steps'] = steps_override
    out = Path(root) / cfg['name'] / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        saved_config = json.loads((out / 'config.json').read_text())
        if saved_config != dict(cfg, seed=seed):
            raise ValueError(f'{out}: existing run has a different config; choose a new --root.')
        return json.loads((out / 'summary.json').read_text())
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    env = gym.make(cfg['env'])
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    dim, actions = env.observation_space.shape[0], env.action_space.n
    if cfg['algorithm'] == 'tabular':
        agent = TabularAgent(dim, actions, bins=cfg['bins'])
    else:
        agent = DQNAgent(dim, actions, **{k: cfg[k] for k in
            ['lr', 'width', 'gamma', 'buffer_size', 'target_update', 'no_target'] if k in cfg})
    (out / 'config.json').write_text(json.dumps(dict(cfg, seed=seed), indent=2))
    start = time.perf_counter()
    training_seconds = 0.
    eval_seconds = 0.
    threshold = 475 if cfg['env'] == 'CartPole-v1' else -100
    reached = None
    with (out / 'training.csv').open('w', buffering=1) as tf, (out / 'evaluation.csv').open('w', buffering=1) as ef:
        tw, ew = csv.writer(tf), csv.writer(ef)
        tw.writerow(['step', 'episode', 'return', 'length', 'epsilon'])
        ew.writerow(['step', 'mean_return', 'std_return', 'episodes', 'elapsed_seconds'])
        def log_eval(step):
            nonlocal eval_seconds, reached
            t = time.perf_counter()
            # Fixed validation seeds across checkpoints, disjoint from training/final seeds.
            vals = evaluate(agent, cfg['env'], [100000 + seed * 1000 + j for j in range(10)])
            eval_seconds += time.perf_counter() - t
            ew.writerow([step, np.mean(vals), np.std(vals, ddof=1), 10, time.perf_counter()-start])
            if step and np.mean(vals) >= threshold and reached is None:
                reached = step
        log_eval(0)
        episode, total, length = 0, 0., 0
        for step in range(1, cfg['steps'] + 1):
            t = time.perf_counter()
            decay_steps = max(1, int(cfg['steps'] * cfg.get('epsilon_fraction', 0.5)))
            epsilon = 1. + min(1., (step - 1) / decay_steps) * (cfg.get('epsilon_end', 0.05) - 1.)
            action = int(rng.integers(actions)) if rng.random() < epsilon else agent.action(obs, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if cfg['algorithm'] == 'tabular':
                agent.update(obs, action, reward, next_obs, terminated)
            else:
                agent.replay.add(obs, action, reward, next_obs, terminated)
                if step >= cfg.get('learning_starts', 1000) and step % cfg.get('train_frequency', 4) == 0:
                    agent.learn(cfg.get('batch_size', 64), rng)
                if not agent.no_target and step % agent.target_update == 0:
                    agent.sync_target()
            total += reward
            length += 1
            obs = next_obs
            if terminated or truncated:
                episode += 1
                tw.writerow([step, episode, total, length, epsilon])
                obs, _ = env.reset()
                total, length = 0., 0
            training_seconds += time.perf_counter() - t
            if step % cfg['eval_interval'] == 0 or step == cfg['steps']:
                log_eval(step)
        t = time.perf_counter()
        final = evaluate(agent, cfg['env'], [900000 + seed * 1000 + j for j in range(100)])
        eval_seconds += time.perf_counter() - t
    if cfg['algorithm'] == 'tabular':
        np.save(out / 'q_table.npy', agent.q)
    else:
        torch.save(agent.online.state_dict(), out / 'model.pt')
    (out / 'final_returns.json').write_text(json.dumps(final))
    summary = dict(config=cfg['name'], seed=seed, steps=cfg['steps'],
                   final_mean=float(np.mean(final)), final_std_episodes=float(np.std(final, ddof=1)),
                   threshold=threshold, threshold_passed=bool(np.mean(final) >= threshold),
                   first_validation_threshold_step=reached, training_seconds=training_seconds,
                   evaluation_seconds=eval_seconds, wall_seconds=time.perf_counter()-start,
                   hardware=dict(platform=platform.platform(), cpu=platform.processor(),
                                 logical_cpus=os.cpu_count(), torch_threads=1, device='cpu',
                                 cpu_quota='2 cores', memory_limit_gib=8),
                   versions=dict(python=platform.python_version(), gymnasium=gym.__version__,
                                 torch=torch.__version__, numpy=np.__version__))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    env.close()
    print(json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int)
    p.add_argument('--root', default='results')
    args = p.parse_args()
    configs = json.loads(Path('experiments/configs.json').read_text())
    run(next(c for c in configs if c['name'] == args.config), args.seed, args.root, args.steps)

if __name__ == '__main__':
    main()
