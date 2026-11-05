"""Record a typical final policy (seed nearest median final score), one fixed episode."""
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import imageio.v2 as imageio
from agents.tabular import TabularAgent
from agents.dqn import DQNAgent


def main():
    torch.set_num_threads(1)
    configs=json.loads(Path('experiments/configs.json').read_text())
    output=Path('results/videos');output.mkdir(exist_ok=True)
    metadata=[]
    for name in ['cartpole_q_10','cartpole_dqn','acrobot_default']:
        cfg=next(c for c in configs if c['name']==name)
        runs=[]
        for seed in [0,1,2]:
            p=Path('results')/name/f'seed_{seed}'
            if (p/'summary.json').exists():runs.append((seed,json.loads((p/'summary.json').read_text())['final_mean']))
        if len(runs)!=3: continue
        median=np.median([r for _,r in runs]);seed=min(runs,key=lambda sr:abs(sr[1]-median))[0]
        p=Path('results')/name/f'seed_{seed}'
        env=gym.make(cfg['env'],render_mode='rgb_array');obs,_=env.reset(seed=800000)
        if cfg['algorithm']=='tabular':
            a=TabularAgent(4,2,bins=cfg['bins']);a.q=np.load(p/'q_table.npy')
        else:
            a=DQNAgent(env.observation_space.shape[0],env.action_space.n,width=cfg['width']);a.online.load_state_dict(torch.load(p/'model.pt',weights_only=True))
        frames=[];total=0;step=0
        while True:
            if step%2==0:frames.append(env.render()[::2,::2])
            obs,r,t,tr,_=env.step(a.action(obs));total+=r;step+=1
            if t or tr:break
        imageio.mimsave(output/(name+'.gif'),frames,duration=round(2000/env.metadata['render_fps']),loop=0)
        metadata.append(dict(config=name,training_seed=seed,video_seed=800000,return_=total,selection='Training seed closest to median final mean; video episode is illustrative, not evaluation evidence.'))
        env.close()
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(metadata)

if __name__=='__main__':main()
