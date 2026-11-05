"""Aggregate only completed runs; no synthetic results or extrapolated curves."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=['#2166ac','#d6604d','#4d9221','#762a83','#e08214']
LABELS={'cartpole_q_6':'Q-learning: 6 bins','cartpole_q_10':'Q-learning: 10 bins','cartpole_q_20':'Q-learning: 20 bins','cartpole_dqn':'DQN','acrobot_default':'DQN base','acrobot_lr_0.0003':'lr=0.0003','acrobot_lr_0.003':'lr=0.003','acrobot_epsilon_0.25':'Exploration 25%','acrobot_epsilon_0.8':'Exploration 80%','acrobot_no_target':'Without target network'}


def load_csv(path):
    with path.open() as f: return list(csv.DictReader(f))


def main():
    root=Path('results'); figs=root/'figures';figs.mkdir(exist_ok=True)
    configs=json.loads(Path('experiments/configs.json').read_text())
    data={}; rows=[]
    for cfg in configs:
        runs=[]
        for seed in [0,1,2]:
            p=root/cfg['name']/f'seed_{seed}'
            if not (p/'summary.json').exists(): continue
            s=json.loads((p/'summary.json').read_text());runs.append((p,s))
        if not runs: continue
        data[cfg['name']]=runs
        vals=np.array([s['final_mean'] for _,s in runs]); wall=np.array([s['wall_seconds'] for _,s in runs])
        reached=[s['first_validation_threshold_step'] for _,s in runs]
        aucs=[]; early=[]
        for p,_ in runs:
            curve=load_csv(p/'evaluation.csv')
            xs=np.array([int(r['step']) for r in curve]); ys=np.array([float(r['mean_return']) for r in curve])
            aucs.append(float(np.trapezoid(ys,xs)/xs[-1]))
            early.append(float(ys[1]))
        rows.append(dict(config=cfg['name'],seeds=len(runs),final_mean=float(vals.mean()),
             final_sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.,final_variance=float(vals.var(ddof=1)) if len(vals)>1 else 0.,
             evaluation_auc_mean=float(np.mean(aucs)),evaluation_auc_sd=float(np.std(aucs,ddof=1)) if len(aucs)>1 else 0.,
             first_10k_mean=float(np.mean(early)),wall_mean_seconds=float(wall.mean()),wall_total_seconds=float(wall.sum()),
             training_mean_seconds=float(np.mean([s['training_seconds'] for _,s in runs])),
             final_threshold_seeds=sum(s['threshold_passed'] for _,s in runs),
             first_validation_threshold_steps=reached,
             final_seed_means=vals.tolist()))
    (root/'aggregate.json').write_text(json.dumps(rows,indent=2))
    if rows:
        with (root/'aggregate.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    groups={
      'tier1_evaluation':['cartpole_q_6','cartpole_q_10','cartpole_q_20','cartpole_dqn'],
      'tier2_learning_rate':['acrobot_lr_0.0003','acrobot_default','acrobot_lr_0.003'],
      'tier2_exploration':['acrobot_epsilon_0.25','acrobot_default','acrobot_epsilon_0.8'],
      'tier2_ablation':['acrobot_default','acrobot_no_target']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for name, keys in groups.items():
        fig,ax=plt.subplots(figsize=(7.2,4.1))
        for color,key in zip(COLORS,keys):
            if key not in data: continue
            curves=[load_csv(p/'evaluation.csv') for p,_ in data[key]]
            x=np.array([int(r['step']) for r in curves[0]])
            if not all([int(r['step']) for r in c]==list(x) for c in curves): raise ValueError('Unequal evaluation grids')
            y=np.array([[float(r['mean_return']) for r in c] for c in curves])
            mean=y.mean(0);sd=y.std(0,ddof=1) if len(y)>1 else np.zeros_like(mean)
            ax.plot(x,mean,label=LABELS[key],color=color);ax.fill_between(x,mean-sd,mean+sd,alpha=.15,color=color)
        threshold=475 if name.startswith('tier1') else -100
        ax.axhline(threshold,color='gray',ls='--',label='Success threshold')
        ax.set(xlabel='Training environment steps',ylabel='Greedy evaluation return (10 episodes)')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figs/(name+'.png'),dpi=180);plt.close(fig)
    # Training curves are separate and include exploration. Fixed 10k-step bins.
    for group in ['tier1','tier2']:
        keys=groups['tier1_evaluation'] if group=='tier1' else ['acrobot_default','acrobot_no_target']
        fig,ax=plt.subplots(figsize=(7.2,4.1))
        edges=np.arange(0,200001,10000);centers=(edges[1:]+edges[:-1])/2
        for color,key in zip(COLORS,keys):
            if key not in data: continue
            ys=[]
            for p,_ in data[key]:
                c=load_csv(p/'training.csv');steps=np.array([int(r['step']) for r in c]);ret=np.array([float(r['return']) for r in c])
                ys.append([ret[(steps>lo)&(steps<=hi)].mean() if np.any((steps>lo)&(steps<=hi)) else np.nan for lo,hi in zip(edges[:-1],edges[1:])])
            ys=np.array(ys);mean=np.nanmean(ys,axis=0);sd=np.nanstd(ys,axis=0,ddof=1)
            ax.plot(centers,mean,label=LABELS[key],color=color);ax.fill_between(centers,mean-sd,mean+sd,alpha=.15,color=color)
        ax.set(xlabel='Training environment steps',ylabel='Training return (10k-step bins)');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figs/(group+'_training.png'),dpi=180);plt.close(fig)
    summary=['# Resultados medidos', '', 'Media final: 100 episodios por agente, tres semillas por configuración. ± es la desviación estándar entre las tres medias. Éxito es el número de semillas que alcanzan el umbral final.', '', '| Configuración | Semillas | Retorno final ± DE | Éxito | Tiempo medio (s) |', '|---|---:|---:|---:|---:|']
    for r in rows:
        summary.append(f"| {r['config']} | {r['seeds']} | {r['final_mean']:.2f} ± {r['final_sd']:.2f} | {r['final_threshold_seeds']}/{r['seeds']} | {r['wall_mean_seconds']:.1f} |")
    summary += ['', 'Los cruces periódicos de umbral usan 10 episodios y se registran aparte en `aggregate.json`. No certifican por sí solos éxito final. Los tiempos son de ejecuciones que comparten una cuota de dos CPU. Véase el informe para interpretar resultados negativos y sensibilidad.', '', 'Estas cifras se generan automáticamente desde los registros; no se seleccionó el mejor checkpoint.']
    (root/'RESUMEN.md').write_text('\n'.join(summary)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
