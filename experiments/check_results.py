"""Validate that submission metrics match raw logs and the planned protocol."""
import csv,json
from pathlib import Path
import numpy as np


def main():
    configs=json.loads(Path('experiments/configs.json').read_text())
    aggregate={r['config']:r for r in json.loads(Path('results/aggregate.json').read_text())}
    count=0
    for cfg in configs:
        final_means=[]
        for seed in [0,1,2]:
            p=Path('results')/cfg['name']/f'seed_{seed}'
            saved=json.loads((p/'config.json').read_text())
            assert saved==dict(cfg,seed=seed),(cfg['name'],seed,'config mismatch')
            summary=json.loads((p/'summary.json').read_text())
            final=json.loads((p/'final_returns.json').read_text())
            assert len(final)==100
            assert np.isclose(summary['final_mean'],np.mean(final))
            with (p/'evaluation.csv').open() as f:rows=list(csv.DictReader(f))
            assert [int(r['step']) for r in rows]==list(range(0,200001,10000))
            assert all(int(r['episodes'])==10 for r in rows)
            with (p/'training.csv').open() as f:training=list(csv.DictReader(f))
            assert sum(int(r['length']) for r in training)<=200000
            assert max(int(r['step']) for r in training)<=200000
            assert np.isfinite(final).all()
            assert (p/('q_table.npy' if cfg['algorithm']=='tabular' else 'model.pt')).exists()
            final_means.append(summary['final_mean']);count+=1
        assert aggregate[cfg['name']]['seeds']==3
        assert np.isclose(aggregate[cfg['name']]['final_mean'],np.mean(final_means))
        assert np.isclose(aggregate[cfg['name']]['final_sd'],np.std(final_means,ddof=1))
    print(f'PASS: {count} runs; exact configs; 21 evaluation checkpoints each; 100 final episodes; matching aggregate metrics.')

if __name__=='__main__':main()
