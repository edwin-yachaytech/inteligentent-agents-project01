"""Run a named group with at most two workers; resume completed runs."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from experiments.train import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=['all', 'tier1', 'tier2'], default='all')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = p.parse_args()
    configs = json.loads(Path('experiments/configs.json').read_text())
    selected = [c for c in configs if args.group == 'all' or c['group'] == args.group]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, c, s) for c in selected for s in args.seeds]
        for job in as_completed(jobs):
            job.result()

if __name__ == '__main__':
    main()
