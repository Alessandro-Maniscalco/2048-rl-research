"""Compare frozen policies on 128 fixed seeds throughout online training.

python -m research.policy_curve_experiment --out runs/research/policy_curves
Active-board scores are logged after EVERY update, but are not policy quality:
boards differ in age and their scores reset when an episode ends.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rl2048.afterstate_compare import evaluate_batch
from rl2048.agents.ntuple import row_tables
from rl2048.vector_game import legal_masks
from research.transformer_td_experiment import config, name, train_one, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=262144)
    p.add_argument('--interval',type=int,default=32768)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    args=p.parse_args()
    if args.steps<4096 or args.steps%128 or args.interval<128 or args.interval%128:
        raise ValueError('Use >=4096 steps and positive checkpoint intervals divisible by128.')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    candidates=[config('mlp_q','exponents',1),config('transformer_q','exponents',1),config('transformer_q','exponents',3)]
    common=dict(monitor_games=128,monitor_interval=args.interval,monitor_initial=True,dense_metrics=True,
        save_monitor_checkpoints=True,monitor_seed_start=8690000,final_seed_start=8700000)
    rng=np.random.default_rng(42)
    def random_decision(b):
        return np.where(legal_masks(b,*row_tables()),rng.random((len(b),4)),-np.inf)
    baseline=evaluate_batch(None,range(8690000,8690128),decision=random_decision)
    write_json(args.out/'random_baseline.json',baseline|dict(policy_rng_seed=42))
    manifest=dict(status='running',steps=args.steps,interval=args.interval,training_seeds=args.seeds,
        monitor_games=128,monitor_seed_start=8690000,final_seed_start=8700000,
        description='Mean raw final score of frozen greedy policies on fixed complete games; live-board mean is a separate diagnostic.',
        cpu_gpu_execution='Sequential: GPU inference, CPU game step, replay assembly, GPU update. No overlapping actor thread.',
        device_selection='CPU MLP and MPS Transformer, using preceding local batch512 benchmark.',runs=[])
    write_json(args.out/'results.json',manifest)
    for c in candidates:
        device='mps' if c['architecture']=='transformer_q' and torch.backends.mps.is_available() else 'cpu'
        for seed in args.seeds:
            r=train_one(c|common,seed,args.steps,args.out/name(c)/f'seed{seed}',device)
            manifest['runs'].append(r)
            write_json(args.out/'results.json',manifest)
            print('RESULT',name(c),seed,r['mean_score'],flush=True)
    manifest['status']='finished'
    write_json(args.out/'results.json',manifest)


if __name__=='__main__':main()
