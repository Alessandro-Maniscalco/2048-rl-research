"""A predeclared held-out comparison; no training or parameter selection here."""
import hashlib
import json
from pathlib import Path
import torch
from rl2048.agents.neural import NeuralAgent
from rl2048.evaluate import evaluate

root=Path('runs/research')
names=['dqn16_exponents','dqn16_control_seed1','dqn16_control_seed2',
       'dqn16_corner_snake','dqn16_winner_seed1','dqn16_winner_seed2',
       'dueling16_seed0','dueling16_seed1','dueling16_seed2',
       'dqn16_gamma1','dqn16_gamma1_seed1','dqn16_gamma1_seed2',
       'dqn16_long','ppo_cpu300','ppo_cnn300','bc_expert_cnn_symmetry',
       'awr_expert_cnn_symmetry','iql_expert_cnn_symmetry']
if __name__=='__main__':
    torch.set_num_threads(1)
    # Record every checkpoint before observing any held-out result.
    manifest={name:hashlib.sha256((root/name/'last/policy.pt').read_bytes()).hexdigest() for name in names}
    (root/'neural_heldout_manifest.json').write_text(json.dumps({'checkpoints':manifest,
        'seeds':list(range(7300000,7300100)), 'selection':'All listed variants frozen before this test'},indent=2))
    results={}
    for name in names:
        folder=root/name
        agent=NeuralAgent.load(folder/'last')
        summary,episodes=evaluate(agent,seeds=range(7300000,7300100),max_steps=40000)
        summary['max_score']=max(row['score'] for row in episodes)
        data={'summary':summary,'episodes':episodes,'split':'held-out test','policy_sha256':manifest[name]}
        (folder/'test.json').write_text(json.dumps(data,indent=2))
        results[name]=summary
        (root/'neural_heldout.json').write_text(json.dumps(results,indent=2))
        print(name,summary['mean_score'],summary['tile_reaching_rates']['2048'],flush=True)
