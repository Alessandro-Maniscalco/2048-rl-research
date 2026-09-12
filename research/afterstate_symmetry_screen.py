"""Can eight views of the SAME frozen scalar network improve action ranking?

This averages afterstate values over all rotations/reflections. It changes only
inference; it supplies no corner/snake features and does not augment training.
Always retain the original checkpoint and this inference configuration together.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from rl2048.agents.afterstate_mlp import AfterstateMLPAgent
from rl2048.afterstate_compare import evaluate_batch
from rl2048.view import save_replay


class SymmetryAverage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model
        cells=np.arange(16).reshape(4,4)
        self.register_buffer('permutations',torch.tensor(np.stack(
            [np.rot90(cells,k).ravel() for k in range(4)]+
            [np.rot90(np.fliplr(cells),k).ravel() for k in range(4)])))

    @torch.no_grad()
    def forward(self, boards):
        views=boards.reshape(-1,16)[:,self.permutations].reshape(-1,16)
        return self.model(views).reshape(-1,8).mean(1)


def run(args):
    device=getattr(args,'device','cpu')
    reference=getattr(args,'identity_reference',None)
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(args.checkpoint,args.out/'frozen')
    torch.set_num_threads(1)
    config=dict(source=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(
        (args.out/'frozen/policy.pt').read_bytes()).hexdigest(),games=100,seed_start=8910000,
        device=device,torch_threads=1,seconds_per_variant=args.seconds,
        hypothesis='The score objective and game physics are rotation/reflection symmetric. Averaging one frozen afterstate model over8equivalentviews may reduce orientation-specific value errors. It may also dilute useful learned patterns, so a gain is not assumed. This is an inference ensemble, not the earlier training-augmentation experiment.',
        comparison='Matched-device identity control versus8-view mean on the same100selectionseeds, same frozen leaf and root-only action scoring. No extra chance search, new data or trained weights; inference costs differ.')
    cached_identity=None
    if reference is not None:
        source_config=json.loads((reference.parent/'config.json').read_text())
        digest=hashlib.sha256((reference.parent/'last/policy.pt').read_bytes()).hexdigest()
        cached_identity=json.loads(reference.read_text())
        episodes=cached_identity['episodes']
        if (source_config['device']!=device or digest!=config['checkpoint_sha256']
                or not cached_identity['summary']['complete'] or len(episodes)!=100
                or sorted(e['seed'] for e in episodes)!=list(range(8910000,8910100))
                or not all(e['complete'] and not e['truncated'] for e in episodes)):
            raise ValueError('Cached identity requires identical frozen LAST weights, device and complete100seed suite')
        config['identity_reference']=str(reference.resolve())
        config['timing_note']='Cached identity timing comes from its original run; concurrent loads may differ.'
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    results={}
    for mode in ('identity','d4_mean'):
        if args.stop_file.exists(): break
        agent=AfterstateMLPAgent.load(args.out/'frozen',device)
        if mode=='d4_mean':
            agent.learner.policy=SymmetryAverage(agent.learner.policy).to(device)
            agent.display_name+=' · eight-view value average'
        def decide(boards):
            if args.stop_file.exists(): raise InterruptedError('User stop')
            return agent.learner.decision(boards)[0]
        try:
            result=cached_identity if mode=='identity' and cached_identity is not None else evaluate_batch(
                None,range(8910000,8910100),decision=decide,deadline=time.time()+args.seconds)
        except InterruptedError: break
        results[mode]=result
        (args.out/f'{mode}.json').write_text(json.dumps(result,indent=2))
        (args.out/'results.json').write_text(json.dumps(results,indent=2))
        print(mode,json.dumps(result['summary']),flush=True)
        if result['summary']['complete'] and not args.stop_file.exists():
            save_replay(agent,args.out/f'replay_{mode}.html',seed=8930100)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--identity-reference',type=Path,
        help='Reuse an identical completed LAST-policy evaluation; weights, device and seeds are verified')
    parser.add_argument('--stop-file',type=Path,required=True)
    run(parser.parse_args())
