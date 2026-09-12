"""Bounded sequential MPS planning checks; never launches another learner."""
import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from research.afterstate_search_experiment import run, write


def pilot(checkpoint, out, stop_file):
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json',dict(started_epoch=time.time(),checkpoint=str(checkpoint.resolve()),
        device='mps',games=8,seed_start=8910000,depths=[2,3],seconds_per_depth=600,
        reason='Full planner-call benchmark on seven fixed diagnostic boards favored MPS over CPU, mostly by about7x. Test complete games and a matched MPS depth2control before extrapolating speed or comparing policy strength. CPU/MPS symmetry ties change trajectories. One short inference study shares the GPU with the single learner; record contention, do not interpret timing as an isolated hardware benchmark.'))
    for depth in (2,3):
        if stop_file.exists():break
        write(out/'status.json',dict(depth=depth,phase='evaluating',updated_epoch=time.time()))
        run(SimpleNamespace(checkpoint=checkpoint,out=out/f'depth{depth}',games=8,
            seed_start=8910000,spawn_batch=256,depth=depth,device='mps',resume=False,
            seconds=600,stop_file=stop_file))
        gc.collect();torch.mps.empty_cache()
    results={str(d):json.loads((out/f'depth{d}/evaluation.json').read_text())
             for d in (2,3) if (out/f'depth{d}/evaluation.json').exists()}
    write(out/'results.json',results)
    write(out/'status.json',dict(phase='complete' if len(results)==2 and all(r['complete'] for r in results.values()) else 'saved_partial',updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stop-file',type=Path,required=True)
    a=p.parse_args();pilot(a.checkpoint,a.out,a.stop_file)
