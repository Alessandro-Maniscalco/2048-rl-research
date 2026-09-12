"""One frozen final selection, one new test suite, and a reproducible replay."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from rl2048.agents.neural import NeuralAgent
from rl2048.evaluate import evaluate
from rl2048.view import save_replay

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    selection=json.loads((a.out/'selection.json').read_text())
    assert hashlib.sha256((a.run/'policy.pt').read_bytes()).hexdigest()==selection['sha256']
    if (a.out/'final_test.json').exists():
        raise RuntimeError('The held-out suite was already consumed; do not retest or tune against it.')
    agent=NeuralAgent.load(a.run)
    summary,episodes=evaluate(agent,seeds=selection['seeds'],max_steps=40000)
    summary['max_score']=max(r['score'] for r in episodes)
    (a.out/'final_test.json').write_text(json.dumps(dict(summary=summary,episodes=episodes,selection=selection),indent=2))
    save_replay(agent,a.out/'replay.html',seed=max(episodes,key=lambda r:r['score'])['seed'])
    print(json.dumps(summary),flush=True)
