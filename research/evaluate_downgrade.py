import json
from pathlib import Path
from rl2048.agents.ntuple import NTupleAgent
from rl2048.ntuple_train import evaluate_fast
from rl2048.view import save_replay

if __name__=='__main__':
    result={}
    for checkpoint in ('native8_otd','published/checkpoint'):
        agent=NTupleAgent.load(Path('runs/research')/checkpoint,mmap_mode='r')
        settings=[(1,0),(2,0),(2,16384)] if checkpoint=='native8_otd' else [(2,32768)]
        for depth,threshold in settings:
            agent.downgrade_threshold=threshold
            summary,episodes=evaluate_fast(agent,range(6_000_000,6_000_100),depth=depth,workers=4)
            result[checkpoint+f'/depth{depth}/downgrade{threshold}']={'summary':summary,'episodes':episodes}
            Path('runs/research/downgrade_validation.json').write_text(json.dumps(result,indent=2))
            print(checkpoint,depth,threshold,{k:v for k,v in summary.items() if k!='seeds'},flush=True)
        if checkpoint.startswith('published'):
            agent.depth=2;agent.display_name='Published pretrained reference · depth 2 + tile downgrading'
            print('REPLAY',save_replay(agent,'runs/research/published_downgrade_replay.html'),flush=True)
