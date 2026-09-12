"""A declared score-search set, separate from final held-out evaluations.

The best replay is selected from 1,000 games and is never presented as typical.
"""
import json
from pathlib import Path
from rl2048.agents.ntuple import NTupleAgent
from rl2048.ntuple_train import evaluate_fast
from rl2048.fast_replay import save_fast_replay

if __name__=='__main__':
    for checkpoint,label,threshold in [('native8_otd','local',16384),('published/checkpoint','published',32768)]:
        agent=NTupleAgent.load(Path('runs/research')/checkpoint,mmap_mode='r')
        agent.depth=2;agent.downgrade_threshold=threshold
        agent.display_name=('Locally trained 8×6 TD' if label=='local' else 'Published pretrained reference')+' · search + tile downgrading'
        summary,episodes=evaluate_fast(agent,range(10_000_000,10_001_000),workers=8)
        result={'summary':summary,'episodes':episodes,'selection':'Score search: maximum selected from 1000 games','provenance':agent.metadata}
        Path(f'runs/research/{label}_score_search.json').write_text(json.dumps(result,indent=2))
        best=max(episodes,key=lambda row:row['score'])
        print(label,{k:v for k,v in summary.items() if k!='seeds'},'BEST',best,flush=True)
        recorded=save_fast_replay(agent,f'runs/research/{label}_record_1000_replay.html',best['seed'])
        assert recorded['score']==best['score']
        print('REPLAY',recorded,flush=True)
