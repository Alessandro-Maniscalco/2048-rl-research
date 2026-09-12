import json
from pathlib import Path
from rl2048.agents.ntuple import NTupleAgent
from rl2048.ntuple_train import evaluate_fast
from rl2048.view import save_replay

if __name__=='__main__':
    for checkpoint in ['local_tc_numba/best','published/checkpoint']:
        agent=NTupleAgent.load(Path('runs/research')/checkpoint,mmap_mode='r')
        results={}
        for depth in ([1,2,3] if checkpoint.startswith('local') else [3]):
            summary,records=evaluate_fast(agent,range(6_000_000,6_000_100),depth=depth,workers=6)
            results[str(depth)]={'summary':summary,'episodes':records}
            (Path('runs/research')/checkpoint/'search_validation.json').write_text(json.dumps(results,indent=2))
            print(checkpoint,depth,{k:v for k,v in summary.items() if k!='seeds'},flush=True)
        agent.depth=3
        replay=Path('runs/research')/('local_best_replay.html' if checkpoint.startswith('local') else 'published_reference_replay.html')
        print('REPLAY',checkpoint,save_replay(agent,replay),flush=True)
