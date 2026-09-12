"""Frozen-policy tests of reward omission, snake rules, and forbidding up."""
import json
from pathlib import Path
import numpy as np
from numba import njit
from concurrent.futures import ThreadPoolExecutor
from rl2048.fast2048 import slide,spawn,value
from rl2048.agents.ntuple import NTupleAgent,row_tables


@njit(cache=True,nogil=True)
def ablation_game(weights,patterns,rows,gains,seed,mode):
    np.random.seed(seed)
    board=np.zeros(16,np.uint8); spawn(board); spawn(board)
    priority=np.array([1,2,3,4,8,7,6,5,9,10,11,12,16,15,14,13])
    score=0
    for t in range(40_000):
        best=-1e30; chosen=-1; legal_count=0
        for action in range(4):
            after,reward,changed=slide(board,action,rows,gains)
            if not changed:
                continue
            legal_count+=1
            if mode=='never_up' and action==0:
                continue
            if mode in ('snake_bottom_left','snake_top_right'):
                estimate=0.
                for j in range(16):
                    position=j if mode=='snake_bottom_left' else 15-j
                    estimate+=(1<<int(after[j]))*priority[position]**3 if after[j] else 0
                estimate+=1000*np.sum(after==0)
            else:
                estimate=max(0.,value(after,weights,patterns))
                if mode!='value_only':
                    estimate+=reward
            if estimate>best:
                best,chosen=estimate,action
        if chosen<0:
            return score,t,1<<int(board.max()),legal_count>0
        board,reward,_=slide(board,chosen,rows,gains)
        spawn(board); score+=reward
    return score,40_000,1<<int(board.max()),True


if __name__=='__main__':
    agent=NTupleAgent.load('runs/research/native_otd',mmap_mode='r')
    rows,gains=row_tables()
    results={}
    for mode in ('reward_plus_value','value_only','never_up','snake_bottom_left','snake_top_right'):
        seeds=list(range(6_200_000,6_200_100))
        first=ablation_game(agent.weights,agent.patterns,rows,gains,seeds[0],mode)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(ablation_game,agent.weights,agent.patterns,rows,gains,s,mode) for s in seeds[1:]]
            outcomes=[first]+[f.result() for f in futures]
        result={'mean_score':float(np.mean([r[0] for r in outcomes])),
                'max_score':int(max(r[0] for r in outcomes)), 'max_tile':int(max(r[2] for r in outcomes)),
                'artificial_stops':sum(bool(r[3]) for r in outcomes), 'seeds':seeds,
                'episodes':[dict(seed=s,score=int(r[0]),length=int(r[1]),max_tile=int(r[2]),truncated=bool(r[3])) for s,r in zip(seeds,outcomes)]}
        results[mode]=result
        Path('runs/research/ablations.json').write_text(json.dumps(results,indent=2))
        print(mode,{k:v for k,v in result.items() if k not in ('seeds','episodes')},flush=True)
