"""Exact finite-horizon score and survival checks on a saved endgame board.

Enumerate every legal move and every possible spawn, with memoization. No
heuristic, sampled spawn, model, hidden environment RNG or discounted reward.
The score and survival objectives are optimized separately and may pick different
future policies. A hard node/time limit fails explicitly; no approximate answer
is silently returned as exact.
"""
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import write
from rl2048.agents.ntuple import row_tables
from rl2048.fast2048 import slide
from rl2048.game import ACTION_NAMES


class ExactEndgame:
    def __init__(self,max_nodes=500000,seconds=60):
        self.rows,self.rewards=row_tables()
        self.max_nodes=max_nodes;self.seconds=seconds;self.started=time.perf_counter();self.nodes=0
        # Per-instance caches avoid retaining finished diagnostic instances.
        self.legal=lru_cache(maxsize=None)(self._legal)
        self.value=lru_cache(maxsize=None)(self._value)

    def _legal(self,key):
        board=np.asarray(key,dtype=np.uint8)
        result=[]
        for action in range(4):
            after,reward,changed=slide(board,action,self.rows,self.rewards)
            if changed:result.append((action,int(reward),tuple(int(x) for x in after)))
        return tuple(result)

    @staticmethod
    def spawns(after):
        empty=[i for i,x in enumerate(after) if x==0]
        if not empty:raise ValueError('A valid move must leave an empty cell')
        for i in empty:
            for tile,p in [(1,.9),(2,.1)]:
                child=list(after);child[i]=tile
                yield tuple(child),p/len(empty)

    def _value(self,key,horizon):
        self.nodes+=1
        if self.nodes>self.max_nodes or time.perf_counter()-self.started>self.seconds:
            raise RuntimeError('Exact diagnostic budget exceeded')
        actions=self.legal(key)
        if not actions:return 0.,0.
        if horizon==0:return 0.,1.
        best_score=best_survival=0.
        for _,reward,after in actions:
            score=float(reward);survival=0.
            for child,p in self.spawns(after):
                next_score,next_survival=self.value(child,horizon-1)
                score+=p*next_score;survival+=p*next_survival
            best_score=max(best_score,score);best_survival=max(best_survival,survival)
        return best_score,best_survival

    def query(self,board,horizon):
        if horizon<1:raise ValueError('Positive horizon required')
        raw=np.asarray(board,dtype=np.int64).reshape(16)
        if np.any(raw<0) or np.any((raw!=0)&((raw&(raw-1))!=0)):raise ValueError('Power-of-two tiles required')
        ranks=np.zeros(16,dtype=np.uint8);ranks[raw>0]=np.log2(raw[raw>0]).astype(np.uint8)
        if ranks.max()>17:raise ValueError('Beyond validated tile rank')
        values={}
        for action,reward,after in self.legal(tuple(int(x) for x in ranks)):
            score=float(reward);survival=0.
            for child,p in self.spawns(after):
                v,q=self.value(child,horizon-1);score+=p*v;survival+=p*q
            values[ACTION_NAMES[action]]=dict(expected_additional_raw_score=score,
                probability_still_alive_after_h_moves=survival,immediate_merge_points=reward)
        return values


def main():
    base=Path('runs/research/endgame_tablebase')
    replay=json.loads((base/'pilot_seed8972000/replay.json').read_text())
    board=replay['frames'][-2]['board']
    solver=ExactEndgame();results=[];error=None
    for horizon in (1,2,3,4,5,6,8):
        try:values=solver.query(board,horizon)
        except RuntimeError as e:error=str(e);break
        results.append(dict(horizon=horizon,actions=values,nodes=solver.nodes,
            elapsed_seconds=time.perf_counter()-solver.started))
        print(json.dumps(results[-1]),flush=True)
    write(base/'pilot_seed8972000/exact_final_move.json',dict(board=board,
        original_action=ACTION_NAMES[replay['frames'][-1]['action']],results=results,
        requested_horizons=[1,2,3,4,5,6,8],complete=error is None,error=error,
        objective='Undiscounted additional raw merge points within h moves; separately maximize chance of remaining alive after h moves.',
        limitation='Different objectives can use different optimal future policies. Finite-horizon values do not establish full-game optimality. '
            'No heuristic leaves, learned value, future RNG or spawn sampling; probabilities enumerated exactly subject to ordinary floating point.'))


if __name__=='__main__':main()
