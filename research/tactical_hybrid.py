"""Candidate policy: inspect risky moves using exact eight-move raw returns.

Not used by the frozen100-game evaluation. Immediate risk only triggers the
calculation; it is not the objective or a prohibition. Preserve the original
choice on ties or budget exhaustion. A short horizon can still be shortsighted,
so this candidate requires a fresh paired game comparison before promotion.
"""
import time

import numpy as np

from research.exact_endgame_diagnostic import ExactEndgame
from rl2048.agents.hybrid_endgame import HybridEndgameAgent
from rl2048.agents.spawn_safety import immediate_death_risks
from rl2048.game import ACTION_NAMES


def tactical_choice(board,proposed,max_nodes=500000,seconds=30):
    raw=np.asarray(board)
    if np.count_nonzero(raw)<15:return proposed,None
    ranks=np.zeros((1,16),dtype=np.uint8);flat=raw.reshape(16)
    ranks[0,flat>0]=np.log2(flat[flat>0]).astype(np.uint8)
    risks,legal=immediate_death_risks(ranks)
    if not legal[0,proposed]:raise ValueError('Illegal original proposal')
    if risks[0,proposed]<=risks[0].min()+1e-12:return proposed,None
    solver=ExactEndgame(max_nodes=max_nodes,seconds=seconds)
    result=dict(proposed=proposed,horizon=8,risks=[float(r) if np.isfinite(r) else None for r in risks[0]],
                objective='Expected additional raw merge points over8moves, no heuristic leaf value')
    try:
        values=solver.query(raw,8)
        best=proposed;best_score=values[ACTION_NAMES[proposed]]['expected_additional_raw_score']
        for action,name in enumerate(ACTION_NAMES):
            if name in values and values[name]['expected_additional_raw_score']>best_score+1e-8:
                best=action;best_score=values[name]['expected_additional_raw_score']
        result.update(complete=True,values=values,selected=best,overrode=best!=proposed)
    except RuntimeError as error:
        best=proposed
        result.update(complete=False,error=str(error),selected=proposed,overrode=False)
    finally:
        result.update(nodes=solver.nodes,seconds=time.perf_counter()-solver.started)
        solver.legal.cache_clear();solver.value.cache_clear()
    return best,result


class TacticalHybridAgent(HybridEndgameAgent):
    name='hybrid_with_exact_tactical_check'
    display_name='Solved formations + compact search + exact tactical check'

    def act(self,board,action_mask):
        started=time.perf_counter()
        proposed=super().act(board,action_mask)
        action,check=tactical_choice(board,proposed)
        if check is not None:
            saved=self.last_decision.copy()
            self.last_decision.update(tactical=check)
            if action!=proposed:
                self.last_decision=dict(action=action,mode='exact-tactical',depth=8,nodes=0,
                    exact_dp_states=check['nodes'],search_scores=None,
                    high_rank_abstraction=False,two32768_search_downgrade=False,
                    proposal=saved,tactical=check)
        self.last_decision['seconds']=time.perf_counter()-started
        return action
