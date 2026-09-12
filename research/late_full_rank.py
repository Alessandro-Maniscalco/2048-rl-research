"""Use full-rank search after 65,536, keeping frozen table recommendations first."""
import time

import numpy as np

from research.full_rank_search import FullRankSearch
from research.work_budget_hybrid import WorkBudgetHybridAgent


class LateFullRankAgent(WorkBudgetHybridAgent):
    name='hybrid_with_full_rank_late_search'
    display_name='Frozen formations + full-rank late search'

    def __init__(self,work_target=4000000,depth=8):
        super().__init__(work_target)
        self.full_rank_depth=depth
        self.full_rank=None

    def act(self,board,action_mask):
        if np.max(board)<65536:
            return super().act(board,action_mask)
        started=time.perf_counter()
        recommendation=self.tables.query(board)
        if recommendation['kind']:
            action=recommendation['action']
            if not action_mask[action]:raise ValueError('Illegal frozen-table recommendation')
            self.last_decision=dict(action=action,mode=f"frozen-{recommendation['kind']}",
                depth=0,nodes=0,search_scores=None,high_rank_abstraction=False,
                two32768_search_downgrade=False,table_probability=recommendation['probability'],
                table_queries=1,table_hit=True,work_target=self.work_target,work_calls=[],
                summed_iterative_layer_nodes=0,seconds=time.perf_counter()-started)
            return action
        if self.full_rank is None:self.full_rank=FullRankSearch()
        answer=self.full_rank.query(board,self.full_rank_depth)
        action=answer['action']
        if not answer['has_move'] or not action_mask[action]:
            raise ValueError('Full-rank search returned no legal move')
        self.last_decision=dict(action=action,mode='full-rank-search',depth=self.full_rank_depth,
            nodes=0,node_count_available=False,search_scores=None,
            heuristic_value=answer['heuristic_value'],high_rank_abstraction=False,
            two32768_search_downgrade=False,table_queries=1,table_hit=False,
            work_target=None,work_calls=[],summed_iterative_layer_nodes=0,
            seconds=time.perf_counter()-started)
        return action
