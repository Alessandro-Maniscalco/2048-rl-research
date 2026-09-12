"""Use frozen solved formations, then the compact engine for other positions."""
import time

import numpy as np

from rl2048.agents.endgame import EndgameAgent
from rl2048.agents.frozen_tablebase import FrozenTablebase


class HybridEndgameAgent(EndgameAgent):
    name='frozen_tables_then_compact'
    display_name='Frozen solved formations + compact adaptive search'

    def __init__(self,time_scale=1.):
        super().__init__(time_scale)
        self.tables=FrozenTablebase()

    def act(self,board,action_mask):
        started=time.perf_counter()
        recommendation=self.tables.query(board)
        if recommendation['kind']:
            action=recommendation['action']
            if not action_mask[action]:raise ValueError('Frozen table recommended an illegal move')
            self.last_decision=dict(action=action,mode=f"frozen-{recommendation['kind']}",
                depth=0,seconds=time.perf_counter()-started,nodes=0,search_scores=None,
                high_rank_abstraction=False,two32768_search_downgrade=False,
                table_probability=recommendation['probability'],
                table_queries=1,table_hit=True)
            return action
        action=super().act(board,action_mask)
        self.last_decision['table_queries']=1
        self.last_decision['table_hit']=False
        self.last_decision['seconds']=time.perf_counter()-started
        return action
