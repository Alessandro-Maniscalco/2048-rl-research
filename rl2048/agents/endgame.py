"""Adapter for game-difficulty's compact search/embedded-table player.

The Python game remains exact at all tile ranks. Only the planner's observation
uses upstream's 4-bit abstraction: ranks >=15 become unmergeable placeholders.
Before the first65536, two32768tiles are downgraded in the SEARCH INPUT as in
upstream AItest, so the planner can choose their merge. No game tiles/rewards
are edited. The engine never receives the environment RNG or future spawn.
"""
from pathlib import Path
import sys
import time

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'third_party/2048EndgameTablebase'


def load_native():
    for folder in (ROOT/'runs/research/endgame_tablebase/build_deps',VENDOR):
        if str(folder) not in sys.path:sys.path.insert(0,str(folder))
    from native_core import ai_core
    from engine_core.AIPlayer import CoreAILogic
    return ai_core,CoreAILogic


def pack_ranks(ranks):
    values=np.asarray(ranks).reshape(16)
    if np.any(values<0) or np.any(values>15):raise ValueError('Packed ranks must be0..15')
    packed=0
    for rank in values:packed=(packed<<4)|int(rank)
    return packed


class EndgameAgent:
    name='external_compact_endgame'
    display_name='Compact endgame tables + adaptive CPU search'

    def __init__(self,time_scale=1.):
        if time_scale<=0:raise ValueError('Positive search-time multiplier required')
        self.core,logic=load_native()
        self.player=self.core.AIPlayer(0);self.player.max_threads=1
        self.player.update_spawn_rate(.1)
        self.logic=logic();self.logic.time_limit_ratio=time_scale
        self.time_scale=time_scale;self.rng=np.random.default_rng(0);self.last_decision={}

    def act(self,board,action_mask):
        if not np.asarray(action_mask).any():raise ValueError('No legal action')
        original=np.asarray(board)
        ranks=np.zeros(16,dtype=np.uint8);flat=original.reshape(16)
        ranks[flat>0]=np.log2(flat[flat>0]).astype(np.uint8)
        capped=np.minimum(ranks,15);packed=pack_ranks(capped)
        downgraded=False
        if ranks.max()<16:
            search_board=int(self.core.resolve_32768_doubles(packed))
            downgraded=search_board!=packed
        else:search_board=packed
        # Keep the upstream board/count representation consistent after a65536.
        observation=np.where(capped>0,1<<capped.astype(np.int64),0).reshape(4,4)
        counts=np.bincount(capped,minlength=16)
        self.player.reset_board(search_board)
        before=time.perf_counter()
        native_action=int(self.logic.calculate_step(self.player,observation,counts))
        action={1:3,2:1,3:0,4:2}.get(native_action,-1)
        if action<0 or not action_mask[action]:
            raise ValueError(f'Native planner returned illegal action{native_action} on an exact live board')
        self.last_decision=dict(action=action,native_action=native_action,mode=self.logic.last_move,
            depth=int(self.logic.last_depth),seconds=time.perf_counter()-before,
            nodes=int(self.player.node),search_scores=list(self.player.scores),
            high_rank_abstraction=bool(ranks.max()>15),two32768_search_downgrade=downgraded,
            search_board_hex=f'{search_board:016x}')
        return action
