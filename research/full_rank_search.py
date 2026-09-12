"""ctypes access to the isolated full-rank heuristic search; never spawns tiles."""
import ctypes
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIBRARY=ROOT/'runs/research/endgame_tablebase/full_rank_bridge/libfull_rank_search.dylib'


def encode(board):
    raw=np.asarray(board,dtype=np.int64)
    if raw.shape!=(4,4) or np.any(raw<0) or np.any((raw!=0)&((raw&(raw-1))!=0)):
        raise ValueError('A4x4board of power-of-two tiles or zeros is required')
    flat=raw.reshape(16);ranks=np.zeros(16,dtype=np.int32)
    ranks[flat>0]=np.log2(flat[flat>0]).astype(np.int32)
    if ranks.max()>17:raise ValueError('Input exceeds validated rank17')
    return ranks


class FullRankSearch:
    def __init__(self):
        self.lib=ctypes.CDLL(str(LIBRARY))
        pointer=ctypes.POINTER(ctypes.c_int)
        self.lib.full_rank_query.argtypes=[pointer,ctypes.c_int,pointer,pointer]
        self.lib.full_rank_query.restype=ctypes.c_int
        self.lib.full_rank_slide.argtypes=[pointer,ctypes.c_int,pointer,pointer]
        self.lib.full_rank_slide.restype=ctypes.c_int

    def query(self,board,depth=8):
        ranks=encode(board);action=ctypes.c_int(-1);value=ctypes.c_int()
        code=self.lib.full_rank_query(ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     depth,ctypes.byref(action),ctypes.byref(value))
        if code<0:raise ValueError('Native query rejected board or depth')
        return dict(action=action.value,heuristic_value=value.value,has_move=bool(code),depth=depth)

    def slide(self,board,action):
        ranks=encode(board);output=np.zeros(16,dtype=np.int32);reward=ctypes.c_int()
        code=self.lib.full_rank_slide(ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),action,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),ctypes.byref(reward))
        if code<0:raise ValueError('Native move rejected board or direction')
        board=np.where(output>0,1<<output.astype(np.int64),0).reshape(4,4)
        return board,reward.value,bool(code)
