"""Read-only solved-subgoal recommendations from the existing MacroXue tables."""
import ctypes
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
LIBRARY=ROOT/'runs/research/endgame_tablebase/frozen_bridge/libfrozen_tables.dylib'
CACHE=ROOT/'runs/research/tablebase_search/cache'


class FrozenTablebase:
    def __init__(self,cache=CACHE):
        self.lib=ctypes.CDLL(str(LIBRARY))
        self.lib.frozen_tables_open.argtypes=[ctypes.c_char_p]
        self.lib.frozen_tables_open.restype=ctypes.c_void_p
        self.lib.frozen_tables_query.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double)]
        self.lib.frozen_tables_query.restype=ctypes.c_int
        self.lib.frozen_tables_close.argtypes=[ctypes.c_void_p]
        self.lib.frozen_tables_close.restype=None
        self.handle=self.lib.frozen_tables_open(str(cache).encode())
        if not self.handle:raise ValueError('Existing valid table cache directory required')

    def query(self,board):
        raw=np.asarray(board,dtype=np.int64)
        if raw.shape!=(4,4) or np.any(raw<0) or np.any((raw!=0)&((raw&(raw-1))!=0)):
            raise ValueError('Expected a4x4board of exact powers of two or zero')
        ranks=np.zeros(16,dtype=np.int32);flat=raw.reshape(16)
        ranks[flat>0]=np.log2(flat[flat>0]).astype(np.int32)
        if ranks.max()>17:raise ValueError('Rank exceeds the validated adapter range0..17')
        action=ctypes.c_int(-1);probability=ctypes.c_double()
        kind=self.lib.frozen_tables_query(self.handle,ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                          ctypes.byref(action),ctypes.byref(probability))
        if kind<0:raise ValueError('Native table query rejected the input')
        return dict(kind=kind,action=action.value,probability=probability.value)

    def close(self):
        if self.handle:self.lib.frozen_tables_close(self.handle);self.handle=None
