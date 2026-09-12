"""Query the isolated wide-key prototype with cache enabled or disabled."""
import ctypes
import time

from research.full_rank_search import ROOT, encode

LIBRARY = ROOT / 'runs/research/endgame_tablebase/cached_full_rank_bridge/libcached_full_rank_search.dylib'


class CachedFullRankSearch:
    def __init__(self):
        self.lib = ctypes.CDLL(str(LIBRARY))
        pointer = ctypes.POINTER(ctypes.c_int)
        self.lib.cached_full_rank_query.argtypes = [pointer, ctypes.c_int, ctypes.c_int,
            pointer, pointer, ctypes.POINTER(ctypes.c_ulonglong)]
        self.lib.cached_full_rank_query.restype = ctypes.c_int

    def query(self, board, depth=8, enabled=True):
        ranks = encode(board)
        action, value = ctypes.c_int(-1), ctypes.c_int()
        stats = (ctypes.c_ulonglong * 4)()
        start = time.perf_counter()
        code = self.lib.cached_full_rank_query(ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            depth, int(enabled), ctypes.byref(action), ctypes.byref(value), stats)
        seconds = time.perf_counter() - start
        if code < 0:
            raise ValueError('Native query rejected board or depth')
        return dict(answer=dict(action=action.value, heuristic_value=value.value,
            has_move=bool(code), depth=depth), enabled=enabled, seconds=seconds,
            stats=dict(zip(('lookups', 'hits', 'updates', 'tile_calls'), stats)))
