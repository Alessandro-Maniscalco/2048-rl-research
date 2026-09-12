import numpy as np
import pytest

from research.full_rank_search import FullRankSearch, encode
from rl2048.game import move, legal_actions


def test_native_moves_preserve_large_tile_merges_and_raw_rewards():
    player=FullRankSearch()
    rng=np.random.default_rng(76421)
    for _ in range(80):
        ranks=rng.integers(0,18,(4,4))
        board=np.where(ranks>0,1<<ranks,0).astype(np.int64)
        for action in range(4):
            actual=player.slide(board,action);expected=move(board,action)
            np.testing.assert_array_equal(actual[0],expected[0])
            assert actual[1:]==expected[1:]
    board=np.zeros((4,4),dtype=np.int64);board[3,:2]=65536
    after,reward,changed=player.slide(board,3)
    assert changed and reward==131072 and after[3,0]==131072


def test_query_is_legal_repeatable_and_does_not_mutate_input():
    player=FullRankSearch()
    board=np.array([[4,16,4,2],[2,4,8,4],[0,0,16384,16384],[65536,2,4,4]],dtype=np.int64)
    before=board.copy()
    first=player.query(board,depth=3)
    assert first==player.query(board,depth=3)
    assert first['has_move'] and legal_actions(board)[first['action']]
    np.testing.assert_array_equal(before,board)
    with pytest.raises(ValueError):encode(np.full((4,4),3))
    with pytest.raises(ValueError):player.query(board,depth=11)
