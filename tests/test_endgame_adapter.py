import numpy as np
import pytest

from rl2048.agents.endgame import EndgameAgent,pack_ranks,load_native,VENDOR
from rl2048.agents.ntuple import encode
from rl2048.game import Game2048,move,legal_actions


pytestmark=pytest.mark.skipif(not list((VENDOR/'native_core').glob('ai_core*.so')),reason='Optional native engine not built')


def test_packing_directions_and_native_rules_before_rank15():
    load_native()
    from engine_core.BoardMover import s_move_board,decode_board
    rng=np.random.default_rng(1026)
    for _ in range(100):
        ranks=rng.integers(0,11,size=(4,4),dtype=np.uint8)
        board=np.where(ranks>0,1<<ranks.astype(np.int64),0)
        packed=pack_ranks(ranks)
        np.testing.assert_array_equal(decode_board(np.uint64(packed)),board)
        for ours,native in [(0,3),(1,2),(2,4),(3,1)]:
            after,reward,_=move(board,ours)
            actual,gain=s_move_board(np.uint64(packed),native)
            np.testing.assert_array_equal(decode_board(actual),after)
            assert gain==reward


def test_planner_does_not_mutate_board_or_rng_and_chooses_legal():
    env=Game2048();board,info=env.reset(seed=7119);agent=EndgameAgent(.1)
    initial=board.copy();rng_state=repr(env.np_random.bit_generator.state)
    action=agent.act(board,info['action_mask'])
    assert info['action_mask'][action]
    np.testing.assert_array_equal(board,initial)
    assert repr(env.np_random.bit_generator.state)==rng_state


def test_large_tile_game_remains_exact_and_planner_only_downgrades_input():
    core,_=load_native()
    board=np.array([[32768,32768,0,0],[2,4,8,0],[0,0,0,0],[0,0,0,0]])
    packed=pack_ranks(encode(board));changed=int(core.resolve_32768_doubles(packed))
    assert changed!=packed
    after,reward,valid=move(board,3)
    assert valid and reward==65536 and after[0,0]==65536
    assert board[0,0]==32768 and board[0,1]==32768


def test_terminal_native_root_has_no_out_of_range_score_access():
    core,_=load_native()
    board=np.array([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,2]])
    assert not legal_actions(board).any()
    player=core.AIPlayer(pack_ranks(encode(board)));player.max_threads=1;player.start_search(1)
    assert player.best_operation==0
