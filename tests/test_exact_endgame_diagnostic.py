import numpy as np
import pytest

from research.exact_endgame_diagnostic import ExactEndgame

FINAL=np.array([[256,1024,4096,32768],[64,512,2048,8192],[8,16,32,128],[2,2,8,2]])


def test_hand_calculated_two_move_score_and_survival():
    solver=ExactEndgame()
    one=solver.query(FINAL,1)
    assert set(one)=={'left','right'}
    assert one['left']['expected_additional_raw_score']==4
    assert one['right']['expected_additional_raw_score']==4
    assert one['left']['probability_still_alive_after_h_moves']==pytest.approx(.9)
    assert one['right']['probability_still_alive_after_h_moves']==pytest.approx(.1)
    two=solver.query(FINAL,2)
    assert two['left']['expected_additional_raw_score']==pytest.approx(4+.9*4)
    assert two['right']['expected_additional_raw_score']==pytest.approx(4+.1*8)
    assert two['left']['probability_still_alive_after_h_moves']==pytest.approx(.09)
    assert two['right']['probability_still_alive_after_h_moves']==pytest.approx(.1)


def test_budget_exhaustion_is_not_an_approximate_result():
    with pytest.raises(RuntimeError,match='budget exceeded'):
        ExactEndgame(max_nodes=1).query(FINAL,3)
