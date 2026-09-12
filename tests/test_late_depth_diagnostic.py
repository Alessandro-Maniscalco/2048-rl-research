import numpy as np

from research.late_depth_diagnostic import inspect_configuration


def test_late_game_assignment_is_no_longer_overwritten_in_isolated_variant():
    board=np.array([[32768,8192,2048,4],[16384,4096,1024,32],[512,256,64,16],[2,0,4,16]])
    assert board.sum()==65418
    original=inspect_configuration(board,last_sum=65346,last_depth=37)
    corrected=inspect_configuration(board,last_sum=65346,last_depth=37,corrected=True)
    assert original['initial_depth']==5 and original['max_depth']==48
    assert corrected['initial_depth']==33 and corrected['max_depth']==60
    assert corrected['time_limit']==.8
    board[0,0]=16384
    assert inspect_configuration(board)==inspect_configuration(board,corrected=True)
