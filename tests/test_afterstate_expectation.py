import numpy as np
import pytest
import torch

from rl2048.afterstate_expectation import spawn_outcomes, weighted_spawn_values
from rl2048.agents.afterstate_mlp import AfterstateLearner
from rl2048.afterstate_compare import moves
from rl2048.agents.ntuple import row_tables


def test_spawn_mass_and_no_board_mutation():
    boards = np.arange(1,33,dtype=np.uint8).reshape(2,16)
    boards[0,[0,5]] = 0; boards[1,3] = 0; original = boards.copy()
    states,owners,p = spawn_outcomes(boards)
    np.testing.assert_array_equal(boards,original)
    np.testing.assert_array_equal(owners,[0,0,0,0,1,1])
    np.testing.assert_allclose(p,[.45,.05,.45,.05,.9,.1])
    np.testing.assert_allclose(np.bincount(owners,weights=p),[1,1])
    assert all(np.count_nonzero(s != boards[o]) == 1 for s,o in zip(states,owners))
    with pytest.raises(ValueError): spawn_outcomes(np.ones((1,16),np.uint8))


def test_weighted_expectation_is_not_uniform_branch_average():
    values = torch.tensor([10.,0.,20.,0.,4.,8.])
    owners = torch.tensor([0,0,0,0,1,1])
    p = torch.tensor([.45,.05,.45,.05,.9,.1])
    torch.testing.assert_close(weighted_spawn_values(values,owners,p,2),torch.tensor([13.5,4.4]))


def test_exact_target_matches_independent_spawn_sum_including_terminal():
    torch.set_num_threads(1); torch.manual_seed(3)
    learner = AfterstateLearner(width=8,gamma=.5,spawn_target='expected',target_batch=3)
    # Two legal afterstate shapes, including almost-full states with terminal spawns.
    boards = np.array([[1,2,3,4,2,3,4,5,3,4,5,6,4,5,1,0],
                       [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],np.uint8)
    # Constant positive estimates distinguish terminal-zero from bootstrapping.
    with torch.no_grad():
        learner.policy.layers[-1].bias.fill_(2.)
        learner.target.layers[-1].bias.fill_(6.)
    expected, count, slides = learner.expected_targets(boards)
    manual = []
    saw_terminal = False
    for board in boards:
        empty = np.flatnonzero(board==0); total = 0.
        for cell in empty:
            for rank,mass in [(1,.9),(2,.1)]:
                state=board.copy();state[cell]=rank
                _,gains,legal=moves(state[None],*row_tables())
                if not legal.any(): value=0.;saw_terminal=True
                else: value=float(gains[0,legal[0]].max()/128+.5*6.)
                total += mass/len(empty)*value
        manual.append(total)
    assert saw_terminal and slides==4*count
    # First board: a2spawn permits a4-point merge (90%); a4spawn is terminal (10%).
    torch.testing.assert_close(expected[0],torch.tensor(.9*(4/128+.5*6)))
    torch.testing.assert_close(expected,torch.tensor(manual),atol=1e-6,rtol=1e-6)
