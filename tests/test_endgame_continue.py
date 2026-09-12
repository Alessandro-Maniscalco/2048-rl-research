from copy import deepcopy

import numpy as np
import pytest

from research.endgame_continue import restore_environment
from rl2048.game import Game2048


def test_restore_full_rng_and_exact_next_transition_without_researching():
    env=Game2048();board,info=env.reset(seed=4182)
    frames=[dict(board=board.tolist(),score=0,action=None,reward=0)]
    rng=np.random.default_rng(831)
    for _ in range(30):
        action=int(rng.choice(np.flatnonzero(info['action_mask'])))
        board,reward,done,_,info=env.step(action);assert not done
        frames.append(dict(board=board.tolist(),score=env.score,action=action,reward=reward))
    replay=dict(result=dict(seed=4182),frames=frames)
    restored,_,new_info,done=restore_environment(replay)
    assert not done and restored.score==env.score and restored.steps==env.steps
    assert restored.np_random.bit_generator.state==env.np_random.bit_generator.state
    action=int(np.flatnonzero(new_info['action_mask'])[0])
    a=restored.step(action);b=env.step(action)
    np.testing.assert_array_equal(a[0],b[0]);assert a[1:4]==b[1:4]
    corrupted=deepcopy(replay);corrupted['frames'][20]['reward']+=4
    with pytest.raises(ValueError,match='prefix differs'):
        restore_environment(corrupted)
